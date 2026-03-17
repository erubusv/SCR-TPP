"""Training script for HNSTPP model.

Wiener-Hopf deterministic initialisation is delegated to
``workspace.train.wh_init`` (see wh_init.py).
"""

import os
import sys
import csv
import math
import time
import random
import pickle
import argparse

import yaml
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, '/workspace')
from workspace.models.HNSTPP import HNSTPP
from workspace.train.wh_init import wiener_hopf_initialize


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        prog = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        decay = 0.5 * (1 + math.cos(math.pi * prog))
        return max(self.min_lr_ratio,
                   self.min_lr_ratio + (1 - self.min_lr_ratio) * decay)


class TPPDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def get_collate_fn(pad_id):
    def collate_fn(batch):
        max_len = max(len(x['time']) for x in batch)
        B = len(batch)
        ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        tds = torch.zeros(B, max_len, dtype=torch.float32)
        for i, item in enumerate(batch):
            t = torch.as_tensor(item['time'], dtype=torch.float32)
            e = torch.as_tensor(item['event'], dtype=torch.long)
            n = len(t)
            ids[i, :n] = e
            if n > 0:
                tds[i, :n] = torch.cat([t[:1], t[1:] - t[:-1]])
        return {'input_ids': ids, 'time_diffs': tds,
                'attention_mask': (ids != pad_id).float()}
    return collate_fn


def train_epoch(model, loader, optimizer, scheduler, device, tau,
                grad_clip=5.0):
    model.train()
    totals = {}
    n_batches = 0
    for batch in tqdm(loader, leave=False, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        mo = model(batch, tau=tau)
        ld = model.compute_loss(batch, mo)
        ld['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for key, val in ld.items():
            v = val.item() if isinstance(val, torch.Tensor) else float(val)
            totals[key] = totals.get(key, 0) + v
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


def validate(model, loader, device, tau):
    model.eval()
    total_nll, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ld = model.compute_loss(batch, model(batch, tau=tau))
            total_nll += ld['nll_loss'].item()
            n += 1
    return total_nll / max(n, 1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HNSTPP')
    parser.add_argument('--config', type=str,
                        default='./workspace/train/train_config.yaml')
    parser.add_argument('--data', type=str,
                        default='./data/synthetic_data.pkl')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_name', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    hcfg = cfg.get('hnstpp', {})
    set_seed(args.seed)

    device = torch.device(
        f'cuda:{hcfg.get("gpu", 0)}'
        if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load data ----
    with open(args.data, 'rb') as f:
        dd = pickle.load(f)

    train_data = TPPDataset(dd['train'])
    val_data = TPPDataset(dd['val'])
    metadata = dd.get('metadata', {'num_types': 4})

    time_scale = metadata.get('mean_time_diff', 1.0)
    print(f"Time scale: {time_scale:.4f}")
    for ds in (train_data.data, val_data.data):
        for item in ds:
            item['time'] = [t / time_scale for t in item['time']]

    num_types = metadata.get('num_types', 4)
    pad_token_id = num_types
    batch_size = hcfg.get('batch_size', 32)
    print(f"Train: {len(train_data)},  Val: {len(val_data)}")

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=get_collate_fn(pad_token_id), num_workers=0,
        pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        collate_fn=get_collate_fn(pad_token_id), num_workers=0,
        pin_memory=(device.type == 'cuda'))

    # ---- Model ----
    model_config = {
        'num_types':          num_types,
        'num_rules':          hcfg.get('num_rules', 8),
        'num_bins':           hcfg.get('num_bins', 20),
        'pad_token_id':       pad_token_id,
        'start_tau':          hcfg.get('start_tau', 5.0),
        'epsilon':            hcfg.get('epsilon', 1e-6),
        'i_max':              hcfg.get('i_max', 20.0),
        'max_cap_percentile': hcfg.get('max_cap_percentile', 0.95),
        'max_cap_multiplier': hcfg.get('max_cap_multiplier', 5.0),
        'integral_num_points':hcfg.get('integral_num_points', 64),
        'sign_tau':           hcfg.get('sign_tau', 1.0),
        'init_b0':            hcfg.get('init_b0', -3.0),
        'init_bias_raw':      hcfg.get('init_bias_raw', 0.0),
        'init_w_exc':         hcfg.get('init_w_exc', -1.0),
        'init_w_inh':         hcfg.get('init_w_inh', -1.0),
        'init_sign_std':      hcfg.get('init_sign_std', 0.0),
        'lambda_ortho':       hcfg.get('lambda_ortho', 1e-3),
        'lambda_h_sparse':    hcfg.get('lambda_h_sparse', 1e-3),
        'lambda_h_binary':    hcfg.get('lambda_h_binary', 0.0),
        'lambda_w_sparse':    hcfg.get('lambda_w_sparse', 0.0),
        'lambda_overlap':     hcfg.get('lambda_overlap', 1e-2),
        'lambda_sign_binary': hcfg.get('lambda_sign_binary', 0.0),
        'lambda_head_entropy':hcfg.get('lambda_head_entropy', 1e-3),
        'lambda_smooth':      hcfg.get('lambda_smooth', 1.0),
    }

    model = HNSTPP(model_config).to(device)

    # Set max_cap
    td_list, id_list = [], []
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        td_list.append(batch['time_diffs'].flatten())
        id_list.append(batch['input_ids'].flatten())
    model.set_max_cap(torch.cat(td_list), torch.cat(id_list))

    # ---- Wiener-Hopf initialisation ----
    if 'wh_fixed_target' not in hcfg:
        raise ValueError(
            "wh_fixed_target must be set in config for fixed-target initialisation."
        )
    fixed_target = int(hcfg['wh_fixed_target'])
    wiener_hopf_initialize(
        model, train_data.data, num_types, device,
        delta_t=float(hcfg.get('wh_delta_t', 0.1)),
        max_lag=float(hcfg.get('wh_max_lag', 10.0)),
        lambda_l1=float(hcfg.get('wh_lambda_l1', 1e-3)),
        pair_min_support=int(hcfg.get('wh_pair_min_support', 2)),
        pair_topk_per_target=int(hcfg.get('wh_pair_topk_per_target', 20)),
        max_source_order=int(hcfg.get('wh_max_source_order', 3)),
        phase3_support_tau=float(hcfg.get('wh_phase3_support_tau', 20.0)),
        phase3_source_pool_topk=int(hcfg.get('wh_phase3_source_pool_topk', 24)),
        phase3_beta2=float(hcfg.get('wh_phase3_beta2', 0.5)),
        phase3_beta3=float(hcfg.get('wh_phase3_beta3', 1.0)),
        fixed_target=fixed_target,
    )

    # ---- Optimiser (exclude frozen bias) ----
    lr = float(hcfg.get('learning_rate', 1e-3))
    total_epochs = hcfg.get('epochs', 50)
    grad_clip = float(hcfg.get('grad_clip', 5.0))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr,
                      weight_decay=float(hcfg.get('weight_decay', 1e-4)))
    total_steps = total_epochs * len(train_loader)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=int(total_steps * 0.1),
        total_steps=total_steps)

    start_tau = float(model_config['start_tau'])
    end_tau   = float(hcfg.get('end_tau', 1.0))

    # ---- Save directory ----
    base_save = hcfg.get('model_save_path', './workspace/saved_models/HNSTPP')
    run_name = args.run_name or f'seed{args.seed}'
    save_dir = os.path.join(base_save, run_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'run_config.yaml'), 'w') as f:
        yaml.safe_dump({'model': model_config, 'seed': args.seed,
                        'run_name': run_name}, f)

    log_path = os.path.join(save_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'train_nll', 'val_nll',
            'event_ll', 'integral_loss', 'num_events',
            'b_k_mean', 'W_pos_mean', 'W_neg_mean', 'sign_gate_mean',
            'reg_loss', 'tau', 'epoch_time'])

    # ---- Training loop ----
    print(f"\n{'=' * 50}")
    print(f"Training HNSTPP  |  {total_epochs} epochs  |  "
          f"R={model_config['num_rules']}  M={model_config['num_bins']}  "
          f"tau {start_tau}→{end_tau}")
    print(f"{'=' * 50}\n")

    best_val_nll = float('inf')

    for epoch in range(total_epochs):
        progress = epoch / max(1, total_epochs - 1)
        tau = start_tau + (end_tau - start_tau) * progress
        model.tau = tau

        t0 = time.time()
        m = train_epoch(model, train_loader, optimizer, scheduler,
                        device, tau, grad_clip)
        dt = time.time() - t0

        val_nll = validate(model, val_loader, device, tau=min(tau, 1.0))

        print(f"Epoch {epoch+1:3d}/{total_epochs} | "
              f"Loss {m['total_loss']:.4f} | NLL {m['nll_loss']:.4f} | "
              f"Val {val_nll:.4f} | tau {tau:.2f} | "
              f"b {m['b_k_mean']:.3f} | "
              f"W+ {m['W_pos_mean']:.3f} | W- {m['W_neg_mean']:.3f} | "
              f"{dt:.1f}s")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, m['total_loss'], m['nll_loss'], val_nll,
                m['event_ll'], m['integral_loss'], m['num_events'],
                m['b_k_mean'], m['W_pos_mean'], m['W_neg_mean'],
                m['sign_gate_mean'], m['reg_loss'], tau, dt])

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_model.pt'))
            print(f"  → saved best (val {val_nll:.4f})")

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    print(f"\nBest Val NLL: {best_val_nll:.4f}")

    # ---- Print learned structure ----
    print(f"\n{'=' * 50}")
    print("Learned Structure")
    print(f"{'=' * 50}")
    s = model.get_structure()
    print(f"Base: {s['b_k'].numpy().round(4)}")
    print(f"W+  : {s['W_pos'].numpy().round(4)}")
    print(f"W-  : {s['W_neg'].numpy().round(4)}")
    print(f"Sign: {s['sign_gate'].numpy().round(3)}")
    print(f"Bias: {s['rule_bias'].numpy().round(3)}")

    H, Head = s['H'].numpy(), s['Head'].numpy()
    for r in range(model_config['num_rules']):
        srcs = [k for k in range(num_types) if H[k, r] > 0.5]
        tgt = int(Head[r].argmax())
        gate = float(s['sign_gate'][r])
        sign = 'EXC' if gate > 0.5 else 'INH'
        w = float(s['W_pos'][r]) if gate > 0.5 else float(s['W_neg'][r])
        h = s['kernel_heights'][r].numpy()
        print(f"  Rule {r}: {srcs} → {tgt} [{sign}] "
              f"W={w:.4f}  h={np.round(h, 3)}")

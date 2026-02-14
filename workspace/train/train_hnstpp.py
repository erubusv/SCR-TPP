import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import pickle
import argparse
import pandas as pd
import math
from tqdm.auto import tqdm

from workspace.models.HNSTPP_triangular_latent import HNSTPPTriangularLatent
from workspace.train.generate_dataset import load_config

torch.set_float32_matmul_precision('high')

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(self.min_lr_ratio, self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay)

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
        padded_input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        padded_time_diffs = torch.zeros((B, max_len), dtype=torch.float32)
        attention_masks = torch.zeros((B, max_len), dtype=torch.float32)
        
        for idx, item in enumerate(batch):
            times = torch.as_tensor(item['time'], dtype=torch.float32)
            events = torch.as_tensor(item['event'], dtype=torch.long)
            seq_len = len(times)
            padded_input_ids[idx, :seq_len] = events
            attention_masks[idx, :seq_len] = 1.0
            if seq_len > 0:
                time_diffs = torch.cat([times[:1], times[1:] - times[:-1]])
                padded_time_diffs[idx, :seq_len] = time_diffs
        
        return {'input_ids': padded_input_ids, 'time_diffs': padded_time_diffs, 'attention_mask': attention_masks}
    return collate_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./workspace/train/train_config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    hcfg = cfg.get('hnstpp', {})
    device = torch.device(f'cuda:{hcfg.get("gpu", 0)}' if torch.cuda.is_available() else 'cpu')

    with open(cfg['path']['data_path'], 'rb') as f:
        dataset_dict = pickle.load(f)
    
    train_data = TPPDataset(dataset_dict['train'])
    val_data = TPPDataset(dataset_dict['val'])
    metadata = dataset_dict.get('metadata', {'num_types': 5})
    
    time_scale = metadata.get('mean_time_diff', 1.0)
    print(f"\n Time scale: {time_scale:.4f}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    for dataset in [train_data.data, val_data.data]:
        for item in dataset:
            item['time'] = [t / time_scale for t in item['time']]
    
    pad_token_id = hcfg.get('pad_token_id', cfg.get('pad_token_id', 0))
    batch_size = hcfg.get('batch_size', 32)
    collate_fn = get_collate_fn(pad_token_id)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model_config = dict(hcfg)
    model_config['num_types'] = cfg.get('num_types', metadata.get('num_types', 5))
    model_config['num_event_types'] = cfg.get('num_event_types', metadata.get('num_types', 5))
    model_config['pad_token_id'] = pad_token_id
    model_config['start_tau'] = hcfg.get('start_tau', 5.0)
    model_config['end_tau'] = hcfg.get('end_tau', 0.5)
    model_config['tau_anneal_rate'] = hcfg.get('tau_anneal_rate', 0.95)
    model_config['num_latent_nodes'] = hcfg.get('num_latent_nodes', 0)

    model = HNSTPPTriangularLatent(model_config).to(device)

    phase1_epochs = hcfg.get('phase1_epochs', 30)
    phase2_epochs = hcfg.get('phase2_epochs', 20)
    
    kernel_params, theta_params, base_params, other_params = [], [], [], []
    for name, param in model.named_parameters():
        if 'raw_deltas' in name or 'raw_widths' in name or 'raw_logits' in name:
            kernel_params.append(param)
        elif 'theta' in name or 'rule_target_logits' in name:
            theta_params.append(param)
        elif 'b0' in name:
            base_params.append(param)
        else:
            other_params.append(param)
    
    for p in base_params:
        p.requires_grad = False

    no_decay = ['theta', 'b0', 'raw_deltas', 'raw_widths']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 1e-4},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=float(hcfg.get('start_learning_rate', 1e-3)))

    total_epochs = phase1_epochs + phase2_epochs
    total_steps = total_epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)
    min_lr = float(hcfg.get('min_learning_rate', 1e-5))
    start_lr = float(hcfg.get('start_learning_rate', 1e-3))
    
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr/start_lr)
    
    print("\n=== 2-Phase Staged Training ===")
    print(f"Phase 1 ({phase1_epochs} epochs): Kernel / Theta learning")
    print(f"Phase 2 ({phase2_epochs} epochs): Base intensity learning\n")
    
    best_val_loss = float('inf')
    start_tau = hcfg.get('start_tau', 5.0)
    end_tau = hcfg.get('end_tau', 0.5)
    tau_anneal_rate = hcfg.get('tau_anneal_rate', 0.95)
    
    for epoch in range(total_epochs):
        if epoch < phase1_epochs:
            phase = 1
            use_gumbel = True
            lambda_sparse_current = hcfg.get('lambda_sparse', 1e-3)
            lambda_base_current = 1e6
        else:
            if epoch == phase1_epochs:
                for p in base_params:
                    p.requires_grad = True
                optimizer.add_param_group({'params': base_params, 'weight_decay': 0.0, 'lr': optimizer.param_groups[0]['lr']})
                print(f"\n>>> Phase 2 Started: Base intensity learning enabled")
            phase = 2
            use_gumbel = True
            lambda_sparse_current = hcfg.get('lambda_sparse', 1e-3)
            lambda_base_current = hcfg.get('base_reg_relaxed', 0.1)
        
        current_tau = max(end_tau, start_tau * (tau_anneal_rate ** (epoch * 0.5))) if use_gumbel else 100.0
        
        model.train()
        total_loss = 0
        nll_sum = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:02d} [P{phase}]", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            output = model(batch, current_tau)
            losses = model.compute_loss(batch, output, sparsity_weight=lambda_sparse_current, base_weight=lambda_base_current)
            loss = losses['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            nll_sum += losses['nll_loss']
        
        avg_train_loss = total_loss / len(train_loader)
        avg_nll = nll_sum / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(batch)
                losses = model.compute_loss(batch, output, sparsity_weight=lambda_sparse_current, base_weight=lambda_base_current)
                val_loss += losses['nll_loss']
        
        avg_val_loss = val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} [P{phase}] | NLL: {avg_nll:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = hcfg.get('model_save_path', './workspace/saved_models/synthetic/HNSTPP')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/best_model.pt')
    
    print("\n=== Training Complete ===")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    
    print("\n" + "="*40)
    print("LEARNED RULES INTERPRETATION")
    print("="*40)
    
    model.eval()
    event_names = [f"Type {i}" for i in range(model_config['num_event_types'])]
    struct = model.explain_model_parameters(event_names=event_names)
    
    if struct is not None:
        df_rules, df_interactions, b0_df = struct
        save_dir = hcfg.get('model_save_path', './workspace/saved_models/synthetic/HNSTPP')
        df_rules.to_csv(f"{save_dir}/rules.csv", index=False)
        df_interactions.to_csv(f"{save_dir}/interactions.csv", index=False)
        b0_df.to_csv(f"{save_dir}/base_intensities.csv", index=False)
        
        print("\n[Learned Rules]")
        print(df_rules.to_string(index=False))
        print("\n[Learned Interactions]")
        if not df_interactions.empty:
            print(df_interactions.head(10).to_string(index=False))
        else:
            print("No significant interactions")

if __name__ == "__main__":
    main()

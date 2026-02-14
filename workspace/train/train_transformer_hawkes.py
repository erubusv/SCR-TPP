import argparse
import time
import warnings
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from workspace.models.TransformerHawkes import TransformerHawkes
from workspace.train.dataset_transformer import EventSeqDataset, collate_fn
from workspace.train.generate_dataset import load_config


class WarmupCosineSchedule(LambdaLR):
    """
    Linear Warmup + Cosine Annealing scheduler (same as used in HNSTPP)
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(self.min_lr_ratio, self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay)

# Suppress specific PyTorch nested-tensor prototype warnings
warnings.filterwarnings("ignore", message=".*nested.*tensor.*", category=UserWarning)
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage", category=UserWarning)


def compute_loss_with_mc(model, batch, pad_id, device, num_mc_samples=10):
    types = batch['types'].to(device)
    times = batch['times'].to(device)  # absolute times
    mask = batch['mask'].to(device)
    tgt_type = batch['tgt_type'].to(device)
    tgt_dt = batch['tgt_dt'].to(device)

    logits, all_hid, _ = model(types, times, src_key_padding_mask=mask)

    valid_mask = (~mask) & (tgt_type != pad_id)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # Cross entropy (type prediction)
    logits_flat = logits.view(-1, logits.size(-1))
    tgt_flat = tgt_type.view(-1)
    ce = nn.functional.cross_entropy(logits_flat, tgt_flat, reduction='none').view(tgt_type.shape)

    scale_factor = 1.0 / (times + 1.0) 

    # intensity at actual event time
    actual_dt_exp = (tgt_dt * scale_factor).unsqueeze(-1) 
    raw_intensity = all_hid + model.alpha * actual_dt_exp
    lambda_at_event = model._softplus(raw_intensity, model.beta)

    # gather target type intensity
    safe_tgt = tgt_type.clone()
    safe_tgt[safe_tgt == pad_id] = 0
    target_lambda = lambda_at_event.gather(2, safe_tgt.unsqueeze(-1)).squeeze(-1) 
    event_ll = torch.log(target_lambda + 1e-9)

    # Monte-Carlo integration for non-event term
    B, L = tgt_dt.shape
    device = tgt_dt.device
    random_u = torch.rand(B, L, num_mc_samples, device=device)
    sampled_dt = tgt_dt.unsqueeze(-1) * random_u
    sampled_scaled_dt = sampled_dt * scale_factor.unsqueeze(-1)

    sampled_raw = all_hid.unsqueeze(2) + model.alpha * sampled_scaled_dt.unsqueeze(-1) 
    sampled_lambda = model._softplus(sampled_raw, model.beta) 
    total_lambda_sampled = sampled_lambda.sum(dim=-1)
    avg_lambda = total_lambda_sampled.mean(dim=-1) 

    non_event_ll = -avg_lambda * tgt_dt 

    temporal_loss = -(event_ll + non_event_ll)

    total_loss = (ce * valid_mask).sum() + (temporal_loss * valid_mask).sum()
    n_valid = valid_mask.sum().item()
    return total_loss, n_valid


def compute_metrics_with_mc(model, batch, pad_id, device, num_mc_samples=100, top_k=1):
    with torch.no_grad():
        types = batch['types'].to(device)
        times = batch['times'].to(device)
        mask = batch['mask'].to(device)
        tgt_type = batch['tgt_type'].to(device)
        tgt_dt = batch['tgt_dt'].to(device)

        logits, all_hid, _ = model(types, times, src_key_padding_mask=mask)
        valid_mask = (~mask) & (tgt_type != pad_id)
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            return 0.0, 0.0, 0.0, 0

        # Type CE
        logits_valid = logits[valid_mask]
        tgt_valid = tgt_type[valid_mask]
        type_ce = nn.functional.cross_entropy(logits_valid, tgt_valid, reduction='sum').item()

        # time terms using MC
        scale_factor = 1.0 / (times + 1.0)
        actual_dt_exp = (tgt_dt * scale_factor).unsqueeze(-1)
        raw_intensity = all_hid + model.alpha * actual_dt_exp
        lambda_at_event = model._softplus(raw_intensity, model.beta)
        safe_tgt_type = tgt_type.clone()
        safe_tgt_type[safe_tgt_type == pad_id] = 0
        target_lambda = lambda_at_event.gather(2, safe_tgt_type.unsqueeze(-1)).squeeze(-1)
        event_ll = torch.log(target_lambda + 1e-9)

        B, L = tgt_dt.shape
        device = tgt_dt.device
        random_u = torch.rand(B, L, num_mc_samples, device=device)
        sampled_dt = tgt_dt.unsqueeze(-1) * random_u
        sampled_scaled_dt = sampled_dt * scale_factor.unsqueeze(-1)
        sampled_raw = all_hid.unsqueeze(2) + model.alpha * sampled_scaled_dt.unsqueeze(-1)
        sampled_lambda = model._softplus(sampled_raw, model.beta)
        total_lambda_sampled = sampled_lambda.sum(dim=-1)
        avg_lambda = total_lambda_sampled.mean(dim=-1)

        non_event_ll = -avg_lambda * tgt_dt
        temporal_nll = -(event_ll + non_event_ll)

        # MAE: predict expected dt using predict_event_rates at candidate dt=tgt_dt
        preds = model.predict_event_rates({'types': types, 'times': times, 'tgt_dt': tgt_dt, 'mask': mask}, model_output=(logits, all_hid, None))
        pred_dt = preds['pred_dt']

        mae = torch.abs(pred_dt[valid_mask] - tgt_dt[valid_mask]).sum().item()

        # Top-k accuracy
        k = max(1, int(top_k))
        k = min(k, logits.size(-1))
        topk_idx = logits.topk(k, dim=-1).indices
        topk_matches = (topk_idx == tgt_type.unsqueeze(-1)).any(dim=-1)
        topk = topk_matches[valid_mask].sum().item()

        # total nll (type + time)
        total_time_nll = temporal_nll[valid_mask].sum().item()

        return (type_ce + total_time_nll), mae, topk, n_valid


def train_epoch(model, loader, optimizer, scheduler, device, pad_id, num_mc_samples=10, top_k=1):
    model.train()
    total = {'nll': 0.0, 'mae': 0.0, 'topk': 0.0, 'n_valid': 0}
    start = time.time()

    scaler = GradScaler()

    for batch in loader:
        # compute loss with autocast
        with autocast():
            loss, n_valid = compute_loss_with_mc(model, batch, pad_id, device, num_mc_samples=num_mc_samples)
        if n_valid == 0:
            continue

        loss = loss / n_valid

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)

        scale_before = scaler.get_scale()
        scaler.update()
        scale_after = scaler.get_scale()

        # Only step scheduler when scaler did not reduce the scale (mimic HNSTPP condition)
        if scale_after >= scale_before:
            if scheduler is not None:
                scheduler.step()

        # eval-style metrics on this batch (use more samples for robust metric estimate)
        batch_nll, batch_mae, batch_topk, batch_count = compute_metrics_with_mc(model, batch, pad_id, device, num_mc_samples=min(100, num_mc_samples*10), top_k=top_k)

        total['nll'] += batch_nll
        total['mae'] += batch_mae
        total['topk'] += batch_topk
        total['n_valid'] += batch_count

    elapsed = time.time() - start
    if total['n_valid'] == 0:
        return {'nll': 0.0, 'mae': 0.0, 'topk': 0.0, 'time': elapsed}

    return {'nll': total['nll']/total['n_valid'], 'mae': total['mae']/total['n_valid'], 'topk': total['topk']/total['n_valid'], 'time': elapsed}


def evaluate(model, loader, device, pad_id, num_mc_samples=200, top_k=1):
    model.eval()
    total = {'nll': 0.0, 'mae': 0.0, 'topk': 0.0, 'n_valid': 0}
    with torch.no_grad():
        for batch in loader:
            batch_nll, batch_mae, batch_topk, batch_count = compute_metrics_with_mc(model, batch, pad_id, device, num_mc_samples=num_mc_samples, top_k=top_k)
            total['nll'] += batch_nll
            total['mae'] += batch_mae
            total['topk'] += batch_topk
            total['n_valid'] += batch_count

    if total['n_valid'] == 0:
        return {'nll': 0.0, 'mae': 0.0, 'topk': 0.0}
    return {'nll': total['nll']/total['n_valid'], 'mae': total['mae']/total['n_valid'], 'topk': total['topk']/total['n_valid']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./workspace/train/train_config.yaml')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(f'cuda:{cfg["transformer"]["gpu"]}' if torch.cuda.is_available() else 'cpu')

    data_path = cfg['path']['data_path']
    transformer_cfg = cfg.get('transformer', {})
    pad_id = transformer_cfg.get('pad_token_id', transformer_cfg.get('pad_id', cfg.get('pad_token_id', 0)))

    ds_train = EventSeqDataset(data_path, split='train')
    ds_val = EventSeqDataset(data_path, split='val')

    # Prefer model-specific settings, fall back to global defaults
    batch_size = transformer_cfg.get('batch_size', cfg.get('batch_size', 32))
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    loader_val = DataLoader(ds_val, batch_size=transformer_cfg.get('batch_size', cfg.get('val_batch_size', batch_size)), shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    num_types = cfg.get('num_event_types', ds_train.metadata.get('num_types', 5))

    embed_dim = transformer_cfg.get('embed_dim', cfg.get('embed_dim', 128))
    nhead = transformer_cfg.get('num_heads', cfg.get('num_heads', 4))
    # prefer even number of heads to allow nested tensor optimizations; adjust upward if odd
    if nhead % 2 != 0:
        nhead += 1
        print(f"Adjusted nhead to {nhead} (was odd) to avoid nested-tensor warnings")

    # Ensure d_model is divisible by nhead
    d_model = embed_dim
    if d_model % nhead != 0:
        d_model = d_model + (nhead - (d_model % nhead))

    model = TransformerHawkes(num_types=num_types, pad_id=pad_id, d_model=d_model, nhead=nhead, num_layers=transformer_cfg.get('n_layers', 2)).to(device)

    # learning rate settings (model-specific preferred)
    start_lr = transformer_cfg.get('start_learning_rate', cfg.get('start_learning_rate', 1e-3))
    min_lr = transformer_cfg.get('min_learning_rate', cfg.get('min_learning_rate', 1e-4))
    optimizer = AdamW(model.parameters(), lr=float(start_lr), weight_decay=1e-4)

    # Scheduler setup matching HNSTPP's WarmupCosineSchedule
    steps_per_epoch = len(loader_train)
    epochs = transformer_cfg.get('epochs', args.epochs)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * 0.1)
    min_lr = float(transformer_cfg.get('min_learning_rate', cfg.get('min_learning_rate', 1e-4)))
    start_lr = float(transformer_cfg.get('start_learning_rate', cfg.get('start_learning_rate', 1e-3)))

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr / max(1e-12, start_lr)
    )

    print(f"Model params: {model.count_parameters()}")

    # choose number of epochs: model config preferred, else CLI arg
    train_mc = transformer_cfg.get('train_mc_samples', 10)
    eval_mc = transformer_cfg.get('eval_mc_samples', 200)

    top_k = transformer_cfg.get('top_k', cfg.get('top_k', 1))

    for epoch in range(epochs):
        train_stats = train_epoch(model, loader_train, optimizer, scheduler, device, pad_id, num_mc_samples=train_mc, top_k=top_k)
        val_stats = evaluate(model, loader_val, device, pad_id, num_mc_samples=eval_mc, top_k=top_k)
        print(f"Epoch {epoch+1}: train nll={train_stats['nll']:.4f} mae={train_stats['mae']:.4f} top{top_k}={train_stats['topk']:.3f} time={train_stats['time']:.1f}s | val nll={val_stats['nll']:.4f} mae={val_stats['mae']:.4f} top{top_k}={val_stats['topk']:.3f}")

    # save
    import os
    save_dir = cfg['path'].get('transformer_save_path', './workspace/saved_models/synthetic/transformer')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'cfg': cfg}, f'{save_dir}/best_model.pt')
    print(f"Saved to {save_dir}/best_model.pt")

if __name__ == '__main__':
    main()

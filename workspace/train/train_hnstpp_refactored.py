"""Training script for HNSTPP-Refactored model.

Clean implementation without OOM-related chunking.
Follows refactoring.md for sweep-line integration.
"""

import os
import sys
import torch
import random
import numpy as np
import csv
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import pickle
import argparse
import yaml
import math
import time
from tqdm.auto import tqdm

sys.path.insert(0, '/workspace')
from workspace.models.hnstpp_refactored import HNSTPPRefactored


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self.lr_lambda)
    
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
        
        for idx, item in enumerate(batch):
            times = torch.as_tensor(item['time'], dtype=torch.float32)
            events = torch.as_tensor(item['event'], dtype=torch.long)
            seq_len = len(times)
            
            padded_input_ids[idx, :seq_len] = events
            if seq_len > 0:
                time_diffs = torch.cat([times[:1], times[1:] - times[:-1]])
                padded_time_diffs[idx, :seq_len] = time_diffs
        
        attention_masks = (padded_input_ids != pad_id).float()
        return {
            'input_ids': padded_input_ids,
            'time_diffs': padded_time_diffs,
            'attention_mask': attention_masks
        }
    return collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, scheduler, device, tau, theta_noise_std=0.0, grad_clip=5.0, b_max=None, use_ste=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_event_ll = 0.0
    total_integral = 0.0
    total_num_events = 0.0
    total_lambda_std = 0.0
    total_b_mean = 0.0
    total_wpos_mean = 0.0
    total_wneg_mean = 0.0
    total_sign_gate_mean = 0.0
    total_source_card_mean = 0.0
    total_pr_mean = 0.0
    total_reg = 0.0
    total_gamma_base = 0.0
    total_resp_entropy = 0.0
    total_resp_base_penalty = 0.0
    total_exc_top1 = 0.0
    total_inh_top1 = 0.0
    total_attenuation = 0.0
    total_inh_aux = 0.0
    total_w_grad_norm = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, leave=False, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward
        model_output = model(batch, tau=tau, theta_noise_std=theta_noise_std, b_max=b_max, use_ste=use_ste)
        
        # Compute loss
        loss_dict = model.compute_loss(batch, model_output, tau=tau, include_aux_loss=True)
        loss = loss_dict['total_loss']
        
        # Backward
        loss.backward()
        
        # Gradient clipping (from refactoring.md C2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Track weight gradient magnitude
        w_grad_norm = 0.0
        if model.w_exc_raw.grad is not None:
            w_grad_norm += model.w_exc_raw.grad.norm().item()
        if model.w_inh_raw.grad is not None:
            w_grad_norm += model.w_inh_raw.grad.norm().item()
        total_w_grad_norm += w_grad_norm
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_nll += loss_dict['nll_loss'].item()
        total_event_ll += float(loss_dict.get('event_ll', 0.0))
        total_integral += float(loss_dict.get('integral_loss', 0.0))
        total_num_events += float(loss_dict.get('num_events', 0.0))
        total_lambda_std += float(loss_dict.get('lambda_std_at_events', 0.0))
        total_b_mean += float(loss_dict.get('b_k_mean', 0.0))
        total_wpos_mean += float(loss_dict.get('W_pos_mean', 0.0))
        total_wneg_mean += float(loss_dict.get('W_neg_mean', 0.0))
        total_sign_gate_mean += float(loss_dict.get('sign_gate_mean', 0.0))
        total_source_card_mean += float(loss_dict.get('source_card_mean', 0.0))
        total_pr_mean += float(loss_dict.get('p_r_mean', 0.0))
        total_reg += float(loss_dict.get('reg_loss', 0.0))
        total_gamma_base += float(loss_dict.get('mean_gamma_base', 0.0))
        total_resp_entropy += float(loss_dict.get('resp_entropy', 0.0))
        total_resp_base_penalty += float(loss_dict.get('resp_base_penalty', 0.0))
        total_exc_top1 += float(loss_dict.get('exc_top1_ratio', 0.0))
        total_inh_top1 += float(loss_dict.get('inh_top1_ratio', 0.0))
        total_attenuation += float(loss_dict.get('attenuation_mean', 0.0))
        total_inh_aux += float(loss_dict.get('inh_aux_loss', 0.0))
        num_batches += 1
    
    integral_ratio = total_integral / max(1e-9, -total_event_ll)

    metrics = {
        'train_loss': total_loss / num_batches,
        'train_nll': total_nll / num_batches,
        'event_ll': total_event_ll / num_batches,
        'integral_loss': total_integral / num_batches,
        'num_events': total_num_events / num_batches,
        'integral_ratio': integral_ratio,
        'lambda_std_at_events': total_lambda_std / num_batches,
        'b_k_mean': total_b_mean / num_batches,
        'W_pos_mean': total_wpos_mean / num_batches,
        'W_neg_mean': total_wneg_mean / num_batches,
        'sign_gate_mean': total_sign_gate_mean / num_batches,
        'source_card_mean': total_source_card_mean / num_batches,
        'p_r_mean': total_pr_mean / num_batches,
        'reg_loss': total_reg / num_batches,
        'mean_gamma_base': total_gamma_base / num_batches,
        'resp_entropy': total_resp_entropy / num_batches,
        'resp_base_penalty': total_resp_base_penalty / num_batches,
        'exc_top1_ratio': total_exc_top1 / num_batches,
        'inh_top1_ratio': total_inh_top1 / num_batches,
        'attenuation_mean': total_attenuation / num_batches,
        'inh_aux_loss': total_inh_aux / num_batches,
        'w_grad_norm': total_w_grad_norm / num_batches,
    }

    return metrics


def validate(model, loader, device, tau, theta_noise_std=0.0, b_max=None):
    """Validate model."""
    model.eval()
    total_nll = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model_output = model(batch, tau=tau, theta_noise_std=theta_noise_std, b_max=b_max)
            loss_dict = model.compute_loss(batch, model_output, tau=tau, include_aux_loss=False)
            total_nll += loss_dict['nll_loss'].item()
            num_batches += 1
    
    return total_nll / num_batches


def compute_empirical_excitation_matrix(data_list, num_types, time_scale, window_sizes=[0.5, 1.0, 2.0, 4.0]):
    """Compute empirical pair-wise interaction z-scores from training data.
    
    For each pair (src, tgt), count how many tgt events follow src events within
    a time window, compared to the expected count under independence.
    Uses Poisson z-score normalization for fair comparison across types.
    Returns:
      exc_z: best positive z-score (excitation signal)
      inh_z: best negative z-score (inhibition signal)
      type_rates: observed event rates [per unit time]
    """
    # Count events per type and total time
    type_counts = np.zeros(num_types)
    total_time = 0.0
    for item in data_list:
        times = np.array(item['time'], dtype=np.float64) / time_scale
        events = np.array(item['event'], dtype=np.int64)
        for k in range(num_types):
            type_counts[k] += np.sum(events == k)
        if len(times) > 0:
            total_time += times[-1]
    
    type_rates = type_counts / max(total_time, 1e-9)
    
    # For each window size, compute z-scores
    best_exc_z = np.full((num_types, num_types), -np.inf)
    best_inh_z = np.full((num_types, num_types), np.inf)
    
    for window in window_sizes:
        pair_counts = np.zeros((num_types, num_types))
        src_counts = np.zeros(num_types)  # Total source events
        
        for item in data_list:
            times = np.array(item['time'], dtype=np.float64) / time_scale
            events = np.array(item['event'], dtype=np.int64)
            n = len(times)
            if n < 2:
                continue
            
            # For each source event, count target events in (t_src, t_src + window]
            for i in range(n):
                src = events[i]
                if src >= num_types:
                    continue
                src_counts[src] += 1
                t_src = times[i]
                # Find target events in (t_src, t_src + window]
                for j in range(i + 1, n):
                    dt = times[j] - t_src
                    if dt > window:
                        break
                    if dt <= 0:
                        continue
                    tgt = events[j]
                    if tgt < num_types:
                        pair_counts[src, tgt] += 1
        
        # Expected under independence: N_src * rate_tgt * window
        pair_expected = np.outer(src_counts, type_rates * window)
        
        # Poisson z-score: z = (observed - expected) / sqrt(expected)
        safe_expected = np.maximum(pair_expected, 1.0)
        z_scores = (pair_counts - pair_expected) / np.sqrt(safe_expected)
        
        # Track best z-scores across window sizes
        best_exc_z = np.maximum(best_exc_z, z_scores)
        best_inh_z = np.minimum(best_inh_z, z_scores)
    
    return best_exc_z, best_inh_z, type_rates


def initialize_structure_from_data(model, data_list, num_types, time_scale, num_rules, device):
    """Initialize theta (H logits) and rule_target_logits from empirical cross-correlations.
    
    Uses Poisson z-score tests to detect both excitation and inhibition.
    Adjusts base intensities down for types that are targets of excitation.
    """
    print("\n=== Data-driven Structure Initialization ===")
    exc_z, inh_z, type_rates = compute_empirical_excitation_matrix(
        data_list, num_types, time_scale
    )
    
    print(f"Type rates (scaled): {type_rates}")
    print(f"Excitation z-scores (src→tgt):")
    for src in range(num_types):
        row = [f"{exc_z[src, tgt]:+.1f}" for tgt in range(num_types)]
        print(f"  src={src}: [{', '.join(row)}]")
    print(f"Inhibition z-scores (src→tgt):")
    for src in range(num_types):
        row = [f"{inh_z[src, tgt]:+.1f}" for tgt in range(num_types)]
        print(f"  src={src}: [{', '.join(row)}]")
    
    # Collect interactions: excitation (z > threshold) and inhibition (z < -threshold)
    z_threshold = 3.0  # Roughly p < 0.001
    interactions = []
    for src in range(num_types):
        for tgt in range(num_types):
            if src == tgt:
                continue  # Skip self-interactions (often spurious)
            # Excitation candidate
            if exc_z[src, tgt] > z_threshold:
                interactions.append((
                    exc_z[src, tgt], 'EXC', src, tgt
                ))
            # Inhibition candidate
            if inh_z[src, tgt] < -z_threshold:
                interactions.append((
                    abs(inh_z[src, tgt]), 'INH', src, tgt
                ))
    
    # Sort by absolute score (strongest interactions first)
    interactions.sort(reverse=True)
    
    # Deduplicate: avoid duplicate (src, tgt) pairs, preferring stronger signal
    used_pairs = set()
    assignments = []
    for score, sign, src, tgt in interactions:
        if len(assignments) >= num_rules:
            break
        if (src, tgt) in used_pairs:
            continue
        used_pairs.add((src, tgt))
        assignments.append((score, sign, src, tgt))
    
    print(f"\nTop-{num_rules} interactions for rule initialization:")
    
    with torch.no_grad():
        for r, (score, sign, src, tgt) in enumerate(assignments):
            print(f"  Rule {r}: {src} → {tgt} [{sign}] score={score:.3f}")
            
            # Initialize theta: strong positive logit for the source type
            model.theta.data[:, r] = -3.0  # All sources off by default
            model.theta.data[src, r] = 3.0   # This source on
            
            # Initialize rule_target_logits: strong logit for target type
            model.rule_target_logits.data[r, :] = -3.0
            model.rule_target_logits.data[r, tgt] = 3.0
            
            # Initialize sign
            if sign == 'EXC':
                model.sign_logits.data[r] = 3.0   # Excitatory
                # V9: Asymmetric weight init - excitatory rules start with large W_exc
                model.w_exc_raw.data[r] = 0.5      # softplus(0.5) ≈ 0.97
                model.w_inh_raw.data[r] = -3.0     # softplus(-3) ≈ 0.049
            else:
                model.sign_logits.data[r] = -3.0  # Inhibitory
                # V9: Inhibitory rules start with large W_inh
                model.w_exc_raw.data[r] = -3.0     # softplus(-3) ≈ 0.049
                model.w_inh_raw.data[r] = 0.5      # softplus(0.5) ≈ 0.97
            
            # Initialize source_alpha to focus on this source
            model.source_alpha_logits.data[:, r] = -3.0
            model.source_alpha_logits.data[src, r] = 3.0
        
        # For any remaining unassigned rules, give weak random init
        for r in range(len(assignments), num_rules):
            print(f"  Rule {r}: (unassigned, random init)")
            model.theta.data[:, r] = torch.randn(num_types) * 0.3 - 1.0
            model.rule_target_logits.data[r, :] = torch.randn(num_types) * 0.3
            model.sign_logits.data[r] = torch.randn(1).item() * 1.5
        
        # Initialize base intensity: reduce for types with strong excitation sources
        # The observed rate includes both base and triggered events.
        # For targets of STRONG excitation (z > 80), reduce base significantly.
        excited_targets = {}  # tgt -> max z-score
        for score, sign, _, tgt in assignments:
            if sign == 'EXC' and score > 80:
                excited_targets[tgt] = max(excited_targets.get(tgt, 0), score)
        
        for k in range(num_types):
            rate = max(type_rates[k], 1e-4)
            if k in excited_targets:
                # Reduce base: assume significant portion comes from excitation
                rate = rate * 0.2  # 20% of observed rate as base
                rate = max(rate, 1e-3)  # Floor
            if rate < 20:
                b0_val = np.log(np.exp(rate) - 1) if rate > 0.01 else np.log(rate)
            else:
                b0_val = rate
            model.b0.data[k] = b0_val
        
        # Initialize kernel components with DIVERSE temporal scales
        C = model.num_components
        for r in range(num_rules):
            for k in range(num_types):
                for c in range(C):
                    # Spread peaks across the time window
                    # Component 0: fast (small peak, small width)
                    # Component 1: medium
                    # Component 2: slow (large peak, large width)
                    peak_ratio = (c + 0.5) / C  # 0.17, 0.5, 0.83 for C=3
                    width_ratio = (c + 1.0) / C  # 0.33, 0.67, 1.0 for C=3
                    # sigmoid^{-1}(x) = log(x / (1-x))
                    model.raw_width_logits.data[r, k, c] = np.log(width_ratio / (1 - width_ratio + 1e-6))
                    model.raw_peak_ratio_logits.data[r, k, c] = np.log(peak_ratio / (1 - peak_ratio + 1e-6))
        
        print(f"\nInitialized b0 → softplus → {torch.nn.functional.softplus(model.b0).data.cpu().numpy()}")
        print(f"Kernel components initialized with diverse temporal scales")
    
    return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./workspace/train/train_config_refactored.yaml')
    parser.add_argument('--data', type=str, default='./data/synthetic_data.pkl')
    parser.add_argument('--benchmark', action='store_true', help='Run training speed benchmark')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--start_tau', type=float, default=None)
    parser.add_argument('--end_tau', type=float, default=None)
    parser.add_argument('--lambda_ent', type=float, default=None)
    parser.add_argument('--lambda_resp_base', type=float, default=None)
    parser.add_argument('--base_resp_target', type=float, default=None)
    parser.add_argument('--use_base_hinge', action='store_true')
    parser.add_argument('--theta_noise_std_start', type=float, default=None)
    parser.add_argument('--theta_noise_std_end', type=float, default=None)
    parser.add_argument('--num_rules', type=int, default=None)
    parser.add_argument('--phase1_epochs', type=int, default=None,
                        help='Number of epochs for Phase 1 (base clamped, soft H)')
    parser.add_argument('--base_clamp_value', type=float, default=None,
                        help='Max base intensity during Phase 1')
    parser.add_argument('--base_release_epochs', type=int, default=None,
                        help='Epochs to linearly release base clamp in Phase 2')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    hcfg = cfg.get('hnstpp_refactored')
    if hcfg is None:
        hcfg = {}
    set_seed(args.seed)
    
    device = torch.device(f'cuda:{hcfg.get("gpu", 0)}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    with open(args.data, 'rb') as f:
        dataset_dict = pickle.load(f)
    
    train_data = TPPDataset(dataset_dict['train'])
    val_data = TPPDataset(dataset_dict['val'])
    metadata = dataset_dict.get('metadata', {'num_types': 4})
    
    # Scale times
    time_scale = metadata.get('mean_time_diff', 1.0)
    print(f"Time scale: {time_scale:.4f}")
    for dataset in [train_data.data, val_data.data]:
        for item in dataset:
            item['time'] = [t / time_scale for t in item['time']]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # DataLoader
    num_types = metadata.get('num_types', 4)
    pad_token_id = num_types
    batch_size = hcfg.get('batch_size', 32)
    
    # For benchmark mode, use smaller batch to avoid OOM
    if args.benchmark:
        batch_size = min(batch_size, 16)
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(pad_token_id),
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(pad_token_id),
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    
    # Model config
    start_tau = hcfg.get('start_tau', 5.0)
    end_tau = hcfg.get('end_tau', 1.0)
    if args.start_tau is not None:
        start_tau = float(args.start_tau)
    if args.end_tau is not None:
        end_tau = float(args.end_tau)
    lambda_ent = hcfg.get('lambda_ent', 0.0)
    lambda_resp_base = hcfg.get('lambda_resp_base', 0.0)
    base_resp_target = hcfg.get('base_resp_target', 0.3)
    use_base_hinge = bool(hcfg.get('use_base_hinge', False))
    if args.lambda_ent is not None:
        lambda_ent = float(args.lambda_ent)
    if args.lambda_resp_base is not None:
        lambda_resp_base = float(args.lambda_resp_base)
    if args.base_resp_target is not None:
        base_resp_target = float(args.base_resp_target)
    if args.use_base_hinge:
        use_base_hinge = True
    theta_noise_std_start = hcfg.get('theta_noise_std_start', hcfg.get('theta_noise_std', 0.0))
    theta_noise_std_end = hcfg.get('theta_noise_std_end', 0.0)
    if args.theta_noise_std_start is not None:
        theta_noise_std_start = float(args.theta_noise_std_start)
    if args.theta_noise_std_end is not None:
        theta_noise_std_end = float(args.theta_noise_std_end)

    model_config = {
        'num_types': num_types,
        'num_rules': hcfg.get('num_rules', 8),
        'num_components': hcfg.get('num_components', 3),
        'pad_token_id': pad_token_id,
        'start_tau': start_tau,
        'theta_noise_std': theta_noise_std_start,
        'epsilon': hcfg.get('epsilon', 1e-6),
        'i_max': hcfg.get('i_max', 20.0),
        'max_cap_percentile': hcfg.get('max_cap_percentile', 0.95),
        'max_cap_multiplier': hcfg.get('max_cap_multiplier', 5.0),
        'lambda_ent': lambda_ent,
        'lambda_resp_base': lambda_resp_base,
        'base_resp_target': base_resp_target,
        'use_base_hinge': use_base_hinge,
        'lambda_sparse': hcfg.get('lambda_sparse', 1e-3),
        'lambda_h_sparse': hcfg.get('lambda_h_sparse', hcfg.get('lambda_sparse', 1e-3)),
        'lambda_h_binary': hcfg.get('lambda_h_binary', 0.0),
        'lambda_h_card': hcfg.get('lambda_h_card', 0.0),
        'h_card_target': hcfg.get('h_card_target', 1.0),
        'lambda_w_sparse': hcfg.get('lambda_w_sparse', 0.0),
        'lambda_overlap': hcfg.get('lambda_overlap', 1e-2),
        'lambda_sign_binary': hcfg.get('lambda_sign_binary', 0.0),
        'lambda_ortho': hcfg.get('lambda_ortho', 1e-3),
        'lambda_head_entropy': hcfg.get('lambda_head_entropy', 1e-3),
        'lambda_head_ortho': hcfg.get('lambda_head_ortho', 0.0),
        'lambda_base_l1': hcfg.get('lambda_base_l1', 0.0),
        'sign_tau': hcfg.get('sign_tau', 1.0),
        'integral_method': hcfg.get('integral_method', 'approx'),
        'integral_num_points': int(hcfg.get('integral_num_points', 64)),
        'breakpoint_chunk_size': hcfg.get('breakpoint_chunk_size', 128),
        'init_w_raw': hcfg.get('init_w_raw', -2.0),
        'init_sign_std': hcfg.get('init_sign_std', 0.0),
        'init_b0': hcfg.get('init_b0', -3.0),
        'init_bias_raw': hcfg.get('init_bias_raw', 0.0),
        'init_w_exc': hcfg.get('init_w_exc', -1.0),
        'init_w_inh': hcfg.get('init_w_inh', -1.0),
        'lambda_inh_boost': hcfg.get('lambda_inh_boost', 0.0),
        'inh_margin': hcfg.get('inh_margin', 0.1),
        'lambda_sign_magnitude': hcfg.get('lambda_sign_magnitude', 0.0),
    }
    if args.num_rules is not None:
        model_config['num_rules'] = args.num_rules
    
    model = HNSTPPRefactored(model_config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set max_cap from training data
    all_time_diffs = []
    all_input_ids = []
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        all_time_diffs.append(batch['time_diffs'].flatten())
        all_input_ids.append(batch['input_ids'].flatten())
    all_time_diffs = torch.cat(all_time_diffs)
    all_input_ids = torch.cat(all_input_ids)
    model.set_max_cap(all_time_diffs, all_input_ids)

    # Data-driven structure initialization
    use_warmstart = bool(hcfg.get('warmstart_from_data', False))
    if use_warmstart:
        initialize_structure_from_data(
            model, train_data.data, num_types, time_scale,
            model_config['num_rules'], device
        )

    # Training setup
    total_epochs = hcfg.get('epochs', 50)
    lr = float(hcfg.get('learning_rate', 1e-3))
    grad_clip = float(hcfg.get('grad_clip', 5.0))
    weight_decay = float(hcfg.get('weight_decay', 1e-4))
    
    # Separate parameter groups for differentiated learning rates
    base_lr_mult = float(hcfg.get('base_lr_mult', 1.0))
    structure_lr_mult = float(hcfg.get('structure_lr_mult', 1.0))
    
    param_groups = [
        # Structure parameters: theta, rule_target_logits, source_alpha_logits, sign_logits
        {'params': [model.theta, model.rule_target_logits, model.source_alpha_logits, model.sign_logits],
         'lr': lr * structure_lr_mult, 'name': 'structure'},
        # Kernel parameters
        {'params': [model.raw_width_logits, model.raw_peak_ratio_logits, model.raw_mix_logits],
         'lr': lr, 'name': 'kernel'},
        # Weight and bias parameters (V8: separated exc/inh weights)
        {'params': [model.w_exc_raw, model.w_inh_raw, model.rule_bias_raw],
         'lr': lr, 'name': 'weight'},
        # Base intensity (can be slowed down)
        {'params': [model.b0],
         'lr': lr * base_lr_mult, 'name': 'base'},
    ]
    
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    
    total_steps = total_epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)

    print(f"\n=== Training HNSTPP-Refactored ===")
    print(f"Epochs: {total_epochs}, Batch size: {batch_size}")
    print(f"Num rules: {model_config['num_rules']}, Num components: {model_config['num_components']}")
    print(
        f"Integral: {model_config['integral_method']}"
        + (f" (num_points={model_config['integral_num_points']})" if model_config['integral_method'] != 'exact' else "")
    )
    print(f"Max cap: {model.max_cap:.4f}")
    print(f"lambda_ent: {lambda_ent}, lambda_resp_base: {lambda_resp_base}, base_resp_target: {base_resp_target}")
    print(f"theta_noise_std schedule: {theta_noise_std_start} -> {theta_noise_std_end}")
    print(f"Grad clip: {grad_clip}")
    print()
    
    # Benchmark mode
    if args.benchmark:
        model.train()
        
        # Warmup
        print("Warming up...")
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            model_output = model(batch, tau=start_tau, theta_noise_std=theta_noise_std_start)
            loss_dict = model.compute_loss(batch, model_output, tau=start_tau, include_aux_loss=True)
            loss_dict['total_loss'].backward()
            optimizer.zero_grad(set_to_none=True)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timed run
        print("Benchmarking...")
        num_steps = 20
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            if i >= num_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)
            model_output = model(batch, tau=start_tau, theta_noise_std=theta_noise_std_start)
            loss_dict = model.compute_loss(batch, model_output, tau=start_tau, include_aux_loss=True)
            loss_dict['total_loss'].backward()
            optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        steps_per_sec = num_steps / elapsed
        samples_per_sec = num_steps * batch_size / elapsed
        
        print(f"\n=== Benchmark Results ===")
        print(f"Steps: {num_steps}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Steps/sec: {steps_per_sec:.2f}")
        print(f"Samples/sec: {samples_per_sec:.2f}")
        print(f"Time per epoch (estimated): {len(train_loader) / steps_per_sec:.2f}s")
        return
    
    # Training loop
    best_val_nll = float('inf')
    base_save_dir = hcfg.get('model_save_path', './workspace/saved_models/synthetic/HNSTPP_refactored')
    run_name = args.run_name
    if run_name is None:
        run_name = f"seed{args.seed}"
    save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save run config
    phase1_epochs = int(hcfg.get('phase1_epochs', 0))
    base_clamp_value = float(hcfg.get('base_clamp_value', 0.0))
    base_release_epochs = int(hcfg.get('base_release_epochs', 10))
    if args.phase1_epochs is not None:
        phase1_epochs = int(args.phase1_epochs)
    if args.base_clamp_value is not None:
        base_clamp_value = float(args.base_clamp_value)
    if args.base_release_epochs is not None:
        base_release_epochs = int(args.base_release_epochs)

    run_config = {
        'model': model_config,
        'training': {
            'epochs': total_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'grad_clip': grad_clip,
            'start_tau': start_tau,
            'end_tau': end_tau,
            'theta_noise_std_start': theta_noise_std_start,
            'theta_noise_std_end': theta_noise_std_end,
            'phase1_epochs': phase1_epochs,
            'base_clamp_value': base_clamp_value,
            'base_release_epochs': base_release_epochs,
        },
        'seed': args.seed,
        'run_name': run_name,
    }
    with open(os.path.join(save_dir, 'run_config.yaml'), 'w') as f:
        yaml.safe_dump(run_config, f)

    log_path = os.path.join(save_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_nll', 'val_nll', 'event_ll', 'integral_loss',
            'num_events', 'integral_ratio', 'lambda_std_at_events', 'b_k_mean',
            'W_pos_mean', 'W_neg_mean', 'sign_gate_mean', 'source_card_mean', 'p_r_mean', 'mean_gamma_base', 'resp_entropy',
            'resp_base_penalty', 'exc_top1_ratio', 'inh_top1_ratio', 'attenuation_mean',
            'reg_loss', 'tau', 'theta_noise_std', 'b_max', 'phase', 'epoch_time'
        ])

    phase1_ended = False
    phase2_ended = False
    phase3_epochs = int(hcfg.get('phase3_epochs', 0))
    phase2_total_epochs = total_epochs - phase1_epochs - phase3_epochs
    weight_lr_mult = float(hcfg.get('weight_lr_mult_phase3', 3.0))

    if phase1_epochs > 0:
        print(f"\n=== Phase-based Training ===")
        print(f"Phase 1: {phase1_epochs} epochs (base_lr_mult={base_lr_mult:.2f}, soft H, tau={start_tau})")
        print(f"Phase 2: {phase2_total_epochs} epochs (base released, STE, tau annealing)")
        if phase3_epochs > 0:
            print(f"Phase 3: {phase3_epochs} epochs (structure frozen, weight LR x{weight_lr_mult})")
    
    for epoch in range(total_epochs):
        # Phase-based schedule
        in_phase1 = (epoch < phase1_epochs) and (phase1_epochs > 0)
        in_phase3 = (epoch >= total_epochs - phase3_epochs) and (phase3_epochs > 0)
        in_phase2 = not in_phase1 and not in_phase3

        if in_phase1:
            # Phase 1: Discovery - frozen/slow base, soft H, high tau
            phase_progress = epoch / max(1, phase1_epochs - 1)
            tau = start_tau  # Keep high tau throughout Phase 1
            use_ste = False
            b_max = base_clamp_value if base_clamp_value > 0 else None
        else:
            # Phase transition: unfreeze base LR when entering Phase 2
            if not phase1_ended and phase1_epochs > 0:
                phase1_ended = True
                # Set base param group LR to full
                for pg in optimizer.param_groups:
                    if pg.get('name') == 'base':
                        pg['lr'] = lr  # Full learning rate for base
                        print(f"  → Phase 2: unfreezing base LR to {lr}")
                    elif pg.get('name') == 'structure':
                        pg['lr'] = lr  # Reduce structure LR to normal
                        print(f"  → Phase 2: structure LR set to {lr}")
            
            # Phase 2: Refinement - release base, STE, anneal tau
            if phase1_epochs > 0:
                phase2_epoch = epoch - phase1_epochs
                phase2_progress = phase2_epoch / max(1, phase2_total_epochs - 1)
            else:
                phase2_progress = epoch / max(1, total_epochs - phase3_epochs - 1)
            
            tau = start_tau + (end_tau - start_tau) * phase2_progress
            use_ste = True
            
            # Gradually release base clamp
            if base_clamp_value > 0 and phase1_epochs > 0:
                phase2_epoch = epoch - phase1_epochs
                if phase2_epoch < base_release_epochs:
                    release_progress = phase2_epoch / max(1, base_release_epochs)
                    # Linearly increase b_max from clamp_value to effectively unclamped
                    b_max = base_clamp_value + (10.0 - base_clamp_value) * release_progress
                else:
                    b_max = None  # Unclamped
            else:
                b_max = None

        # Temperature annealing for theta noise
        progress = epoch / max(1, total_epochs - 1)
        theta_noise_std = theta_noise_std_start + (theta_noise_std_end - theta_noise_std_start) * progress
        model.tau = tau
        
        # Phase 3: Weight refinement - freeze structure, boost weight LR
        if in_phase3 and not phase2_ended:
            phase2_ended = True
            print(f"  → Phase 3: Freezing structure, boosting weight LR x{weight_lr_mult}")
            for pg in optimizer.param_groups:
                if pg.get('name') == 'structure':
                    pg['lr'] = 0.0  # Freeze structure
                elif pg.get('name') == 'weight':
                    pg['lr'] = lr * weight_lr_mult
                    print(f"    Weight LR: {pg['lr']:.5f}")
            # Override regularization for Phase 3
            model.config['lambda_overlap'] = float(hcfg.get('lambda_overlap_phase3', 0.0))
            model.config['lambda_sign_binary'] = 0.0
            model.config['lambda_sign_magnitude'] = 0.0
            model.config['lambda_h_card'] = 0.0
            model.config['lambda_h_binary'] = 0.0
        
        if in_phase3:
            tau = end_tau  # Fixed low tau
            use_ste = True
            b_max = None
        
        phase_str = "P1" if in_phase1 else ("P3" if in_phase3 else "P2")
        
        # Train
        epoch_start = time.time()
        current_scheduler = None if in_phase3 else scheduler
        train_metrics = train_epoch(
            model, train_loader, optimizer, current_scheduler, device, tau, theta_noise_std, grad_clip,
            b_max=b_max, use_ste=use_ste
        )
        epoch_time = time.time() - epoch_start
        
        # Validate (always unclamped for fair comparison)
        val_nll = validate(model, val_loader, device, tau=min(tau, 1.0), theta_noise_std=0.0)
        
        b_max_str = f"{b_max:.3f}" if b_max is not None else "None"
        print(f"Epoch {epoch+1:3d}/{total_epochs} [{phase_str}] | "
              f"Loss: {train_metrics['train_loss']:.4f} | NLL: {train_metrics['train_nll']:.4f} | "
              f"Val: {val_nll:.4f} | tau: {tau:.2f} | b_max: {b_max_str} | "
              f"b_k: {train_metrics['b_k_mean']:.4f} | W+: {train_metrics['W_pos_mean']:.3f} | W-: {train_metrics['W_neg_mean']:.3f} | "
              f"p_r: {train_metrics['p_r_mean']:.4f} | γ_b: {train_metrics['mean_gamma_base']:.3f} | "
              f"atten: {train_metrics['attenuation_mean']:.3f} | inh_aux: {train_metrics.get('inh_aux_loss', 0):.4f} | "
              f"∇w: {train_metrics.get('w_grad_norm', 0):.4f} | {epoch_time:.1f}s")

        # Log metrics
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics['train_loss'],
                train_metrics['train_nll'],
                val_nll,
                train_metrics['event_ll'],
                train_metrics['integral_loss'],
                train_metrics['num_events'],
                train_metrics['integral_ratio'],
                train_metrics['lambda_std_at_events'],
                train_metrics['b_k_mean'],
                train_metrics['W_pos_mean'],
                train_metrics['W_neg_mean'],
                train_metrics['sign_gate_mean'],
                train_metrics['source_card_mean'],
                train_metrics['p_r_mean'],
                train_metrics['mean_gamma_base'],
                train_metrics['resp_entropy'],
                train_metrics['resp_base_penalty'],
                train_metrics['exc_top1_ratio'],
                train_metrics['inh_top1_ratio'],
                train_metrics['attenuation_mean'],
                train_metrics['reg_loss'],
                tau,
                theta_noise_std,
                b_max if b_max is not None else '',
                phase_str,
                epoch_time,
            ])
        
        # Save best
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f"  → Saved best model (val_nll: {val_nll:.4f})")
    
    # Always save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    
    print(f"\n=== Training Complete ===")
    print(f"Best Val NLL: {best_val_nll:.4f}")
    
    # Print learned structure
    print("\n=== Learned Structure ===")
    structure = model.get_structure()
    print(f"Base intensities: {structure['b_k'].numpy()}")
    print(f"W_pos: {structure['W_pos'].numpy()}")
    print(f"W_neg: {structure['W_neg'].numpy()}")
    print(f"Sign gate: {structure['sign_gate'].numpy()}")
    print(f"Rule bias: {structure['rule_bias'].numpy()}")
    
    # Print rule details
    H = structure['H'].numpy()
    Head = structure['Head'].numpy()
    for r in range(model_config['num_rules']):
        sources = [int(k) for k in range(num_types) if H[k, r] > 0.5]
        target = int(Head[r].argmax())
        w_p = float(structure['W_pos'][r])
        w_n = float(structure['W_neg'][r])
        gate = float(structure['sign_gate'][r])
        sign = "EXC" if gate > 0.5 else "INH"
        w_eff = w_p if gate > 0.5 else w_n
        print(f"  Rule {r}: {sources} → {target} [{sign}] W={w_eff:.4f} (W+={w_p:.4f}, W-={w_n:.4f}, gate={gate:.3f})")
        # Print kernel info for active sources
        for src in sources:
            peaks = structure['peaks'][r, src].numpy()
            widths = structure['widths'][r, src].numpy()
            mixes = structure['mix_weights'][r, src].numpy()
            print(f"    Kernel[src={src}]: peaks={peaks}, widths={widths}, mix={mixes}")


if __name__ == '__main__':
    main()

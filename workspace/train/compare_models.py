import argparse
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from workspace.train.dataset_transformer import EventSeqDataset, collate_fn
from workspace.train.generate_dataset import load_config
from workspace.models.TransformerHawkes import TransformerHawkes
from workspace.models.HNSTPP import HNSTPP
from workspace.models.NSTPP import NSTPP
from workspace.models.HRTPP import HRTPP
from workspace.models.CLNN import CLNN
from workspace.models.CLUSTER import CLUSTER
from workspace.models.TELLER import TELLER
from workspace.train.train_transformer_hawkes import compute_metrics_with_mc
import os
import pandas as pd

# Suppress nested-tensor prototype warnings (benign for our use)
warnings.filterwarnings("ignore", message=".*nested.*tensor.*", category=UserWarning)
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage", category=UserWarning)


def eval_transformer(checkpoint, cfg, device):
    tcfg = cfg.get('transformer', {})
    pad_id = tcfg.get('pad_token_id', cfg.get('pad_token_id', 0))
    num_types = cfg.get('num_event_types')

    # Prefer using the config saved with the checkpoint to match shapes
    state = torch.load(checkpoint, map_location=device)
    saved_cfg = state.get('cfg', {})
    tcfg_saved = saved_cfg.get('transformer', {}) if saved_cfg else {}

    d_model = tcfg_saved.get('embed_dim', saved_cfg.get('embed_dim', tcfg.get('embed_dim', 128)))
    nhead = tcfg_saved.get('num_heads', saved_cfg.get('num_heads', tcfg.get('num_heads', 4)))
    if nhead % 2 != 0:
        nhead += 1

    if d_model % nhead != 0:
        d_model = d_model + (nhead - (d_model % nhead))

    model = TransformerHawkes(num_types=num_types, pad_id=pad_id, d_model=d_model, nhead=nhead).to(device)
    # load state permissively to handle older checkpoints (backwards-compatible)
    missing, unexpected = model.load_state_dict(state['model_state'], strict=False), None
    if isinstance(missing, tuple):
        # PyTorch <=1.12 returns (missing_keys, unexpected_keys) when strict=False; newer versions return None
        pass
    # Note: we intentionally load permissively; parameters missing from older checkpoints are initialized randomly
    model.eval()

    ds = EventSeqDataset(cfg['path']['data_path'], split='test')
    loader = DataLoader(ds, batch_size=tcfg.get('batch_size', 32), collate_fn=lambda b: collate_fn(b, pad_id))

    total = {'nll':0.0, 'mae':0.0, 'top_k':0.0, 'n':0}
    times = []
    eval_mc = cfg.get('transformer', {}).get('eval_mc_samples', 200)
    top_k = tcfg.get('top_k', 1)
    with torch.no_grad():
        for batch in loader:
            t0 = time.time()
            # use compute_metrics_with_mc for consistent evaluation
            batch_nll, batch_mae, batch_topk, batch_count = compute_metrics_with_mc(model, batch, pad_id, device, num_mc_samples=eval_mc, top_k=top_k)
            t1 = time.time()
            times.append(t1-t0)

            if batch_count == 0:
                continue

            total['nll'] += batch_nll
            total['mae'] += batch_mae
            total['top_k'] += batch_topk
            total['n'] += batch_count

    avg = {
        'nll_per_event': total['nll']/max(1,total['n']),
        'mae': total['mae']/max(1,total['n']),
        'top_k': total['top_k']/max(1,total['n']),
        'mean_inference_latency_batch': sum(times)/len(times),
        'n_params': model.count_parameters()
    }
    return avg


def eval_hnstpp(checkpoint_dir, cfg, device):
    # load HNSTPP checkpoint
    state = torch.load(checkpoint_dir + '/best_model.pt', map_location=device)
    # instantiate model with hnstpp config
    hcfg = cfg.get('hnstpp', {})
    model_cfg = dict(hcfg)
    model_cfg['num_types'] = cfg.get('num_types', 5)
    model_cfg['num_event_types'] = cfg.get('num_event_types', 5)
    model_cfg['pad_token_id'] = hcfg.get('pad_token_id', cfg.get('pad_token_id', 0))
    top_k = hcfg.get('top_k', 1)

    model = HNSTPP(model_cfg).to(device)
    model.load_state_dict(state)
    model.eval()

    from workspace.train.train_hnstpp import get_collate_fn
    # reuse generate_dataset load
    data_path = cfg['path']['data_path']
    import pickle
    with open(data_path, 'rb') as f:
        dset = pickle.load(f)
    test_data = dset['test']
    pad_id = model_cfg['pad_token_id']
    collate = get_collate_fn(pad_id)
    loader = DataLoader(test_data, batch_size=hcfg.get('batch_size', 32), collate_fn=collate)

    total = {'type_ce':0.0, 'time_nll':0.0, 'mae':0.0, 'top_k':0.0, 'n_events':0}
    times = []
    with torch.no_grad():
        for batch in loader:
            t0 = time.time()
            # move tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}
            model_output = model(batch)
            t1 = time.time()
            times.append(t1-t0)

            # Build target next-event type and dt from input_ids/time_diffs
            input_ids = batch['input_ids']  # (B,L)
            time_diffs = batch['time_diffs']  # (B,L)
            mask = batch.get('attention_mask', torch.ones_like(input_ids, dtype=torch.float))

            B, L = input_ids.shape
            tgt_type = torch.full((B, L), pad_id, dtype=torch.long, device=device)
            tgt_dt = torch.zeros((B, L), dtype=torch.float, device=device)
            for i in range(B):
                ln = int(mask[i].sum().item())
                if ln <= 1:
                    continue
                ids = input_ids[i, :ln]
                t = torch.cumsum(time_diffs[i, :ln], dim=0)
                tgt_type[i, :ln-1] = ids[1:]
                tgt_dt[i, :ln-1] = t[1:] - t[:-1]

            preds = model.predict_event_rates(batch, model_output=model_output)
            total_rate = preds['total_rate']  # (B,L)
            probs = preds['prob']  # (B,L,num_types)
            pred_dt = preds['pred_dt']  # (B,L)

            valid = (mask == 1) & (tgt_type != pad_id)
            n = valid.sum().item()
            if n == 0:
                continue

            # Type cross entropy (negative log prob of true type)
            # Replace pad ids in target before gather to avoid out-of-bounds indexing, then mask
            safe_tgt_type = tgt_type.clone()
            safe_tgt_type[safe_tgt_type == pad_id] = 0
            # Gather prob for true type using gather to avoid advanced index broadcasting issues
            true_probs_all = probs.gather(-1, safe_tgt_type.unsqueeze(-1)).squeeze(-1)  # (B,L)
            true_probs = true_probs_all[valid]
            type_ce = -torch.log(true_probs + 1e-9).sum().item()

            # Time NLL: use exponential approx: -log(total_rate) + total_rate * dt
            rate_valid = total_rate[valid]
            dt_valid = tgt_dt[valid]
            time_nll = (-torch.log(rate_valid + 1e-9) + rate_valid * dt_valid).sum().item()

            mae = torch.abs(pred_dt[valid] - dt_valid).sum().item()

            preds_topk = probs.topk(top_k, dim=-1).indices
            topk = (preds_topk == tgt_type.unsqueeze(-1)).any(dim=-1).sum().item()

            total['type_ce'] += type_ce
            total['time_nll'] += time_nll
            total['mae'] += mae
            total['top_k'] += topk
            total['n_events'] += n

    avg = {
        'nll_per_event': (total['type_ce'] + total['time_nll']) / max(1, total['n_events']),
        'mae': total['mae'] / max(1, total['n_events']),
        'top_k': total['top_k'] / max(1, total['n_events']),
        'mean_inference_latency_batch': sum(times)/len(times),
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./workspace/train/train_config.yaml')
    parser.add_argument('--transformer_ckpt', type=str, default='/workspace/workspace/saved_models/synthetic/transformer/best_model.pt')
    parser.add_argument('--hnstpp_ckpt_dir', type=str, default='/workspace/workspace/saved_models/synthetic/HNSTPP')
    parser.add_argument('--run_baselines', action='store_true', help='Evaluate implemented baseline models (untrained) on synthetic test set')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    results = []

    # Evaluate transformer and HNSTPP if checkpoints provided
    if args.transformer_ckpt:
        res_t = eval_transformer(args.transformer_ckpt, cfg, device)
        res_t['model'] = 'transformer'
        results.append(res_t)

    if args.hnstpp_ckpt_dir:
        res_h = eval_hnstpp(args.hnstpp_ckpt_dir, cfg, device)
        res_h['model'] = 'hnstpp'
        results.append(res_h)

    # Optionally run lightweight eval of implemented baselines (untrained weights)
    if args.run_baselines:
        # Prepare dataset and loader (test split)
        ds = EventSeqDataset(cfg['path']['data_path'], split='test')
        pad_id = cfg.get('pad_token_id', 0)
        loader = DataLoader(ds, batch_size=cfg.get('batch_size', 32), collate_fn=lambda b: collate_fn(b, pad_id))

        def eval_model_instance(model, loader, device, pad_id, top_k=1):
            model.to(device).eval()
            total = {'type_ce':0.0, 'time_nll':0.0, 'mae':0.0, 'top_k':0.0, 'n_events':0}
            times = []
            with torch.no_grad():
                for batch in loader:
                    t0 = time.time()
                    # adapt batch for BaseTPP models
                    batch_device = {k: v.to(device) for k, v in batch.items()}
                    wrapped = {
                        'events': batch_device['types'],
                        'times': batch_device['times'],
                        'mask': batch_device['mask']
                    }
                    model_output = model(wrapped)
                    t1 = time.time(); times.append(t1-t0)

                    # compute metrics
                    tgt_type = batch_device['tgt_type']
                    tgt_dt = batch_device['tgt_dt']
                    mask = batch_device['mask']
                    valid = (~mask) & (tgt_type != pad_id)
                    n = valid.sum().item()
                    if n == 0:
                        continue
                    # Type CE
                    logits = model_output['type_logits']
                    logits_flat = logits.view(-1, logits.size(-1))
                    tgt_flat = tgt_type.view(-1).clone()
                    # replace pad ids with zero to avoid out-of-range targets; will mask them below
                    tgt_flat[tgt_flat == pad_id] = 0
                    ce = nn.functional.cross_entropy(logits_flat, tgt_flat, reduction='none').view(tgt_type.shape)
                    total['type_ce'] += ce[valid].sum().item()
                    # Time NLL using exponential approx
                    rate = model_output['time_rate']
                    rate_valid = rate[valid]
                    dt_valid = tgt_dt[valid]
                    time_nll = (-torch.log(rate_valid + 1e-9) + rate_valid * dt_valid).sum().item()
                    total['time_nll'] += time_nll
                    # MAE: use predicted dt = 1 / rate
                    pred_dt = (1.0 / (rate + 1e-9))[valid]
                    total['mae'] += torch.abs(pred_dt - dt_valid).sum().item()
                    # Top-k
                    k = min(top_k, logits.size(-1))
                    topk_idx = logits.topk(k, dim=-1).indices
                    topk_matches = (topk_idx == tgt_type.unsqueeze(-1)).any(dim=-1)
                    total['top_k'] += topk_matches[valid].sum().item()
                    total['n_events'] += n

            avg = {
                'nll_per_event': (total['type_ce'] + total['time_nll'])/max(1,total['n_events']),
                'mae': total['mae']/max(1,total['n_events']),
                'top_k': total['top_k']/max(1,total['n_events']),
                'mean_inference_latency_batch': sum(times)/len(times) if len(times)>0 else 0.0,
                'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            return avg

        # instantiate baseline models using config
        baseline_cfg = {
            'num_event_types': cfg.get('num_event_types', 5),
            'pad_token_id': cfg.get('pad_token_id', 0),
            'embed_dim': cfg.get('embed_dim', 32),
            'hidden_dim': cfg.get('hidden_dim', 64)
        }

        models_to_run = [
            ('hrtpp', HRTPP(baseline_cfg)),
            ('clnn', CLNN(baseline_cfg)),
            ('cluster', CLUSTER(baseline_cfg)),
            ('teller', TELLER(baseline_cfg)),
        ]
        
        # NSTPP needs additional config for the neuro-symbolic components
        nstpp_cfg = dict(baseline_cfg)
        nstpp_cfg['num_types'] = cfg.get('num_types', 5)
        nstpp_cfg['num_heads'] = cfg.get('hnstpp', {}).get('num_heads', 3)
        nstpp_cfg['start_tau'] = cfg.get('hnstpp', {}).get('start_tau', 1.0)
        nstpp_cfg['mc_samples'] = cfg.get('hnstpp', {}).get('mc_samples', 5)
        nstpp_cfg['lambda_sparse'] = cfg.get('hnstpp', {}).get('lambda_sparse', 1e-3)
        models_to_run.insert(0, ('nstpp', NSTPP(nstpp_cfg)))

        for name, mdl in models_to_run:
            print('Evaluating', name)
            res = eval_model_instance(mdl, loader, device, pad_id, top_k=cfg.get('top_k', 1))
            res['model'] = name
            results.append(res)

    if results:
        df = pd.DataFrame(results)
        os.makedirs('./workspace/benchmarks', exist_ok=True)
        out_csv = './workspace/benchmarks/transformer_vs_hnstpp.csv'
        df.to_csv(out_csv, index=False)
        print('Saved results to', out_csv)
        print(df)
    else:
        print('No models evaluated. Provide checkpoints with --transformer_ckpt and/or --hnstpp_ckpt_dir')

if __name__ == '__main__':
    main()

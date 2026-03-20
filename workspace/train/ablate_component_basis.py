"""Ablation runner for the fixed-target component-basis prototype."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time

import numpy as np
import torch
import yaml

from workspace.models.HNSTPP_component_basis import HNSTPPComponentBasis
from workspace.train.component_basis_init import build_component_basis_structure
from workspace.train.train_hnstpp import TPPDataset, get_collate_fn


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(data_path: str):
    with open(data_path, "rb") as f:
        dd = pickle.load(f)
    metadata = dd["metadata"]
    time_scale = float(metadata.get("mean_time_diff", 1.0))
    train = dd["train"]
    val = dd["val"]
    for ds in (train, val):
        for item in ds:
            item["time"] = [float(t) / time_scale for t in item["time"]]
    return train, val, metadata


def estimate_max_cap(train_data: list[dict], num_types: int, pct: float = 0.95, mul: float = 5.0) -> float:
    ds = TPPDataset(train_data)
    batch = get_collate_fn(num_types)([ds[i] for i in range(min(64, len(ds)))])
    valid = batch["input_ids"] != num_types
    td = batch["time_diffs"][valid].cpu().float().numpy()
    td = td[td > 0]
    if len(td) == 0:
        return 1.0
    return float(np.percentile(td, pct * 100.0) * mul)


def gt_rules_from_config(config: dict):
    gt = set()
    for rule in config.get("rules", []):
        srcs = tuple(sorted(int(s) for s in rule["condition"].keys()))
        if float(rule.get("W_pos", 0.0)) > float(rule.get("W_neg", 0.0)):
            sign = "exc"
        elif float(rule.get("W_neg", 0.0)) > float(rule.get("W_pos", 0.0)):
            sign = "inh"
        else:
            continue
        gt.add((srcs, sign, int(rule["target"])))
    return gt


def predict_rules(model: HNSTPPComponentBasis, threshold: float = 1e-3):
    st = model.get_structure()
    atom_sources = st["atom_sources"].numpy().astype(int)
    w_exc = st["w_exc"].numpy()
    w_inh = st["w_inh"].numpy()
    target = int(st["fixed_target"])

    preds = set()
    for atom_id in range(atom_sources.shape[0]):
        srcs = tuple(sorted(int(s) for s in atom_sources[atom_id] if int(s) >= 0))
        if not srcs:
            continue
        we = float(w_exc[atom_id])
        wi = float(w_inh[atom_id])
        if max(we, wi) <= threshold:
            continue
        sign = "exc" if we >= wi else "inh"
        preds.add((srcs, sign, target))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixed_target", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta_t", type=float, default=0.1)
    ap.add_argument("--max_lag", type=float, default=10.0)
    ap.add_argument("--source_pool_topk", type=int, default=8)
    ap.add_argument("--max_source_order", type=int, default=3)
    ap.add_argument("--topk_per_order", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    train_data, _, metadata = prepare_data(args.data)
    cfg = load_yaml(args.config)
    device = torch.device(args.device)

    max_cap = estimate_max_cap(train_data, int(metadata["num_types"]))
    print(f"estimated max_cap = {max_cap:.4f}")

    t0 = time.time()
    structure = build_component_basis_structure(
        train_data,
        int(metadata["num_types"]),
        fixed_target=int(args.fixed_target),
        device=device,
        delta_t=float(args.delta_t),
        max_lag=float(args.max_lag),
        source_pool_topk=int(args.source_pool_topk),
        max_source_order=int(args.max_source_order),
        topk_per_order=int(args.topk_per_order),
    )
    elapsed = time.time() - t0

    structure_cfg = {
        "source_defs": [
            {
                "source": int(sd.source),
                "score": float(sd.score),
                "peak": float(sd.peak),
                "width": float(sd.width),
                "beta": float(sd.beta),
                "alpha": float(sd.alpha),
            }
            for sd in structure["source_defs"]
        ],
        "atoms": [
            {
                "atom_id": int(atom.atom_id),
                "component_id": int(atom.component_id),
                "sources": [int(s) for s in atom.sources],
                "order": int(atom.order),
                "score": float(atom.score),
                "sign": int(atom.sign),
            }
            for atom in structure["atoms"]
        ],
        "base_rates": [float(x) for x in structure["base_rates"]],
        "w_exc": [float(x) for x in structure["w_exc"]],
        "w_inh": [float(x) for x in structure["w_inh"]],
    }

    model = HNSTPPComponentBasis(
        {
            "num_types": int(metadata["num_types"]),
            "pad_token_id": int(metadata["num_types"]),
            "fixed_target": int(args.fixed_target),
            "max_cap": float(max_cap),
            "structure": structure_cfg,
            "epsilon": 1e-6,
            "i_max": 20.0,
            "integral_num_points": 64,
        }
    ).to(device)

    # Runtime smoke test on one batch.
    ds = TPPDataset(train_data)
    batch = get_collate_fn(int(metadata["num_types"]))([ds[i] for i in range(min(16, len(ds)))])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        loss_dict = model.compute_loss(batch, model())
    print(f"runtime nll = {float(loss_dict['nll_loss'].item()):.6f}")

    gt = gt_rules_from_config(cfg)
    preds = predict_rules(model)
    hit = sorted(gt & preds)
    miss = sorted(gt - preds)
    extra = sorted(preds - gt)

    out = {
        "elapsed_sec": elapsed,
        "recall": len(hit) / max(len(gt), 1),
        "hit_count": len(hit),
        "gt_count": len(gt),
        "num_sources": len(structure_cfg["source_defs"]),
        "num_atoms": len(structure_cfg["atoms"]),
        "hit": hit,
        "miss": miss,
        "extra": extra,
    }
    print("===RESULT===")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

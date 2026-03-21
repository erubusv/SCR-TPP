"""Initializer-only evaluation for fixed-target synthetic overlap datasets."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import yaml

from workspace.models.HNSTPP import HNSTPP
from workspace.train.train_hnstpp import TPPDataset, get_collate_fn
from workspace.train.wh_init import wiener_hopf_initialize


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(num_types: int, num_rules: int):
    cfg = {
        "num_types": num_types,
        "num_rules": num_rules,
        "num_bins": 20,
        "pad_token_id": num_types,
        "start_tau": 5.0,
        "epsilon": 1.0e-6,
        "i_max": 20.0,
        "max_cap_percentile": 0.95,
        "max_cap_multiplier": 5.0,
        "integral_num_points": 64,
        "sign_tau": 1.0,
        "init_b0": -3.0,
        "init_bias_raw": 0.0,
        "init_w_exc": -1.0,
        "init_w_inh": -1.0,
        "init_sign_std": 0.0,
        "lambda_ortho": 0.0,
        "lambda_h_sparse": 0.0,
        "lambda_h_binary": 0.0,
        "lambda_w_sparse": 0.0,
        "lambda_overlap": 0.0,
        "lambda_sign_binary": 0.0,
        "lambda_head_entropy": 0.0,
        "lambda_smooth": 0.0,
    }
    return HNSTPP(cfg)


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


def set_model_cap(model: HNSTPP, train_data: list[dict]):
    ds = TPPDataset(train_data)
    collate_fn = get_collate_fn(model.pad_token_id)
    batch = collate_fn([ds[i] for i in range(min(64, len(ds)))])
    model.set_max_cap(batch["time_diffs"].flatten(), batch["input_ids"].flatten())


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


def node_label(node: int, fixed_target: int) -> str:
    if int(node) == int(fixed_target):
        return "T"
    if 0 <= int(node) < 26:
        return chr(ord("A") + int(node))
    return f"S{int(node)}"


def format_rule(rule, fixed_target: int) -> str:
    srcs, sign, target = rule
    lhs = " and ".join(node_label(int(s), fixed_target) for s in srcs)
    rhs = node_label(int(target), fixed_target)
    sign_txt = "excitation" if sign == "exc" else "inhibition"
    return f"{lhs} -> {rhs} : {sign_txt}"


def print_rule_block(title: str, rules, fixed_target: int):
    print(title)
    if not rules:
        print("  - none")
        return
    for rule in rules:
        print(f"  - {format_rule(rule, fixed_target)}")


def print_ranked_rule_block(title: str, rows, fixed_target: int):
    print(title)
    if not rows:
        print("  - none")
        return
    for row in rows:
        print(f"  - {format_rule(row['rule'], fixed_target)} [score={row['score']:.6f}, margin={row['margin']:.6f}]")


def predicted_rules(model: HNSTPP):
    st = model.get_structure()
    rows = []

    H = st["H"].numpy().astype(int)
    Head = st["Head"].numpy().astype(int)
    W_pos = st["W_pos"].numpy()
    W_neg = st["W_neg"].numpy()
    family_active = st["family_hyp_active"].numpy().astype(bool)
    family_target = st["family_hyp_target"].numpy().astype(int)
    family_sources = st["family_hyp_sources"].numpy().astype(int)
    family_weight = st["family_hyp_weight"].numpy()

    for r in range(H.shape[1]):
        if family_active[r]:
            srcs = tuple(sorted(int(s) for s in family_sources[r] if int(s) >= 0))
            if srcs:
                score = float(np.max(family_weight[r]))
                if score > 1.0e-4:
                    rows.append(
                        {
                            "rule": (srcs, "inh", int(family_target[r])),
                            "score": score,
                            "margin": score,
                            "order": len(srcs),
                        }
                    )
            continue

        srcs = tuple(sorted(int(s) for s in np.where(H[:, r] > 0)[0].tolist()))
        if not srcs:
            continue
        tgt = int(np.argmax(Head[r]))
        w_exc = float(W_pos[r])
        w_inh = float(W_neg[r])
        score = max(w_exc, w_inh)
        if score <= 1.0e-4:
            continue
        sign = "exc" if w_exc >= w_inh else "inh"
        rows.append(
            {
                "rule": (srcs, sign, tgt),
                "score": float(score),
                "margin": float(abs(w_exc - w_inh)),
                "order": len(srcs),
            }
        )

    return rows, int(family_active.sum())


def select_top_rules(rows, k: int):
    best_by_lhs = {}
    for row in rows:
        key = (row["rule"][0], row["rule"][2])
        cur = best_by_lhs.get(key)
        sort_key = (float(row["score"]), float(row["margin"]), int(row["order"]))
        if cur is None or sort_key > cur["sort_key"]:
            best_by_lhs[key] = {**row, "sort_key": sort_key}
    ranked = sorted(best_by_lhs.values(), key=lambda x: x["sort_key"], reverse=True)
    return ranked[: max(int(k), 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_rules", type=int, default=12)
    ap.add_argument("--fixed_target", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wh_lambda_l1", type=float, default=1.0e-3)
    ap.add_argument("--wh_pair_min_support", type=int, default=2)
    ap.add_argument("--wh_pair_topk_per_target", type=int, default=20)
    ap.add_argument("--wh_max_source_order", type=int, default=3)
    ap.add_argument("--wh_phase3_support_tau", type=float, default=20.0)
    ap.add_argument("--wh_delta_t", type=float, default=0.1)
    ap.add_argument("--wh_max_lag", type=float, default=10.0)
    ap.add_argument("--wh_phase4_source_cap", type=float, default=None)
    ap.add_argument("--wh_phase4_fixed_kernel_shape", type=str, default="flat")
    ap.add_argument("--wh_phase4_fixed_kernel_peak", type=float, default=1.0)
    ap.add_argument("--wh_phase4_fixed_kernel_sigma", type=float, default=0.75)
    ap.add_argument("--wh_phase4_fixed_kernel_amp", type=float, default=0.5)
    ap.add_argument("--wh_phase4_fixed_kernel_trunc_mult", type=float, default=3.0)
    ap.add_argument("--freeze_kernel_after_init", action="store_true")
    ap.add_argument("--wh_phase6_protect_best_inh_single", action="store_true")
    ap.add_argument("--no-wh_phase6_protect_best_inh_single", dest="wh_phase6_protect_best_inh_single", action="store_false")
    ap.add_argument("--wh_phase6_family_conflict_penalty", type=float, default=0.35)
    ap.add_argument("--prediction_k", type=int, default=None)
    ap.set_defaults(wh_phase6_protect_best_inh_single=True)
    args = ap.parse_args()

    set_seed(args.seed)
    train_data, _, metadata = prepare_data(args.data)
    cfg = load_yaml(args.config)

    device = torch.device(args.device)
    model = build_model(int(metadata["num_types"]), int(args.num_rules)).to(device)
    set_model_cap(model, train_data)

    t0 = time.time()
    wiener_hopf_initialize(
        model,
        train_data,
        int(metadata["num_types"]),
        device,
        delta_t=float(args.wh_delta_t),
        max_lag=float(args.wh_max_lag),
        lambda_l1=float(args.wh_lambda_l1),
        pair_min_support=int(args.wh_pair_min_support),
        pair_topk_per_target=int(args.wh_pair_topk_per_target),
        max_source_order=int(args.wh_max_source_order),
        phase3_support_tau=float(args.wh_phase3_support_tau),
        phase4_source_cap=args.wh_phase4_source_cap,
        phase4_fixed_kernel_shape=str(args.wh_phase4_fixed_kernel_shape),
        phase4_fixed_kernel_peak=float(args.wh_phase4_fixed_kernel_peak),
        phase4_fixed_kernel_sigma=float(args.wh_phase4_fixed_kernel_sigma),
        phase4_fixed_kernel_amp=float(args.wh_phase4_fixed_kernel_amp),
        phase4_fixed_kernel_trunc_mult=float(args.wh_phase4_fixed_kernel_trunc_mult),
        freeze_kernel_after_init=bool(args.freeze_kernel_after_init),
        phase6_protect_best_inh_single=bool(args.wh_phase6_protect_best_inh_single),
        phase6_family_conflict_penalty=float(args.wh_phase6_family_conflict_penalty),
        fixed_target=int(args.fixed_target),
    )
    elapsed = time.time() - t0

    gt = gt_rules_from_config(cfg)
    prediction_k = int(args.prediction_k) if args.prediction_k is not None else len(gt)
    ranked_preds, family_active = predicted_rules(model)
    selected_rows = select_top_rules(ranked_preds, prediction_k)
    preds = {row["rule"] for row in selected_rows}
    hit = sorted(gt & preds)
    miss = sorted(gt - preds)
    extra = sorted(preds - gt)

    print("===RULE REPORT===")
    print_rule_block("True rules:", sorted(gt), int(args.fixed_target))
    print_ranked_rule_block(f"Top-{prediction_k} predicted rules:", selected_rows, int(args.fixed_target))
    print_rule_block("Matched rules:", hit, int(args.fixed_target))
    print_rule_block("Missing rules:", miss, int(args.fixed_target))
    print_rule_block("Extra predicted rules:", extra, int(args.fixed_target))

    out = {
        "elapsed_sec": elapsed,
        "recall": len(hit) / max(len(gt), 1),
        "hit_count": len(hit),
        "gt_count": len(gt),
        "prediction_k": prediction_k,
        "family_active": family_active,
        "pred": [row["rule"] for row in selected_rows],
        "hit": hit,
        "miss": miss,
        "extra": extra,
    }
    print("===RESULT===")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

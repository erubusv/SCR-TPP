from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from data.synthetic import create_rules_from_config, generate_exp_data
import rule_dependent_kernel_active_set as rd
from runtime_resources import configure_runtime_resources, default_cpu_threads


ROOT = Path("/workspace")
PAPER_SUITE = ROOT / "data" / "paper_suite"
RESULTS_PATH = PAPER_SUITE / "paper_benchmark_results.json"


BENCHMARKS = [
    {"name": "logical_clean_plus", "config": PAPER_SUITE / "paper_logical_clean_plus.yaml"},
    {"name": "logical_shared", "config": PAPER_SUITE / "paper_logical_shared.yaml"},
    {"name": "logical_context", "config": PAPER_SUITE / "paper_logical_context.yaml"},
    {"name": "kernel_triangular", "config": PAPER_SUITE / "paper_kernel_robustness_triangular.yaml"},
    {"name": "kernel_exponential", "config": PAPER_SUITE / "paper_kernel_robustness_exponential.yaml"},
    {"name": "kernel_gaussian", "config": PAPER_SUITE / "paper_kernel_robustness_gaussian.yaml"},
    {"name": "num_predicates_10", "config": PAPER_SUITE / "paper_num_predicates_10.yaml"},
    {"name": "num_predicates_20", "config": PAPER_SUITE / "paper_num_predicates_20.yaml"},
    {"name": "num_predicates_30", "config": PAPER_SUITE / "paper_num_predicates_30.yaml"},
    {"name": "ablation_excitation_only", "config": PAPER_SUITE / "paper_ablation_excitation_only.yaml"},
    {"name": "ablation_inhibition_only", "config": PAPER_SUITE / "paper_ablation_inhibition_only.yaml"},
    {"name": "ablation_mixed_sign", "config": PAPER_SUITE / "paper_ablation_mixed_sign.yaml"},
]


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_dataset(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    output_path = Path(cfg["path"]["output_path"])
    rules = create_rules_from_config(cfg)
    data = generate_exp_data(
        rules=rules,
        num_samples=cfg.get("num_samples", 5000),
        time_horizon=cfg.get("time_horizon", 100.0),
        base_intensities=cfg.get("base_intensity", {}),
        max_len=cfg.get("max_len", 512),
        seed=cfg.get("seed"),
    )
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    rng.shuffle(data)
    n = len(data)
    dataset = {
        "train": data[: int(0.8 * n)],
        "val": data[int(0.8 * n) : int(0.9 * n)],
        "test": data[int(0.9 * n) :],
        "metadata": {
            "num_types": cfg["num_event_types"],
            "config": cfg,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    return output_path


def evaluate_config(
    config_path: Path,
    device_name: str,
    *,
    regenerate_dataset: bool = True,
) -> dict:
    cfg = load_yaml(config_path)
    data_path = generate_dataset(config_path) if regenerate_dataset else Path(cfg["path"]["output_path"])
    target = int(cfg["rules"][0]["target"])

    train, val, metadata = rd.load_dataset(str(data_path))
    config = rd.load_yaml(str(config_path))
    gt = rd.gt_rules_from_config(config)
    num_types = int(metadata["num_types"])
    time_horizon = float(config.get("time_horizon", max(max(seq["time"]) for seq in train + val)))
    source_ids = tuple(sorted(k for k in range(num_types) if k != target))

    train_arrays = rd.build_seq_event_arrays(train, num_types)
    val_arrays = rd.build_seq_event_arrays(val, num_types)
    global_kernels = rd.estimate_source_kernels(
        train_arrays,
        source_ids=source_ids,
        target=target,
        max_lag=10.0,
        num_bins=40,
        num_knots=7,
        time_horizon=float(time_horizon),
    )
    grid_step = rd.auto_grid_step(global_kernels)

    tr_event_seq, tr_event_times = rd.collect_target_events(train, target=target)
    va_event_seq, va_event_times = rd.collect_target_events(val, target=target)
    tr_grid_seq, tr_grid_times, tr_grid_w = rd.build_midpoint_grid(train, time_horizon=float(time_horizon), step=grid_step)
    va_grid_seq, va_grid_times, va_grid_w = rd.build_midpoint_grid(val, time_horizon=float(time_horizon), step=grid_step)

    a_train_event, _, _, _ = rd.build_global_activity(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        kernels=global_kernels,
        source_ids=source_ids,
        tr_event_seq=tr_event_seq,
        tr_event_times=tr_event_times,
        tr_grid_seq=tr_grid_seq,
        tr_grid_times=tr_grid_times,
        va_event_seq=va_event_seq,
        va_event_times=va_event_times,
        va_grid_seq=va_grid_seq,
        va_grid_times=va_grid_times,
    )
    basis_cache = rd.SourceBasisCache(
        source_ids=source_ids,
        knots=next(iter(global_kernels.values())).knots,
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        train_event_seq_ids=tr_event_seq,
        train_event_times=tr_event_times,
        train_grid_seq_ids=tr_grid_seq,
        train_grid_times=tr_grid_times,
        val_event_seq_ids=va_event_seq,
        val_event_times=va_event_times,
        val_grid_seq_ids=va_grid_seq,
        val_grid_times=va_grid_times,
    )
    from conjunctive_rule_initializer import build_event_lag_bin_cache

    train_event_lag_bin_cache = build_event_lag_bin_cache(
        train_arrays,
        source_ids=source_ids,
        target_seq_ids=tr_event_seq,
        target_times=tr_event_times,
        max_lag=10.0,
        num_bins=40,
    )
    subsets = rd.subset_list(source_ids, 3)
    src_to_col_global = {int(s): j for j, s in enumerate(source_ids)}
    rule_heights = rd.initialize_rule_specific_heights(
        subsets=subsets,
        source_ids=source_ids,
        global_kernels=global_kernels,
        global_activity_event=a_train_event,
        src_to_col_global=src_to_col_global,
        train_arrays=train_arrays,
        train_event_lag_bin_cache=train_event_lag_bin_cache,
        max_lag=10.0,
        num_bins=40,
        time_horizon=float(time_horizon),
    )
    init_arrays = rd.compute_rule_feature_arrays(
        subsets=subsets,
        basis_cache=basis_cache,
        rule_heights=rule_heights,
        kernels=global_kernels,
    )

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch_basis_cache = rd.TorchBasisCache(basis_cache, device)

    bic, mu, exc_params, inh_params, rule_heights, active_rules = rd.run_active_set(
        subsets=subsets,
        init_arrays=init_arrays,
        rule_heights=rule_heights,
        basis_cache=basis_cache,
        grid_weights_train=np.asarray(tr_grid_w, dtype=np.float64),
        grid_weights_val=np.asarray(va_grid_w, dtype=np.float64),
        max_rules=12,
        opt_steps=60,
        lr=0.05,
        device=device,
        torch_basis_cache=torch_basis_cache,
        penalize_kernel_df=False,
    )
    bic, mu, exc_params, inh_params, rule_heights, active_rules = rd.family_attribution_refine(
        active_rules=active_rules,
        subsets=subsets,
        basis_cache=basis_cache,
        rule_heights=rule_heights,
        mu=float(mu),
        exc_params=exc_params,
        inh_params=inh_params,
        grid_weights_train=np.asarray(tr_grid_w, dtype=np.float64),
        grid_weights_val=np.asarray(va_grid_w, dtype=np.float64),
        device=device,
        torch_basis_cache=torch_basis_cache,
        opt_steps=60,
        lr=0.05,
        passes=1,
    )
    bic, mu, exc_params, inh_params, rule_heights, active_rules, chosen_scale = rd.choose_post_prune_by_penalty_scale(
        active_rules=active_rules,
        subsets=subsets,
        basis_cache=basis_cache,
        rule_heights=rule_heights,
        mu=float(mu),
        exc_params=exc_params,
        inh_params=inh_params,
        grid_weights_train=np.asarray(tr_grid_w, dtype=np.float64),
        grid_weights_val=np.asarray(va_grid_w, dtype=np.float64),
        device=device,
        torch_basis_cache=torch_basis_cache,
        opt_steps=60,
        lr=0.05,
        min_order=2,
        penalty_scale_grid=[0.6, 0.8, 1.0],
    )

    preds = rd.summarize_results(subsets=subsets, exc_params=exc_params, inh_params=inh_params, target=target)
    matched = sorted(gt & preds)
    missing = sorted(gt - preds)
    extra = sorted(preds - gt)
    return {
        "data_path": str(data_path),
        "config_path": str(config_path),
        "target": int(target),
        "device": str(device),
        "bic": float(bic),
        "post_prune_penalty_scale": float(chosen_scale),
        "selected_rule_count": int(len(preds)),
        "true_rule_count": int(len(gt)),
        "matched": [rd.format_rule(r, target) for r in matched],
        "missing": [rd.format_rule(r, target) for r in missing],
        "extra": [rd.format_rule(r, target) for r in extra],
        "predicted": [rd.format_rule(r, target) for r in sorted(preds)],
        "recall": float(len(matched)) / max(len(gt), 1),
        "precision": float(len(matched)) / max(len(preds), 1),
    }


def parse_name_filter(spec: str) -> set[str]:
    return {part.strip() for part in str(spec).split(",") if part.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("BENCHMARK_DEVICE", "cuda:0"))
    ap.add_argument("--cpu_threads", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--reuse_dataset", action="store_true")
    args = ap.parse_args()

    cpu_threads = configure_runtime_resources(None if int(args.cpu_threads) <= 0 else int(args.cpu_threads))
    requested = parse_name_filter(args.only)
    benchmarks = BENCHMARKS
    if requested:
        benchmarks = [item for item in BENCHMARKS if item["name"] in requested]
        missing = sorted(requested - {item["name"] for item in benchmarks})
        if missing:
            raise ValueError(f"unknown benchmark names: {missing}")

    print(
        json.dumps(
            {
                "cpu_threads": int(cpu_threads),
                "default_cpu_threads": int(default_cpu_threads()),
                "device": str(args.device),
                "reuse_dataset": bool(args.reuse_dataset),
                "benchmarks": [item["name"] for item in benchmarks],
            },
            indent=2,
        ),
        flush=True,
    )

    out = {}
    for item in benchmarks:
        name = item["name"]
        print("RUNNING", name, flush=True)
        out[name] = evaluate_config(item["config"], str(args.device), regenerate_dataset=not bool(args.reuse_dataset))
        print(json.dumps({name: out[name]}, indent=2), flush=True)
        RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print("WROTE", RESULTS_PATH)


if __name__ == "__main__":
    main()

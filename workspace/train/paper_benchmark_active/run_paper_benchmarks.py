from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.synthetic import (
    create_rules_from_config,
    generate_canonical_loglink_data,
    generate_exact_log_rewrite_data,
    generate_exp_data,
)
import rule_dependent_kernel_active_set as rd
from runtime_resources import configure_runtime_resources, default_cpu_threads


ROOT = Path(__file__).resolve().parents[3]
PAPER_SUITE = ROOT / "data" / "paper_suite"
PAPER_CONFIGS = PAPER_SUITE / "configs"
FINAL_SYNTHETIC_CONFIGS = PAPER_CONFIGS / "final_logical_tpp"
RESULTS_PATH = PAPER_SUITE / "results" / "paper_benchmark_results_final_logical_tpp.json"
FINAL_FAMILY_ATTRIBUTION_PASSES = 1
FINAL_POST_PRUNE_MIN_ORDER = 1
FINAL_POST_PRUNE_MAX_DROP_SIZE = 2
FINAL_POST_PRUNE_PENALTY_SCALE = 1.0
FINAL_COMPONENT_SAME_SIGN_ONLY = True


BENCHMARKS = [
    {"name": "logical_clean_plus", "config": FINAL_SYNTHETIC_CONFIGS / "logical_clean_plus.yaml"},
    {"name": "logical_shared", "config": FINAL_SYNTHETIC_CONFIGS / "logical_shared.yaml"},
    {"name": "logical_context", "config": FINAL_SYNTHETIC_CONFIGS / "logical_context.yaml"},
    {"name": "kernel_triangular", "config": FINAL_SYNTHETIC_CONFIGS / "kernel_triangular.yaml"},
    {"name": "kernel_exponential", "config": FINAL_SYNTHETIC_CONFIGS / "kernel_exponential.yaml"},
    {"name": "kernel_gaussian", "config": FINAL_SYNTHETIC_CONFIGS / "kernel_gaussian.yaml"},
    {"name": "num_predicates_10", "config": FINAL_SYNTHETIC_CONFIGS / "num_predicates_10.yaml"},
    {"name": "num_predicates_20", "config": FINAL_SYNTHETIC_CONFIGS / "num_predicates_20.yaml"},
    {"name": "ablation_excitation_only", "config": FINAL_SYNTHETIC_CONFIGS / "ablation_excitation_only.yaml"},
    {"name": "ablation_inhibition_only", "config": FINAL_SYNTHETIC_CONFIGS / "ablation_inhibition_only.yaml"},
    {"name": "ablation_mixed_sign", "config": FINAL_SYNTHETIC_CONFIGS / "ablation_mixed_sign.yaml"},
]


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        workspace_prefix = Path("/workspace")
        try:
            rel = path.relative_to(workspace_prefix)
        except ValueError:
            return path
        return ROOT / rel
    return ROOT / path


def resolve_compute_device(requested: str) -> torch.device:
    req = str(requested).strip().lower()
    if req in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(requested))


def _rule_sign(rule_cfg: dict) -> str | None:
    w_pos = float(rule_cfg.get("W_pos", 0.0))
    w_neg = float(rule_cfg.get("W_neg", 0.0))
    if w_pos > w_neg:
        return "exc"
    if w_neg > w_pos:
        return "inh"
    return None


def _truth_rule_lookup(config: dict) -> dict[tuple[tuple[int, ...], str, int], dict]:
    lookup: dict[tuple[tuple[int, ...], str, int], dict] = {}
    for rule_cfg in config.get("rules", []):
        sign = _rule_sign(rule_cfg)
        if sign is None:
            continue
        key = (
            tuple(sorted(int(s) for s in rule_cfg["condition"].keys())),
            sign,
            int(rule_cfg["target"]),
        )
        lookup[key] = rule_cfg
    return lookup


def _triangular_kernel_values(knots: np.ndarray, peak: float, width: float, mix_weight: float) -> np.ndarray:
    out = np.zeros_like(knots, dtype=np.float64)
    eps = 1e-12
    width = max(float(width), eps)
    peak = float(np.clip(float(peak), eps, width - eps))
    mask = (knots > 0.0) & (knots <= width)
    if not np.any(mask):
        return out
    dt = knots[mask]
    vals = np.where(
        dt <= peak,
        float(mix_weight) * (dt / peak),
        float(mix_weight) * ((width - dt) / max(width - peak, eps)),
    )
    out[mask] = np.maximum(vals, 0.0)
    return out


def _gaussian_kernel_values(
    knots: np.ndarray,
    peak: float,
    sigma: float,
    mix_weight: float,
    support_mult: float,
) -> np.ndarray:
    out = np.zeros_like(knots, dtype=np.float64)
    eps = 1e-12
    sigma = max(float(sigma), eps)
    peak = max(float(peak), eps)
    support = peak + max(float(support_mult), 0.0) * sigma
    mask = (knots > 0.0) & (knots <= support)
    if not np.any(mask):
        return out
    z = (knots[mask] - peak) / sigma
    out[mask] = float(mix_weight) * np.exp(-0.5 * z * z)
    return out


def _exponential_kernel_values(
    knots: np.ndarray,
    peak: float,
    tau: float,
    mix_weight: float,
    support_mult: float,
) -> np.ndarray:
    out = np.zeros_like(knots, dtype=np.float64)
    eps = 1e-12
    tau = max(float(tau), eps)
    peak = max(float(peak), eps)
    support = peak + max(float(support_mult), 0.0) * tau
    mask = (knots > peak) & (knots <= support)
    if not np.any(mask):
        return out
    z = (knots[mask] - peak) / tau
    out[mask] = float(mix_weight) * np.exp(-z)
    return out


def _truth_kernel_heights(
    *,
    kernel_type: str,
    source_condition: dict,
    knots: np.ndarray,
) -> np.ndarray:
    peaks = list(source_condition.get("peaks", []))
    widths = list(source_condition.get("widths", []))
    mix_weights = list(source_condition.get("mix_weights", []))
    support_mults = list(source_condition.get("support_mults", []))
    n_comp = len(peaks)
    if len(widths) != n_comp or len(mix_weights) != n_comp:
        raise ValueError("kernel condition expects peaks/widths/mix_weights with equal lengths")
    if not support_mults:
        support_mults = [3.0] * n_comp
    elif len(support_mults) != n_comp:
        raise ValueError("kernel condition support_mults length must match peaks length")

    vals = np.zeros_like(knots, dtype=np.float64)
    for peak, width, mix_weight, support_mult in zip(peaks, widths, mix_weights, support_mults):
        if kernel_type == "gaussian":
            vals += _gaussian_kernel_values(knots, peak, width, mix_weight, support_mult)
        elif kernel_type == "exponential":
            vals += _exponential_kernel_values(knots, peak, width, mix_weight, support_mult)
        else:
            vals += _triangular_kernel_values(knots, peak, width, mix_weight)
    return rd.normalize_piecewise_area(knots, vals)


def _compute_kernel_recovery(
    *,
    config: dict,
    target: int,
    matched: list[tuple[tuple[int, ...], str, int]],
    subsets: list[tuple[int, ...]],
    rule_heights: dict[tuple[int, int], np.ndarray],
    knots: np.ndarray,
) -> dict:
    subset_to_idx = {tuple(int(s) for s in subset): int(idx) for idx, subset in enumerate(subsets)}
    truth_lookup = _truth_rule_lookup(config)
    details: list[dict] = []
    l1_vals: list[float] = []
    l2_vals: list[float] = []
    knots = np.asarray(knots, dtype=np.float64)

    for rule in sorted(matched):
        subset, _sign, _target = rule
        idx = subset_to_idx.get(tuple(int(s) for s in subset))
        truth_rule = truth_lookup.get(rule)
        if idx is None or truth_rule is None:
            continue
        kernel_type = str(truth_rule.get("kernel", config.get("kernel", "triangular")))
        for src in subset:
            src = int(src)
            est_raw = np.asarray(rule_heights[(int(idx), src)], dtype=np.float64)
            est = rd.normalize_piecewise_area(knots, est_raw)
            source_condition = truth_rule["condition"].get(src)
            if source_condition is None:
                source_condition = truth_rule["condition"].get(str(src))
            if source_condition is None:
                raise KeyError(f"missing source {src} condition for truth rule {rule}")
            truth = _truth_kernel_heights(
                kernel_type=kernel_type,
                source_condition=source_condition,
                knots=knots,
            )
            l1 = float(np.trapz(np.abs(est - truth), x=knots))
            l2 = float(math.sqrt(float(np.trapz((est - truth) ** 2, x=knots))))
            l1_vals.append(l1)
            l2_vals.append(l2)
            details.append(
                {
                    "rule": rd.format_rule(rule, target),
                    "source": int(src),
                    "kernel_type": str(kernel_type),
                    "l1": l1,
                    "l2": l2,
                    "knots": [float(x) for x in knots],
                    "estimated": [float(x) for x in est],
                    "truth": [float(x) for x in truth],
                }
            )

    return {
        "num_rule_source_pairs": int(len(details)),
        "mean_l1": float(np.mean(l1_vals)) if l1_vals else 0.0,
        "mean_l2": float(np.mean(l2_vals)) if l2_vals else 0.0,
        "max_l1": float(np.max(l1_vals)) if l1_vals else 0.0,
        "max_l2": float(np.max(l2_vals)) if l2_vals else 0.0,
        "details": details,
    }


def generate_dataset(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    output_path = resolve_repo_path(cfg["path"]["output_path"])
    rules = create_rules_from_config(cfg)
    generation_model = str(cfg.get("synthetic_generation_model", "")).strip()
    if generation_model == "canonical_loglink":
        data = generate_canonical_loglink_data(
            rules=rules,
            num_samples=cfg.get("num_samples", 5000),
            time_horizon=cfg.get("time_horizon", 100.0),
            base_intensities=cfg.get("base_intensity", {}),
            max_len=cfg.get("max_len", 512),
            seed=cfg.get("seed"),
        )
    elif generation_model == "exact_log_multiplicative_rewrite":
        data = generate_exact_log_rewrite_data(
            rules=rules,
            num_samples=cfg.get("num_samples", 5000),
            time_horizon=cfg.get("time_horizon", 100.0),
            base_intensities=cfg.get("base_intensity", {}),
            max_len=cfg.get("max_len", 512),
            seed=cfg.get("seed"),
        )
    else:
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
    forward_variant: str = "baseline",
) -> dict:
    t_start = time.perf_counter()
    cfg = load_yaml(config_path)
    data_path = generate_dataset(config_path) if regenerate_dataset else resolve_repo_path(cfg["path"]["output_path"])
    target = int(cfg["rules"][0]["target"])

    train, val, metadata = rd.load_dataset(str(data_path))
    config = rd.load_yaml(str(config_path))
    intensity_model = str(config.get("intensity_model", "multiplicative"))
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

    device = resolve_compute_device(device_name)
    torch_basis_cache = rd.TorchBasisCache(basis_cache, device)

    forward_variant = str(forward_variant).strip().lower() or "baseline"
    forward_kwargs = dict(
        subsets=subsets,
        init_arrays=init_arrays,
        score_arrays=init_arrays,
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
        num_val_sequences=len(val),
        intensity_model=str(intensity_model),
    )
    if forward_variant == "seed_partition":
        bic, mu, exc_params, inh_params, rule_heights, active_rules = rd.run_active_set_seed_partition(
            **forward_kwargs,
        )
    else:
        bic, mu, exc_params, inh_params, rule_heights, active_rules = rd.run_active_set(
            **forward_kwargs,
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
        passes=int(FINAL_FAMILY_ATTRIBUTION_PASSES),
        num_val_sequences=len(val),
        intensity_model=str(intensity_model),
    )
    bic, mu, exc_params, inh_params, rule_heights, active_rules = rd.post_prune_irreducible_rules(
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
        min_order=int(FINAL_POST_PRUNE_MIN_ORDER),
        max_drop_size=int(FINAL_POST_PRUNE_MAX_DROP_SIZE),
        penalize_kernel_df=False,
        penalty_scale=float(FINAL_POST_PRUNE_PENALTY_SCALE),
        num_val_sequences=len(val),
        intensity_model=str(intensity_model),
    )
    bic, mu, exc_params, inh_params, rule_heights, active_rules = rd.component_subset_search(
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
        penalize_kernel_df=False,
        penalty_scale=float(FINAL_POST_PRUNE_PENALTY_SCALE),
        num_val_sequences=len(val),
        same_sign_only=bool(FINAL_COMPONENT_SAME_SIGN_ONLY),
        intensity_model=str(intensity_model),
    )
    preds = rd.summarize_results(subsets=subsets, exc_params=exc_params, inh_params=inh_params, target=target)
    matched = sorted(gt & preds)
    missing = sorted(gt - preds)
    extra = sorted(preds - gt)
    kernel_recovery = _compute_kernel_recovery(
        config=config,
        target=target,
        matched=matched,
        subsets=subsets,
        rule_heights=rule_heights,
        knots=np.asarray(next(iter(global_kernels.values())).knots, dtype=np.float64),
    )
    return {
        "benchmark": str(config_path.stem).replace(".yaml", ""),
        "data_path": str(data_path),
        "config_path": str(config_path),
        "target": int(target),
        "device": str(device),
        "intensity_model": str(intensity_model),
        "forward_variant": str(forward_variant),
        "bic": float(bic),
        "elapsed_sec": float(time.perf_counter() - t_start),
        "family_attribution_passes": int(FINAL_FAMILY_ATTRIBUTION_PASSES),
        "post_prune_kernel_df": False,
        "post_prune_penalty_scale": float(FINAL_POST_PRUNE_PENALTY_SCALE),
        "post_prune_min_order": int(FINAL_POST_PRUNE_MIN_ORDER),
        "post_prune_max_drop_size": int(FINAL_POST_PRUNE_MAX_DROP_SIZE),
        "component_subset_search": True,
        "component_same_sign_only": bool(FINAL_COMPONENT_SAME_SIGN_ONLY),
        "selected_rule_count": int(len(preds)),
        "true_rule_count": int(len(gt)),
        "matched": [rd.format_rule(r, target) for r in matched],
        "missing": [rd.format_rule(r, target) for r in missing],
        "extra": [rd.format_rule(r, target) for r in extra],
        "predicted": [rd.format_rule(r, target) for r in sorted(preds)],
        "recall": float(len(matched)) / max(len(gt), 1),
        "precision": float(len(matched)) / max(len(preds), 1),
        "kernel_recovery": kernel_recovery,
    }


def parse_name_filter(spec: str) -> set[str]:
    return {part.strip() for part in str(spec).split(",") if part.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("BENCHMARK_DEVICE", "auto"))
    ap.add_argument("--cpu_threads", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--reuse_dataset", action="store_true")
    ap.add_argument("--result_path", default="")
    ap.add_argument("--forward_variant", default=os.environ.get("BENCHMARK_FORWARD_VARIANT", "baseline"))
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
                "forward_variant": str(args.forward_variant),
                "benchmarks": [item["name"] for item in benchmarks],
            },
            indent=2,
        ),
        flush=True,
    )

    out = {}
    results_path = RESULTS_PATH if not str(args.result_path).strip() else resolve_repo_path(str(args.result_path).strip())
    results_path.parent.mkdir(parents=True, exist_ok=True)
    for item in benchmarks:
        name = item["name"]
        print("RUNNING", name, flush=True)
        out[name] = evaluate_config(
            item["config"],
            str(args.device),
            regenerate_dataset=not bool(args.reuse_dataset),
            forward_variant=str(args.forward_variant),
        )
        print(json.dumps({name: out[name]}, indent=2), flush=True)
        results_path.write_text(json.dumps(out, indent=2))
    print("WROTE", results_path)


if __name__ == "__main__":
    main()

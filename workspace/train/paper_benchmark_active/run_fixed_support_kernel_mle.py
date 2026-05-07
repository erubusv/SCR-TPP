from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/workspace")
CONFIG_DIR = ROOT / "data" / "paper_suite" / "configs" / "hetero_source_2000_adjusted"
BASE_RESULT_DIR = Path(
    os.environ.get(
        "BASE_RESULT_DIR",
        str(ROOT / "data" / "paper_suite" / "results" / "scr_tpp_support_full33"),
    )
)
DEFAULT_RESULT_DIR = ROOT / "data" / "paper_suite" / "results" / "fixed_support_mhat_kernel_mle_full33_20260503"
SEEDS = (111, 222, 333)
CPU_THREADS_PER_WORKER = 12
OPT_STEPS = 300
KNOT_STRATEGY = "lag_quantile"
FINAL_KERNEL_MLE_SMOOTHNESS_RIDGE = 0.0
FINAL_KERNEL_NORMALIZATION = "area"
MAX_WITNESS_FRONTIER_PASSES = 1

BENCHMARKS = (
    "logical_clean_plus",
    "logical_shared",
    "logical_context",
    "kernel_triangular",
    "kernel_exponential",
    "kernel_gaussian",
    "num_predicates_10",
    "num_predicates_20",
    "ablation_excitation_only",
    "ablation_inhibition_only",
    "ablation_mixed_sign",
)

# Scheduling estimates for fixed-support final kernel MLE only.
EST_SEC = {
    "logical_clean_plus": 35,
    "logical_shared": 30,
    "logical_context": 40,
    "kernel_triangular": 35,
    "kernel_exponential": 35,
    "kernel_gaussian": 45,
    "num_predicates_10": 35,
    "num_predicates_20": 55,
    "ablation_excitation_only": 35,
    "ablation_inhibition_only": 35,
    "ablation_mixed_sign": 40,
}


def _install_paths() -> None:
    for path in (
        "/workspace",
        "/workspace/workspace/train",
        "/workspace/workspace/train/paper_benchmark_active",
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault(
        "PYTHONPATH",
        "/workspace:/workspace/workspace/train:/workspace/workspace/train/paper_benchmark_active",
    )


def m_hat_from_n_eff(n_eff: int) -> int:
    raw = max(3, int(math.floor(float(n_eff) ** (1.0 / 3.0))))
    if raw % 2 == 0:
        raw -= 1
    return max(3, int(raw))


def _source_token(token: str) -> int:
    token = str(token).strip()
    if len(token) == 1 and token.isalpha():
        return ord(token.upper()) - ord("A")
    if token.upper().startswith("X") and token[1:].isdigit():
        return int(token[1:])
    if token.isdigit():
        return int(token)
    raise ValueError(f"cannot parse source token: {token!r}")


def parse_formatted_rule(text: str, target: int) -> tuple[tuple[int, ...], str, int]:
    body, rest = str(text).split("->", 1)
    sign = "exc" if "excitation" in rest else "inh"
    parts = [part.strip() for part in re.split(r"\s+and\s+", body.strip()) if part.strip()]
    return (tuple(sorted(_source_token(part) for part in parts)), sign, int(target))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    tmp.replace(path)


def integration_grid_step_for_kernels(kernels: dict[int, Any]) -> float:
    spacings: list[np.ndarray] = []
    for kernel in kernels.values():
        delta = np.diff(np.asarray(kernel.knots, dtype=np.float64))
        delta = delta[delta > 1e-12]
        if delta.size:
            spacings.append(delta)
    if not spacings:
        return 0.5
    typical_spacing = float(np.median(np.concatenate(spacings)))
    return float(np.clip(typical_spacing / 2.0, 0.1, 0.5))


def _strictly_increasing_knots(values: np.ndarray, *, max_lag: float, num_knots: int) -> np.ndarray:
    knots = np.asarray(values, dtype=np.float64).copy()
    knots[0] = 0.0
    knots[-1] = float(max_lag)
    if int(num_knots) <= 2:
        return knots
    min_gap = max(float(max_lag), 1.0) * 1e-9
    for i in range(1, int(num_knots) - 1):
        lower = float(knots[i - 1]) + min_gap
        upper = float(max_lag) - float(int(num_knots) - 1 - i) * min_gap
        knots[i] = min(max(float(knots[i]), lower), upper)
    return knots


def collect_conditional_rule_source_lags(
    train_arrays: list[dict[int, np.ndarray]],
    *,
    subset: tuple[int, ...],
    source: int,
    target: int,
    max_lag: float,
) -> np.ndarray:
    lags: list[np.ndarray] = []
    source = int(source)
    others = tuple(int(src) for src in subset if int(src) != source)
    for by_type in train_arrays:
        tgt = by_type.get(int(target), np.zeros((0,), dtype=np.float64))
        src_events = by_type.get(source, np.zeros((0,), dtype=np.float64))
        if tgt.size == 0 or src_events.size == 0:
            continue
        other_events = {
            int(other): by_type.get(int(other), np.zeros((0,), dtype=np.float64))
            for other in others
        }
        for t in tgt:
            ok = True
            for other, events in other_events.items():
                if events.size == 0:
                    ok = False
                    break
                left_o = np.searchsorted(events, float(t) - float(max_lag), side="left")
                right_o = np.searchsorted(events, float(t), side="left")
                if right_o <= left_o:
                    ok = False
                    break
            if not ok:
                continue
            left = np.searchsorted(src_events, float(t) - float(max_lag), side="left")
            right = np.searchsorted(src_events, float(t), side="left")
            if right <= left:
                continue
            dts = float(t) - src_events[left:right]
            dts = dts[(dts > 0.0) & (dts <= float(max_lag))]
            if dts.size:
                lags.append(np.asarray(dts, dtype=np.float64))
    if not lags:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(lags).astype(np.float64, copy=False)


def conditional_rule_source_knots(
    train_arrays: list[dict[int, np.ndarray]],
    *,
    subset: tuple[int, ...],
    source: int,
    target: int,
    max_lag: float,
    num_knots: int,
    fallback_knots: np.ndarray,
) -> np.ndarray:
    num_knots = int(num_knots)
    lags = collect_conditional_rule_source_lags(
        train_arrays,
        subset=tuple(int(x) for x in subset),
        source=int(source),
        target=int(target),
        max_lag=float(max_lag),
    )
    if lags.size < max(1, num_knots - 2):
        return np.asarray(fallback_knots, dtype=np.float64).copy()
    probs = np.linspace(0.0, 1.0, num_knots, dtype=np.float64)
    knots = np.quantile(lags, probs, method="linear")
    return _strictly_increasing_knots(knots, max_lag=float(max_lag), num_knots=num_knots)


def prepare_state_with_num_knots(config_path: Path, device_name: str, *, dataset_seed: int, num_knots: int) -> dict[str, Any]:
    import run_paper_benchmarks as rpb
    import rule_dependent_kernel_active_set as rd
    from conjunctive_rule_initializer import (
        SourceBasisCache,
        auto_grid_step,
        build_midpoint_grid,
        build_seq_event_arrays,
        collect_target_events,
        estimate_source_kernels,
        feasible_subset_list,
    )

    cfg = rpb._config_with_dataset_seed(rpb.load_yaml(config_path), int(dataset_seed))
    data_path = rpb.resolve_repo_path(cfg["path"]["output_path"])
    train, val, metadata = rpb._load_rule_discovery_dataset(data_path)
    config = metadata.get("config", {}) or cfg
    target = int(config["rules"][0]["target"])
    num_types = int(metadata["num_types"])
    source_ids = tuple(sorted(k for k in range(num_types) if k != target))
    time_horizon = float(config.get("time_horizon", 120.0))
    max_lag = rpb._config_max_lag(config)

    train_arrays = build_seq_event_arrays(train, num_types)
    val_arrays = build_seq_event_arrays(val, num_types)
    global_exc = estimate_source_kernels(
        train_arrays,
        source_ids=source_ids,
        target=target,
        max_lag=max_lag,
        num_bins=40,
        num_knots=int(num_knots),
        time_horizon=time_horizon,
        sign="exc",
        knot_strategy=KNOT_STRATEGY,
    )
    global_inh = estimate_source_kernels(
        train_arrays,
        source_ids=source_ids,
        target=target,
        max_lag=max_lag,
        num_bins=40,
        num_knots=int(num_knots),
        time_horizon=time_horizon,
        sign="inh",
        knot_strategy=KNOT_STRATEGY,
    )
    if KNOT_STRATEGY == "lag_quantile":
        grid_step = integration_grid_step_for_kernels(global_exc)
    else:
        grid_step = auto_grid_step(global_exc)

    tr_event_seq, tr_event_times = collect_target_events(train, target=target)
    va_event_seq, va_event_times = collect_target_events(val, target=target)
    tr_grid_seq, tr_grid_times, tr_grid_w = build_midpoint_grid(train, time_horizon=time_horizon, step=grid_step)
    va_grid_seq, va_grid_times, va_grid_w = build_midpoint_grid(val, time_horizon=time_horizon, step=grid_step)
    source_knots = {int(src): np.asarray(global_exc[int(src)].knots, dtype=np.float64) for src in source_ids}
    knots = np.asarray(next(iter(source_knots.values())), dtype=np.float64)
    basis_cache = SourceBasisCache(
        source_ids=source_ids,
        knots=knots,
        source_knots=source_knots,
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
    subsets, _source_components, _source_presence_masks = feasible_subset_list(
        source_ids=source_ids,
        max_order=int(rpb.FINAL_MAX_RULE_ORDER),
        basis_cache=basis_cache,
    )
    rule_heights_exc = rd.initialize_rule_specific_heights(
        subsets=subsets,
        source_ids=source_ids,
        global_kernels=global_exc,
    )
    rule_heights_inh = rd.initialize_rule_specific_heights(
        subsets=subsets,
        source_ids=source_ids,
        global_kernels=global_inh,
    )
    device = rpb.resolve_compute_device(device_name)
    return {
        "config": config,
        "target": int(target),
        "subsets": subsets,
        "source_ids": source_ids,
        "train_arrays": train_arrays,
        "val_arrays": val_arrays,
        "basis_cache": basis_cache,
        "torch_basis_cache": rd.TorchBasisCache(basis_cache, device),
        "rule_heights": rule_heights_exc,
        "template_rule_heights": {"exc": rule_heights_exc, "inh": rule_heights_inh},
        "grid_weights_train": np.asarray(tr_grid_w, dtype=np.float64),
        "grid_weights_val": np.asarray(va_grid_w, dtype=np.float64),
        "num_sequences": int(len(val)),
        "knots": knots,
        "source_knots": source_knots,
        "device": device,
        "grid_step": float(grid_step),
        "data_path": str(data_path),
    }


def apply_conditional_rule_source_grid(
    prepared: dict[str, Any],
    *,
    active_rules: list[Any],
    num_knots: int,
) -> dict[str, Any]:
    import rule_dependent_kernel_active_set as rd
    from conjunctive_rule_initializer import SourceBasisCache

    source_knots = {
        int(src): np.asarray(knots, dtype=np.float64)
        for src, knots in prepared["source_knots"].items()
    }
    max_lag = float(np.asarray(prepared["knots"], dtype=np.float64)[-1])
    rule_source_knots: dict[tuple[int, int], np.ndarray] = {}
    rule_heights = {
        (int(idx), int(src)): np.asarray(vals, dtype=np.float64).copy()
        for (idx, src), vals in prepared["rule_heights"].items()
    }
    template_rule_heights = {
        str(sign): {
            (int(idx), int(src)): np.asarray(vals, dtype=np.float64).copy()
            for (idx, src), vals in height_map.items()
        }
        for sign, height_map in prepared["template_rule_heights"].items()
    }
    for ar in active_rules:
        idx = int(ar.idx)
        subset = tuple(int(src) for src in prepared["subsets"][idx])
        for src in subset:
            fallback = source_knots[int(src)]
            knots = conditional_rule_source_knots(
                prepared["train_arrays"],
                subset=subset,
                source=int(src),
                target=int(prepared["target"]),
                max_lag=max_lag,
                num_knots=int(num_knots),
                fallback_knots=fallback,
            )
            rule_source_knots[(idx, int(src))] = knots
            old_knots = source_knots[int(src)]
            for heights_map in (rule_heights, template_rule_heights["exc"], template_rule_heights["inh"]):
                old_h = np.asarray(heights_map[(idx, int(src))], dtype=np.float64)
                interp = np.interp(knots, old_knots, old_h, left=old_h[0], right=old_h[-1])
                heights_map[(idx, int(src))] = np.asarray(interp, dtype=np.float64)

    base_cache = prepared["basis_cache"]
    basis_cache = SourceBasisCache(
        source_ids=tuple(int(src) for src in prepared["source_ids"]),
        knots=np.asarray(prepared["knots"], dtype=np.float64),
        source_knots=source_knots,
        rule_source_knots=rule_source_knots,
        train_arrays=prepared["train_arrays"],
        val_arrays=prepared["val_arrays"],
        train_event_seq_ids=base_cache.train_event_seq_ids,
        train_event_times=base_cache.train_event_times,
        train_grid_seq_ids=base_cache.train_grid_seq_ids,
        train_grid_times=base_cache.train_grid_times,
        val_event_seq_ids=base_cache.val_event_seq_ids,
        val_event_times=base_cache.val_event_times,
        val_grid_seq_ids=base_cache.val_grid_seq_ids,
        val_grid_times=base_cache.val_grid_times,
    )
    out = dict(prepared)
    out["basis_cache"] = basis_cache
    out["torch_basis_cache"] = rd.TorchBasisCache(basis_cache, prepared["device"])
    out["rule_heights"] = rule_heights
    out["template_rule_heights"] = template_rule_heights
    out["rule_source_knots"] = rule_source_knots
    return out


def apply_max_witness_frontier_grid(
    prepared: dict[str, Any],
    *,
    active_rules: list[Any],
    fitted_result: Any,
    num_knots: int,
    n_eff: int,
) -> tuple[dict[str, Any], Any]:
    import rule_dependent_kernel_active_set as rd
    from conjunctive_rule_initializer import SourceBasisCache

    num_knots = int(num_knots)
    basis_cache = prepared["basis_cache"]
    source_knots = {
        int(src): np.asarray(knots, dtype=np.float64)
        for src, knots in prepared["source_knots"].items()
    }
    old_rule_source_knots = {
        (int(idx), int(src)): np.asarray(knots, dtype=np.float64)
        for (idx, src), knots in prepared.get("rule_source_knots", {}).items()
    }
    max_lag = float(np.asarray(prepared["knots"], dtype=np.float64)[-1])
    active_rules = sorted(list(active_rules), key=lambda ar: (int(ar.idx), str(ar.sign)))
    rule_specs = [
        (ar, tuple(int(src) for src in prepared["subsets"][int(ar.idx)]))
        for ar in active_rules
    ]
    sign = np.asarray([1.0 if str(ar.sign) == "exc" else -1.0 for ar, _subset in rule_specs], dtype=np.float64)
    beta = np.asarray(
        [
            float(fitted_result.exc_params.get(int(ar.idx), fitted_result.inh_params.get(int(ar.idx), 0.0)))
            for ar, _subset in rule_specs
        ],
        dtype=np.float64,
    )

    source_states: dict[tuple[int, int], dict[str, Any]] = {}
    for ar, subset in rule_specs:
        idx = int(ar.idx)
        for src in subset:
            src = int(src)
            knots = old_rule_source_knots.get((idx, src), source_knots.get(src, np.asarray(prepared["knots"], dtype=np.float64)))
            heights = np.asarray(fitted_result.rule_heights[(idx, src)], dtype=np.float64)
            tr_ev, tr_gr, _va_ev, _va_gr = basis_cache.arrays_for_rule_source(idx, src)
            ev_w, ev_q, ev_lag = _witness_response_and_winners(tr_ev, heights=heights, knots=knots)
            gr_w, gr_q, gr_lag = _witness_response_and_winners(tr_gr, heights=heights, knots=knots)
            source_states[(idx, src)] = {
                "event_w": ev_w,
                "grid_w": gr_w,
                "event_q": ev_q,
                "event_lag": ev_lag,
                "grid_q": gr_q,
                "grid_lag": gr_lag,
                "old_knots": knots,
            }

    event_features = []
    grid_features = []
    for ar, subset in rule_specs:
        idx = int(ar.idx)
        ev_parts = [source_states[(idx, int(src))]["event_w"] for src in subset]
        gr_parts = [source_states[(idx, int(src))]["grid_w"] for src in subset]
        event_features.append(np.prod(np.stack(ev_parts, axis=0), axis=0))
        grid_features.append(np.prod(np.stack(gr_parts, axis=0), axis=0))
    event_features_arr = np.stack(event_features, axis=0) if event_features else np.zeros((0, 0), dtype=np.float64)
    grid_features_arr = np.stack(grid_features, axis=0) if grid_features else np.zeros((0, 0), dtype=np.float64)
    signed_beta = sign * beta
    signed_grid = np.matmul(signed_beta, grid_features_arr) if grid_features_arr.size else np.zeros((0,), dtype=np.float64)
    lambda_grid = float(fitted_result.mu) * np.exp(np.clip(signed_grid, -40.0, 40.0))
    grid_weights = np.asarray(prepared["grid_weights_train"], dtype=np.float64)

    rule_source_knots: dict[tuple[int, int], np.ndarray] = {}
    for rule_row, (ar, subset) in enumerate(rule_specs):
        idx = int(ar.idx)
        for src in subset:
            src = int(src)
            state = source_states[(idx, src)]
            other_event = np.ones_like(state["event_w"], dtype=np.float64)
            other_grid = np.ones_like(state["grid_w"], dtype=np.float64)
            for other in subset:
                other = int(other)
                if other == src:
                    continue
                other_event *= source_states[(idx, other)]["event_w"]
                other_grid *= source_states[(idx, other)]["grid_w"]
            event_coeff = np.abs(signed_beta[rule_row] * other_event)
            grid_coeff = np.abs((-grid_weights * lambda_grid) * signed_beta[rule_row] * other_grid)
            ev_q = np.asarray(state["event_q"], dtype=np.int64)
            gr_q = np.asarray(state["grid_q"], dtype=np.int64)
            ev_weight = event_coeff[ev_q] if ev_q.size else np.zeros((0,), dtype=np.float64)
            gr_weight = grid_coeff[gr_q] if gr_q.size else np.zeros((0,), dtype=np.float64)
            lags = np.concatenate([
                np.asarray(state["event_lag"], dtype=np.float64),
                np.asarray(state["grid_lag"], dtype=np.float64),
            ])
            weights = np.concatenate([ev_weight, gr_weight])
            if lags.size < max(1, num_knots - 2) or float(np.sum(weights)) <= 1e-12:
                knots = np.asarray(state["old_knots"], dtype=np.float64).copy()
            else:
                knots = _local_kkt_interval_refined_knots(
                    np.asarray(state["old_knots"], dtype=np.float64),
                    lags,
                    weights,
                    max_lag=float(max_lag),
                )
            rule_source_knots[(idx, src)] = knots

    rule_heights = {
        (int(idx), int(src)): np.asarray(vals, dtype=np.float64).copy()
        for (idx, src), vals in fitted_result.rule_heights.items()
    }
    template_rule_heights = {
        str(sign_name): {
            (int(idx), int(src)): np.asarray(vals, dtype=np.float64).copy()
            for (idx, src), vals in height_map.items()
        }
        for sign_name, height_map in prepared["template_rule_heights"].items()
    }
    for (idx, src), knots in rule_source_knots.items():
        old_knots = old_rule_source_knots.get((idx, src), source_knots.get(src, np.asarray(prepared["knots"], dtype=np.float64)))
        for heights_map in (rule_heights, template_rule_heights["exc"], template_rule_heights["inh"]):
            if (idx, src) not in heights_map:
                continue
            old_h = np.asarray(heights_map[(idx, src)], dtype=np.float64)
            heights_map[(idx, src)] = np.asarray(np.interp(knots, old_knots, old_h, left=old_h[0], right=old_h[-1]), dtype=np.float64)

    base_cache = prepared["basis_cache"]
    new_basis_cache = SourceBasisCache(
        source_ids=tuple(int(src) for src in prepared["source_ids"]),
        knots=np.asarray(prepared["knots"], dtype=np.float64),
        source_knots=source_knots,
        rule_source_knots=rule_source_knots,
        train_arrays=prepared["train_arrays"],
        val_arrays=prepared["val_arrays"],
        train_event_seq_ids=base_cache.train_event_seq_ids,
        train_event_times=base_cache.train_event_times,
        train_grid_seq_ids=base_cache.train_grid_seq_ids,
        train_grid_times=base_cache.train_grid_times,
        val_event_seq_ids=base_cache.val_event_seq_ids,
        val_event_times=base_cache.val_event_times,
        val_grid_seq_ids=base_cache.val_grid_seq_ids,
        val_grid_times=base_cache.val_grid_times,
    )
    out = dict(prepared)
    out["basis_cache"] = new_basis_cache
    out["torch_basis_cache"] = rd.TorchBasisCache(new_basis_cache, prepared["device"])
    out["rule_heights"] = rule_heights
    out["template_rule_heights"] = template_rule_heights
    out["rule_source_knots"] = rule_source_knots
    out["max_witness_frontier"] = {
        "mode": "local_kkt_interval_refinement",
    }
    warm_result = rd.SupportEvalResult(
        bic=float(fitted_result.bic),
        mu=float(fitted_result.mu),
        exc_params=dict(fitted_result.exc_params),
        inh_params=dict(fitted_result.inh_params),
        rule_heights=rule_heights,
        arrays_out=None,
        active_rules=list(fitted_result.active_rules),
    )
    return out, warm_result


def _safe_trapz(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.trapz(np.asarray(values, dtype=np.float64), x=np.asarray(grid, dtype=np.float64)))


def _density_on_grid(values: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float]:
    vals = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    area = _safe_trapz(vals, grid)
    if area <= 1e-12:
        return np.zeros_like(vals, dtype=np.float64), float(area)
    return vals / float(area), float(area)


def _normalize_area_numpy(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    vals = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    knots = np.asarray(knots, dtype=np.float64)
    if vals.size <= 1:
        return np.ones_like(vals, dtype=np.float64)
    area = _safe_trapz(vals, knots)
    if area > 1e-12:
        return vals / float(area)
    span = max(float(knots[-1] - knots[0]), 1e-12)
    return np.ones_like(vals, dtype=np.float64) / span


def _witness_response_and_winners(
    data: Any,
    *,
    heights: np.ndarray,
    knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heights = _normalize_area_numpy(heights, knots)
    out = np.zeros((int(data.num_queries),), dtype=np.float64)
    if data.query_index.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return out, np.zeros((0,), dtype=np.int64), empty
    vals = (
        np.asarray(data.left_weight, dtype=np.float64) * heights[np.asarray(data.left_index, dtype=np.int64)]
        + np.asarray(data.right_weight, dtype=np.float64) * heights[np.asarray(data.right_index, dtype=np.int64)]
    )
    vals = np.maximum(vals, 0.0)
    qidx = np.asarray(data.query_index, dtype=np.int64)
    np.maximum.at(out, qidx, vals)
    candidate_lags = (
        np.asarray(data.left_weight, dtype=np.float64) * knots[np.asarray(data.left_index, dtype=np.int64)]
        + np.asarray(data.right_weight, dtype=np.float64) * knots[np.asarray(data.right_index, dtype=np.int64)]
    )
    winning = (out[qidx] > 1e-12) & (np.abs(vals - out[qidx]) <= 1e-10 * np.maximum(out[qidx], 1.0))
    return out, qidx[winning], candidate_lags[winning]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    probs = np.asarray(probs, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.zeros_like(probs, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    total = float(cdf[-1])
    if total <= 1e-12:
        return np.quantile(values, probs, method="linear")
    cdf = cdf / total
    return np.interp(np.clip(probs, 0.0, 1.0), cdf, values, left=values[0], right=values[-1])


def _local_kkt_interval_refined_knots(
    old_knots: np.ndarray,
    lags: np.ndarray,
    weights: np.ndarray,
    *,
    max_lag: float,
) -> np.ndarray:
    old_knots = np.asarray(old_knots, dtype=np.float64)
    num_knots = int(old_knots.size)
    if num_knots < 4:
        return old_knots.copy()
    lags = np.asarray(lags, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    mask = np.isfinite(lags) & np.isfinite(weights) & (weights > 0.0)
    lags = lags[mask]
    weights = weights[mask]
    if lags.size == 0 or float(np.sum(weights)) <= 1e-12:
        return old_knots.copy()

    bins = np.searchsorted(old_knots, lags, side="right") - 1
    bins = np.clip(bins, 0, num_knots - 2)
    interval_mass = np.bincount(bins, weights=weights, minlength=num_knots - 1).astype(np.float64)
    top_interval = int(np.argmax(interval_mass))
    if float(interval_mass[top_interval]) <= 1e-12:
        return old_knots.copy()
    in_top = bins == top_interval
    if not np.any(in_top):
        return old_knots.copy()
    new_knot = float(_weighted_quantile(lags[in_top], weights[in_top], np.asarray([0.5], dtype=np.float64))[0])
    left = float(old_knots[top_interval])
    right = float(old_knots[top_interval + 1])
    min_gap = max(float(max_lag), 1.0) * 1e-9
    if not (left + min_gap < new_knot < right - min_gap):
        return old_knots.copy()

    donor_scores: list[tuple[float, int]] = []
    protected = {top_interval, top_interval + 1}
    for knot_idx in range(1, num_knots - 1):
        if knot_idx in protected:
            continue
        score = float(interval_mass[knot_idx - 1] + interval_mass[knot_idx])
        donor_scores.append((score, int(knot_idx)))
    if not donor_scores:
        return old_knots.copy()
    _score, donor_idx = min(donor_scores, key=lambda item: (item[0], item[1]))
    candidate = np.delete(old_knots, donor_idx)
    candidate = np.sort(np.concatenate([candidate, np.asarray([new_knot], dtype=np.float64)]))
    return _strictly_increasing_knots(candidate, max_lag=float(max_lag), num_knots=num_knots)


def _cdf_from_density(density: np.ndarray, grid: np.ndarray) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    cdf = np.zeros_like(density, dtype=np.float64)
    if density.size <= 1:
        return cdf
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf[1:] = np.cumsum(increments)
    if cdf[-1] > 1e-12:
        cdf = cdf / cdf[-1]
    return np.clip(cdf, 0.0, 1.0)


def _kernel_distribution_metrics(est: np.ndarray, truth: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    est_density, est_area = _density_on_grid(est, grid)
    truth_density, truth_area = _density_on_grid(truth, grid)
    if est_area <= 1e-12 and truth_area <= 1e-12:
        return {
            "area_est": float(est_area),
            "area_truth": float(truth_area),
            "total_variation": 0.0,
            "wasserstein_1": 0.0,
            "ks": 0.0,
            "energy": 0.0,
            "hellinger": 0.0,
            "js_divergence": 0.0,
        }
    if est_area <= 1e-12 or truth_area <= 1e-12:
        return {
            "area_est": float(est_area),
            "area_truth": float(truth_area),
            "total_variation": 1.0,
            "wasserstein_1": float(grid[-1] - grid[0]),
            "ks": 1.0,
            "energy": float(math.sqrt(2.0 * max(float(grid[-1] - grid[0]), 0.0))),
            "hellinger": 1.0,
            "js_divergence": float(math.log(2.0)),
        }
    cdf_est = _cdf_from_density(est_density, grid)
    cdf_truth = _cdf_from_density(truth_density, grid)
    cdf_diff = cdf_est - cdf_truth
    tv = 0.5 * _safe_trapz(np.abs(est_density - truth_density), grid)
    w1 = _safe_trapz(np.abs(cdf_diff), grid)
    ks = float(np.max(np.abs(cdf_diff))) if cdf_diff.size else 0.0
    energy = math.sqrt(max(2.0 * _safe_trapz(cdf_diff * cdf_diff, grid), 0.0))
    hellinger = math.sqrt(max(0.5 * _safe_trapz((np.sqrt(est_density) - np.sqrt(truth_density)) ** 2, grid), 0.0))
    eps = 1e-12
    mix = 0.5 * (est_density + truth_density)
    js = 0.5 * _safe_trapz(est_density * np.log((est_density + eps) / (mix + eps)), grid)
    js += 0.5 * _safe_trapz(truth_density * np.log((truth_density + eps) / (mix + eps)), grid)
    return {
        "area_est": float(est_area),
        "area_truth": float(truth_area),
        "total_variation": float(tv),
        "wasserstein_1": float(w1),
        "ks": float(ks),
        "energy": float(energy),
        "hellinger": float(hellinger),
        "js_divergence": float(max(js, 0.0)),
    }


def _peak_summary(values: np.ndarray, grid: np.ndarray, *, modal_grid: np.ndarray | None = None) -> dict[str, Any]:
    _ = modal_grid
    vals = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    grid = np.asarray(grid, dtype=np.float64)
    peak = float(np.max(vals)) if vals.size else 0.0
    if vals.size == 0 or peak <= 1e-12:
        return {"valid": False, "peak_lag": None, "peak_interval": None}
    peak_lag = float(grid[int(np.argmax(vals))])
    # FWHM modal core: a source pair is "equal" when their half-maximum
    # temporal cores overlap. This is a shape-based relation definition, not a
    # data-fitted margin.
    mask = vals >= 0.5 * peak
    core_grid = grid[mask]
    interval = [float(np.min(core_grid)), float(np.max(core_grid))]
    return {
        "valid": True,
        "peak_lag": peak_lag,
        "peak_interval": [float(interval[0]), float(interval[1])],
        "peak_interval_type": "fwhm_modal_core",
    }


def _relation_from_peaks(a: dict[str, Any], b: dict[str, Any]) -> str:
    if not bool(a.get("valid")) or not bool(b.get("valid")):
        return "none"
    a0, a1 = [float(x) for x in a["peak_interval"]]
    b0, b1 = [float(x) for x in b["peak_interval"]]
    # Larger source-to-target lag means the source happened earlier.
    if a0 > b1:
        return "before"
    if a1 < b0:
        return "after"
    return "equal"


def _temporal_relations_for_rule(
    *,
    rule_text: str,
    source_summaries: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    sources = sorted(int(src) for src in source_summaries)
    rows: list[dict[str, Any]] = []
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1:]:
            est_rel = _relation_from_peaks(source_summaries[src_a]["estimated"], source_summaries[src_b]["estimated"])
            truth_rel = _relation_from_peaks(source_summaries[src_a]["truth"], source_summaries[src_b]["truth"])
            rows.append(
                {
                    "rule": rule_text,
                    "source_a": int(src_a),
                    "source_b": int(src_b),
                    "estimated_relation": est_rel,
                    "truth_relation": truth_rel,
                    "match": bool(est_rel == truth_rel),
                    "estimated_peak_a": source_summaries[src_a]["estimated"].get("peak_lag"),
                    "estimated_peak_b": source_summaries[src_b]["estimated"].get("peak_lag"),
                    "truth_peak_a": source_summaries[src_a]["truth"].get("peak_lag"),
                    "truth_peak_b": source_summaries[src_b]["truth"].get("peak_lag"),
                    "estimated_interval_a": source_summaries[src_a]["estimated"].get("peak_interval"),
                    "estimated_interval_b": source_summaries[src_b]["estimated"].get("peak_interval"),
                    "truth_interval_a": source_summaries[src_a]["truth"].get("peak_interval"),
                    "truth_interval_b": source_summaries[src_b]["truth"].get("peak_interval"),
                }
            )
    return rows


def native_kernel_recovery(
    *,
    prepared: dict[str, Any],
    matched: list[tuple[tuple[int, ...], str, int]],
    rule_heights: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any]:
    import run_paper_benchmarks as rpb
    from conjunctive_rule_initializer import normalize_piecewise_score

    truth_lookup = rpb._truth_rule_lookup(prepared["config"])
    subset_to_idx = {tuple(int(x) for x in subset): int(idx) for idx, subset in enumerate(prepared["subsets"])}
    source_knots = {int(k): np.asarray(v, dtype=np.float64) for k, v in prepared.get("source_knots", {}).items()}
    rule_source_knots = {
        (int(idx), int(src)): np.asarray(v, dtype=np.float64)
        for (idx, src), v in prepared.get("rule_source_knots", {}).items()
    }
    default_knots = np.asarray(prepared["knots"], dtype=np.float64)
    details: list[dict[str, Any]] = []
    l1_vals: list[float] = []
    l2_vals: list[float] = []
    for rule in sorted(matched):
        subset, _sign, target = rule
        idx = subset_to_idx[tuple(int(x) for x in subset)]
        truth_rule = truth_lookup[rule]
        kernel_type = str(truth_rule.get("kernel", prepared["config"].get("kernel", "triangular")))
        for src in subset:
            src = int(src)
            knots = rule_source_knots.get((idx, src), source_knots.get(src, default_knots))
            est = normalize_piecewise_score(np.asarray(rule_heights[(idx, src)], dtype=np.float64))
            source_condition = truth_rule["condition"].get(src) or truth_rule["condition"].get(str(src))
            truth = rpb._truth_kernel_heights(
                kernel_type=kernel_type,
                source_condition=source_condition,
                knots=knots,
            )
            l1 = _safe_trapz(np.abs(est - truth), knots)
            l2 = math.sqrt(max(_safe_trapz((est - truth) ** 2, knots), 0.0))
            l1_vals.append(l1)
            l2_vals.append(l2)
            details.append(
                {
                    "rule": rpb.format_rule(rule, int(target)),
                    "source": int(src),
                    "kernel_type": kernel_type,
                    "l1": float(l1),
                    "l2": float(l2),
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


def dense_kernel_recovery(
    *,
    prepared: dict[str, Any],
    matched: list[tuple[tuple[int, ...], str, int]],
    rule_heights: dict[tuple[int, int], np.ndarray],
    dense_grid_size: int = 121,
) -> dict[str, Any]:
    import run_paper_benchmarks as rpb
    from conjunctive_rule_initializer import normalize_piecewise_score

    default_knots = np.asarray(prepared["knots"], dtype=np.float64)
    source_knots = {int(k): np.asarray(v, dtype=np.float64) for k, v in prepared.get("source_knots", {}).items()}
    rule_source_knots = {
        (int(idx), int(src)): np.asarray(v, dtype=np.float64)
        for (idx, src), v in prepared.get("rule_source_knots", {}).items()
    }
    truth_lookup = rpb._truth_rule_lookup(prepared["config"])
    subset_to_idx = {tuple(int(x) for x in subset): int(idx) for idx, subset in enumerate(prepared["subsets"])}
    details: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    l1_vals: list[float] = []
    l2_vals: list[float] = []
    dist_keys = ("total_variation", "wasserstein_1", "ks", "energy", "hellinger", "js_divergence")
    dist_vals: dict[str, list[float]] = {key: [] for key in dist_keys}
    for rule in sorted(matched):
        subset, _sign, target = rule
        idx = subset_to_idx[tuple(int(x) for x in subset)]
        truth_rule = truth_lookup[rule]
        kernel_type = str(truth_rule.get("kernel", prepared["config"].get("kernel", "triangular")))
        rule_text = rpb.format_rule(rule, int(target))
        source_summaries: dict[int, dict[str, dict[str, Any]]] = {}
        for src in subset:
            src = int(src)
            knots = rule_source_knots.get((idx, src), source_knots.get(src, default_knots))
            dense_grid = np.linspace(float(knots[0]), float(knots[-1]), int(dense_grid_size), dtype=np.float64)
            est_native = normalize_piecewise_score(np.asarray(rule_heights[(idx, src)], dtype=np.float64))
            est_dense = np.interp(dense_grid, knots, est_native, left=est_native[0], right=est_native[-1])
            est_dense = normalize_piecewise_score(np.maximum(est_dense, 0.0))
            source_condition = truth_rule["condition"].get(src) or truth_rule["condition"].get(str(src))
            truth_dense = rpb._truth_kernel_heights(
                kernel_type=kernel_type,
                source_condition=source_condition,
                knots=dense_grid,
            )
            l1 = _safe_trapz(np.abs(est_dense - truth_dense), dense_grid)
            l2 = math.sqrt(max(_safe_trapz((est_dense - truth_dense) ** 2, dense_grid), 0.0))
            dist = _kernel_distribution_metrics(est_dense, truth_dense, dense_grid)
            l1_vals.append(l1)
            l2_vals.append(l2)
            for key in dist_keys:
                dist_vals[key].append(float(dist[key]))
            est_peak = _peak_summary(est_dense, dense_grid, modal_grid=knots)
            truth_peak = _peak_summary(truth_dense, dense_grid, modal_grid=knots)
            source_summaries[int(src)] = {"estimated": est_peak, "truth": truth_peak}
            details.append(
                {
                    "rule": rule_text,
                    "source": int(src),
                    "kernel_type": kernel_type,
                    "l1": l1,
                    "l2": l2,
                    "ise": float(l2 * l2),
                    **dist,
                    "estimated_peak_lag": est_peak.get("peak_lag"),
                    "truth_peak_lag": truth_peak.get("peak_lag"),
                    "estimated_peak_interval": est_peak.get("peak_interval"),
                    "truth_peak_interval": truth_peak.get("peak_interval"),
                    "knots": [float(x) for x in knots],
                    "dense_grid": [float(x) for x in dense_grid],
                    "estimated_native": [float(x) for x in est_native],
                    "estimated_dense": [float(x) for x in est_dense],
                    "truth_dense": [float(x) for x in truth_dense],
                }
            )
        relation_rows.extend(_temporal_relations_for_rule(rule_text=rule_text, source_summaries=source_summaries))
    relation_matches = [row["match"] for row in relation_rows if row["truth_relation"] != "none"]
    return {
        "num_rule_source_pairs": int(len(details)),
        "mean_l1": float(np.mean(l1_vals)) if l1_vals else 0.0,
        "mean_l2": float(np.mean(l2_vals)) if l2_vals else 0.0,
        "max_l1": float(np.max(l1_vals)) if l1_vals else 0.0,
        "max_l2": float(np.max(l2_vals)) if l2_vals else 0.0,
        "mean_ise": float(np.mean([x * x for x in l2_vals])) if l2_vals else 0.0,
        **{
            f"mean_{key}": float(np.mean(values)) if values else 0.0
            for key, values in dist_vals.items()
        },
        **{
            f"max_{key}": float(np.max(values)) if values else 0.0
            for key, values in dist_vals.items()
        },
        "temporal_relation_accuracy": float(np.mean(relation_matches)) if relation_matches else 1.0,
        "num_temporal_relation_pairs": int(len(relation_rows)),
        "temporal_relations": relation_rows,
        "details": details,
    }


def fit_case(name: str, seed: int, gpu_id: int, result_dir: Path) -> dict[str, Any]:
    import run_paper_benchmarks as rpb
    import rule_dependent_kernel_active_set as rd

    base_path = BASE_RESULT_DIR / f"{name}_seed{int(seed)}.json"
    base = json.loads(base_path.read_text())
    if float(base.get("recall", 0.0)) != 1.0 or float(base.get("precision", 0.0)) != 1.0:
        raise RuntimeError(f"base support is not exact for {name} seed={seed}")

    target = int(base["target"])
    n_eff = int(base.get("final_selection_sequences", 0) or 0)
    if n_eff <= 0:
        raise RuntimeError(f"base result has invalid final_selection_sequences: {n_eff}")
    m_hat = m_hat_from_n_eff(n_eff)
    prepared = prepare_state_with_num_knots(
        Path(base["config_path"]),
        f"cuda:{int(gpu_id)}",
        dataset_seed=int(seed),
        num_knots=int(m_hat),
    )
    subset_to_idx = {tuple(int(x) for x in subset): int(idx) for idx, subset in enumerate(prepared["subsets"])}
    matched_rules = [parse_formatted_rule(text, target) for text in base["matched"]]
    support_rules = [parse_formatted_rule(text, target) for text in base["predicted"]]
    active_rules = [
        rd.ActiveRule(idx=int(subset_to_idx[tuple(rule[0])]), sign=str(rule[1]))
        for rule in support_rules
    ]
    prepared = apply_conditional_rule_source_grid(
        prepared,
        active_rules=active_rules,
        num_knots=int(m_hat),
    )

    started = time.perf_counter()
    result = rd.evaluate_support_exact(
        active_rules=active_rules,
        subsets=prepared["subsets"],
        basis_cache=prepared["basis_cache"],
        base_rule_heights=prepared["rule_heights"],
        template_rule_heights=prepared["template_rule_heights"],
        grid_weights_train=prepared["grid_weights_train"],
        grid_weights_val=prepared["grid_weights_val"],
        device=prepared["device"],
        torch_basis_cache=prepared["torch_basis_cache"],
        opt_steps=int(OPT_STEPS),
        lr=0.05,
        penalize_kernel_df=True,
        penalty_scale=1.0,
        num_val_sequences=int(n_eff),
        kernel_smoothness_ridge=float(FINAL_KERNEL_MLE_SMOOTHNESS_RIDGE),
        kernel_normalization=str(FINAL_KERNEL_NORMALIZATION),
        support_cache=None,
        warm_start_result=None,
    )
    best_prepared = prepared
    best_result = result
    selected_post_selection_stage = "conditional_lag_quantile"
    post_selection_candidates = [
        {
            "stage": selected_post_selection_stage,
            "bic": float(result.bic),
        }
    ]
    working_prepared = prepared
    working_result = result
    for frontier_pass in range(int(MAX_WITNESS_FRONTIER_PASSES)):
        frontier_prepared, warm_result = apply_max_witness_frontier_grid(
            working_prepared,
            active_rules=active_rules,
            fitted_result=working_result,
            num_knots=int(m_hat),
            n_eff=int(n_eff),
        )
        frontier_result = rd.evaluate_support_exact(
            active_rules=active_rules,
            subsets=frontier_prepared["subsets"],
            basis_cache=frontier_prepared["basis_cache"],
            base_rule_heights=frontier_prepared["rule_heights"],
            template_rule_heights=frontier_prepared["template_rule_heights"],
            grid_weights_train=frontier_prepared["grid_weights_train"],
            grid_weights_val=frontier_prepared["grid_weights_val"],
            device=frontier_prepared["device"],
            torch_basis_cache=frontier_prepared["torch_basis_cache"],
            opt_steps=int(OPT_STEPS),
            lr=0.05,
            penalize_kernel_df=True,
            penalty_scale=1.0,
            num_val_sequences=int(n_eff),
            kernel_smoothness_ridge=float(FINAL_KERNEL_MLE_SMOOTHNESS_RIDGE),
            kernel_normalization=str(FINAL_KERNEL_NORMALIZATION),
            support_cache=None,
            warm_start_result=warm_result,
        )
        stage_name = f"local_kkt_interval_refinement_pass_{int(frontier_pass) + 1}"
        post_selection_candidates.append(
            {
                "stage": stage_name,
                "bic": float(frontier_result.bic),
            }
        )
        working_prepared = frontier_prepared
        working_result = frontier_result
        if float(frontier_result.bic) < float(best_result.bic):
            best_prepared = frontier_prepared
            best_result = frontier_result
            selected_post_selection_stage = stage_name
    prepared = best_prepared
    result = best_result
    fit_sec = float(time.perf_counter() - started)
    kernel_recovery_native = native_kernel_recovery(
        prepared=prepared,
        matched=matched_rules,
        rule_heights=result.rule_heights,
    )
    kernel_recovery_dense = dense_kernel_recovery(
        prepared=prepared,
        matched=matched_rules,
        rule_heights=result.rule_heights,
    )
    g_blocks = rd.kernel_group_count(list(result.active_rules), prepared["subsets"])
    block_dim = rd.model_param_dim(list(result.active_rules), prepared["subsets"], int(m_hat))
    full_sieve_dim = float(1 + len(result.active_rules) + int(g_blocks) * (int(m_hat) - 1))
    log_n = math.log(float(rd.bic_sample_size(int(n_eff))))
    full_sieve_bic = float(result.bic + (full_sieve_dim - block_dim) * log_n)
    exc_params = {int(k): float(v) for k, v in result.exc_params.items()}
    inh_params = {int(k): float(v) for k, v in result.inh_params.items()}
    rule_parameter_rows = []
    for rule in result.active_rules:
        subset = tuple(int(x) for x in prepared["subsets"][int(rule.idx)])
        beta = exc_params.get(int(rule.idx), inh_params.get(int(rule.idx), 0.0))
        rule_parameter_rows.append(
            {
                "sources": list(subset),
                "sign": str(rule.sign),
                "target": int(target),
                "beta": float(beta),
            }
        )

    return {
        "status": "ok",
        "benchmark": str(name),
        "seed": int(seed),
        "target": int(target),
        "base_result_path": str(base_path),
        "config_path": str(base["config_path"]),
        "data_path": str(prepared["data_path"]),
        "support_fixed": [str(x) for x in base["predicted"]],
        "support_metrics": {
            "recall": float(base["recall"]),
            "precision": float(base["precision"]),
            "missing": list(base.get("missing", [])),
            "extra": list(base.get("extra", [])),
        },
        "assumptions": {
            "effective_sample_size": "independent_trajectories",
            "n_eff": int(n_eff),
            "kernel_smoothness": "one_dimensional_lipschitz_s1",
            "kernel_constraints": "nonnegative_area_normalized_internal_peak_normalized_for_reporting",
            "initial_knot_strategy": "rule_source_conditional_empirical_lag_quantile",
            "final_knot_strategy": "local_kkt_interval_refinement",
            "max_witness_frontier_passes": int(MAX_WITNESS_FRONTIER_PASSES),
            "max_witness_frontier": dict(prepared.get("max_witness_frontier", {})),
            "post_selection_candidate_rule": "select lowest exact fixed-support BIC",
            "selected_post_selection_stage": str(selected_post_selection_stage),
            "post_selection_candidates": list(post_selection_candidates),
            "kernel_normalization": str(FINAL_KERNEL_NORMALIZATION),
            "kernel_smoothness_ridge": float(FINAL_KERNEL_MLE_SMOOTHNESS_RIDGE),
            "support_mutation": "disabled_fixed_support",
            "m_hat_formula": "odd_floor(n_eff^(1/3)) with minimum 3",
        },
        "m_hat": int(m_hat),
        "opt_steps": int(OPT_STEPS),
        "mu": float(result.mu),
        "exc_params": exc_params,
        "inh_params": inh_params,
        "rule_parameters": rule_parameter_rows,
        "bic_block_mdl_scale": float(result.bic),
        "full_sieve_dim": float(full_sieve_dim),
        "block_dim": float(block_dim),
        "kernel_block_count": int(g_blocks),
        "full_sieve_bic": float(full_sieve_bic),
        "knots": [float(x) for x in prepared["knots"]],
        "source_knots": {
            str(int(src)): [float(x) for x in np.asarray(knots, dtype=np.float64)]
            for src, knots in prepared["source_knots"].items()
        },
        "rule_source_knots": {
            f"{int(idx)}:{int(src)}": [float(x) for x in np.asarray(knots, dtype=np.float64)]
            for (idx, src), knots in prepared.get("rule_source_knots", {}).items()
        },
        "kernel_recovery_native": kernel_recovery_native,
        "kernel_recovery_dense": kernel_recovery_dense,
        "runtime_sec": float(fit_sec),
        "worker_gpu": int(gpu_id),
    }


def _task_estimate(task: dict[str, Any]) -> int:
    return int(EST_SEC.get(str(task["name"]), 40))


def worker_main(gpu_id: int, tasks: mp.Queue, done: mp.Queue, result_dir: str) -> None:
    _install_paths()
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
    from runtime_resources import configure_runtime_resources

    configure_runtime_resources(CPU_THREADS_PER_WORKER)
    result_dir_path = Path(result_dir)
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        name = str(task["name"])
        seed = int(task["seed"])
        out_path = result_dir_path / f"{name}_seed{seed}.json"
        start = time.perf_counter()
        print(f"[gpu{gpu_id}] START {name} seed={seed} est={_task_estimate(task)}s", flush=True)
        try:
            payload = fit_case(name, seed, int(gpu_id), result_dir_path)
            payload["wall_sec"] = float(time.perf_counter() - start)
            write_json(out_path, payload)
            summary = {
                "status": "ok",
                "benchmark": name,
                "seed": seed,
                "gpu": int(gpu_id),
                "m_hat": payload["m_hat"],
                "mean_l1_dense": payload["kernel_recovery_dense"]["mean_l1"],
                "max_l1_dense": payload["kernel_recovery_dense"]["max_l1"],
                "runtime_sec": payload["runtime_sec"],
                "wall_sec": payload["wall_sec"],
            }
            print(f"[gpu{gpu_id}] DONE {json.dumps(summary, ensure_ascii=False)}", flush=True)
            done.put(summary)
        except BaseException as exc:
            payload = {
                "status": "error",
                "benchmark": name,
                "seed": seed,
                "gpu": int(gpu_id),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "wall_sec": float(time.perf_counter() - start),
            }
            write_json(out_path, payload)
            print(f"[gpu{gpu_id}] ERROR {json.dumps(payload, ensure_ascii=False)}", flush=True)
            done.put(payload)


def aggregate(result_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    temporal_relation_rows: list[dict[str, Any]] = []
    for name in BENCHMARKS:
        for seed in SEEDS:
            path = result_dir / f"{name}_seed{int(seed)}.json"
            if not path.exists():
                rows.append({"status": "missing_file", "benchmark": name, "seed": int(seed)})
                continue
            data = json.loads(path.read_text())
            if data.get("status") != "ok":
                rows.append({"status": data.get("status", "error"), "benchmark": name, "seed": int(seed)})
                continue
            dense = data["kernel_recovery_dense"]
            native = data["kernel_recovery_native"]
            rows.append(
                {
                    "status": "ok",
                    "benchmark": name,
                    "seed": int(seed),
                    "m_hat": int(data["m_hat"]),
                    "support_recall": float(data["support_metrics"]["recall"]),
                    "support_precision": float(data["support_metrics"]["precision"]),
                    "mean_l1_dense": float(dense["mean_l1"]),
                    "mean_l2_dense": float(dense["mean_l2"]),
                    "mean_ise_dense": float(dense.get("mean_ise", 0.0)),
                    "mean_total_variation_dense": float(dense.get("mean_total_variation", 0.0)),
                    "mean_wasserstein_1_dense": float(dense.get("mean_wasserstein_1", 0.0)),
                    "mean_ks_dense": float(dense.get("mean_ks", 0.0)),
                    "mean_energy_dense": float(dense.get("mean_energy", 0.0)),
                    "mean_hellinger_dense": float(dense.get("mean_hellinger", 0.0)),
                    "mean_js_divergence_dense": float(dense.get("mean_js_divergence", 0.0)),
                    "max_l1_dense": float(dense["max_l1"]),
                    "max_l2_dense": float(dense["max_l2"]),
                    "max_total_variation_dense": float(dense.get("max_total_variation", 0.0)),
                    "max_wasserstein_1_dense": float(dense.get("max_wasserstein_1", 0.0)),
                    "max_ks_dense": float(dense.get("max_ks", 0.0)),
                    "max_energy_dense": float(dense.get("max_energy", 0.0)),
                    "max_hellinger_dense": float(dense.get("max_hellinger", 0.0)),
                    "max_js_divergence_dense": float(dense.get("max_js_divergence", 0.0)),
                    "temporal_relation_accuracy": float(dense.get("temporal_relation_accuracy", 1.0)),
                    "num_temporal_relation_pairs": int(dense.get("num_temporal_relation_pairs", 0)),
                    "mean_l1_native": float(native["mean_l1"]),
                    "mean_l2_native": float(native["mean_l2"]),
                    "max_l1_native": float(native["max_l1"]),
                    "max_l2_native": float(native["max_l2"]),
                    "mu": float(data["mu"]),
                    "full_sieve_bic": float(data["full_sieve_bic"]),
                    "kernel_block_count": int(data["kernel_block_count"]),
                    "runtime_sec": float(data["runtime_sec"]),
                    "wall_sec": float(data.get("wall_sec", data["runtime_sec"])),
                    "worker_gpu": int(data["worker_gpu"]),
                }
            )
            for detail in dense["details"]:
                kernel_rows.append(
                    {
                        "benchmark": name,
                        "seed": int(seed),
                        "rule": detail["rule"],
                        "source": int(detail["source"]),
                        "kernel_type": detail["kernel_type"],
                        "l1_dense": float(detail["l1"]),
                        "l2_dense": float(detail["l2"]),
                        "ise_dense": float(detail.get("ise", float(detail["l2"]) ** 2)),
                        "total_variation": float(detail.get("total_variation", 0.0)),
                        "wasserstein_1": float(detail.get("wasserstein_1", 0.0)),
                        "ks": float(detail.get("ks", 0.0)),
                        "energy": float(detail.get("energy", 0.0)),
                        "hellinger": float(detail.get("hellinger", 0.0)),
                        "js_divergence": float(detail.get("js_divergence", 0.0)),
                        "area_est": float(detail.get("area_est", 0.0)),
                        "area_truth": float(detail.get("area_truth", 0.0)),
                        "estimated_peak_lag": detail.get("estimated_peak_lag"),
                        "truth_peak_lag": detail.get("truth_peak_lag"),
                        "m_hat": int(data["m_hat"]),
                    }
                )
            for relation in dense.get("temporal_relations", []):
                temporal_relation_rows.append(
                    {
                        "benchmark": name,
                        "seed": int(seed),
                        "rule": relation["rule"],
                        "source_a": int(relation["source_a"]),
                        "source_b": int(relation["source_b"]),
                        "estimated_relation": str(relation["estimated_relation"]),
                        "truth_relation": str(relation["truth_relation"]),
                        "match": bool(relation["match"]),
                        "estimated_peak_a": relation.get("estimated_peak_a"),
                        "estimated_peak_b": relation.get("estimated_peak_b"),
                        "truth_peak_a": relation.get("truth_peak_a"),
                        "truth_peak_b": relation.get("truth_peak_b"),
                        "estimated_interval_a0": relation.get("estimated_interval_a", [None, None])[0],
                        "estimated_interval_a1": relation.get("estimated_interval_a", [None, None])[1],
                        "estimated_interval_b0": relation.get("estimated_interval_b", [None, None])[0],
                        "estimated_interval_b1": relation.get("estimated_interval_b", [None, None])[1],
                        "truth_interval_a0": relation.get("truth_interval_a", [None, None])[0],
                        "truth_interval_a1": relation.get("truth_interval_a", [None, None])[1],
                        "truth_interval_b0": relation.get("truth_interval_b", [None, None])[0],
                        "truth_interval_b1": relation.get("truth_interval_b", [None, None])[1],
                    }
                )
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "result_dir": str(result_dir),
        "base_result_dir": str(BASE_RESULT_DIR),
        "complete_ok": bool(len(ok_rows) == len(BENCHMARKS) * len(SEEDS)),
        "num_rows": int(len(rows)),
        "num_ok": int(len(ok_rows)),
        "mean_l1_dense": float(np.mean([row["mean_l1_dense"] for row in ok_rows])) if ok_rows else None,
        "mean_l2_dense": float(np.mean([row["mean_l2_dense"] for row in ok_rows])) if ok_rows else None,
        "mean_total_variation_dense": float(np.mean([row["mean_total_variation_dense"] for row in ok_rows])) if ok_rows else None,
        "mean_wasserstein_1_dense": float(np.mean([row["mean_wasserstein_1_dense"] for row in ok_rows])) if ok_rows else None,
        "mean_ks_dense": float(np.mean([row["mean_ks_dense"] for row in ok_rows])) if ok_rows else None,
        "mean_energy_dense": float(np.mean([row["mean_energy_dense"] for row in ok_rows])) if ok_rows else None,
        "mean_hellinger_dense": float(np.mean([row["mean_hellinger_dense"] for row in ok_rows])) if ok_rows else None,
        "mean_js_divergence_dense": float(np.mean([row["mean_js_divergence_dense"] for row in ok_rows])) if ok_rows else None,
        "mean_temporal_relation_accuracy": float(np.mean([row["temporal_relation_accuracy"] for row in ok_rows])) if ok_rows else None,
        "max_l1_dense": float(np.max([row["max_l1_dense"] for row in ok_rows])) if ok_rows else None,
        "max_l2_dense": float(np.max([row["max_l2_dense"] for row in ok_rows])) if ok_rows else None,
        "runtime_sum_sec": float(np.sum([row["runtime_sec"] for row in ok_rows])) if ok_rows else None,
        "mean_runtime_sec": float(np.mean([row["runtime_sec"] for row in ok_rows])) if ok_rows else None,
        "rows": rows,
    }
    write_json(result_dir / "summary.json", summary)
    write_csv(rows, result_dir / "summary_rows.csv")
    write_csv(kernel_rows, result_dir / "kernel_detail_rows.csv")
    write_csv(temporal_relation_rows, result_dir / "temporal_relation_rows.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-support M_hat kernel-shape MLE for the 33 synthetic paper cases.")
    parser.add_argument("--result_dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--only", default="")
    parser.add_argument("--seeds", default="111,222,333")
    args = parser.parse_args()

    _install_paths()
    mp.set_start_method("spawn", force=True)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x.strip()) for x in str(args.seeds).split(",") if x.strip())
    only = {x.strip() for x in str(args.only).split(",") if x.strip()}
    names = tuple(name for name in BENCHMARKS if not only or name in only)
    tasks_list = [{"name": name, "seed": int(seed)} for name in names for seed in seeds]
    tasks_list = sorted(tasks_list, key=lambda task: -_task_estimate(task))
    tasks: mp.Queue = mp.Queue()
    done: mp.Queue = mp.Queue()
    for task in tasks_list:
        tasks.put(task)

    gpus = tuple(int(x.strip()) for x in str(args.gpus).split(",") if x.strip())
    start = time.perf_counter()
    workers = [
        mp.Process(target=worker_main, args=(gpu, tasks, done, str(result_dir)), daemon=False)
        for gpu in gpus
    ]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join()
    summary = aggregate(result_dir)
    summary["total_wall_sec"] = float(time.perf_counter() - start)
    write_json(result_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

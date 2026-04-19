"""End-to-end sparse rule-dependent kernel learner.

This experimental path keeps rule-specific kernels from the start:

    g_{U,s}(tau) = sum_m h_{U,s,m} phi_m(tau),  h_{U,s,m} >= 0,  int g = 1

Rules are added by one-sided point-process score tests and active rules are
jointly optimized with their kernel heights and rule weights. The raw
convolution basis responses are computed once and reused throughout.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from conjunctive_rule_initializer import (
    SourceBasisCache,
    auto_grid_step,
    base_rate_fit,
    bounded_source_activity,
    build_midpoint_grid,
    build_seq_event_arrays,
    collect_target_events,
    collect_weighted_lag_hist,
    estimate_source_kernels,
    format_rule,
    gt_rules_from_config,
    init_piecewise_heights,
    load_dataset,
    load_yaml,
    normalize_piecewise_area,
    normalized_kernel_response,
    print_rule_block,
    subset_list,
    summarize_results,
    trapz_area_weights,
)
from runtime_resources import configure_deterministic_research, configure_runtime_resources


@dataclass(frozen=True)
class ActiveRule:
    idx: int
    sign: str


class TorchBasisCache:
    def __init__(self, basis_cache: SourceBasisCache, device: torch.device):
        self._basis_cache = basis_cache
        self._device = device
        self._cache: dict[tuple[int, str], torch.Tensor] = {}

    def arrays(self, src: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src = int(src)
        keys = [
            (src, "tr_ev"),
            (src, "tr_gr"),
            (src, "va_ev"),
            (src, "va_gr"),
        ]
        missing = [key for key in keys if key not in self._cache]
        if missing:
            b_tr_ev, b_tr_gr, b_va_ev, b_va_gr = self._basis_cache.arrays(src)
            self._cache[(src, "tr_ev")] = torch.tensor(b_tr_ev, dtype=torch.float32, device=self._device)
            self._cache[(src, "tr_gr")] = torch.tensor(b_tr_gr, dtype=torch.float32, device=self._device)
            self._cache[(src, "va_ev")] = torch.tensor(b_va_ev, dtype=torch.float32, device=self._device)
            self._cache[(src, "va_gr")] = torch.tensor(b_va_gr, dtype=torch.float32, device=self._device)
        return (
            self._cache[(src, "tr_ev")],
            self._cache[(src, "tr_gr")],
            self._cache[(src, "va_ev")],
            self._cache[(src, "va_gr")],
        )


def rule_param_dim(subset: tuple[int, ...], num_knots: int) -> int:
    return int(1 + len(tuple(subset)) * max(int(num_knots) - 1, 0))


def model_param_dim(active_rules: list[ActiveRule], subsets, num_knots: int) -> int:
    return int(1 + sum(rule_param_dim(subsets[int(ar.idx)], int(num_knots)) for ar in active_rules))


def bic_sample_size(num_sequences: int) -> int:
    # For the benchmark suite we observe independent trajectories and
    # approximate the continuous-time likelihood on each of them. The grid
    # points are numerical quadrature locations, not additional independent
    # observations, so BIC should scale with the number of sequences rather
    # than the number of quadrature points.
    return max(int(num_sequences), 2)


def inverse_softplus(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), 1e-8)
    out = np.empty_like(x, dtype=np.float64)
    mask = x > 20.0
    out[mask] = x[mask]
    out[~mask] = np.log(np.expm1(x[~mask]))
    return out


def build_global_activity(
    *,
    train_arrays,
    val_arrays,
    kernels,
    source_ids,
    tr_event_seq,
    tr_event_times,
    tr_grid_seq,
    tr_grid_times,
    va_event_seq,
    va_event_times,
    va_grid_seq,
    va_grid_times,
):
    from conjunctive_rule_initializer import compute_source_signal_matrix

    z_train_event = compute_source_signal_matrix(
        train_arrays, kernels, source_ids=source_ids, seq_ids=tr_event_seq, times=tr_event_times
    )
    z_val_event = compute_source_signal_matrix(
        val_arrays, kernels, source_ids=source_ids, seq_ids=va_event_seq, times=va_event_times
    )
    z_train_grid = compute_source_signal_matrix(
        train_arrays, kernels, source_ids=source_ids, seq_ids=tr_grid_seq, times=tr_grid_times
    )
    z_val_grid = compute_source_signal_matrix(
        val_arrays, kernels, source_ids=source_ids, seq_ids=va_grid_seq, times=va_grid_times
    )
    return (
        bounded_source_activity(z_train_event),
        bounded_source_activity(z_train_grid),
        bounded_source_activity(z_val_event),
        bounded_source_activity(z_val_grid),
    )


def initialize_rule_specific_heights(
    *,
    subsets,
    source_ids,
    global_kernels,
    global_activity_event,
    src_to_col_global,
    train_arrays,
    train_event_lag_bin_cache,
    max_lag,
    num_bins,
    time_horizon,
):
    total_time = float(len(train_arrays)) * float(time_horizon)
    source_counts = {int(s): 0 for s in source_ids}
    for by_type in train_arrays:
        for src in source_ids:
            source_counts[int(src)] += int(by_type.get(int(src), np.zeros((0,), dtype=np.float64)).size)

    edges = np.linspace(0.0, float(max_lag), int(num_bins) + 1)
    bin_width = float(edges[1] - edges[0])
    out: dict[tuple[int, int], np.ndarray] = {}
    for idx, subset in enumerate(subsets):
        subset = tuple(int(s) for s in subset)
        for src in subset:
            other = [int(u) for u in subset if int(u) != int(src)]
            if other:
                weights = np.ones((global_activity_event.shape[0],), dtype=np.float64)
                for u in other:
                    weights *= np.clip(global_activity_event[:, src_to_col_global[int(u)]], 0.0, 1.0)
            else:
                weights = np.ones((global_activity_event.shape[0],), dtype=np.float64)
            effective_weight = float(np.sum(weights))
            if effective_weight <= 1e-8:
                out[(int(idx), int(src))] = np.asarray(global_kernels[int(src)].heights, dtype=np.float64).copy()
                continue
            hist = collect_weighted_lag_hist(
                source=int(src),
                event_weights=weights,
                event_lag_bin_cache=train_event_lag_bin_cache,
            )
            src_rate = float(source_counts[int(src)]) / max(total_time, 1e-8)
            expected = np.full_like(hist, src_rate * effective_weight * bin_width)
            heights = init_piecewise_heights(
                hist=hist,
                edges=edges,
                expected=expected,
                knots=np.asarray(global_kernels[int(src)].knots, dtype=np.float64),
            )
            out[(int(idx), int(src))] = heights
    return out


def compute_rule_feature_arrays(
    *,
    subsets,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    kernels,
):
    area_weights = trapz_area_weights(next(iter(kernels.values())).knots)
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for idx, subset in enumerate(subsets):
        subset = tuple(int(s) for s in subset)
        tr_parts = []
        tr_grid_parts = []
        va_parts = []
        va_grid_parts = []
        for src in subset:
            b_tr_ev, b_tr_gr, b_va_ev, b_va_gr = basis_cache.arrays(int(src))
            z_tr_ev, h = normalized_kernel_response(b_tr_ev, rule_heights[(int(idx), int(src))], area_weights)
            z_tr_gr, _ = normalized_kernel_response(b_tr_gr, rule_heights[(int(idx), int(src))], area_weights)
            z_va_ev, _ = normalized_kernel_response(b_va_ev, rule_heights[(int(idx), int(src))], area_weights)
            z_va_gr, _ = normalized_kernel_response(b_va_gr, rule_heights[(int(idx), int(src))], area_weights)
            rule_heights[(int(idx), int(src))] = h
            tr_parts.append(bounded_source_activity(z_tr_ev))
            tr_grid_parts.append(bounded_source_activity(z_tr_gr))
            va_parts.append(bounded_source_activity(z_va_ev))
            va_grid_parts.append(bounded_source_activity(z_va_gr))
        out[int(idx)] = (
            np.prod(np.column_stack(tr_parts), axis=1),
            np.prod(np.column_stack(tr_grid_parts), axis=1),
            np.prod(np.column_stack(va_parts), axis=1),
            np.prod(np.column_stack(va_grid_parts), axis=1),
        )
    return out


def compute_active_rule_feature_arrays(
    *,
    active_rules: list[ActiveRule],
    subsets,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    kernels,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not active_rules:
        return {}
    area_weights = trapz_area_weights(next(iter(kernels.values())).knots)
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    seen_idx: set[int] = set()
    for ar in active_rules:
        idx = int(ar.idx)
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        subset = tuple(int(s) for s in subsets[idx])
        tr_parts = []
        tr_grid_parts = []
        va_parts = []
        va_grid_parts = []
        for src in subset:
            b_tr_ev, b_tr_gr, b_va_ev, b_va_gr = basis_cache.arrays(int(src))
            z_tr_ev, h = normalized_kernel_response(b_tr_ev, rule_heights[(idx, int(src))], area_weights)
            z_tr_gr, _ = normalized_kernel_response(b_tr_gr, rule_heights[(idx, int(src))], area_weights)
            z_va_ev, _ = normalized_kernel_response(b_va_ev, rule_heights[(idx, int(src))], area_weights)
            z_va_gr, _ = normalized_kernel_response(b_va_gr, rule_heights[(idx, int(src))], area_weights)
            rule_heights[(idx, int(src))] = h
            tr_parts.append(bounded_source_activity(z_tr_ev))
            tr_grid_parts.append(bounded_source_activity(z_tr_gr))
            va_parts.append(bounded_source_activity(z_va_ev))
            va_grid_parts.append(bounded_source_activity(z_va_gr))
        out[idx] = (
            np.prod(np.column_stack(tr_parts), axis=1),
            np.prod(np.column_stack(tr_grid_parts), axis=1),
            np.prod(np.column_stack(va_parts), axis=1),
            np.prod(np.column_stack(va_grid_parts), axis=1),
        )
    return out

def rule_score(
    *,
    feat_event: np.ndarray,
    feat_grid: np.ndarray,
    mu: float,
    exc_event: np.ndarray,
    inh_event: np.ndarray,
    exc_grid: np.ndarray,
    inh_grid: np.ndarray,
    grid_weights_train: np.ndarray,
    penalty: float,
    intensity_model: str = "multiplicative",
) -> tuple[float, str, float]:
    if str(intensity_model) == "canonical_loglink":
        base_lp_grid = exc_grid - inh_grid
        base_mass = grid_weights_train * np.exp(np.clip(base_lp_grid, -40.0, 40.0))
        base_norm = float(np.sum(base_mass))
        if base_norm <= 1e-12:
            return -1e18, "exc", 1e-4

        s_event = float(np.sum(feat_event))
        n_events = int(feat_event.size)

        def _profile_gain(sign: str) -> tuple[float, float]:
            coef = profiled_coef_canonical(
                feat_event=feat_event,
                feat_grid=feat_grid,
                base_lp_grid=base_lp_grid,
                grid_weights_train=grid_weights_train,
                sign=str(sign),
            )
            signed = 1.0 if str(sign) == "exc" else -1.0
            expo = np.exp(np.clip(signed * float(coef) * feat_grid, -40.0, 40.0))
            new_norm = float(np.dot(base_mass, expo))
            if not np.isfinite(new_norm) or new_norm <= 1e-300:
                return -1e18, float(max(coef, 1e-4))
            ll_gain = signed * float(coef) * s_event - float(n_events) * math.log(new_norm / base_norm)
            return float(ll_gain - float(penalty)), float(max(coef, 1e-4))

        gain_exc, coef_exc = _profile_gain("exc")
        gain_inh, coef_inh = _profile_gain("inh")
    else:
        eta_event = np.clip(float(mu) + exc_event, 1e-8, None)
        eta_grid = np.clip(float(mu) + exc_grid, 1e-8, None)
        exp_neg_i_grid = np.exp(-inh_grid)
        lam0_grid = eta_grid * exp_neg_i_grid

        inv_eta_event = 1.0 / eta_event
        inv_eta_sq = inv_eta_event * inv_eta_event
        g_exc = float(np.dot(feat_event, inv_eta_event) - np.dot(feat_grid, grid_weights_train * exp_neg_i_grid))
        h_exc = float(np.dot(feat_event * inv_eta_sq, feat_event))
        gain_exc = -1e18
        coef_exc = 0.0
        if g_exc > 1e-10 and h_exc > 1e-10:
            coef_exc = float(g_exc / h_exc)
            gain_exc = 0.5 * float(g_exc * coef_exc) - float(penalty)

        g_inh = float(-np.sum(feat_event) + np.dot(feat_grid, grid_weights_train * lam0_grid))
        h_inh = float(np.dot(feat_grid * (grid_weights_train * lam0_grid), feat_grid))
        gain_inh = -1e18
        coef_inh = 0.0
        if g_inh > 1e-10 and h_inh > 1e-10:
            coef_inh = float(g_inh / h_inh)
            gain_inh = 0.5 * float(g_inh * coef_inh) - float(penalty)

    if gain_exc >= gain_inh:
        return gain_exc, "exc", max(coef_exc, 1e-4)
    return gain_inh, "inh", max(coef_inh, 1e-4)


def profiled_coef_canonical(
    *,
    feat_event: np.ndarray,
    feat_grid: np.ndarray,
    base_lp_grid: np.ndarray,
    grid_weights_train: np.ndarray,
    sign: str,
) -> float:
    feat_event = np.asarray(feat_event, dtype=np.float64)
    feat_grid = np.asarray(feat_grid, dtype=np.float64)
    base_lp_grid = np.asarray(base_lp_grid, dtype=np.float64)
    grid_weights_train = np.asarray(grid_weights_train, dtype=np.float64)
    if feat_event.size == 0 or feat_grid.size == 0:
        return 1e-4
    max_feat = float(np.max(feat_grid))
    if max_feat <= 1e-12:
        return 1e-4

    signed = 1.0 if str(sign) == "exc" else -1.0
    s_event = float(np.sum(feat_event))
    base_mass = grid_weights_train * np.exp(np.clip(base_lp_grid, -40.0, 40.0))
    g0 = float(np.sum(base_mass))
    if g0 <= 1e-12:
        return 1e-4

    def _deriv(gamma: float) -> float:
        expo = np.exp(np.clip(signed * float(gamma) * feat_grid, -40.0, 40.0))
        g = float(np.dot(base_mass, expo))
        if g <= 1e-300:
            return -1e18
        num = float(np.dot(base_mass, feat_grid * expo))
        return signed * s_event - signed * float(feat_event.size) * num / g

    d0 = _deriv(0.0)
    if not np.isfinite(d0) or d0 <= 1e-10:
        return 1e-4

    gamma_cap = 40.0 / max_feat
    lo = 0.0
    hi = min(1.0, gamma_cap)
    while hi < gamma_cap and _deriv(hi) > 0.0:
        lo = hi
        hi = min(2.0 * hi, gamma_cap)
    if hi >= gamma_cap and _deriv(hi) > 0.0:
        return float(max(gamma_cap, 1e-4))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _deriv(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(max(0.5 * (lo + hi), 1e-4))


def profiled_gain_canonical_sign(
    *,
    feat_event: np.ndarray,
    feat_grid: np.ndarray,
    base_lp_grid: np.ndarray,
    grid_weights_train: np.ndarray,
    sign: str,
    penalty: float,
) -> tuple[float, float]:
    base_lp_grid = np.asarray(base_lp_grid, dtype=np.float64)
    base_mass = np.asarray(grid_weights_train, dtype=np.float64) * np.exp(np.clip(base_lp_grid, -40.0, 40.0))
    base_norm = float(np.sum(base_mass))
    if base_norm <= 1e-12 or not np.isfinite(base_norm):
        return -1e18, 1e-4
    coef = profiled_coef_canonical(
        feat_event=feat_event,
        feat_grid=feat_grid,
        base_lp_grid=base_lp_grid,
        grid_weights_train=grid_weights_train,
        sign=str(sign),
    )
    signed = 1.0 if str(sign) == "exc" else -1.0
    expo = np.exp(np.clip(signed * float(coef) * np.asarray(feat_grid, dtype=np.float64), -40.0, 40.0))
    new_norm = float(np.dot(base_mass, expo))
    if not np.isfinite(new_norm) or new_norm <= 1e-300:
        return -1e18, float(max(coef, 1e-4))
    s_event = float(np.sum(np.asarray(feat_event, dtype=np.float64)))
    n_events = int(np.asarray(feat_event).size)
    ll_gain = signed * float(coef) * s_event - float(n_events) * math.log(new_norm / base_norm)
    return float(ll_gain - float(penalty)), float(max(coef, 1e-4))


def optimize_active_set_torch(
    *,
    active_rules: list[ActiveRule],
    subsets,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    init_mu: float,
    init_coef_map: dict[tuple[int, str], float],
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None = None,
    steps: int,
    lr: float,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    bic_num_sequences: int | None = None,
    intensity_model: str = "multiplicative",
    kernel_anchor_heights: dict[tuple[int, int], np.ndarray] | None = None,
    kernel_anchor_ridge: float = 0.0,
    kernel_tie_mode: str = "none",
) -> tuple[float, float, dict[int, float], dict[int, float], dict[tuple[int, int], np.ndarray], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    if not active_rules:
        mu = base_rate_fit(0, float(np.sum(grid_weights_train)))
        return float("inf"), mu, {}, {}, rule_heights, {}

    knots = basis_cache.knots
    area_weights = torch.tensor(trapz_area_weights(knots), dtype=torch.float32, device=device)
    gw_tr = torch.tensor(grid_weights_train, dtype=torch.float32, device=device)
    gw_va = torch.tensor(grid_weights_val, dtype=torch.float32, device=device)

    if torch_basis_cache is None:
        torch_basis_cache = TorchBasisCache(basis_cache, device)

    raw_mu = torch.nn.Parameter(torch.tensor([inverse_softplus(np.asarray([init_mu], dtype=np.float64))[0]], dtype=torch.float32, device=device))
    raw_coef = torch.nn.ParameterList()
    raw_heights: dict[tuple[object, ...], torch.nn.Parameter] = {}
    anchor_tensors: dict[tuple[object, ...], torch.Tensor] = {}

    def param_key_for(ar: ActiveRule, src: int):
        if str(kernel_tie_mode) == "source":
            return ("src", int(src))
        if str(kernel_tie_mode) == "source_sign":
            return ("src_sign", int(src), str(ar.sign))
        return ("rule_src", int(ar.idx), int(src))

    for ar in active_rules:
        coef0 = float(init_coef_map.get((int(ar.idx), str(ar.sign)), 0.1))
        raw_coef.append(torch.nn.Parameter(torch.tensor([inverse_softplus(np.asarray([coef0], dtype=np.float64))[0]], dtype=torch.float32, device=device)))
        for src in subsets[int(ar.idx)]:
            key = (int(ar.idx), int(src))
            pkey = param_key_for(ar, int(src))
            if pkey not in raw_heights:
                h0 = np.asarray(rule_heights[key], dtype=np.float64)
                raw_heights[pkey] = torch.nn.Parameter(torch.tensor(inverse_softplus(h0), dtype=torch.float32, device=device))
                if kernel_anchor_heights is not None and key in kernel_anchor_heights:
                    h_anchor = np.asarray(kernel_anchor_heights[key], dtype=np.float64)
                    h_anchor = h_anchor / max(float(np.dot(trapz_area_weights(knots), h_anchor)), 1e-12)
                    anchor_tensors[pkey] = torch.tensor(h_anchor, dtype=torch.float32, device=device)

    unique_sources = sorted({int(src) for ar in active_rules for src in subsets[int(ar.idx)]})
    basis_tensors = {int(src): torch_basis_cache.arrays(int(src)) for src in unique_sources}
    rule_specs = [
        (
            ar,
            tuple(int(src) for src in subsets[int(ar.idx)]),
            tuple(param_key_for(ar, int(src)) for src in subsets[int(ar.idx)]),
        )
        for ar in active_rules
    ]
    first_src = int(rule_specs[0][1][0])
    first_tr_ev, first_tr_gr, first_va_ev, first_va_gr = basis_tensors[first_src]
    tr_event_len = int(first_tr_ev.shape[0])
    tr_grid_len = int(first_tr_gr.shape[0])
    va_event_len = int(first_va_ev.shape[0])
    va_grid_len = int(first_va_gr.shape[0])
    empty_val = torch.zeros((0,), dtype=torch.float32, device=device)

    params = [raw_mu] + list(raw_coef) + list(raw_heights.values())
    opt = torch.optim.Adam(params, lr=float(lr))

    best_state = None
    best_bic = float("inf")

    def current_model(emit_numpy: bool = False, include_val: bool = True):
        mu = F.softplus(raw_mu[0]) + 1e-8
        exc_event = torch.zeros((tr_event_len,), dtype=torch.float32, device=device)
        inh_event = torch.zeros((tr_event_len,), dtype=torch.float32, device=device)
        exc_grid = torch.zeros((tr_grid_len,), dtype=torch.float32, device=device)
        inh_grid = torch.zeros((tr_grid_len,), dtype=torch.float32, device=device)
        if include_val:
            exc_event_va = torch.zeros((va_event_len,), dtype=torch.float32, device=device)
            inh_event_va = torch.zeros((va_event_len,), dtype=torch.float32, device=device)
            exc_grid_va = torch.zeros((va_grid_len,), dtype=torch.float32, device=device)
            inh_grid_va = torch.zeros((va_grid_len,), dtype=torch.float32, device=device)
        else:
            exc_event_va = empty_val
            inh_event_va = empty_val
            exc_grid_va = empty_val
            inh_grid_va = empty_val
        arrays_out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = {} if emit_numpy else None
        coef_out: dict[tuple[int, str], float] = {}
        heights_out: dict[tuple[int, int], np.ndarray] | None = {} if emit_numpy else None
        kernel_reg = torch.zeros((), dtype=torch.float32, device=device)

        for j, (ar, rule_sources, rule_pkeys) in enumerate(rule_specs):
            coef = F.softplus(raw_coef[j][0]) + 1e-8
            if emit_numpy:
                coef_out[(int(ar.idx), str(ar.sign))] = float(coef.detach().cpu().item())
            feat_tr_ev = torch.ones((tr_event_len,), dtype=torch.float32, device=device)
            feat_tr_gr = torch.ones((tr_grid_len,), dtype=torch.float32, device=device)
            if include_val:
                feat_va_ev = torch.ones((va_event_len,), dtype=torch.float32, device=device)
                feat_va_gr = torch.ones((va_grid_len,), dtype=torch.float32, device=device)
            else:
                feat_va_ev = empty_val
                feat_va_gr = empty_val
            for src, pkey in zip(rule_sources, rule_pkeys):
                key = (int(ar.idx), int(src))
                h = F.softplus(raw_heights[pkey]) + 1e-8
                h = h / torch.clamp(torch.dot(area_weights, h), min=1e-8)
                if float(kernel_anchor_ridge) > 0.0 and pkey in anchor_tensors:
                    kernel_reg = kernel_reg + torch.sum((h - anchor_tensors[pkey]) ** 2)
                if heights_out is not None:
                    heights_out[key] = h.detach().cpu().numpy().astype(np.float64, copy=True)
                b_tr_ev, b_tr_gr, b_va_ev, b_va_gr = basis_tensors[int(src)]
                z_tr_ev = b_tr_ev @ h
                z_tr_gr = b_tr_gr @ h
                feat_tr_ev = feat_tr_ev * (1.0 - torch.exp(-torch.clamp(z_tr_ev, min=0.0)))
                feat_tr_gr = feat_tr_gr * (1.0 - torch.exp(-torch.clamp(z_tr_gr, min=0.0)))
                if include_val:
                    z_va_ev = b_va_ev @ h
                    z_va_gr = b_va_gr @ h
                    feat_va_ev = feat_va_ev * (1.0 - torch.exp(-torch.clamp(z_va_ev, min=0.0)))
                    feat_va_gr = feat_va_gr * (1.0 - torch.exp(-torch.clamp(z_va_gr, min=0.0)))
            if arrays_out is not None:
                arrays_out[int(ar.idx)] = (
                    feat_tr_ev.detach().cpu().numpy().astype(np.float64, copy=False),
                    feat_tr_gr.detach().cpu().numpy().astype(np.float64, copy=False),
                    feat_va_ev.detach().cpu().numpy().astype(np.float64, copy=False),
                    feat_va_gr.detach().cpu().numpy().astype(np.float64, copy=False),
                )
            if ar.sign == "exc":
                exc_event = exc_event + feat_tr_ev * coef
                exc_grid = exc_grid + feat_tr_gr * coef
                exc_event_va = exc_event_va + feat_va_ev * coef
                exc_grid_va = exc_grid_va + feat_va_gr * coef
            else:
                inh_event = inh_event + feat_tr_ev * coef
                inh_grid = inh_grid + feat_tr_gr * coef
                inh_event_va = inh_event_va + feat_va_ev * coef
                inh_grid_va = inh_grid_va + feat_va_gr * coef
        return (
            mu,
            exc_event,
            inh_event,
            exc_grid,
            inh_grid,
            exc_event_va,
            inh_event_va,
            exc_grid_va,
            inh_grid_va,
            kernel_reg,
            coef_out,
            heights_out,
            arrays_out,
        )

    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        (
            mu,
            exc_event,
            inh_event,
            exc_grid,
            inh_grid,
            exc_event_va,
            inh_event_va,
            exc_grid_va,
            inh_grid_va,
            kernel_reg,
            _coef_out,
            _heights_out,
            _arrays_out,
        ) = current_model(emit_numpy=False, include_val=False)

        if str(intensity_model) == "canonical_loglink":
            train_nll = -(torch.log(mu) * float(exc_event.numel()) + exc_event.sum() - inh_event.sum()) + (gw_tr * mu * torch.exp(torch.clamp(exc_grid - inh_grid, min=-40.0, max=40.0))).sum()
        else:
            eta_event = torch.clamp(mu + exc_event, min=1e-8)
            eta_grid = torch.clamp(mu + exc_grid, min=1e-8)
            train_nll = -(torch.log(eta_event) - inh_event).sum() + (gw_tr * eta_grid * torch.exp(-inh_grid)).sum()
        if float(kernel_anchor_ridge) > 0.0:
            train_nll = train_nll + float(kernel_anchor_ridge) * kernel_reg
        train_nll.backward()
        opt.step()

        if step % 10 == 0 or step == int(steps) - 1:
            with torch.no_grad():
                (
                    mu_b,
                    exc_event_b,
                    inh_event_b,
                    exc_grid_b,
                    inh_grid_b,
                    exc_event_va_b,
                    inh_event_va_b,
                    exc_grid_va_b,
                    inh_grid_va_b,
                    _kernel_reg_b,
                    _coef_out_b,
                    _heights_out_b,
                    _arrays_out_b,
                ) = current_model(emit_numpy=False)
                if str(intensity_model) == "canonical_loglink":
                    ll_val = (torch.log(mu_b) * float(exc_event_va_b.numel()) + exc_event_va_b.sum() - inh_event_va_b.sum()) - (gw_va * mu_b * torch.exp(torch.clamp(exc_grid_va_b - inh_grid_va_b, min=-40.0, max=40.0))).sum()
                else:
                    eta_ev_va = torch.clamp(mu_b + exc_event_va_b, min=1e-8)
                    eta_gr_va = torch.clamp(mu_b + exc_grid_va_b, min=1e-8)
                    ll_val = (torch.log(eta_ev_va) - inh_event_va_b).sum() - (gw_va * eta_gr_va * torch.exp(-inh_grid_va_b)).sum()
                n_eff = bic_sample_size(int(bic_num_sequences) if bic_num_sequences is not None else int(exc_grid_va_b.numel()))
                bic = float(
                    (-2.0 * ll_val).detach().cpu().item()
                    + (
                        float(penalty_scale) * float(model_param_dim(active_rules, subsets, int(len(knots)))) * math.log(float(n_eff))
                        if penalize_kernel_df
                        else float(penalty_scale) * float(1 + len(active_rules)) * math.log(float(n_eff))
                    )
                )
                if bic < best_bic:
                    (
                        mu_emit,
                        _exc_event_emit,
                        _inh_event_emit,
                        _exc_grid_emit,
                        _inh_grid_emit,
                        _exc_event_va_emit,
                        _inh_event_va_emit,
                        _exc_grid_va_emit,
                        _inh_grid_va_emit,
                        _kernel_reg_emit,
                        coef_out_b,
                        heights_out_b,
                        arrays_out_b,
                    ) = current_model(emit_numpy=True)
                    best_bic = bic
                    best_state = (
                        float(mu_emit.detach().cpu().item()),
                        coef_out_b,
                        heights_out_b,
                        arrays_out_b,
                    )

    if best_state is None:
        raise RuntimeError("Active-set optimization failed to produce a valid state")

    mu_fit, coef_out, heights_out, arrays_out = best_state
    exc_params = {int(idx): float(coef) for (idx, sign), coef in coef_out.items() if sign == "exc"}
    inh_params = {int(idx): float(coef) for (idx, sign), coef in coef_out.items() if sign == "inh"}
    for key, h in heights_out.items():
        rule_heights[key] = np.asarray(h, dtype=np.float64)
    return best_bic, float(mu_fit), exc_params, inh_params, rule_heights, arrays_out


def contributions_from_active(
    *,
    active_rules: list[ActiveRule],
    arrays_out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    tr_event_len: int,
    tr_grid_len: int,
):
    exc_ev = np.zeros((tr_event_len,), dtype=np.float64)
    inh_ev = np.zeros((tr_event_len,), dtype=np.float64)
    exc_gr = np.zeros((tr_grid_len,), dtype=np.float64)
    inh_gr = np.zeros((tr_grid_len,), dtype=np.float64)
    for ar in active_rules:
        arr = arrays_out[int(ar.idx)]
        if ar.sign == "exc":
            coef = float(exc_params.get(int(ar.idx), 0.0))
            exc_ev += coef * arr[0]
            exc_gr += coef * arr[1]
        else:
            coef = float(inh_params.get(int(ar.idx), 0.0))
            inh_ev += coef * arr[0]
            inh_gr += coef * arr[1]
    return exc_ev, inh_ev, exc_gr, inh_gr


def fit_constant_bic(
    n_events_val: int,
    grid_weights_val: np.ndarray,
    n_events_train: int,
    grid_weights_train: np.ndarray,
    *,
    n_val_sequences: int,
):
    mu = base_rate_fit(n_events_train, float(np.sum(grid_weights_train)))
    ll_val = float(n_events_val) * math.log(max(mu, 1e-8)) - float(np.sum(grid_weights_val)) * mu
    bic = -2.0 * ll_val + math.log(float(bic_sample_size(int(n_val_sequences))))
    return float(bic), float(mu)


def maximal_nested_families(active_rules: list[ActiveRule], subsets) -> list[list[ActiveRule]]:
    rules = list(active_rules)
    maximal: list[tuple[int, ...]] = []
    for ar in rules:
        subset = tuple(int(s) for s in subsets[int(ar.idx)])
        if any(set(subset).issubset(set(subsets[int(other.idx)])) and len(tuple(subsets[int(other.idx)])) > len(subset) for other in rules):
            continue
        maximal.append(subset)
    families: list[list[ActiveRule]] = []
    seen: set[tuple[int, ...]] = set()
    for max_subset in maximal:
        family = [
            ar
            for ar in rules
            if set(tuple(int(s) for s in subsets[int(ar.idx)])).issubset(set(max_subset))
        ]
        sizes = {len(tuple(subsets[int(ar.idx)])) for ar in family}
        if len(family) < 2 or len(sizes) < 2:
            continue
        key = tuple(sorted(int(ar.idx) for ar in family))
        if key in seen:
            continue
        seen.add(key)
        families.append(family)
    return families


def validation_bic_from_arrays(
    *,
    active_rules: list[ActiveRule],
    arrays_out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    mu: float,
    subsets,
    num_knots: int,
    grid_weights_val: np.ndarray,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    num_sequences_val: int | None = None,
    intensity_model: str = "multiplicative",
) -> float:
    if not active_rules:
        raise ValueError("validation_bic_from_arrays requires a non-empty active_rules list")

    va_event_len = int(next(iter(arrays_out.values()))[2].size)
    exc_ev = np.zeros((va_event_len,), dtype=np.float64)
    inh_ev = np.zeros((va_event_len,), dtype=np.float64)
    exc_gr = np.zeros((grid_weights_val.size,), dtype=np.float64)
    inh_gr = np.zeros((grid_weights_val.size,), dtype=np.float64)
    for ar in active_rules:
        arr = arrays_out[int(ar.idx)]
        if ar.sign == "exc":
            coef = float(exc_params.get(int(ar.idx), 0.0))
            exc_ev += coef * arr[2]
            exc_gr += coef * arr[3]
        else:
            coef = float(inh_params.get(int(ar.idx), 0.0))
            inh_ev += coef * arr[2]
            inh_gr += coef * arr[3]
    if str(intensity_model) == "canonical_loglink":
        ll_val = float(va_event_len * math.log(max(float(mu), 1e-8)) + np.sum(exc_ev) - np.sum(inh_ev) - np.dot(grid_weights_val, float(mu) * np.exp(np.clip(exc_gr - inh_gr, -40.0, 40.0))))
    else:
        eta_ev = np.clip(float(mu) + exc_ev, 1e-8, None)
        eta_gr = np.clip(float(mu) + exc_gr, 1e-8, None)
        ll_val = float(np.sum(np.log(eta_ev) - inh_ev) - np.dot(grid_weights_val, eta_gr * np.exp(-inh_gr)))
    n_eff = bic_sample_size(int(num_sequences_val) if num_sequences_val is not None else int(grid_weights_val.size))
    return float(
        -2.0 * ll_val
        + (
            float(penalty_scale) * float(model_param_dim(active_rules, subsets, int(num_knots))) * math.log(float(n_eff))
            if penalize_kernel_df
            else float(penalty_scale) * float(1 + len(active_rules)) * math.log(float(n_eff))
        )
    )


def build_init_coef_map(
    active_rules: list[ActiveRule],
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    fallback: float = 0.1,
) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    for ar in active_rules:
        if ar.sign == "exc":
            out[(int(ar.idx), "exc")] = float(exc_params.get(int(ar.idx), fallback))
        else:
            out[(int(ar.idx), "inh")] = float(inh_params.get(int(ar.idx), fallback))
    return out


def family_attribution_refine(
    *,
    active_rules: list[ActiveRule],
    subsets,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    mu: float,
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None = None,
    opt_steps: int,
    lr: float,
    passes: int,
    penalize_kernel_df: bool = False,
    num_val_sequences: int | None = None,
    intensity_model: str = "multiplicative",
    kernel_anchor_heights: dict[tuple[int, int], np.ndarray] | None = None,
    kernel_anchor_ridge: float = 0.0,
    kernel_tie_mode: str = "none",
    return_arrays: bool = False,
):
    if not active_rules or int(passes) <= 0:
        if bool(return_arrays):
            return float("inf"), float(mu), exc_params, inh_params, rule_heights, active_rules, {}
        return float("inf"), float(mu), exc_params, inh_params, rule_heights, active_rules

    knots = np.asarray(basis_cache.knots, dtype=np.float64)
    area_weights = trapz_area_weights(knots)
    families = maximal_nested_families(active_rules, subsets)
    if not families:
        init_coef_map = build_init_coef_map(active_rules, exc_params, inh_params, fallback=0.1)
        bic, mu_fit, exc_fit, inh_fit, rule_heights, _ = optimize_active_set_torch(
            active_rules=active_rules,
            subsets=subsets,
            basis_cache=basis_cache,
            rule_heights=rule_heights,
            init_mu=mu,
            init_coef_map=init_coef_map,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            steps=max(40, int(opt_steps)),
            lr=lr,
            penalize_kernel_df=bool(penalize_kernel_df),
            bic_num_sequences=num_val_sequences,
            intensity_model=str(intensity_model),
            kernel_anchor_heights=kernel_anchor_heights,
            kernel_anchor_ridge=float(kernel_anchor_ridge),
            kernel_tie_mode=str(kernel_tie_mode),
        )
        if bool(return_arrays):
            return bic, float(mu_fit), exc_fit, inh_fit, rule_heights, active_rules, _
        return bic, float(mu_fit), exc_fit, inh_fit, rule_heights, active_rules

    cur_mu = float(mu)
    cur_exc = dict(exc_params)
    cur_inh = dict(inh_params)
    cur_rules = list(active_rules)
    cur_heights = rule_heights
    best_bic = float("inf")

    for _ in range(int(passes)):
        init_coef_map = build_init_coef_map(cur_rules, cur_exc, cur_inh, fallback=0.1)
        best_bic, mu_fit, exc_fit, inh_fit, cur_heights, arrays_out = optimize_active_set_torch(
            active_rules=cur_rules,
            subsets=subsets,
            basis_cache=basis_cache,
            rule_heights=cur_heights,
            init_mu=cur_mu,
            init_coef_map=init_coef_map,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            steps=max(40, int(opt_steps) // 2),
            lr=lr,
            penalize_kernel_df=bool(penalize_kernel_df),
            bic_num_sequences=num_val_sequences,
            intensity_model=str(intensity_model),
            kernel_anchor_heights=kernel_anchor_heights,
            kernel_anchor_ridge=float(kernel_anchor_ridge),
            kernel_tie_mode=str(kernel_tie_mode),
        )
        cur_mu = float(mu_fit)
        cur_exc = dict(exc_fit)
        cur_inh = dict(inh_fit)

        exc_ev, inh_ev, exc_gr, inh_gr = contributions_from_active(
            active_rules=cur_rules,
            arrays_out=arrays_out,
            exc_params=cur_exc,
            inh_params=cur_inh,
            tr_event_len=int(next(iter(arrays_out.values()))[0].size),
            tr_grid_len=int(next(iter(arrays_out.values()))[1].size),
        )
        if str(intensity_model) == "canonical_loglink":
            lam_ev = float(cur_mu) * np.exp(np.clip(exc_ev - inh_ev, -40.0, 40.0))
            lam_gr = float(cur_mu) * np.exp(np.clip(exc_gr - inh_gr, -40.0, 40.0))
        else:
            eta_ev = np.clip(float(cur_mu) + exc_ev, 1e-8, None)
            eta_gr = np.clip(float(cur_mu) + exc_gr, 1e-8, None)
            lam_gr = eta_gr * np.exp(-inh_gr)

        stat_sum: dict[tuple[int, int], np.ndarray] = {}
        stat_count: dict[tuple[int, int], float] = {}

        for family in families:
            family_inh = [ar for ar in family if ar.sign == "inh"]
            fam_inh_gr = None
            if family_inh:
                fam_inh_gr = np.zeros_like(next(iter(arrays_out.values()))[1], dtype=np.float64)
                for ar in family_inh:
                    fam_inh_gr += float(cur_inh.get(int(ar.idx), 0.0)) * arrays_out[int(ar.idx)][1]
                fam_inh_gr = np.clip(fam_inh_gr, 1e-8, None)

            for ar in family:
                subset = tuple(int(s) for s in subsets[int(ar.idx)])
                if ar.sign == "exc":
                    coef = float(cur_exc.get(int(ar.idx), 0.0))
                    if coef <= 1e-10:
                        continue
                    if str(intensity_model) == "canonical_loglink":
                        # In the log-link model, excitation enters linearly in the
                        # event log-likelihood term, so event-side attribution should
                        # follow the rule's direct event contribution rather than the
                        # multiplicative-model rate share.
                        weights = coef * arrays_out[int(ar.idx)][0]
                    else:
                        weights = coef * arrays_out[int(ar.idx)][0] / eta_ev
                    if float(np.sum(weights)) <= 1e-10:
                        continue
                    for src in subset:
                        b_tr_ev, _, _, _ = basis_cache.arrays(int(src))
                        stat = np.asarray(weights @ b_tr_ev, dtype=np.float64)
                        key = (int(ar.idx), int(src))
                        stat_sum[key] = stat if key not in stat_sum else stat_sum[key] + stat
                        stat_count[key] = stat_count.get(key, 0.0) + 1.0
                else:
                    coef = float(cur_inh.get(int(ar.idx), 0.0))
                    if coef <= 1e-10 or fam_inh_gr is None:
                        continue
                    share = coef * arrays_out[int(ar.idx)][1] / fam_inh_gr
                    weights = np.asarray(grid_weights_train, dtype=np.float64) * lam_gr * share
                    if float(np.sum(weights)) <= 1e-10:
                        continue
                    for src in subset:
                        _, b_tr_gr, _, _ = basis_cache.arrays(int(src))
                        stat = np.asarray(weights @ b_tr_gr, dtype=np.float64)
                        key = (int(ar.idx), int(src))
                        stat_sum[key] = stat if key not in stat_sum else stat_sum[key] + stat
                        stat_count[key] = stat_count.get(key, 0.0) + 1.0

        for key, stat in stat_sum.items():
            avg = stat / max(float(stat_count.get(key, 1.0)), 1.0)
            if float(np.sum(avg)) <= 1e-12:
                continue
            cur_heights[key] = normalize_piecewise_area(knots, np.maximum(avg, 0.0))

    init_coef_map = build_init_coef_map(cur_rules, cur_exc, cur_inh, fallback=0.1)
    best_bic, mu_fit, exc_fit, inh_fit, cur_heights, arrays_out = optimize_active_set_torch(
        active_rules=cur_rules,
        subsets=subsets,
        basis_cache=basis_cache,
        rule_heights=cur_heights,
        init_mu=cur_mu,
        init_coef_map=init_coef_map,
        grid_weights_train=grid_weights_train,
        grid_weights_val=grid_weights_val,
        device=device,
        torch_basis_cache=torch_basis_cache,
        steps=max(60, int(opt_steps)),
        lr=lr,
        penalize_kernel_df=bool(penalize_kernel_df),
        bic_num_sequences=num_val_sequences,
        intensity_model=str(intensity_model),
        kernel_anchor_heights=kernel_anchor_heights,
        kernel_anchor_ridge=float(kernel_anchor_ridge),
        kernel_tie_mode=str(kernel_tie_mode),
    )
    if bool(return_arrays):
        return best_bic, float(mu_fit), exc_fit, inh_fit, cur_heights, cur_rules, arrays_out
    return best_bic, float(mu_fit), exc_fit, inh_fit, cur_heights, cur_rules




def run_active_set(
    *,
    subsets,
    init_arrays,
    score_arrays=None,
    rule_heights,
    basis_cache,
    grid_weights_train,
    grid_weights_val,
    max_rules,
    opt_steps,
    lr,
    device,
    torch_basis_cache: TorchBasisCache | None = None,
    penalize_kernel_df,
    num_val_sequences: int,
    intensity_model: str = "multiplicative",
):
    n_events_train = int(next(iter(init_arrays.values()))[0].size)
    n_grid_train = int(next(iter(init_arrays.values()))[1].size)
    n_events_val = int(next(iter(init_arrays.values()))[2].size)
    base_bic, mu = fit_constant_bic(
        n_events_val=n_events_val,
        grid_weights_val=grid_weights_val,
        n_events_train=n_events_train,
        grid_weights_train=grid_weights_train,
        n_val_sequences=int(num_val_sequences),
    )
    active_rules: list[ActiveRule] = []
    exc_params: dict[int, float] = {}
    inh_params: dict[int, float] = {}
    arrays_active: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    best_bic = base_bic
    while len(active_rules) < int(max_rules):
        exc_ev, inh_ev, exc_gr, inh_gr = contributions_from_active(
            active_rules=active_rules,
            arrays_out=arrays_active,
            exc_params=exc_params,
            inh_params=inh_params,
            tr_event_len=n_events_train,
            tr_grid_len=n_grid_train,
        ) if active_rules else (
            np.zeros((n_events_train,), dtype=np.float64),
            np.zeros((n_events_train,), dtype=np.float64),
            np.zeros((n_grid_train,), dtype=np.float64),
            np.zeros((n_grid_train,), dtype=np.float64),
        )

        active_idx_set = {int(ar.idx) for ar in active_rules}
        stale_trials = []
        if score_arrays is None:
            score_arrays = init_arrays

        for idx, arr in init_arrays.items():
            if int(idx) in active_idx_set:
                continue
            score_arr = score_arrays.get(int(idx), arr)
            candidate_penalty = (
                0.5 * float(rule_param_dim(subsets[int(idx)], basis_cache.knots.size)) * math.log(float(bic_sample_size(int(num_val_sequences))))
                if penalize_kernel_df
                else 0.5 * math.log(float(bic_sample_size(int(num_val_sequences))))
            )
            gain, sign, coef0 = rule_score(
                feat_event=score_arr[0],
                feat_grid=score_arr[1],
                mu=mu,
                exc_event=exc_ev,
                inh_event=inh_ev,
                exc_grid=exc_gr,
                inh_grid=inh_gr,
                grid_weights_train=grid_weights_train,
                penalty=candidate_penalty,
                intensity_model=str(intensity_model),
            )
            stale_trials.append((float(gain), int(idx), str(sign), float(coef0)))

        if not stale_trials:
            break

        stale_trials.sort(key=lambda x: x[0], reverse=True)
        best_trial = max(stale_trials, key=lambda x: x[0])
        if best_trial[0] <= 1e-8:
            break

        _, idx_add, sign_add, coef0 = best_trial
        trial_active = active_rules + [ActiveRule(idx=int(idx_add), sign=str(sign_add))]
        init_coef_map = build_init_coef_map(trial_active, exc_params, inh_params, fallback=float(coef0))
        init_coef_map[(int(idx_add), str(sign_add))] = float(coef0)
        bic, mu_fit, exc_fit, inh_fit, rule_heights, arrays_fit = optimize_active_set_torch(
            active_rules=trial_active,
            subsets=subsets,
            basis_cache=basis_cache,
            rule_heights=rule_heights,
            init_mu=mu,
            init_coef_map=init_coef_map,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            steps=opt_steps,
            lr=lr,
            penalize_kernel_df=bool(penalize_kernel_df),
            bic_num_sequences=int(num_val_sequences),
            intensity_model=str(intensity_model),
        )
        if float(bic) + 1e-8 < best_bic:
            best_bic = float(bic)
            mu = float(mu_fit)
            active_rules = trial_active
            exc_params = exc_fit
            inh_params = inh_fit
            arrays_active = arrays_fit
            continue
        break

    improved = True
    while improved and active_rules:
        improved = False
        for j in range(len(active_rules)):
            trial_active = active_rules[:j] + active_rules[j + 1 :]
            if not trial_active:
                trial_bic, trial_mu = fit_constant_bic(
                    n_events_val=n_events_val,
                    grid_weights_val=grid_weights_val,
                    n_events_train=n_events_train,
                    grid_weights_train=grid_weights_train,
                    n_val_sequences=int(num_val_sequences),
                )
                if float(trial_bic) + 1e-8 < best_bic:
                    best_bic = float(trial_bic)
                    mu = float(trial_mu)
                    active_rules = []
                    exc_params = {}
                    inh_params = {}
                    arrays_active = {}
                    improved = True
                    break
                continue
            init_coef_map = build_init_coef_map(trial_active, exc_params, inh_params, fallback=0.1)
            bic, mu_fit, exc_fit, inh_fit, rule_heights, arrays_fit = optimize_active_set_torch(
                active_rules=trial_active,
                subsets=subsets,
                basis_cache=basis_cache,
                rule_heights=rule_heights,
                init_mu=mu,
                init_coef_map=init_coef_map,
                grid_weights_train=grid_weights_train,
                grid_weights_val=grid_weights_val,
                device=device,
                torch_basis_cache=torch_basis_cache,
                steps=max(40, opt_steps // 2),
                lr=lr,
                penalize_kernel_df=bool(penalize_kernel_df),
                bic_num_sequences=int(num_val_sequences),
                intensity_model=str(intensity_model),
            )
            if float(bic) + 1e-8 < best_bic:
                best_bic = float(bic)
                mu = float(mu_fit)
                active_rules = trial_active
                exc_params = exc_fit
                inh_params = inh_fit
                arrays_active = arrays_fit
                improved = True
                break
    return best_bic, mu, exc_params, inh_params, rule_heights, active_rules

@functools.lru_cache(maxsize=None)
def support_partitions(support: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    items = tuple(int(s) for s in sorted(support))
    if not items:
        return [tuple()]

    def _rec(rest: tuple[int, ...]) -> list[list[list[int]]]:
        if not rest:
            return [[]]
        head, tail = int(rest[0]), tuple(int(x) for x in rest[1:])
        tail_parts = _rec(tail)
        out: list[list[list[int]]] = []
        for part in tail_parts:
            out.append([[head]] + [list(block) for block in part])
            for j in range(len(part)):
                updated = [list(block) for block in part]
                updated[j] = sorted(updated[j] + [head])
                out.append(updated)
        return out

    unique: set[tuple[tuple[int, ...], ...]] = set()
    for part in _rec(items):
        canon = tuple(sorted(tuple(sorted(int(x) for x in block)) for block in part))
        unique.add(canon)
    return sorted(unique, key=lambda part: (len(part), part))


def exact_seed_partition_step(
    *,
    seed_idx: int,
    active_rules: list[ActiveRule],
    subsets,
    init_arrays,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    mu: float,
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None,
    opt_steps: int,
    lr: float,
    penalize_kernel_df: bool,
    num_val_sequences: int,
    intensity_model: str,
) -> tuple[float, float, dict[int, float], dict[int, float], dict[tuple[int, int], np.ndarray], list[ActiveRule], tuple[int, ...]]:
    seed_support = tuple(int(s) for s in subsets[int(seed_idx)])
    seed_set = set(seed_support)
    subset_to_idx = {tuple(int(s) for s in subset): int(idx) for idx, subset in enumerate(subsets)}
    inside_active = [ar for ar in active_rules if set(int(s) for s in subsets[int(ar.idx)]).issubset(seed_set)]
    outside_active = [ar for ar in active_rules if ar not in inside_active]
    active_signs = {int(ar.idx): str(ar.sign) for ar in active_rules}

    n_events_train = int(next(iter(init_arrays.values()))[0].size)
    n_grid_train = int(next(iter(init_arrays.values()))[1].size)
    exc_ev, inh_ev, exc_gr, inh_gr = contributions_from_active(
        active_rules=outside_active,
        arrays_out={int(ar.idx): init_arrays[int(ar.idx)] for ar in outside_active},
        exc_params=exc_params,
        inh_params=inh_params,
        tr_event_len=n_events_train,
        tr_grid_len=n_grid_train,
    ) if outside_active else (
        np.zeros((n_events_train,), dtype=np.float64),
        np.zeros((n_events_train,), dtype=np.float64),
        np.zeros((n_grid_train,), dtype=np.float64),
        np.zeros((n_grid_train,), dtype=np.float64),
    )
    penalty = (
        0.5 * math.log(float(bic_sample_size(int(num_val_sequences))))
        if not penalize_kernel_df
        else None
    )

    block_rule_cache: dict[tuple[int, ...], ActiveRule] = {}
    block_coef_cache: dict[tuple[int, str], float] = {}

    def resolve_block_rule(block: tuple[int, ...]) -> ActiveRule:
        block = tuple(sorted(int(s) for s in block))
        idx = int(subset_to_idx[block])
        if idx in active_signs:
            sign = str(active_signs[idx])
            coef0 = float(exc_params[idx] if sign == "exc" else inh_params[idx])
        else:
            arr = init_arrays[idx]
            candidate_penalty = (
                0.5 * float(rule_param_dim(subsets[int(idx)], basis_cache.knots.size)) * math.log(float(bic_sample_size(int(num_val_sequences))))
                if penalize_kernel_df
                else float(penalty)
            )
            _gain, sign, coef0 = rule_score(
                feat_event=arr[0],
                feat_grid=arr[1],
                mu=mu,
                exc_event=exc_ev,
                inh_event=inh_ev,
                exc_grid=exc_gr,
                inh_grid=inh_gr,
                grid_weights_train=grid_weights_train,
                penalty=float(candidate_penalty),
                intensity_model=str(intensity_model),
            )
        block_coef_cache[(idx, str(sign))] = float(coef0)
        rule = ActiveRule(idx=idx, sign=str(sign))
        block_rule_cache[block] = rule
        return rule

    n_events_val = int(next(iter(init_arrays.values()))[2].size)
    if active_rules:
        current_bic, current_mu, current_exc, current_inh, current_heights, _ = optimize_active_set_torch(
            active_rules=active_rules,
            subsets=subsets,
            basis_cache=basis_cache,
            rule_heights=rule_heights,
            init_mu=mu,
            init_coef_map=build_init_coef_map(active_rules, exc_params, inh_params, fallback=0.1),
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            steps=max(40, int(opt_steps)),
            lr=lr,
            penalize_kernel_df=bool(penalize_kernel_df),
            bic_num_sequences=int(num_val_sequences),
            intensity_model=str(intensity_model),
        )
        best_state = (
            float(current_bic),
            float(current_mu),
            dict(current_exc),
            dict(current_inh),
            current_heights,
            list(active_rules),
        )
    else:
        current_bic, current_mu = fit_constant_bic(
            n_events_val=n_events_val,
            grid_weights_val=grid_weights_val,
            n_events_train=n_events_train,
            grid_weights_train=grid_weights_train,
            n_val_sequences=int(num_val_sequences),
        )
        best_state = (
            float(current_bic),
            float(current_mu),
            {},
            {},
            rule_heights,
            [],
        )

    for partition in support_partitions(seed_support):
        chart_rules = [resolve_block_rule(block) for block in partition]
        cand_rules = sorted(outside_active + chart_rules, key=lambda ar: (int(ar.idx), str(ar.sign)))
        init_coef_map = build_init_coef_map(cand_rules, exc_params, inh_params, fallback=0.1)
        for ar in chart_rules:
            key = (int(ar.idx), str(ar.sign))
            if key in block_coef_cache:
                init_coef_map[key] = float(block_coef_cache[key])
        cand_bic, cand_mu, cand_exc, cand_inh, cand_heights, _ = optimize_active_set_torch(
            active_rules=cand_rules,
            subsets=subsets,
            basis_cache=basis_cache,
            rule_heights=rule_heights,
            init_mu=mu,
            init_coef_map=init_coef_map,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            steps=max(30, int(opt_steps) // 2),
            lr=lr,
            penalize_kernel_df=bool(penalize_kernel_df),
            bic_num_sequences=int(num_val_sequences),
            intensity_model=str(intensity_model),
        )
        cand_state = (
            float(cand_bic),
            float(cand_mu),
            dict(cand_exc),
            dict(cand_inh),
            cand_heights,
            cand_rules,
        )
        if float(cand_state[0]) + 1e-8 < float(best_state[0]):
            best_state = cand_state
    return (*best_state, seed_support)


def run_active_set_seed_partition(
    *,
    subsets,
    init_arrays,
    score_arrays=None,
    rule_heights,
    basis_cache,
    grid_weights_train,
    grid_weights_val,
    max_rules,
    opt_steps,
    lr,
    device,
    torch_basis_cache: TorchBasisCache | None = None,
    penalize_kernel_df,
    num_val_sequences: int,
    intensity_model: str = "multiplicative",
):
    n_events_train = int(next(iter(init_arrays.values()))[0].size)
    n_grid_train = int(next(iter(init_arrays.values()))[1].size)
    n_events_val = int(next(iter(init_arrays.values()))[2].size)
    base_bic, mu = fit_constant_bic(
        n_events_val=n_events_val,
        grid_weights_val=grid_weights_val,
        n_events_train=n_events_train,
        grid_weights_train=grid_weights_train,
        n_val_sequences=int(num_val_sequences),
    )
    active_rules: list[ActiveRule] = []
    exc_params: dict[int, float] = {}
    inh_params: dict[int, float] = {}
    best_bic = float(base_bic)

    while len(active_rules) < int(max_rules):
        exc_ev, inh_ev, exc_gr, inh_gr = contributions_from_active(
            active_rules=active_rules,
            arrays_out={int(ar.idx): init_arrays[int(ar.idx)] for ar in active_rules},
            exc_params=exc_params,
            inh_params=inh_params,
            tr_event_len=n_events_train,
            tr_grid_len=n_grid_train,
        ) if active_rules else (
            np.zeros((n_events_train,), dtype=np.float64),
            np.zeros((n_events_train,), dtype=np.float64),
            np.zeros((n_grid_train,), dtype=np.float64),
            np.zeros((n_grid_train,), dtype=np.float64),
        )

        active_idx_set = {int(ar.idx) for ar in active_rules}
        stale_trials = []
        if score_arrays is None:
            score_arrays = init_arrays

        for idx, arr in init_arrays.items():
            if int(idx) in active_idx_set:
                continue
            score_arr = score_arrays.get(int(idx), arr)
            candidate_penalty = (
                0.5 * float(rule_param_dim(subsets[int(idx)], basis_cache.knots.size)) * math.log(float(bic_sample_size(int(num_val_sequences))))
                if penalize_kernel_df
                else 0.5 * math.log(float(bic_sample_size(int(num_val_sequences))))
            )
            gain, sign, coef0 = rule_score(
                feat_event=score_arr[0],
                feat_grid=score_arr[1],
                mu=mu,
                exc_event=exc_ev,
                inh_event=inh_ev,
                exc_grid=exc_gr,
                inh_grid=inh_gr,
                grid_weights_train=grid_weights_train,
                penalty=candidate_penalty,
                intensity_model=str(intensity_model),
            )
            stale_trials.append((float(gain), int(idx), str(sign), float(coef0)))

        if not stale_trials:
            break

        stale_trials.sort(key=lambda x: x[0], reverse=True)
        best_trial = max(stale_trials, key=lambda x: x[0])
        if float(best_trial[0]) <= 1e-8:
            break

        _, idx_add, _sign_add, _coef0 = best_trial
        trial_bic, trial_mu, trial_exc, trial_inh, trial_heights, trial_rules, _seed_support = exact_seed_partition_step(
            seed_idx=int(idx_add),
            active_rules=active_rules,
            subsets=subsets,
            init_arrays=init_arrays,
            basis_cache=basis_cache,
            rule_heights=rule_heights,
            mu=mu,
            exc_params=exc_params,
            inh_params=inh_params,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            opt_steps=int(opt_steps),
            lr=lr,
            penalize_kernel_df=bool(penalize_kernel_df),
            num_val_sequences=int(num_val_sequences),
            intensity_model=str(intensity_model),
        )
        if float(trial_bic) + 1e-8 < best_bic:
            best_bic = float(trial_bic)
            mu = float(trial_mu)
            active_rules = list(trial_rules)
            exc_params = dict(trial_exc)
            inh_params = dict(trial_inh)
            rule_heights = trial_heights
            continue
        break

    improved = True
    while improved and active_rules:
        improved = False
        for j in range(len(active_rules)):
            trial_active = active_rules[:j] + active_rules[j + 1 :]
            if not trial_active:
                trial_bic, trial_mu = fit_constant_bic(
                    n_events_val=n_events_val,
                    grid_weights_val=grid_weights_val,
                    n_events_train=n_events_train,
                    grid_weights_train=grid_weights_train,
                    n_val_sequences=int(num_val_sequences),
                )
                if float(trial_bic) + 1e-8 < best_bic:
                    best_bic = float(trial_bic)
                    mu = float(trial_mu)
                    active_rules = []
                    exc_params = {}
                    inh_params = {}
                    improved = True
                    break
                continue
            init_coef_map = build_init_coef_map(trial_active, exc_params, inh_params, fallback=0.1)
            bic, mu_fit, exc_fit, inh_fit, rule_heights, _arrays_fit = optimize_active_set_torch(
                active_rules=trial_active,
                subsets=subsets,
                basis_cache=basis_cache,
                rule_heights=rule_heights,
                init_mu=mu,
                init_coef_map=init_coef_map,
                grid_weights_train=grid_weights_train,
                grid_weights_val=grid_weights_val,
                device=device,
                torch_basis_cache=torch_basis_cache,
                steps=max(40, opt_steps // 2),
                lr=lr,
                penalize_kernel_df=bool(penalize_kernel_df),
                bic_num_sequences=int(num_val_sequences),
                intensity_model=str(intensity_model),
            )
            if float(bic) + 1e-8 < best_bic:
                best_bic = float(bic)
                mu = float(mu_fit)
                active_rules = trial_active
                exc_params = exc_fit
                inh_params = inh_fit
                improved = True
                break
    return best_bic, mu, exc_params, inh_params, rule_heights, active_rules


def post_prune_irreducible_rules(
    *,
    active_rules: list[ActiveRule],
    subsets,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    mu: float,
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None = None,
    opt_steps: int,
    lr: float,
    min_order: int,
    max_drop_size: int = 1,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    num_val_sequences: int | None = None,
    intensity_model: str = "multiplicative",
):
    if not active_rules:
        return float("inf"), float(mu), exc_params, inh_params, rule_heights, active_rules

    init_coef_map = build_init_coef_map(active_rules, exc_params, inh_params, fallback=0.1)
    # Exact refit for the current active set before nested pruning.
    best_bic, mu_fit, exc_fit, inh_fit, rule_heights, arrays_out = optimize_active_set_torch(
        active_rules=active_rules,
        subsets=subsets,
        basis_cache=basis_cache,
        rule_heights=rule_heights,
        init_mu=mu,
        init_coef_map=init_coef_map,
        grid_weights_train=grid_weights_train,
        grid_weights_val=grid_weights_val,
        device=device,
        torch_basis_cache=torch_basis_cache,
        steps=max(60, int(opt_steps)),
        lr=lr,
        penalize_kernel_df=bool(penalize_kernel_df),
        penalty_scale=float(penalty_scale),
        bic_num_sequences=num_val_sequences,
        intensity_model=str(intensity_model),
    )
    mu = float(mu_fit)
    exc_params = dict(exc_fit)
    inh_params = dict(inh_fit)
    active_rules = list(active_rules)
    train_event_len = int(next(iter(arrays_out.values()))[0].size)
    val_event_len = int(next(iter(arrays_out.values()))[2].size)

    improved = True
    while improved:
        improved = False
        removable = [j for j, ar in enumerate(active_rules) if len(tuple(subsets[int(ar.idx)])) >= int(min_order)]
        best_drop = None
        for drop_size in range(1, min(int(max_drop_size), len(removable)) + 1):
            for idxs in itertools.combinations(removable, drop_size):
                idx_set = set(int(j) for j in idxs)
                trial_active = [ar for j, ar in enumerate(active_rules) if j not in idx_set]
                if not trial_active:
                    trial_bic, trial_mu = fit_constant_bic(
                        n_events_val=val_event_len,
                        grid_weights_val=grid_weights_val,
                        n_events_train=train_event_len,
                        grid_weights_train=grid_weights_train,
                        n_val_sequences=int(num_val_sequences if num_val_sequences is not None else len(grid_weights_val)),
                    )
                    trial = (
                        float(trial_bic),
                        float(trial_mu),
                        {},
                        {},
                        rule_heights,
                        {},
                        trial_active,
                    )
                else:
                    init_coef_map = build_init_coef_map(trial_active, exc_params, inh_params, fallback=0.1)
                    trial_bic, trial_mu, trial_exc, trial_inh, trial_heights, trial_arrays = optimize_active_set_torch(
                        active_rules=trial_active,
                        subsets=subsets,
                        basis_cache=basis_cache,
                        rule_heights=rule_heights,
                        init_mu=mu,
                        init_coef_map=init_coef_map,
                        grid_weights_train=grid_weights_train,
                        grid_weights_val=grid_weights_val,
                        device=device,
                        torch_basis_cache=torch_basis_cache,
                        steps=max(25, int(opt_steps) // 4),
                        lr=lr,
                        penalize_kernel_df=bool(penalize_kernel_df),
                        penalty_scale=float(penalty_scale),
                        bic_num_sequences=num_val_sequences,
                        intensity_model=str(intensity_model),
                    )
                    trial = (
                        float(trial_bic),
                        float(trial_mu),
                        dict(trial_exc),
                        dict(trial_inh),
                        trial_heights,
                        trial_arrays,
                        trial_active,
                    )
                if best_drop is None or trial[0] < best_drop[0]:
                    best_drop = trial
        if best_drop is None:
            break
        trial_bic, trial_mu, trial_exc, trial_inh, trial_heights, trial_arrays, trial_active = best_drop
        if float(trial_bic) + 1e-8 < float(best_bic):
            best_bic = float(trial_bic)
            mu = float(trial_mu)
            exc_params = dict(trial_exc)
            inh_params = dict(trial_inh)
            rule_heights = trial_heights
            arrays_out = trial_arrays
            active_rules = trial_active
            improved = True
    return best_bic, mu, exc_params, inh_params, rule_heights, active_rules


def canonical_profile_backward_prune(
    *,
    active_rules: list[ActiveRule],
    subsets,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    kernels,
    mu: float,
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None = None,
    opt_steps: int,
    lr: float,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    num_val_sequences: int | None = None,
) -> tuple[float, float, dict[int, float], dict[int, float], dict[tuple[int, int], np.ndarray], list[ActiveRule]]:
    if not active_rules:
        return float("inf"), float(mu), exc_params, inh_params, rule_heights, active_rules

    cur_rules = list(active_rules)
    cur_mu = float(mu)
    cur_exc = dict(exc_params)
    cur_inh = dict(inh_params)
    cur_heights = rule_heights
    n_eff = bic_sample_size(int(num_val_sequences) if num_val_sequences is not None else int(grid_weights_val.size))
    log_n = math.log(float(n_eff))
    changed = False
    arrays_out = compute_active_rule_feature_arrays(
        active_rules=cur_rules,
        subsets=subsets,
        basis_cache=basis_cache,
        rule_heights=cur_heights,
        kernels=kernels,
    )

    while cur_rules:
        tr_event_len = int(next(iter(arrays_out.values()))[0].size)
        tr_grid_len = int(next(iter(arrays_out.values()))[1].size)
        total_exc_ev, total_inh_ev, total_exc_gr, total_inh_gr = contributions_from_active(
            active_rules=cur_rules,
            arrays_out=arrays_out,
            exc_params=cur_exc,
            inh_params=cur_inh,
            tr_event_len=tr_event_len,
            tr_grid_len=tr_grid_len,
        )

        best_drop: tuple[float, int] | None = None
        for j, ar in enumerate(cur_rules):
            idx = int(ar.idx)
            sign = str(ar.sign)
            coef = float(cur_exc.get(idx, 0.0) if sign == "exc" else cur_inh.get(idx, 0.0))
            feat_event = np.asarray(arrays_out[idx][0], dtype=np.float64)
            feat_grid = np.asarray(arrays_out[idx][1], dtype=np.float64)
            if sign == "exc":
                base_exc_gr = total_exc_gr - coef * feat_grid
                base_inh_gr = total_inh_gr
            else:
                base_exc_gr = total_exc_gr
                base_inh_gr = total_inh_gr - coef * feat_grid
            base_lp_grid = np.asarray(base_exc_gr - base_inh_gr, dtype=np.float64)
            penalty = 0.5 * float(penalty_scale) * log_n * (
                float(rule_param_dim(subsets[idx], basis_cache.knots.size)) if penalize_kernel_df else 1.0
            )
            gain, _ = profiled_gain_canonical_sign(
                feat_event=feat_event,
                feat_grid=feat_grid,
                base_lp_grid=base_lp_grid,
                grid_weights_train=grid_weights_train,
                sign=sign,
                penalty=penalty,
            )
            if best_drop is None or float(gain) < float(best_drop[0]):
                best_drop = (float(gain), int(j))

        if best_drop is None or float(best_drop[0]) > 1e-8:
            break

        drop_j = int(best_drop[1])
        drop_rule = cur_rules[drop_j]
        trial_rules = cur_rules[:drop_j] + cur_rules[drop_j + 1 :]
        if not trial_rules:
            break
        idx = int(drop_rule.idx)
        if str(drop_rule.sign) == "exc":
            cur_exc.pop(idx, None)
        else:
            cur_inh.pop(idx, None)
        cur_rules = list(trial_rules)
        changed = True

    if changed and cur_rules:
        init_coef_map = build_init_coef_map(cur_rules, cur_exc, cur_inh, fallback=0.1)
        final_bic, cur_mu, cur_exc, cur_inh, cur_heights, _arrays_out = optimize_active_set_torch(
            active_rules=cur_rules,
            subsets=subsets,
            basis_cache=basis_cache,
            rule_heights=cur_heights,
            init_mu=cur_mu,
            init_coef_map=init_coef_map,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            steps=max(40, int(opt_steps)),
            lr=lr,
            penalize_kernel_df=bool(penalize_kernel_df),
            penalty_scale=float(penalty_scale),
            bic_num_sequences=num_val_sequences,
            intensity_model="canonical_loglink",
        )
        return float(final_bic), float(cur_mu), dict(cur_exc), dict(cur_inh), cur_heights, cur_rules

    final_bic = validation_bic_from_arrays(
        active_rules=cur_rules,
        mu=float(cur_mu),
        exc_params=cur_exc,
        inh_params=cur_inh,
        arrays_out=arrays_out,
        grid_weights_val=grid_weights_val,
        subsets=subsets,
        num_knots=basis_cache.knots.size,
        penalize_kernel_df=bool(penalize_kernel_df),
        penalty_scale=float(penalty_scale),
        num_sequences_val=num_val_sequences,
        intensity_model="canonical_loglink",
    )
    return float(final_bic), float(cur_mu), dict(cur_exc), dict(cur_inh), cur_heights, cur_rules


def build_overlap_components(
    active_rules: list[ActiveRule],
    subsets,
    *,
    same_sign_only: bool = True,
) -> list[list[ActiveRule]]:
    if not active_rules:
        return []
    n = len(active_rules)
    adj = [[] for _ in range(n)]
    subset_sets = [set(int(s) for s in subsets[int(ar.idx)]) for ar in active_rules]
    for i in range(n):
        for j in range(i + 1, n):
            if bool(same_sign_only) and str(active_rules[i].sign) != str(active_rules[j].sign):
                continue
            if subset_sets[i].intersection(subset_sets[j]):
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * n
    comps: list[list[ActiveRule]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp_idx = []
        while stack:
            cur = stack.pop()
            comp_idx.append(cur)
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        comps.append([active_rules[k] for k in sorted(comp_idx)])
    return comps


def component_subset_search(
    *,
    active_rules: list[ActiveRule],
    subsets,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    mu: float,
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None = None,
    opt_steps: int,
    lr: float,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    num_val_sequences: int | None = None,
    same_sign_only: bool = True,
    intensity_model: str = "multiplicative",
):
    if not active_rules:
        return float("inf"), float(mu), exc_params, inh_params, rule_heights, active_rules

    cur_rules = list(active_rules)
    cur_mu = float(mu)
    cur_exc = dict(exc_params)
    cur_inh = dict(inh_params)
    cur_heights = rule_heights
    best_bic = float("inf")

    while True:
        init_coef_map = build_init_coef_map(cur_rules, cur_exc, cur_inh, fallback=0.1)
        best_bic, mu_fit, exc_fit, inh_fit, cur_heights, arrays_out = optimize_active_set_torch(
            active_rules=cur_rules,
            subsets=subsets,
            basis_cache=basis_cache,
            rule_heights=cur_heights,
            init_mu=cur_mu,
            init_coef_map=init_coef_map,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            steps=max(40, int(opt_steps)),
            lr=lr,
            penalize_kernel_df=bool(penalize_kernel_df),
            penalty_scale=float(penalty_scale),
            bic_num_sequences=num_val_sequences,
            intensity_model=str(intensity_model),
        )
        cur_mu = float(mu_fit)
        cur_exc = dict(exc_fit)
        cur_inh = dict(inh_fit)
        comps = build_overlap_components(cur_rules, subsets, same_sign_only=bool(same_sign_only))
        changed = False
        next_rules: list[ActiveRule] = []
        for comp in comps:
            if len(comp) <= 1:
                next_rules.extend(comp)
                continue
            outside = [ar for ar in cur_rules if ar not in comp]
            best_subset = list(comp)
            best_subset_score = float(best_bic)
            for mask in range(1 << len(comp)):
                subset_rules = [comp[j] for j in range(len(comp)) if (mask >> j) & 1]
                cand_rules = outside + subset_rules
                if not cand_rules:
                    cand_score, _ = fit_constant_bic(
                        n_events_val=int(next(iter(arrays_out.values()))[2].size),
                        grid_weights_val=grid_weights_val,
                        n_events_train=int(next(iter(arrays_out.values()))[0].size),
                        grid_weights_train=grid_weights_train,
                        n_val_sequences=int(num_val_sequences if num_val_sequences is not None else len(grid_weights_val)),
                    )
                elif len(cand_rules) == len(cur_rules) and all(a == b for a, b in zip(cand_rules, cur_rules)):
                    cand_score = float(best_bic)
                else:
                    cand_score = validation_bic_from_arrays(
                        active_rules=cand_rules,
                        arrays_out=arrays_out,
                        exc_params=cur_exc,
                        inh_params=cur_inh,
                        mu=cur_mu,
                        subsets=subsets,
                        num_knots=basis_cache.knots.size,
                        grid_weights_val=grid_weights_val,
                        penalize_kernel_df=bool(penalize_kernel_df),
                        penalty_scale=float(penalty_scale),
                        num_sequences_val=num_val_sequences,
                        intensity_model=str(intensity_model),
                    )
                if float(cand_score) + 1e-8 < float(best_subset_score):
                    best_subset_score = float(cand_score)
                    best_subset = subset_rules
            if len(best_subset) != len(comp) or any(a != b for a, b in zip(best_subset, comp)):
                changed = True
            next_rules.extend(best_subset)
        next_rules = sorted(next_rules, key=lambda ar: (int(ar.idx), str(ar.sign)))
        cur_rules = next_rules
        if not changed:
            return best_bic, cur_mu, cur_exc, cur_inh, cur_heights, cur_rules

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixed_target", type=int, required=True)
    ap.add_argument("--max_order", type=int, default=3)
    ap.add_argument("--max_lag", type=float, default=10.0)
    ap.add_argument("--kernel_num_bins", type=int, default=40)
    ap.add_argument("--kernel_num_knots", type=int, default=7)
    ap.add_argument("--grid_step", type=float, default=0.0)
    ap.add_argument("--max_rules", type=int, default=12)
    ap.add_argument("--opt_steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cpu_threads", type=int, default=0)
    ap.add_argument("--family_attribution_passes", type=int, default=1)
    ap.add_argument("--post_prune_min_order", type=int, default=1)
    ap.add_argument("--post_prune_max_drop_size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    configure_runtime_resources(None if int(args.cpu_threads) <= 0 else int(args.cpu_threads))
    configure_deterministic_research(seed=int(args.seed), deterministic=True)

    train, val, metadata = load_dataset(args.data)
    config = load_yaml(args.config)
    intensity_model = str(config.get("intensity_model", "multiplicative"))
    target = int(args.fixed_target)
    gt = gt_rules_from_config(config)
    num_types = int(metadata["num_types"])
    time_horizon = float(config.get("time_horizon", max(max(seq["time"]) for seq in train + val)))
    source_ids = tuple(sorted(k for k in range(num_types) if k != target))

    train_arrays = build_seq_event_arrays(train, num_types)
    val_arrays = build_seq_event_arrays(val, num_types)
    global_kernels = estimate_source_kernels(
        train_arrays,
        source_ids=source_ids,
        target=target,
        max_lag=float(args.max_lag),
        num_bins=int(args.kernel_num_bins),
        num_knots=int(args.kernel_num_knots),
        time_horizon=float(time_horizon),
    )
    grid_step = float(args.grid_step) if float(args.grid_step) > 0.0 else auto_grid_step(global_kernels)

    tr_event_seq, tr_event_times = collect_target_events(train, target=target)
    va_event_seq, va_event_times = collect_target_events(val, target=target)
    tr_grid_seq, tr_grid_times, tr_grid_w = build_midpoint_grid(train, time_horizon=float(time_horizon), step=grid_step)
    va_grid_seq, va_grid_times, va_grid_w = build_midpoint_grid(val, time_horizon=float(time_horizon), step=grid_step)

    a_train_event, a_train_grid, a_val_event, a_val_grid = build_global_activity(
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

    basis_cache = SourceBasisCache(
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
        max_lag=float(args.max_lag),
        num_bins=int(args.kernel_num_bins),
    )
    subsets = subset_list(source_ids, int(args.max_order))
    src_to_col_global = {int(s): j for j, s in enumerate(source_ids)}
    rule_heights = initialize_rule_specific_heights(
        subsets=subsets,
        source_ids=source_ids,
        global_kernels=global_kernels,
        global_activity_event=a_train_event,
        src_to_col_global=src_to_col_global,
        train_arrays=train_arrays,
        train_event_lag_bin_cache=train_event_lag_bin_cache,
        max_lag=float(args.max_lag),
        num_bins=int(args.kernel_num_bins),
        time_horizon=float(time_horizon),
    )
    init_arrays = compute_rule_feature_arrays(
        subsets=subsets,
        basis_cache=basis_cache,
        rule_heights=rule_heights,
        kernels=global_kernels,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch_basis_cache = TorchBasisCache(basis_cache, device)

    bic, mu, exc_params, inh_params, rule_heights, active_rules = run_active_set(
        subsets=subsets,
        init_arrays=init_arrays,
        rule_heights=rule_heights,
        basis_cache=basis_cache,
        grid_weights_train=np.asarray(tr_grid_w, dtype=np.float64),
        grid_weights_val=np.asarray(va_grid_w, dtype=np.float64),
        max_rules=int(args.max_rules),
        opt_steps=int(args.opt_steps),
        lr=float(args.lr),
        device=device,
        torch_basis_cache=torch_basis_cache,
        penalize_kernel_df=False,
        num_val_sequences=len(val),
        intensity_model=intensity_model,
    )
    if int(args.family_attribution_passes) > 0:
        bic, mu, exc_params, inh_params, rule_heights, active_rules = family_attribution_refine(
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
            opt_steps=int(args.opt_steps),
            lr=float(args.lr),
            passes=int(args.family_attribution_passes),
            num_val_sequences=len(val),
            intensity_model=intensity_model,
        )
    chosen_post_prune_scale = 1.0
    bic, mu, exc_params, inh_params, rule_heights, active_rules = post_prune_irreducible_rules(
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
        opt_steps=int(args.opt_steps),
        lr=float(args.lr),
        min_order=int(args.post_prune_min_order),
        max_drop_size=int(args.post_prune_max_drop_size),
        penalize_kernel_df=False,
        penalty_scale=1.0,
        num_val_sequences=len(val),
        intensity_model=intensity_model,
    )
    bic, mu, exc_params, inh_params, rule_heights, active_rules = component_subset_search(
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
        opt_steps=int(args.opt_steps),
        lr=float(args.lr),
        penalize_kernel_df=False,
        penalty_scale=1.0,
        num_val_sequences=len(val),
        same_sign_only=True,
        intensity_model=intensity_model,
    )

    preds = summarize_results(subsets=subsets, exc_params=exc_params, inh_params=inh_params, target=target)
    hit = sorted(gt & preds)
    miss = sorted(gt - preds)
    extra = sorted(preds - gt)
    print("===E2E RULE-DEPENDENT ACTIVE-SET REPORT===")
    print(
        "best_params:",
        {
            "target": target,
            "selected_rule_count": int(len(preds)),
            "train_target_events": int(len(tr_event_times)),
            "val_target_events": int(len(va_event_times)),
            "grid_step": float(grid_step),
            "grid_train": int(len(tr_grid_times)),
            "grid_val": int(len(va_grid_times)),
            "bic": float(bic),
            "mu": float(mu),
            "device": str(device),
            "intensity_model": intensity_model,
            "post_prune_penalty_scale": None if chosen_post_prune_scale is None else float(chosen_post_prune_scale),
        },
    )
    print_rule_block("True rules:", sorted(gt), target)
    print_rule_block("Predicted rules:", sorted(preds), target)
    print_rule_block("Matched rules:", hit, target)
    print_rule_block("Missing rules:", miss, target)
    print_rule_block("Extra predicted rules:", extra, target)
    print("Estimated rule-dependent kernels:")
    for idx in sorted(set(list(exc_params.keys()) + list(inh_params.keys()))):
        for src in subsets[int(idx)]:
            print(
                {
                    "subset": tuple(int(s) for s in subsets[int(idx)]),
                    "source": int(src),
                    "kernel_knots": [float(x) for x in global_kernels[int(src)].knots],
                    "kernel_heights": [float(x) for x in normalize_piecewise_area(global_kernels[int(src)].knots, rule_heights[(int(idx), int(src))])],
                }
            )


if __name__ == "__main__":
    main()

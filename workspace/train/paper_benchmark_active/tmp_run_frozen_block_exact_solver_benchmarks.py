from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency path
    threadpool_limits = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rule_dependent_kernel_active_set as rd
import tmp_inhibition_profile_block_validate as proto


PAPER_SUITE = REPO_ROOT / "data" / "paper_suite"
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


@dataclass
class ExcState:
    support: tuple[int, ...]
    mu: float
    params: dict[int, float]
    bic: float
    tr_ev: np.ndarray
    tr_gr: np.ndarray
    va_ev: np.ndarray
    va_gr: np.ndarray
    eta_ev_tr: np.ndarray


@dataclass
class InhState:
    support: tuple[int, ...]
    params: dict[int, float]
    bic: float
    tr_ev: np.ndarray
    tr_gr: np.ndarray
    va_ev: np.ndarray
    va_gr: np.ndarray
    residual_tr: np.ndarray


def penalty_per_rule_from_arrays(arrays_all) -> float:
    sample = next(iter(arrays_all.values()))
    n_eff = max(int(np.asarray(sample[2]).size + np.asarray(sample[3]).size), 2)
    return 0.5 * math.log(float(n_eff))


def local_data_path(config_path: Path) -> Path:
    return config_path.with_suffix(".exact_block.local.pkl")


def format_rules(rules, target: int) -> list[str]:
    return [rd.format_rule(r, int(target)) for r in sorted(rules)]


def build_block_columns(arrays_all, support: tuple[int, ...], part_ev: int, part_gr: int):
    if not support:
        sample = next(iter(arrays_all.values()))
        n_ev = int(np.asarray(sample[part_ev]).size)
        n_gr = int(np.asarray(sample[part_gr]).size)
        return np.zeros((n_ev, 0), dtype=np.float64), np.zeros((n_gr, 0), dtype=np.float64)
    mat_ev = np.column_stack([np.asarray(arrays_all[int(idx)][part_ev], dtype=np.float64) for idx in support]).astype(np.float64, copy=False)
    mat_gr = np.column_stack([np.asarray(arrays_all[int(idx)][part_gr], dtype=np.float64) for idx in support]).astype(np.float64, copy=False)
    return mat_ev, mat_gr


def sum_block_contrib(arrays_all, params: dict[int, float], part_ev: int, part_gr: int, length_ev: int, length_gr: int):
    out_ev = np.zeros((length_ev,), dtype=np.float64)
    out_gr = np.zeros((length_gr,), dtype=np.float64)
    for idx, coef in params.items():
        arr = arrays_all[int(idx)]
        out_ev += float(coef) * np.asarray(arr[part_ev], dtype=np.float64)
        out_gr += float(coef) * np.asarray(arr[part_gr], dtype=np.float64)
    return out_ev, out_gr


def validation_bic(ll_val: float, total_rules: int, n_eff: int) -> float:
    return float(-2.0 * float(ll_val) + float(1 + int(total_rules)) * math.log(float(max(n_eff, 2))))


def blas_single_thread_context():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


def evaluate_support_batch(tasks, cache, solve_uncached, workers: int):
    if not tasks:
        return []

    ordered_supports: list[tuple[int, ...]] = []
    missing: dict[tuple[int, ...], dict[int, float] | None] = {}
    for support, warm in tasks:
        key = tuple(sorted(int(x) for x in support))
        ordered_supports.append(key)
        if key not in cache and key not in missing:
            missing[key] = warm

    if missing:
        missing_items = list(missing.items())
        if int(workers) <= 1 or len(missing_items) <= 1:
            for support, warm in missing_items:
                cache[support] = solve_uncached(support, warm)
        else:
            max_workers = min(int(workers), len(missing_items))
            with blas_single_thread_context():
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    future_map = {
                        ex.submit(solve_uncached, support, warm): support
                        for support, warm in missing_items
                    }
                    for fut in as_completed(future_map):
                        cache[future_map[fut]] = fut.result()

    return [cache[key] for key in ordered_supports]


def eligible_candidate_ids(arrays_all, support_set: set[int], forbidden: set[int]) -> list[int]:
    return [int(idx) for idx in arrays_all.keys() if int(idx) not in support_set and int(idx) not in forbidden]


def filter_candidate_ids(candidate_ids: list[int], allowed_ids: set[int] | None) -> list[int]:
    if allowed_ids is None:
        return [int(idx) for idx in candidate_ids]
    allow = set(int(idx) for idx in allowed_ids)
    return [int(idx) for idx in candidate_ids if int(idx) in allow]


def subset_sources(subsets, idx: int) -> tuple[int, ...]:
    return tuple(int(s) for s in subsets[int(idx)])


def motif_related_sources(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
    lhs_set = set(int(x) for x in lhs)
    rhs_set = set(int(x) for x in rhs)
    if not lhs_set or not rhs_set:
        return False
    inter = len(lhs_set & rhs_set)
    if inter <= 0:
        return False
    if lhs_set < rhs_set or rhs_set < lhs_set:
        return True
    if len(lhs_set) == len(rhs_set) and inter >= max(1, len(lhs_set) - 1):
        return True
    return False


def inactive_center_scores(
    *,
    subsets,
    active_support: tuple[int, ...],
    forbidden: set[int],
) -> list[tuple[int, int, int, int]]:
    active_source_map = {int(idx): subset_sources(subsets, int(idx)) for idx in active_support}
    active_set = set(int(idx) for idx in active_support)
    scored = []
    for idx, subset in enumerate(subsets):
        idx_i = int(idx)
        sources = tuple(int(s) for s in subset)
        order = len(sources)
        if order not in (1, 2, 3) or idx_i in active_set or idx_i in forbidden:
            continue
        contain_count = 0
        one_edit_count = 0
        overlap_mass = 0
        src_set = set(sources)
        for active_idx, active_sources in active_source_map.items():
            active_set_src = set(active_sources)
            inter = len(src_set & active_set_src)
            if src_set < active_set_src:
                contain_count += 1
                overlap_mass += inter
            elif len(src_set) == len(active_set_src) and inter >= max(1, len(src_set) - 1):
                one_edit_count += 1
                overlap_mass += inter
        if contain_count <= 0 and one_edit_count <= 0:
            continue
        scored.append((int(contain_count), int(one_edit_count), int(overlap_mass), idx_i))
    scored.sort(key=lambda item: (item[0], item[1], item[2], -item[3]), reverse=True)
    return scored


def build_inh_center_block(
    *,
    subsets,
    current_support: tuple[int, ...],
    forbidden: set[int],
    center_top_k: int,
    center_min_support: int,
):
    if int(center_top_k) <= 0:
        return set(), {}
    scored = inactive_center_scores(
        subsets=subsets,
        active_support=current_support,
        forbidden=forbidden,
    )
    center_ids: list[int] = []
    center_to_active: dict[int, set[int]] = {}
    for contain_count, one_edit_count, _overlap_mass, idx in scored:
        support_count = int(contain_count + one_edit_count)
        if support_count < int(center_min_support):
            continue
        center_ids.append(int(idx))
        if len(center_ids) >= int(center_top_k):
            break
    center_set = set(int(idx) for idx in center_ids)
    for idx in center_ids:
        center_sources = subset_sources(subsets, int(idx))
        touched = {
            int(active_idx)
            for active_idx in current_support
            if motif_related_sources(center_sources, subset_sources(subsets, int(active_idx)))
        }
        if touched:
            center_to_active[int(idx)] = touched
    drop_to_centers: dict[int, set[int]] = {}
    for center_idx, active_ids in center_to_active.items():
        for active_idx in active_ids:
            drop_to_centers.setdefault(int(active_idx), set()).add(int(center_idx))
    return center_set, drop_to_centers


def chunked(iterable: list[int], size: int):
    for start in range(0, len(iterable), int(size)):
        yield iterable[start : start + int(size)]


def fit_exc_support(
    *,
    support: tuple[int, ...],
    arrays_all,
    inh_params: dict[int, float],
    gw_tr: np.ndarray,
    gw_va: np.ndarray,
    init_mu: float,
    init_params: dict[int, float] | None,
    total_inh_rules: int,
):
    n_ev_tr = int(next(iter(arrays_all.values()))[0].size)
    n_gr_tr = int(next(iter(arrays_all.values()))[1].size)
    n_ev_va = int(next(iter(arrays_all.values()))[2].size)
    n_gr_va = int(next(iter(arrays_all.values()))[3].size)
    inh_tr_ev, inh_tr_gr = sum_block_contrib(arrays_all, inh_params, 0, 1, n_ev_tr, n_gr_tr)
    inh_va_ev, inh_va_gr = sum_block_contrib(arrays_all, inh_params, 2, 3, n_ev_va, n_gr_va)
    c_tr = np.exp(-np.clip(inh_tr_gr, 0.0, 60.0))
    c_va = np.exp(-np.clip(inh_va_gr, 0.0, 60.0))
    x_tr_ev, x_tr_gr = build_block_columns(arrays_all, support, 0, 1)
    x_va_ev, x_va_gr = build_block_columns(arrays_all, support, 2, 3)

    def unpack(z: np.ndarray):
        mu = float(z[0])
        alpha = np.asarray(z[1:], dtype=np.float64)
        return mu, alpha

    def objective(z: np.ndarray):
        mu, alpha = unpack(z)
        exc_ev = x_tr_ev @ alpha if alpha.size else np.zeros((n_ev_tr,), dtype=np.float64)
        exc_gr = x_tr_gr @ alpha if alpha.size else np.zeros((n_gr_tr,), dtype=np.float64)
        eta_ev = np.clip(mu + exc_ev, 1e-8, None)
        eta_gr = np.clip(mu + exc_gr, 1e-8, None)
        val = float(-np.sum(np.log(eta_ev)) + np.sum((gw_tr * c_tr) * eta_gr))
        grad_mu = float(-np.sum(1.0 / eta_ev) + np.sum(gw_tr * c_tr))
        if alpha.size:
            grad_alpha = -(x_tr_ev / eta_ev[:, None]).sum(axis=0) + x_tr_gr.T @ (gw_tr * c_tr)
            grad = np.concatenate([[grad_mu], np.asarray(grad_alpha, dtype=np.float64)])
        else:
            grad = np.asarray([grad_mu], dtype=np.float64)
        return val, grad

    z0 = np.zeros((1 + len(support),), dtype=np.float64)
    z0[0] = max(float(init_mu), 1e-6)
    if init_params:
        for j, idx in enumerate(support):
            z0[1 + j] = max(float(init_params.get(int(idx), 0.1)), 0.0)
    else:
        z0[1:] = 0.1
    bounds = [(1e-8, None)] * (1 + len(support))
    res = minimize(
        fun=lambda z: objective(z)[0],
        x0=z0,
        jac=lambda z: objective(z)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7},
    )
    mu_fit, alpha_fit = unpack(np.maximum(np.asarray(res.x, dtype=np.float64), 0.0))
    tr_ev = np.asarray(x_tr_ev @ alpha_fit if alpha_fit.size else np.zeros((n_ev_tr,), dtype=np.float64), dtype=np.float64)
    tr_gr = np.asarray(x_tr_gr @ alpha_fit if alpha_fit.size else np.zeros((n_gr_tr,), dtype=np.float64), dtype=np.float64)
    va_ev = np.asarray(x_va_ev @ alpha_fit if alpha_fit.size else np.zeros((n_ev_va,), dtype=np.float64), dtype=np.float64)
    va_gr = np.asarray(x_va_gr @ alpha_fit if alpha_fit.size else np.zeros((n_gr_va,), dtype=np.float64), dtype=np.float64)
    eta_ev_va = np.clip(mu_fit + va_ev, 1e-8, None)
    eta_gr_va = np.clip(mu_fit + va_gr, 1e-8, None)
    ll_val = float(np.sum(np.log(eta_ev_va) - inh_va_ev) - np.dot(gw_va, eta_gr_va * c_va))
    bic = validation_bic(ll_val, total_rules=int(total_inh_rules + len(support)), n_eff=int(n_ev_va + n_gr_va))
    return ExcState(
        support=support,
        mu=float(mu_fit),
        params={int(idx): float(alpha_fit[j]) for j, idx in enumerate(support)},
        bic=float(bic),
        tr_ev=tr_ev,
        tr_gr=tr_gr,
        va_ev=va_ev,
        va_gr=va_gr,
        eta_ev_tr=np.clip(mu_fit + tr_ev, 1e-8, None),
    )


def fit_inh_support(
    *,
    support: tuple[int, ...],
    arrays_all,
    mu: float,
    exc_params: dict[int, float],
    gw_tr: np.ndarray,
    gw_va: np.ndarray,
    init_params: dict[int, float] | None,
    total_exc_rules: int,
):
    n_ev_tr = int(next(iter(arrays_all.values()))[0].size)
    n_gr_tr = int(next(iter(arrays_all.values()))[1].size)
    n_ev_va = int(next(iter(arrays_all.values()))[2].size)
    n_gr_va = int(next(iter(arrays_all.values()))[3].size)
    exc_tr_ev, exc_tr_gr = sum_block_contrib(arrays_all, exc_params, 0, 1, n_ev_tr, n_gr_tr)
    exc_va_ev, exc_va_gr = sum_block_contrib(arrays_all, exc_params, 2, 3, n_ev_va, n_gr_va)
    eta_gr_tr = np.clip(float(mu) + exc_tr_gr, 1e-8, None)
    eta_ev_va = np.clip(float(mu) + exc_va_ev, 1e-8, None)
    eta_gr_va = np.clip(float(mu) + exc_va_gr, 1e-8, None)
    x_tr_ev, x_tr_gr = build_block_columns(arrays_all, support, 0, 1)
    x_va_ev, x_va_gr = build_block_columns(arrays_all, support, 2, 3)
    event_vec = np.asarray(x_tr_ev.sum(axis=0) if x_tr_ev.size else np.zeros((0,), dtype=np.float64), dtype=np.float64)

    if not support:
        tr_ev = np.zeros((n_ev_tr,), dtype=np.float64)
        tr_gr = np.zeros((n_gr_tr,), dtype=np.float64)
        va_ev = np.zeros((n_ev_va,), dtype=np.float64)
        va_gr = np.zeros((n_gr_va,), dtype=np.float64)
        ll_val = float(np.sum(np.log(eta_ev_va)) - np.dot(gw_va, eta_gr_va))
        bic = validation_bic(ll_val, total_rules=int(total_exc_rules), n_eff=int(n_ev_va + n_gr_va))
        residual_tr = np.asarray(gw_tr * eta_gr_tr, dtype=np.float64)
        return InhState(
            support=(),
            params={},
            bic=float(bic),
            tr_ev=tr_ev,
            tr_gr=tr_gr,
            va_ev=va_ev,
            va_gr=va_gr,
            residual_tr=residual_tr,
        )

    def objective(beta: np.ndarray):
        xb = x_tr_gr @ beta if beta.size else 0.0
        exp_term = np.exp(-np.clip(xb, 0.0, 60.0))
        val = float(np.dot(event_vec, beta) + np.sum(gw_tr * eta_gr_tr * exp_term))
        grad = event_vec - x_tr_gr.T @ (gw_tr * eta_gr_tr * exp_term) if beta.size else np.zeros((0,), dtype=np.float64)
        return val, grad

    beta0 = np.asarray([max(float(init_params.get(int(idx), 0.1)), 0.0) for idx in support], dtype=np.float64) if init_params else np.full((len(support),), 0.1, dtype=np.float64)
    res = minimize(
        fun=lambda z: objective(z)[0],
        x0=beta0,
        jac=lambda z: objective(z)[1],
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(support),
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7},
    )
    beta_fit = np.maximum(np.asarray(res.x, dtype=np.float64), 0.0)
    tr_ev = np.asarray(x_tr_ev @ beta_fit if beta_fit.size else np.zeros((n_ev_tr,), dtype=np.float64), dtype=np.float64)
    tr_gr = np.asarray(x_tr_gr @ beta_fit if beta_fit.size else np.zeros((n_gr_tr,), dtype=np.float64), dtype=np.float64)
    va_ev = np.asarray(x_va_ev @ beta_fit if beta_fit.size else np.zeros((n_ev_va,), dtype=np.float64), dtype=np.float64)
    va_gr = np.asarray(x_va_gr @ beta_fit if beta_fit.size else np.zeros((n_gr_va,), dtype=np.float64), dtype=np.float64)
    ll_val = float(np.sum(np.log(eta_ev_va) - va_ev) - np.dot(gw_va, eta_gr_va * np.exp(-np.clip(va_gr, 0.0, 60.0))))
    bic = validation_bic(ll_val, total_rules=int(total_exc_rules + len(support)), n_eff=int(n_ev_va + n_gr_va))
    residual_tr = gw_tr * eta_gr_tr * np.exp(-np.clip(tr_gr, 0.0, 60.0))
    return InhState(
        support=support,
        params={int(idx): float(beta_fit[j]) for j, idx in enumerate(support)},
        bic=float(bic),
        tr_ev=tr_ev,
        tr_gr=tr_gr,
        va_ev=va_ev,
        va_gr=va_gr,
        residual_tr=residual_tr,
    )


def positive_exc_adds(state: ExcState, arrays_all, inh_params, gw_tr: np.ndarray, forbidden: set[int]) -> list[int]:
    n_ev_tr = int(next(iter(arrays_all.values()))[0].size)
    n_gr_tr = int(next(iter(arrays_all.values()))[1].size)
    inh_tr_ev, inh_tr_gr = sum_block_contrib(arrays_all, inh_params, 0, 1, n_ev_tr, n_gr_tr)
    del inh_tr_ev
    c_tr = np.exp(-np.clip(inh_tr_gr, 0.0, 60.0))
    support_set = set(int(x) for x in state.support)
    eligible = eligible_candidate_ids(arrays_all, support_set, forbidden)
    out = []
    inv_eta = np.asarray(1.0 / state.eta_ev_tr, dtype=np.float64)
    grid_weight = np.asarray(gw_tr * c_tr, dtype=np.float64)
    for ids in chunked(eligible, 256):
        ev_mat = np.column_stack([np.asarray(arrays_all[int(idx)][0], dtype=np.float64) for idx in ids]).astype(np.float64, copy=False)
        gr_mat = np.column_stack([np.asarray(arrays_all[int(idx)][1], dtype=np.float64) for idx in ids]).astype(np.float64, copy=False)
        grad0 = -(ev_mat.T @ inv_eta) + (gr_mat.T @ grid_weight)
        out.extend(int(idx) for idx, g in zip(ids, grad0.tolist()) if float(g) < -1e-10)
    return out


def positive_inh_adds(
    state: InhState,
    arrays_all,
    forbidden: set[int],
    *,
    allowed_ids: set[int] | None = None,
    residual_override: np.ndarray | None = None,
) -> list[int]:
    support_set = set(int(x) for x in state.support)
    eligible = filter_candidate_ids(eligible_candidate_ids(arrays_all, support_set, forbidden), allowed_ids)
    out = []
    residual = np.asarray(state.residual_tr if residual_override is None else residual_override, dtype=np.float64)
    for ids in chunked(eligible, 256):
        ev_mat = np.column_stack([np.asarray(arrays_all[int(idx)][0], dtype=np.float64) for idx in ids]).astype(np.float64, copy=False)
        gr_mat = np.column_stack([np.asarray(arrays_all[int(idx)][1], dtype=np.float64) for idx in ids]).astype(np.float64, copy=False)
        g0 = (gr_mat.T @ residual) - ev_mat.sum(axis=0)
        out.extend(int(idx) for idx, g in zip(ids, g0.tolist()) if float(g) > 1e-10)
    return out


def screened_inh_adds(
    state: InhState,
    arrays_all,
    forbidden: set[int],
    penalty_per_rule: float,
    screen_workers: int,
    *,
    allowed_ids: set[int] | None = None,
    residual_override: np.ndarray | None = None,
) -> list[tuple[float, int, float]]:
    residual = np.asarray(state.residual_tr if residual_override is None else residual_override, dtype=np.float64)
    derivative_ok = positive_inh_adds(
        state,
        arrays_all,
        forbidden,
        allowed_ids=allowed_ids,
        residual_override=residual,
    )

    def score_one(idx: int):
        feat_ev = np.asarray(arrays_all[int(idx)][0], dtype=np.float64)
        feat_gr = np.asarray(arrays_all[int(idx)][1], dtype=np.float64)
        gain, beta1 = proto.scalar_gain(
            int(idx),
            event_sum_map={int(idx): float(np.sum(feat_ev))},
            grid_col_map={int(idx): feat_gr},
            residual=residual,
            penalty_per_rule=float(penalty_per_rule),
        )
        return float(gain), int(idx), float(beta1)

    out = []
    if int(screen_workers) <= 1 or len(derivative_ok) <= 1:
        for idx in derivative_ok:
                gain, idx, beta1 = score_one(int(idx))
                if float(gain) > 1e-10:
                    out.append((float(gain), int(idx), float(beta1)))
    else:
        with blas_single_thread_context():
            with ThreadPoolExecutor(max_workers=min(int(screen_workers), len(derivative_ok))) as ex:
                future_map = {ex.submit(score_one, int(idx)): int(idx) for idx in derivative_ok}
                for fut in as_completed(future_map):
                    gain, idx, beta1 = fut.result()
                    if float(gain) > 1e-10:
                        out.append((float(gain), int(idx), float(beta1)))
    out.sort(reverse=True)
    return out


def safe_inh_swap_superset(
    state: InhState,
    arrays_all,
    forbidden: set[int],
    penalty_per_rule: float,
    screen_workers: int,
) -> list[tuple[float, int, float]]:
    if not state.support:
        upper_residual = np.asarray(state.residual_tr, dtype=np.float64)
    else:
        boost_cols = []
        for idx in state.support:
            coef = float(state.params.get(int(idx), 0.0))
            if coef <= 0.0:
                continue
            boost_cols.append(coef * np.asarray(arrays_all[int(idx)][1], dtype=np.float64))
        if boost_cols:
            max_boost = np.maximum.reduce(boost_cols)
            upper_residual = np.asarray(state.residual_tr, dtype=np.float64) * np.exp(np.clip(max_boost, 0.0, 60.0))
        else:
            upper_residual = np.asarray(state.residual_tr, dtype=np.float64)
    return screened_inh_adds(
        state,
        arrays_all,
        forbidden,
        penalty_per_rule,
        int(screen_workers),
        residual_override=upper_residual,
    )


def exact_scalar_exc_gain(
    *,
    feat_ev: np.ndarray,
    feat_gr: np.ndarray,
    eta_ev: np.ndarray,
    grid_weight: np.ndarray,
    penalty_per_rule: float,
) -> tuple[float, float]:
    feat_ev = np.asarray(feat_ev, dtype=np.float64)
    feat_gr = np.asarray(feat_gr, dtype=np.float64)
    eta_ev = np.asarray(eta_ev, dtype=np.float64)
    grid_weight = np.asarray(grid_weight, dtype=np.float64)
    if feat_ev.size == 0:
        return float(-penalty_per_rule), 0.0
    grid_cost = float(np.dot(grid_weight, feat_gr))

    def deriv(beta: float) -> float:
        return float(grid_cost - np.sum(feat_ev / np.clip(eta_ev + float(beta) * feat_ev, 1e-8, None)))

    g0 = deriv(0.0)
    if g0 >= -1e-10:
        return float(-penalty_per_rule), 0.0

    hi = 1.0
    for _ in range(64):
        if deriv(hi) >= 0.0:
            break
        hi *= 2.0
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if deriv(mid) >= 0.0:
            hi = mid
        else:
            lo = mid
    beta = float(hi)
    delta = float(-np.sum(np.log(np.clip(eta_ev + beta * feat_ev, 1e-8, None) / np.clip(eta_ev, 1e-8, None))) + beta * grid_cost)
    gain = float(-(delta + float(penalty_per_rule)))
    return float(gain), float(beta)


def screened_exc_adds(
    state: ExcState,
    arrays_all,
    inh_params: dict[int, float],
    gw_tr: np.ndarray,
    forbidden: set[int],
    penalty_per_rule: float,
    screen_workers: int,
    *,
    allowed_ids: set[int] | None = None,
    eta_ev_override: np.ndarray | None = None,
) -> list[tuple[float, int, float]]:
    n_ev_tr = int(next(iter(arrays_all.values()))[0].size)
    n_gr_tr = int(next(iter(arrays_all.values()))[1].size)
    _inh_tr_ev, inh_tr_gr = sum_block_contrib(arrays_all, inh_params, 0, 1, n_ev_tr, n_gr_tr)
    del _inh_tr_ev
    grid_weight = np.asarray(gw_tr * np.exp(-np.clip(inh_tr_gr, 0.0, 60.0)), dtype=np.float64)
    eta_ev = np.asarray(state.eta_ev_tr if eta_ev_override is None else eta_ev_override, dtype=np.float64)
    support_set = set(int(x) for x in state.support)
    eligible = filter_candidate_ids(eligible_candidate_ids(arrays_all, support_set, forbidden), allowed_ids)

    def score_one(idx: int):
        feat_ev = np.asarray(arrays_all[int(idx)][0], dtype=np.float64)
        feat_gr = np.asarray(arrays_all[int(idx)][1], dtype=np.float64)
        gain, alpha1 = exact_scalar_exc_gain(
            feat_ev=feat_ev,
            feat_gr=feat_gr,
            eta_ev=eta_ev,
            grid_weight=grid_weight,
            penalty_per_rule=float(penalty_per_rule),
        )
        return float(gain), int(idx), float(alpha1)

    out: list[tuple[float, int, float]] = []
    if int(screen_workers) <= 1 or len(eligible) <= 1:
        for idx in eligible:
            gain, idx, alpha1 = score_one(int(idx))
            if float(gain) > 1e-10:
                out.append((float(gain), int(idx), float(alpha1)))
    else:
        with blas_single_thread_context():
            with ThreadPoolExecutor(max_workers=min(int(screen_workers), len(eligible))) as ex:
                future_map = {ex.submit(score_one, int(idx)): int(idx) for idx in eligible}
                for fut in as_completed(future_map):
                    gain, idx, alpha1 = fut.result()
                    if float(gain) > 1e-10:
                        out.append((float(gain), int(idx), float(alpha1)))
    out.sort(reverse=True)
    return out


def safe_exc_swap_superset(
    state: ExcState,
    arrays_all,
    inh_params: dict[int, float],
    gw_tr: np.ndarray,
    forbidden: set[int],
    penalty_per_rule: float,
    screen_workers: int,
) -> list[tuple[float, int, float]]:
    if not state.support:
        eta_lower = np.asarray(state.eta_ev_tr, dtype=np.float64)
    else:
        drop_cols = []
        for idx in state.support:
            coef = float(state.params.get(int(idx), 0.0))
            if coef <= 0.0:
                continue
            drop_cols.append(coef * np.asarray(arrays_all[int(idx)][0], dtype=np.float64))
        if drop_cols:
            max_drop = np.maximum.reduce(drop_cols)
            eta_lower = np.clip(np.asarray(state.eta_ev_tr, dtype=np.float64) - max_drop, 1e-8, None)
        else:
            eta_lower = np.asarray(state.eta_ev_tr, dtype=np.float64)
    return screened_exc_adds(
        state,
        arrays_all,
        inh_params,
        gw_tr,
        forbidden,
        penalty_per_rule,
        int(screen_workers),
        eta_ev_override=eta_lower,
    )


def active_inh_interaction_block(
    current: InhState,
    arrays_all,
    inactive_ids: set[int],
    *,
    tol: float = 1e-12,
) -> set[int]:
    if not current.support or not inactive_ids:
        return set()
    residual = np.asarray(current.residual_tr, dtype=np.float64)
    active_ids: set[int] = set()
    inactive_sorted = sorted(int(idx) for idx in inactive_ids)
    for active_idx in current.support:
        active_gr = np.asarray(arrays_all[int(active_idx)][1], dtype=np.float64)
        active_weighted = residual * active_gr
        for ids in chunked(inactive_sorted, 256):
            gr_mat = np.column_stack([np.asarray(arrays_all[int(idx)][1], dtype=np.float64) for idx in ids]).astype(np.float64, copy=False)
            if np.max(active_weighted @ gr_mat) > float(tol):
                active_ids.add(int(active_idx))
                break
    return active_ids


def active_exc_interaction_block(
    current: ExcState,
    arrays_all,
    inactive_ids: set[int],
    *,
    tol: float = 1e-12,
) -> set[int]:
    if not current.support or not inactive_ids:
        return set()
    eta = np.clip(np.asarray(current.eta_ev_tr, dtype=np.float64), 1e-8, None)
    weight = 1.0 / np.square(eta)
    active_ids: set[int] = set()
    inactive_sorted = sorted(int(idx) for idx in inactive_ids)
    for active_idx in current.support:
        active_ev = np.asarray(arrays_all[int(active_idx)][0], dtype=np.float64)
        active_weighted = weight * active_ev
        for ids in chunked(inactive_sorted, 256):
            ev_mat = np.column_stack([np.asarray(arrays_all[int(idx)][0], dtype=np.float64) for idx in ids]).astype(np.float64, copy=False)
            if np.max(active_weighted @ ev_mat) > float(tol):
                active_ids.add(int(active_idx))
                break
    return active_ids


def validation_loss_vector_from_parts(
    *,
    mu: float,
    exc_va_ev: np.ndarray,
    exc_va_gr: np.ndarray,
    inh_va_ev: np.ndarray,
    inh_va_gr: np.ndarray,
    gw_va: np.ndarray,
) -> np.ndarray:
    eta_ev = np.clip(np.asarray(float(mu) + np.asarray(exc_va_ev, dtype=np.float64), dtype=np.float64), 1e-8, None)
    eta_gr = np.clip(np.asarray(float(mu) + np.asarray(exc_va_gr, dtype=np.float64), dtype=np.float64), 1e-8, None)
    inh_ev = np.asarray(inh_va_ev, dtype=np.float64)
    inh_gr = np.asarray(inh_va_gr, dtype=np.float64)
    gw_va = np.asarray(gw_va, dtype=np.float64)
    event_losses = -np.log(eta_ev) + inh_ev
    grid_losses = gw_va * eta_gr * np.exp(-np.clip(inh_gr, 0.0, 60.0))
    return np.concatenate([event_losses, grid_losses]).astype(np.float64, copy=False)


def validation_loss_vector(exc_state: ExcState, inh_state: InhState, gw_va: np.ndarray) -> np.ndarray:
    return validation_loss_vector_from_parts(
        mu=float(exc_state.mu),
        exc_va_ev=np.asarray(exc_state.va_ev, dtype=np.float64),
        exc_va_gr=np.asarray(exc_state.va_gr, dtype=np.float64),
        inh_va_ev=np.asarray(inh_state.va_ev, dtype=np.float64),
        inh_va_gr=np.asarray(inh_state.va_gr, dtype=np.float64),
        gw_va=np.asarray(gw_va, dtype=np.float64),
    )


def empirical_bernstein_radius(diff: np.ndarray, delta: float) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    if diff.size <= 1:
        return float("inf")
    n = float(diff.size)
    var_hat = float(np.var(diff, ddof=1))
    width = float(np.max(diff) - np.min(diff))
    delta = float(min(max(delta, 1e-12), 1.0 - 1e-12))
    log_term = math.log(3.0 / delta)
    return float(math.sqrt(2.0 * var_hat * log_term / n) + 3.0 * width * log_term / max(n - 1.0, 1.0))


def validation_split_indices(arrays_all):
    sample = next(iter(arrays_all.values()))
    n_ev_va = int(np.asarray(sample[2]).size)
    n_gr_va = int(np.asarray(sample[3]).size)
    ev_idx = np.arange(n_ev_va, dtype=np.int64)
    gr_idx = np.arange(n_gr_va, dtype=np.int64)
    ev_a = ev_idx[::2]
    ev_b = ev_idx[1::2]
    gr_a = gr_idx[::2]
    gr_b = gr_idx[1::2]
    if ev_b.size == 0:
        ev_b = ev_a.copy()
    if gr_b.size == 0:
        gr_b = gr_a.copy()
    idx_a = np.concatenate([ev_a, n_ev_va + gr_a]).astype(np.int64, copy=False)
    idx_b = np.concatenate([ev_b, n_ev_va + gr_b]).astype(np.int64, copy=False)
    return idx_a, idx_b


def frozen_drop_validation_loss_vector(
    *,
    sign: str,
    drop_idx: int,
    exc_state: ExcState,
    inh_state: InhState,
    arrays_all,
    gw_va: np.ndarray,
) -> np.ndarray:
    rule_va_ev = np.asarray(arrays_all[int(drop_idx)][2], dtype=np.float64)
    rule_va_gr = np.asarray(arrays_all[int(drop_idx)][3], dtype=np.float64)
    if str(sign) == "exc":
        coef = float(exc_state.params.get(int(drop_idx), 0.0))
        exc_va_ev = np.asarray(exc_state.va_ev, dtype=np.float64) - coef * rule_va_ev
        exc_va_gr = np.asarray(exc_state.va_gr, dtype=np.float64) - coef * rule_va_gr
        inh_va_ev = np.asarray(inh_state.va_ev, dtype=np.float64)
        inh_va_gr = np.asarray(inh_state.va_gr, dtype=np.float64)
    else:
        coef = float(inh_state.params.get(int(drop_idx), 0.0))
        exc_va_ev = np.asarray(exc_state.va_ev, dtype=np.float64)
        exc_va_gr = np.asarray(exc_state.va_gr, dtype=np.float64)
        inh_va_ev = np.asarray(inh_state.va_ev, dtype=np.float64) - coef * rule_va_ev
        inh_va_gr = np.asarray(inh_state.va_gr, dtype=np.float64) - coef * rule_va_gr
    return validation_loss_vector_from_parts(
        mu=float(exc_state.mu),
        exc_va_ev=exc_va_ev,
        exc_va_gr=exc_va_gr,
        inh_va_ev=inh_va_ev,
        inh_va_gr=inh_va_gr,
        gw_va=np.asarray(gw_va, dtype=np.float64),
    )


def refit_fixed_support_pair(
    *,
    exc_support: tuple[int, ...],
    inh_support: tuple[int, ...],
    arrays_all,
    gw_tr: np.ndarray,
    gw_va: np.ndarray,
    init_mu: float,
    init_exc_params: dict[int, float],
    init_inh_params: dict[int, float],
    cycles: int = 2,
):
    exc_state = fit_exc_support(
        support=tuple(exc_support),
        arrays_all=arrays_all,
        inh_params=dict(init_inh_params),
        gw_tr=gw_tr,
        gw_va=gw_va,
        init_mu=float(init_mu),
        init_params=dict(init_exc_params),
        total_inh_rules=int(len(inh_support)),
    )
    inh_state = fit_inh_support(
        support=tuple(inh_support),
        arrays_all=arrays_all,
        mu=float(exc_state.mu),
        exc_params=exc_state.params,
        gw_tr=gw_tr,
        gw_va=gw_va,
        init_params=dict(init_inh_params),
        total_exc_rules=int(len(exc_support)),
    )
    for _ in range(max(int(cycles), 1)):
        exc_state = fit_exc_support(
            support=tuple(exc_support),
            arrays_all=arrays_all,
            inh_params=inh_state.params,
            gw_tr=gw_tr,
            gw_va=gw_va,
            init_mu=float(exc_state.mu),
            init_params=exc_state.params,
            total_inh_rules=int(len(inh_support)),
        )
        inh_state = fit_inh_support(
            support=tuple(inh_support),
            arrays_all=arrays_all,
            mu=float(exc_state.mu),
            exc_params=exc_state.params,
            gw_tr=gw_tr,
            gw_va=gw_va,
            init_params=inh_state.params,
            total_exc_rules=int(len(exc_support)),
        )
    exc_state = fit_exc_support(
        support=tuple(exc_support),
        arrays_all=arrays_all,
        inh_params=inh_state.params,
        gw_tr=gw_tr,
        gw_va=gw_va,
        init_mu=float(exc_state.mu),
        init_params=exc_state.params,
        total_inh_rules=int(len(inh_support)),
    )
    return exc_state, inh_state


def confidence_calibrated_exact_prune(
    *,
    exc_state: ExcState,
    inh_state: InhState,
    arrays_all,
    gw_tr: np.ndarray,
    gw_va: np.ndarray,
    delta: float,
    screen_top_k: int = 1,
    epsilon: float = 0.0,
):
    current_loss = validation_loss_vector(exc_state, inh_state, gw_va)
    idx_a, idx_b = validation_split_indices(arrays_all)
    current_loss_a = np.asarray(current_loss[idx_a], dtype=np.float64)
    current_loss_b = np.asarray(current_loss[idx_b], dtype=np.float64)
    tests = int(len(exc_state.support) + len(inh_state.support))
    if tests <= 0:
        return exc_state, inh_state, []

    screen_delta = float(delta) / float(max(tests, 1))
    screen_candidates = []
    for drop_idx in exc_state.support:
        cand_loss = frozen_drop_validation_loss_vector(
            sign="exc",
            drop_idx=int(drop_idx),
            exc_state=exc_state,
            inh_state=inh_state,
            arrays_all=arrays_all,
            gw_va=gw_va,
        )
        diff_a = np.asarray(cand_loss[idx_a] - current_loss_a, dtype=np.float64)
        mean_a = float(np.mean(diff_a))
        radius_a = empirical_bernstein_radius(diff_a, screen_delta)
        screen_candidates.append(
            {
                "sign": "exc",
                "drop_idx": int(drop_idx),
                "screen_mean_diff": float(mean_a),
                "screen_radius": float(radius_a),
                "screen_ucb": float(mean_a + radius_a),
            }
        )
    for drop_idx in inh_state.support:
        cand_loss = frozen_drop_validation_loss_vector(
            sign="inh",
            drop_idx=int(drop_idx),
            exc_state=exc_state,
            inh_state=inh_state,
            arrays_all=arrays_all,
            gw_va=gw_va,
        )
        diff_a = np.asarray(cand_loss[idx_a] - current_loss_a, dtype=np.float64)
        mean_a = float(np.mean(diff_a))
        radius_a = empirical_bernstein_radius(diff_a, screen_delta)
        screen_candidates.append(
            {
                "sign": "inh",
                "drop_idx": int(drop_idx),
                "screen_mean_diff": float(mean_a),
                "screen_radius": float(radius_a),
                "screen_ucb": float(mean_a + radius_a),
            }
        )

    screen_candidates = sorted(
        screen_candidates,
        key=lambda item: (float(item["screen_ucb"]), float(item["screen_mean_diff"]), str(item["sign"]), int(item["drop_idx"])),
    )
    selected = screen_candidates[: max(0, int(screen_top_k))]
    if not selected:
        return exc_state, inh_state, [{
            "stage": "screen",
            "tested_rules": int(tests),
            "screen_delta": float(screen_delta),
            "selected_top_k": 0,
            "candidates": [],
        }]

    n_eff_a = int(max(len(idx_a), 2))
    n_eff_b = int(max(len(idx_b), 2))
    penalty_total_a = float(0.5 * math.log(float(n_eff_a)))
    family = []
    k = int(len(selected))
    for mask in range(1, 1 << k):
        chosen = [selected[j] for j in range(k) if (mask >> j) & 1]
        exc_drop = {int(c["drop_idx"]) for c in chosen if str(c["sign"]) == "exc"}
        inh_drop = {int(c["drop_idx"]) for c in chosen if str(c["sign"]) == "inh"}
        exc_support2 = tuple(int(x) for x in exc_state.support if int(x) not in exc_drop)
        inh_support2 = tuple(int(x) for x in inh_state.support if int(x) not in inh_drop)
        init_exc = {int(k0): float(v0) for k0, v0 in exc_state.params.items() if int(k0) not in exc_drop}
        init_inh = {int(k0): float(v0) for k0, v0 in inh_state.params.items() if int(k0) not in inh_drop}
        family.append(
            {
                "drop_rules": [{"sign": str(c["sign"]), "drop_idx": int(c["drop_idx"])} for c in chosen],
                "drop_count": int(len(chosen)),
                "exc_support": exc_support2,
                "inh_support": inh_support2,
                "init_exc": init_exc,
                "init_inh": init_inh,
            }
        )

    exact_delta = float(delta) / float(max(len(family), 1))
    exact_candidates = []
    for fam in family:
        cand_exc, cand_inh = refit_fixed_support_pair(
            exc_support=tuple(fam["exc_support"]),
            inh_support=tuple(fam["inh_support"]),
            arrays_all=arrays_all,
            gw_tr=gw_tr,
            gw_va=gw_va,
            init_mu=float(exc_state.mu),
            init_exc_params=dict(fam["init_exc"]),
            init_inh_params=dict(fam["init_inh"]),
        )
        cand_loss = validation_loss_vector(cand_exc, cand_inh, gw_va)
        diff_a = np.asarray(cand_loss[idx_a] - current_loss_a, dtype=np.float64)
        diff_b = np.asarray(cand_loss[idx_b] - current_loss_b, dtype=np.float64)
        mean_a = float(np.mean(diff_a))
        mean_b = float(np.mean(diff_b))
        radius_b = empirical_bernstein_radius(diff_b, exact_delta)
        lcb_total_b = float(float(n_eff_b) * (mean_b - radius_b))
        total_diff_a = float(float(n_eff_a) * mean_a)
        select_score = float(total_diff_a - float(fam["drop_count"]) * penalty_total_a)
        exact_candidates.append(
            {
                **fam,
                "exact_mean_diff_a": float(mean_a),
                "exact_total_diff_a": float(total_diff_a),
                "exact_mean_diff": float(mean_b),
                "exact_radius": float(radius_b),
                "exact_lcb_total_b": float(lcb_total_b),
                "select_score": float(select_score),
                "in_confidence_set": bool(lcb_total_b <= float(epsilon)),
                "cand_exc": cand_exc,
                "cand_inh": cand_inh,
            }
        )

    prune_logs = [
        {
            "stage": "screen",
            "tested_rules": int(tests),
            "screen_delta": float(screen_delta),
            "selected_top_k": int(len(selected)),
            "candidates": [
                {
                    "sign": str(cand["sign"]),
                    "drop_idx": int(cand["drop_idx"]),
                    "screen_mean_diff": float(cand["screen_mean_diff"]),
                    "screen_radius": float(cand["screen_radius"]),
                    "screen_ucb": float(cand["screen_ucb"]),
                }
                for cand in screen_candidates
            ],
        },
        {
            "stage": "exact",
            "tested_rules": int(len(family)),
            "exact_delta": float(exact_delta),
            "epsilon": float(epsilon),
            "selection_penalty_total_a": float(penalty_total_a),
            "confidence_count": int(sum(1 for cand in exact_candidates if bool(cand["in_confidence_set"]))),
            "candidates": [
                {
                    "drop_rules": list(cand["drop_rules"]),
                    "drop_count": int(cand["drop_count"]),
                    "exact_mean_diff_a": float(cand["exact_mean_diff_a"]),
                    "exact_total_diff_a": float(cand["exact_total_diff_a"]),
                    "select_score": float(cand["select_score"]),
                    "exact_mean_diff": float(cand["exact_mean_diff"]),
                    "exact_radius": float(cand["exact_radius"]),
                    "exact_lcb_total_b": float(cand["exact_lcb_total_b"]),
                    "in_confidence_set": bool(cand["in_confidence_set"]),
                }
                for cand in sorted(exact_candidates, key=lambda item: (float(item["select_score"]), -int(item["drop_count"]), float(item["exact_lcb_total_b"])))
            ],
        },
    ]
    confident = [cand for cand in exact_candidates if bool(cand["in_confidence_set"])]
    improving = [cand for cand in confident if float(cand["select_score"]) < 0.0]
    if not improving:
        return exc_state, inh_state, prune_logs
    chosen = min(
        improving,
        key=lambda cand: (float(cand["select_score"]), -int(cand["drop_count"]), float(cand["exact_lcb_total_b"]), float(cand["cand_exc"].bic), float(cand["cand_inh"].bic)),
    )
    prune_logs[-1]["chosen_model"] = {
        "drop_rules": list(chosen["drop_rules"]),
        "drop_count": int(chosen["drop_count"]),
        "exact_mean_diff_a": float(chosen["exact_mean_diff_a"]),
        "exact_total_diff_a": float(chosen["exact_total_diff_a"]),
        "select_score": float(chosen["select_score"]),
        "exact_mean_diff": float(chosen["exact_mean_diff"]),
        "exact_radius": float(chosen["exact_radius"]),
        "exact_lcb_total_b": float(chosen["exact_lcb_total_b"]),
    }
    return chosen["cand_exc"], chosen["cand_inh"], prune_logs


def best_exc_neighbor(
    *,
    current: ExcState,
    arrays_all,
    inh_params: dict[int, float],
    gw_tr: np.ndarray,
    gw_va: np.ndarray,
    forbidden: set[int],
    support_workers: int,
    safe_swap_superset: bool,
    active_interaction_block: bool,
    scalar_warm_reuse: bool,
):
    cache: dict[tuple[int, ...], ExcState] = {tuple(current.support): current}

    def solve_uncached(support: tuple[int, ...], warm: dict[int, float] | None):
        return fit_exc_support(
            support=support,
            arrays_all=arrays_all,
            inh_params=inh_params,
            gw_tr=gw_tr,
            gw_va=gw_va,
            init_mu=float(current.mu),
            init_params=warm,
            total_inh_rules=int(len(inh_params)),
        )

    best = current
    penalty_per_rule = penalty_per_rule_from_arrays(arrays_all)
    safe_superset_scored: list[tuple[float, int, float]] = []
    safe_superset_ids: set[int] | None = None
    if bool(safe_swap_superset):
        safe_superset_scored = safe_exc_swap_superset(
            current,
            arrays_all,
            inh_params,
            gw_tr,
            forbidden,
            penalty_per_rule,
            int(support_workers),
        )
        safe_superset_ids = {int(idx) for _gain, idx, _alpha1 in safe_superset_scored}
    add_candidates_scored = screened_exc_adds(
        current,
        arrays_all,
        inh_params,
        gw_tr,
        forbidden,
        penalty_per_rule,
        int(support_workers),
        allowed_ids=safe_superset_ids,
    )
    add_tasks = []
    for _gain, idx, alpha1 in add_candidates_scored:
        support = tuple(sorted(current.support + (int(idx),)))
        warm = dict(current.params)
        warm[int(idx)] = float(alpha1 if bool(scalar_warm_reuse) else 0.1)
        add_tasks.append((support, warm))
    add_states = evaluate_support_batch(add_tasks, cache, solve_uncached, int(support_workers))
    add_evals = 0
    drop_evals = 0
    swap_evals = 0
    for cand in add_states:
        add_evals += 1
        if cand.bic + 1e-8 < best.bic:
            best = cand

    drop_tasks = []
    drop_ids = []
    for drop_idx in current.support:
        support = tuple(int(x) for x in current.support if int(x) != int(drop_idx))
        warm = {int(k): float(v) for k, v in current.params.items() if int(k) != int(drop_idx)}
        drop_ids.append(int(drop_idx))
        drop_tasks.append((support, warm))
    dropped_states = evaluate_support_batch(drop_tasks, cache, solve_uncached, int(support_workers))

    swap_tasks = []
    interaction_drop_ids = set(int(x) for x in current.support)
    if bool(active_interaction_block) and safe_superset_ids:
        interaction_drop_ids = active_exc_interaction_block(current, arrays_all, safe_superset_ids)
        if not interaction_drop_ids:
            interaction_drop_ids = set(int(x) for x in current.support)
    for drop_idx, dropped in zip(drop_ids, dropped_states):
        drop_evals += 1
        if dropped.bic + 1e-8 < best.bic:
            best = dropped
        if int(drop_idx) not in interaction_drop_ids:
            continue
        swap_candidates_scored = screened_exc_adds(
            dropped,
            arrays_all,
            inh_params,
            gw_tr,
            forbidden,
            penalty_per_rule,
            int(support_workers),
            allowed_ids=safe_superset_ids,
        )
        for _gain, idx, alpha1 in swap_candidates_scored:
            if int(idx) == int(drop_idx):
                continue
            support2 = tuple(sorted(dropped.support + (int(idx),)))
            if support2 == tuple(current.support):
                continue
            warm2 = dict(dropped.params)
            warm2[int(idx)] = float(alpha1 if bool(scalar_warm_reuse) else 0.1)
            swap_tasks.append((support2, warm2))
    swap_states = evaluate_support_batch(swap_tasks, cache, solve_uncached, int(support_workers))
    for cand in swap_states:
        swap_evals += 1
        if cand.bic + 1e-8 < best.bic:
            best = cand
    return best, {
        "safe_superset_candidates": int(len(safe_superset_ids or set())),
        "interaction_drop_candidates": int(len(interaction_drop_ids)),
        "add_candidates": int(len(add_candidates_scored)),
        "add_evals": int(add_evals),
        "drop_evals": int(drop_evals),
        "swap_evals": int(swap_evals),
        "support_evals": int(len(cache)),
    }


def best_inh_neighbor(
    *,
    current: InhState,
    subsets,
    arrays_all,
    mu: float,
    exc_params: dict[int, float],
    gw_tr: np.ndarray,
    gw_va: np.ndarray,
    forbidden: set[int],
    support_workers: int,
    block_center_top_k: int,
    block_center_min_support: int,
    block_swap_only: bool,
    safe_swap_superset: bool,
    active_interaction_block: bool,
    scalar_warm_reuse: bool,
):
    cache: dict[tuple[int, ...], InhState] = {tuple(current.support): current}

    def solve_uncached(support: tuple[int, ...], warm: dict[int, float] | None):
        return fit_inh_support(
            support=support,
            arrays_all=arrays_all,
            mu=float(mu),
            exc_params=exc_params,
            gw_tr=gw_tr,
            gw_va=gw_va,
            init_params=warm,
            total_exc_rules=int(len(exc_params)),
        )

    best = current
    penalty_per_rule = penalty_per_rule_from_arrays(arrays_all)
    safe_superset_scored: list[tuple[float, int]] = []
    safe_superset_ids: set[int] | None = None
    if bool(safe_swap_superset):
        safe_superset_scored = safe_inh_swap_superset(
            current,
            arrays_all,
            forbidden,
            penalty_per_rule,
            int(support_workers),
        )
        safe_superset_ids = {int(idx) for _gain, idx, _beta1 in safe_superset_scored}
    block_center_set, drop_to_centers = build_inh_center_block(
        subsets=subsets,
        current_support=tuple(current.support),
        forbidden=forbidden,
        center_top_k=int(block_center_top_k),
        center_min_support=int(block_center_min_support),
    )
    add_candidates_scored = screened_inh_adds(
        current,
        arrays_all,
        forbidden,
        penalty_per_rule,
        int(support_workers),
        allowed_ids=safe_superset_ids,
    )
    if int(block_center_top_k) > 0 and not block_center_set:
        add_candidates_scored = []
    elif block_center_set:
        add_candidates_scored = [(float(gain), int(idx), float(beta1)) for gain, idx, beta1 in add_candidates_scored if int(idx) in block_center_set]
    add_candidates = [int(idx) for _gain, idx, _beta1 in add_candidates_scored]
    add_tasks = []
    for _gain, idx, beta1 in add_candidates_scored:
        support = tuple(sorted(current.support + (int(idx),)))
        warm = dict(current.params)
        warm[int(idx)] = float(beta1 if bool(scalar_warm_reuse) else 0.1)
        add_tasks.append((support, warm))
    add_states = evaluate_support_batch(add_tasks, cache, solve_uncached, int(support_workers))
    add_evals = 0
    drop_evals = 0
    swap_evals = 0
    for cand in add_states:
        add_evals += 1
        if cand.bic + 1e-8 < best.bic:
            best = cand

    drop_tasks = []
    drop_ids = []
    candidate_drop_ids = list(current.support)
    interaction_drop_ids = set(int(x) for x in current.support)
    if bool(active_interaction_block) and safe_superset_ids:
        interaction_drop_ids = active_inh_interaction_block(current, arrays_all, safe_superset_ids)
        if not interaction_drop_ids:
            interaction_drop_ids = set(int(x) for x in current.support)
        candidate_drop_ids = [int(drop_idx) for drop_idx in candidate_drop_ids if int(drop_idx) in interaction_drop_ids]
    if int(block_center_top_k) > 0 and not block_center_set:
        candidate_drop_ids = []
    elif block_center_set:
        candidate_drop_ids = [int(drop_idx) for drop_idx in current.support if int(drop_idx) in drop_to_centers]
    for drop_idx in candidate_drop_ids:
        support = tuple(int(x) for x in current.support if int(x) != int(drop_idx))
        warm = {int(k): float(v) for k, v in current.params.items() if int(k) != int(drop_idx)}
        drop_ids.append(int(drop_idx))
        drop_tasks.append((support, warm))
    dropped_states = evaluate_support_batch(drop_tasks, cache, solve_uncached, int(support_workers))

    swap_tasks = []
    for drop_idx, dropped in zip(drop_ids, dropped_states):
        drop_evals += 1
        if not bool(block_swap_only) and dropped.bic + 1e-8 < best.bic:
            best = dropped
        swap_candidates_scored = screened_inh_adds(
            dropped,
            arrays_all,
            forbidden,
            penalty_per_rule,
            int(support_workers),
            allowed_ids=safe_superset_ids,
        )
        if block_center_set:
            allowed_centers = drop_to_centers.get(int(drop_idx), set())
            if not allowed_centers:
                continue
            swap_candidates_scored = [(float(gain), int(idx), float(beta1)) for gain, idx, beta1 in swap_candidates_scored if int(idx) in allowed_centers]
        for _gain, idx, beta1 in swap_candidates_scored:
            if int(idx) == int(drop_idx):
                continue
            support2 = tuple(sorted(dropped.support + (int(idx),)))
            if support2 == tuple(current.support):
                continue
            warm2 = dict(dropped.params)
            warm2[int(idx)] = float(beta1 if bool(scalar_warm_reuse) else 0.1)
            swap_tasks.append((support2, warm2))
    swap_states = evaluate_support_batch(swap_tasks, cache, solve_uncached, int(support_workers))
    for cand in swap_states:
        swap_evals += 1
        if cand.bic + 1e-8 < best.bic:
            best = cand
    return best, {
        "safe_superset_candidates": int(len(safe_superset_ids or set())),
        "interaction_drop_candidates": int(len(interaction_drop_ids)),
        "block_centers": int(len(block_center_set)),
        "block_drop_candidates": int(len(drop_ids)),
        "block_universe_rules": int(len(set(int(idx) for idx in block_center_set) | set(int(idx) for idx in drop_to_centers))),
        "add_candidates": int(len(add_candidates)),
        "add_evals": int(add_evals),
        "drop_evals": int(drop_evals),
        "swap_evals": int(swap_evals),
        "support_evals": int(len(cache)),
    }


def final_prediction(problem, exc_params: dict[int, float], inh_params: dict[int, float]) -> list[str]:
    return sorted(
        rd.summarize_results(
            subsets=problem["subsets"],
            exc_params=exc_params,
            inh_params=inh_params,
            target=int(problem["target"]),
        )
    )


def run_exact_block_solver(
    problem,
    *,
    opt_steps: int,
    lr: float,
    max_rounds: int,
    device: torch.device,
    support_workers: int,
    inh_block_center_top_k: int,
    inh_block_center_min_support: int,
    inh_block_swap_only: bool,
    inh_safe_swap_superset: bool,
    exc_safe_swap_superset: bool,
    active_interaction_block: bool,
    scalar_warm_reuse: bool,
    confidence_prune: bool,
    confidence_delta: float,
    confidence_screen_top_k: int,
    confidence_epsilon: float,
):
    anchor = proto.run_anchor(
        problem,
        opt_steps=int(opt_steps),
        lr=float(lr),
        device=device,
        use_postprune_anchor=False,
    )
    arrays_all = anchor["arrays_all"]
    gw_tr = np.asarray(problem["grid_weights_train"], dtype=np.float64)
    gw_va = np.asarray(problem["grid_weights_val"], dtype=np.float64)
    exc_support = tuple(sorted(int(k) for k in anchor["exc_params"]))
    inh_support = tuple(sorted(int(k) for k in anchor["inh_params"]))

    exc_state = fit_exc_support(
        support=exc_support,
        arrays_all=arrays_all,
        inh_params=anchor["inh_params"],
        gw_tr=gw_tr,
        gw_va=gw_va,
        init_mu=float(anchor["mu"]),
        init_params=anchor["exc_params"],
        total_inh_rules=int(len(inh_support)),
    )
    inh_state = fit_inh_support(
        support=inh_support,
        arrays_all=arrays_all,
        mu=float(exc_state.mu),
        exc_params=exc_state.params,
        gw_tr=gw_tr,
        gw_va=gw_va,
        init_params=anchor["inh_params"],
        total_exc_rules=int(len(exc_support)),
    )
    exc_state = fit_exc_support(
        support=exc_state.support,
        arrays_all=arrays_all,
        inh_params=inh_state.params,
        gw_tr=gw_tr,
        gw_va=gw_va,
        init_mu=float(exc_state.mu),
        init_params=exc_state.params,
        total_inh_rules=int(len(inh_state.support)),
    )

    logs = []
    for round_idx in range(int(max_rounds)):
        changed = False
        best_exc, exc_stats = best_exc_neighbor(
            current=exc_state,
            arrays_all=arrays_all,
            inh_params=inh_state.params,
            gw_tr=gw_tr,
            gw_va=gw_va,
            forbidden=set(int(x) for x in inh_state.support),
            support_workers=int(support_workers),
            safe_swap_superset=bool(exc_safe_swap_superset),
            active_interaction_block=bool(active_interaction_block),
            scalar_warm_reuse=bool(scalar_warm_reuse),
        )
        if best_exc.bic + 1e-8 < exc_state.bic:
            exc_state = best_exc
            changed = True

        best_inh, inh_stats = best_inh_neighbor(
            current=inh_state,
            subsets=problem["subsets"],
            arrays_all=arrays_all,
            mu=float(exc_state.mu),
            exc_params=exc_state.params,
            gw_tr=gw_tr,
            gw_va=gw_va,
            forbidden=set(int(x) for x in exc_state.support),
            support_workers=int(support_workers),
            block_center_top_k=int(inh_block_center_top_k),
            block_center_min_support=int(inh_block_center_min_support),
            block_swap_only=bool(inh_block_swap_only),
            safe_swap_superset=bool(inh_safe_swap_superset),
            active_interaction_block=bool(active_interaction_block),
            scalar_warm_reuse=bool(scalar_warm_reuse),
        )
        if best_inh.bic + 1e-8 < inh_state.bic:
            inh_state = best_inh
            changed = True

        exc_state = fit_exc_support(
            support=exc_state.support,
            arrays_all=arrays_all,
            inh_params=inh_state.params,
            gw_tr=gw_tr,
            gw_va=gw_va,
            init_mu=float(exc_state.mu),
            init_params=exc_state.params,
            total_inh_rules=int(len(inh_state.support)),
        )
        logs.append(
            {
                "round": int(round_idx + 1),
                "exc_bic": float(exc_state.bic),
                "inh_bic": float(inh_state.bic),
                "exc_rules": int(len(exc_state.support)),
                "inh_rules": int(len(inh_state.support)),
                "exc_safe_superset_candidates": int(exc_stats["safe_superset_candidates"]),
                "exc_interaction_drop_candidates": int(exc_stats["interaction_drop_candidates"]),
                "exc_add_candidates": int(exc_stats["add_candidates"]),
                "inh_safe_superset_candidates": int(inh_stats["safe_superset_candidates"]),
                "inh_interaction_drop_candidates": int(inh_stats["interaction_drop_candidates"]),
                "inh_add_candidates": int(inh_stats["add_candidates"]),
                "inh_block_centers": int(inh_stats["block_centers"]),
                "inh_block_drop_candidates": int(inh_stats["block_drop_candidates"]),
                "inh_block_universe_rules": int(inh_stats["block_universe_rules"]),
                "exc_add_evals": int(exc_stats["add_evals"]),
                "inh_add_evals": int(inh_stats["add_evals"]),
                "exc_drop_evals": int(exc_stats["drop_evals"]),
                "inh_drop_evals": int(inh_stats["drop_evals"]),
                "exc_swap_evals": int(exc_stats["swap_evals"]),
                "inh_swap_evals": int(inh_stats["swap_evals"]),
                "exc_support_evals": int(exc_stats["support_evals"]),
                "inh_support_evals": int(inh_stats["support_evals"]),
                "changed": bool(changed),
            }
        )
        if not changed:
            break

    prune_logs = []
    if bool(confidence_prune):
        exc_state, inh_state, prune_logs = confidence_calibrated_exact_prune(
            exc_state=exc_state,
            inh_state=inh_state,
            arrays_all=arrays_all,
            gw_tr=gw_tr,
            gw_va=gw_va,
            delta=float(confidence_delta),
            screen_top_k=int(confidence_screen_top_k),
            epsilon=float(confidence_epsilon),
        )

    preds = final_prediction(problem, exc_state.params, inh_state.params)
    return {
        "anchor_bic": float(anchor["bic"]),
        "exact_bic": float(exc_state.bic),
        "anchor_exc_rules": int(len(anchor["exc_params"])),
        "anchor_inh_rules": int(len(anchor["inh_params"])),
        "exc_params": exc_state.params,
        "inh_params": inh_state.params,
        "preds": preds,
        "logs": logs,
        "prune_logs": prune_logs,
    }


def evaluate_benchmark(
    *,
    name: str,
    config_path: Path,
    opt_steps: int,
    lr: float,
    max_rounds: int,
    cleanup_data: bool,
    device: torch.device,
    support_workers: int,
    inh_block_center_top_k: int,
    inh_block_center_min_support: int,
    inh_block_swap_only: bool,
    inh_safe_swap_superset: bool,
    exc_safe_swap_superset: bool,
    active_interaction_block: bool,
    scalar_warm_reuse: bool,
    confidence_prune: bool,
    confidence_delta: float,
    confidence_screen_top_k: int,
    confidence_epsilon: float,
):
    data_path = local_data_path(config_path)
    started = time.time()
    proto.ensure_local_dataset(config_path, data_path)
    try:
        problem = proto.setup_problem(data_path, config_path)
        solved = run_exact_block_solver(
            problem,
            opt_steps=int(opt_steps),
            lr=float(lr),
            max_rounds=int(max_rounds),
            device=device,
            support_workers=int(support_workers),
            inh_block_center_top_k=int(inh_block_center_top_k),
            inh_block_center_min_support=int(inh_block_center_min_support),
            inh_block_swap_only=bool(inh_block_swap_only),
            inh_safe_swap_superset=bool(inh_safe_swap_superset),
            exc_safe_swap_superset=bool(exc_safe_swap_superset),
            active_interaction_block=bool(active_interaction_block),
            scalar_warm_reuse=bool(scalar_warm_reuse),
            confidence_prune=bool(confidence_prune),
            confidence_delta=float(confidence_delta),
            confidence_screen_top_k=int(confidence_screen_top_k),
            confidence_epsilon=float(confidence_epsilon),
        )
        target = int(problem["target"])
        gt = sorted(problem["gt"])
        pred_set = set(solved["preds"])
        gt_set = set(gt)
        matched = sorted(gt_set & pred_set)
        missing = sorted(gt_set - pred_set)
        extra = sorted(pred_set - gt_set)
        return {
            "name": str(name),
            "config_path": str(config_path),
            "elapsed_sec": float(time.time() - started),
            "anchor_bic": float(solved["anchor_bic"]),
            "bic": float(solved["exact_bic"]),
            "inh_block_center_top_k": int(inh_block_center_top_k),
            "inh_block_center_min_support": int(inh_block_center_min_support),
            "inh_block_swap_only": bool(inh_block_swap_only),
            "inh_safe_swap_superset": bool(inh_safe_swap_superset),
            "exc_safe_swap_superset": bool(exc_safe_swap_superset),
            "active_interaction_block": bool(active_interaction_block),
            "scalar_warm_reuse": bool(scalar_warm_reuse),
            "confidence_prune": bool(confidence_prune),
            "confidence_delta": float(confidence_delta),
            "confidence_screen_top_k": int(confidence_screen_top_k),
            "confidence_epsilon": float(confidence_epsilon),
            "true_rule": format_rules(gt, target),
            "matched_rule": format_rules(matched, target),
            "missing_rule": format_rules(missing, target),
            "extra_rule": format_rules(extra, target),
            "recall": float(len(matched)) / max(len(gt_set), 1),
            "precision": float(len(matched)) / max(len(pred_set), 1),
            "round_logs": solved["logs"],
            "prune_logs": solved.get("prune_logs", []),
        }
    finally:
        if bool(cleanup_data):
            try:
                data_path.unlink(missing_ok=True)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="all")
    ap.add_argument("--opt_steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max_rounds", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--support_workers", type=int, default=1)
    ap.add_argument("--inh_block_center_top_k", type=int, default=0)
    ap.add_argument("--inh_block_center_min_support", type=int, default=2)
    ap.add_argument("--inh_block_swap_only", action="store_true")
    ap.add_argument("--inh_safe_swap_superset", action="store_true")
    ap.add_argument("--exc_safe_swap_superset", action="store_true")
    ap.add_argument("--active_interaction_block", action="store_true")
    ap.add_argument("--scalar_warm_reuse", action="store_true")
    ap.add_argument("--confidence_prune", action="store_true")
    ap.add_argument("--confidence_delta", type=float, default=0.05)
    ap.add_argument("--confidence_screen_top_k", type=int, default=1)
    ap.add_argument("--confidence_epsilon", type=float, default=0.0)
    ap.add_argument("--keep_data", action="store_true")
    ap.add_argument("--json_out", default="data/paper_suite/tmp_frozen_block_exact_solver_results.json")
    args = ap.parse_args()

    thread_cap = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(max(thread_cap, 1))
    torch.set_num_interop_threads(1)
    if str(args.device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    wanted = None if str(args.benchmarks).strip().lower() == "all" else {x.strip() for x in str(args.benchmarks).split(",") if x.strip()}
    items = [item for item in BENCHMARKS if wanted is None or item["name"] in wanted]
    if not items:
        raise ValueError("no benchmarks selected")

    results = {}
    out_path = REPO_ROOT / str(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for item in items:
        name = str(item["name"])
        print(f"RUNNING {name}", flush=True)
        try:
            result = evaluate_benchmark(
                name=name,
                config_path=Path(item["config"]),
                opt_steps=int(args.opt_steps),
                lr=float(args.lr),
                max_rounds=int(args.max_rounds),
                cleanup_data=not bool(args.keep_data),
                device=device,
                support_workers=int(args.support_workers),
                inh_block_center_top_k=int(args.inh_block_center_top_k),
                inh_block_center_min_support=int(args.inh_block_center_min_support),
                inh_block_swap_only=bool(args.inh_block_swap_only),
                inh_safe_swap_superset=bool(args.inh_safe_swap_superset),
                exc_safe_swap_superset=bool(args.exc_safe_swap_superset),
                active_interaction_block=bool(args.active_interaction_block),
                scalar_warm_reuse=bool(args.scalar_warm_reuse),
                confidence_prune=bool(args.confidence_prune),
                confidence_delta=float(args.confidence_delta),
                confidence_screen_top_k=int(args.confidence_screen_top_k),
                confidence_epsilon=float(args.confidence_epsilon),
            )
        except Exception as exc:
            result = {
                "name": name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        results[name] = result
        out_path.write_text(json.dumps(results, indent=2))
        print(json.dumps({name: result}, indent=2), flush=True)
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()

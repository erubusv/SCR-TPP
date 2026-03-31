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


def positive_inh_adds(state: InhState, arrays_all, forbidden: set[int]) -> list[int]:
    support_set = set(int(x) for x in state.support)
    eligible = eligible_candidate_ids(arrays_all, support_set, forbidden)
    out = []
    residual = np.asarray(state.residual_tr, dtype=np.float64)
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
) -> list[tuple[float, int]]:
    derivative_ok = positive_inh_adds(state, arrays_all, forbidden)

    def score_one(idx: int):
        feat_ev = np.asarray(arrays_all[int(idx)][0], dtype=np.float64)
        feat_gr = np.asarray(arrays_all[int(idx)][1], dtype=np.float64)
        gain, _beta1 = proto.scalar_gain(
            int(idx),
            event_sum_map={int(idx): float(np.sum(feat_ev))},
            grid_col_map={int(idx): feat_gr},
            residual=state.residual_tr,
            penalty_per_rule=float(penalty_per_rule),
        )
        return float(gain), int(idx)

    out = []
    if int(screen_workers) <= 1 or len(derivative_ok) <= 1:
        for idx in derivative_ok:
            gain, idx = score_one(int(idx))
            if float(gain) > 1e-10:
                out.append((float(gain), int(idx)))
    else:
        with blas_single_thread_context():
            with ThreadPoolExecutor(max_workers=min(int(screen_workers), len(derivative_ok))) as ex:
                future_map = {ex.submit(score_one, int(idx)): int(idx) for idx in derivative_ok}
                for fut in as_completed(future_map):
                    gain, idx = fut.result()
                    if float(gain) > 1e-10:
                        out.append((float(gain), int(idx)))
    out.sort(reverse=True)
    return out


def best_exc_neighbor(
    *,
    current: ExcState,
    arrays_all,
    inh_params: dict[int, float],
    gw_tr: np.ndarray,
    gw_va: np.ndarray,
    forbidden: set[int],
    support_workers: int,
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
    add_candidates = sorted(
        positive_exc_adds(current, arrays_all, inh_params, gw_tr, forbidden),
        key=int,
    )
    add_tasks = []
    for idx in add_candidates:
        support = tuple(sorted(current.support + (int(idx),)))
        warm = dict(current.params)
        warm[int(idx)] = 0.1
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
    for drop_idx, dropped in zip(drop_ids, dropped_states):
        drop_evals += 1
        if dropped.bic + 1e-8 < best.bic:
            best = dropped
        swap_candidates = sorted(
            positive_exc_adds(dropped, arrays_all, inh_params, gw_tr, forbidden),
            key=int,
        )
        for idx in swap_candidates:
            if int(idx) == int(drop_idx):
                continue
            support2 = tuple(sorted(dropped.support + (int(idx),)))
            if support2 == tuple(current.support):
                continue
            warm2 = dict(dropped.params)
            warm2[int(idx)] = 0.1
            swap_tasks.append((support2, warm2))
    swap_states = evaluate_support_batch(swap_tasks, cache, solve_uncached, int(support_workers))
    for cand in swap_states:
        swap_evals += 1
        if cand.bic + 1e-8 < best.bic:
            best = cand
    return best, {
        "add_candidates": int(len(add_candidates)),
        "add_evals": int(add_evals),
        "drop_evals": int(drop_evals),
        "swap_evals": int(swap_evals),
        "support_evals": int(len(cache)),
    }


def best_inh_neighbor(
    *,
    current: InhState,
    arrays_all,
    mu: float,
    exc_params: dict[int, float],
    gw_tr: np.ndarray,
    gw_va: np.ndarray,
    forbidden: set[int],
    support_workers: int,
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
    add_candidates_scored = screened_inh_adds(current, arrays_all, forbidden, penalty_per_rule, int(support_workers))
    add_candidates = [int(idx) for _gain, idx in add_candidates_scored]
    add_tasks = []
    for idx in add_candidates:
        support = tuple(sorted(current.support + (int(idx),)))
        warm = dict(current.params)
        warm[int(idx)] = 0.1
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
    for drop_idx, dropped in zip(drop_ids, dropped_states):
        drop_evals += 1
        if dropped.bic + 1e-8 < best.bic:
            best = dropped
        swap_candidates_scored = screened_inh_adds(dropped, arrays_all, forbidden, penalty_per_rule, int(support_workers))
        for _gain, idx in swap_candidates_scored:
            if int(idx) == int(drop_idx):
                continue
            support2 = tuple(sorted(dropped.support + (int(idx),)))
            if support2 == tuple(current.support):
                continue
            warm2 = dict(dropped.params)
            warm2[int(idx)] = 0.1
            swap_tasks.append((support2, warm2))
    swap_states = evaluate_support_batch(swap_tasks, cache, solve_uncached, int(support_workers))
    for cand in swap_states:
        swap_evals += 1
        if cand.bic + 1e-8 < best.bic:
            best = cand
    return best, {
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


def run_exact_block_solver(problem, *, opt_steps: int, lr: float, max_rounds: int, device: torch.device, support_workers: int):
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
        )
        if best_exc.bic + 1e-8 < exc_state.bic:
            exc_state = best_exc
            changed = True

        best_inh, inh_stats = best_inh_neighbor(
            current=inh_state,
            arrays_all=arrays_all,
            mu=float(exc_state.mu),
            exc_params=exc_state.params,
            gw_tr=gw_tr,
            gw_va=gw_va,
            forbidden=set(int(x) for x in exc_state.support),
            support_workers=int(support_workers),
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
                "exc_add_candidates": int(exc_stats["add_candidates"]),
                "inh_add_candidates": int(inh_stats["add_candidates"]),
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
    }


def evaluate_benchmark(*, name: str, config_path: Path, opt_steps: int, lr: float, max_rounds: int, cleanup_data: bool, device: torch.device, support_workers: int):
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
            "true_rule": format_rules(gt, target),
            "matched_rule": format_rules(matched, target),
            "missing_rule": format_rules(missing, target),
            "extra_rule": format_rules(extra, target),
            "recall": float(len(matched)) / max(len(gt_set), 1),
            "precision": float(len(matched)) / max(len(pred_set), 1),
            "round_logs": solved["logs"],
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

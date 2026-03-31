from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import brentq, minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.synthetic import create_rules_from_config, generate_exp_data
import rule_dependent_kernel_active_set as rd


@dataclass
class SupportState:
    support: tuple[int, ...]
    delta_nll: float
    beta: np.ndarray
    residual: np.ndarray
    improvement: float


def ensure_local_dataset(config_path: Path, out_path: Path):
    if out_path.exists():
        return out_path
    cfg = yaml.safe_load(config_path.read_text())
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(dataset, f)
    return out_path


def setup_problem(data_path: Path, config_path: Path):
    train, val, metadata = rd.load_dataset(str(data_path))
    config = rd.load_yaml(str(config_path))
    target = int(config["rules"][0]["target"])
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
    return {
        "config": config,
        "gt": gt,
        "target": target,
        "global_kernels": global_kernels,
        "basis_cache": basis_cache,
        "subsets": subsets,
        "rule_heights": rule_heights,
        "grid_weights_train": np.asarray(tr_grid_w, dtype=np.float64),
        "grid_weights_val": np.asarray(va_grid_w, dtype=np.float64),
    }


def build_arrays_all(problem):
    return rd.compute_rule_feature_arrays(
        subsets=problem["subsets"],
        basis_cache=problem["basis_cache"],
        rule_heights=problem["rule_heights"],
        kernels=problem["global_kernels"],
    )


def summarize_rules(subsets, inh_params, target):
    return sorted(
        rd.summarize_results(
            subsets=subsets,
            exc_params={},
            inh_params=inh_params,
            target=target,
        )
    )


def run_anchor(problem, *, opt_steps: int, lr: float, device: torch.device, use_postprune_anchor: bool):
    arrays_all = build_arrays_all(problem)
    torch_basis_cache = rd.TorchBasisCache(problem["basis_cache"], device)
    bic, mu, exc_params, inh_params, rule_heights, active_rules = rd.run_active_set(
        subsets=problem["subsets"],
        init_arrays=arrays_all,
        rule_heights=problem["rule_heights"],
        basis_cache=problem["basis_cache"],
        grid_weights_train=problem["grid_weights_train"],
        grid_weights_val=problem["grid_weights_val"],
        max_rules=12,
        opt_steps=int(opt_steps),
        lr=float(lr),
        device=device,
        torch_basis_cache=torch_basis_cache,
        penalize_kernel_df=False,
    )
    bic, mu, exc_params, inh_params, rule_heights, active_rules = rd.family_attribution_refine(
        active_rules=active_rules,
        subsets=problem["subsets"],
        basis_cache=problem["basis_cache"],
        rule_heights=rule_heights,
        mu=float(mu),
        exc_params=exc_params,
        inh_params=inh_params,
        grid_weights_train=problem["grid_weights_train"],
        grid_weights_val=problem["grid_weights_val"],
        device=device,
        torch_basis_cache=torch_basis_cache,
        opt_steps=int(opt_steps),
        lr=float(lr),
        passes=1,
    )
    if bool(use_postprune_anchor):
        bic, mu, exc_params, inh_params, rule_heights, active_rules, _scale = rd.choose_post_prune_by_penalty_scale(
            active_rules=active_rules,
            subsets=problem["subsets"],
            basis_cache=problem["basis_cache"],
            rule_heights=rule_heights,
            mu=float(mu),
            exc_params=exc_params,
            inh_params=inh_params,
            grid_weights_train=problem["grid_weights_train"],
            grid_weights_val=problem["grid_weights_val"],
            device=device,
            torch_basis_cache=torch_basis_cache,
            opt_steps=int(opt_steps),
            lr=float(lr),
            min_order=2,
            penalty_scale_grid=[0.6, 0.8, 1.0],
        )
    problem["rule_heights"] = rule_heights
    arrays_all = build_arrays_all(problem)
    active_rules = list(active_rules)
    exc_params = {int(k): float(v) for k, v in exc_params.items()}
    inh_params = {int(k): float(v) for k, v in inh_params.items()}
    exc_ev, inh_ev, exc_gr, inh_gr = rd.contributions_from_active(
        active_rules=active_rules,
        arrays_out=arrays_all,
        exc_params=exc_params,
        inh_params=inh_params,
        tr_event_len=int(next(iter(arrays_all.values()))[0].size),
        tr_grid_len=int(next(iter(arrays_all.values()))[1].size),
    )
    eta_grid = np.clip(float(mu) + exc_gr, 1e-8, None)
    base_grid_mass = problem["grid_weights_train"] * eta_grid * np.exp(-inh_gr)
    return {
        "bic": float(bic),
        "mu": float(mu),
        "active_rules": active_rules,
        "exc_params": exc_params,
        "inh_params": inh_params,
        "arrays_all": arrays_all,
        "base_grid_mass": base_grid_mass,
        "exc_ev": exc_ev,
        "inh_ev": inh_ev,
        "exc_gr": exc_gr,
        "inh_gr": inh_gr,
    }


def optimize_support(
    support: tuple[int, ...],
    *,
    event_sum_map: dict[int, float],
    grid_col_map: dict[int, np.ndarray],
    base_grid_mass: np.ndarray,
    penalty_per_rule: float,
    warm_beta: np.ndarray | None = None,
) -> SupportState:
    if not support:
        return SupportState(
            support=(),
            delta_nll=0.0,
            beta=np.zeros((0,), dtype=np.float64),
            residual=base_grid_mass,
            improvement=0.0,
        )

    event_vec = np.asarray([event_sum_map[int(idx)] for idx in support], dtype=np.float64)
    grid_mat = np.column_stack([grid_col_map[int(idx)] for idx in support]).astype(np.float64, copy=False)

    def objective(beta: np.ndarray):
        xb = grid_mat @ beta
        exp_term = np.exp(-np.clip(xb, 0.0, 60.0))
        val = float(np.dot(event_vec, beta) + np.sum(base_grid_mass * (exp_term - 1.0)))
        grad = event_vec - grid_mat.T @ (base_grid_mass * exp_term)
        return val, grad

    x0 = np.zeros((len(support),), dtype=np.float64) if warm_beta is None else np.maximum(np.asarray(warm_beta, dtype=np.float64), 0.0)
    res = minimize(
        fun=lambda x: objective(x)[0],
        x0=x0,
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(support),
        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6},
    )
    beta = np.maximum(np.asarray(res.x, dtype=np.float64), 0.0)
    xb = grid_mat @ beta
    residual = base_grid_mass * np.exp(-np.clip(xb, 0.0, 60.0))
    delta_nll = float(np.dot(event_vec, beta) + np.sum(residual - base_grid_mass))
    improvement = float(-delta_nll - float(penalty_per_rule) * len(support))
    return SupportState(
        support=support,
        delta_nll=delta_nll,
        beta=beta,
        residual=residual,
        improvement=improvement,
    )


def scalar_gain(
    idx: int,
    *,
    event_sum_map: dict[int, float],
    grid_col_map: dict[int, np.ndarray],
    residual: np.ndarray,
    penalty_per_rule: float,
) -> tuple[float, float]:
    a = float(event_sum_map[int(idx)])
    x = np.asarray(grid_col_map[int(idx)], dtype=np.float64)
    g0 = float(np.dot(residual, x) - a)
    if g0 <= 1e-12:
        return 0.0, 0.0

    def grad(t: float):
        return a - float(np.dot(residual * x, np.exp(-np.clip(x * t, 0.0, 60.0))))

    hi = 1.0
    while grad(hi) < 0.0 and hi < 256.0:
        hi *= 2.0
    t_star = float(brentq(grad, 0.0, hi))
    delta_nll = float(a * t_star + np.sum(residual * (np.exp(-np.clip(x * t_star, 0.0, 60.0)) - 1.0)))
    gain = float(-delta_nll - float(penalty_per_rule))
    return max(gain, 0.0), t_star


def search_best_additions(
    candidate_ids: list[int],
    *,
    event_sum_map: dict[int, float],
    grid_col_map: dict[int, np.ndarray],
    base_grid_mass: np.ndarray,
    penalty_per_rule: float,
    max_add: int,
):
    cache: dict[tuple[int, ...], SupportState] = {
        (): SupportState(
            support=(),
            delta_nll=0.0,
            beta=np.zeros((0,), dtype=np.float64),
            residual=base_grid_mass,
            improvement=0.0,
        )
    }
    best = cache[()]
    node_count = 0

    def solve_child(parent: SupportState, add_idx: int) -> SupportState:
        support = tuple(sorted(parent.support + (int(add_idx),)))
        if support in cache:
            return cache[support]
        gain1, beta1 = scalar_gain(
            int(add_idx),
            event_sum_map=event_sum_map,
            grid_col_map=grid_col_map,
            residual=parent.residual,
            penalty_per_rule=penalty_per_rule,
        )
        warm = []
        parent_pos = {idx: j for j, idx in enumerate(parent.support)}
        for idx in support:
            if idx in parent_pos:
                warm.append(float(parent.beta[parent_pos[idx]]))
            elif idx == int(add_idx):
                warm.append(float(beta1))
            else:
                warm.append(0.0)
        state = optimize_support(
            support,
            event_sum_map=event_sum_map,
            grid_col_map=grid_col_map,
            base_grid_mass=base_grid_mass,
            penalty_per_rule=penalty_per_rule,
            warm_beta=np.asarray(warm, dtype=np.float64),
        )
        cache[support] = state
        return state

    def recurse(state: SupportState, remaining: tuple[int, ...]):
        nonlocal best, node_count
        node_count += 1
        if state.improvement > best.improvement + 1e-10:
            best = state
        if not remaining or len(state.support) >= int(max_add):
            return

        gains = []
        for idx in remaining:
            gain, beta1 = scalar_gain(
                int(idx),
                event_sum_map=event_sum_map,
                grid_col_map=grid_col_map,
                residual=state.residual,
                penalty_per_rule=penalty_per_rule,
            )
            gains.append((float(gain), int(idx), float(beta1)))
        gains.sort(reverse=True)
        slots_left = int(max_add) - len(state.support)
        ub = float(state.improvement + sum(max(0.0, g[0]) for g in gains[:slots_left]))
        if ub <= best.improvement + 1e-10:
            return

        pivot_gain, pivot_idx, _pivot_beta = gains[0]
        tail = tuple(int(idx) for idx in remaining if int(idx) != int(pivot_idx))
        if float(pivot_gain) > 1e-12:
            child = solve_child(state, int(pivot_idx))
            recurse(child, tail)
        recurse(state, tail)

    recurse(cache[()], tuple(int(idx) for idx in candidate_ids))
    return best, node_count, cache


def format_rule_idxs(rule_ids, subsets, target):
    return [rd.format_rule((tuple(int(s) for s in subsets[int(idx)]), "inh", int(target)), int(target)) for idx in rule_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/paper_suite/paper_ablation_inhibition_only.yaml")
    ap.add_argument("--data_out", default="data/paper_suite/paper_ablation_inhibition_only.local.pkl")
    ap.add_argument("--opt_steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--all_inactive", action="store_true")
    ap.add_argument("--include_singletons", action="store_true")
    ap.add_argument("--top_pairs", type=int, default=4)
    ap.add_argument("--top_raw_triplets", type=int, default=15)
    ap.add_argument("--top_exclusive_triplets", type=int, default=12)
    ap.add_argument("--keep_triplets", type=int, default=15)
    ap.add_argument("--max_add", type=int, default=4)
    ap.add_argument("--use_postprune_anchor", action="store_true")
    args = ap.parse_args()

    thread_cap = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(max(thread_cap, 1))
    torch.set_num_interop_threads(1)

    config_path = Path(args.config)
    data_path = ensure_local_dataset(config_path, Path(args.data_out))
    problem = setup_problem(data_path, config_path)
    device = torch.device("cpu")

    anchor = run_anchor(
        problem,
        opt_steps=int(args.opt_steps),
        lr=float(args.lr),
        device=device,
        use_postprune_anchor=bool(args.use_postprune_anchor),
    )
    subsets = problem["subsets"]
    target = int(problem["target"])
    n_eff = float(max(anchor["arrays_all"][0][2].size + problem["grid_weights_val"].size, 2))
    penalty_per_rule = 0.5 * math.log(n_eff)
    active_idx_set = {int(ar.idx) for ar in anchor["active_rules"]}

    pair_scores = []
    triplet_raw = []
    singleton_scores = []
    order_groups: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for idx, subset in enumerate(subsets):
        order = len(tuple(subset))
        if order not in (1, 2, 3) or int(idx) in active_idx_set:
            continue
        gain, sign, _coef0 = rd.rule_score(
            feat_event=anchor["arrays_all"][int(idx)][0],
            feat_grid=anchor["arrays_all"][int(idx)][1],
            mu=float(anchor["mu"]),
            exc_event=anchor["exc_ev"],
            inh_event=anchor["inh_ev"],
            exc_grid=anchor["exc_gr"],
            inh_grid=anchor["inh_gr"],
            grid_weights_train=problem["grid_weights_train"],
            penalty=float(penalty_per_rule),
        )
        if str(sign) != "inh":
            continue
        order_groups[int(order)].append(int(idx))
        if order == 1:
            singleton_scores.append((float(gain), int(idx)))
        elif order == 2:
            pair_scores.append((float(gain), int(idx)))
        else:
            triplet_raw.append((float(gain), int(idx)))

    singleton_scores.sort(reverse=True)
    pair_scores.sort(reverse=True)
    pair_gain_map = {int(idx): float(gain) for gain, idx in pair_scores}

    exclusive_triplets = []
    for raw_gain, idx in triplet_raw:
        subset = tuple(int(s) for s in subsets[int(idx)])
        pair_subs = [tuple(sorted((subset[a], subset[b]))) for a in range(3) for b in range(a + 1, 3)]
        pair_best = max(float(pair_gain_map.get(int(subsets.index(psub)) if psub in subsets else -1, 0.0)) for psub in pair_subs)
        exclusive_triplets.append((float(raw_gain - pair_best), float(raw_gain), int(idx)))

    triplet_raw.sort(reverse=True)
    exclusive_triplets.sort(reverse=True)

    if bool(args.all_inactive):
        candidate_ids = []
        if bool(args.include_singletons):
            candidate_ids.extend(sorted(order_groups[1]))
        candidate_ids.extend(sorted(order_groups[2]))
        candidate_ids.extend(sorted(order_groups[3]))
        pair_ids = sorted(order_groups[2])
        triplet_ids = sorted(order_groups[3])
        singleton_ids = sorted(order_groups[1]) if bool(args.include_singletons) else []
    else:
        triplet_ids = []
        seen = set()
        for _gain, idx in triplet_raw[: int(args.top_raw_triplets)]:
            if int(idx) not in seen:
                seen.add(int(idx))
                triplet_ids.append(int(idx))
        for _exclusive, _raw, idx in exclusive_triplets[: int(args.top_exclusive_triplets)]:
            if int(idx) not in seen:
                seen.add(int(idx))
                triplet_ids.append(int(idx))
        triplet_ids = sorted(
            triplet_ids,
            key=lambda idx: (
                next((x[0] for x in exclusive_triplets if x[2] == idx), -1e18),
                next((x[0] for x in triplet_raw if x[1] == idx), -1e18),
            ),
            reverse=True,
        )[: int(args.keep_triplets)]
        singleton_ids = [int(idx) for _gain, idx in singleton_scores[: int(args.top_pairs)]] if bool(args.include_singletons) else []
        pair_ids = [int(idx) for _gain, idx in pair_scores[: int(args.top_pairs)]]
        candidate_ids = singleton_ids + pair_ids + [idx for idx in triplet_ids if idx not in pair_ids and idx not in singleton_ids]

    event_sum_map = {int(idx): float(np.sum(anchor["arrays_all"][int(idx)][0])) for idx in candidate_ids}
    grid_col_map = {int(idx): np.asarray(anchor["arrays_all"][int(idx)][1], dtype=np.float64) for idx in candidate_ids}

    best_state, node_count, _cache = search_best_additions(
        candidate_ids,
        event_sum_map=event_sum_map,
        grid_col_map=grid_col_map,
        base_grid_mass=np.asarray(anchor["base_grid_mass"], dtype=np.float64),
        penalty_per_rule=float(penalty_per_rule),
        max_add=int(args.max_add),
    )

    selected_additions = list(best_state.support)
    added_rules = [rd.ActiveRule(idx=int(idx), sign="inh") for idx in selected_additions]
    final_active = list(anchor["active_rules"]) + added_rules
    init_coef_map = rd.build_init_coef_map(final_active, anchor["exc_params"], anchor["inh_params"], fallback=0.1)
    for idx, beta in zip(best_state.support, best_state.beta.tolist()):
        init_coef_map[(int(idx), "inh")] = float(max(beta, 1e-4))
    torch_basis_cache = rd.TorchBasisCache(problem["basis_cache"], device)
    refit_bic, refit_mu, _exc_fit, refit_inh, _rule_heights, _arrays_fit = rd.optimize_active_set_torch(
        active_rules=final_active,
        subsets=subsets,
        basis_cache=problem["basis_cache"],
        rule_heights=problem["rule_heights"],
        init_mu=float(anchor["mu"]),
        init_coef_map=init_coef_map,
        grid_weights_train=problem["grid_weights_train"],
        grid_weights_val=problem["grid_weights_val"],
        device=device,
        torch_basis_cache=torch_basis_cache,
        steps=max(30, int(args.opt_steps)),
        lr=float(args.lr),
        penalize_kernel_df=False,
    )

    anchor_rules = summarize_rules(subsets, anchor["inh_params"], target)
    refit_rules = summarize_rules(subsets, refit_inh, target)
    gt = sorted(problem["gt"])
    gt_set = set(gt)
    refit_set = set(refit_rules)
    hit = sorted(gt_set & refit_set)
    miss = sorted(gt_set - refit_set)
    extra = sorted(refit_set - gt_set)

    print("=== INHIBITION PROFILE BLOCK VALIDATION ===")
    print(
        {
            "thread_cap": int(thread_cap),
            "baseline_bic": float(anchor["bic"]),
            "baseline_rule_count": int(len(anchor_rules)),
            "candidate_count": int(len(candidate_ids)),
            "candidate_singletons": int(len(singleton_ids)),
            "candidate_pairs": int(len(pair_ids)),
            "candidate_triplets": int(len(triplet_ids)),
            "search_nodes": int(node_count),
            "best_addition_count": int(len(best_state.support)),
            "profile_improvement": float(best_state.improvement),
            "refit_bic": float(refit_bic),
            "refit_bic_delta_vs_baseline": float(anchor["bic"] - refit_bic),
            "refit_mu": float(refit_mu),
        }
    )
    rd.print_rule_block("Baseline anchor rules:", anchor_rules, target)
    if singleton_ids:
        print("Candidate singleton rules:")
        for txt in format_rule_idxs(singleton_ids, subsets, target):
            print("  -", txt)
    print("Candidate pair rules:")
    for txt in format_rule_idxs(pair_ids, subsets, target):
        print("  -", txt)
    print("Candidate triplet rules:")
    for txt in format_rule_idxs(triplet_ids, subsets, target):
        print("  -", txt)
    print("Best add-on rules:")
    if best_state.support:
        for txt in format_rule_idxs(best_state.support, subsets, target):
            print("  -", txt)
    else:
        print("  - none")
    rd.print_rule_block("Refit predicted rules:", refit_rules, target)
    rd.print_rule_block("Matched rules:", hit, target)
    rd.print_rule_block("Missing rules:", miss, target)
    rd.print_rule_block("Extra predicted rules:", extra, target)


if __name__ == "__main__":
    main()

"""Research sweeps for hierarchical exact-cell selectors on overlap synthetic data."""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np
import yaml

from workspace.train.ablate_component_basis import gt_rules_from_config, prepare_data
from workspace.train.component_basis_init import _build_source_cache, phase1_pairwise_screen, phase2_source_evidence
from workspace.train.research_exact_cell_selectors import (
    build_partitions,
    format_rule,
    node_label,
    parse_csv_floats,
    print_rule_block,
)


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def component_cell_stats(X: np.ndarray, y: np.ndarray):
    d = X.shape[1]
    eta = np.zeros((1 << d,), dtype=np.float64)
    support = np.zeros((1 << d,), dtype=np.float64)
    for bits in range(1 << d):
        mask = np.ones((len(y),), dtype=bool)
        for j in range(d):
            mask &= (X[:, j] == ((bits >> j) & 1))
        cnt = int(mask.sum())
        support[bits] = float(cnt)
        if cnt > 0:
            eta[bits] = logit(float(y[mask].mean()))
    return eta, support


def mobius_rows(eta: np.ndarray, support: np.ndarray, source_ids: tuple[int, ...], max_order: int = 3):
    d = len(source_ids)
    rows = []
    for order in range(1, min(max_order, d) + 1):
        for idxs in itertools.combinations(range(d), order):
            bits_u = 0
            coef = 0.0
            for j in idxs:
                bits_u |= 1 << j
            for r in range(order + 1):
                for sub in itertools.combinations(idxs, r):
                    bits = 0
                    for j in sub:
                        bits |= 1 << j
                    coef += ((-1) ** (order - r)) * float(eta[bits])
            rows.append((tuple(source_ids[j] for j in idxs), float(coef), float(support[bits_u])))
    return rows


def parent_rows(eta: np.ndarray, support: np.ndarray, source_ids: tuple[int, ...], max_order: int = 3):
    d = len(source_ids)
    rows = []
    for order in range(1, min(max_order, d) + 1):
        for idxs in itertools.combinations(range(d), order):
            bits_u = 0
            for j in idxs:
                bits_u |= 1 << j
            best = None
            if order == 1:
                parent_bits = 0
                sup = min(float(support[bits_u]), float(support[parent_bits]))
                if sup > 0.0:
                    delta = float(eta[bits_u] - eta[parent_bits])
                    best = (delta, sup)
            else:
                for anchor in idxs:
                    parent_bits = bits_u & ~(1 << anchor)
                    sup = min(float(support[bits_u]), float(support[parent_bits]))
                    if sup <= 0.0:
                        continue
                    delta = float(eta[bits_u] - eta[parent_bits])
                    cand = (delta, sup)
                    if best is None or abs(delta) * sup > abs(best[0]) * best[1]:
                        best = cand
            if best is None:
                continue
            rows.append((tuple(source_ids[j] for j in idxs), float(best[0]), float(best[1])))
    return rows


def hybrid_rows(eta: np.ndarray, support: np.ndarray, source_ids: tuple[int, ...], max_order: int = 3):
    base_m = {tuple(srcs): (coef, sup) for srcs, coef, sup in mobius_rows(eta, support, source_ids, max_order=max_order)}
    base_p = {tuple(srcs): (coef, sup) for srcs, coef, sup in parent_rows(eta, support, source_ids, max_order=max_order)}
    out = []
    keys = sorted(set(base_m) | set(base_p), key=lambda xs: (len(xs), xs))
    for srcs in keys:
        order = len(srcs)
        if order <= 2 and srcs in base_p:
            coef, sup = base_p[srcs]
        elif srcs in base_m:
            coef, sup = base_m[srcs]
        else:
            coef, sup = base_p[srcs]
        out.append((srcs, float(coef), float(sup)))
    return out


def orderwise_rows(eta: np.ndarray, support: np.ndarray, source_ids: tuple[int, ...], max_order: int = 3):
    base_m = {tuple(srcs): (coef, sup) for srcs, coef, sup in mobius_rows(eta, support, source_ids, max_order=max_order)}
    base_p = {tuple(srcs): (coef, sup) for srcs, coef, sup in parent_rows(eta, support, source_ids, max_order=max_order)}
    out = []
    keys = sorted(set(base_m) | set(base_p), key=lambda xs: (len(xs), xs))
    for srcs in keys:
        order = len(srcs)
        if order <= 2 and srcs in base_m:
            coef, sup = base_m[srcs]
        elif order >= 3 and srcs in base_p:
            coef, sup = base_p[srcs]
        elif srcs in base_m:
            coef, sup = base_m[srcs]
        else:
            coef, sup = base_p[srcs]
        out.append((srcs, float(coef), float(sup)))
    return out


def orderwise_alt_rows(eta: np.ndarray, support: np.ndarray, source_ids: tuple[int, ...], max_order: int = 3):
    base_m = {tuple(srcs): (coef, sup) for srcs, coef, sup in mobius_rows(eta, support, source_ids, max_order=max_order)}
    base_p = {tuple(srcs): (coef, sup) for srcs, coef, sup in parent_rows(eta, support, source_ids, max_order=max_order)}
    out = []
    keys = sorted(set(base_m) | set(base_p), key=lambda xs: (len(xs), xs))
    for srcs in keys:
        order = len(srcs)
        if order == 2 and srcs in base_m:
            coef, sup = base_m[srcs]
        elif srcs in base_p:
            coef, sup = base_p[srcs]
        else:
            coef, sup = base_m[srcs]
        out.append((srcs, float(coef), float(sup)))
    return out


def build_rows(mode: str, eta: np.ndarray, support: np.ndarray, source_ids: tuple[int, ...], max_order: int = 3):
    if mode == "mobius":
        return mobius_rows(eta, support, source_ids, max_order=max_order)
    if mode == "parent":
        return parent_rows(eta, support, source_ids, max_order=max_order)
    if mode == "hybrid":
        return hybrid_rows(eta, support, source_ids, max_order=max_order)
    if mode == "orderwise":
        return orderwise_rows(eta, support, source_ids, max_order=max_order)
    if mode == "orderwise_alt":
        return orderwise_alt_rows(eta, support, source_ids, max_order=max_order)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixed_target", type=int, default=6)
    ap.add_argument("--partition_mode", choices=["manual", "singleton", "all_partitions"], default="all_partitions")
    ap.add_argument("--max_block", type=int, default=4)
    ap.add_argument("--mode_values", default="mobius,parent,hybrid")
    ap.add_argument("--thr_values", default="0.05,0.15")
    ap.add_argument("--support_pow_values", default="1.0")
    ap.add_argument("--order_bonus_values", default="2.0")
    ap.add_argument("--sup_pen_values", default="0.5,2.0")
    ap.add_argument("--topn", type=int, default=40)
    args = ap.parse_args()

    train_data, _, metadata = prepare_data(args.data)
    cfg = load_yaml(args.config)
    gt = gt_rules_from_config(cfg)
    num_types = int(metadata["num_types"])

    n_val = min(max(1, int(len(train_data) * 0.15)), len(train_data) - 1)
    tr_data = train_data[n_val:]

    phase1 = phase1_pairwise_screen(
        tr_data,
        num_types,
        int(args.fixed_target),
        delta_t=0.1,
        max_lag=10.0,
        source_pool_topk=8,
    )
    phase2 = phase2_source_evidence(tr_data, phase1, fixed_target=int(args.fixed_target))
    tr_cache = _build_source_cache(
        tr_data,
        phase2["source_defs"],
        fixed_target=int(args.fixed_target),
        num_types=num_types,
        int_grid_mult=1,
    )
    source_ids = [int(sd.source) for sd in phase2["source_defs"]]
    local_by_source = {s: i for i, s in enumerate(source_ids)}
    components_list = build_partitions(
        args.partition_mode,
        tuple(sorted(source_ids)),
        max_block=int(args.max_block),
    )
    print(f"partition_mode={args.partition_mode} num_partitions={len(components_list)}")
    mode_values = [x.strip() for x in args.mode_values.split(",") if x.strip()]
    thr_values = parse_csv_floats(args.thr_values)
    support_pow_values = parse_csv_floats(args.support_pow_values)
    order_bonus_values = parse_csv_floats(args.order_bonus_values)
    sup_pen_values = parse_csv_floats(args.sup_pen_values)

    results = []
    total = (
        len(mode_values)
        * len(components_list)
        * len(thr_values)
        * len(support_pow_values)
        * len(order_bonus_values)
        * len(sup_pen_values)
    )
    counter = 0
    y = tr_cache["event_y"].astype(np.int64)
    for mode, thr, support_pow, order_bonus, sup_pen in itertools.product(
        mode_values,
        thr_values,
        support_pow_values,
        order_bonus_values,
        sup_pen_values,
    ):
        X_all = (tr_cache["event_q"] > float(thr)).astype(np.int64)
        for components in components_list:
            scored = []
            for comp in components:
                idxs = [local_by_source[s] for s in comp]
                eta, support = component_cell_stats(X_all[:, idxs], y)
                rows = build_rows(mode, eta, support, tuple(comp), max_order=3)
                for srcs, coef, sup in rows:
                    sign = "exc" if coef > 0.0 else "inh"
                    score = abs(coef) * (max(sup, 1.0) ** float(support_pow))
                    scored.append((tuple(sorted(srcs)), sign, float(score), len(srcs)))

            ranked = []
            for srcs, sign, score, order in scored:
                sup = 0.0
                set_r = set(srcs)
                for srcs2, sign2, score2, _ in scored:
                    if sign != sign2:
                        continue
                    if set_r < set(srcs2):
                        sup = max(sup, score2)
                adj = score + float(order_bonus) * max(0, order - 1) - float(sup_pen) * sup
                ranked.append((adj, score, order, (srcs, sign, int(args.fixed_target))))
            ranked.sort(reverse=True)
            pred = set(x[3] for x in ranked[: len(gt)])
            results.append(
                (
                    len(gt & pred),
                    mode,
                    thr,
                    support_pow,
                    order_bonus,
                    sup_pen,
                    tuple(tuple(int(s) for s in comp) for comp in components),
                    sorted(pred),
                )
            )
            counter += 1
            if counter % 5000 == 0:
                print(f"progress={counter}/{total}")

    results.sort(reverse=True)
    seen = set()
    for item in results[: args.topn]:
        key = (item[0], item[1], tuple(item[-2]), tuple(item[-1]))
        if key in seen:
            continue
        seen.add(key)
        print(item)

    if results:
        best = results[0]
        preds = set(best[-1])
        hit = sorted(gt & preds)
        miss = sorted(gt - preds)
        extra = sorted(preds - gt)
        print("===BEST REPORT===")
        print(
            "best_params:",
            {
                "hit_count": int(best[0]),
                "mode": best[1],
                "thr": float(best[2]),
                "support_pow": float(best[3]),
                "order_bonus": float(best[4]),
                "sup_pen": float(best[5]),
                "components": best[6],
            },
        )
        print_rule_block("True rules:", sorted(gt), int(args.fixed_target))
        print_rule_block("Predicted rules:", sorted(preds), int(args.fixed_target))
        print_rule_block("Matched rules:", hit, int(args.fixed_target))
        print_rule_block("Missing rules:", miss, int(args.fixed_target))
        print_rule_block("Extra predicted rules:", extra, int(args.fixed_target))


if __name__ == "__main__":
    main()

"""Research sweeps for stagewise residual exact-cell selectors."""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import yaml

from workspace.train.ablate_component_basis import gt_rules_from_config, prepare_data
from workspace.train.component_basis_init import _build_source_cache, phase1_pairwise_screen, phase2_source_evidence
from workspace.train.research_exact_cell_selectors import build_partitions, parse_csv_floats, print_rule_block
from workspace.train.research_hier_cell_selectors import build_rows, component_cell_stats


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def subset_bits(idxs: tuple[int, ...]) -> int:
    bits = 0
    for j in idxs:
        bits |= 1 << j
    return bits


def apply_rule_to_residual(res_eta: np.ndarray, idxs: tuple[int, ...], coef: float):
    bits_u = subset_bits(idxs)
    for bits in range(res_eta.shape[0]):
        if (bits & bits_u) == bits_u:
            res_eta[bits] -= float(coef)


def select_stage(
    res_tables,
    components,
    mode: str,
    order: int,
    support_pow: float,
    sup_pen: float,
    fixed_target: int,
):
    rows = []
    for comp_id, comp in enumerate(components):
        eta = res_tables[comp_id]
        support = res_tables[f"support_{comp_id}"]
        cand_rows = build_rows(mode, eta, support, tuple(comp), max_order=3)
        scored_rows = []
        for srcs, coef, sup in cand_rows:
            sign = "exc" if float(coef) > 0.0 else "inh"
            score = abs(float(coef)) * (max(float(sup), 1.0) ** float(support_pow))
            scored_rows.append((tuple(sorted(srcs)), sign, float(score), float(coef), int(len(srcs))))
        for srcs, sign, score, coef, row_order in scored_rows:
            if row_order != order:
                continue
            sup_shadow = 0.0
            set_r = set(srcs)
            for srcs2, sign2, score2, _, row_order2 in scored_rows:
                if sign != sign2:
                    continue
                if row_order2 <= row_order:
                    continue
                if set_r < set(srcs2):
                    sup_shadow = max(sup_shadow, float(score2))
            adj_score = score - float(sup_pen) * sup_shadow
            rows.append((float(adj_score), float(coef), comp_id, tuple(sorted(srcs)), sign, int(fixed_target)))
    rows.sort(reverse=True)
    dedup = []
    seen = set()
    for row in rows:
        key = (row[3], row[4], row[5])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
    return dedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixed_target", type=int, default=6)
    ap.add_argument("--partition_mode", choices=["manual", "singleton", "all_partitions"], default="all_partitions")
    ap.add_argument("--max_block", type=int, default=4)
    ap.add_argument("--mode_values", default="mobius,parent,orderwise,orderwise_alt")
    ap.add_argument("--thr_values", default="0.05,0.15")
    ap.add_argument("--support_pow_values", default="1.0")
    ap.add_argument("--sup_pen_values", default="0.0,0.25,0.5,1.0")
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
    sup_pen_values = parse_csv_floats(args.sup_pen_values)
    quota_values = [(k1, k2, k3) for k1 in range(8) for k2 in range(8) for k3 in range(8) if k1 + k2 + k3 == len(gt)]
    y = tr_cache["event_y"].astype(np.int64)

    results = []
    for thr, support_pow, sup_pen, components in itertools.product(thr_values, support_pow_values, sup_pen_values, components_list):
        X_all = (tr_cache["event_q"] > float(thr)).astype(np.int64)
        base_tables = {}
        for comp_id, comp in enumerate(components):
            idxs = [local_by_source[s] for s in comp]
            eta, support = component_cell_stats(X_all[:, idxs], y)
            base_tables[comp_id] = eta
            base_tables[f"support_{comp_id}"] = support

        for mode1, mode2, mode3, quota in itertools.product(mode_values, mode_values, mode_values, quota_values):
            res_tables = {}
            for key, val in base_tables.items():
                res_tables[key] = val.copy()
            selected = []

            for order, mode, k in ((1, mode1, quota[0]), (2, mode2, quota[1]), (3, mode3, quota[2])):
                if k <= 0:
                    continue
                rows = select_stage(res_tables, components, mode, order, support_pow, sup_pen, int(args.fixed_target))
                chosen = rows[:k]
                selected.extend((srcs, sign, tgt) for _, _, _, srcs, sign, tgt in chosen)
                for _, coef, comp_id, srcs, _, _ in chosen:
                    local_idxs = tuple(components[comp_id].index(s) for s in srcs)
                    apply_rule_to_residual(res_tables[comp_id], local_idxs, coef)

            pred = tuple(sorted(set(selected)))
            results.append(
                (
                    len(gt & set(pred)),
                    float(thr),
                    float(support_pow),
                    float(sup_pen),
                    mode1,
                    mode2,
                    mode3,
                    quota,
                    tuple(tuple(int(s) for s in comp) for comp in components),
                    pred,
                )
            )

    results.sort(reverse=True)
    seen = set()
    for item in results[: args.topn]:
        key = (item[0], item[3], item[4], item[5], item[6], item[7], item[8], item[9])
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
                "thr": best[1],
                "support_pow": best[2],
                "sup_pen": best[3],
                "mode1": best[4],
                "mode2": best[5],
                "mode3": best[6],
                "quota": best[7],
                "components": best[8],
            },
        )
        print_rule_block("True rules:", sorted(gt), int(args.fixed_target))
        print_rule_block("Predicted rules:", sorted(preds), int(args.fixed_target))
        print_rule_block("Matched rules:", hit, int(args.fixed_target))
        print_rule_block("Missing rules:", miss, int(args.fixed_target))
        print_rule_block("Extra predicted rules:", extra, int(args.fixed_target))


if __name__ == "__main__":
    main()

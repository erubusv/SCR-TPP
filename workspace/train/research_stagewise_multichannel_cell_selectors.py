"""Stagewise residual exact-cell selectors with signed multichannel source evidence.

This keeps the current best stagewise hierarchy, but replaces single-channel
source evidence q_s(t) with up to two latent channels per source, one for the
strongest negative lag lobe and one for the strongest positive lag lobe.
Source-level rules are recovered by selecting the best channel combination for
each source subset/sign pair, then residualising on the chosen latent combo.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import yaml

from workspace.train.ablate_component_basis import gt_rules_from_config, prepare_data
from workspace.train.research_channel_rate_selectors import (
    ChannelDef,
    build_channel_cache,
    estimate_channel_bounding,
    phase1_multichannel_screen,
)
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


def collapse_channel_rows(
    eta: np.ndarray,
    support: np.ndarray,
    channel_ids: tuple[int, ...],
    channel_by_id: dict[int, ChannelDef],
    *,
    mode: str,
    support_pow: float,
) -> list[dict]:
    best_by_key: dict[tuple[tuple[int, ...], str], dict] = {}
    for combo, coef, sup in build_rows(mode, eta, support, channel_ids, max_order=3):
        combo = tuple(int(x) for x in combo)
        if len(combo) == 0:
            continue
        srcs = tuple(sorted(int(channel_by_id[ch].source) for ch in combo))
        if len(set(srcs)) != len(combo):
            continue
        sign = "exc" if float(coef) > 0.0 else "inh"
        score = abs(float(coef)) * (max(float(sup), 1.0) ** float(support_pow))
        row = {
            "srcs": srcs,
            "sign": sign,
            "score": float(score),
            "coef": float(coef),
            "support": float(sup),
            "combo": combo,
            "order": int(len(srcs)),
        }
        key = (srcs, sign)
        cur = best_by_key.get(key)
        if cur is None or float(row["score"]) > float(cur["score"]):
            best_by_key[key] = row
    return list(best_by_key.values())


def select_stage(
    res_tables,
    component_defs: list[dict],
    channel_by_id: dict[int, ChannelDef],
    *,
    mode: str,
    order: int,
    support_pow: float,
    sup_pen: float,
    fixed_target: int,
):
    rows = []
    for comp_id, comp in enumerate(component_defs):
        eta = res_tables[comp_id]
        support = res_tables[f"support_{comp_id}"]
        collapsed = collapse_channel_rows(
            eta,
            support,
            tuple(int(ch) for ch in comp["channel_ids"]),
            channel_by_id,
            mode=mode,
            support_pow=support_pow,
        )
        for row in collapsed:
            if int(row["order"]) != int(order):
                continue
            sup_shadow = 0.0
            set_r = set(row["srcs"])
            for row2 in collapsed:
                if row["sign"] != row2["sign"]:
                    continue
                if int(row2["order"]) <= int(row["order"]):
                    continue
                if set_r < set(row2["srcs"]):
                    sup_shadow = max(sup_shadow, float(row2["score"]))
            adj_score = float(row["score"] - float(sup_pen) * sup_shadow)
            rows.append(
                (
                    adj_score,
                    float(row["coef"]),
                    comp_id,
                    tuple(int(s) for s in row["srcs"]),
                    str(row["sign"]),
                    int(fixed_target),
                    tuple(int(ch) for ch in row["combo"]),
                )
            )
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
    ap.add_argument("--manual_partition", default=None)
    ap.add_argument("--mode_values", default="parent")
    ap.add_argument("--thr_values", default="0.12,0.2,0.25")
    ap.add_argument("--support_pow_values", default="1.0,1.25,1.75,3.0")
    ap.add_argument("--sup_pen_values", default="0.5,1.0,2.0,2.5,10.0")
    ap.add_argument("--delta_t", type=float, default=0.1)
    ap.add_argument("--max_lag", type=float, default=10.0)
    ap.add_argument("--source_pool_topk", type=int, default=8)
    ap.add_argument("--min_mass_frac", type=float, default=0.01)
    ap.add_argument("--adaptive_split", action="store_true")
    ap.add_argument("--conflict_balance_min", type=float, default=0.0)
    ap.add_argument("--conflict_sep_min", type=float, default=0.0)
    ap.add_argument("--conflict_gain_min", type=float, default=0.0)
    ap.add_argument("--split_thr", type=float, default=0.2)
    ap.add_argument(
        "--source_kernel_mode",
        choices=[
            "triangular_exact",
            "triangular_pwlin",
            "gaussian_exact",
            "gaussian_pwlin",
            "exponential_exact",
            "exponential_pwlin",
        ],
        default="triangular_pwlin",
    )
    ap.add_argument("--source_kernel_num_bins", type=int, default=50)
    ap.add_argument("--source_kernel_support_mult", type=float, default=3.0)
    ap.add_argument("--topn", type=int, default=40)
    args = ap.parse_args()

    train_data, _, metadata = prepare_data(args.data)
    cfg = load_yaml(args.config)
    gt = gt_rules_from_config(cfg)
    num_types = int(metadata["num_types"])
    n_val = min(max(1, int(len(train_data) * 0.15)), len(train_data) - 1)
    va_data = train_data[:n_val]
    tr_data = train_data[n_val:]

    phase1 = phase1_multichannel_screen(
        tr_data,
        num_types,
        int(args.fixed_target),
        delta_t=float(args.delta_t),
        max_lag=float(args.max_lag),
        source_pool_topk=int(args.source_pool_topk),
        min_mass_frac=float(args.min_mass_frac),
        val_data_list=va_data,
        adaptive_split=bool(args.adaptive_split),
        conflict_balance_min=float(args.conflict_balance_min),
        conflict_sep_min=float(args.conflict_sep_min),
        conflict_gain_min=float(args.conflict_gain_min),
        split_thr=float(args.split_thr),
        kernel_eval_mode=str(args.source_kernel_mode),
        kernel_num_bins=int(args.source_kernel_num_bins),
        kernel_max_cap=float(args.max_lag),
        kernel_support_mult=float(args.source_kernel_support_mult),
    )
    channels = estimate_channel_bounding(
        tr_data,
        phase1["channels"],
        fixed_target=int(args.fixed_target),
        kernel_eval_mode=str(args.source_kernel_mode),
        kernel_num_bins=int(args.source_kernel_num_bins),
        kernel_max_cap=float(args.max_lag),
        kernel_support_mult=float(args.source_kernel_support_mult),
    )
    channel_cache = build_channel_cache(
        tr_data,
        channels,
        fixed_target=int(args.fixed_target),
        num_types=num_types,
        int_grid_mult=1,
        kernel_eval_mode=str(args.source_kernel_mode),
        kernel_num_bins=int(args.source_kernel_num_bins),
        kernel_max_cap=float(args.max_lag),
        kernel_support_mult=float(args.source_kernel_support_mult),
    )

    channel_by_id = {int(ch.channel_id): ch for ch in channels}
    channel_col_by_id = {int(ch.channel_id): i for i, ch in enumerate(channels)}
    source_ids = tuple(sorted({int(ch.source) for ch in channels}))
    components_list = build_partitions(
        args.partition_mode,
        source_ids,
        max_block=int(args.max_block),
        manual_partition_text=args.manual_partition,
    )
    print(f"partition_mode={args.partition_mode} num_partitions={len(components_list)}")
    print("source_pool:", list(source_ids))
    print("source_summaries:", phase1.get("source_summaries", []))
    print(
        "channels:",
        [
            {
                "id": int(ch.channel_id),
                "src": int(ch.source),
                "sign_prior": int(ch.sign_prior),
                "peak": round(float(ch.peak), 3),
                "width": round(float(ch.width), 3),
                "beta": round(float(ch.beta), 3),
                "alpha": round(float(ch.alpha), 3),
            }
            for ch in channels
        ],
    )

    mode_values = [x.strip() for x in args.mode_values.split(",") if x.strip()]
    thr_values = parse_csv_floats(args.thr_values)
    support_pow_values = parse_csv_floats(args.support_pow_values)
    sup_pen_values = parse_csv_floats(args.sup_pen_values)
    quota_values = [(k1, k2, k3) for k1 in range(8) for k2 in range(8) for k3 in range(8) if k1 + k2 + k3 == len(gt)]

    results = []
    for thr, support_pow, sup_pen, components in itertools.product(
        thr_values,
        support_pow_values,
        sup_pen_values,
        components_list,
    ):
        X_all = (channel_cache["event_q"] > float(thr)).astype(np.int64)
        y = channel_cache["event_y"].astype(np.int64)
        component_defs = []
        base_tables = {}
        for comp_id, comp_sources in enumerate(components):
            local_channels = [
                int(ch.channel_id)
                for ch in channels
                if int(ch.source) in set(int(s) for s in comp_sources)
            ]
            local_cols = [int(channel_col_by_id[ch]) for ch in local_channels]
            component_defs.append(
                {
                    "sources": tuple(int(s) for s in comp_sources),
                    "channel_ids": tuple(int(ch) for ch in local_channels),
                    "channel_pos": {int(ch): i for i, ch in enumerate(local_channels)},
                }
            )
            if local_cols:
                eta, support = component_cell_stats(X_all[:, local_cols], y)
            else:
                eta = np.zeros((1,), dtype=np.float64)
                support = np.zeros((1,), dtype=np.float64)
            base_tables[comp_id] = eta
            base_tables[f"support_{comp_id}"] = support

        for mode1, mode2, mode3, quota in itertools.product(mode_values, mode_values, mode_values, quota_values):
            res_tables = {key: val.copy() for key, val in base_tables.items()}
            selected = []
            for order, mode, k in ((1, mode1, quota[0]), (2, mode2, quota[1]), (3, mode3, quota[2])):
                if k <= 0:
                    continue
                rows = select_stage(
                    res_tables,
                    component_defs,
                    channel_by_id,
                    mode=mode,
                    order=order,
                    support_pow=float(support_pow),
                    sup_pen=float(sup_pen),
                    fixed_target=int(args.fixed_target),
                )
                chosen = rows[:k]
                selected.extend((srcs, sign, tgt) for _, _, _, srcs, sign, tgt, _ in chosen)
                for _, coef, comp_id, _, _, _, combo in chosen:
                    local_idxs = tuple(int(component_defs[comp_id]["channel_pos"][int(ch)]) for ch in combo)
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
        key = (item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9])
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

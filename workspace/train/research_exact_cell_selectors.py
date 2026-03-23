"""Research sweeps for exact-cell component selectors on overlap synthetic data."""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Iterable

import numpy as np
import yaml

from workspace.train.ablate_component_basis import gt_rules_from_config, prepare_data
from workspace.train.component_basis_init import _build_source_cache, phase1_pairwise_screen, phase2_source_evidence


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_manual_partition(text: str) -> tuple[tuple[int, ...], ...]:
    blocks = []
    for raw_block in text.split("|"):
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        block = tuple(sorted(int(x.strip()) for x in raw_block.split(",") if x.strip()))
        if block:
            blocks.append(block)
    return tuple(sorted(blocks))


def node_label(node: int, fixed_target: int) -> str:
    if int(node) == int(fixed_target):
        return "T"
    if 0 <= int(node) < 26:
        return chr(ord("A") + int(node))
    return f"S{int(node)}"


def format_rule(rule: tuple[tuple[int, ...], str, int], fixed_target: int) -> str:
    srcs, sign, target = rule
    lhs = " and ".join(node_label(int(s), fixed_target) for s in srcs)
    rhs = node_label(int(target), fixed_target)
    sign_txt = "excitation" if sign == "exc" else "inhibition"
    return f"{lhs} -> {rhs} : {sign_txt}"


def print_rule_block(title: str, rules: list[tuple[tuple[int, ...], str, int]], fixed_target: int):
    print(title)
    if not rules:
        print("  - none")
        return
    for rule in rules:
        print(f"  - {format_rule(rule, fixed_target)}")


def logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def family_mobius(X: np.ndarray, y: np.ndarray, source_ids: tuple[int, ...], max_order: int = 3):
    d = X.shape[1]
    cell_eta = {}
    cell_support = {}
    for bits in range(1 << d):
        mask = np.ones((len(y),), dtype=bool)
        for j in range(d):
            mask &= (X[:, j] == ((bits >> j) & 1))
        support = int(mask.sum())
        if support <= 0:
            cell_eta[bits] = 0.0
            cell_support[bits] = 0.0
            continue
        cell_eta[bits] = logit(float(y[mask].mean()))
        cell_support[bits] = float(support)

    rows = []
    for order in range(1, min(max_order, d) + 1):
        for idxs in itertools.combinations(range(d), order):
            coef = 0.0
            for r in range(order + 1):
                for sub in itertools.combinations(idxs, r):
                    bits = 0
                    for j in sub:
                        bits |= (1 << j)
                    coef += ((-1) ** (order - r)) * cell_eta[bits]
            bits_u = 0
            for j in idxs:
                bits_u |= (1 << j)
            rows.append((tuple(source_ids[j] for j in idxs), float(coef), float(cell_support[bits_u])))
    return rows


def _parts(seq: tuple[int, ...], max_block: int) -> Iterable[tuple[tuple[int, ...], ...]]:
    if not seq:
        yield ()
        return
    first = seq[0]
    for part in _parts(seq[1:], max_block=max_block):
        yield ((first,),) + part
        for i, block in enumerate(part):
            if len(block) >= max_block:
                continue
            new_block = tuple(sorted(block + (first,)))
            new_part = part[:i] + (new_block,) + part[i + 1 :]
            yield tuple(sorted(tuple(sorted(b)) for b in new_part))


def unique_partitions(seq: tuple[int, ...], max_block: int = 4) -> list[tuple[tuple[int, ...], ...]]:
    seen: set[tuple[tuple[int, ...], ...]] = set()
    out: list[tuple[tuple[int, ...], ...]] = []
    for part in _parts(seq, max_block=max_block):
        part = tuple(sorted(tuple(sorted(block)) for block in part))
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def build_partitions(
    mode: str,
    source_ids: tuple[int, ...],
    *,
    max_block: int,
    manual_partition_text: str | None = None,
) -> list[tuple[tuple[int, ...], ...]]:
    if mode == "manual":
        if manual_partition_text:
            return [parse_manual_partition(manual_partition_text)]
        return [((0, 1, 4, 5), (2, 3))]
    if mode == "singleton":
        return [tuple((s,) for s in source_ids)]
    if mode == "all_partitions":
        return unique_partitions(source_ids, max_block=max_block)
    raise ValueError(f"unknown partition mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixed_target", type=int, default=6)
    ap.add_argument("--partition_mode", choices=["manual", "singleton", "all_partitions"], default="manual")
    ap.add_argument("--max_block", type=int, default=4)
    ap.add_argument("--manual_partition", default=None)
    ap.add_argument("--topn", type=int, default=80)
    ap.add_argument("--thr_values", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--support_pow_values", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--order_bonus_values", default="0.0,0.1,0.2,0.5,1.0,2.0")
    ap.add_argument("--sup_pen_values", default="0.0,0.1,0.2,0.5,1.0,2.0")
    args = ap.parse_args()

    train_data, _, metadata = prepare_data(args.data)
    cfg = load_yaml(args.config)
    gt = gt_rules_from_config(cfg)
    num_types = int(metadata["num_types"])

    n_val = min(max(1, int(len(train_data) * 0.15)), len(train_data) - 1)
    val_data = train_data[:n_val]
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
        manual_partition_text=args.manual_partition,
    )
    print(f"partition_mode={args.partition_mode} num_partitions={len(components_list)}")
    thr_values = parse_csv_floats(args.thr_values)
    support_pow_values = parse_csv_floats(args.support_pow_values)
    order_bonus_values = parse_csv_floats(args.order_bonus_values)
    sup_pen_values = parse_csv_floats(args.sup_pen_values)

    results = []
    total = (
        len(components_list)
        * len(thr_values)
        * len(support_pow_values)
        * len(order_bonus_values)
        * len(sup_pen_values)
    )
    counter = 0
    for thr, support_pow, order_bonus, sup_pen in itertools.product(
        thr_values,
        support_pow_values,
        order_bonus_values,
        sup_pen_values,
    ):
        X = (tr_cache["event_q"] > float(thr)).astype(np.int64)
        y = tr_cache["event_y"].astype(np.int64)
        for components in components_list:
            base_rows = []
            for comp in components:
                idxs = [local_by_source[s] for s in comp]
                base_rows.extend(family_mobius(X[:, idxs], y, tuple(comp), max_order=3))

            scored = []
            for srcs, coef, support in base_rows:
                sign = "exc" if coef > 0.0 else "inh"
                score = abs(coef) * (max(support, 1.0) ** float(support_pow))
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
        key = (item[0], tuple(item[-2]), tuple(item[-1]))
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
                "thr": float(best[1]),
                "support_pow": float(best[2]),
                "order_bonus": float(best[3]),
                "sup_pen": float(best[4]),
                "components": best[5],
            },
        )
        print_rule_block("True rules:", sorted(gt), int(args.fixed_target))
        print_rule_block("Predicted rules:", sorted(preds), int(args.fixed_target))
        print_rule_block("Matched rules:", hit, int(args.fixed_target))
        print_rule_block("Missing rules:", miss, int(args.fixed_target))
        print_rule_block("Extra predicted rules:", extra, int(args.fixed_target))


if __name__ == "__main__":
    main()

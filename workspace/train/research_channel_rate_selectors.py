"""Research sweeps for latent-channel rate-based selectors on overlap synthetic data.

This prototype addresses a structural failure of the exact-cell selector:
for sign-conflict sources such as A and C, Phase 2 compresses multiple signed lag
lobes into a single scalar evidence q_s(t). Here we instead split each source into
at most one positive and one negative temporal channel, score source-subset rules by
the best channel combination, and use continuous-time target-rate contrasts rather
than event-time class logits.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import yaml

from workspace.train.ablate_component_basis import gt_rules_from_config, prepare_data
from workspace.train.component_basis_init import _estimate_base_rates, _raw_source_signal_at_queries, _sequence_lag_hist
from workspace.train.research_exact_cell_selectors import format_rule, parse_csv_floats, print_rule_block


@dataclass
class ChannelDef:
    source: int
    channel_id: int
    sign_prior: int
    score: float
    peak: float
    width: float
    beta: float
    alpha: float


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pair_excess_hist(
    data_list: list[dict],
    num_types: int,
    src: int,
    fixed_target: int,
    *,
    delta_t: float,
    max_lag: float,
) -> tuple[np.ndarray, np.ndarray]:
    rates, _ = _estimate_base_rates(data_list, num_types)
    bins = max(1, int(math.ceil(max_lag / max(delta_t, 1e-6))))
    hist = np.zeros((bins,), dtype=np.float64)
    total_target = 0
    for item in data_list:
        e = np.asarray(item["event"], dtype=np.int64)
        total_target += int(np.sum(e == int(fixed_target)))
    for item in data_list:
        t = np.asarray(item["time"], dtype=np.float64)
        e = np.asarray(item["event"], dtype=np.int64)
        if len(t) == 0:
            continue
        _sequence_lag_hist(
            t[e == int(src)],
            t[e == int(fixed_target)],
            max_lag,
            delta_t,
            hist,
        )
    expected = float(rates[int(src)]) * float(delta_t) * float(max(total_target, 1))
    centers = (np.arange(bins, dtype=np.float64) + 0.5) * delta_t
    excess = hist - expected
    return centers, excess


def strongest_signed_lobes(
    centers: np.ndarray,
    excess: np.ndarray,
    *,
    delta_t: float,
    max_lag: float,
    min_mass_frac: float = 0.01,
) -> list[dict]:
    total_mass = float(np.abs(excess).sum())
    if total_mass <= 1e-8:
        return []
    segs: list[dict] = []
    start = None
    seg_sign = 0
    for i, val in enumerate(excess):
        cur_sign = 1 if val > 0.0 else (-1 if val < 0.0 else 0)
        if cur_sign == 0:
            if start is not None:
                segs.append({"sign": seg_sign, "start": start, "end": i - 1})
                start = None
                seg_sign = 0
            continue
        if start is None:
            start = i
            seg_sign = cur_sign
            continue
        if cur_sign != seg_sign:
            segs.append({"sign": seg_sign, "start": start, "end": i - 1})
            start = i
            seg_sign = cur_sign
    if start is not None:
        segs.append({"sign": seg_sign, "start": start, "end": len(excess) - 1})

    out: list[dict] = []
    for sign in (-1, 1):
        best = None
        for seg in segs:
            if int(seg["sign"]) != sign:
                continue
            sl = slice(int(seg["start"]), int(seg["end"]) + 1)
            weights = np.abs(excess[sl])
            mass = float(weights.sum())
            if mass < min_mass_frac * total_mass:
                continue
            mu = float((centers[sl] * weights).sum() / max(mass, 1e-8))
            var = float((((centers[sl] - mu) ** 2) * weights).sum() / max(mass, 1e-8))
            sigma = max(math.sqrt(max(var, 1e-8)), delta_t)
            peak = float(np.clip(mu, delta_t, max_lag * 0.95))
            width = float(np.clip(mu + 2.0 * sigma, peak + delta_t, max_lag))
            row = {
                "sign_prior": int(sign),
                "score": mass,
                "peak": peak,
                "width": width,
            }
            if best is None or float(row["score"]) > float(best["score"]):
                best = row
        if best is not None:
            out.append(best)
    return out


def phase1_multichannel_screen(
    data_list: list[dict],
    num_types: int,
    fixed_target: int,
    *,
    delta_t: float,
    max_lag: float,
    source_pool_topk: int,
    min_mass_frac: float,
) -> dict:
    rates, total_time = _estimate_base_rates(data_list, num_types)
    pair_scores = []
    for src in range(num_types):
        if src == int(fixed_target):
            continue
        _, excess = pair_excess_hist(
            data_list,
            num_types,
            int(src),
            int(fixed_target),
            delta_t=delta_t,
            max_lag=max_lag,
        )
        pair_scores.append({"source": int(src), "score": float(np.abs(excess).sum())})
    pair_scores.sort(key=lambda x: float(x["score"]), reverse=True)
    if source_pool_topk > 0:
        pair_scores = pair_scores[: min(int(source_pool_topk), len(pair_scores))]

    channels: list[ChannelDef] = []
    next_id = 0
    for row in pair_scores:
        src = int(row["source"])
        centers, excess = pair_excess_hist(
            data_list,
            num_types,
            src,
            int(fixed_target),
            delta_t=delta_t,
            max_lag=max_lag,
        )
        lobes = strongest_signed_lobes(
            centers,
            excess,
            delta_t=delta_t,
            max_lag=max_lag,
            min_mass_frac=float(min_mass_frac),
        )
        for lobe in lobes:
            channels.append(
                ChannelDef(
                    source=src,
                    channel_id=next_id,
                    sign_prior=int(lobe["sign_prior"]),
                    score=float(lobe["score"]),
                    peak=float(lobe["peak"]),
                    width=float(lobe["width"]),
                    beta=0.0,
                    alpha=1.0,
                )
            )
            next_id += 1

    return {
        "base_rates": rates,
        "total_time": total_time,
        "pair_scores": pair_scores,
        "channels": channels,
    }


def estimate_channel_bounding(
    data_list: list[dict],
    channels: list[ChannelDef],
    *,
    fixed_target: int,
    beta_quantile: float = 0.6,
) -> list[ChannelDef]:
    out = []
    for ch in channels:
        vals = []
        for item in data_list:
            t = np.asarray(item["time"], dtype=np.float64)
            e = np.asarray(item["event"], dtype=np.int64)
            if len(t) == 0:
                continue
            src_times = t[e == int(ch.source)]
            tgt_times = t[e == int(fixed_target)]
            if len(src_times) == 0 or len(tgt_times) == 0:
                continue
            z = _raw_source_signal_at_queries(tgt_times, src_times, ch.peak, ch.width)
            if len(z) > 0:
                vals.append(z.astype(np.float64))
        if vals:
            z_all = np.concatenate(vals)
            beta = float(np.quantile(z_all, beta_quantile))
            above = z_all[z_all > beta + 1e-6]
            if len(above) > 0:
                med = float(np.median(above))
                alpha = float(np.log(2.0) / max(med - beta, 1e-3))
            else:
                alpha = 1.0
        else:
            beta = 0.25
            alpha = 1.0
        out.append(
            ChannelDef(
                source=int(ch.source),
                channel_id=int(ch.channel_id),
                sign_prior=int(ch.sign_prior),
                score=float(ch.score),
                peak=float(ch.peak),
                width=float(ch.width),
                beta=float(beta),
                alpha=float(alpha),
            )
        )
    return out


def build_channel_cache(
    data_list: list[dict],
    channels: list[ChannelDef],
    *,
    fixed_target: int,
    num_types: int,
    int_grid_mult: int = 1,
) -> dict:
    from workspace.train.component_basis_init import _build_query_grid

    rows_q: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    grid_q_rows: list[np.ndarray] = []
    grid_w_rows: list[np.ndarray] = []
    grid_times_list, grid_weights_list, _ = _build_query_grid(data_list, int_grid_mult=int_grid_mult)
    seq_local = 0

    def q_matrix(query_times: np.ndarray, event_times: np.ndarray, event_types: np.ndarray) -> np.ndarray:
        q = np.zeros((len(query_times), len(channels)), dtype=np.float32)
        for j, ch in enumerate(channels):
            src_times = event_times[event_types == int(ch.source)]
            if len(src_times) == 0:
                continue
            z = _raw_source_signal_at_queries(query_times, src_times, ch.peak, ch.width)
            q[:, j] = 1.0 - np.exp(-ch.alpha * np.maximum(z - ch.beta, 0.0))
        return np.clip(q, 0.0, 1.0)

    for item in data_list:
        times = np.asarray(item["time"], dtype=np.float64)
        events = np.asarray(item["event"], dtype=np.int64)
        if len(times) == 0:
            continue
        rows_q.append(q_matrix(times, times, events))
        rows_y.append((events == int(fixed_target)).astype(np.float32))
        g_t = grid_times_list[seq_local]
        g_w = grid_weights_list[seq_local]
        grid_q_rows.append(q_matrix(g_t, times, events))
        grid_w_rows.append(g_w.astype(np.float32))
        seq_local += 1

    if rows_q:
        event_q = np.concatenate(rows_q, axis=0).astype(np.float32)
        event_y = np.concatenate(rows_y, axis=0).astype(np.float32)
    else:
        event_q = np.zeros((1, len(channels)), dtype=np.float32)
        event_y = np.zeros((1,), dtype=np.float32)
    if grid_q_rows:
        grid_q = np.concatenate(grid_q_rows, axis=0).astype(np.float32)
        grid_w = np.concatenate(grid_w_rows, axis=0).astype(np.float32)
    else:
        grid_q = np.zeros((1, len(channels)), dtype=np.float32)
        grid_w = np.ones((1,), dtype=np.float32)
    return {
        "event_q": event_q,
        "event_y": event_y,
        "grid_q": grid_q,
        "grid_w": grid_w,
        "fixed_target": int(fixed_target),
        "num_types": int(num_types),
    }


def binary_rate_parent_delta(
    event_q: np.ndarray,
    event_y: np.ndarray,
    grid_q: np.ndarray,
    grid_w: np.ndarray,
    channel_idxs: tuple[int, ...],
    thr: float,
) -> tuple[float, float]:
    d = len(channel_idxs)
    Xev = (event_q[:, channel_idxs] > float(thr)).astype(np.int8)
    Xgr = (grid_q[:, channel_idxs] > float(thr)).astype(np.int8)
    eta = np.zeros((1 << d,), dtype=np.float64)
    support = np.zeros((1 << d,), dtype=np.float64)
    for bits in range(1 << d):
        mev = np.ones((Xev.shape[0],), dtype=bool)
        mgr = np.ones((Xgr.shape[0],), dtype=bool)
        for j in range(d):
            bit = (bits >> j) & 1
            mev &= (Xev[:, j] == bit)
            mgr &= (Xgr[:, j] == bit)
        n_t = float(event_y[mev].sum())
        t_w = float(grid_w[mgr].sum())
        support[bits] = t_w
        eta[bits] = math.log((n_t + 1e-6) / max(t_w + 1e-6, 1e-6))

    bits_u = (1 << d) - 1
    if d == 1:
        sup = min(float(support[bits_u]), float(support[0]))
        return float(eta[bits_u] - eta[0]), float(sup)

    best = None
    for anchor in range(d):
        parent_bits = bits_u & ~(1 << anchor)
        sup = min(float(support[bits_u]), float(support[parent_bits]))
        if sup <= 0.0:
            continue
        delta = float(eta[bits_u] - eta[parent_bits])
        cand = (delta, sup)
        if best is None or abs(delta) * sup > abs(best[0]) * best[1]:
            best = cand
    if best is None:
        return 0.0, 0.0
    return float(best[0]), float(best[1])


def source_subset_candidates(
    channel_cache_tr: dict,
    channel_cache_va: dict,
    channels: list[ChannelDef],
    source_ids: tuple[int, ...],
    *,
    thr: float,
    support_pow: float,
) -> list[dict]:
    by_source: dict[int, list[int]] = defaultdict(list)
    for j, ch in enumerate(channels):
        by_source[int(ch.source)].append(int(j))

    rows = []
    for order in (1, 2, 3):
        for srcs in itertools.combinations(source_ids, order):
            combos = itertools.product(*[by_source[int(s)] for s in srcs])
            best_by_sign = {}
            for combo in combos:
                if len(set(int(channels[idx].source) for idx in combo)) != order:
                    continue
                delta_tr, sup_tr = binary_rate_parent_delta(
                    channel_cache_tr["event_q"],
                    channel_cache_tr["event_y"],
                    channel_cache_tr["grid_q"],
                    channel_cache_tr["grid_w"],
                    tuple(int(i) for i in combo),
                    thr,
                )
                delta_va, sup_va = binary_rate_parent_delta(
                    channel_cache_va["event_q"],
                    channel_cache_va["event_y"],
                    channel_cache_va["grid_q"],
                    channel_cache_va["grid_w"],
                    tuple(int(i) for i in combo),
                    thr,
                )
                if delta_tr == 0.0 or delta_va == 0.0 or np.sign(delta_tr) != np.sign(delta_va):
                    continue
                sign = "exc" if delta_tr > 0.0 else "inh"
                coef = float(np.sign(delta_tr) * min(abs(delta_tr), abs(delta_va)))
                sup = float(min(sup_tr, sup_va))
                score = abs(coef) * (max(sup, 1.0) ** float(support_pow))
                cur = best_by_sign.get(sign)
                if cur is None or float(score) > float(cur["score"]):
                    best_by_sign[sign] = {
                        "srcs": tuple(int(s) for s in srcs),
                        "sign": sign,
                        "score": float(score),
                        "coef": float(coef),
                        "support": float(sup),
                        "combo": tuple(int(i) for i in combo),
                    }
            rows.extend(best_by_sign.values())
    return rows


def adjusted_scores(rows: list[dict], shadow_pen: float) -> list[dict]:
    out = []
    for row in rows:
        src_set = set(row["srcs"])
        sup_shadow = 0.0
        for row2 in rows:
            if row["sign"] != row2["sign"]:
                continue
            if len(row2["srcs"]) <= len(row["srcs"]):
                continue
            if src_set < set(row2["srcs"]):
                sup_shadow = max(sup_shadow, float(row2["score"]))
        out.append({**row, "adj_score": float(row["score"] - float(shadow_pen) * sup_shadow)})
    return out


def select_with_quota(rows: list[dict], fixed_target: int, quota: tuple[int, int, int]) -> tuple[tuple[tuple[int, ...], str, int], ...]:
    chosen = []
    for order, k in enumerate(quota, start=1):
        if int(k) <= 0:
            continue
        pool = [r for r in rows if len(r["srcs"]) == int(order)]
        pool.sort(key=lambda r: (float(r["adj_score"]), float(r["score"]), abs(float(r["coef"]))), reverse=True)
        seen_lhs = set()
        count = 0
        for row in pool:
            lhs = tuple(row["srcs"])
            if lhs in seen_lhs:
                continue
            seen_lhs.add(lhs)
            chosen.append((lhs, str(row["sign"]), int(fixed_target)))
            count += 1
            if count >= int(k):
                break
    return tuple(sorted(set(chosen)))


def print_ranked_block(title: str, rows: list[dict], fixed_target: int):
    print(title)
    if not rows:
        print("  - none")
        return
    for row in rows:
        rule = (tuple(row["srcs"]), row["sign"], int(fixed_target))
        print(
            f"  - {format_rule(rule, fixed_target)} "
            f"[adj={row['adj_score']:.6f}, score={row['score']:.6f}, coef={row['coef']:.6f}, "
            f"support={row['support']:.3f}, combo={row['combo']}]"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixed_target", type=int, default=6)
    ap.add_argument("--delta_t", type=float, default=0.1)
    ap.add_argument("--max_lag", type=float, default=10.0)
    ap.add_argument("--source_pool_topk", type=int, default=8)
    ap.add_argument("--min_mass_frac", type=float, default=0.01)
    ap.add_argument("--thr_values", default="0.1,0.12,0.15,0.2,0.25")
    ap.add_argument("--support_pow_values", default="0.5,1.0,1.25,1.5")
    ap.add_argument("--shadow_pen_values", default="0.0,0.25,0.5,1.0,2.0")
    ap.add_argument("--topn", type=int, default=30)
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
    )
    channels = estimate_channel_bounding(
        tr_data,
        phase1["channels"],
        fixed_target=int(args.fixed_target),
    )
    tr_cache = build_channel_cache(
        tr_data,
        channels,
        fixed_target=int(args.fixed_target),
        num_types=num_types,
        int_grid_mult=1,
    )
    va_cache = build_channel_cache(
        va_data,
        channels,
        fixed_target=int(args.fixed_target),
        num_types=num_types,
        int_grid_mult=1,
    )
    source_ids = tuple(sorted({int(ch.source) for ch in channels}))
    print("Source pool:", list(source_ids))
    print("Channels:")
    for ch in channels:
        print(
            f"  ch={ch.channel_id} src={ch.source} sign_prior={ch.sign_prior:+d} "
            f"score={ch.score:.3f} peak={ch.peak:.3f} width={ch.width:.3f} "
            f"beta={ch.beta:.3f} alpha={ch.alpha:.3f}"
        )

    thr_values = parse_csv_floats(args.thr_values)
    support_pow_values = parse_csv_floats(args.support_pow_values)
    shadow_pen_values = parse_csv_floats(args.shadow_pen_values)
    quota_values = [(k1, k2, k3) for k1 in range(len(gt) + 1) for k2 in range(len(gt) + 1) for k3 in range(len(gt) + 1) if k1 + k2 + k3 == len(gt)]

    base_rows_by_thr = {}
    for thr in thr_values:
        base_rows_by_thr[float(thr)] = source_subset_candidates(
            tr_cache,
            va_cache,
            channels,
            source_ids,
            thr=float(thr),
            support_pow=1.0,
        )

    results = []
    for thr, support_pow, shadow_pen, quota in itertools.product(
        thr_values,
        support_pow_values,
        shadow_pen_values,
        quota_values,
    ):
        rows = []
        for row in base_rows_by_thr[float(thr)]:
            rows.append({**row, "score": abs(float(row["coef"])) * (max(float(row["support"]), 1.0) ** float(support_pow))})
        rows = adjusted_scores(rows, shadow_pen=float(shadow_pen))
        pred = select_with_quota(rows, int(args.fixed_target), quota)
        results.append(
            (
                len(gt & set(pred)),
                float(thr),
                float(support_pow),
                float(shadow_pen),
                quota,
                pred,
                rows,
            )
        )

    results.sort(reverse=True)
    seen = set()
    for item in results[: args.topn]:
        key = (item[0], item[1], item[2], item[3], item[4], item[5])
        if key in seen:
            continue
        seen.add(key)
        print(item[:6])

    if results:
        best = results[0]
        preds = set(best[5])
        hit = sorted(gt & preds)
        miss = sorted(gt - preds)
        extra = sorted(preds - gt)
        selected_rows = [
            row for row in best[6]
            if (tuple(row["srcs"]), row["sign"], int(args.fixed_target)) in preds
        ]
        selected_rows.sort(key=lambda r: (float(r["adj_score"]), float(r["score"]), abs(float(r["coef"]))), reverse=True)
        print("===BEST REPORT===")
        print(
            "best_params:",
            {
                "hit_count": int(best[0]),
                "thr": float(best[1]),
                "support_pow": float(best[2]),
                "shadow_pen": float(best[3]),
                "quota": best[4],
            },
        )
        print_rule_block("True rules:", sorted(gt), int(args.fixed_target))
        print_ranked_block(f"Predicted top-{len(gt)} rules:", selected_rows, int(args.fixed_target))
        print_rule_block("Matched rules:", hit, int(args.fixed_target))
        print_rule_block("Missing rules:", miss, int(args.fixed_target))
        print_rule_block("Extra predicted rules:", extra, int(args.fixed_target))


if __name__ == "__main__":
    main()

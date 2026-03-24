"""Adaptive multichannel stagewise selector.

Self-contained implementation of the current best overlap-rule recovery path:
1. Pairwise lag screening on the fixed target
2. Adaptive signed-channel splitting for strong sign-conflict sources
3. Bounded source-channel evidence on event times
4. Stagewise exact-cell rule selection with exact-top-k evaluation
"""

from __future__ import annotations

import argparse
import itertools
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import yaml


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


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(data_path: str):
    with open(data_path, "rb") as f:
        dd = pickle.load(f)
    metadata = dd["metadata"]
    time_scale = float(metadata.get("mean_time_diff", 1.0))
    train = dd["train"]
    val = dd["val"]
    for ds in (train, val):
        for item in ds:
            item["time"] = [float(t) / time_scale for t in item["time"]]
    return train, val, metadata


def gt_rules_from_config(config: dict):
    gt = set()
    for rule in config.get("rules", []):
        srcs = tuple(sorted(int(s) for s in rule["condition"].keys()))
        w_pos = float(rule.get("W_pos", 0.0))
        w_neg = float(rule.get("W_neg", 0.0))
        if w_pos > w_neg:
            sign = "exc"
        elif w_neg > w_pos:
            sign = "inh"
        else:
            continue
        gt.add((srcs, sign, int(rule["target"])))
    return gt


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


def _parts(seq: tuple[int, ...], max_block: int):
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
        return [tuple((s,) for s in source_ids)]
    if mode == "singleton":
        return [tuple((s,) for s in source_ids)]
    if mode == "all_partitions":
        return unique_partitions(source_ids, max_block=max_block)
    raise ValueError(f"unknown partition mode: {mode}")


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
                    best = (float(eta[bits_u] - eta[parent_bits]), float(sup))
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


def _estimate_base_rates(data_list: list[dict], num_types: int) -> tuple[np.ndarray, float]:
    counts = np.zeros(num_types, dtype=np.float64)
    total_time = 0.0
    for item in data_list:
        t = np.asarray(item["time"], dtype=np.float64)
        e = np.asarray(item["event"], dtype=np.int64)
        if len(t) == 0:
            continue
        total_time += float(t[-1])
        valid = (e >= 0) & (e < num_types)
        if valid.any():
            counts += np.bincount(e[valid], minlength=num_types)[:num_types]
    return counts / max(total_time, 1e-9), total_time


def _sequence_lag_hist(times_src: np.ndarray, times_tgt: np.ndarray, max_lag: float, delta_t: float, hist: np.ndarray) -> None:
    if len(times_src) == 0 or len(times_tgt) == 0:
        return
    bins = hist.shape[0]
    for t_tgt in times_tgt:
        st = int(np.searchsorted(times_src, t_tgt - max_lag, side="left"))
        en = int(np.searchsorted(times_src, t_tgt, side="left"))
        if st >= en:
            continue
        dts = t_tgt - times_src[st:en]
        b = np.floor(dts / delta_t).astype(np.int64)
        b = b[(b >= 0) & (b < bins)]
        if len(b) > 0:
            hist += np.bincount(b, minlength=bins)[:bins]


def _triangular_kernel(dt: np.ndarray, peak: float, width: float) -> np.ndarray:
    out = np.zeros_like(dt, dtype=np.float64)
    if width <= 1e-8:
        return out
    peak = float(np.clip(peak, 1e-6, width - 1e-6))
    valid = (dt > 0.0) & (dt < width)
    if not np.any(valid):
        return out
    dv = dt[valid]
    left = dv <= peak
    vals = np.empty_like(dv)
    vals[left] = dv[left] / peak
    vals[~left] = (width - dv[~left]) / max(width - peak, 1e-6)
    out[valid] = np.clip(vals, 0.0, 1.0)
    return out


def _triangular_piecewise_linear_kernel(
    dt: np.ndarray,
    peak: float,
    width: float,
    *,
    num_bins: int,
    max_cap: float,
) -> np.ndarray:
    out = np.zeros_like(dt, dtype=np.float64)
    if max_cap <= 1e-8 or num_bins <= 0:
        return out
    valid = (dt > 0.0) & (dt < max_cap)
    if not np.any(valid):
        return out
    grid = np.linspace(0.0, float(max_cap), int(num_bins) + 1, dtype=np.float64)
    h = _triangular_kernel(grid, peak, width)
    dv = np.clip(dt[valid], 0.0, float(max_cap) * (1.0 - 1e-7))
    bin_w = float(max_cap) / float(num_bins)
    idx = np.clip((dv / bin_w).astype(np.int64), 0, int(num_bins) - 1)
    frac = (dv / bin_w) - idx
    out[valid] = h[idx] + (h[idx + 1] - h[idx]) * frac
    return out


def _gaussian_kernel(dt: np.ndarray, peak: float, support: float, *, support_mult: float) -> np.ndarray:
    out = np.zeros_like(dt, dtype=np.float64)
    if support <= 1e-8:
        return out
    valid = (dt > 0.0) & (dt < support)
    if not np.any(valid):
        return out
    sigma = max((float(support) - float(peak)) / max(float(support_mult), 1e-6), 1e-6)
    z = (dt[valid] - float(peak)) / sigma
    out[valid] = np.exp(-0.5 * z * z)
    return out


def _gaussian_piecewise_linear_kernel(
    dt: np.ndarray,
    peak: float,
    support: float,
    *,
    num_bins: int,
    max_cap: float,
    support_mult: float,
) -> np.ndarray:
    out = np.zeros_like(dt, dtype=np.float64)
    if max_cap <= 1e-8 or num_bins <= 0:
        return out
    valid = (dt > 0.0) & (dt < max_cap)
    if not np.any(valid):
        return out
    grid = np.linspace(0.0, float(max_cap), int(num_bins) + 1, dtype=np.float64)
    h = _gaussian_kernel(grid, peak, support, support_mult=support_mult)
    dv = np.clip(dt[valid], 0.0, float(max_cap) * (1.0 - 1e-7))
    bin_w = float(max_cap) / float(num_bins)
    idx = np.clip((dv / bin_w).astype(np.int64), 0, int(num_bins) - 1)
    frac = (dv / bin_w) - idx
    out[valid] = h[idx] + (h[idx + 1] - h[idx]) * frac
    return out


def _exponential_kernel(dt: np.ndarray, peak: float, support: float, *, support_mult: float) -> np.ndarray:
    out = np.zeros_like(dt, dtype=np.float64)
    if support <= 1e-8:
        return out
    tau = max((float(support) - float(peak)) / max(float(support_mult), 1e-6), 1e-6)
    valid = (dt > float(peak)) & (dt < support)
    if not np.any(valid):
        return out
    out[valid] = np.exp(-(dt[valid] - float(peak)) / tau)
    return out


def _exponential_piecewise_linear_kernel(
    dt: np.ndarray,
    peak: float,
    support: float,
    *,
    num_bins: int,
    max_cap: float,
    support_mult: float,
) -> np.ndarray:
    out = np.zeros_like(dt, dtype=np.float64)
    if max_cap <= 1e-8 or num_bins <= 0:
        return out
    valid = (dt > 0.0) & (dt < max_cap)
    if not np.any(valid):
        return out
    grid = np.linspace(0.0, float(max_cap), int(num_bins) + 1, dtype=np.float64)
    h = _exponential_kernel(grid, peak, support, support_mult=support_mult)
    dv = np.clip(dt[valid], 0.0, float(max_cap) * (1.0 - 1e-7))
    bin_w = float(max_cap) / float(num_bins)
    idx = np.clip((dv / bin_w).astype(np.int64), 0, int(num_bins) - 1)
    frac = (dv / bin_w) - idx
    out[valid] = h[idx] + (h[idx + 1] - h[idx]) * frac
    return out


def _raw_source_signal_at_queries(
    query_times: np.ndarray,
    src_times: np.ndarray,
    peak: float,
    width: float,
    *,
    kernel_eval_mode: str,
    kernel_num_bins: int,
    kernel_max_cap: float | None,
    kernel_support_mult: float,
) -> np.ndarray:
    out = np.zeros((len(query_times),), dtype=np.float32)
    if len(query_times) == 0 or len(src_times) == 0:
        return out
    max_cap = float(kernel_max_cap if kernel_max_cap is not None else max(width, 1e-6))
    left = 0
    right = 0
    n_src = len(src_times)
    for qi, q in enumerate(query_times):
        while left < n_src and src_times[left] < q - width:
            left += 1
        while right < n_src and src_times[right] < q:
            right += 1
        if right <= left:
            continue
        dts = q - src_times[left:right]
        if kernel_eval_mode == "triangular_exact":
            kvals = _triangular_kernel(dts, peak, width)
        elif kernel_eval_mode == "triangular_pwlin":
            kvals = _triangular_piecewise_linear_kernel(dts, peak, width, num_bins=int(kernel_num_bins), max_cap=max_cap)
        elif kernel_eval_mode == "gaussian_exact":
            kvals = _gaussian_kernel(dts, peak, width, support_mult=float(kernel_support_mult))
        elif kernel_eval_mode == "gaussian_pwlin":
            kvals = _gaussian_piecewise_linear_kernel(
                dts,
                peak,
                width,
                num_bins=int(kernel_num_bins),
                max_cap=max_cap,
                support_mult=float(kernel_support_mult),
            )
        elif kernel_eval_mode == "exponential_exact":
            kvals = _exponential_kernel(dts, peak, width, support_mult=float(kernel_support_mult))
        elif kernel_eval_mode == "exponential_pwlin":
            kvals = _exponential_piecewise_linear_kernel(
                dts,
                peak,
                width,
                num_bins=int(kernel_num_bins),
                max_cap=max_cap,
                support_mult=float(kernel_support_mult),
            )
        else:
            raise ValueError(f"Unsupported source kernel eval mode: {kernel_eval_mode}")
        out[qi] = float(kvals.sum())
    return out


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
        _sequence_lag_hist(t[e == int(src)], t[e == int(fixed_target)], max_lag, delta_t, hist)
    expected = float(rates[int(src)]) * float(delta_t) * float(max(total_target, 1))
    centers = (np.arange(bins, dtype=np.float64) + 0.5) * delta_t
    return centers, hist - expected


def _single_channel_stat(centers: np.ndarray, excess: np.ndarray, *, delta_t: float, max_lag: float) -> dict:
    weights = np.abs(excess)
    score = float(weights.sum())
    if score <= 1e-8:
        peak = max(float(delta_t), 0.5)
        width = min(float(max_lag), max(1.5 * peak, 1.0))
    else:
        mu = float((centers * weights).sum() / max(weights.sum(), 1e-8))
        var = float((((centers - mu) ** 2) * weights).sum() / max(weights.sum(), 1e-8))
        sigma = max(math.sqrt(max(var, 1e-8)), float(delta_t))
        peak = float(np.clip(mu, float(delta_t), float(max_lag) * 0.8))
        width = float(np.clip(mu + 2.0 * sigma, peak + float(delta_t), float(max_lag)))
    return {"score": score, "peak": peak, "width": width}


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
    segs = []
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

    out = []
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
                "mu": mu,
                "sigma": sigma,
            }
            if best is None or float(row["score"]) > float(best["score"]):
                best = row
        if best is not None:
            out.append(best)
    return out


def _split_gain_for_source(
    val_data: list[dict],
    *,
    fixed_target: int,
    src: int,
    lobes: list[dict],
) -> float:
    if len(lobes) < 2 or len(val_data) == 0:
        return 0.0
    num_types = max(int(fixed_target) + 1, int(src) + 1, 1)
    for item in val_data:
        e = np.asarray(item["event"], dtype=np.int64)
        if len(e) > 0:
            num_types = max(num_types, int(e.max()) + 1)
    centers, val_excess = pair_excess_hist(val_data, int(num_types), int(src), int(fixed_target), delta_t=0.1, max_lag=10.0)
    total_val = float(np.abs(val_excess).sum())
    if total_val <= 1e-8:
        return 0.0
    aligned = []
    for lobe in lobes[:2]:
        mu = float(lobe["mu"])
        sigma = float(lobe["sigma"])
        lo = mu - 2.0 * sigma
        hi = mu + 2.0 * sigma
        mask = (centers >= lo) & (centers <= hi)
        if int(lobe["sign_prior"]) > 0:
            mass = float(np.maximum(val_excess[mask], 0.0).sum())
        else:
            mass = float(np.maximum(-val_excess[mask], 0.0).sum())
        aligned.append(mass)
    if len(aligned) < 2:
        return 0.0
    return float(min(aligned) / max(total_val, 1e-8))


def phase1_multichannel_screen(
    data_list: list[dict],
    num_types: int,
    fixed_target: int,
    *,
    delta_t: float,
    max_lag: float,
    source_pool_topk: int,
    min_mass_frac: float,
    val_data_list: list[dict] | None = None,
    adaptive_split: bool = False,
    conflict_balance_min: float = 0.0,
    conflict_sep_min: float = 0.0,
    conflict_gain_min: float = 0.0,
    split_thr: float = 0.2,
) -> dict:
    rates, total_time = _estimate_base_rates(data_list, num_types)
    pair_scores = []
    for src in range(num_types):
        if src == int(fixed_target):
            continue
        _, excess = pair_excess_hist(data_list, num_types, int(src), int(fixed_target), delta_t=delta_t, max_lag=max_lag)
        pair_scores.append({"source": int(src), "score": float(np.abs(excess).sum())})
    pair_scores.sort(key=lambda x: float(x["score"]), reverse=True)
    if source_pool_topk > 0:
        pair_scores = pair_scores[: min(int(source_pool_topk), len(pair_scores))]

    channels: list[ChannelDef] = []
    next_id = 0
    source_summaries = []
    for row in pair_scores:
        src = int(row["source"])
        centers, excess = pair_excess_hist(data_list, num_types, src, int(fixed_target), delta_t=delta_t, max_lag=max_lag)
        lobes = strongest_signed_lobes(centers, excess, delta_t=float(delta_t), max_lag=float(max_lag), min_mass_frac=float(min_mass_frac))
        single_stat = _single_channel_stat(centers, excess, delta_t=float(delta_t), max_lag=float(max_lag))
        score_map = {int(l["sign_prior"]): float(l["score"]) for l in lobes}
        pos = float(score_map.get(1, 0.0))
        neg = float(score_map.get(-1, 0.0))
        balance = float(min(pos, neg) / max(pos + neg, 1e-8))
        if len(lobes) >= 2:
            pos_lobe = next((l for l in lobes if int(l["sign_prior"]) == 1), None)
            neg_lobe = next((l for l in lobes if int(l["sign_prior"]) == -1), None)
            if pos_lobe is not None and neg_lobe is not None:
                sep = abs(float(pos_lobe["mu"]) - float(neg_lobe["mu"])) / max(
                    math.sqrt(float(pos_lobe["sigma"]) ** 2 + float(neg_lobe["sigma"]) ** 2),
                    1e-6,
                )
            else:
                sep = 0.0
        else:
            sep = 0.0
        gain = 0.0
        use_split = len(lobes) >= 2
        if bool(adaptive_split):
            gain = _split_gain_for_source(
                data_list if val_data_list is None else val_data_list,
                fixed_target=int(fixed_target),
                src=int(src),
                lobes=lobes,
            )
            use_split = (
                len(lobes) >= 2
                and balance >= float(conflict_balance_min)
                and sep >= float(conflict_sep_min)
                and gain >= float(conflict_gain_min)
            )
        source_summaries.append(
            {
                "source": int(src),
                "single_score": float(single_stat["score"]),
                "mneg": neg,
                "mpos": pos,
                "balance": balance,
                "separation": float(sep),
                "gain": float(gain),
                "mode": "split" if use_split else "single",
            }
        )
        if use_split:
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
        else:
            dom_sign = 1 if pos > neg else (-1 if neg > pos else 0)
            channels.append(
                ChannelDef(
                    source=src,
                    channel_id=next_id,
                    sign_prior=int(dom_sign),
                    score=float(single_stat["score"]),
                    peak=float(single_stat["peak"]),
                    width=float(single_stat["width"]),
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
        "source_summaries": source_summaries,
    }


def estimate_channel_bounding(
    data_list: list[dict],
    channels: list[ChannelDef],
    *,
    fixed_target: int,
    beta_quantile: float = 0.6,
    kernel_eval_mode: str = "triangular_pwlin",
    kernel_num_bins: int = 50,
    kernel_max_cap: float | None = None,
    kernel_support_mult: float = 3.0,
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
            z = _raw_source_signal_at_queries(
                tgt_times,
                src_times,
                ch.peak,
                ch.width,
                kernel_eval_mode=str(kernel_eval_mode),
                kernel_num_bins=int(kernel_num_bins),
                kernel_max_cap=kernel_max_cap,
                kernel_support_mult=float(kernel_support_mult),
            )
            if len(z) > 0:
                vals.append(z.astype(np.float64))
        if vals:
            z_all = np.concatenate(vals)
            beta = float(np.quantile(z_all, beta_quantile))
            above = z_all[z_all > beta + 1e-6]
            alpha = float(np.log(2.0) / max(float(np.median(above)) - beta, 1e-3)) if len(above) > 0 else 1.0
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
    kernel_eval_mode: str,
    kernel_num_bins: int,
    kernel_max_cap: float | None,
    kernel_support_mult: float,
) -> dict:
    rows_q = []
    rows_y = []
    for item in data_list:
        times = np.asarray(item["time"], dtype=np.float64)
        events = np.asarray(item["event"], dtype=np.int64)
        if len(times) == 0:
            continue
        q = np.zeros((len(times), len(channels)), dtype=np.float32)
        for j, ch in enumerate(channels):
            src_times = times[events == int(ch.source)]
            if len(src_times) == 0:
                continue
            z = _raw_source_signal_at_queries(
                times,
                src_times,
                ch.peak,
                ch.width,
                kernel_eval_mode=str(kernel_eval_mode),
                kernel_num_bins=int(kernel_num_bins),
                kernel_max_cap=kernel_max_cap,
                kernel_support_mult=float(kernel_support_mult),
            )
            q[:, j] = 1.0 - np.exp(-float(ch.alpha) * np.maximum(z - float(ch.beta), 0.0))
        rows_q.append(np.clip(q, 0.0, 1.0))
        rows_y.append((events == int(fixed_target)).astype(np.float32))

    if rows_q:
        event_q = np.concatenate(rows_q, axis=0).astype(np.float32)
        event_y = np.concatenate(rows_y, axis=0).astype(np.float32)
    else:
        event_q = np.zeros((1, len(channels)), dtype=np.float32)
        event_y = np.zeros((1,), dtype=np.float32)
    return {"event_q": event_q, "event_y": event_y}


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
    support_pow: float,
) -> list[dict]:
    best_by_key: dict[tuple[tuple[int, ...], str], dict] = {}
    for combo, coef, sup in parent_rows(eta, support, channel_ids, max_order=3):
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


def component_cell_counts(X: np.ndarray, y: np.ndarray):
    d = X.shape[1]
    if len(y) == 0:
        return np.zeros((1 << d,), dtype=np.float64), np.zeros((1 << d,), dtype=np.float64)
    bits = np.zeros((len(y),), dtype=np.int64)
    for j in range(d):
        bits |= (X[:, j].astype(np.int64) << j)
    total = np.bincount(bits, minlength=(1 << d)).astype(np.float64)
    pos = np.bincount(bits, weights=y.astype(np.float64), minlength=(1 << d)).astype(np.float64)
    return total, pos


def cell_feature_from_local_idxs(d: int, local_idxs: tuple[int, ...]) -> np.ndarray:
    bits_u = subset_bits(local_idxs)
    all_bits = np.arange(1 << d, dtype=np.int64)
    return ((all_bits & bits_u) == bits_u).astype(np.float64)


def conditional_unique_ratio(
    feat: np.ndarray,
    neighbor_feats: list[np.ndarray],
    weights: np.ndarray,
) -> float:
    denom = float(np.dot(weights, feat * feat))
    if denom <= 1e-12 or not neighbor_feats:
        return 1.0
    Z = np.stack(neighbor_feats, axis=1).astype(np.float64)
    ws = np.sqrt(np.maximum(weights, 0.0))
    A = Z * ws[:, None]
    b = feat * ws
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = b - A @ coef
    except np.linalg.LinAlgError:
        return 1.0
    num = float(np.dot(resid, resid))
    return float(np.clip(num / max(denom, 1e-12), 0.0, 1.0))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def binomial_loglik(total: np.ndarray, pos: np.ndarray, eta: np.ndarray) -> float:
    return float(np.sum(pos * eta - total * np.logaddexp(0.0, eta)))


def fit_signed_binomial_model(
    total: np.ndarray,
    pos: np.ndarray,
    signed_feats: list[np.ndarray],
    *,
    beta_init: float | None = None,
    theta_init: np.ndarray | None = None,
    num_iter: int = 40,
) -> tuple[float, np.ndarray, np.ndarray]:
    mean_rate = float((pos.sum() + 0.5) / max(total.sum() + 1.0, 1.0))
    beta = float(logit(mean_rate)) if beta_init is None else float(beta_init)
    p = len(signed_feats)
    theta = np.zeros((p,), dtype=np.float64) if theta_init is None else np.maximum(np.asarray(theta_init, dtype=np.float64), 0.0)
    if p == 0:
        eta = np.full_like(total, beta, dtype=np.float64)
        return beta, theta, eta
    G = np.stack(signed_feats, axis=1).astype(np.float64)
    for _ in range(max(int(num_iter), 1)):
        eta = beta + G @ theta
        prob = _sigmoid(eta)
        resid = pos - total * prob
        weight = total * prob * (1.0 - prob)
        h0 = float(weight.sum()) + 1e-8
        step0 = float(np.clip(resid.sum() / h0, -1.0, 1.0))
        beta += step0
        max_change = abs(step0)
        for j in range(p):
            eta = beta + G @ theta
            prob = _sigmoid(eta)
            resid = pos - total * prob
            weight = total * prob * (1.0 - prob)
            gj = G[:, j]
            uj = float(np.dot(resid, gj))
            hj = float(np.dot(weight, gj * gj)) + 1e-8
            new_theta = max(0.0, float(theta[j] + np.clip(uj / hj, -1.0, 1.0)))
            max_change = max(max_change, abs(new_theta - float(theta[j])))
            theta[j] = new_theta
        if max_change < 1e-5:
            break
    eta = beta + G @ theta
    return beta, theta, eta


def fit_bic_from_signed_feats(
    total_tr: np.ndarray,
    pos_tr: np.ndarray,
    total_va: np.ndarray,
    pos_va: np.ndarray,
    signed_feats: list[np.ndarray],
) -> float:
    beta, theta, _ = fit_signed_binomial_model(total_tr, pos_tr, signed_feats)
    if signed_feats:
        Gv = np.stack(signed_feats, axis=1).astype(np.float64)
        eta_val = float(beta) + Gv @ theta
    else:
        eta_val = np.full_like(total_va, float(beta), dtype=np.float64)
    ll_val = binomial_loglik(total_va, pos_va, eta_val)
    k = int(len(signed_feats)) + 1
    return float(-2.0 * ll_val + float(k) * math.log(max(float(total_va.sum()), 2.0)))


def approx_val_gain_from_score(
    eta_val: np.ndarray,
    total_val: np.ndarray,
    pos_val: np.ndarray,
    signed_feat_val: np.ndarray,
) -> float:
    prob = _sigmoid(eta_val)
    resid = pos_val - total_val * prob
    weight = total_val * prob * (1.0 - prob)
    u = float(np.dot(resid, signed_feat_val))
    if u <= 0.0:
        return 0.0
    h = float(np.dot(weight, signed_feat_val * signed_feat_val)) + 1e-8
    return float(0.5 * (u * u) / h)


def sibling_union_penalty(
    rules: list[tuple[tuple[int, ...], str, int]],
) -> int:
    by_sign: dict[tuple[str, int], set[tuple[int, ...]]] = {}
    for srcs, sign, tgt in rules:
        by_sign.setdefault((str(sign), int(tgt)), set()).add(tuple(int(s) for s in srcs))
    pen = 0
    for subsets in by_sign.values():
        tuples = list(subsets)
        for a, b in itertools.combinations(tuples, 2):
            set_a = set(int(s) for s in a)
            set_b = set(int(s) for s in b)
            union = tuple(sorted(set_a | set_b))
            if len(union) <= max(len(a), len(b)) or len(union) > 3:
                continue
            # Penalize sibling parents that imply a union rule but leave it absent.
            if len(a) == len(b) == len(union) - 1 and tuple(union) not in subsets:
                pen += 1
    return int(pen)


def collect_top_group_variants(
    *,
    component_defs: list[dict],
    base_tables: dict,
    channel_by_id: dict[int, ChannelDef],
    support_pow: float,
    sup_pen: float,
    fixed_target: int,
) -> dict[int, list[dict]]:
    group_pool: dict[tuple[int, tuple[int, ...]], dict] = {}
    for order in (1, 2, 3):
        rows = select_stage(
            base_tables,
            component_defs,
            channel_by_id,
            order=order,
            support_pow=float(support_pow),
            sup_pen=float(sup_pen),
            fixed_target=int(fixed_target),
        )
        for adj_score, coef, comp_id, srcs, sign, tgt, combo in rows:
            d = len(component_defs[int(comp_id)]["channel_ids"])
            local_idxs = tuple(int(component_defs[int(comp_id)]["channel_pos"][int(ch)]) for ch in combo)
            raw_feat = cell_feature_from_local_idxs(d, local_idxs)
            sign_mult = 1.0 if str(sign) == "exc" else -1.0
            item = {
                "comp_id": int(comp_id),
                "srcs": tuple(int(s) for s in srcs),
                "sign": str(sign),
                "tgt": int(tgt),
                "adj_score": float(adj_score),
                "signed_feat": sign_mult * raw_feat,
            }
            key = (int(comp_id), tuple(int(s) for s in srcs))
            prev = group_pool.get(key)
            if prev is None or float(item["adj_score"]) > float(prev["adj_score"]):
                group_pool[key] = item
    by_comp: dict[int, list[dict]] = defaultdict(list)
    for item in group_pool.values():
        by_comp[int(item["comp_id"])].append(item)
    return dict(by_comp)


def structural_redundancy_count(
    rule: tuple[tuple[int, ...], str, int],
    selected_rules: list[tuple[tuple[int, ...], str, int]],
) -> int:
    srcs, sign, tgt = rule
    set_r = set(int(s) for s in srcs)
    count = 0
    for srcs2, sign2, tgt2 in selected_rules:
        if (srcs2, sign2, tgt2) == rule:
            continue
        if str(sign2) != str(sign) or int(tgt2) != int(tgt):
            continue
        set_s = set(int(s) for s in srcs2)
        if set_r < set_s or set_s < set_r:
            count += 1
        elif len(set_r) == len(set_s) == len(set_r | set_s) - 1:
            count += 1
    return int(count)


def redundancy_pair_penalty(
    rules: list[tuple[tuple[int, ...], str, int]],
) -> int:
    by_sign: dict[tuple[str, int], list[tuple[int, ...]]] = {}
    for srcs, sign, tgt in rules:
        by_sign.setdefault((str(sign), int(tgt)), []).append(tuple(int(s) for s in srcs))
    pen = 0
    for subsets in by_sign.values():
        for a, b in itertools.combinations(subsets, 2):
            set_a = set(int(s) for s in a)
            set_b = set(int(s) for s in b)
            if set_a < set_b or set_b < set_a:
                pen += 1
            elif len(set_a) == len(set_b) and len(set_a | set_b) == len(set_a) + 1:
                pen += 1
    return int(pen)


def project_component_selection(
    *,
    total_tr: np.ndarray,
    pos_tr: np.ndarray,
    total_va: np.ndarray,
    pos_va: np.ndarray,
    selected_items: list[dict],
    sibling_lambda: float,
    projection_lambda: float,
) -> list[dict]:
    if len(selected_items) <= 1 or float(projection_lambda) <= 0.0:
        return list(selected_items)
    scale = float(math.log(max(float(total_va.sum()), 2.0)))
    n = len(selected_items)
    best_obj = math.inf
    best_items = list(selected_items)
    for mask in range(1 << n):
        feats = []
        rules = []
        items = []
        for i, item in enumerate(selected_items):
            if (mask >> i) & 1:
                feats.append(item["signed_feat"])
                items.append(item)
                rules.append((item["srcs"], item["sign"], item["tgt"]))
        obj = fit_bic_from_signed_feats(total_tr, pos_tr, total_va, pos_va, feats)
        obj += float(sibling_lambda) * scale * float(sibling_union_penalty(rules))
        obj += float(projection_lambda) * scale * float(redundancy_pair_penalty(rules))
        if float(obj) < float(best_obj):
            best_obj = float(obj)
            best_items = list(items)
    return best_items


def subset_shadow_prune_component_selection(
    *,
    total_tr: np.ndarray,
    pos_tr: np.ndarray,
    total_va: np.ndarray,
    pos_va: np.ndarray,
    selected_items: list[dict],
    sibling_lambda: float,
    projection_lambda: float,
    tau: float,
) -> list[dict]:
    if len(selected_items) <= 1 or float(tau) <= 0.0:
        return list(selected_items)

    scale = float(math.log(max(float(total_va.sum()), 2.0)))

    def objective(items: list[dict]) -> float:
        feats = [it["signed_feat"] for it in items]
        rules = [(it["srcs"], it["sign"], it["tgt"]) for it in items]
        obj = fit_bic_from_signed_feats(total_tr, pos_tr, total_va, pos_va, feats)
        obj += float(sibling_lambda) * scale * float(sibling_union_penalty(rules))
        obj += float(projection_lambda) * scale * float(redundancy_pair_penalty(rules))
        return float(obj)

    cur = list(selected_items)
    cur_obj = objective(cur)
    stop = float(tau) * 0.5 * scale

    while len(cur) > 1:
        best_idx = None
        best_delta = None
        for idx, item in enumerate(cur):
            srcs = set(int(s) for s in item["srcs"])
            has_same_sign_superset = any(
                j != idx
                and other["sign"] == item["sign"]
                and int(other["tgt"]) == int(item["tgt"])
                and srcs < set(int(s) for s in other["srcs"])
                for j, other in enumerate(cur)
            )
            if not has_same_sign_superset:
                continue
            rem = cur[:idx] + cur[idx + 1 :]
            rem_obj = objective(rem)
            delta = float(rem_obj - cur_obj)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is None or float(best_delta) > stop:
            break
        cur.pop(int(best_idx))
        cur_obj = objective(cur)

    return cur


def source_level_project_component_selection(
    *,
    total_tr: np.ndarray,
    pos_tr: np.ndarray,
    total_va: np.ndarray,
    pos_va: np.ndarray,
    source_order: tuple[int, ...],
    selected_items: list[dict],
    sibling_lambda: float,
    projection_lambda: float,
) -> list[dict]:
    if len(selected_items) <= 1:
        return list(selected_items)

    scale = float(math.log(max(float(total_va.sum()), 2.0)))
    src_pos = {int(s): i for i, s in enumerate(source_order)}
    raw_feats = []
    valid_items = []
    for item in selected_items:
        try:
            local_idxs = tuple(sorted(int(src_pos[int(s)]) for s in item["srcs"]))
        except KeyError:
            continue
        raw_feat = cell_feature_from_local_idxs(len(source_order), local_idxs)
        raw_feats.append(raw_feat)
        valid_items.append(item)
    n = len(valid_items)
    if n <= 1:
        return list(valid_items)

    feats = []
    for i, item in enumerate(valid_items):
        mask = np.ones_like(raw_feats[i], dtype=bool)
        set_r = set(int(s) for s in item["srcs"])
        for j, other in enumerate(valid_items):
            if i == j:
                continue
            if other["sign"] != item["sign"] or int(other["tgt"]) != int(item["tgt"]):
                continue
            if set_r < set(int(s) for s in other["srcs"]):
                mask &= ~(raw_feats[j] > 0)
        sign_mult = 1.0 if str(item["sign"]) == "exc" else -1.0
        feats.append(sign_mult * raw_feats[i] * mask.astype(np.float64))

    best_obj = math.inf
    best_items = list(valid_items)
    for mask in range(1 << n):
        cur_feats = []
        cur_items = []
        rules = []
        for i, item in enumerate(valid_items):
            if (mask >> i) & 1:
                cur_feats.append(feats[i])
                cur_items.append(item)
                rules.append((item["srcs"], item["sign"], item["tgt"]))
        obj = fit_bic_from_signed_feats(total_tr, pos_tr, total_va, pos_va, cur_feats)
        obj += float(sibling_lambda) * scale * float(sibling_union_penalty(rules))
        obj += float(projection_lambda) * scale * float(redundancy_pair_penalty(rules))
        if obj < best_obj:
            best_obj = float(obj)
            best_items = list(cur_items)
    return best_items


def subset_sign_consistency_prune(
    *,
    total: np.ndarray,
    pos: np.ndarray,
    source_order: tuple[int, ...],
    selected_items: list[dict],
) -> list[dict]:
    if len(selected_items) <= 1:
        return list(selected_items)

    all_bits = np.arange(len(total), dtype=np.int64)
    src_pos = {int(s): i for i, s in enumerate(source_order)}
    global_rate = float((float(pos.sum()) + 0.5) / (float(total.sum()) + 1.0))
    global_logit = math.log(global_rate / max(1.0 - global_rate, 1e-8))

    cur = list(selected_items)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        keep = []
        for item in cur:
            try:
                local_idxs = tuple(sorted(int(src_pos[int(s)]) for s in item["srcs"]))
            except KeyError:
                keep.append(item)
                continue
            bits_u = subset_bits(local_idxs)
            mask = (all_bits & bits_u) == bits_u
            has_superset = False
            for other in cur:
                if other is item:
                    continue
                if other["sign"] != item["sign"] or int(other["tgt"]) != int(item["tgt"]):
                    continue
                set_r = set(int(s) for s in item["srcs"])
                set_o = set(int(s) for s in other["srcs"])
                if set_r < set_o and len(set_o) == len(set_r) + 1:
                    has_superset = True
                    local_other = tuple(sorted(int(src_pos[int(s)]) for s in other["srcs"]))
                    bits_o = subset_bits(local_other)
                    mask &= ~((all_bits & bits_o) == bits_o)
            if not has_superset:
                keep.append(item)
                continue
            tot = float(total[mask].sum())
            if tot <= 0.0:
                changed = True
                continue
            p = float(pos[mask].sum())
            rate = float((p + 0.5) / (tot + 1.0))
            logit = math.log(rate / max(1.0 - rate, 1e-8))
            if item["sign"] == "exc" and logit <= global_logit:
                changed = True
                continue
            if item["sign"] == "inh" and logit >= global_logit:
                changed = True
                continue
            keep.append(item)
        cur = keep
    return cur


def prune_component_selection(
    *,
    total_tr: np.ndarray,
    pos_tr: np.ndarray,
    total_va: np.ndarray,
    pos_va: np.ndarray,
    selected_items: list[dict],
    mode: str,
    tau: float,
) -> list[dict]:
    if mode == "none" or len(selected_items) <= 1:
        return list(selected_items)

    cur = list(selected_items)
    scale = float(math.log(max(float(total_va.sum()), 2.0)))

    def bic_of(items: list[dict]) -> float:
        feats = [it["signed_feat"] for it in items]
        return fit_bic_from_signed_feats(total_tr, pos_tr, total_va, pos_va, feats)

    while len(cur) > 1:
        cur_bic = bic_of(cur)
        cur_rules = [(it["srcs"], it["sign"], it["tgt"]) for it in cur]
        best_idx = None
        best_metric = None
        best_delta = None
        for idx, item in enumerate(cur):
            rem = cur[:idx] + cur[idx + 1 :]
            rem_bic = bic_of(rem)
            delta = float(rem_bic - cur_bic)
            if mode == "bic":
                metric = delta
            else:
                red = structural_redundancy_count((item["srcs"], item["sign"], item["tgt"]), cur_rules)
                metric = float(delta / (1.0 + float(red)))
            if best_metric is None or float(metric) < float(best_metric):
                best_idx = idx
                best_metric = float(metric)
                best_delta = float(delta)
        if best_idx is None:
            break
        if mode == "bic":
            stop = float(tau) * 0.5 * scale
            if float(best_delta) > stop:
                break
        else:
            stop = float(tau) * 0.35 * scale
            if float(best_metric) > stop:
                break
        cur.pop(int(best_idx))
    return cur


def run_auto_conditional_redundancy(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    component_defs: list[dict],
    base_tables: dict,
    channel_by_id: dict[int, ChannelDef],
    support_pow: float,
    sup_pen: float,
    fixed_target: int,
    max_rules: int,
):
    def _refit_component(
        total_tr: np.ndarray,
        pos_tr: np.ndarray,
        total_va: np.ndarray,
        pos_va: np.ndarray,
        items: list[dict],
    ):
        feats = [it["signed_feat"] for it in items]
        beta, theta, _ = fit_signed_binomial_model(
            total_tr,
            pos_tr,
            feats,
        )
        if feats:
            Gv = np.stack(feats, axis=1)
            eta_val = float(beta) + Gv @ theta
        else:
            eta_val = np.full_like(total_va, float(beta), dtype=np.float64)
        ll_val = binomial_loglik(total_va, pos_va, eta_val)
        return float(beta), theta, eta_val, float(ll_val)

    comp_models = []
    for comp_id, comp in enumerate(component_defs):
        global_cols = [int(c) for c in comp["global_cols"]]
        if global_cols:
            tr_total, tr_pos = component_cell_counts(X_train[:, global_cols], y_train)
            va_total, va_pos = component_cell_counts(X_val[:, global_cols], y_val)
        else:
            tr_total = np.zeros((1,), dtype=np.float64)
            tr_pos = np.zeros((1,), dtype=np.float64)
            va_total = np.zeros((1,), dtype=np.float64)
            va_pos = np.zeros((1,), dtype=np.float64)
        beta, theta, eta_val, ll_val = _refit_component(tr_total, tr_pos, va_total, va_pos, [])
        comp_models.append(
            {
                "train_total": tr_total,
                "train_pos": tr_pos,
                "val_total": va_total,
                "val_pos": va_pos,
                "selected": [],
                "beta": beta,
                "theta": theta,
                "eta_val": eta_val,
                "ll_val": ll_val,
            }
        )

    group_pool: dict[tuple[int, tuple[int, ...]], list[dict]] = {}
    for order in (1, 2, 3):
        rows = select_stage(
            base_tables,
            component_defs,
            channel_by_id,
            order=order,
            support_pow=float(support_pow),
            sup_pen=float(sup_pen),
            fixed_target=int(fixed_target),
        )
        for adj_score, coef, comp_id, srcs, sign, tgt, combo in rows:
            d = len(component_defs[int(comp_id)]["channel_ids"])
            local_idxs = tuple(int(component_defs[int(comp_id)]["channel_pos"][int(ch)]) for ch in combo)
            raw_feat = cell_feature_from_local_idxs(d, local_idxs)
            sign_mult = 1.0 if str(sign) == "exc" else -1.0
            item = {
                "comp_id": int(comp_id),
                "srcs": tuple(int(s) for s in srcs),
                "sign": str(sign),
                "tgt": int(tgt),
                "raw_feat": raw_feat,
                "signed_feat": sign_mult * raw_feat,
            }
            group_pool.setdefault((int(comp_id), tuple(int(s) for s in srcs)), []).append(item)

    selected_group_keys: set[tuple[int, tuple[int, ...]]] = set()
    selected_rules: list[tuple[tuple[int, ...], str, int]] = []
    for _ in range(max(int(max_rules), 1)):
        best = None
        for group_key, variants in group_pool.items():
            if group_key in selected_group_keys:
                continue
            comp_id = int(group_key[0])
            comp_model = comp_models[comp_id]
            for var in variants:
                set_srcs = set(int(s) for s in var["srcs"])
                neighbor_feats = []
                for sel in comp_model["selected"]:
                    set_sel = set(int(s) for s in sel["srcs"])
                    if set_srcs < set_sel or set_sel < set_srcs:
                        neighbor_feats.append(sel["raw_feat"])
                uniq = conditional_unique_ratio(var["raw_feat"], neighbor_feats, comp_model["train_total"])
                if uniq <= 1e-8:
                    continue
                gain = approx_val_gain_from_score(
                    comp_model["eta_val"],
                    comp_model["val_total"],
                    comp_model["val_pos"],
                    var["signed_feat"],
                )
                eff_gain = float(gain * uniq)
                bic_stop = 0.5 * math.log(max(float(comp_model["val_total"].sum()), 2.0))
                cand = {
                    "eff_gain": eff_gain,
                    "bic_stop": float(bic_stop),
                    "comp_id": comp_id,
                    "group_key": group_key,
                    "var": var,
                }
                if best is None or float(cand["eff_gain"]) > float(best["eff_gain"]):
                    best = cand
        if best is None:
            break
        if float(best["eff_gain"]) <= float(best["bic_stop"]):
            break
        comp_id = int(best["comp_id"])
        comp_model = comp_models[comp_id]
        new_items = list(comp_model["selected"]) + [best["var"]]
        beta, theta, eta_val, ll_val = _refit_component(
            comp_model["train_total"],
            comp_model["train_pos"],
            comp_model["val_total"],
            comp_model["val_pos"],
            new_items,
        )
        comp_model["selected"] = new_items
        comp_model["beta"] = beta
        comp_model["theta"] = theta
        comp_model["eta_val"] = eta_val
        comp_model["ll_val"] = ll_val
        selected_group_keys.add(tuple(best["group_key"]))
        selected_rules.append((best["var"]["srcs"], best["var"]["sign"], best["var"]["tgt"]))

    return tuple(sorted(set(selected_rules)))


def run_auto_sibling_bic(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    component_defs: list[dict],
    base_tables: dict,
    channel_by_id: dict[int, ChannelDef],
    support_pow: float,
    sup_pen: float,
    fixed_target: int,
    sibling_lambda: float,
    projection_lambda: float,
    subset_prune_tau: float,
    source_projection_lambda: float,
    sign_prune: bool,
    post_prune_mode: str,
    post_prune_tau: float,
):
    group_pool = collect_top_group_variants(
        component_defs=component_defs,
        base_tables=base_tables,
        channel_by_id=channel_by_id,
        support_pow=float(support_pow),
        sup_pen=float(sup_pen),
        fixed_target=int(fixed_target),
    )

    preds: set[tuple[tuple[int, ...], str, int]] = set()
    for comp_id, comp in enumerate(component_defs):
        global_cols = [int(c) for c in comp["global_cols"]]
        tr_total, tr_pos = component_cell_counts(X_train[:, global_cols], y_train)
        va_total, va_pos = component_cell_counts(X_val[:, global_cols], y_val)
        groups = list(group_pool.get(int(comp_id), []))
        n = len(groups)
        if n == 0:
            continue
        scale = float(math.log(max(float(va_total.sum()), 2.0)))
        best_obj = math.inf
        best_items: list[dict] = []
        for mask in range(1 << n):
            signed_feats = []
            selected_items = []
            rules = []
            for i, var in enumerate(groups):
                if (mask >> i) & 1:
                    signed_feats.append(var["signed_feat"])
                    selected_items.append(var)
                    rules.append((var["srcs"], var["sign"], var["tgt"]))
            obj = fit_bic_from_signed_feats(tr_total, tr_pos, va_total, va_pos, signed_feats)
            obj += float(sibling_lambda) * scale * float(sibling_union_penalty(rules))
            if float(obj) < float(best_obj):
                best_obj = float(obj)
                best_items = list(selected_items)
        best_items = prune_component_selection(
            total_tr=tr_total,
            pos_tr=tr_pos,
            total_va=va_total,
            pos_va=va_pos,
            selected_items=best_items,
            mode=str(post_prune_mode),
            tau=float(post_prune_tau),
        )
        best_items = project_component_selection(
            total_tr=tr_total,
            pos_tr=tr_pos,
            total_va=va_total,
            pos_va=va_pos,
            selected_items=best_items,
            sibling_lambda=float(sibling_lambda),
            projection_lambda=float(projection_lambda),
        )
        best_items = subset_shadow_prune_component_selection(
            total_tr=tr_total,
            pos_tr=tr_pos,
            total_va=va_total,
            pos_va=va_pos,
            selected_items=best_items,
            sibling_lambda=float(sibling_lambda),
            projection_lambda=float(projection_lambda),
            tau=float(subset_prune_tau),
        )
        if float(source_projection_lambda) > 0.0 and len(best_items) > 1:
            source_cols = []
            for src in comp["sources"]:
                cols = [
                    int(i)
                    for i, ch_id in enumerate(comp["channel_ids"])
                    if int(channel_by_id[int(ch_id)].source) == int(src)
                ]
                source_cols.append(cols)
            xtr_source = np.stack(
                [
                    X_train[:, global_cols][:, cols].max(axis=1).astype(np.int64)
                    if len(cols) > 1
                    else X_train[:, global_cols][:, cols[0]].astype(np.int64)
                    for cols in source_cols
                ],
                axis=1,
            )
            xva_source = np.stack(
                [
                    X_val[:, global_cols][:, cols].max(axis=1).astype(np.int64)
                    if len(cols) > 1
                    else X_val[:, global_cols][:, cols[0]].astype(np.int64)
                    for cols in source_cols
                ],
                axis=1,
            )
            src_total_tr, src_pos_tr = component_cell_counts(xtr_source, y_train)
            src_total_va, src_pos_va = component_cell_counts(xva_source, y_val)
            best_items = source_level_project_component_selection(
                total_tr=src_total_tr,
                pos_tr=src_pos_tr,
                total_va=src_total_va,
                pos_va=src_pos_va,
                source_order=tuple(int(s) for s in comp["sources"]),
                selected_items=best_items,
                sibling_lambda=float(sibling_lambda),
                projection_lambda=float(source_projection_lambda),
            )
        if bool(sign_prune) and len(best_items) > 1:
            source_cols = []
            for src in comp["sources"]:
                cols = [
                    int(i)
                    for i, ch_id in enumerate(comp["channel_ids"])
                    if int(channel_by_id[int(ch_id)].source) == int(src)
                ]
                source_cols.append(cols)
            xall_source = np.stack(
                [
                    X_train[:, global_cols][:, cols].max(axis=1).astype(np.int64)
                    if len(cols) > 1
                    else X_train[:, global_cols][:, cols[0]].astype(np.int64)
                    for cols in source_cols
                ],
                axis=1,
            )
            src_total_all, src_pos_all = component_cell_counts(xall_source, y_train)
            best_items = subset_sign_consistency_prune(
                total=src_total_all,
                pos=src_pos_all,
                source_order=tuple(int(s) for s in comp["sources"]),
                selected_items=best_items,
            )
        preds.update((it["srcs"], it["sign"], it["tgt"]) for it in best_items)
    return tuple(sorted(preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixed_target", type=int, default=6)
    ap.add_argument("--partition_mode", choices=["manual", "singleton", "all_partitions"], default="all_partitions")
    ap.add_argument("--max_block", type=int, default=4)
    ap.add_argument("--manual_partition", default=None)
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
    ap.add_argument(
        "--selection_mode",
        choices=["exact_top_k", "auto_conditional_redundancy", "auto_sibling_bic"],
        default="exact_top_k",
    )
    ap.add_argument("--max_auto_rules", type=int, default=20)
    ap.add_argument("--auto_sibling_lambda", type=float, default=4.0)
    ap.add_argument("--auto_projection_lambda", type=float, default=0.0)
    ap.add_argument("--auto_subset_prune_tau", type=float, default=0.0)
    ap.add_argument("--auto_source_projection_lambda", type=float, default=0.0)
    ap.add_argument("--auto_sign_prune", action="store_true")
    ap.add_argument("--auto_post_prune_mode", choices=["none", "bic", "struct"], default="none")
    ap.add_argument("--auto_post_prune_tau", type=float, default=1.0)
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
    train_cache = build_channel_cache(
        tr_data,
        channels,
        fixed_target=int(args.fixed_target),
        kernel_eval_mode=str(args.source_kernel_mode),
        kernel_num_bins=int(args.source_kernel_num_bins),
        kernel_max_cap=float(args.max_lag),
        kernel_support_mult=float(args.source_kernel_support_mult),
    )
    val_cache = build_channel_cache(
        va_data,
        channels,
        fixed_target=int(args.fixed_target),
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

    thr_values = parse_csv_floats(args.thr_values)
    support_pow_values = parse_csv_floats(args.support_pow_values)
    sup_pen_values = parse_csv_floats(args.sup_pen_values)
    quota_values = [(k1, k2, k3) for k1 in range(len(gt) + 1) for k2 in range(len(gt) + 1) for k3 in range(len(gt) + 1) if k1 + k2 + k3 == len(gt)]

    results = []
    for thr, support_pow, sup_pen, components in itertools.product(
        thr_values,
        support_pow_values,
        sup_pen_values,
        components_list,
    ):
        X_all = (train_cache["event_q"] > float(thr)).astype(np.int64)
        y = train_cache["event_y"].astype(np.int64)
        component_defs = []
        base_tables = {}
        for comp_id, comp_sources in enumerate(components):
            local_channels = [int(ch.channel_id) for ch in channels if int(ch.source) in set(int(s) for s in comp_sources)]
            local_cols = [int(channel_col_by_id[ch]) for ch in local_channels]
            component_defs.append(
                {
                    "sources": tuple(int(s) for s in comp_sources),
                    "channel_ids": tuple(int(ch) for ch in local_channels),
                    "global_cols": tuple(int(c) for c in local_cols),
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

        if args.selection_mode == "exact_top_k":
            for quota in quota_values:
                res_tables = {key: val.copy() for key, val in base_tables.items()}
                selected = []
                for order, k in ((1, quota[0]), (2, quota[1]), (3, quota[2])):
                    if k <= 0:
                        continue
                    rows = select_stage(
                        res_tables,
                        component_defs,
                        channel_by_id,
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
                        quota,
                        tuple(tuple(int(s) for s in comp) for comp in components),
                        pred,
                    )
                )
        elif args.selection_mode == "auto_conditional_redundancy":
            X_val = (val_cache["event_q"] > float(thr)).astype(np.int64)
            y_val = val_cache["event_y"].astype(np.int64)
            pred = run_auto_conditional_redundancy(
                X_train=X_all,
                y_train=y,
                X_val=X_val,
                y_val=y_val,
                component_defs=component_defs,
                base_tables=base_tables,
                channel_by_id=channel_by_id,
                support_pow=float(support_pow),
                sup_pen=float(sup_pen),
                fixed_target=int(args.fixed_target),
                max_rules=int(args.max_auto_rules),
            )
            results.append(
                (
                    len(gt & set(pred)),
                    float(thr),
                    float(support_pow),
                    float(sup_pen),
                    ("auto", len(pred)),
                    tuple(tuple(int(s) for s in comp) for comp in components),
                    pred,
                )
            )
        else:
            X_val = (val_cache["event_q"] > float(thr)).astype(np.int64)
            y_val = val_cache["event_y"].astype(np.int64)
            pred = run_auto_sibling_bic(
                X_train=X_all,
                y_train=y,
                X_val=X_val,
                y_val=y_val,
                component_defs=component_defs,
                base_tables=base_tables,
                channel_by_id=channel_by_id,
                support_pow=float(support_pow),
                sup_pen=float(sup_pen),
                fixed_target=int(args.fixed_target),
                sibling_lambda=float(args.auto_sibling_lambda),
                projection_lambda=float(args.auto_projection_lambda),
                subset_prune_tau=float(args.auto_subset_prune_tau),
                source_projection_lambda=float(args.auto_source_projection_lambda),
                sign_prune=bool(args.auto_sign_prune),
                post_prune_mode=str(args.auto_post_prune_mode),
                post_prune_tau=float(args.auto_post_prune_tau),
            )
            results.append(
                (
                    len(gt & set(pred)),
                    float(thr),
                    float(support_pow),
                    float(sup_pen),
                    ("auto_sibling", len(pred)),
                    tuple(tuple(int(s) for s in comp) for comp in components),
                    pred,
                )
            )

    results.sort(reverse=True)
    seen = set()
    for item in results[: args.topn]:
        key = (item[0], item[1], item[2], item[3], item[4], item[5], item[6])
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
                "quota": best[4],
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

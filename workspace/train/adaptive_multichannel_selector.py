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
    channel_cache = build_channel_cache(
        tr_data,
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
        X_all = (channel_cache["event_q"] > float(thr)).astype(np.int64)
        y = channel_cache["event_y"].astype(np.int64)
        component_defs = []
        base_tables = {}
        for comp_id, comp_sources in enumerate(components):
            local_channels = [int(ch.channel_id) for ch in channels if int(ch.source) in set(int(s) for s in comp_sources)]
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

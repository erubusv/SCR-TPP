"""Partition-free conjunctive logical-rule initializer.

Mainline philosophy:
1. Estimate one nonnegative piecewise-linear lag kernel per source from source-target lag excess.
2. Build source responses z_s(t) by convolving source events with that kernel.
3. Convert each source response into a bounded activity
       a_s(t) = 1 - exp(-z_s(t))
4. Define each subset rule feature as a conjunction
       rho_U(t) = prod(a_s(t) for s in U)
5. Use a point-process objective
       sum log lambda(t_i) - integral lambda(t) dt
   with sign-exclusive subset states {off, exc, inh}.

This file is the clean replacement path for overlap-rule discovery. It does not
use partitions or anchor classification in the default path.
"""

from __future__ import annotations

import itertools
import pickle
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class SourceKernel:
    source: int
    knots: np.ndarray
    heights: np.ndarray
    slopes: np.ndarray = field(init=False, repr=False)
    intercepts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.knots = np.asarray(self.knots, dtype=np.float64)
        self.heights = np.asarray(self.heights, dtype=np.float64)
        if self.knots.ndim != 1 or self.heights.ndim != 1 or self.knots.size != self.heights.size:
            raise ValueError("SourceKernel expects 1D knots/heights with equal length")
        if self.knots.size < 2:
            raise ValueError("SourceKernel requires at least two knots")
        delta = np.diff(self.knots)
        self.slopes = np.diff(self.heights) / np.maximum(delta, 1e-8)
        self.intercepts = self.heights[:-1] - self.slopes * self.knots[:-1]

    @property
    def peak(self) -> float:
        return float(self.knots[int(np.argmax(self.heights))])

    @property
    def width(self) -> float:
        return float(self.knots[-1])

    @property
    def min_spacing(self) -> float:
        return float(np.min(np.diff(self.knots)))

    def eval_sum(self, event_times: np.ndarray, t: float) -> float:
        if event_times.size == 0:
            return 0.0
        right = np.searchsorted(event_times, t, side="left")
        if right <= 0:
            return 0.0
        prefix = np.empty((event_times.size + 1,), dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(event_times, out=prefix[1:])

        total = 0.0
        for left_dt, right_dt, slope, intercept in zip(
            self.knots[:-1], self.knots[1:], self.slopes, self.intercepts
        ):
            high = np.searchsorted(event_times, t - float(left_dt), side="left")
            low = np.searchsorted(event_times, t - float(right_dt), side="left")
            if high <= low:
                continue
            count = float(high - low)
            sum_times = float(prefix[high] - prefix[low])
            total += float(slope) * (float(t) * count - sum_times) + float(intercept) * count
        return max(float(total), 0.0)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(path: str):
    with open(path, "rb") as f:
        ds = pickle.load(f)
    return ds["train"], ds["val"], ds["metadata"]


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


def build_seq_event_arrays(split: list[dict], num_types: int) -> list[dict[int, np.ndarray]]:
    out = []
    for seq in split:
        by_type = defaultdict(list)
        for t, e in zip(seq["time"], seq["event"]):
            by_type[int(e)].append(float(t))
        out.append({k: np.asarray(v, dtype=np.float64) for k, v in by_type.items()})
    return out


def collect_lag_hist(
    seq_arrays: list[dict[int, np.ndarray]],
    *,
    source: int,
    target: int,
    max_lag: float,
    num_bins: int,
) -> np.ndarray:
    edges = np.linspace(0.0, max_lag, int(num_bins) + 1)
    hist = np.zeros((int(num_bins),), dtype=np.float64)
    for by_type in seq_arrays:
        src = by_type.get(int(source), np.zeros((0,), dtype=np.float64))
        tgt = by_type.get(int(target), np.zeros((0,), dtype=np.float64))
        if src.size == 0 or tgt.size == 0:
            continue
        for t in tgt:
            left = np.searchsorted(src, t - max_lag, side="left")
            right = np.searchsorted(src, t, side="left")
            if right <= left:
                continue
            dts = t - src[left:right]
            hist += np.histogram(dts, bins=edges)[0]
    return hist


def build_event_lag_bin_cache(
    seq_arrays: list[dict[int, np.ndarray]],
    *,
    source_ids: tuple[int, ...],
    target_seq_ids: np.ndarray,
    target_times: np.ndarray,
    max_lag: float,
    num_bins: int,
) -> dict[int, np.ndarray]:
    edges = np.linspace(0.0, max_lag, int(num_bins) + 1)
    cache: dict[int, np.ndarray] = {}
    n_events = int(target_times.size)
    for src in source_ids:
        mat = np.zeros((n_events, int(num_bins)), dtype=np.float32)
        for i, (seq_idx, t) in enumerate(zip(target_seq_ids.tolist(), target_times.tolist())):
            src_times = seq_arrays[int(seq_idx)].get(int(src), np.zeros((0,), dtype=np.float64))
            if src_times.size == 0:
                continue
            left = np.searchsorted(src_times, float(t) - max_lag, side="left")
            right = np.searchsorted(src_times, float(t), side="left")
            if right <= left:
                continue
            dts = float(t) - src_times[left:right]
            mat[i, :] = np.histogram(dts, bins=edges)[0].astype(np.float32, copy=False)
        cache[int(src)] = mat
    return cache


def normalize_piecewise_area(knots: np.ndarray, heights: np.ndarray) -> np.ndarray:
    area = float(np.trapz(np.asarray(heights, dtype=np.float64), x=np.asarray(knots, dtype=np.float64)))
    if area <= 1e-12:
        span = float(knots[-1] - knots[0])
        if span <= 1e-12:
            return np.ones_like(heights, dtype=np.float64)
        return np.full_like(heights, fill_value=1.0 / span, dtype=np.float64)
    return np.asarray(heights, dtype=np.float64) / area


def init_piecewise_heights(
    *,
    hist: np.ndarray,
    edges: np.ndarray,
    expected: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    centers = 0.5 * (edges[:-1] + edges[1:])
    excess = np.abs(np.asarray(hist, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    smooth = np.convolve(excess, np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64), mode="same")
    smooth /= 9.0
    smooth = np.maximum(smooth, 0.0)
    if float(np.sum(smooth)) <= 1e-12:
        heights = np.ones_like(knots, dtype=np.float64)
    else:
        heights = np.interp(
            np.asarray(knots, dtype=np.float64),
            centers,
            smooth,
            left=float(smooth[0]),
            right=0.0,
        )
        heights = np.maximum(heights, 0.0)
    return normalize_piecewise_area(knots, heights)


def estimate_source_kernels(
    seq_arrays: list[dict[int, np.ndarray]],
    *,
    source_ids: tuple[int, ...],
    target: int,
    max_lag: float,
    num_bins: int,
    num_knots: int,
    time_horizon: float,
) -> dict[int, SourceKernel]:
    target_count = 0
    total_time = float(len(seq_arrays)) * float(time_horizon)
    source_counts = {int(s): 0 for s in source_ids}
    for by_type in seq_arrays:
        target_count += int(by_type.get(int(target), np.zeros((0,), dtype=np.float64)).size)
        for src in source_ids:
            source_counts[int(src)] += int(by_type.get(int(src), np.zeros((0,), dtype=np.float64)).size)

    edges = np.linspace(0.0, max_lag, int(num_bins) + 1)
    bin_width = float(edges[1] - edges[0])
    knots = np.linspace(0.0, float(max_lag), int(num_knots), dtype=np.float64)
    kernels: dict[int, SourceKernel] = {}
    for src in source_ids:
        hist = collect_lag_hist(
            seq_arrays,
            source=int(src),
            target=int(target),
            max_lag=max_lag,
            num_bins=num_bins,
        )
        src_rate = float(source_counts[int(src)]) / max(total_time, 1e-8)
        expected = np.full_like(hist, src_rate * float(target_count) * bin_width)
        heights = init_piecewise_heights(hist=hist, edges=edges, expected=expected, knots=knots)
        kernels[int(src)] = SourceKernel(
            source=int(src),
            knots=knots.copy(),
            heights=heights,
        )
    return kernels


def collect_weighted_lag_hist(
    *,
    source: int,
    event_weights: np.ndarray,
    event_lag_bin_cache: dict[int, np.ndarray],
) -> np.ndarray:
    mat = event_lag_bin_cache.get(int(source))
    if mat is None:
        raise KeyError(int(source))
    weights = np.asarray(event_weights, dtype=np.float64)
    return (weights @ mat.astype(np.float64, copy=False)).astype(np.float64, copy=False)


def collect_target_events(split: list[dict], *, target: int) -> tuple[np.ndarray, np.ndarray]:
    seq_ids: list[int] = []
    times: list[float] = []
    for seq_idx, seq in enumerate(split):
        for t, e in zip(seq["time"], seq["event"]):
            if int(e) == int(target):
                seq_ids.append(int(seq_idx))
                times.append(float(t))
    return np.asarray(seq_ids, dtype=np.int64), np.asarray(times, dtype=np.float64)


def auto_grid_step(kernels: dict[int, SourceKernel]) -> float:
    min_spacing = min(float(ker.min_spacing) for ker in kernels.values())
    return float(np.clip(min_spacing / 2.0, 0.1, 0.5))


def build_midpoint_grid(split: list[dict], *, time_horizon: float, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_ids: list[np.ndarray] = []
    times: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    midpoint = 0.5 * float(step)
    for seq_idx, _seq in enumerate(split):
        grid = np.arange(midpoint, float(time_horizon), float(step), dtype=np.float64)
        if grid.size == 0:
            continue
        seq_ids.append(np.full(grid.shape, fill_value=int(seq_idx), dtype=np.int64))
        times.append(grid)
        weights.append(np.full(grid.shape, fill_value=float(step), dtype=np.float64))
    if not times:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.concatenate(seq_ids),
        np.concatenate(times),
        np.concatenate(weights),
    )


def compute_source_signal_matrix(
    seq_arrays: list[dict[int, np.ndarray]],
    kernels: dict[int, SourceKernel],
    *,
    source_ids: tuple[int, ...],
    seq_ids: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    z = np.zeros((len(times), len(source_ids)), dtype=np.float64)
    if len(times) == 0:
        return z

    order = np.argsort(seq_ids, kind="stable")
    seq_ids_sorted = seq_ids[order]
    uniq, starts = np.unique(seq_ids_sorted, return_index=True)
    ends = np.r_[starts[1:], len(order)]

    for col, src in enumerate(source_ids):
        kernel = kernels[int(src)]
        for seq_idx, st, en in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
            idxs = order[st:en]
            q = times[idxs]
            src_events = seq_arrays[int(seq_idx)].get(int(src), np.zeros((0,), dtype=np.float64))
            if src_events.size == 0:
                continue
            prefix = np.empty((src_events.size + 1,), dtype=np.float64)
            prefix[0] = 0.0
            np.cumsum(src_events, out=prefix[1:])
            total = np.zeros((len(q),), dtype=np.float64)
            for left_dt, right_dt, slope, intercept in zip(
                kernel.knots[:-1],
                kernel.knots[1:],
                kernel.slopes,
                kernel.intercepts,
            ):
                high = np.searchsorted(src_events, q - float(left_dt), side="left")
                low = np.searchsorted(src_events, q - float(right_dt), side="left")
                count = (high - low).astype(np.float64)
                if not np.any(count > 0.0):
                    continue
                sum_times = prefix[high] - prefix[low]
                total += float(slope) * (count * q - sum_times) + float(intercept) * count
            z[idxs, col] = np.maximum(total, 0.0)
    return z


def bounded_source_activity(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-np.maximum(z, 0.0))


def subset_list(source_ids: tuple[int, ...], max_order: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for order in range(1, int(max_order) + 1):
        out.extend(tuple(int(source_ids[i]) for i in idxs) for idxs in itertools.combinations(range(len(source_ids)), order))
    return out


def trapz_area_weights(knots: np.ndarray) -> np.ndarray:
    knots = np.asarray(knots, dtype=np.float64)
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError("trapz_area_weights expects at least two knots")
    w = np.zeros((knots.size,), dtype=np.float64)
    delta = np.diff(knots)
    w[0] = 0.5 * float(delta[0])
    w[-1] = 0.5 * float(delta[-1])
    if knots.size > 2:
        w[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    return w


def basis_kernel(knots: np.ndarray, basis_idx: int) -> SourceKernel:
    heights = np.zeros((len(knots),), dtype=np.float64)
    heights[int(basis_idx)] = 1.0
    return SourceKernel(source=-1, knots=np.asarray(knots, dtype=np.float64), heights=heights)


class SourceBasisCache:
    def __init__(
        self,
        *,
        source_ids: tuple[int, ...],
        knots: np.ndarray,
        train_arrays: list[dict[int, np.ndarray]],
        val_arrays: list[dict[int, np.ndarray]],
        train_event_seq_ids: np.ndarray,
        train_event_times: np.ndarray,
        train_grid_seq_ids: np.ndarray,
        train_grid_times: np.ndarray,
        val_event_seq_ids: np.ndarray,
        val_event_times: np.ndarray,
        val_grid_seq_ids: np.ndarray,
        val_grid_times: np.ndarray,
    ):
        self.source_ids = tuple(int(s) for s in source_ids)
        self.knots = np.asarray(knots, dtype=np.float64)
        self.train_arrays = train_arrays
        self.val_arrays = val_arrays
        self.train_event_seq_ids = train_event_seq_ids
        self.train_event_times = train_event_times
        self.train_grid_seq_ids = train_grid_seq_ids
        self.train_grid_times = train_grid_times
        self.val_event_seq_ids = val_event_seq_ids
        self.val_event_times = val_event_times
        self.val_grid_seq_ids = val_grid_seq_ids
        self.val_grid_times = val_grid_times
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def arrays(self, source: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = int(source)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        mats_tr_ev: list[np.ndarray] = []
        mats_tr_gr: list[np.ndarray] = []
        mats_va_ev: list[np.ndarray] = []
        mats_va_gr: list[np.ndarray] = []
        for basis_idx in range(self.knots.size):
            ker = basis_kernel(self.knots, basis_idx)
            tr_ev = compute_source_signal_matrix(
                self.train_arrays,
                {key: ker},
                source_ids=(key,),
                seq_ids=self.train_event_seq_ids,
                times=self.train_event_times,
            )[:, 0]
            tr_gr = compute_source_signal_matrix(
                self.train_arrays,
                {key: ker},
                source_ids=(key,),
                seq_ids=self.train_grid_seq_ids,
                times=self.train_grid_times,
            )[:, 0]
            va_ev = compute_source_signal_matrix(
                self.val_arrays,
                {key: ker},
                source_ids=(key,),
                seq_ids=self.val_event_seq_ids,
                times=self.val_event_times,
            )[:, 0]
            va_gr = compute_source_signal_matrix(
                self.val_arrays,
                {key: ker},
                source_ids=(key,),
                seq_ids=self.val_grid_seq_ids,
                times=self.val_grid_times,
            )[:, 0]
            mats_tr_ev.append(tr_ev)
            mats_tr_gr.append(tr_gr)
            mats_va_ev.append(va_ev)
            mats_va_gr.append(va_gr)
        out = (
            np.column_stack(mats_tr_ev),
            np.column_stack(mats_tr_gr),
            np.column_stack(mats_va_ev),
            np.column_stack(mats_va_gr),
        )
        self.cache[key] = out
        return out


def normalized_kernel_response(basis_matrix: np.ndarray, raw_heights: np.ndarray, area_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(np.asarray(raw_heights, dtype=np.float64), 1e-8)
    area = float(np.dot(area_weights, x))
    if area <= 1e-12:
        area = 1.0
    heights = x / area
    z = basis_matrix @ heights
    return np.maximum(z, 0.0), heights


def base_rate_fit(n_events: int, exposure: float) -> float:
    return max(float(n_events) / max(float(exposure), 1e-8), 1e-8)


def summarize_results(
    *,
    subsets: list[tuple[int, ...]],
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    target: int,
) -> set[tuple[tuple[int, ...], str, int]]:
    preds = set()
    for idx in sorted(exc_params):
        preds.add((tuple(sorted(int(s) for s in subsets[int(idx)])), "exc", int(target)))
    for idx in sorted(inh_params):
        preds.add((tuple(sorted(int(s) for s in subsets[int(idx)])), "inh", int(target)))
    return preds

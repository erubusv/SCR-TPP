"""Partition-free conjunctive logical-rule initializer.

Mainline philosophy:
1. Estimate one nonnegative fixed-knot continuous score kernel per source from
   source-target lag excess.
2. Constrain each source kernel directly as a pointwise score function with
       0 <= g_s(tau) <= 1.
3. Build source witness activations
       a_s(t) = max_i g_s(t - t_i^s).
4. Define each subset rule feature as a conjunction
       rho_U(t) = prod(a_s(t) for s in U).
5. Use a point-process objective
       sum log lambda(t_i) - integral lambda(t) dt
   with sign-exclusive subset states {off, exc, inh}.

This file is the clean replacement path for overlap-rule discovery. It does not
use partitions or anchor classification in the default path.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import yaml
from numba import njit


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
        self.heights = normalize_piecewise_score(self.heights)
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

    def eval_value(self, dt: np.ndarray | float) -> np.ndarray | float:
        vals = np.interp(
            np.asarray(dt, dtype=np.float64),
            self.knots,
            self.heights,
            left=0.0,
            right=0.0,
        )
        vals = np.clip(vals, 0.0, 1.0)
        if np.ndim(dt) == 0:
            return float(vals)
        return vals


@dataclass(frozen=True)
class WitnessQueryData:
    num_queries: int
    query_index: np.ndarray
    left_index: np.ndarray
    right_index: np.ndarray
    left_weight: np.ndarray
    right_weight: np.ndarray


@njit(cache=True)
def _build_witness_query_arrays_for_sequence(
    q: np.ndarray,
    src_events: np.ndarray,
    knots: np.ndarray,
    max_support: float,
):
    n_q = q.size
    n_src = src_events.size
    if n_q == 0 or n_src == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float64)
        return empty_i, empty_i, empty_i, empty_f, empty_f

    total_pairs = 0
    left = 0
    right = 0
    for j in range(n_q):
        t = q[j]
        while left < n_src and src_events[left] < t - max_support:
            left += 1
        while right < n_src and src_events[right] < t:
            right += 1
        if right > left:
            total_pairs += right - left

    query_index = np.empty((total_pairs,), dtype=np.int64)
    left_index = np.empty((total_pairs,), dtype=np.int64)
    right_index = np.empty((total_pairs,), dtype=np.int64)
    left_weight = np.empty((total_pairs,), dtype=np.float64)
    right_weight = np.empty((total_pairs,), dtype=np.float64)

    left = 0
    right = 0
    out = 0
    num_knots = knots.size
    for j in range(n_q):
        t = q[j]
        while left < n_src and src_events[left] < t - max_support:
            left += 1
        while right < n_src and src_events[right] < t:
            right += 1
        for k in range(left, right):
            dt = t - src_events[k]
            if dt <= 0.0 or dt > max_support:
                continue
            li = 0
            while li + 1 < num_knots - 1 and knots[li + 1] < dt:
                li += 1
            ri = li + 1
            lk = knots[li]
            rk = knots[ri]
            delta = rk - lk
            if delta <= 1e-12:
                rw = 0.0
            else:
                rw = (dt - lk) / delta
                if rw < 0.0:
                    rw = 0.0
                elif rw > 1.0:
                    rw = 1.0
            query_index[out] = j
            left_index[out] = li
            right_index[out] = ri
            right_weight[out] = rw
            left_weight[out] = 1.0 - rw
            out += 1

    return (
        query_index[:out],
        left_index[:out],
        right_index[:out],
        left_weight[:out],
        right_weight[:out],
    )


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


def normalize_piecewise_score(heights: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(heights, dtype=np.float64), 0.0)
    if x.size == 0:
        return x
    peak = float(np.max(x))
    if peak <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip(x / peak, 0.0, 1.0)


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


def collect_source_target_lags(
    seq_arrays: list[dict[int, np.ndarray]],
    *,
    source: int,
    target: int,
    max_lag: float,
) -> np.ndarray:
    lags: list[np.ndarray] = []
    for by_type in seq_arrays:
        src = by_type.get(int(source), np.zeros((0,), dtype=np.float64))
        tgt = by_type.get(int(target), np.zeros((0,), dtype=np.float64))
        if src.size == 0 or tgt.size == 0:
            continue
        for t in tgt:
            left = np.searchsorted(src, t - float(max_lag), side="left")
            right = np.searchsorted(src, t, side="left")
            if right <= left:
                continue
            dts = t - src[left:right]
            dts = dts[(dts > 0.0) & (dts <= float(max_lag))]
            if dts.size:
                lags.append(np.asarray(dts, dtype=np.float64))
    if not lags:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(lags).astype(np.float64, copy=False)


def _strictly_increasing_knots(values: np.ndarray, *, max_lag: float, num_knots: int) -> np.ndarray:
    knots = np.asarray(values, dtype=np.float64).copy()
    if knots.size != int(num_knots):
        raise ValueError("knot candidate size must equal num_knots")
    max_lag = float(max_lag)
    knots[0] = 0.0
    knots[-1] = max_lag
    if int(num_knots) <= 2:
        return knots
    min_gap = max(max_lag, 1.0) * 1e-9
    for i in range(1, int(num_knots) - 1):
        lower = knots[i - 1] + min_gap
        upper = max_lag - float(int(num_knots) - 1 - i) * min_gap
        knots[i] = min(max(float(knots[i]), lower), upper)
    return knots


def build_lag_quantile_knots(
    seq_arrays: list[dict[int, np.ndarray]],
    *,
    source: int,
    target: int,
    max_lag: float,
    num_knots: int,
) -> np.ndarray:
    num_knots = int(num_knots)
    if num_knots < 2:
        raise ValueError("num_knots must be at least 2")
    max_lag = float(max_lag)
    lags = collect_source_target_lags(
        seq_arrays,
        source=int(source),
        target=int(target),
        max_lag=max_lag,
    )
    if lags.size < max(1, num_knots - 2):
        return np.linspace(0.0, max_lag, num_knots, dtype=np.float64)
    if num_knots == 2:
        return np.asarray([0.0, max_lag], dtype=np.float64)
    probs = np.linspace(0.0, 1.0, num_knots, dtype=np.float64)
    candidates = np.quantile(lags, probs, method="linear")
    candidates[0] = 0.0
    candidates[-1] = max_lag
    return _strictly_increasing_knots(candidates, max_lag=max_lag, num_knots=num_knots)


def init_piecewise_heights(
    *,
    hist: np.ndarray,
    edges: np.ndarray,
    expected: np.ndarray,
    knots: np.ndarray,
    sign: str = "exc",
) -> np.ndarray:
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.asarray(hist, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if str(sign) == "inh":
        signal = np.maximum(expected - hist, 0.0)
    else:
        signal = np.maximum(hist - expected, 0.0)
    if float(np.sum(signal)) <= 1e-12:
        return np.ones_like(knots, dtype=np.float64)
    heights = np.interp(
        np.asarray(knots, dtype=np.float64),
        centers,
        signal,
        left=float(signal[0]),
        right=0.0,
    )
    return normalize_piecewise_score(np.maximum(heights, 0.0))


def estimate_source_kernels(
    seq_arrays: list[dict[int, np.ndarray]],
    *,
    source_ids: tuple[int, ...],
    target: int,
    max_lag: float,
    num_bins: int,
    num_knots: int,
    time_horizon: float,
    sign: str = "exc",
    knot_strategy: str = "uniform",
) -> dict[int, SourceKernel]:
    target_count = 0
    total_time = float(len(seq_arrays)) * float(time_horizon)
    source_counts = {int(s): 0 for s in source_ids}
    for by_type in seq_arrays:
        target_count += int(by_type.get(int(target), np.zeros((0,), dtype=np.float64)).size)
        for src in source_ids:
            source_counts[int(src)] += int(by_type.get(int(src), np.zeros((0,), dtype=np.float64)).size)

    edges = np.linspace(0.0, float(max_lag), int(num_bins) + 1)
    bin_width = float(edges[1] - edges[0])
    uniform_knots = np.linspace(0.0, float(max_lag), int(num_knots), dtype=np.float64)
    kernels: dict[int, SourceKernel] = {}
    for src in source_ids:
        if str(knot_strategy) == "lag_quantile":
            knots = build_lag_quantile_knots(
                seq_arrays,
                source=int(src),
                target=int(target),
                max_lag=float(max_lag),
                num_knots=int(num_knots),
            )
        elif str(knot_strategy) == "uniform":
            knots = uniform_knots.copy()
        else:
            raise ValueError(f"unknown knot_strategy={knot_strategy!r}")
        hist = collect_lag_hist(
            seq_arrays,
            source=int(src),
            target=int(target),
            max_lag=float(max_lag),
            num_bins=int(num_bins),
        )
        src_rate = float(source_counts[int(src)]) / max(total_time, 1e-8)
        expected = np.full_like(hist, src_rate * float(target_count) * bin_width)
        kernels[int(src)] = SourceKernel(
            source=int(src),
            knots=knots.copy(),
            heights=init_piecewise_heights(
                hist=hist,
                edges=edges,
                expected=expected,
                knots=knots,
                sign=str(sign),
            ),
        )
    return kernels


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


def compute_source_witness_query_data(
    seq_arrays: list[dict[int, np.ndarray]],
    *,
    source: int,
    knots: np.ndarray,
    seq_ids: np.ndarray,
    times: np.ndarray,
) -> WitnessQueryData:
    knots = np.asarray(knots, dtype=np.float64)
    num_queries = int(len(times))
    if num_queries == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float64)
        return WitnessQueryData(
            num_queries=0,
            query_index=empty_i,
            left_index=empty_i,
            right_index=empty_i,
            left_weight=empty_f,
            right_weight=empty_f,
        )

    query_index_parts: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    left_weight_parts: list[np.ndarray] = []
    right_weight_parts: list[np.ndarray] = []

    order = np.argsort(seq_ids, kind="stable")
    seq_ids_sorted = seq_ids[order]
    uniq, starts = np.unique(seq_ids_sorted, return_index=True)
    ends = np.r_[starts[1:], len(order)]
    max_support = float(knots[-1])

    for seq_idx, st, en in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        idxs = order[st:en]
        q = np.asarray(times[idxs], dtype=np.float64)
        src_events = seq_arrays[int(seq_idx)].get(int(source), np.zeros((0,), dtype=np.float64))
        if src_events.size == 0:
            continue
        local_q_idx, left_idx, right_idx, left_w, right_w = _build_witness_query_arrays_for_sequence(
            q,
            src_events,
            knots,
            max_support,
        )
        if local_q_idx.size == 0:
            continue
        query_index_parts.append(np.asarray(idxs[local_q_idx], dtype=np.int64))
        left_parts.append(left_idx)
        right_parts.append(right_idx)
        left_weight_parts.append(left_w)
        right_weight_parts.append(right_w)

    if not query_index_parts:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float64)
        return WitnessQueryData(
            num_queries=num_queries,
            query_index=empty_i,
            left_index=empty_i,
            right_index=empty_i,
            left_weight=empty_f,
            right_weight=empty_f,
        )
    return WitnessQueryData(
        num_queries=num_queries,
        query_index=np.concatenate(query_index_parts),
        left_index=np.concatenate(left_parts),
        right_index=np.concatenate(right_parts),
        left_weight=np.concatenate(left_weight_parts),
        right_weight=np.concatenate(right_weight_parts),
    )


def witness_response_from_query_data(data: WitnessQueryData, raw_heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heights = normalize_piecewise_score(raw_heights)
    out = np.zeros((int(data.num_queries),), dtype=np.float64)
    if data.query_index.size == 0:
        return out, heights
    vals = (
        data.left_weight * heights[data.left_index]
        + data.right_weight * heights[data.right_index]
    )
    np.maximum.at(out, data.query_index, np.clip(vals, 0.0, 1.0))
    return out, heights


def source_train_presence_masks(
    basis_cache: "SourceBasisCache",
    *,
    source_ids: tuple[int, ...],
) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for src in source_ids:
        tr_ev, tr_gr, _va_ev, _va_gr = basis_cache.arrays(int(src))
        mask = np.zeros((int(tr_ev.num_queries) + int(tr_gr.num_queries),), dtype=bool)
        if tr_ev.query_index.size:
            mask[np.asarray(tr_ev.query_index, dtype=np.int64)] = True
        if tr_gr.query_index.size:
            offset = int(tr_ev.num_queries)
            mask[offset + np.asarray(tr_gr.query_index, dtype=np.int64)] = True
        masks[int(src)] = mask
    return masks


def source_overlap_components_from_masks(
    source_ids: tuple[int, ...],
    masks: dict[int, np.ndarray],
) -> list[tuple[int, ...]]:
    source_ids = tuple(int(s) for s in source_ids)
    n = len(source_ids)
    adj = [[] for _ in range(n)]
    for i in range(n):
        si = int(source_ids[i])
        mi = np.asarray(masks[si], dtype=bool)
        for j in range(i + 1, n):
            sj = int(source_ids[j])
            mj = np.asarray(masks[sj], dtype=bool)
            if bool(np.any(mi & mj)):
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * n
    comps: list[tuple[int, ...]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(int(source_ids[cur]))
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        comps.append(tuple(sorted(comp)))
    return comps


def feasible_subset_list(
    *,
    source_ids: tuple[int, ...],
    max_order: int,
    basis_cache: "SourceBasisCache",
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], dict[int, np.ndarray]]:
    masks = source_train_presence_masks(basis_cache, source_ids=source_ids)
    components = source_overlap_components_from_masks(source_ids, masks)
    out: list[tuple[int, ...]] = []
    for comp in components:
        comp = tuple(sorted(int(s) for s in comp))
        if not comp:
            continue
        comp_pos = {int(src): i for i, src in enumerate(comp)}
        comp_masks = {
            int(src): np.asarray(masks[int(src)], dtype=bool)
            for src in comp
        }
        level: list[tuple[tuple[int, ...], np.ndarray]] = [
            ((int(src),), comp_masks[int(src)].copy())
            for src in comp
            if bool(np.any(comp_masks[int(src)]))
        ]
        out.extend(subset for subset, _joint in level)
        order = 1
        while level and order < min(int(max_order), len(comp)):
            next_level: list[tuple[tuple[int, ...], np.ndarray]] = []
            for subset, joint in level:
                start = int(comp_pos[int(subset[-1])]) + 1
                for src in comp[start:]:
                    joint_next = joint & comp_masks[int(src)]
                    if not bool(np.any(joint_next)):
                        continue
                    next_level.append((subset + (int(src),), joint_next))
            out.extend(subset for subset, _joint in next_level)
            level = next_level
            order += 1
    out = sorted(set(out), key=lambda subset: (len(subset), subset))
    return out, components, masks


class SourceBasisCache:
    def __init__(
        self,
        *,
        source_ids: tuple[int, ...],
        knots: np.ndarray,
        source_knots: dict[int, np.ndarray] | None = None,
        rule_source_knots: dict[tuple[int, int], np.ndarray] | None = None,
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
        if source_knots is None:
            self.source_knots = {int(s): self.knots.copy() for s in self.source_ids}
        else:
            self.source_knots = {
                int(s): np.asarray(source_knots[int(s)], dtype=np.float64)
                for s in self.source_ids
            }
        self.rule_source_knots = {
            (int(idx), int(src)): np.asarray(value, dtype=np.float64)
            for (idx, src), value in (rule_source_knots or {}).items()
        }
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
        self.cache: dict[tuple[int | None, int], tuple[WitnessQueryData, WitnessQueryData, WitnessQueryData, WitnessQueryData]] = {}

    def knots_for_source(self, source: int) -> np.ndarray:
        return np.asarray(self.source_knots.get(int(source), self.knots), dtype=np.float64)

    def knots_for_rule_source(self, rule_idx: int | None, source: int) -> np.ndarray:
        if rule_idx is not None:
            hit = self.rule_source_knots.get((int(rule_idx), int(source)))
            if hit is not None:
                return np.asarray(hit, dtype=np.float64)
        return self.knots_for_source(int(source))

    def arrays_for_rule_source(
        self,
        rule_idx: int | None,
        source: int,
    ) -> tuple[WitnessQueryData, WitnessQueryData, WitnessQueryData, WitnessQueryData]:
        key = (None if rule_idx is None else int(rule_idx), int(source))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        source_key = int(source)
        knots = self.knots_for_rule_source(rule_idx, source_key)
        out = (
            compute_source_witness_query_data(
                self.train_arrays,
                source=source_key,
                knots=knots,
                seq_ids=self.train_event_seq_ids,
                times=self.train_event_times,
            ),
            compute_source_witness_query_data(
                self.train_arrays,
                source=source_key,
                knots=knots,
                seq_ids=self.train_grid_seq_ids,
                times=self.train_grid_times,
            ),
            compute_source_witness_query_data(
                self.val_arrays,
                source=source_key,
                knots=knots,
                seq_ids=self.val_event_seq_ids,
                times=self.val_event_times,
            ),
            compute_source_witness_query_data(
                self.val_arrays,
                source=source_key,
                knots=knots,
                seq_ids=self.val_grid_seq_ids,
                times=self.val_grid_times,
            ),
        )
        self.cache[key] = out
        return out

    def arrays(self, source: int) -> tuple[WitnessQueryData, WitnessQueryData, WitnessQueryData, WitnessQueryData]:
        return self.arrays_for_rule_source(None, int(source))


def base_rate_fit(n_events: int, exposure: float) -> float:
    return max(float(n_events) / max(float(exposure), 1e-8), 1e-8)


def summarize_results(
    *,
    subsets: list[tuple[int, ...]],
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    target: int,
    beta_threshold: float = 0.0,
) -> set[tuple[tuple[int, ...], str, int]]:
    preds = set()
    for idx in sorted(exc_params):
        if float(exc_params[idx]) > float(beta_threshold):
            preds.add((tuple(sorted(int(s) for s in subsets[int(idx)])), "exc", int(target)))
    for idx in sorted(inh_params):
        if float(inh_params[idx]) > float(beta_threshold):
            preds.add((tuple(sorted(int(s) for s in subsets[int(idx)])), "inh", int(target)))
    return preds

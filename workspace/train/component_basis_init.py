"""Fixed-target component-basis discovery and initialisation.

Prototype implementation of the design:
1. Pairwise lag screening
2. Bounded per-source evidence q_s(t)
3. Residual higher-order hyperedge discovery
4. Source-component assembly
5. Conjunctive subset basis construction
6. Target-specific sparse fit with component penalties
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import torch


@dataclass
class SourceDef:
    source: int
    score: float
    peak: float
    width: float
    beta: float
    alpha: float


@dataclass
class AtomDef:
    atom_id: int
    component_id: int
    sources: tuple[int, ...]
    order: int
    score: float
    sign: int


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _banner(msg: str) -> None:
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60)


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
    rates = counts / max(total_time, 1e-9)
    return rates, total_time


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


def phase1_pairwise_screen(
    data_list: list[dict],
    num_types: int,
    fixed_target: int,
    *,
    delta_t: float = 0.1,
    max_lag: float = 10.0,
    source_pool_topk: int = 8,
) -> dict:
    """Pairwise lag screening for source relevance and lag priors."""
    _banner("Component Basis Phase 1: Pairwise Screening")
    rates, total_time = _estimate_base_rates(data_list, num_types)
    bins = max(1, int(math.ceil(max_lag / max(delta_t, 1e-6))))
    centers = (np.arange(bins, dtype=np.float64) + 0.5) * delta_t

    pair_stats = []
    total_target = 0
    for item in data_list:
        e = np.asarray(item["event"], dtype=np.int64)
        total_target += int(np.sum(e == int(fixed_target)))

    for src in range(num_types):
        if src == int(fixed_target):
            continue
        hist = np.zeros(bins, dtype=np.float64)
        for item in data_list:
            t = np.asarray(item["time"], dtype=np.float64)
            e = np.asarray(item["event"], dtype=np.int64)
            if len(t) == 0:
                continue
            src_times = t[e == int(src)]
            tgt_times = t[e == int(fixed_target)]
            _sequence_lag_hist(src_times, tgt_times, max_lag, delta_t, hist)

        expected = float(rates[src]) * float(delta_t) * float(max(total_target, 1))
        excess = hist - expected
        weights = np.abs(excess)
        score = float(weights.sum())
        if score <= 1e-8:
            peak = max(delta_t, 0.5)
            width = min(max_lag, max(1.5 * peak, 1.0))
        else:
            mu = float((centers * weights).sum() / max(weights.sum(), 1e-8))
            var = float(((centers - mu) ** 2 * weights).sum() / max(weights.sum(), 1e-8))
            sigma = max(math.sqrt(max(var, 1e-8)), delta_t)
            peak = float(np.clip(mu, delta_t, max_lag * 0.8))
            width = float(np.clip(mu + 2.0 * sigma, peak + delta_t, max_lag))

        pair_stats.append({
            "source": int(src),
            "score": score,
            "peak": peak,
            "width": width,
        })

    pair_stats.sort(key=lambda x: float(x["score"]), reverse=True)
    if source_pool_topk > 0:
        pair_stats = pair_stats[: min(source_pool_topk, len(pair_stats))]
    print("  Source pool:", [int(x["source"]) for x in pair_stats])
    return {
        "base_rates": rates,
        "total_time": total_time,
        "pair_stats": pair_stats,
    }


def _raw_source_signal_at_queries(query_times: np.ndarray, src_times: np.ndarray, peak: float, width: float) -> np.ndarray:
    out = np.zeros((len(query_times),), dtype=np.float32)
    if len(query_times) == 0 or len(src_times) == 0:
        return out
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
        out[qi] = float(_triangular_kernel(dts, peak, width).sum())
    return out


def _estimate_source_bounding(
    data_list: list[dict],
    source_stats: list[dict],
    *,
    fixed_target: int,
    beta_quantile: float = 0.6,
) -> list[SourceDef]:
    source_defs: list[SourceDef] = []
    for stat in source_stats:
        src = int(stat["source"])
        peak = float(stat["peak"])
        width = float(stat["width"])
        target_z = []
        for item in data_list:
            times = np.asarray(item["time"], dtype=np.float64)
            events = np.asarray(item["event"], dtype=np.int64)
            if len(times) == 0:
                continue
            src_times = times[events == src]
            tgt_times = times[events == int(fixed_target)]
            if len(src_times) == 0 or len(tgt_times) == 0:
                continue
            z = _raw_source_signal_at_queries(tgt_times, src_times, peak, width)
            if len(z) > 0:
                target_z.append(z.astype(np.float64))
        if target_z:
            z_all = np.concatenate(target_z)
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
        source_defs.append(
            SourceDef(
                source=src,
                score=float(stat["score"]),
                peak=peak,
                width=width,
                beta=beta,
                alpha=alpha,
            )
        )
    return source_defs


def phase2_source_evidence(
    data_list: list[dict],
    phase1_state: dict,
    *,
    fixed_target: int,
    beta_quantile: float = 0.6,
) -> dict:
    """Bounded per-source evidence q_s(t)."""
    _banner("Component Basis Phase 2: Bounded Source Evidence")
    source_defs = _estimate_source_bounding(
        data_list,
        phase1_state["pair_stats"],
        fixed_target=fixed_target,
        beta_quantile=beta_quantile,
    )
    for sd in source_defs:
        print(
            f"  src={sd.source} score={sd.score:.3f} peak={sd.peak:.3f} "
            f"width={sd.width:.3f} beta={sd.beta:.3f} alpha={sd.alpha:.3f}"
        )
    return {
        "base_rates": phase1_state["base_rates"],
        "source_defs": source_defs,
    }


def _build_query_grid(data_list: list[dict], *, int_grid_mult: int = 2) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    total_events = sum(len(item["time"]) for item in data_list)
    m_grid = max(total_events * max(int_grid_mult, 1), 1)
    all_T = [float(np.asarray(item["time"], dtype=np.float64)[-1]) + 1e-6 for item in data_list if len(item["time"]) > 0]
    if not all_T:
        return [], [], []
    sum_T = max(sum(all_T), 1e-9)
    grid_times_list: list[np.ndarray] = []
    grid_weights_list: list[np.ndarray] = []
    seq_ids: list[np.ndarray] = []
    seq_local = 0
    for item in data_list:
        t = np.asarray(item["time"], dtype=np.float64)
        if len(t) == 0:
            continue
        T_s = float(t[-1]) + 1e-6
        n_pts = max(8, int(m_grid * T_s / sum_T))
        g_t = np.linspace(0.0, T_s, n_pts + 1)
        dt = T_s / n_pts
        w = np.full(n_pts + 1, dt, dtype=np.float32)
        w[0] *= 0.5
        w[-1] *= 0.5
        grid_times_list.append(g_t)
        grid_weights_list.append(w)
        seq_ids.append(np.full(n_pts + 1, seq_local, dtype=np.int64))
        seq_local += 1
    return grid_times_list, grid_weights_list, seq_ids


def _sequence_q_matrix(query_times: np.ndarray, event_times: np.ndarray, event_types: np.ndarray, source_defs: list[SourceDef]) -> np.ndarray:
    q = np.zeros((len(query_times), len(source_defs)), dtype=np.float32)
    if len(query_times) == 0:
        return q
    for j, sd in enumerate(source_defs):
        src_times = event_times[event_types == int(sd.source)]
        if len(src_times) == 0:
            continue
        z = _raw_source_signal_at_queries(query_times, src_times, sd.peak, sd.width)
        q[:, j] = 1.0 - np.exp(-sd.alpha * np.maximum(z - sd.beta, 0.0))
    return np.clip(q, 0.0, 1.0)


def _build_source_cache(
    data_list: list[dict],
    source_defs: list[SourceDef],
    *,
    fixed_target: int,
    num_types: int,
    int_grid_mult: int = 2,
) -> dict:
    rows_q: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    grid_q_rows: list[np.ndarray] = []
    grid_w_rows: list[np.ndarray] = []

    grid_times_list, grid_weights_list, _ = _build_query_grid(data_list, int_grid_mult=int_grid_mult)

    seq_local = 0
    for item in data_list:
        times = np.asarray(item["time"], dtype=np.float64)
        events = np.asarray(item["event"], dtype=np.int64)
        if len(times) == 0:
            continue

        q_ev = _sequence_q_matrix(times, times, events, source_defs)
        y = (events == int(fixed_target)).astype(np.float32)
        rows_q.append(q_ev)
        rows_y.append(y)

        g_t = grid_times_list[seq_local]
        g_w = grid_weights_list[seq_local]
        q_gr = _sequence_q_matrix(g_t, times, events, source_defs)
        grid_q_rows.append(q_gr)
        grid_w_rows.append(g_w)
        seq_local += 1

    if rows_q:
        event_q = np.concatenate(rows_q, axis=0).astype(np.float32)
        event_y = np.concatenate(rows_y, axis=0).astype(np.float32)
    else:
        event_q = np.zeros((1, len(source_defs)), dtype=np.float32)
        event_y = np.zeros((1,), dtype=np.float32)
    if grid_q_rows:
        grid_q = np.concatenate(grid_q_rows, axis=0).astype(np.float32)
        grid_w = np.concatenate(grid_w_rows, axis=0).astype(np.float32)
    else:
        grid_q = np.zeros((1, len(source_defs)), dtype=np.float32)
        grid_w = np.ones((1,), dtype=np.float32)
    return {
        "event_q": event_q,
        "event_y": event_y,
        "grid_q": grid_q,
        "grid_w": grid_w,
        "num_types": int(num_types),
        "fixed_target": int(fixed_target),
    }


def _ridge_project(y: np.ndarray, X: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32)
    gram = X.T @ X
    gram.flat[:: gram.shape[0] + 1] += ridge
    rhs = X.T @ y
    try:
        sol = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return sol.astype(np.float32)


def _subset_product(q: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray:
    if len(idxs) == 1:
        return q[:, idxs[0]].astype(np.float32)
    prod = np.ones((q.shape[0],), dtype=np.float32)
    for idx in idxs:
        prod *= q[:, idx]
    return prod


def _residual_interaction_score(
    q_train: np.ndarray,
    y_train: np.ndarray,
    q_val: np.ndarray,
    y_val: np.ndarray,
    idxs: tuple[int, ...],
    ridge: float = 1e-4,
    support_tau: float = 20.0,
) -> tuple[float, int]:
    p_train = _subset_product(q_train, idxs)
    p_val = _subset_product(q_val, idxs)
    support = float(p_train.sum())
    strict_subsets = []
    for order in range(1, len(idxs)):
        strict_subsets.extend(combinations(idxs, order))
    if strict_subsets:
        x_train = np.stack([_subset_product(q_train, ss) for ss in strict_subsets], axis=1)
        x_val = np.stack([_subset_product(q_val, ss) for ss in strict_subsets], axis=1)
        alpha = _ridge_project(p_train, x_train, ridge=ridge)
        r_train = p_train - x_train @ alpha
        r_val = p_val - x_val @ alpha
    else:
        r_train = p_train
        r_val = p_val

    yc_train = y_train - float(y_train.mean())
    yc_val = y_val - float(y_val.mean())
    num_train = float(np.dot(yc_train, r_train))
    num_val = float(np.dot(yc_val, r_val))
    den_train = float(np.dot(r_train, r_train)) + 1e-8
    den_val = float(np.dot(r_val, r_val)) + 1e-8
    if num_train == 0.0 or num_val == 0.0 or np.sign(num_train) != np.sign(num_val):
        return 0.0, 0
    score_train = (num_train * num_train) / den_train
    score_val = (num_val * num_val) / den_val
    shrink = math.sqrt(support / max(support + support_tau, 1e-8))
    sign = 1 if num_train > 0.0 else -1
    return float(min(score_train, score_val) * shrink), int(sign)


def phase3_discover_hyperedges(
    train_cache: dict,
    val_cache: dict,
    source_defs: list[SourceDef],
    *,
    max_source_order: int = 3,
    topk_per_order: int = 8,
    support_tau: float = 20.0,
    min_score: float = 1e-4,
) -> list[dict]:
    """Residual higher-order hyperedge discovery."""
    _banner("Component Basis Phase 3: Hyperedge Discovery")
    rows = []
    source_ids = [int(sd.source) for sd in source_defs]
    local_to_source = {i: s for i, s in enumerate(source_ids)}

    for order in range(2, max_source_order + 1):
        for idxs in combinations(range(len(source_defs)), order):
            score, sign = _residual_interaction_score(
                train_cache["event_q"],
                train_cache["event_y"],
                val_cache["event_q"],
                val_cache["event_y"],
                idxs,
                support_tau=support_tau,
            )
            if score <= min_score or sign == 0:
                continue
            rows.append({
                "sources": tuple(local_to_source[i] for i in idxs),
                "order": int(order),
                "score": float(score),
                "sign": int(sign),
            })

    selected = []
    for order in range(2, max_source_order + 1):
        rows_o = [r for r in rows if int(r["order"]) == order]
        rows_o.sort(key=lambda x: float(x["score"]), reverse=True)
        selected.extend(rows_o[: min(topk_per_order, len(rows_o))])

    for row in selected:
        print(f"  order={row['order']} sources={row['sources']} score={row['score']:.4f} sign={row['sign']:+d}")
    return selected


def phase4_build_components(
    source_defs: list[SourceDef],
    hyperedges: list[dict],
    *,
    max_source_order: int = 3,
) -> dict:
    """Assemble source components and subset basis."""
    _banner("Component Basis Phase 4: Components and Basis")
    source_ids = [int(sd.source) for sd in source_defs]
    idx_by_source = {s: i for i, s in enumerate(source_ids)}
    uf = _UnionFind(len(source_ids))
    for edge in hyperedges:
        srcs = [int(s) for s in edge["sources"]]
        for a, b in combinations(srcs, 2):
            uf.union(idx_by_source[a], idx_by_source[b])

    comps: dict[int, list[int]] = {}
    for src in source_ids:
        root = uf.find(idx_by_source[src])
        comps.setdefault(root, []).append(int(src))
    components = sorted([tuple(sorted(v)) for v in comps.values()], key=lambda x: (len(x), x))

    edge_map = {
        tuple(sorted(int(s) for s in edge["sources"])): edge
        for edge in hyperedges
    }
    atoms: list[AtomDef] = []
    atom_id = 0
    for comp_id, comp in enumerate(components):
        for order in range(1, min(max_source_order, len(comp)) + 1):
            for srcs in combinations(comp, order):
                edge = edge_map.get(tuple(int(s) for s in srcs))
                if order == 1:
                    score = 0.0
                    sign = 0
                else:
                    score = float(edge["score"]) if edge is not None else 0.0
                    sign = int(edge["sign"]) if edge is not None else 0
                atoms.append(
                    AtomDef(
                        atom_id=atom_id,
                        component_id=comp_id,
                        sources=tuple(int(s) for s in srcs),
                        order=int(order),
                        score=score,
                        sign=sign,
                    )
                )
                atom_id += 1

    print("  Components:", components)
    print(f"  Total atoms: {len(atoms)}")
    return {
        "components": components,
        "atoms": atoms,
    }


def _build_atom_cache(source_cache: dict, source_defs: list[SourceDef], atoms: list[AtomDef]) -> tuple[np.ndarray, np.ndarray]:
    local_by_source = {int(sd.source): i for i, sd in enumerate(source_defs)}
    p_event = np.zeros((source_cache["event_q"].shape[0], len(atoms)), dtype=np.float32)
    p_grid = np.zeros((source_cache["grid_q"].shape[0], len(atoms)), dtype=np.float32)
    for atom in atoms:
        idxs = tuple(local_by_source[int(s)] for s in atom.sources)
        p_event[:, atom.atom_id] = _subset_product(source_cache["event_q"], idxs)
        p_grid[:, atom.atom_id] = _subset_product(source_cache["grid_q"], idxs)
    return p_event, p_grid


def _target_cached_nll(
    w_exc: torch.Tensor,
    w_inh: torch.Tensor,
    p_event: torch.Tensor,
    p_grid: torch.Tensor,
    y_event: torch.Tensor,
    base_tgt: torch.Tensor,
    grid_w: torch.Tensor,
) -> torch.Tensor:
    e_ev = torch.matmul(p_event, w_exc)
    i_ev = torch.matmul(p_event, w_inh).clamp(max=20.0)
    lam_ev = (base_tgt + e_ev).clamp(min=1e-8) * torch.exp(-i_ev)
    log_ll = (torch.log(lam_ev.clamp(min=1e-8)) * y_event).sum()

    e_gr = torch.matmul(p_grid, w_exc)
    i_gr = torch.matmul(p_grid, w_inh).clamp(max=20.0)
    lam_gr = (base_tgt + e_gr).clamp(min=0.0) * torch.exp(-i_gr)
    integral = (lam_gr * grid_w).sum()
    n_tgt = max(float(y_event.sum().item()), 1.0)
    return (-log_ll + integral) / n_tgt


def phase5_fit_component_basis(
    train_cache: dict,
    val_cache: dict,
    source_defs: list[SourceDef],
    components_state: dict,
    base_rates: np.ndarray,
    *,
    fixed_target: int,
    device: torch.device,
    steps: int = 300,
    lr: float = 5e-2,
    lambda_comp: float = 5e-3,
    lambda_atom: float = 2e-3,
    lambda_overlap: float = 1e-3,
) -> dict:
    """Target-specific sparse fit on the subset basis."""
    _banner("Component Basis Phase 5: Sparse Component Fit")
    atoms = components_state["atoms"]
    components = components_state["components"]
    p_event_np, p_grid_np = _build_atom_cache(train_cache, source_defs, atoms)
    vp_event_np, vp_grid_np = _build_atom_cache(val_cache, source_defs, atoms)

    atom_scores = []
    local_by_source = {int(sd.source): i for i, sd in enumerate(source_defs)}
    for atom in atoms:
        idxs = tuple(local_by_source[int(s)] for s in atom.sources)
        score, sign = _residual_interaction_score(
            train_cache["event_q"],
            train_cache["event_y"],
            val_cache["event_q"],
            val_cache["event_y"],
            idxs,
            support_tau=20.0,
        )
        atom_scores.append((float(score), int(sign)))

    score_vals = [abs(s) for s, _ in atom_scores if abs(s) > 0]
    score_scale = float(np.quantile(np.asarray(score_vals, dtype=np.float64), 0.9)) if score_vals else 1.0

    init_we = np.zeros((len(atoms),), dtype=np.float32)
    init_wi = np.zeros((len(atoms),), dtype=np.float32)
    for atom in atoms:
        score, sign = atom_scores[atom.atom_id]
        mag = float(np.clip(abs(score) / max(score_scale, 1e-6), 0.0, 1.0))
        base = 0.2 if atom.order == 1 else 0.3
        if sign > 0:
            init_we[atom.atom_id] = base * mag
        elif sign < 0:
            init_wi[atom.atom_id] = base * mag

    p_event = torch.as_tensor(p_event_np, dtype=torch.float32, device=device)
    p_grid = torch.as_tensor(p_grid_np, dtype=torch.float32, device=device)
    vp_event = torch.as_tensor(vp_event_np, dtype=torch.float32, device=device)
    vp_grid = torch.as_tensor(vp_grid_np, dtype=torch.float32, device=device)
    y_event = torch.as_tensor(train_cache["event_y"], dtype=torch.float32, device=device)
    vy_event = torch.as_tensor(val_cache["event_y"], dtype=torch.float32, device=device)
    grid_w = torch.as_tensor(train_cache["grid_w"], dtype=torch.float32, device=device)
    vgrid_w = torch.as_tensor(val_cache["grid_w"], dtype=torch.float32, device=device)
    base_tgt = torch.tensor(float(base_rates[int(fixed_target)]), dtype=torch.float32, device=device)

    w_exc = torch.nn.Parameter(torch.as_tensor(init_we, dtype=torch.float32, device=device))
    w_inh = torch.nn.Parameter(torch.as_tensor(init_wi, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([w_exc, w_inh], lr=lr)

    comp_to_atoms = []
    for comp_id in range(len(components)):
        idxs = [atom.atom_id for atom in atoms if int(atom.component_id) == comp_id]
        comp_to_atoms.append(torch.as_tensor(idxs, dtype=torch.long, device=device))

    order_pen = torch.as_tensor(
        [1.0 if atom.order == 1 else (1.15 if atom.order == 2 else 1.35) for atom in atoms],
        dtype=torch.float32,
        device=device,
    )

    best = None
    best_val = None
    for step in range(max(steps, 0)):
        opt.zero_grad()
        nll = _target_cached_nll(w_exc, w_inh, p_event, p_grid, y_event, base_tgt, grid_w)
        comp_pen = torch.tensor(0.0, dtype=torch.float32, device=device)
        for idxs in comp_to_atoms:
            v = torch.cat([w_exc[idxs], w_inh[idxs]], dim=0)
            comp_pen = comp_pen + torch.sqrt((v * v).sum() + 1e-8)
        atom_pen = torch.sqrt(w_exc * w_exc + w_inh * w_inh + 1e-8)
        loss = (
            nll
            + float(lambda_comp) * comp_pen
            + float(lambda_atom) * (order_pen * atom_pen).sum()
            + float(lambda_overlap) * (w_exc * w_inh).sum()
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            w_exc.clamp_(min=0.0)
            w_inh.clamp_(min=0.0)

        if step % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                val_nll = _target_cached_nll(w_exc, w_inh, vp_event, vp_grid, vy_event, base_tgt, vgrid_w)
            print(
                f"  step {step:3d} train_nll={float(nll.item()):.6f} "
                f"val_nll={float(val_nll.item()):.6f} "
                f"alive={int(((w_exc + w_inh) > 1e-4).sum().item())}"
            )
            if best_val is None or float(val_nll.item()) < best_val:
                best_val = float(val_nll.item())
                best = (w_exc.detach().clone(), w_inh.detach().clone())

    if best is None:
        best = (w_exc.detach().clone(), w_inh.detach().clone())
    return {
        "w_exc": best[0].detach().cpu().numpy(),
        "w_inh": best[1].detach().cpu().numpy(),
        "atoms": atoms,
        "components": components,
        "source_defs": source_defs,
        "base_rates": base_rates,
        "fixed_target": int(fixed_target),
    }


def build_component_basis_structure(
    data_list: list[dict],
    num_types: int,
    *,
    fixed_target: int,
    device: torch.device,
    delta_t: float = 0.1,
    max_lag: float = 10.0,
    source_pool_topk: int = 8,
    max_source_order: int = 3,
    topk_per_order: int = 8,
    val_ratio: float = 0.15,
    int_grid_mult: int = 2,
) -> dict:
    if len(data_list) <= 1:
        train_data = data_list
        val_data = data_list
    else:
        n_val = min(max(1, int(len(data_list) * val_ratio)), len(data_list) - 1)
        val_data = data_list[:n_val]
        train_data = data_list[n_val:]

    phase1 = phase1_pairwise_screen(
        train_data,
        num_types,
        fixed_target,
        delta_t=delta_t,
        max_lag=max_lag,
        source_pool_topk=source_pool_topk,
    )
    phase2 = phase2_source_evidence(
        train_data,
        phase1,
        fixed_target=fixed_target,
    )
    train_cache = _build_source_cache(
        train_data,
        phase2["source_defs"],
        fixed_target=fixed_target,
        num_types=num_types,
        int_grid_mult=int_grid_mult,
    )
    val_cache = _build_source_cache(
        val_data,
        phase2["source_defs"],
        fixed_target=fixed_target,
        num_types=num_types,
        int_grid_mult=1,
    )
    hyperedges = phase3_discover_hyperedges(
        train_cache,
        val_cache,
        phase2["source_defs"],
        max_source_order=max_source_order,
        topk_per_order=topk_per_order,
    )
    phase4 = phase4_build_components(
        phase2["source_defs"],
        hyperedges,
        max_source_order=max_source_order,
    )
    phase5 = phase5_fit_component_basis(
        train_cache,
        val_cache,
        phase2["source_defs"],
        phase4,
        phase2["base_rates"],
        fixed_target=fixed_target,
        device=device,
    )
    return phase5

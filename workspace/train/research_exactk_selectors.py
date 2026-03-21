"""Research sweeps for exact-k selector variants on component-basis outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import yaml

from workspace.train.ablate_component_basis import gt_rules_from_config, prepare_data
from workspace.train.component_basis_init import (
    _build_atom_cache,
    _build_source_cache,
    _subset_product,
    _target_cached_nll,
    build_component_basis_structure,
)


@dataclass
class AtomRow:
    srcs: tuple[int, ...]
    order: int
    weight: float
    weight_sign: str
    disc: float
    delta: float
    plain_score: float
    plain_sign: str | None
    gated_score: float
    gated_sign: str | None


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def score_stat(
    p_train: np.ndarray,
    y_train: np.ndarray,
    p_val: np.ndarray,
    y_val: np.ndarray,
    support_tau: float = 20.0,
) -> tuple[float, int]:
    support = float(p_train.sum())
    yc_train = y_train - y_train.mean()
    yc_val = y_val - y_val.mean()
    num_train = float(np.dot(yc_train, p_train))
    num_val = float(np.dot(yc_val, p_val))
    den_train = float(np.dot(p_train, p_train)) + 1e-8
    den_val = float(np.dot(p_val, p_val)) + 1e-8
    if num_train == 0.0 or num_val == 0.0 or np.sign(num_train) != np.sign(num_val):
        return 0.0, 0
    score = min((num_train * num_train) / den_train, (num_val * num_val) / den_val)
    score *= np.sqrt(support / max(support + support_tau, 1e-8))
    return float(score), (1 if num_train > 0.0 else -1)


def build_rows(
    train_data: list[dict],
    metadata: dict,
    fixed_target: int,
    *,
    delta_t: float = 0.1,
    max_lag: float = 10.0,
    source_pool_topk: int = 8,
    max_source_order: int = 3,
    topk_per_order: int = 8,
) -> tuple[dict, list[AtomRow]]:
    device = torch.device("cpu")
    structure = build_component_basis_structure(
        train_data,
        int(metadata["num_types"]),
        fixed_target=fixed_target,
        device=device,
        delta_t=delta_t,
        max_lag=max_lag,
        source_pool_topk=source_pool_topk,
        max_source_order=max_source_order,
        topk_per_order=topk_per_order,
    )

    n_val = min(max(1, int(len(train_data) * 0.15)), len(train_data) - 1)
    val_data = train_data[:n_val]
    tr_data = train_data[n_val:]

    source_defs = structure["source_defs"]
    atoms = structure["atoms"]
    families = [tuple(comp) for comp in structure["components"]]
    local_by_source = {int(sd.source): i for i, sd in enumerate(source_defs)}

    tr_cache = _build_source_cache(
        tr_data,
        source_defs,
        fixed_target=fixed_target,
        num_types=int(metadata["num_types"]),
        int_grid_mult=2,
    )
    va_cache = _build_source_cache(
        val_data,
        source_defs,
        fixed_target=fixed_target,
        num_types=int(metadata["num_types"]),
        int_grid_mult=1,
    )

    we = np.asarray(structure["w_exc"], dtype=np.float32)
    wi = np.asarray(structure["w_inh"], dtype=np.float32)
    ve, vg = _build_atom_cache(va_cache, source_defs, atoms)
    ve_t = torch.as_tensor(ve, dtype=torch.float32)
    vg_t = torch.as_tensor(vg, dtype=torch.float32)
    vy_t = torch.as_tensor(va_cache["event_y"], dtype=torch.float32)
    vgw_t = torch.as_tensor(va_cache["grid_w"], dtype=torch.float32)
    base_tgt = torch.tensor(float(structure["base_rates"][fixed_target]), dtype=torch.float32)
    full_nll = float(
        _target_cached_nll(torch.as_tensor(we), torch.as_tensor(wi), ve_t, vg_t, vy_t, base_tgt, vgw_t)
    )

    def prod(q: np.ndarray, srcs: tuple[int, ...]) -> np.ndarray:
        idxs = tuple(local_by_source[int(s)] for s in srcs)
        return _subset_product(q, idxs)

    def gated_feature(q: np.ndarray, srcs: tuple[int, ...]) -> np.ndarray:
        out = prod(q, srcs)
        srcset = set(int(s) for s in srcs)
        for fam in families:
            famset = set(int(s) for s in fam)
            if len(fam) <= len(srcs) or not srcset.issubset(famset):
                continue
            extra = tuple(sorted(int(s) for s in famset - srcset))
            out = out * (1.0 - prod(q, extra))
        return np.clip(out, 0.0, 1.0)

    rows: list[AtomRow] = []
    for atom in atoms:
        srcs = tuple(int(s) for s in atom.sources)
        a = int(atom.atom_id)
        weight = max(float(we[a]), float(wi[a]))
        if weight <= 1.0e-4:
            continue
        weight_sign = "exc" if float(we[a]) >= float(wi[a]) else "inh"
        we0 = we.copy()
        wi0 = wi.copy()
        we0[a] = 0.0
        wi0[a] = 0.0
        delta = float(
            _target_cached_nll(torch.as_tensor(we0), torch.as_tensor(wi0), ve_t, vg_t, vy_t, base_tgt, vgw_t)
            - full_nll
        )
        plain_tr = prod(tr_cache["event_q"], srcs)
        plain_va = prod(va_cache["event_q"], srcs)
        gated_tr = gated_feature(tr_cache["event_q"], srcs)
        gated_va = gated_feature(va_cache["event_q"], srcs)
        plain_score, plain_sign_raw = score_stat(plain_tr, tr_cache["event_y"], plain_va, va_cache["event_y"])
        gated_score, gated_sign_raw = score_stat(gated_tr, tr_cache["event_y"], gated_va, va_cache["event_y"])
        plain_sign = "exc" if plain_sign_raw > 0 else ("inh" if plain_sign_raw < 0 else None)
        gated_sign = "exc" if gated_sign_raw > 0 else ("inh" if gated_sign_raw < 0 else None)
        rows.append(
            AtomRow(
                srcs=srcs,
                order=int(atom.order),
                weight=weight,
                weight_sign=weight_sign,
                disc=float(atom.score),
                delta=delta,
                plain_score=plain_score,
                plain_sign=plain_sign,
                gated_score=gated_score,
                gated_sign=gated_sign,
            )
        )
    return structure, rows


def get_sign(row: AtomRow, sign_mode: str) -> str:
    if sign_mode == "weight":
        return row.weight_sign
    if sign_mode == "plain":
        return row.plain_sign or row.weight_sign
    if sign_mode == "gated":
        return row.gated_sign or row.plain_sign or row.weight_sign
    if sign_mode == "single_gated_other_plain":
        if row.order == 1:
            return row.gated_sign or row.plain_sign or row.weight_sign
        return row.plain_sign or row.weight_sign
    if sign_mode == "single_gated_other_weight":
        if row.order == 1:
            return row.gated_sign or row.plain_sign or row.weight_sign
        return row.weight_sign
    raise ValueError(sign_mode)


def rank_rows(
    rows: list[AtomRow],
    *,
    score_mode: tuple[str, ...],
    sign_mode: str,
    score_scale: dict[str, float],
    order_bonus: float,
    sup_pen: float,
    fixed_target: int,
    k: int,
) -> list[tuple[tuple[int, ...], str, int]]:
    ranked = []
    for row in rows:
        sign = get_sign(row, sign_mode)
        set_r = set(row.srcs)
        sup = 0.0
        for other in rows:
            if row is other or sign != get_sign(other, sign_mode):
                continue
            if set_r < set(other.srcs):
                sup = max(sup, float(other.disc) / max(score_scale["disc"], 1e-6))
        base = 0.0
        for key in score_mode:
            base += float(getattr(row, key)) / max(score_scale[key], 1e-6)
        adj = base + order_bonus * max(0, row.order - 1) - sup_pen * sup
        ranked.append((adj, row.order, (row.srcs, sign, fixed_target)))
    ranked.sort(reverse=True)
    return [x[2] for x in ranked[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixed_target", type=int, default=6)
    args = ap.parse_args()

    train_data, _, metadata = prepare_data(args.data)
    cfg = load_yaml(args.config)
    gt = gt_rules_from_config(cfg)
    _, rows = build_rows(train_data, metadata, int(args.fixed_target))

    score_scale = {}
    for key in ("weight", "disc", "delta", "plain_score", "gated_score"):
        vals = [float(getattr(r, key)) for r in rows if float(getattr(r, key)) > 0.0]
        score_scale[key] = (
            float(np.quantile(np.asarray(vals, dtype=np.float64), 0.9))
            if vals
            else 1.0
        )

    score_modes = [
        ("weight",),
        ("disc",),
        ("delta",),
        ("plain_score",),
        ("gated_score",),
        ("weight", "disc"),
        ("weight", "delta"),
        ("disc", "delta"),
        ("plain_score", "disc"),
        ("gated_score", "disc"),
        ("weight", "plain_score"),
        ("weight", "gated_score"),
    ]
    sign_modes = [
        "weight",
        "plain",
        "gated",
        "single_gated_other_plain",
        "single_gated_other_weight",
    ]

    results = []
    for score_mode, sign_mode, order_bonus, sup_pen in product(
        score_modes,
        sign_modes,
        [0.0, 0.1, 0.2, 0.5, 1.0],
        [0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
    ):
        pred = set(
            rank_rows(
                rows,
                score_mode=score_mode,
                sign_mode=sign_mode,
                score_scale=score_scale,
                order_bonus=order_bonus,
                sup_pen=sup_pen,
                fixed_target=int(args.fixed_target),
                k=len(gt),
            )
        )
        results.append(
            (
                len(gt & pred),
                score_mode,
                sign_mode,
                order_bonus,
                sup_pen,
                sorted(pred),
            )
        )

    results.sort(reverse=True)
    seen = set()
    for item in results[:80]:
        key = (item[0], tuple(item[-1]))
        if key in seen:
            continue
        seen.add(key)
        print(item)


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()

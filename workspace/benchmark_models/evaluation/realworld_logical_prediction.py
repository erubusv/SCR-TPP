from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ..baselines.teller_official import OFFICIAL_ROOT, _as_teller_dataset
from ..core.schema import NormalizedRule, normalize_rules
from ..baselines.logical_rule_baselines import CLNNModel, CLUSTERModel, NSTPPModel


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_split(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _seq_horizon(seq: dict[str, Any], default_horizon: float) -> float:
    times = [float(t) for t in seq.get("time", [])]
    if not times:
        return float(default_horizon)
    return max(float(default_horizon), float(max(times)))


def _events_by_type(seq: dict[str, Any]) -> dict[int, np.ndarray]:
    by_type: dict[int, list[float]] = {}
    for t, event in zip(seq.get("time", []), seq.get("event", [])):
        by_type.setdefault(int(event), []).append(float(t))
    return {event: np.asarray(times, dtype=np.float64) for event, times in by_type.items()}


def _last_lag(times: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.searchsorted(times, query, side="left") - 1
    valid = pos >= 0
    last = np.zeros_like(query, dtype=np.float64)
    if np.any(valid):
        last[valid] = times[pos[valid]]
    lag = query - last
    return valid, last, lag


def _paper_design_for_queries(
    *,
    sequences: list[dict[str, Any]],
    query_by_sequence: list[np.ndarray],
    source_ids: tuple[int, ...],
) -> dict[str, np.ndarray]:
    total_q = int(sum(len(q) for q in query_by_sequence))
    p = len(source_ids)
    facts = np.zeros((total_q, p), dtype=np.float64)
    clocks = np.zeros((total_q, p), dtype=np.float64)
    last_times = np.zeros((total_q, p), dtype=np.float64)
    valid = np.zeros((total_q, p), dtype=np.float64)
    offset = 0
    for seq, query in zip(sequences, query_by_sequence):
        q = np.asarray(query, dtype=np.float64)
        n = len(q)
        if n == 0:
            continue
        by_type = _events_by_type(seq)
        for src_pos, src in enumerate(source_ids):
            times = by_type.get(int(src), np.asarray([], dtype=np.float64))
            if len(times) == 0:
                clocks[offset:offset + n, src_pos] = 1e6
                continue
            src_valid, src_last, lag = _last_lag(times, q)
            facts[offset:offset + n, src_pos] = src_valid.astype(np.float64)
            clocks[offset:offset + n, src_pos] = np.where(src_valid, lag, 1e6)
            last_times[offset:offset + n, src_pos] = src_last
            valid[offset:offset + n, src_pos] = src_valid.astype(np.float64)
        offset += n
    return {"facts": facts, "clocks": clocks, "last_times": last_times, "valid": valid}


def _scr_rule_features(
    *,
    sequences: list[dict[str, Any]],
    query_by_sequence: list[np.ndarray],
    details: list[dict[str, Any]],
) -> np.ndarray:
    total_q = int(sum(len(q) for q in query_by_sequence))
    if not details:
        return np.zeros((total_q, 0), dtype=np.float64)
    out = np.zeros((total_q, len(details)), dtype=np.float64)
    kernel_by_rule: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    for detail in details:
        by_source: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for kernel in detail.get("kernel_distribution_by_source", []) or []:
            source = int(kernel["source"])
            knots = np.asarray(kernel.get("knots", []), dtype=np.float64)
            heights = np.asarray(kernel.get("shape_heights", []), dtype=np.float64)
            if len(knots) and len(knots) == len(heights):
                by_source[source] = (knots, np.maximum(heights, 0.0))
        kernel_by_rule.append(by_source)
    offset = 0
    for seq, query in zip(sequences, query_by_sequence):
        q = np.asarray(query, dtype=np.float64)
        n = len(q)
        if n == 0:
            continue
        by_type = _events_by_type(seq)
        for ridx, detail in enumerate(details):
            activation = np.ones(n, dtype=np.float64)
            for src in detail.get("sources", []) or []:
                src = int(src)
                times = by_type.get(src, np.asarray([], dtype=np.float64))
                if len(times) == 0 or src not in kernel_by_rule[ridx]:
                    activation *= 0.0
                    continue
                valid, _last, lag = _last_lag(times, q)
                knots, heights = kernel_by_rule[ridx][src]
                values = np.zeros(n, dtype=np.float64)
                inside = valid & (lag >= knots[0]) & (lag <= knots[-1])
                if np.any(inside):
                    values[inside] = np.interp(lag[inside], knots, heights, left=0.0, right=0.0)
                activation *= values
            out[offset:offset + n, ridx] = activation
        offset += n
    return out


def _target_event_queries(sequences: list[dict[str, Any]], target: int) -> list[np.ndarray]:
    return [
        np.asarray(
            [float(t) for t, event in zip(seq.get("time", []), seq.get("event", [])) if int(event) == int(target)],
            dtype=np.float64,
        )
        for seq in sequences
    ]


def _grid_queries(sequences: list[dict[str, Any]], *, grid_size: int, time_horizon: float) -> tuple[list[np.ndarray], np.ndarray]:
    queries: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for seq in sequences:
        horizon = _seq_horizon(seq, float(time_horizon))
        grid = (np.arange(int(grid_size), dtype=np.float64) + 0.5) * horizon / max(int(grid_size), 1)
        queries.append(grid)
        weights.append(np.full(int(grid_size), horizon / max(int(grid_size), 1), dtype=np.float64))
    return queries, np.concatenate(weights) if weights else np.zeros(0, dtype=np.float64)


def _raw_output_path(row: dict[str, Any]) -> Path | None:
    metadata = dict(row.get("model_metadata", {}) or {})
    raw_path = metadata.get("raw_output_path")
    if raw_path:
        return Path(str(raw_path))
    command = list(metadata.get("command", []) or [])
    for idx, part in enumerate(command[:-1]):
        if str(part) == "--output":
            return Path(str(command[idx + 1]))
    return None


def _payload_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("model_metadata", {}) or {})
    payload_meta = metadata.get("payload_metadata")
    if isinstance(payload_meta, dict):
        return payload_meta
    raw_path = _raw_output_path(row)
    if raw_path is not None and raw_path.exists():
        payload = json.loads(raw_path.read_text())
        if isinstance(payload.get("metadata"), dict):
            return dict(payload["metadata"])
    return {}


def _teller_master_is_finite(row: dict[str, Any]) -> bool:
    meta = _payload_metadata(row)
    master = meta.get("teller_master_params")
    if not isinstance(master, dict):
        return False
    base = master.get("base")
    weights = master.get("weights", [])
    return bool(base == base and all(float(w) == float(w) for w in weights))


def _teller_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    raw_path = _raw_output_path(row)
    if raw_path is None or not raw_path.exists():
        return None
    payload = json.loads(raw_path.read_text())
    if not isinstance(payload, dict):
        return None
    return payload


def _load_official_teller_model(path: str | Path) -> Any:
    official_root = OFFICIAL_ROOT.resolve()
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    import logic_learning  # noqa: F401  # type: ignore

    with Path(path).open("rb") as f:
        return pickle.load(f)


def _teller_params(row: dict[str, Any], *, target: int) -> tuple[Any, list[dict[str, Any]], np.ndarray, float, dict[str, Any]] | None:
    payload = _teller_payload(row)
    if payload is None:
        return None
    metadata = dict(payload.get("metadata", {}) or {})
    master = metadata.get("teller_master_params")
    if not isinstance(master, dict):
        return None
    model_path = metadata.get("official_model_pickle_path")
    if not model_path or not Path(str(model_path)).exists():
        return None
    model = _load_official_teller_model(str(model_path))
    base = float(master.get("base", float("nan")))
    weights = np.asarray(master.get("weights", []), dtype=np.float64)
    templates = [model.logic_template[int(target)][rule_id] for rule_id in sorted(model.logic_template.get(int(target), {}).keys())]
    if not np.isfinite(base) or len(weights) != len(templates) or not np.all(np.isfinite(weights)):
        return None
    return model, templates, weights, base, metadata


def _teller_feature_matrices(
    *,
    sequences: list[dict[str, Any]],
    target: int,
    event_by_sequence: list[np.ndarray],
    grid_by_sequence: list[np.ndarray],
    grid_weights_by_sequence: list[np.ndarray],
    model: Any,
    templates: list[dict[str, Any]],
    time_window: float,
    tolerance: float,
    decay_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_events = int(sum(len(q) for q in event_by_sequence))
    total_grid = int(sum(len(q) for q in grid_by_sequence))
    flat_grid_weights = (
        np.concatenate([np.asarray(w, dtype=np.float64) for w in grid_weights_by_sequence if len(w)])
        if any(len(w) for w in grid_weights_by_sequence)
        else np.zeros(0, dtype=np.float64)
    )
    if not templates:
        weights = (
            np.concatenate([np.asarray(w, dtype=np.float64) for w in grid_weights_by_sequence if len(w)])
            if any(len(w) for w in grid_weights_by_sequence)
            else np.zeros(0, dtype=np.float64)
        )
        return np.zeros((total_events, 0), dtype=np.float64), np.zeros((total_grid, 0), dtype=np.float64), weights
    old_window = float(model.time_window)
    old_tol = float(model.Time_tolerance)
    old_decay = float(model.decay_rate)
    model.time_window = float(time_window)
    model.Time_tolerance = float(tolerance)
    model.decay_rate = float(decay_rate)
    teller_dataset = _as_teller_dataset(sequences, num_event_types=int(model.num_predicate))
    event_cols = [np.zeros(total_events, dtype=np.float64) for _ in templates]
    grid_cols = [np.zeros(total_grid, dtype=np.float64) for _ in templates]
    event_offset = 0
    grid_offset = 0
    try:
        for sample_id, (_seq, event_query, grid_query) in enumerate(zip(sequences, event_by_sequence, grid_by_sequence)):
            history = teller_dataset[int(sample_id)]
            for local_idx, t in enumerate(event_query):
                for ridx, template in enumerate(templates):
                    feat = model.get_feature(float(t), int(target), history, template)
                    effect = model.get_formula_effect(float(t), int(target), history, template)
                    event_cols[ridx][event_offset + local_idx] = float((feat * effect).detach().cpu().numpy()[0])
            event_offset += len(event_query)
            for local_idx, t in enumerate(grid_query):
                for ridx, template in enumerate(templates):
                    feat = model.get_feature(float(t), int(target), history, template)
                    effect = model.get_formula_effect(float(t), int(target), history, template)
                    grid_cols[ridx][grid_offset + local_idx] = float((feat * effect).detach().cpu().numpy()[0])
            grid_offset += len(grid_query)
    finally:
        model.time_window = old_window
        model.Time_tolerance = old_tol
        model.decay_rate = old_decay
    event_x = np.column_stack(event_cols) if event_cols else np.zeros((total_events, 0), dtype=np.float64)
    grid_x = np.column_stack(grid_cols) if grid_cols else np.zeros((total_grid, 0), dtype=np.float64)
    event_x[~np.isfinite(event_x)] = 0.0
    grid_x[~np.isfinite(grid_x)] = 0.0
    return event_x, grid_x, flat_grid_weights


def _evaluate_teller_loglink(
    *,
    sequences: list[dict[str, Any]],
    target: int,
    time_horizon: float,
    grid_size: int,
    teller: tuple[Any, list[dict[str, Any]], np.ndarray, float, dict[str, Any]],
) -> dict[str, float | int | None]:
    model, templates, weights, base, metadata = teller
    event_queries = _target_event_queries(sequences, target)
    grid_queries, flat_weights = _grid_queries(sequences, grid_size=grid_size, time_horizon=time_horizon)
    per_seq_weights: list[np.ndarray] = []
    offset = 0
    for grid in grid_queries:
        per_seq_weights.append(flat_weights[offset:offset + len(grid)])
        offset += len(grid)
    event_x, grid_x, grid_weights = _teller_feature_matrices(
        sequences=sequences,
        target=int(target),
        event_by_sequence=event_queries,
        grid_by_sequence=grid_queries,
        grid_weights_by_sequence=per_seq_weights,
        model=model,
        templates=templates,
        time_window=float(metadata.get("time_window", 30.0)),
        tolerance=float(metadata.get("time_tolerance", metadata.get("tolerance", 0.1))),
        decay_rate=float(metadata.get("decay_rate", 1.0)),
    )
    event_eta = float(base) + event_x @ weights if len(weights) else np.full(event_x.shape[0], float(base))
    grid_eta = float(base) + grid_x @ weights if len(weights) else np.full(grid_x.shape[0], float(base))
    event_count = int(event_eta.shape[0])
    integral = float(np.sum(grid_weights * np.exp(np.clip(grid_eta, -30.0, 30.0))))
    if event_count == 0:
        return {"nll": None, "target_event_count": 0, "integral": integral}
    nll = float((-float(np.sum(event_eta)) + integral) / max(event_count, 1))
    return {"nll": nll, "target_event_count": event_count, "integral": integral}


def _teller_time_prediction_metrics(
    *,
    sequences: list[dict[str, Any]],
    target: int,
    time_horizon: float,
    teller: tuple[Any, list[dict[str, Any]], np.ndarray, float, dict[str, Any]],
    prediction_grid_size: int,
) -> dict[str, float | int | None]:
    model, templates, weights, base, metadata = teller
    sq_errors: list[float] = []
    abs_errors: list[float] = []
    hist_sequences: list[dict[str, Any]] = []
    grids: list[np.ndarray] = []
    prev_times: list[float] = []
    true_dts: list[float] = []
    per_seq_weights: list[np.ndarray] = []
    dts: list[float] = []
    for seq in sequences:
        times = [float(t) for t in seq.get("time", [])]
        events = [int(e) for e in seq.get("event", [])]
        if len(times) < 2:
            continue
        horizon = _seq_horizon(seq, float(time_horizon))
        for idx in range(1, len(times)):
            if int(events[idx]) != int(target):
                continue
            prev_t = float(times[idx - 1])
            true_dt = max(float(times[idx]) - prev_t, 0.0)
            if true_dt <= 0.0:
                continue
            upper = max(horizon, float(times[idx]))
            dt = (upper - prev_t) / max(int(prediction_grid_size), 1)
            grid = prev_t + (np.arange(int(prediction_grid_size), dtype=np.float64) + 0.5) * dt
            hist_sequences.append(_history_until(seq, prev_t))
            grids.append(grid)
            prev_times.append(prev_t)
            true_dts.append(true_dt)
            per_seq_weights.append(np.full(len(grid), dt, dtype=np.float64))
            dts.append(dt)
    if not grids:
        return {"time_rmse": None, "time_mae": None, "time_count": 0}
    _event_x, grid_x, _grid_weights = _teller_feature_matrices(
        sequences=hist_sequences,
        target=int(target),
        event_by_sequence=[np.zeros(0, dtype=np.float64) for _ in hist_sequences],
        grid_by_sequence=grids,
        grid_weights_by_sequence=per_seq_weights,
        model=model,
        templates=templates,
        time_window=float(metadata.get("time_window", 30.0)),
        tolerance=float(metadata.get("time_tolerance", metadata.get("tolerance", 0.1))),
        decay_rate=float(metadata.get("decay_rate", 1.0)),
    )
    eta_all = float(base) + grid_x @ weights if len(weights) else np.full(grid_x.shape[0], float(base))
    lam_all = np.exp(np.clip(eta_all, -30.0, 30.0))
    offset = 0
    for grid, prev_t, true_dt, dt in zip(grids, prev_times, true_dts, dts):
        lam = lam_all[offset:offset + len(grid)]
        offset += len(grid)
        pred_dt = _expected_wait_from_grid(grid=grid, lam=lam, prev_t=prev_t, dt=float(dt))
        err = float(pred_dt - true_dt)
        sq_errors.append(err * err)
        abs_errors.append(abs(err))
    return {
        "time_rmse": float(math.sqrt(sum(sq_errors) / len(sq_errors))),
        "time_mae": float(sum(abs_errors) / len(abs_errors)),
        "time_count": int(len(sq_errors)),
    }


def _load_paper_checkpoint(row: dict[str, Any], device: str) -> tuple[Any, dict[str, Any]] | None:
    payload_meta = _payload_metadata(row)
    ckpt_path = payload_meta.get("paper_intensity_checkpoint_path")
    if not ckpt_path:
        return None
    ckpt = torch.load(str(ckpt_path), map_location=device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model_name = str(ckpt["model"])
    init_args = dict(ckpt["init_args"])
    if model_name == "CLNN":
        model = CLNNModel(**init_args)
    elif model_name == "NSTPP":
        model = NSTPPModel(**init_args)
    elif model_name == "CLUSTER":
        model = CLUSTERModel(**init_args)
    else:
        raise ValueError(f"unsupported paper checkpoint model: {model_name}")
    torch_device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch_device)
    model.eval()
    return model, ckpt


def _paper_model_intensity(
    *,
    model: Any,
    ckpt: dict[str, Any],
    sequences: list[dict[str, Any]],
    query_by_sequence: list[np.ndarray],
    device: str,
) -> np.ndarray:
    source_ids = tuple(int(src) for src in ckpt["source_ids"])
    design = _paper_design_for_queries(
        sequences=sequences,
        query_by_sequence=query_by_sequence,
        source_ids=source_ids,
    )
    torch_device = next(model.parameters()).device
    total = int(design["facts"].shape[0])
    if total == 0:
        return np.zeros(0, dtype=np.float64)
    chunk_size = 65536 if torch_device.type == "cuda" else 262144
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, total, int(chunk_size)):
            end = min(start + int(chunk_size), total)
            facts = torch.tensor(design["facts"][start:end], dtype=torch.float32, device=torch_device)
            clocks = torch.tensor(design["clocks"][start:end], dtype=torch.float32, device=torch_device)
            last_times = torch.tensor(design["last_times"][start:end], dtype=torch.float32, device=torch_device)
            valid = torch.tensor(design["valid"][start:end], dtype=torch.float32, device=torch_device)
            if str(ckpt["model"]) == "CLNN":
                eta = model.rho + model.formula_truth(clocks) @ model.w_phi
                lam = torch.exp(torch.clamp(eta, -30.0, 30.0))
            elif str(ckpt["model"]) == "NSTPP":
                phi = model.features(
                    facts,
                    last_times,
                    valid,
                    delta=float(ckpt.get("delta", 0.0)),
                    sample_gumbel=False,
                )
                lam = torch.nn.functional.softplus(model.base_raw) + phi @ torch.nn.functional.softplus(model.gamma_raw)
            elif str(ckpt["model"]) == "CLUSTER":
                dummy_grid = facts[:0]
                dummy_last = last_times[:0]
                dummy_valid = valid[:0]
                lam, _grid_lam, _event_phi, _grid_phi = model.marginal_lambdas(
                    facts,
                    dummy_grid,
                    last_times,
                    dummy_last,
                    valid,
                    dummy_valid,
                    delta=float(ckpt.get("delta", 0.0)),
                    use_temporal=bool(ckpt.get("use_temporal", True)),
                )
            else:
                raise ValueError(f"unsupported paper model: {ckpt['model']}")
            out.append(lam.detach().cpu().numpy().astype(np.float64))
            del facts, clocks, last_times, valid, lam
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def _base_rate_params(train_sequences: list[dict[str, Any]], *, target: int, time_horizon: float) -> dict[str, float]:
    event_count = sum(1 for seq in train_sequences for event in seq.get("event", []) if int(event) == int(target))
    exposure = max(float(len(train_sequences)) * float(time_horizon), 1e-8)
    return {"mu": max(float(event_count) / exposure, 1e-8)}


def _evaluate_scr_loglink(
    *,
    sequences: list[dict[str, Any]],
    target: int,
    time_horizon: float,
    grid_size: int,
    params: dict[str, Any],
    scr_details: list[dict[str, Any]] | None = None,
) -> dict[str, float | int | None]:
    event_queries = _target_event_queries(sequences, target)
    grid_queries, grid_weights = _grid_queries(sequences, grid_size=grid_size, time_horizon=time_horizon)
    event_x = _scr_rule_features(sequences=sequences, query_by_sequence=event_queries, details=scr_details or [])
    grid_x = _scr_rule_features(sequences=sequences, query_by_sequence=grid_queries, details=scr_details or [])
    signs = np.asarray(
        [1.0 if str(detail.get("sign")) == "excitation" else -1.0 for detail in (scr_details or [])],
        dtype=np.float64,
    )
    beta = np.asarray(params.get("beta", []), dtype=np.float64)
    mu = max(float(params.get("mu", 1e-8)), 1e-8)
    if len(beta):
        event_eta = math.log(mu) + event_x @ (beta * signs)
        grid_eta = math.log(mu) + grid_x @ (beta * signs)
    else:
        event_eta = np.full(event_x.shape[0], math.log(mu), dtype=np.float64)
        grid_eta = np.full(grid_x.shape[0], math.log(mu), dtype=np.float64)
    event_count = int(event_eta.shape[0])
    if event_count == 0:
        return {"nll": None, "target_event_count": 0, "integral": float(np.sum(grid_weights * np.exp(np.clip(grid_eta, -30, 30))))}
    integral = float(np.sum(grid_weights * np.exp(np.clip(grid_eta, -30, 30))))
    nll = float((-float(np.sum(event_eta)) + integral) / max(event_count, 1))
    return {"nll": nll, "target_event_count": event_count, "integral": integral}


def _evaluate_paper_intensity(
    *,
    sequences: list[dict[str, Any]],
    target: int,
    time_horizon: float,
    grid_size: int,
    model: Any | None,
    ckpt: dict[str, Any] | None,
    base_params: dict[str, float] | None,
    device: str,
) -> dict[str, float | int | None]:
    event_queries = _target_event_queries(sequences, target)
    grid_queries, grid_weights = _grid_queries(sequences, grid_size=grid_size, time_horizon=time_horizon)
    if model is None or ckpt is None:
        mu = max(float((base_params or {}).get("mu", 1e-8)), 1e-8)
        event_lambda = np.full(sum(len(q) for q in event_queries), mu, dtype=np.float64)
        grid_lambda = np.full(len(grid_weights), mu, dtype=np.float64)
    else:
        event_lambda = _paper_model_intensity(
            model=model,
            ckpt=ckpt,
            sequences=sequences,
            query_by_sequence=event_queries,
            device=device,
        )
        grid_lambda = _paper_model_intensity(
            model=model,
            ckpt=ckpt,
            sequences=sequences,
            query_by_sequence=grid_queries,
            device=device,
        )
    event_count = int(len(event_lambda))
    if event_count == 0:
        return {"nll": None, "target_event_count": 0, "integral": float(np.sum(grid_weights * grid_lambda))}
    integral = float(np.sum(grid_weights * grid_lambda))
    nll = float((-np.sum(np.log(np.clip(event_lambda, 1e-12, None))) + integral) / max(event_count, 1))
    return {"nll": nll, "target_event_count": event_count, "integral": integral}


def _history_until(seq: dict[str, Any], cutoff: float) -> dict[str, Any]:
    times: list[float] = []
    events: list[int] = []
    for t, event in zip(seq.get("time", []), seq.get("event", [])):
        if float(t) > float(cutoff):
            break
        times.append(float(t))
        events.append(int(event))
    return {"time": times, "event": events, "sequence_id": seq.get("sequence_id")}


def _expected_wait_from_grid(*, grid: np.ndarray, lam: np.ndarray, prev_t: float, dt: float) -> float:
    cumhaz_left = np.maximum(np.cumsum(lam * float(dt)) - lam * float(dt), 0.0)
    density = lam * np.exp(-cumhaz_left)
    denom = float(np.sum(density * float(dt)))
    if denom <= 1e-12:
        return 1.0 / max(float(lam[0]) if len(lam) else 0.0, 1e-8)
    pred_t = float(np.sum(grid * density * float(dt)) / denom)
    return max(pred_t - float(prev_t), 0.0)


def _scr_time_prediction_metrics(
    *,
    sequences: list[dict[str, Any]],
    target: int,
    time_horizon: float,
    params: dict[str, Any],
    scr_details: list[dict[str, Any]] | None = None,
    prediction_grid_size: int = 64,
) -> dict[str, float | int | None]:
    sq_errors: list[float] = []
    abs_errors: list[float] = []
    beta = np.asarray(params.get("beta", []), dtype=np.float64)
    mu = max(float(params.get("mu", 1e-8)), 1e-8)
    signs = np.asarray(
        [1.0 if str(detail.get("sign")) == "excitation" else -1.0 for detail in (scr_details or [])],
        dtype=np.float64,
    )
    hist_sequences: list[dict[str, Any]] = []
    grids: list[np.ndarray] = []
    prev_times: list[float] = []
    true_dts: list[float] = []
    dts: list[float] = []
    for seq in sequences:
        times = [float(t) for t in seq.get("time", [])]
        events = [int(e) for e in seq.get("event", [])]
        if len(times) < 2:
            continue
        horizon = _seq_horizon(seq, float(time_horizon))
        for idx in range(1, len(times)):
            if int(events[idx]) != int(target):
                continue
            prev_t = float(times[idx - 1])
            true_dt = max(float(times[idx]) - prev_t, 0.0)
            if true_dt <= 0.0:
                continue
            upper = max(horizon, float(times[idx]))
            grid = prev_t + (np.arange(int(prediction_grid_size), dtype=np.float64) + 0.5) * (upper - prev_t) / max(int(prediction_grid_size), 1)
            hist_sequences.append(_history_until(seq, prev_t))
            grids.append(grid)
            prev_times.append(prev_t)
            true_dts.append(true_dt)
            dts.append((upper - prev_t) / max(int(prediction_grid_size), 1))
    if not grids:
        return {"time_rmse": None, "time_mae": None, "time_count": 0}
    x_all = _scr_rule_features(sequences=hist_sequences, query_by_sequence=grids, details=scr_details or [])
    if len(beta):
        eta_all = math.log(mu) + x_all @ (beta * signs)
    else:
        eta_all = np.full(x_all.shape[0], math.log(mu), dtype=np.float64)
    lam_all = np.exp(np.clip(eta_all, -30.0, 30.0))
    offset = 0
    for grid, prev_t, true_dt, dt in zip(grids, prev_times, true_dts, dts):
        lam = lam_all[offset:offset + len(grid)]
        offset += len(grid)
        pred_dt = _expected_wait_from_grid(grid=grid, lam=lam, prev_t=prev_t, dt=float(dt))
        err = float(pred_dt - true_dt)
        sq_errors.append(err * err)
        abs_errors.append(abs(err))
    return {
        "time_rmse": float(math.sqrt(sum(sq_errors) / len(sq_errors))),
        "time_mae": float(sum(abs_errors) / len(abs_errors)),
        "time_count": int(len(sq_errors)),
    }


def _paper_time_prediction_metrics(
    *,
    sequences: list[dict[str, Any]],
    target: int,
    time_horizon: float,
    model: Any | None,
    ckpt: dict[str, Any] | None,
    base_params: dict[str, float] | None,
    device: str,
    prediction_grid_size: int,
) -> dict[str, float | int | None]:
    sq_errors: list[float] = []
    abs_errors: list[float] = []
    hist_sequences: list[dict[str, Any]] = []
    grids: list[np.ndarray] = []
    prev_times: list[float] = []
    true_dts: list[float] = []
    dts: list[float] = []
    for seq in sequences:
        times = [float(t) for t in seq.get("time", [])]
        events = [int(e) for e in seq.get("event", [])]
        if len(times) < 2:
            continue
        horizon = _seq_horizon(seq, float(time_horizon))
        for idx in range(1, len(times)):
            if int(events[idx]) != int(target):
                continue
            prev_t = float(times[idx - 1])
            true_dt = max(float(times[idx]) - prev_t, 0.0)
            if true_dt <= 0.0:
                continue
            upper = max(horizon, float(times[idx]))
            grid = prev_t + (np.arange(int(prediction_grid_size), dtype=np.float64) + 0.5) * (upper - prev_t) / max(int(prediction_grid_size), 1)
            hist_sequences.append(_history_until(seq, prev_t))
            grids.append(grid)
            prev_times.append(prev_t)
            true_dts.append(true_dt)
            dts.append((upper - prev_t) / max(int(prediction_grid_size), 1))
    if not grids:
        return {"time_rmse": None, "time_mae": None, "time_count": 0}
    if model is None or ckpt is None:
        mu = max(float((base_params or {}).get("mu", 1e-8)), 1e-8)
        lam_all = np.full(sum(len(grid) for grid in grids), mu, dtype=np.float64)
    else:
        lam_all = _paper_model_intensity(
            model=model,
            ckpt=ckpt,
            sequences=hist_sequences,
            query_by_sequence=grids,
            device=device,
        )
    offset = 0
    for grid, prev_t, true_dt, dt in zip(grids, prev_times, true_dts, dts):
        lam = lam_all[offset:offset + len(grid)]
        offset += len(grid)
        pred_dt = _expected_wait_from_grid(grid=grid, lam=lam, prev_t=prev_t, dt=float(dt))
        err = float(pred_dt - true_dt)
        sq_errors.append(err * err)
        abs_errors.append(abs(err))
    return {
        "time_rmse": float(math.sqrt(sum(sq_errors) / len(sq_errors))),
        "time_mae": float(sum(abs_errors) / len(abs_errors)),
        "time_count": int(len(sq_errors)),
    }


def _scr_params(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details = list(row.get("learned_rule_parameter_details", []) or [])
    if not details:
        details = list((row.get("model_metadata", {}) or {}).get("learned_rule_parameter_details", []) or [])
    beta = np.asarray([float(detail.get("beta", 0.0)) for detail in details], dtype=np.float64)
    mu = float(row.get("mu") or (row.get("model_metadata", {}) or {}).get("mu") or 1e-8)
    return details, {"mu": mu, "beta": beta}


def _row_rules(row: dict[str, Any], target: int) -> list[NormalizedRule]:
    return normalize_rules(row.get("learned_rules", []) or row.get("predicted_rules", []), default_target=int(target))


def evaluate_row(
    *,
    row: dict[str, Any],
    split: dict[str, Any],
    target: int,
    time_horizon: float,
    grid_size: int,
    device: str,
    prediction_grid_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    train = list(split.get("train", []))
    test = list(split.get("test", []))
    metadata = dict(row.get("model_metadata", {}) or {})
    config = dict((split.get("metadata", {}) or {}).get("config", {}) or {})
    time_window = float(metadata.get("time_window", config.get("time_window", time_horizon)))
    rules = _row_rules(row, target)
    if str(row.get("model")) == "SCR-TPP":
        details, params = _scr_params(row)
        feature_builder = "scr"
        paper_model = None
        paper_ckpt = None
        base_params = None
    elif str(row.get("model")) in {"CLNN", "NSTPP", "CLUSTER"}:
        loaded = _load_paper_checkpoint(row, str(device))
        if loaded is None:
            raise FileNotFoundError(
                f"{row.get('model')} row has no paper_intensity_checkpoint_path; rerun the baseline after the checkpoint patch"
            )
        paper_model, paper_ckpt = loaded
        details = []
        params = {}
        feature_builder = f"{row.get('model')}_paper_intensity"
        base_params = None
    else:
        teller = _teller_params(row, target=int(target))
        details = []
        params = {}
        feature_builder = "TELLER_paper_loglink" if teller is not None else "TELLER_invalid_or_missing_master"
        paper_model = None
        paper_ckpt = None
        base_params = _base_rate_params(train, target=int(target), time_horizon=float(time_horizon))
    infer_start = time.perf_counter()
    if str(row.get("model")) == "SCR-TPP":
        likelihood = _evaluate_scr_loglink(
            sequences=test,
            target=int(target),
            time_horizon=float(time_horizon),
            grid_size=int(grid_size),
            params=params,
            scr_details=details,
        )
        timing = _scr_time_prediction_metrics(
            sequences=test,
            target=int(target),
            time_horizon=float(time_horizon),
            params=params,
            scr_details=details,
            prediction_grid_size=int(prediction_grid_size),
        )
    else:
        if str(row.get("model")).startswith("TELLER"):
            teller = _teller_params(row, target=int(target))
            if teller is None:
                likelihood = {"nll": None, "target_event_count": None}
                timing = {"time_rmse": None, "time_mae": None, "time_count": None}
            else:
                likelihood = _evaluate_teller_loglink(
                    sequences=test,
                    target=int(target),
                    time_horizon=float(time_horizon),
                    grid_size=int(grid_size),
                    teller=teller,
                )
                timing = _teller_time_prediction_metrics(
                    sequences=test,
                    target=int(target),
                    time_horizon=float(time_horizon),
                    teller=teller,
                    prediction_grid_size=int(prediction_grid_size),
                )
        else:
            likelihood = _evaluate_paper_intensity(
                sequences=test,
                target=int(target),
                time_horizon=float(time_horizon),
                grid_size=int(grid_size),
                model=paper_model,
                ckpt=paper_ckpt,
                base_params=base_params,
                device=str(device),
            )
            timing = _paper_time_prediction_metrics(
                sequences=test,
                target=int(target),
                time_horizon=float(time_horizon),
                model=paper_model,
                ckpt=paper_ckpt,
                base_params=base_params,
                device=str(device),
                prediction_grid_size=int(prediction_grid_size),
            )
    inference_runtime = float(time.perf_counter() - infer_start)
    base_training_time = row.get("training_time_sec", row.get("runtime_sec"))
    training_time = None if base_training_time is None else float(base_training_time)
    return {
        "model": row.get("model"),
        "dataset": row.get("dataset"),
        "target": row.get("target"),
        "target_event_id": int(target),
        "seed": row.get("seed"),
        "nll": likelihood.get("nll"),
        "time_mae": timing.get("time_mae"),
        "time_rmse": timing.get("time_rmse"),
        "type_acc": None,
        "topk": {},
        "runtime_sec": float(time.perf_counter() - started) + float(row.get("runtime_sec", 0.0) or 0.0),
        "training_time_sec": training_time,
        "inference_time_sec": inference_runtime,
        "learned_rules": [rule.to_dict() for rule in rules],
        "learned_rule_count": len(rules),
        "model_metadata": {
            "implementation_type": metadata.get("implementation_type"),
            "evaluation_scope": "train_fit_test_target_mark",
            "prediction_evaluator": feature_builder,
            "source_rule_runtime_sec": row.get("runtime_sec"),
            "prediction_extra_fit_runtime_sec": 0.0,
            "target_event_count": likelihood.get("target_event_count"),
            "time_prediction_count": timing.get("time_count"),
            "time_window": time_window,
            "grid_size": int(grid_size),
            "note": (
                "Rules are learned on the train split only. Prediction metrics are computed "
                "on the held-out test split. SCR-TPP uses its learned beta/kernel parameters. "
                "CLNN/NSTPP/CLUSTER use the trained restricted paper-based model checkpoint and their "
                "paper intensity family directly. TELLER uses its exported branch-and-price "
                "master intercept and signed rule weights directly."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate logical rule models on held-out target-event prediction.")
    ap.add_argument("--rule_input_manifest", required=True)
    ap.add_argument("--rule_results_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid_size", type=int, default=96)
    ap.add_argument("--prediction_grid_size", type=int, default=64)
    args = ap.parse_args()

    manifest = json.loads(Path(args.rule_input_manifest).read_text())
    target = int(manifest["target_event_id"])
    rows = _read_jsonl(args.rule_results_jsonl)
    out: list[dict[str, Any]] = []
    for row in rows:
        seed = str(row.get("seed"))
        if seed not in manifest["split_paths"]:
            continue
        split = _load_split(manifest["split_paths"][seed])
        config = (split.get("metadata", {}) or {}).get("config", {}) or {}
        time_horizon = float(config.get("time_horizon", config.get("time_window", 1.0)))
        out.append(
            evaluate_row(
                row=row,
                split=split,
                target=target,
                time_horizon=time_horizon,
                grid_size=int(args.grid_size),
                device=str(args.device),
                prediction_grid_size=int(args.prediction_grid_size),
            )
        )
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"output": str(output), "rows": len(out)}, indent=2))


if __name__ == "__main__":
    main()

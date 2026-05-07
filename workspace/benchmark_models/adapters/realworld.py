from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def _rank_event_labels(values: pd.Series) -> list[str]:
    counts = values.astype(str).value_counts(dropna=False)
    rows = [(str(label), int(count)) for label, count in counts.items()]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [label for label, _ in rows]


def _select_event_labels(
    values: pd.Series,
    *,
    top_k: int | None,
    always_keep: set[str] | None = None,
) -> list[str]:
    ranked = _rank_event_labels(values)
    keep = set(always_keep or set())
    if top_k is None or top_k <= 0 or top_k >= len(ranked):
        selected = set(ranked)
    else:
        selected = set(ranked[:top_k])
        missing_required = [label for label in sorted(keep) if label not in selected and label in set(ranked)]
        for label in missing_required:
            if len(selected) >= top_k:
                removable = [candidate for candidate in ranked if candidate in selected and candidate not in keep]
                if removable:
                    selected.remove(removable[-1])
            selected.add(label)
    return [label for label in ranked if label in selected]


def events_csv_to_sequences(
    *,
    csv_path: str | Path,
    sequence_col: str,
    time_col: str,
    event_col: str,
    top_k_event_types: int | None = None,
    target_event_label: str | None = None,
    min_events_per_sequence: int = 2,
    start_at_zero: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert a generic event-log CSV into the sequence dict format.

    Required columns are sequence id, event time, and event label. Event labels
    are mapped deterministically by frequency, with lexical tie-breaking. When a
    target label is supplied it is kept even under top-k filtering.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    missing = [col for col in (sequence_col, time_col, event_col) if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df[[sequence_col, time_col, event_col]].dropna()
    df[event_col] = df[event_col].astype(str)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    keep_required = {str(target_event_label)} if target_event_label is not None else set()
    selected_labels = _select_event_labels(
        df[event_col],
        top_k=top_k_event_types,
        always_keep=keep_required,
    )
    df = df[df[event_col].isin(set(selected_labels))].copy()
    event_to_id = {label: idx for idx, label in enumerate(selected_labels)}
    sequences: list[dict[str, Any]] = []
    for sequence_id, group in df.groupby(sequence_col, sort=True):
        group = group.sort_values(time_col, kind="mergesort")
        if len(group) < int(min_events_per_sequence):
            continue
        times = [float(value) for value in group[time_col].tolist()]
        if start_at_zero and times:
            offset = times[0]
            times = [float(value - offset) for value in times]
        events = [int(event_to_id[str(label)]) for label in group[event_col].tolist()]
        sequences.append(
            {
                "time": times,
                "event": events,
                "sequence_id": str(sequence_id),
            }
        )
    target_event_id = None
    if target_event_label is not None and str(target_event_label) in event_to_id:
        target_event_id = int(event_to_id[str(target_event_label)])
    metadata = {
        "source_csv": str(csv_path),
        "sequence_col": sequence_col,
        "time_col": time_col,
        "event_col": event_col,
        "num_sequences": len(sequences),
        "dim_process": len(event_to_id),
        "event_to_id": event_to_id,
        "id_to_event": {str(idx): label for label, idx in event_to_id.items()},
        "top_k_event_types": top_k_event_types,
        "target_event_label": target_event_label,
        "target_event_id": target_event_id,
        "min_events_per_sequence": int(min_events_per_sequence),
        "start_at_zero": bool(start_at_zero),
    }
    return sequences, metadata


def write_sequence_pickle(
    *,
    sequences: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"sequences": sequences, "metadata": metadata}, f)
    meta_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    return str(output_path)

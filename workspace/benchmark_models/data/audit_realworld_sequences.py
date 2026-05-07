from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ..adapters.realworld import events_csv_to_sequences


def _sequence_stats(sequences: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(seq.get("event", [])) for seq in sequences]
    event_counts = Counter(int(ev) for seq in sequences for ev in seq.get("event", []))
    max_time = max((max(seq.get("time", [0.0])) for seq in sequences if seq.get("time")), default=0.0)
    return {
        "num_sequences": len(sequences),
        "num_events": int(sum(lengths)),
        "min_sequence_length": int(min(lengths, default=0)),
        "median_sequence_length": float(pd.Series(lengths).median()) if lengths else 0.0,
        "mean_sequence_length": float(pd.Series(lengths).mean()) if lengths else 0.0,
        "max_sequence_length": int(max(lengths, default=0)),
        "dim_process": int(len(event_counts)),
        "max_time": float(max_time),
        "top_event_counts": [
            {"event": int(event), "count": int(count)}
            for event, count in event_counts.most_common(20)
        ],
    }


def _load_pickle(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and {"sequences", "metadata"}.issubset(payload):
        return list(payload["sequences"]), dict(payload["metadata"])
    if isinstance(payload, dict) and {"train", "val", "test"}.issubset(payload):
        sequences = list(payload.get("train", [])) + list(payload.get("val", [])) + list(payload.get("test", []))
        return sequences, dict(payload.get("metadata", {}))
    if isinstance(payload, list):
        return list(payload), {}
    raise ValueError("unsupported sequence pickle format")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit real-world event sequences before running TPP benchmarks.")
    ap.add_argument("--input_pickle")
    ap.add_argument("--input_csv")
    ap.add_argument("--sequence_col")
    ap.add_argument("--time_col")
    ap.add_argument("--event_col")
    ap.add_argument("--top_k_event_types", type=int)
    ap.add_argument("--target_event_label")
    ap.add_argument("--min_events_per_sequence", type=int, default=2)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    if bool(args.input_pickle) == bool(args.input_csv):
        raise ValueError("pass exactly one of --input_pickle or --input_csv")
    if args.input_pickle:
        sequences, metadata = _load_pickle(args.input_pickle)
    else:
        if not args.sequence_col or not args.time_col or not args.event_col:
            raise ValueError("--input_csv requires --sequence_col, --time_col, --event_col")
        sequences, metadata = events_csv_to_sequences(
            csv_path=args.input_csv,
            sequence_col=args.sequence_col,
            time_col=args.time_col,
            event_col=args.event_col,
            top_k_event_types=args.top_k_event_types,
            target_event_label=args.target_event_label,
            min_events_per_sequence=int(args.min_events_per_sequence),
        )
    out = {
        "input": args.input_pickle or args.input_csv,
        "stats": _sequence_stats(sequences),
        "metadata": metadata,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

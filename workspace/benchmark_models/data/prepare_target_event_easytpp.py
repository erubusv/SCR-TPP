from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

from ..adapters.easytpp import deterministic_split, write_gatech_pickles


def _load_sequences(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "sequences" in payload:
        return list(payload["sequences"]), dict(payload.get("metadata", {}) or {})
    if isinstance(payload, list):
        return list(payload), {}
    raise ValueError("expected a sequence pickle with either a list or {'sequences', 'metadata'}")


def _target_only_sequence(
    seq: dict[str, Any],
    *,
    target_event_id: int,
    min_target_events: int,
) -> dict[str, Any] | None:
    times = [float(t) for t in seq.get("time", [])]
    events = [int(e) for e in seq.get("event", [])]
    if len(times) != len(events):
        raise ValueError("sequence time/event length mismatch")
    target_times = [time for time, event in zip(times, events) if event == int(target_event_id)]
    if len(target_times) < int(min_target_events):
        return None
    return {
        "time": target_times,
        "event": [0 for _ in target_times],
        "sequence_id": seq.get("sequence_id"),
    }


def _filter_split(
    sequences: Sequence[dict[str, Any]],
    *,
    target_event_id: int,
    min_target_events: int,
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    target_event_count = 0
    for seq in sequences:
        filtered = _target_only_sequence(
            seq,
            target_event_id=int(target_event_id),
            min_target_events=int(min_target_events),
        )
        if filtered is None:
            continue
        target_event_count += len(filtered["event"])
        out.append(filtered)
    return out, target_event_count


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create EasyTPP Gatech-format inputs for target-event-only time prediction. "
            "This makes EasyTPP NLL/RMSE evaluate the fixed target event process rather "
            "than the full marked process."
        )
    )
    ap.add_argument("--sequence_pickle", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_event_id", type=int, required=True)
    ap.add_argument("--target_event_label", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--min_target_events", type=int, default=2)
    args = ap.parse_args()

    sequences, source_metadata = _load_sequences(args.sequence_pickle)
    train_raw, dev_raw, test_raw = deterministic_split(
        sequences,
        train_ratio=float(args.train_ratio),
        dev_ratio=float(args.dev_ratio),
    )
    train, train_target_events = _filter_split(
        train_raw,
        target_event_id=int(args.target_event_id),
        min_target_events=int(args.min_target_events),
    )
    dev, dev_target_events = _filter_split(
        dev_raw,
        target_event_id=int(args.target_event_id),
        min_target_events=int(args.min_target_events),
    )
    test, test_target_events = _filter_split(
        test_raw,
        target_event_id=int(args.target_event_id),
        min_target_events=int(args.min_target_events),
    )
    if not train or not dev or not test:
        raise ValueError("target-event filtering produced an empty train/dev/test split")

    out_dir = Path(args.output_dir)
    files = write_gatech_pickles(
        train=train,
        dev=dev,
        test=test,
        dim_process=1,
        output_dir=out_dir,
    )
    manifest = {
        "format": "easytpp_gatech",
        "evaluation_scope": "target_event_process",
        "target_event_id": int(args.target_event_id),
        "target_event_label": str(args.target_event_label),
        "source_sequence_pickle": str(args.sequence_pickle),
        "source_dim_process": source_metadata.get("dim_process"),
        "dim_process": 1,
        "min_target_events_per_sequence": int(args.min_target_events),
        "original_split_sequences": {
            "train": len(train_raw),
            "dev": len(dev_raw),
            "test": len(test_raw),
        },
        "target_process_split_sequences": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        },
        "target_process_event_counts": {
            "train": train_target_events,
            "dev": dev_target_events,
            "test": test_target_events,
        },
        "dropped_sequences": {
            "train": len(train_raw) - len(train),
            "dev": len(dev_raw) - len(dev),
            "test": len(test_raw) - len(test),
        },
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

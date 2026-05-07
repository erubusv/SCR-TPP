from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence


def sequence_to_gatech_events(seq: dict[str, Any]) -> list[dict[str, Any]]:
    times = [float(t) for t in seq.get("time", [])]
    events = [int(e) for e in seq.get("event", [])]
    if len(times) != len(events):
        raise ValueError("sequence time/event length mismatch")
    last_by_type: dict[int, float] = {}
    out: list[dict[str, Any]] = []
    prev = 0.0
    for idx, (time, event_type) in enumerate(zip(times, events)):
        if idx > 0 and time < times[idx - 1]:
            raise ValueError("event times must be non-decreasing")
        last_same = last_by_type.get(event_type, 0.0)
        out.append(
            {
                "idx_event": int(idx),
                "time_since_last_event": float(time - prev),
                "time_since_last_same_event": float(time - last_same),
                "type_event": int(event_type),
                "time_since_start": float(time),
            }
        )
        prev = float(time)
        last_by_type[int(event_type)] = float(time)
    return out


def convert_split_to_gatech(sequences: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [sequence_to_gatech_events(seq) for seq in sequences]


def write_gatech_pickles(
    *,
    train: Sequence[dict[str, Any]],
    dev: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    dim_process: int,
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train": {"dim_process": int(dim_process), "train": convert_split_to_gatech(train)},
        "dev": {"dim_process": int(dim_process), "dev": convert_split_to_gatech(dev)},
        "test": {"dim_process": int(dim_process), "test": convert_split_to_gatech(test)},
    }
    paths = {}
    for split, payload in payloads.items():
        path = output_dir / f"{split}.pkl"
        with path.open("wb") as f:
            pickle.dump(payload, f)
        paths[split] = str(path)
    manifest = {
        "format": "easytpp_gatech",
        "dim_process": int(dim_process),
        "files": paths,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paths


def deterministic_split(
    sequences: Sequence[dict[str, Any]],
    *,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.0 < train_ratio < 1.0 and 0.0 <= dev_ratio < 1.0 and train_ratio + dev_ratio < 1.0):
        raise ValueError("invalid split ratios")
    n = len(sequences)
    n_train = int(round(n * float(train_ratio)))
    n_dev = int(round(n * float(dev_ratio)))
    train = list(sequences[:n_train])
    dev = list(sequences[n_train:n_train + n_dev])
    test = list(sequences[n_train + n_dev:])
    return train, dev, test

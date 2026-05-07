from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.easytpp import write_gatech_pickles


def _load_sequences(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "sequences" not in payload:
        raise ValueError(f"expected sequence pickle with a 'sequences' field: {path}")
    return list(payload["sequences"]), dict(payload.get("metadata", {}) or {})


def _write_pickle(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return str(path)


def _counts(sequences: list[dict[str, Any]], target: int) -> dict[str, int]:
    event_counts: Counter[int] = Counter()
    target_positive = 0
    for seq in sequences:
        has_target = False
        times = seq.get("time", [])
        events = seq.get("event", [])
        if len(times) != len(events):
            raise ValueError(f"sequence {seq.get('sequence_id')} has mismatched time/event lengths")
        for i, (time, event) in enumerate(zip(times, events)):
            if i and float(time) < float(times[i - 1]):
                raise ValueError(f"sequence {seq.get('sequence_id')} has decreasing event times")
            event_id = int(event)
            event_counts[event_id] += 1
            has_target = has_target or event_id == int(target)
        target_positive += int(has_target)
    return {
        "sequences": len(sequences),
        "events": int(sum(event_counts.values())),
        "target_events": int(event_counts[int(target)]),
        "non_target_events": int(sum(v for k, v in event_counts.items() if int(k) != int(target))),
        "target_positive_sequences": int(target_positive),
    }


def _split_random(
    sequences: list[dict[str, Any]],
    *,
    seed: int,
    train_size: int,
    val_size: int,
    test_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = int(train_size) + int(val_size) + int(test_size)
    if total > len(sequences):
        raise ValueError(f"requested {total} sequences, but only {len(sequences)} are available")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(sequences), dtype=np.int64)
    rng.shuffle(idx)
    train_idx = idx[: int(train_size)]
    val_idx = idx[int(train_size) : int(train_size) + int(val_size)]
    test_idx = idx[int(train_size) + int(val_size) : total]
    return (
        [sequences[int(i)] for i in train_idx],
        [sequences[int(i)] for i in val_idx],
        [sequences[int(i)] for i in test_idx],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare random 5000/1000/1000 MIMIC low-urine splits.")
    ap.add_argument("--source_pickle", default="data/realworld_prepared/mimic_low_urine/sequences.pkl")
    ap.add_argument("--output_dir", default="data/realworld_prepared/mimic_low_urine_random_5000")
    ap.add_argument("--seeds", default="111,222,333")
    ap.add_argument("--train_size", type=int, default=5000)
    ap.add_argument("--val_size", type=int, default=1000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--copy_sequences", action="store_true", help="copy instead of symlinking sequences.pkl")
    args = ap.parse_args()

    source_pickle = Path(args.source_pickle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences, metadata = _load_sequences(source_pickle)
    target = int(metadata.get("target_event_id", 0))
    dim_process = int(metadata.get("dim_process", metadata.get("num_types", 0)))
    if dim_process <= 0:
        dim_process = 1 + max(int(e) for seq in sequences for e in seq.get("event", []))

    target_sequence_pickle = output_dir / "sequences.pkl"
    if target_sequence_pickle.exists() or target_sequence_pickle.is_symlink():
        target_sequence_pickle.unlink()
    if args.copy_sequences:
        shutil.copy2(source_pickle, target_sequence_pickle)
    else:
        target_sequence_pickle.symlink_to(source_pickle.resolve())

    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    split_paths: dict[str, str] = {}
    easytpp_paths: dict[str, dict[str, str]] = {}
    summaries = []
    for seed in seeds:
        train, val, test = _split_random(
            sequences,
            seed=int(seed),
            train_size=int(args.train_size),
            val_size=int(args.val_size),
            test_size=int(args.test_size),
        )
        split_metadata = {
            **metadata,
            "num_types": int(dim_process),
            "dim_process": int(dim_process),
            "split_seed": int(seed),
            "split_strategy": "sequence_level_uniform_random_fixed_size",
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "test_size": int(args.test_size),
            "source_sequence_pickle": str(source_pickle),
        }
        split_payload = {"train": train, "val": val, "test": test, "metadata": split_metadata}
        split_paths[str(seed)] = _write_pickle(output_dir / f"split_seed{seed}.pkl", split_payload)
        easytpp_dir = output_dir / f"easytpp_seed{seed}"
        easytpp_paths[str(seed)] = write_gatech_pickles(
            train=train,
            dev=val,
            test=test,
            dim_process=int(dim_process),
            output_dir=easytpp_dir,
        )
        summaries.append(
            {
                "seed": int(seed),
                "train": _counts(train, target),
                "val": _counts(val, target),
                "test": _counts(test, target),
            }
        )

    default_seed = str(seeds[0])
    default_easytpp = output_dir / "easytpp"
    if default_easytpp.exists() or default_easytpp.is_symlink():
        if default_easytpp.is_dir() and not default_easytpp.is_symlink():
            shutil.rmtree(default_easytpp)
        else:
            default_easytpp.unlink()
    default_easytpp.symlink_to((output_dir / f"easytpp_seed{default_seed}").resolve())

    manifest = {
        "dataset": "mimic_iv_low_urine_random_5000",
        "source_pickle": str(source_pickle),
        "sequence_pickle": str(target_sequence_pickle),
        "num_available_sequences": len(sequences),
        "target_event_id": int(target),
        "target_event_label": metadata.get("target_event_label", "low_urine_output"),
        "dim_process": int(dim_process),
        "seeds": seeds,
        "split_paths": split_paths,
        "easytpp_paths": easytpp_paths,
        "default_easytpp_seed": int(default_seed),
        "split_summary": summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

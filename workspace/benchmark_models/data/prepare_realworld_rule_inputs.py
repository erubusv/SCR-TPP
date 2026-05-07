from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_sequences(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "sequences" in payload:
        return list(payload["sequences"]), dict(payload.get("metadata", {}))
    if isinstance(payload, list):
        return list(payload), {}
    raise ValueError(f"unsupported sequence pickle format: {path}")


def _load_predefined_split(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or not {"train", "val", "test"}.issubset(payload):
        raise ValueError("--split_pickle must contain a dict with train/val/test keys")
    train = list(payload.get("train", []))
    dev = list(payload.get("val", []))
    test = list(payload.get("test", []))
    if not train or not dev or not test:
        raise ValueError("--split_pickle produced an empty train/val/test split")
    return train, dev, test, dict(payload.get("metadata", {}) or {})


def _split_by_seed(
    sequences: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.0 < train_ratio < 1.0 and 0.0 <= dev_ratio < 1.0 and train_ratio + dev_ratio < 1.0):
        raise ValueError("invalid split ratios")
    idx = np.arange(len(sequences), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    n_train = int(round(len(idx) * float(train_ratio)))
    n_dev = int(round(len(idx) * float(dev_ratio)))
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]
    return (
        [sequences[int(i)] for i in train_idx],
        [sequences[int(i)] for i in dev_idx],
        [sequences[int(i)] for i in test_idx],
    )


def _max_time(sequences: list[dict[str, Any]]) -> float:
    return max((max(map(float, seq.get("time", []) or [0.0])) for seq in sequences), default=0.0)


def _event_counts(sequences: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for seq in sequences:
        for event in seq.get("event", []):
            event_id = int(event)
            counts[event_id] = int(counts.get(event_id, 0)) + 1
    return counts


def _write_pickle(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return str(path)


def _dummy_source(target: int, num_event_types: int) -> int:
    for event_id in range(int(num_event_types)):
        if int(event_id) != int(target):
            return int(event_id)
    raise ValueError("need at least two event types for rule baselines")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create train-only rule-discovery inputs from a real-world sequence pickle.")
    ap.add_argument("--sequence_pickle", required=True)
    ap.add_argument(
        "--split_pickle",
        help=(
            "Optional predefined train/val/test split. Use this for real-world "
            "prediction benchmarks so logical rule baselines and EasyTPP use "
            "the identical sequence split."
        ),
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--target_event_id", type=int, required=True)
    ap.add_argument("--target_event_label", required=True)
    ap.add_argument("--seeds", default="111,222,333")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--max_lag_days", type=float, default=30.0)
    args = ap.parse_args()

    sequences, metadata = _load_sequences(args.sequence_pickle)
    if not sequences:
        raise ValueError("no sequences available")
    num_event_types = int(metadata.get("dim_process", metadata.get("num_types", 0)))
    if num_event_types <= 1:
        num_event_types = 1 + max(int(event) for seq in sequences for event in seq.get("event", []))
    target = int(args.target_event_id)
    if not (0 <= target < num_event_types):
        raise ValueError(f"target_event_id={target} outside [0,{num_event_types})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    predefined_split: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None = None
    if args.split_pickle:
        train_fixed, dev_fixed, test_fixed, split_metadata = _load_predefined_split(args.split_pickle)
        predefined_split = (train_fixed, dev_fixed, test_fixed, split_metadata)
        split_seed = int(split_metadata.get("split_seed", seeds[0] if seeds else 111))
        seeds = [split_seed]
        split_num_types = int(split_metadata.get("num_types", split_metadata.get("dim_process", num_event_types)))
        if int(split_num_types) != int(num_event_types):
            raise ValueError(
                f"split num_types={split_num_types} does not match sequence num_event_types={num_event_types}"
            )
    time_horizon = float(_max_time(sequences))
    max_lag = max(float(args.max_lag_days), 1.0)
    dummy_src = _dummy_source(target, num_event_types)
    config = {
        "dataset": str(args.dataset_name),
        "num_event_types": int(num_event_types),
        "time_horizon": float(time_horizon),
        "time_window": float(time_horizon),
        "activation_mode": "product_max_witness",
        "intensity_model": "canonical_loglink",
        "kernel": "triangular",
        "target_event_label": str(args.target_event_label),
        "realworld_rule_discovery": True,
        "rules": [
            {
                "target": int(target),
                "sign": "excitation",
                "condition": {
                    int(dummy_src): {
                        "peaks": [float(max_lag) / 2.0],
                        "widths": [float(max_lag)],
                        "mix_weights": [1.0],
                        "support_mults": [1.0],
                    }
                },
                "note": "dummy target anchor; real-world discovery has no ground-truth rule labels",
            }
        ],
    }
    config_path = out_dir / "rule_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    split_rows = []
    train_only_paths: dict[str, str] = {}
    split_paths: dict[str, str] = {}
    config_paths: dict[str, str] = {}
    for seed in seeds:
        if predefined_split is None:
            train, dev, test = _split_by_seed(
                sequences,
                seed=int(seed),
                train_ratio=float(args.train_ratio),
                dev_ratio=float(args.dev_ratio),
            )
            source_split_metadata = metadata
            split_strategy = "sequence_level_seeded_shuffle"
        else:
            train, dev, test, source_split_metadata = predefined_split
            split_strategy = str(source_split_metadata.get("split_strategy", "predefined_sequence_split"))
        seed_config_path = out_dir / f"rule_config_seed{seed}.yaml"
        seed_config = {
            **config,
            "seed": int(seed),
            "path": {"output_path": str((out_dir / f"rule_train_seed{seed}.pkl").resolve())},
        }
        split_payload = {
            "train": train,
            "val": dev,
            "test": test,
            "metadata": {
                "num_types": int(num_event_types),
                "dim_process": int(num_event_types),
                "config": seed_config,
                "source_metadata": metadata,
                "source_split_metadata": source_split_metadata,
                "split_seed": int(seed),
                "split_strategy": split_strategy,
            },
        }
        train_only_payload = {
            "train": train,
            "val": [],
            "test": [],
            "metadata": {
                **split_payload["metadata"],
                "split_strategy": "train_only_rule_discovery_from_sequence_level_seeded_shuffle",
            },
        }
        split_paths[str(seed)] = _write_pickle(out_dir / f"split_seed{seed}.pkl", split_payload)
        train_only_paths[str(seed)] = _write_pickle(out_dir / f"rule_train_seed{seed}.pkl", train_only_payload)
        seed_config_path.write_text(yaml.safe_dump(seed_config, sort_keys=False))
        config_paths[str(seed)] = str(seed_config_path)
        counts = _event_counts(train)
        split_rows.append(
            {
                "seed": int(seed),
                "train_sequences": len(train),
                "dev_sequences": len(dev),
                "test_sequences": len(test),
                "train_events": int(sum(counts.values())),
                "train_target_events": int(counts.get(target, 0)),
            }
        )

    manifest = {
        "dataset": str(args.dataset_name),
        "sequence_pickle": str(args.sequence_pickle),
        "split_pickle": str(args.split_pickle) if args.split_pickle else None,
        "num_sequences": len(sequences),
        "num_event_types": int(num_event_types),
        "target_event_id": int(target),
        "target_event_label": str(args.target_event_label),
        "train_ratio": float(args.train_ratio),
        "dev_ratio": float(args.dev_ratio),
        "test_ratio": float(1.0 - float(args.train_ratio) - float(args.dev_ratio)),
        "time_horizon": float(time_horizon),
        "max_lag_days": float(max_lag),
        "config_path": str(config_path),
        "config_paths": config_paths,
        "split_paths": split_paths,
        "rule_train_paths": train_only_paths,
        "split_summary": split_rows,
        "source_metadata": metadata,
        "split_policy": (
            "predefined_split_shared_with_prediction_benchmarks"
            if args.split_pickle
            else "seeded_shuffle_generated_by_rule_input_adapter"
        ),
    }
    manifest_path = out_dir / "rule_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

from ..adapters.easytpp import deterministic_split, write_gatech_pickles
from ..core.prediction_external import (
    load_prediction_manifest,
    run_external_prediction_baseline,
    write_prediction_example_manifest,
)
from ..adapters.realworld import events_csv_to_sequences, write_sequence_pickle
from ..core.summarize import write_jsonl


def _load_existing_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _jsonl_has_requested_models(path: Path, requested_models: list[str]) -> bool:
    if not path.exists():
        return False
    seen = set()
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                model = row.get("model")
                if model is not None:
                    seen.add(str(model))
    except Exception:
        return False
    return set(requested_models).issubset(seen)


def _run_prediction_with_oom_batch_fallback(
    *,
    spec,
    dataset: str,
    target: str,
    train_path,
    dev_path,
    test_path,
    output_path: Path,
    seed: int,
    extra_values: dict,
    min_batch_size: int = 8,
):
    current_values = dict(extra_values)
    batch_size = int(current_values.get("batch_size", 0) or 0)
    while True:
        try:
            return run_external_prediction_baseline(
                spec=spec,
                dataset=dataset,
                target=target,
                train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                output_path=output_path,
                seed=seed,
                extra_values=current_values,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "CUDA out of memory" not in message or batch_size <= min_batch_size:
                raise
            batch_size = max(min_batch_size, batch_size // 2)
            current_values["batch_size"] = batch_size


def _acquire_prediction_result_lock(result_path: Path, requested_models: list[str]) -> Path | None:
    """Avoid duplicate GPU jobs when the same result group is pre-launched."""
    if _jsonl_has_requested_models(result_path, requested_models):
        return None
    lock_path = Path(str(result_path) + ".lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"pid": os.getpid(), "result_path": str(result_path)}) + "\n")
            return lock_path
        except FileExistsError:
            while lock_path.exists():
                if _jsonl_has_requested_models(result_path, requested_models):
                    return None
                time.sleep(10)
            if _jsonl_has_requested_models(result_path, requested_models):
                return None


def load_sequence_pickle(path: str | Path) -> tuple[list[dict], int, dict[str, list[dict]] | None, dict]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and {"train", "val", "test"}.issubset(data):
        train = list(data.get("train", []))
        dev = list(data.get("val", []))
        test = list(data.get("test", []))
        predefined_splits = {"train": train, "dev": dev, "test": test}
        if not train or not dev or not test:
            predefined_splits = None
        sequences = train + dev + test
        metadata = data.get("metadata", {})
        dim_process = int(metadata.get("num_types", metadata.get("dim_process", 0)))
    elif isinstance(data, dict) and {"sequences", "metadata"}.issubset(data):
        sequences = list(data.get("sequences", []))
        metadata = data.get("metadata", {})
        dim_process = int(metadata.get("dim_process", metadata.get("num_types", 0)))
        predefined_splits = None
    elif isinstance(data, list):
        sequences = list(data)
        dim_process = 1 + max(int(ev) for seq in sequences for ev in seq.get("event", []))
        predefined_splits = None
    else:
        raise ValueError("expected paper-suite dict or list of sequence dicts")
    if dim_process <= 0:
        raise ValueError("cannot infer dim_process")
    return sequences, dim_process, predefined_splits, dict(metadata or {})


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare real-world prediction benchmark inputs.")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["mimic_iv", "bpi2017", "generic_event_log", "generic_pickle"],
    )
    ap.add_argument("--target", default="low_urine_output")
    ap.add_argument("--input_pickle", help="pre-extracted sequence pickle; raw MIMIC is not committed")
    ap.add_argument("--input_csv", help="generic event-log CSV for MIMIC-IV or another event-log dataset")
    ap.add_argument("--sequence_col", help="CSV sequence id column")
    ap.add_argument("--time_col", help="CSV event time column")
    ap.add_argument("--event_col", help="CSV event type column")
    ap.add_argument("--top_k_event_types", type=int, help="keep top-k event labels by frequency")
    ap.add_argument("--target_event_label", help="event label to keep under top-k filtering")
    ap.add_argument("--min_events_per_sequence", type=int, default=2)
    ap.add_argument("--no_start_at_zero", action="store_true")
    ap.add_argument("--write_sequence_pickle", action="store_true")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--prepare_easytpp", action="store_true")
    ap.add_argument("--prediction_manifest", help="external prediction baseline manifest")
    ap.add_argument("--models", default="", help="comma-separated prediction models to run")
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--gpu", type=int, default=-1, help="GPU id passed to EasyTPP wrappers; -1 uses CPU")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_epoch", type=int, default=20)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--time_emb_size", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--max_len", type=int)
    ap.add_argument("--target_prediction_grid_size", type=int, default=128)
    ap.add_argument("--prediction_results_jsonl", help="path for real-world result rows")
    ap.add_argument("--write_prediction_example_manifest")
    args = ap.parse_args()

    if args.write_prediction_example_manifest:
        write_prediction_example_manifest(args.write_prediction_example_manifest)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.input_pickle) == bool(args.input_csv):
        raise ValueError("pass exactly one of --input_pickle or --input_csv")
    source_metadata = {}
    input_ref = args.input_pickle or args.input_csv
    predefined_splits = None
    if args.input_pickle:
        sequences, dim_process, predefined_splits, source_metadata = load_sequence_pickle(args.input_pickle)
    else:
        required_csv_args = [args.sequence_col, args.time_col, args.event_col]
        if any(value is None for value in required_csv_args):
            raise ValueError("--input_csv requires --sequence_col, --time_col, and --event_col")
        sequences, source_metadata = events_csv_to_sequences(
            csv_path=args.input_csv,
            sequence_col=str(args.sequence_col),
            time_col=str(args.time_col),
            event_col=str(args.event_col),
            top_k_event_types=args.top_k_event_types,
            target_event_label=args.target_event_label,
            min_events_per_sequence=args.min_events_per_sequence,
            start_at_zero=not args.no_start_at_zero,
        )
        dim_process = int(source_metadata["dim_process"])
        if args.write_sequence_pickle:
            source_metadata["sequence_pickle"] = write_sequence_pickle(
                sequences=sequences,
                metadata=source_metadata,
                output_path=out_dir / "sequences.pkl",
            )
    if predefined_splits is None:
        train, dev, test = deterministic_split(
            sequences,
            train_ratio=float(args.train_ratio),
            dev_ratio=float(args.dev_ratio),
        )
    else:
        train = predefined_splits["train"]
        dev = predefined_splits["dev"]
        test = predefined_splits["test"]
    manifest = {
        "dataset": args.dataset,
        "target": args.target,
        "input": str(input_ref),
        "num_sequences": len(sequences),
        "dim_process": int(dim_process),
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "source_metadata": source_metadata,
    }
    if args.prepare_easytpp:
        manifest["easytpp_files"] = write_gatech_pickles(
            train=train,
            dev=dev,
            test=test,
            dim_process=dim_process,
            output_dir=out_dir / "easytpp",
        )
    requested_models = [model.strip() for model in args.models.split(",") if model.strip()]
    if requested_models:
        result_path = Path(args.prediction_results_jsonl or str(out_dir / "prediction_results.jsonl"))
        lock_path = _acquire_prediction_result_lock(result_path, requested_models)
        if lock_path is None:
            manifest["prediction_results_jsonl"] = str(result_path)
            manifest["prediction_models"] = requested_models
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(json.dumps(manifest, indent=2))
            return
        if int(dim_process) <= 1 and source_metadata.get("target_event_id") is not None:
            raise ValueError(
                "target-mark real-world prediction must use the full marked event history. "
                "Do not pass target-only EasyTPP inputs for the main benchmark."
            )
        if not args.prediction_manifest:
            raise ValueError("--models requires --prediction_manifest")
        if "easytpp_files" not in manifest:
            manifest["easytpp_files"] = write_gatech_pickles(
                train=train,
                dev=dev,
                test=test,
                dim_process=dim_process,
                output_dir=out_dir / "easytpp",
            )
        specs = load_prediction_manifest(args.prediction_manifest)
        target_event_id = source_metadata.get("target_event_id")
        target_event_label = source_metadata.get("target_event_label") or args.target_event_label or args.target
        extra_values = {}
        if target_event_id is not None:
            extra_values = {
                "target_event_id": int(target_event_id),
                "target_event_label": str(target_event_label),
                "gpu": int(args.gpu),
                "batch_size": int(args.batch_size),
                "max_epoch": int(args.max_epoch),
                "learning_rate": float(args.learning_rate),
                "hidden_size": int(args.hidden_size),
                "time_emb_size": int(args.time_emb_size),
                "num_layers": int(args.num_layers),
                "num_heads": int(args.num_heads),
                "mc_samples": int(args.mc_samples),
                "max_len": 0 if args.max_len is None else int(args.max_len),
                "target_prediction_grid_size": int(args.target_prediction_grid_size),
            }
        else:
            extra_values = {
                "gpu": int(args.gpu),
                "batch_size": int(args.batch_size),
                "max_epoch": int(args.max_epoch),
                "learning_rate": float(args.learning_rate),
                "hidden_size": int(args.hidden_size),
                "time_emb_size": int(args.time_emb_size),
                "num_layers": int(args.num_layers),
                "num_heads": int(args.num_heads),
                "mc_samples": int(args.mc_samples),
                "max_len": 0 if args.max_len is None else int(args.max_len),
                "target_prediction_grid_size": int(args.target_prediction_grid_size),
            }
            needs_target = any(
                "{target_event_id}" in str(part)
                for spec in specs.values()
                for part in (spec.command if isinstance(spec.command, list) else [spec.command])
            )
            if needs_target:
                raise ValueError(
                    "prediction manifest requires {target_event_id}, but input metadata has no target_event_id"
                )
        try:
            result_rows = _load_existing_jsonl_rows(result_path)
            completed = {str(row.get("model")) for row in result_rows}
            for model in requested_models:
                if model in completed:
                    continue
                if model not in specs:
                    raise KeyError(f"model {model!r} not found in prediction manifest")
                result = _run_prediction_with_oom_batch_fallback(
                    spec=specs[model],
                    dataset=args.dataset,
                    target=args.target,
                    train_path=manifest["easytpp_files"]["train"],
                    dev_path=manifest["easytpp_files"]["dev"],
                    test_path=manifest["easytpp_files"]["test"],
                    output_path=out_dir / "prediction_outputs" / f"{model}_seed{args.seed}.json",
                    seed=int(args.seed),
                    extra_values=extra_values,
                )
                result_rows.append(result.to_dict())
                write_jsonl(result_rows, result_path)
            manifest["prediction_results_jsonl"] = str(result_path)
            manifest["prediction_models"] = requested_models
        finally:
            lock_path.unlink(missing_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

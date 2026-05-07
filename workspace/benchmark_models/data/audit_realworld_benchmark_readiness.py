from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def _seq_ids(sequences: list[dict[str, Any]]) -> list[str]:
    return [str(seq.get("sequence_id")) for seq in sequences]


def _event_counts(sequences: list[dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for seq in sequences:
        times = seq.get("time", [])
        events = seq.get("event", [])
        if len(times) != len(events):
            raise ValueError(f"sequence {seq.get('sequence_id')} has mismatched time/event lengths")
        for idx, (t, event) in enumerate(zip(times, events)):
            if idx and float(t) < float(times[idx - 1]):
                raise ValueError(f"sequence {seq.get('sequence_id')} has decreasing event times")
            counts[int(event)] += 1
    return counts


def _add(checks: list[dict[str, Any]], name: str, ok: bool, details: dict[str, Any] | str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "details": details})


def _manifest_commands_include(manifest: dict[str, Any], *tokens: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for spec in manifest.get("baselines", []):
        command = [str(part) for part in spec.get("command", [])]
        out[str(spec.get("name"))] = all(token in command for token in tokens)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict audit before MIMIC real-world benchmark execution.")
    ap.add_argument("--prepared_dir", default="data/realworld_prepared/mimic_low_urine")
    ap.add_argument("--split_pickle", default="data/realworld_prepared/mimic_low_urine/split_seed111.pkl")
    ap.add_argument("--rule_input_manifest", default="data/realworld_prepared/mimic_low_urine/rule_inputs/rule_input_manifest.json")
    ap.add_argument("--prediction_manifest", default="workspace/benchmark_models/configs/prediction_baselines.easytpp_official.yaml")
    ap.add_argument("--rule_baseline_manifest", default="workspace/benchmark_models/configs/baselines.official_or_faithful.yaml")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    prepared_dir = Path(args.prepared_dir)
    checks: list[dict[str, Any]] = []

    sequence_payload = _load_pickle(prepared_dir / "sequences.pkl")
    metadata = dict(sequence_payload.get("metadata", {}) or {})
    sequences = list(sequence_payload.get("sequences", []))
    target = int(metadata.get("target_event_id", -1))
    dim_process = int(metadata.get("dim_process", metadata.get("num_types", 0)))
    id_to_event = {int(k): str(v) for k, v in dict(metadata.get("id_to_event", {})).items()}
    _add(
        checks,
        "sequence_pickle_has_single_target_metadata",
        target >= 0 and target < dim_process and bool(metadata.get("target_event_label")),
        {"target_event_id": target, "target_event_label": metadata.get("target_event_label"), "dim_process": dim_process},
    )
    counts_all = _event_counts(sequences)
    _add(
        checks,
        "full_sequences_keep_non_target_history",
        dim_process > 1 and sum(v for k, v in counts_all.items() if k != target) > 0,
        {"num_sequences": len(sequences), "target_events": int(counts_all[target]), "non_target_events": int(sum(v for k, v in counts_all.items() if k != target))},
    )

    split = _load_pickle(args.split_pickle)
    split_counts: dict[str, dict[str, int]] = {}
    split_ids: dict[str, set[str]] = {}
    for split_name, key in (("train", "train"), ("dev", "val"), ("test", "test")):
        seqs = list(split.get(key, []))
        counts = _event_counts(seqs)
        split_counts[split_name] = {
            "sequences": len(seqs),
            "events": int(sum(counts.values())),
            "target_events": int(counts[target]),
            "non_target_events": int(sum(v for k, v in counts.items() if k != target)),
            "target_sequences": int(sum(any(int(e) == target for e in seq.get("event", [])) for seq in seqs)),
        }
        split_ids[split_name] = set(_seq_ids(seqs))
    no_overlap = not (
        split_ids["train"] & split_ids["dev"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["dev"] & split_ids["test"]
    )
    _add(
        checks,
        "sequence_split_is_disjoint_and_target_present",
        no_overlap and all(row["target_events"] > 0 and row["non_target_events"] > 0 for row in split_counts.values()),
        split_counts,
    )

    easy_paths = {
        "train": prepared_dir / "easytpp" / "train.pkl",
        "dev": prepared_dir / "easytpp" / "dev.pkl",
        "test": prepared_dir / "easytpp" / "test.pkl",
    }
    easy_counts: dict[str, dict[str, int]] = {}
    for split_name, path in easy_paths.items():
        payload = _load_pickle(path)
        seqs = payload[split_name]
        type_counts = Counter(int(ev["type_event"]) for seq in seqs for ev in seq)
        easy_counts[split_name] = {
            "dim_process": int(payload.get("dim_process", 0)),
            "sequences": len(seqs),
            "events": int(sum(type_counts.values())),
            "target_events": int(type_counts[target]),
            "non_target_events": int(sum(v for k, v in type_counts.items() if k != target)),
        }
    _add(
        checks,
        "easytpp_inputs_are_full_marked_process",
        all(row["dim_process"] == dim_process and row["non_target_events"] > 0 for row in easy_counts.values()),
        easy_counts,
    )

    rule_manifest = _load_json(args.rule_input_manifest)
    first_seed = str(sorted(rule_manifest["split_paths"].keys(), key=int)[0])
    rule_split = _load_pickle(rule_manifest["split_paths"][first_seed])
    same_split = all(
        _seq_ids(list(rule_split[rule_key])) == _seq_ids(list(split[split_key]))
        for rule_key, split_key in (("train", "train"), ("val", "val"), ("test", "test"))
    )
    train_only = _load_pickle(rule_manifest["rule_train_paths"][first_seed])
    _add(
        checks,
        "logical_rule_split_matches_prediction_split",
        same_split,
        {"seed": int(first_seed), "split_policy": rule_manifest.get("split_policy")},
    )
    _add(
        checks,
        "logical_rule_learning_receives_train_only_payload",
        bool(train_only.get("train")) and not train_only.get("val") and not train_only.get("test"),
        {"train_sequences": len(train_only.get("train", [])), "val_sequences": len(train_only.get("val", [])), "test_sequences": len(train_only.get("test", []))},
    )
    rule_config = _load_yaml(rule_manifest["config_paths"][first_seed])
    dummy_sources = sorted({int(src) for rule in rule_config.get("rules", []) for src in rule.get("condition", {}).keys()})
    _add(
        checks,
        "logical_rule_config_uses_single_target_and_non_target_anchor",
        int(rule_config["num_event_types"]) == dim_process
        and all(int(rule.get("target")) == target for rule in rule_config.get("rules", []))
        and all(src != target for src in dummy_sources),
        {"config_path": rule_manifest["config_paths"][first_seed], "dummy_sources": dummy_sources},
    )

    prediction_manifest = _load_yaml(args.prediction_manifest)
    target_gpu_tokens = _manifest_commands_include(prediction_manifest, "--target_event_id", "--target_event_label", "--gpu")
    _add(
        checks,
        "easytpp_manifest_passes_target_and_gpu",
        all(target_gpu_tokens.values()),
        target_gpu_tokens,
    )
    rule_baseline_manifest = _load_yaml(args.rule_baseline_manifest)
    exclude_target_tokens = _manifest_commands_include(rule_baseline_manifest, "--exclude_target_from_sources")
    _add(
        checks,
        "logical_baseline_manifest_excludes_target_from_sources",
        all(exclude_target_tokens.values()),
        exclude_target_tokens,
    )
    rule_device_tokens = _manifest_commands_include(rule_baseline_manifest, "--device")
    device_required = {
        name: ok
        for name, ok in rule_device_tokens.items()
        if name in {"CLNN", "NSTPP", "CLUSTER"}
    }
    _add(
        checks,
        "gpu_logical_baseline_manifest_passes_device",
        all(device_required.values()),
        device_required,
    )

    output = {
        "prepared_dir": str(prepared_dir),
        "target_event_id": target,
        "target_event_label": metadata.get("target_event_label"),
        "id_to_event": id_to_event,
        "checks": checks,
        "ok": all(row["ok"] for row in checks),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    if not output["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from ..core.external import load_baseline_manifest, run_external_rule_baseline
from ..core.schema import NormalizedRule
from ..core.summarize import write_jsonl


def _rule_key(rule: NormalizedRule) -> tuple[tuple[int, ...], str, int]:
    return (tuple(rule.sources), str(rule.sign), int(rule.target))


def _set_jaccard(left: set[tuple[tuple[int, ...], str, int]], right: set[tuple[tuple[int, ...], str, int]]) -> float:
    if not left and not right:
        return 1.0
    return float(len(left & right)) / float(max(len(left | right), 1))


def _source_overlap(rule_a: NormalizedRule, rule_b: NormalizedRule) -> float:
    if int(rule_a.target) != int(rule_b.target) or str(rule_a.sign) != str(rule_b.sign):
        return 0.0
    a = set(int(src) for src in rule_a.sources)
    b = set(int(src) for src in rule_b.sources)
    return float(len(a & b)) / float(max(len(a | b), 1))


def _greedy_source_jaccard(left: list[NormalizedRule], right: list[NormalizedRule]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            score = _source_overlap(a, b)
            if score > 0.0:
                pairs.append((float(score), int(i), int(j)))
    pairs.sort(reverse=True)
    used_l: set[int] = set()
    used_r: set[int] = set()
    total = 0.0
    for score, i, j in pairs:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        total += float(score)
    return float(total) / float(max(max(len(left), len(right)), 1))


def _stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    out = []
    for model, model_rows in sorted(by_model.items()):
        exact_vals: list[float] = []
        source_vals: list[float] = []
        for left, right in itertools.combinations(model_rows, 2):
            left_rules = [NormalizedRule.from_obj(rule) for rule in left.get("learned_rules", [])]
            right_rules = [NormalizedRule.from_obj(rule) for rule in right.get("learned_rules", [])]
            exact_vals.append(_set_jaccard({_rule_key(rule) for rule in left_rules}, {_rule_key(rule) for rule in right_rules}))
            source_vals.append(_greedy_source_jaccard(left_rules, right_rules))
        out.append(
            {
                "model": model,
                "num_splits": len(model_rows),
                "mean_exact_rule_jaccard_across_splits": float(sum(exact_vals) / len(exact_vals)) if exact_vals else None,
                "mean_signed_source_jaccard_across_splits": float(sum(source_vals) / len(source_vals)) if source_vals else None,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run real-world rule-discovery baselines on train-only split pickles.")
    ap.add_argument("--rule_input_manifest", required=True)
    ap.add_argument("--baseline_manifest", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="")
    args = ap.parse_args()

    manifest = json.loads(Path(args.rule_input_manifest).read_text())
    specs = load_baseline_manifest(args.baseline_manifest)
    requested = [part.strip() for part in str(args.models).split(",") if part.strip()]
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    if not seeds:
        seeds = [int(seed) for seed in sorted(manifest["rule_train_paths"].keys(), key=int)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        dataset_path = manifest["rule_train_paths"][str(seed)]
        for model in requested:
            if model not in specs:
                raise KeyError(f"model {model!r} not in {args.baseline_manifest}")
            raw_output = out_dir / "raw" / f"{model}_seed{seed}.json"
            rules, metadata = run_external_rule_baseline(
                spec=specs[model],
                config_path=manifest["config_path"],
                dataset_path=dataset_path,
                output_path=raw_output,
                seed=int(seed),
                extra_values={"device": str(args.device)},
            )
            rows.append(
                {
                    "model": str(model),
                    "dataset": str(manifest["dataset"]),
                    "target": str(manifest["target_event_label"]),
                    "target_event_id": int(manifest["target_event_id"]),
                    "seed": int(seed),
                    "split": "train",
                    "learned_rule_count": len(rules),
                    "learned_rules": [rule.to_dict() for rule in rules],
                    "runtime_sec": float(metadata.get("runtime_sec", 0.0)),
                    "training_time_sec": float(metadata.get("runtime_sec", 0.0)),
                    "model_metadata": metadata,
                }
            )

    results_path = out_dir / "rule_results.jsonl"
    write_jsonl(rows, results_path)
    summary = {
        "dataset": str(manifest["dataset"]),
        "target": str(manifest["target_event_label"]),
        "target_event_id": int(manifest["target_event_id"]),
        "models": requested,
        "seeds": seeds,
        "rows": len(rows),
        "stability": _stability(rows),
    }
    (out_dir / "rule_stability_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"results": str(results_path), **summary}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.schema import SyntheticRuleRecoveryResult, match_rule_sets, true_rules_from_config


def import_ours_result_file(
    *,
    result_path: str | Path,
    config_root: str | Path,
    model_name: str = "ours",
) -> list[SyntheticRuleRecoveryResult]:
    result_path = Path(result_path)
    config_root = Path(config_root)
    payload = json.loads(result_path.read_text())
    seed = _seed_from_result_filename(result_path)
    results: list[SyntheticRuleRecoveryResult] = []
    if isinstance(payload, dict) and "benchmark" in payload and (
        "predicted" in payload or "predicted_rules" in payload or "rule_parameters" in payload
    ):
        row_items = [(str(payload["benchmark"]), payload)]
    elif isinstance(payload, dict):
        row_items = list(payload.items())
    else:
        raise TypeError(f"unsupported ours result payload type in {result_path}: {type(payload)!r}")

    for dataset_name, row in row_items:
        if not isinstance(row, dict):
            continue
        config_path = config_root / f"{dataset_name}.yaml"
        truth = true_rules_from_config(config_path)
        target = int(row["target"])
        predicted = row.get("predicted", row.get("predicted_rules", row.get("rule_parameters", [])))
        runtime = row.get("elapsed_sec", row.get("runtime_sec", row.get("wall_sec")))
        matched = match_rule_sets(predicted, truth, default_target=target)
        result = SyntheticRuleRecoveryResult(
            model=model_name,
            dataset=str(dataset_name),
            seed=int(row.get("dataset_seed", seed)),
            predicted_rules=matched["predicted_rules"],
            matched=matched["matched"],
            missing=matched["missing"],
            extra=matched["extra"],
            recall=matched["recall"],
            precision=matched["precision"],
            f1=matched["f1"],
            source_jaccard_recall=matched["source_jaccard_recall"],
            source_jaccard_precision=matched["source_jaccard_precision"],
            source_jaccard_f1=matched["source_jaccard_f1"],
            signed_source_jaccard_recall=matched["signed_source_jaccard_recall"],
            signed_source_jaccard_precision=matched["signed_source_jaccard_precision"],
            signed_source_jaccard_f1=matched["signed_source_jaccard_f1"],
            source_jaccard_pairs=matched["source_jaccard_pairs"],
            signed_source_jaccard_pairs=matched["signed_source_jaccard_pairs"],
            runtime_sec=float(runtime) if runtime is not None else None,
            model_metadata={
                "source_result_path": str(result_path),
                "algorithm_profile": row.get("algorithm_profile"),
                "final_support_criterion": row.get("final_support_criterion"),
                "implementation_type": "native_runner_import",
            },
        )
        result.validate()
        results.append(result)
    return results


def _seed_from_result_filename(path: Path) -> int:
    stem = path.stem
    if "_seed" in stem:
        token = stem.rsplit("_seed", 1)[-1]
        if token.isdigit():
            return int(token)
    raise ValueError(f"cannot infer seed from result filename: {path.name}")


def import_ours_results_dir(
    *,
    results_dir: str | Path,
    config_root: str | Path,
    model_name: str = "ours",
) -> list[SyntheticRuleRecoveryResult]:
    results = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if "_seed" not in path.stem:
            continue
        results.extend(import_ours_result_file(result_path=path, config_root=config_root, model_name=model_name))
    return results

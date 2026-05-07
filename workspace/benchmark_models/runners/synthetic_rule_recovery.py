from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.external import load_baseline_manifest, run_external_rule_baseline, write_example_manifest
from ..scr_tpp.synthetic_results import import_ours_results_dir
from ..core.schema import SyntheticRuleRecoveryResult, match_rule_sets
from ..core.summarize import synthetic_summary, write_csv, write_jsonl
from ..adapters.synthetic import DEFAULT_CONFIG_ROOT, DEFAULT_DATASET_ROOT, iter_synthetic_cases, write_all_as_train_dataset


def parse_csv_set(spec: str) -> set[str]:
    return {part.strip() for part in str(spec).split(",") if part.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run or import synthetic rule recovery benchmarks.")
    ap.add_argument("--models", default="ours", help="comma-separated model names")
    ap.add_argument("--config_root", default=str(DEFAULT_CONFIG_ROOT))
    ap.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    ap.add_argument("--seeds", default="111,222,333")
    ap.add_argument("--only", default="")
    ap.add_argument("--ours_results_dir", default="")
    ap.add_argument("--baseline_manifest", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_csv", default="")
    ap.add_argument("--write_example_manifest", default="")
    args = ap.parse_args()

    if args.write_example_manifest:
        write_example_manifest(args.write_example_manifest)
        return

    models = parse_csv_set(args.models)
    seeds = tuple(int(seed) for seed in parse_csv_set(args.seeds))
    only = parse_csv_set(args.only) if args.only.strip() else None
    rows: list[dict] = []

    if "ours" in models:
        if not args.ours_results_dir:
            raise ValueError("--ours_results_dir is required when models includes ours")
        rows.extend(
            result.to_dict()
            for result in import_ours_results_dir(
                results_dir=args.ours_results_dir,
                config_root=args.config_root,
                model_name="ours",
            )
        )

    external_models = sorted(models - {"ours"})
    if external_models:
        if not args.baseline_manifest:
            raise ValueError("--baseline_manifest is required for external baselines")
        manifest = load_baseline_manifest(args.baseline_manifest)
        cases = list(
            iter_synthetic_cases(
                config_root=args.config_root,
                dataset_root=args.dataset_root,
                seeds=seeds,
                only=only,
            )
        )
        output_root = Path(args.out_jsonl).with_suffix("")
        for model in external_models:
            if model not in manifest:
                raise ValueError(f"model {model!r} not found in baseline manifest")
            spec = manifest[model]
            for case in cases:
                pred_path = output_root / "external_outputs" / model / f"seed_{case.seed}" / f"{case.name}.json"
                adapted_dataset_path = output_root / "adapted_datasets" / f"seed_{case.seed}" / f"{case.name}.pkl"
                write_all_as_train_dataset(case, adapted_dataset_path)
                predicted, metadata = run_external_rule_baseline(
                    spec=spec,
                    config_path=case.config_path,
                    dataset_path=adapted_dataset_path,
                    output_path=pred_path,
                    seed=case.seed,
                    extra_values={"device": args.device},
                )
                metadata["original_dataset_path"] = str(case.dataset_path)
                metadata["adapted_dataset_path"] = str(adapted_dataset_path)
                matched = match_rule_sets(predicted, case.true_rules())
                result = SyntheticRuleRecoveryResult(
                    model=model,
                    dataset=case.name,
                    seed=case.seed,
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
                    runtime_sec=metadata.get("runtime_sec"),
                    model_metadata=metadata,
                )
                rows.append(result.to_dict())

    write_jsonl(rows, args.out_jsonl)
    if args.out_summary_csv:
        write_csv(synthetic_summary(rows), args.out_summary_csv)
    print(json.dumps({"wrote": args.out_jsonl, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

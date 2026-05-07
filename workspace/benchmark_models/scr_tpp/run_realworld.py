from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..core.schema import normalize_rules


ROOT = Path(__file__).resolve().parents[2]
PAPER_RUNNER_DIR = ROOT / "workspace" / "train" / "paper_benchmark_active"


def _install_paths() -> None:
    for path in (str(ROOT), str(ROOT / "workspace" / "train"), str(PAPER_RUNNER_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault(
        "PYTHONPATH",
        f"{ROOT}:{ROOT / 'workspace' / 'train'}:{PAPER_RUNNER_DIR}",
    )


def _load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _format_result(row: dict[str, Any], *, dataset: str, target_label: str, seed: int) -> dict[str, Any]:
    target = int(row["target"])
    predicted = normalize_rules(row.get("predicted", []), default_target=target)
    nll_proxy = None
    bic = row.get("bic")
    runtime_profile = dict(row.get("runtime_profile", {}) or {})
    final_stage_timings = dict(runtime_profile.get("final_stage_timings_sec", {}) or {})
    return {
        "model": "SCR-TPP",
        "dataset": str(dataset),
        "target": str(target_label),
        "target_event_id": int(target),
        "seed": int(seed),
        "nll": nll_proxy,
        "bic": None if bic is None else float(bic),
        "runtime_sec": float(row.get("elapsed_sec", 0.0)),
        "training_time_sec": float(row.get("elapsed_sec", 0.0)),
        "inference_time_sec": None,
        "support_search_time_sec": float(row.get("elapsed_sec", 0.0)),
        "final_kernel_fit_time_sec": float(final_stage_timings.get("final_free_kernel_polish", 0.0)),
        "learned_rules": [rule.to_dict() for rule in predicted],
        "learned_rule_count": len(predicted),
        "mu": row.get("mu"),
        "learned_rule_parameter_details": row.get("learned_rule_parameter_details", []),
        "model_metadata": {
            "implementation_type": "paper_algorithm_realworld_fixed_target_support_discovery",
            "runner_module": "run_paper_benchmarks_real_world",
            "realworld_memory_note": (
                "The real-world runner uses the same SCR-TPP objective and move set as "
                "the paper runner, but computes sieve basis responses in row chunks "
                "and disables GPU-only speed caches that duplicate multi-GB tensors."
            ),
            "algorithm_profile": row.get("algorithm_profile"),
            "final_support_criterion": row.get("final_support_criterion"),
            "runtime_profile": runtime_profile,
            "mu": row.get("mu"),
            "learned_rule_parameter_details": row.get("learned_rule_parameter_details", []),
            "raw_output": row,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the paper SCR-TPP support learner on real-world train-only splits.")
    ap.add_argument("--rule_input_manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seeds", default="111")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cpu_threads", type=int, default=12)
    ap.add_argument("--sieve_batch_size", type=int, default=2)
    ap.add_argument("--sieve_steps", type=int, default=20)
    ap.add_argument("--exact_batch_size", type=int, default=8)
    args = ap.parse_args()

    _install_paths()
    import runtime_resources
    import run_paper_benchmarks_real_world as runner

    runtime_resources.configure_runtime_resources(int(args.cpu_threads))
    runner.FINAL_SIEVE_BATCH_SIZE = max(1, int(args.sieve_batch_size))
    runner.FINAL_SIEVE_STEPS = max(1, int(args.sieve_steps))
    runner.FINAL_EXACT_BATCH_SIZE = max(1, int(args.exact_batch_size))
    manifest = _load_manifest(args.rule_input_manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    rows = []
    for seed in seeds:
        config_path = manifest["config_paths"][str(seed)]
        start = time.perf_counter()
        raw = runner.evaluate_config(
            Path(config_path),
            str(args.device),
            regenerate_dataset=False,
            dataset_seed=None,
        )
        raw["elapsed_sec"] = float(raw.get("elapsed_sec", time.perf_counter() - start))
        row = _format_result(
            raw,
            dataset=str(manifest["dataset"]),
            target_label=str(manifest["target_event_label"]),
            seed=int(seed),
        )
        rows.append(row)
        (out_dir / f"scr_tpp_seed{seed}.json").write_text(json.dumps(row, indent=2))

    jsonl = out_dir / "scr_tpp_results.jsonl"
    with jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"results": str(jsonl), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

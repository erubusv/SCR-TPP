from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _metadata_time(row: dict[str, Any], key: str) -> float | None:
    if row.get(key) is not None:
        return float(row[key])
    metadata = dict(row.get("model_metadata", {}) or {})
    if metadata.get(key) is not None:
        return float(metadata[key])
    target_metrics = metadata.get("target_mark_test_metrics")
    if isinstance(target_metrics, dict) and target_metrics.get(key) is not None:
        return float(target_metrics[key])
    return None


def _rule_text(rule: dict[str, Any]) -> str:
    sources = ",".join(str(src) for src in rule.get("sources", []))
    sign = str(rule.get("sign", "?"))
    target = rule.get("target", "?")
    rel = rule.get("temporal_relations") or []
    rel_text = "" if not rel else f"; rel={rel}"
    return f"({sources}) -> {target} [{sign}{rel_text}]"


def _scr_params(row: dict[str, Any]) -> list[str]:
    details = list(row.get("learned_rule_parameter_details", []) or [])
    if not details:
        details = list((row.get("model_metadata", {}) or {}).get("learned_rule_parameter_details", []) or [])
    out: list[str] = []
    for detail in details:
        sources = ",".join(str(src) for src in detail.get("sources", []))
        beta = detail.get("beta")
        kernels = []
        for kernel in detail.get("kernel_distribution_by_source", []) or []:
            source = kernel.get("source")
            peak = kernel.get("estimated_peak", kernel.get("peak"))
            relation = kernel.get("temporal_relation")
            kernels.append(f"src {source}: peak={_fmt(peak, 3)}, rel={relation}")
        out.append(f"({sources}) beta={_fmt(beta, 4)}; " + "; ".join(kernels))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a paper-table-style real-world benchmark report.")
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--readiness_audit_json")
    args = ap.parse_args()

    rows = _read_jsonl(args.results_jsonl)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model")), str(row.get("dataset")), str(row.get("target")))].append(row)

    lines: list[str] = [
        "# Real-World Benchmark Results",
        "",
        "## Metric Contract",
        "",
        "- All models are trained on the train split only and evaluated on the held-out test split.",
        "- NLL, MAE, and RMSE are target-event metrics: the history keeps all marks, but the event term and next-event-time error are computed only for the fixed target mark.",
        "- Logical TPP rows additionally list the learned rules. SCR-TPP also exports beta and learned kernel summaries when available.",
        "- Real-world data has no ground-truth rules, so no rule-recovery Jaccard is reported.",
        "",
    ]
    if args.readiness_audit_json:
        audit = json.loads(Path(args.readiness_audit_json).read_text())
        lines += [
            "## Readiness Audit",
            "",
            f"- Audit status: `{audit.get('ok')}`",
            f"- Target: `{audit.get('target_event_label')}` (`{audit.get('target_event_id')}`)",
            "",
        ]

    lines += [
        "## Prediction Metrics",
        "",
        "| model | dataset | target | runs | NLL | MAE | RMSE | train sec | inference sec | runtime sec | learned rules |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, dataset, target), group in sorted(grouped.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    dataset,
                    target,
                    str(len(group)),
                    _fmt(_mean([row.get("nll") for row in group])),
                    _fmt(_mean([row.get("time_mae") for row in group])),
                    _fmt(_mean([row.get("time_rmse") for row in group])),
                    _fmt(_mean([_metadata_time(row, "training_time_sec") for row in group]), 1),
                    _fmt(_mean([_metadata_time(row, "inference_time_sec") for row in group]), 1),
                    _fmt(_mean([row.get("runtime_sec") for row in group]), 1),
                    str(sum(len(row.get("learned_rules", []) or []) for row in group)),
                ]
            )
            + " |"
        )

    lines += ["", "## Learned Rules", ""]
    for row in rows:
        rules = list(row.get("learned_rules", []) or [])
        if not rules:
            continue
        lines += [
            f"### {row.get('model')} seed={row.get('seed', '-')}",
            "",
        ]
        for rule in rules:
            lines.append(f"- `{_rule_text(rule)}`")
        if str(row.get("model")) == "SCR-TPP":
            params = _scr_params(row)
            if params:
                lines += ["", "SCR-TPP parameter summary:"]
                for item in params:
                    lines.append(f"- `{item}`")
        lines.append("")

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps({"output": str(out_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

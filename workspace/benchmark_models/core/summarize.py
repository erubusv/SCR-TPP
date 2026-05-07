from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (list, dict))})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def synthetic_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["dataset"]))].append(row)
    out = []
    for (model, dataset), group in sorted(grouped.items()):
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "runs": len(group),
                "exact_count": sum(
                    1 for row in group
                    if float(row["recall"]) == 1.0 and float(row["precision"]) == 1.0
                ),
                "mean_recall": sum(float(row["recall"]) for row in group) / len(group),
                "mean_precision": sum(float(row["precision"]) for row in group) / len(group),
                "mean_f1": sum(float(row["f1"]) for row in group) / len(group),
                "mean_source_jaccard_recall": sum(
                    float(row.get("source_jaccard_recall", 0.0)) for row in group
                ) / len(group),
                "mean_source_jaccard_precision": sum(
                    float(row.get("source_jaccard_precision", 0.0)) for row in group
                ) / len(group),
                "mean_source_jaccard_f1": sum(
                    float(row.get("source_jaccard_f1", 0.0)) for row in group
                ) / len(group),
                "mean_signed_source_jaccard_recall": sum(
                    float(row.get("signed_source_jaccard_recall", 0.0)) for row in group
                ) / len(group),
                "mean_signed_source_jaccard_precision": sum(
                    float(row.get("signed_source_jaccard_precision", 0.0)) for row in group
                ) / len(group),
                "mean_signed_source_jaccard_f1": sum(
                    float(row.get("signed_source_jaccard_f1", 0.0)) for row in group
                ) / len(group),
                "mean_runtime_sec": sum(float(row["runtime_sec"] or 0.0) for row in group) / len(group),
                "total_missing": sum(len(row.get("missing", [])) for row in group),
                "total_extra": sum(len(row.get("extra", [])) for row in group),
            }
        )
    return out


def _mean_optional(group: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in group if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def realworld_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["dataset"]), str(row["target"]))].append(row)
    out = []
    for (model, dataset, target), group in sorted(grouped.items()):
        out.append(
            {
                "model": model,
                "dataset": dataset,
                "target": target,
                "runs": len(group),
                "mean_nll": _mean_optional(group, "nll"),
                "mean_time_mae": _mean_optional(group, "time_mae"),
                "mean_time_rmse": _mean_optional(group, "time_rmse"),
                "mean_type_acc": _mean_optional(group, "type_acc"),
                "mean_runtime_sec": _mean_optional(group, "runtime_sec"),
                "learned_rule_count": sum(len(row.get("learned_rules", [])) for row in group),
            }
        )
    return out

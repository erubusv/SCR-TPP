from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _label(node: int, target: int) -> str:
    if int(node) == int(target):
        return "T"
    if 0 <= int(node) < 26:
        return chr(ord("A") + int(node))
    return f"S{int(node)}"


def _rule_text(rule: dict[str, Any]) -> str:
    target = int(rule["target"])
    lhs = " and ".join(_label(int(src), target) for src in rule.get("sources", []))
    sign = str(rule.get("sign", "excitation"))
    rels = rule.get("temporal_relations", []) or []
    rel_txt = f" [{','.join(str(rel) for rel in rels)}]" if rels else ""
    return f"{lhs} -> T : {sign}{rel_txt}"


def _overall(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    out = []
    for model, group in sorted(grouped.items()):
        out.append(
            {
                "model": model,
                "runs": len(group),
                "exact": f"{sum(1 for row in group if float(row['recall']) == 1.0 and float(row['precision']) == 1.0)}/{len(group)}",
                "mean_recall": _mean([float(row["recall"]) for row in group]),
                "mean_precision": _mean([float(row["precision"]) for row in group]),
                "mean_f1": _mean([float(row["f1"]) for row in group]),
                "mean_source_jaccard_f1": _mean([float(row.get("source_jaccard_f1", 0.0)) for row in group]),
                "mean_signed_source_jaccard_f1": _mean(
                    [float(row.get("signed_source_jaccard_f1", 0.0)) for row in group]
                ),
                "total_missing": sum(len(row.get("missing", [])) for row in group),
                "total_extra": sum(len(row.get("extra", [])) for row in group),
                "mean_runtime_sec": _mean([float(row.get("runtime_sec") or 0.0) for row in group]),
                "total_runtime_sec": sum(float(row.get("runtime_sec") or 0.0) for row in group),
            }
        )
    return out


def _per_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "exact": f"{sum(1 for row in group if float(row['recall']) == 1.0 and float(row['precision']) == 1.0)}/{len(group)}",
                "mean_recall": _mean([float(row["recall"]) for row in group]),
                "mean_precision": _mean([float(row["precision"]) for row in group]),
                "mean_f1": _mean([float(row["f1"]) for row in group]),
                "mean_source_jaccard_f1": _mean([float(row.get("source_jaccard_f1", 0.0)) for row in group]),
                "mean_signed_source_jaccard_f1": _mean(
                    [float(row.get("signed_source_jaccard_f1", 0.0)) for row in group]
                ),
                "total_missing": sum(len(row.get("missing", [])) for row in group),
                "total_extra": sum(len(row.get("extra", [])) for row in group),
                "mean_runtime_sec": _mean([float(row.get("runtime_sec") or 0.0) for row in group]),
            }
        )
    return out


def _learned_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda item: (str(item["model"]), str(item["dataset"]), int(item["seed"]))):
        rules = "; ".join(_rule_text(rule) for rule in row.get("predicted_rules", [])) or "none"
        out.append(
            {
                "model": str(row["model"]),
                "dataset": str(row["dataset"]),
                "seed": int(row["seed"]),
                "learned_rules": rules,
            }
        )
    return out


def _relation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    out = []
    for model, group in sorted(grouped.items()):
        counts: Counter[str] = Counter()
        rule_count = 0
        for row in group:
            for rule in row.get("predicted_rules", []):
                rels = rule.get("temporal_relations", []) or []
                if rels:
                    rule_count += 1
                    counts.update(str(rel) for rel in rels)
        out.append(
            {
                "model": model,
                "predicted_rules_with_relation_labels": rule_count,
                "relation_label_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
            }
        )
    return out


def _markdown(
    *,
    rows: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    per_dataset: list[dict[str, Any]],
    relation: list[dict[str, Any]],
    learned_rules_path: Path,
    teller_root: Path,
) -> str:
    lines = [
        "# Synthetic Full Comparison: All Available Models",
        "",
        f"TELLER result root: `{teller_root}`",
        "",
        "Synthetic rule recovery uses all generated sequences as train data. Exact match is `(source set, sign, target)`. Source Jaccard is a secondary near-miss metric.",
        "",
        "## Overall Support Metrics",
        "",
        "| model | runs | exact | recall | precision | F1 | source Jaccard F1 | signed Jaccard F1 | missing | extra | mean runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['model']} | {row['runs']} | {row['exact']} | {row['mean_recall']:.3f} | "
            f"{row['mean_precision']:.3f} | {row['mean_f1']:.3f} | "
            f"{row['mean_source_jaccard_f1']:.3f} | {row['mean_signed_source_jaccard_f1']:.3f} | "
            f"{row['total_missing']} | {row['total_extra']} | {row['mean_runtime_sec']:.1f} |"
        )
    lines += [
        "",
        "## Temporal Relation Label Availability",
        "",
        "| model | predicted rules with relation labels | label counts |",
        "|---|---:|---|",
    ]
    for row in relation:
        lines.append(
            f"| {row['model']} | {row['predicted_rules_with_relation_labels']} | "
            f"`{row['relation_label_counts']}` |"
        )
    lines += [
        "",
        "## Per-Dataset Seed Means",
        "",
        "| model | dataset | exact | recall | precision | F1 | source Jaccard F1 | signed Jaccard F1 | missing | extra | runtime s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_dataset:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['exact']} | {row['mean_recall']:.3f} | "
            f"{row['mean_precision']:.3f} | {row['mean_f1']:.3f} | "
            f"{row['mean_source_jaccard_f1']:.3f} | {row['mean_signed_source_jaccard_f1']:.3f} | "
            f"{row['total_missing']} | {row['total_extra']} | {row['mean_runtime_sec']:.1f} |"
        )
    lines += [
        "",
        "## Learned Rules",
        "",
        f"Full learned-rule table is stored at `{learned_rules_path}`.",
        "",
        "The table is kept separate because it contains one row per model/dataset/seed and is large.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate synthetic benchmark result tables.")
    parser.add_argument("--ours_jsonl", default="data/paper_suite/results/benchmark_synthetic_ours_jaccard.jsonl")
    parser.add_argument("--logic_root", default="data/paper_suite/results/benchmark_full33_best_20260504_173529")
    parser.add_argument("--teller_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--doc_path", required=True)
    args = parser.parse_args()

    teller_root = Path(args.teller_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows.extend(_load_jsonl(args.ours_jsonl))
    logic_root = Path(args.logic_root)
    for name in ("clnn", "nstpp", "cluster"):
        rows.extend(_load_jsonl(logic_root / f"{name}.jsonl"))
    for seed in (111, 222, 333):
        rows.extend(_load_jsonl(teller_root / f"seed{seed}.jsonl"))

    if not rows:
        raise RuntimeError("no synthetic result rows found")
    overall = _overall(rows)
    per_dataset = _per_dataset(rows)
    learned = _learned_rules(rows)
    relation = _relation_summary(rows)

    _write_csv(out_dir / "all_models_overall.csv", overall)
    _write_csv(out_dir / "all_models_per_dataset.csv", per_dataset)
    _write_csv(out_dir / "all_models_learned_rules.csv", learned)
    _write_csv(out_dir / "all_models_relation_label_availability.csv", relation)
    (out_dir / "all_models.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    learned_path = out_dir / "all_models_learned_rules.csv"
    doc = _markdown(
        rows=rows,
        overall=overall,
        per_dataset=per_dataset,
        relation=relation,
        learned_rules_path=learned_path,
        teller_root=teller_root,
    )
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text(doc)
    print(json.dumps({"rows": len(rows), "doc_path": str(args.doc_path), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

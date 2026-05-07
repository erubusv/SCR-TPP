from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.schema import NormalizedRule, true_rules_from_config
from ..core.summarize import load_jsonl, synthetic_summary


MODEL_ORDER = {"ours": 0, "CLNN": 1, "NSTPP": 2, "CLUSTER": 3}


def _label(index: int) -> str:
    index = int(index)
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return f"S{index}"


def _try_raw_json(rule: dict[str, Any]) -> dict[str, Any]:
    raw = rule.get("raw")
    if not isinstance(raw, str):
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _rule_text(rule: dict[str, Any] | NormalizedRule) -> str:
    if isinstance(rule, NormalizedRule):
        data = rule.to_dict()
        raw = _try_raw_json(data)
    else:
        data = rule
        raw = _try_raw_json(rule)
    sources = [int(src) for src in data.get("sources", [])]
    target = int(data["target"])
    lhs = " ∧ ".join(_label(src) for src in sources) or "∅"
    sign = str(data.get("sign", "excitation"))
    sign_short = "exc" if sign == "excitation" else "inh"
    suffix = ""
    if "beta" in raw:
        suffix = f" beta={float(raw['beta']):.3f}"
    elif "score" in raw:
        suffix = f" score={float(raw['score']):.3f}"
    elif "W_pos" in raw or "W_neg" in raw:
        mag = float(raw.get("W_pos", 0.0) or raw.get("W_neg", 0.0))
        suffix = f" ({mag:.1f})"
    rels = data.get("temporal_relations", []) or []
    rel_text = ""
    if rels:
        rel_text = " [" + "; ".join(str(rel) for rel in rels[:3])
        if len(rels) > 3:
            rel_text += f"; +{len(rels) - 3}"
        rel_text += "]"
    return f"{lhs} -> {_label(target)} : {sign_short}{suffix}{rel_text}"


def _rule_list(rules: list[dict[str, Any]]) -> str:
    if not rules:
        return "none"
    return "<br>".join(_rule_text(rule) for rule in rules)


def _truth_text(config_path: Path) -> str:
    rules = true_rules_from_config(config_path)
    return "; ".join(_rule_text(rule) for rule in rules)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _model_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    out = []
    for model, group in sorted(grouped.items(), key=lambda item: (MODEL_ORDER.get(item[0], 99), item[0])):
        out.append(
            {
                "model": model,
                "runs": len(group),
                "exact": f"{sum(1 for row in group if float(row['recall']) == 1.0 and float(row['precision']) == 1.0)}/{len(group)}",
                "recall": _mean([float(row["recall"]) for row in group]),
                "precision": _mean([float(row["precision"]) for row in group]),
                "f1": _mean([float(row["f1"]) for row in group]),
                "jaccard_f1": _mean([float(row.get("source_jaccard_f1", 0.0)) for row in group]),
                "signed_jaccard_f1": _mean([float(row.get("signed_source_jaccard_f1", 0.0)) for row in group]),
                "missing": sum(len(row.get("missing", [])) for row in group),
                "extra": sum(len(row.get("extra", [])) for row in group),
                "runtime_sum": sum(float(row.get("runtime_sec") or 0.0) for row in group),
            }
        )
    return out


def _dataset_sections(rows: list[dict[str, Any]], config_root: Path) -> list[str]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    summary_by_key = {(row["model"], row["dataset"]): row for row in synthetic_summary(rows)}
    sections: list[str] = []
    for dataset in sorted(by_dataset):
        cfg = config_root / f"{dataset}.yaml"
        sections.extend([f"## {dataset}", "", f"Truth rules: {_truth_text(cfg)}", ""])
        sections.extend(
            [
                "| model | exact | recall | precision | F1 | Jaccard F1 | signed Jaccard F1 | missing | extra |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        models = sorted({str(row["model"]) for row in by_dataset[dataset]}, key=lambda item: (MODEL_ORDER.get(item, 99), item))
        for model in models:
            s = summary_by_key[(model, dataset)]
            sections.append(
                f"| {model} | {int(s['exact_count'])}/{int(s['runs'])} | {float(s['mean_recall']):.3f} | "
                f"{float(s['mean_precision']):.3f} | {float(s['mean_f1']):.3f} | "
                f"{float(s['mean_source_jaccard_f1']):.3f} | {float(s['mean_signed_source_jaccard_f1']):.3f} | "
                f"{int(s['total_missing'])} | {int(s['total_extra'])} |"
            )
        sections.append("")
        for model in models:
            sections.extend([f"<details>", f"<summary>{model} learned rules by seed</summary>", ""])
            sections.extend(["| seed | recall | precision | learned rules | missing | extra |", "|---:|---:|---:|---|---|---|"])
            for row in sorted((r for r in by_dataset[dataset] if str(r["model"]) == model), key=lambda item: int(item["seed"])):
                sections.append(
                    f"| {int(row['seed'])} | {float(row['recall']):.3f} | {float(row['precision']):.3f} | "
                    f"{_rule_list(row.get('predicted_rules', []))} | "
                    f"{_rule_list(row.get('missing', []))} | {_rule_list(row.get('extra', []))} |"
                )
            sections.extend(["", "</details>", ""])
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic recovery markdown report with learned rules.")
    parser.add_argument("--rows_jsonl", required=True)
    parser.add_argument("--config_root", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.rows_jsonl)
    config_root = Path(args.config_root)
    lines = [
        "# Synthetic Rule Recovery Benchmark: Corrected Baselines",
        "",
        "Date: 2026-05-05",
        "",
        f"Result root: `{args.result_root}`",
        f"Config root: `{config_root}`",
        f"Frozen manifest: `{args.manifest}`",
        "",
        "Synthetic rule discovery uses all generated sequences as train data. Exact match is `(source set, sign, target)`. Source-set Jaccard is a secondary near-miss metric and is computed by one-to-one Hungarian maximum-weight matching.",
        "",
        "This is the current corrected synthetic benchmark table. It includes our model and the corrected CLNN, NSTPP, and CLUSTER runs. TELLER is not included here because the previous TELLER synthetic run used a stale benchmark path and is treated as invalid until rerun with the current official-code adapter.",
        "",
        "## Model-Level Summary",
        "",
        "| model | runs | exact | recall | precision | F1 | source-Jaccard F1 | signed source-Jaccard F1 | missing | extra | runtime sum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _model_summary(rows):
        lines.append(
            f"| {row['model']} | {row['runs']} | {row['exact']} | {row['recall']:.3f} | {row['precision']:.3f} | "
            f"{row['f1']:.3f} | {row['jaccard_f1']:.3f} | {row['signed_jaccard_f1']:.3f} | "
            f"{row['missing']} | {row['extra']} | {row['runtime_sum']:.1f}s |"
        )
    lines.extend(["", "## Dataset-Level Results And Learned Rules", ""])
    lines.extend(_dataset_sections(rows, config_root))
    output = Path(args.out_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(json.dumps({"wrote": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

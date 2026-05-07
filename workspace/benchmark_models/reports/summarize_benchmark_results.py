from __future__ import annotations

import argparse
import json

from ..core.summarize import load_jsonl, realworld_summary, synthetic_summary, write_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize benchmark result JSONL files.")
    ap.add_argument("--synthetic_jsonl", default="")
    ap.add_argument("--synthetic_summary_csv", default="")
    ap.add_argument("--realworld_jsonl", default="")
    ap.add_argument("--realworld_summary_csv", default="")
    args = ap.parse_args()

    out = {}
    if args.synthetic_jsonl:
        rows = load_jsonl(args.synthetic_jsonl)
        summary = synthetic_summary(rows)
        out["synthetic"] = summary
        if args.synthetic_summary_csv:
            write_csv(summary, args.synthetic_summary_csv)
    if args.realworld_jsonl:
        rows = load_jsonl(args.realworld_jsonl)
        summary = realworld_summary(rows)
        out["realworld"] = summary
        if args.realworld_summary_csv:
            write_csv(summary, args.realworld_summary_csv)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

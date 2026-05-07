from __future__ import annotations

import argparse
import gzip
import json
import pickle
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.easytpp import write_gatech_pickles


DEFAULT_TARGET_ACTIVITY = "O_Accepted"
DEFAULT_SOURCE_ACTIVITIES = [
    "O_Create Offer",
    "O_Created",
    "O_Sent (mail and online)",
    "A_Validating",
    "A_Complete",
    "O_Returned",
    "A_Incomplete",
    "A_Submitted",
    "W_Complete application",
    "W_Validate application",
    "W_Handle leads",
    "W_Call incomplete files",
    "O_Sent (online only)",
    "W_Call after offers",
    "W_Assess potential fraud",
]


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _parse_time(value: str) -> float:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def _iter_cases(path: Path):
    in_event = False
    ev: dict[str, str] = {}
    current: list[dict[str, str]] = []
    trace_attrs: dict[str, str] = {}
    with gzip.open(path, "rb") as fh:
        for event, elem in ET.iterparse(fh, events=("start", "end")):
            tag = _strip_ns(elem.tag)
            if event == "start":
                if tag == "trace":
                    current = []
                    trace_attrs = {}
                elif tag == "event":
                    in_event = True
                    ev = {}
            else:
                if tag in {"string", "date", "int", "float", "boolean"}:
                    key = elem.attrib.get("key")
                    value = elem.attrib.get("value")
                    if key is not None and value is not None:
                        if in_event:
                            ev[key] = value
                        else:
                            trace_attrs[key] = value
                elif tag == "event":
                    in_event = False
                    current.append(ev)
                elif tag == "trace":
                    yield trace_attrs, current
                elem.clear()


def _percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    idx = int(round((len(arr) - 1) * float(q)))
    idx = max(0, min(idx, len(arr) - 1))
    return float(arr[idx])


def _split_summary(name: str, sequences: list[dict[str, Any]], *, target: int, id_to_event: dict[int, str]) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    lengths: list[int] = []
    target_counts: list[int] = []
    for seq in sequences:
        times = [float(t) for t in seq["time"]]
        events = [int(e) for e in seq["event"]]
        if len(times) != len(events):
            raise ValueError(f"{seq.get('sequence_id')} has mismatched time/event length")
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(f"{seq.get('sequence_id')} has decreasing event times")
        counts.update(events)
        lengths.append(len(events))
        target_counts.append(sum(1 for event in events if event == int(target)))
    return {
        "split": str(name),
        "sequences": len(sequences),
        "events": int(sum(counts.values())),
        "target_events": int(counts[int(target)]),
        "non_target_events": int(sum(v for k, v in counts.items() if int(k) != int(target))),
        "target_positive_sequences": int(sum(1 for count in target_counts if count > 0)),
        "event_counts": {id_to_event[int(k)]: int(v) for k, v in sorted(counts.items())},
        "event_counts_by_id": {str(int(k)): int(v) for k, v in sorted(counts.items())},
        "length_mean": float(statistics.mean(lengths)) if lengths else 0.0,
        "length_p50": _percentile(lengths, 0.50),
        "length_p95": _percentile(lengths, 0.95),
        "length_max": int(max(lengths) if lengths else 0),
    }


def _write_pickle(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare BPI 2017 target-event dataset for SCR-TPP.")
    ap.add_argument("--raw_xes_gz", default="data/realworld_raw/bpi_2017/BPI_Challenge_2017.xes.gz")
    ap.add_argument("--output_dir", default="data/realworld_prepared/bpi2017_o_accepted_random_5000")
    ap.add_argument("--target_activity", default=DEFAULT_TARGET_ACTIVITY)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--train_size", type=int, default=5000)
    ap.add_argument("--val_size", type=int, default=1000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument(
        "--target_positive_only",
        action="store_true",
        help="If set, sample only cases containing O_Accepted. Default keeps accepted and non-accepted cases.",
    )
    args = ap.parse_args()

    raw_path = Path(args.raw_xes_gz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_activity = str(args.target_activity)
    source_activities = list(DEFAULT_SOURCE_ACTIVITIES)
    if target_activity in source_activities:
        source_activities = [activity for activity in source_activities if activity != target_activity]
    event_to_id = {target_activity: 0}
    for idx, activity in enumerate(source_activities, start=1):
        event_to_id[activity] = int(idx)
    id_to_event = {int(v): str(k) for k, v in event_to_id.items()}
    target = 0
    selected_activities = set(event_to_id)

    started = time.perf_counter()
    sequences: list[dict[str, Any]] = []
    raw_cases = 0
    raw_events = 0
    raw_activity_counts: Counter[str] = Counter()
    target_positive_cases = 0
    source_positive_cases = 0
    for case_idx, (trace_attrs, events) in enumerate(_iter_cases(raw_path)):
        raw_cases += 1
        rows: list[tuple[float, int, str]] = []
        source_positive = False
        target_positive = False
        for raw_idx, ev in enumerate(events):
            if ev.get("lifecycle:transition") != "complete":
                continue
            activity = ev.get("concept:name")
            if not activity:
                continue
            raw_activity_counts[activity] += 1
            raw_events += 1
            if activity not in selected_activities:
                continue
            timestamp = ev.get("time:timestamp")
            if not timestamp:
                continue
            if activity == target_activity:
                target_positive = True
            else:
                source_positive = True
            rows.append((_parse_time(timestamp), raw_idx, activity))
        if target_positive:
            target_positive_cases += 1
        if source_positive:
            source_positive_cases += 1
        if args.target_positive_only and not target_positive:
            continue
        if len(rows) < 2:
            continue
        rows.sort(key=lambda x: (x[0], x[1]))
        first_time = rows[0][0]
        sequence_id = str(trace_attrs.get("concept:name", f"case_{case_idx}"))
        sequences.append(
            {
                "time": [(time_value - first_time) / 86400.0 for time_value, _idx, _activity in rows],
                "event": [event_to_id[activity] for _time, _idx, activity in rows],
                "sequence_id": sequence_id,
                "source_case_id": sequence_id,
            }
        )

    total_requested = int(args.train_size) + int(args.val_size) + int(args.test_size)
    if len(sequences) < total_requested:
        raise ValueError(f"requested {total_requested} sequences, found only {len(sequences)}")
    rng = np.random.default_rng(int(args.seed))
    idx = np.arange(len(sequences), dtype=np.int64)
    rng.shuffle(idx)
    selected = [sequences[int(i)] for i in idx[:total_requested]]
    train = selected[: int(args.train_size)]
    val = selected[int(args.train_size) : int(args.train_size) + int(args.val_size)]
    test = selected[int(args.train_size) + int(args.val_size) :]

    metadata: dict[str, Any] = {
        "dataset": f"bpi2017_{target_activity.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}_random_{int(args.train_size)}",
        "source_dataset": "BPI Challenge 2017",
        "source_url": "https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884",
        "doi": "10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b",
        "raw_xes_gz": str(raw_path),
        "lifecycle_filter": "complete",
        "dim_process": int(len(event_to_id)),
        "num_types": int(len(event_to_id)),
        "event_to_id": event_to_id,
        "id_to_event": {str(k): v for k, v in id_to_event.items()},
        "target_event_id": int(target),
        "target_event_label": target_activity,
        "source_event_ids": list(range(1, len(event_to_id))),
        "source_event_labels": {str(i): id_to_event[i] for i in range(1, len(event_to_id))},
        "time_unit": "days_since_case_first_selected_event",
        "split_seed": int(args.seed),
        "split_strategy": (
            "target_positive_case_level_uniform_random_fixed_size"
            if args.target_positive_only
            else "case_level_uniform_random_fixed_size"
        ),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
    }
    sequence_pickle = output_dir / "sequences.pkl"
    split_pickle = output_dir / f"split_seed{int(args.seed)}.pkl"
    payload = {"sequences": selected, "train": train, "val": val, "test": test, "metadata": metadata}
    _write_pickle(sequence_pickle, payload)
    _write_pickle(split_pickle, {"train": train, "val": val, "test": test, "metadata": metadata})
    easytpp_paths = write_gatech_pickles(
        train=train,
        dev=val,
        test=test,
        dim_process=int(len(event_to_id)),
        output_dir=output_dir / "easytpp",
    )
    split_summary = {
        "train": _split_summary("train", train, target=target, id_to_event=id_to_event),
        "val": _split_summary("val", val, target=target, id_to_event=id_to_event),
        "test": _split_summary("test", test, target=target, id_to_event=id_to_event),
    }
    checks = [
        {
            "name": "target_event_present_in_all_splits",
            "ok": all(split_summary[name]["target_events"] > 0 for name in ("train", "val", "test")),
            "details": {name: split_summary[name]["target_events"] for name in ("train", "val", "test")},
        },
        {
            "name": "source_predicates_exclude_target",
            "ok": 0 not in metadata["source_event_ids"],
            "details": {"target_event_id": 0, "source_event_ids": metadata["source_event_ids"]},
        },
        {
            "name": "target_positive_and_negative_cases_available",
            "ok": all(
                0 < split_summary[name]["target_positive_sequences"] < split_summary[name]["sequences"]
                for name in ("train", "val", "test")
            ),
            "details": {
                name: {
                    "target_positive_sequences": split_summary[name]["target_positive_sequences"],
                    "sequences": split_summary[name]["sequences"],
                }
                for name in ("train", "val", "test")
            },
        },
    ]
    audit = {
        "prepared_dir": str(output_dir),
        "target_event_id": 0,
        "target_event_label": target_activity,
        "id_to_event": metadata["id_to_event"],
        "source_event_labels": metadata["source_event_labels"],
        "split_summary": split_summary,
        "raw_audit": {
            "raw_cases": int(raw_cases),
            "raw_complete_events": int(raw_events),
            "target_positive_cases": int(target_positive_cases),
            "source_positive_cases": int(source_positive_cases),
            "raw_complete_activity_counts": dict(raw_activity_counts.most_common()),
            "preparation_sec": float(time.perf_counter() - started),
        },
        "checks": checks,
        "ok": all(bool(row["ok"]) for row in checks),
    }
    (output_dir / "readiness_audit.json").write_text(json.dumps(audit, indent=2))
    manifest = {
        "dataset": metadata["dataset"],
        "source_dataset": metadata["source_dataset"],
        "source_url": metadata["source_url"],
        "doi": metadata["doi"],
        "sequence_pickle": str(sequence_pickle),
        "split_paths": {str(args.seed): str(split_pickle)},
        "easytpp_paths": easytpp_paths,
        "target_event_id": 0,
        "target_event_label": target_activity,
        "dim_process": int(len(event_to_id)),
        "seeds": [int(args.seed)],
        "split_summary": split_summary,
        "readiness_audit": str(output_dir / "readiness_audit.json"),
        "raw_audit": audit["raw_audit"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = [
        "# BPI 2017 O_Accepted Prepared Dataset",
        "",
        f"- Target: `{target_activity}` id `0`.",
        "- Lifecycle filter: `complete` only.",
        f"- Split: `{metadata['split_strategy']}`.",
        "",
        "## Source Events",
        "| id | activity |",
        "|---:|---|",
    ]
    for i in range(1, len(event_to_id)):
        lines.append(f"| {i} | `{id_to_event[i]}` |")
    lines.extend(
        [
            "",
            "## Split Summary",
            "| split | sequences | events | target events | mean length | p95 length |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "val", "test"):
        row = split_summary[split]
        lines.append(
            f"| {split} | {row['sequences']} | {row['events']} | {row['target_events']} | "
            f"{row['length_mean']:.2f} | {row['length_p95']:.0f} |"
        )
    (output_dir / "bpi2017_o_accepted_dataset_info.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

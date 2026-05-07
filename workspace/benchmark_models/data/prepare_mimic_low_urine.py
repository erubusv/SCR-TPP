from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..adapters.easytpp import write_gatech_pickles


LOW_URINE_LABEL = "low_urine_output"
LOW_URINE_WINDOW_HOURS = 6.0
LOW_URINE_RATE_THRESHOLD_ML_PER_KG_H = 0.5

URINE_OUTPUT_ITEMIDS = {
    226559: "Foley",
    226560: "Void",
    226561: "Condom Cath",
    226563: "Suprapubic",
    226564: "R Nephrostomy",
    226565: "L Nephrostomy",
    226567: "Straight Cath",
    226584: "Ileoconduit",
    226557: "R Ureteral Stent",
    226558: "L Ureteral Stent",
    226627: "OR Urine",
    226631: "PACU Urine",
}

WEIGHT_ITEMIDS_KG = {
    226512: "Admission Weight (Kg)",
    224639: "Daily Weight",
}
WEIGHT_ITEMIDS_LB = {
    226531: "Admission Weight (lbs.)",
}

LOW_SYSBP_ITEMIDS = {
    220050: "Arterial Blood Pressure systolic",
    220179: "Non Invasive Blood Pressure systolic",
    225309: "ART BP Systolic",
}
LOW_MAP_ITEMIDS = {
    220052: "Arterial Blood Pressure mean",
    220181: "Non Invasive Blood Pressure mean",
    225312: "ART BP Mean",
}
HIGH_HR_ITEMIDS = {220045: "Heart Rate"}
HIGH_RR_ITEMIDS = {220210: "Respiratory Rate", 224690: "Respiratory Rate (Total)"}
LOW_SPO2_ITEMIDS = {220277: "O2 saturation pulseoxymetry"}
HIGH_FIO2_ITEMIDS = {223835: "Inspired O2 Fraction"}
FEVER_ITEMIDS = {223762: "Temperature Celsius"}

VASOPRESSOR_ITEMIDS = {
    221906: "Norepinephrine",
    221749: "Phenylephrine",
    229630: "Phenylephrine (50/250)",
    229631: "Phenylephrine (200/250)_OLD_1",
    229632: "Phenylephrine (200/250)",
    222315: "Vasopressin",
    221289: "Epinephrine",
    229617: "Epinephrine.",
    221662: "Dopamine",
    221653: "Dobutamine",
    221986: "Milrinone",
}
CRYSTALLOID_ITEMIDS = {
    225158: "NaCl 0.9%",
    220954: "Saline 0,9%",
    220953: "Ringers",
    220955: "Ringers Lactate",
    220956: "Ringers Acetate",
    226364: "OR Crystalloid Intake",
    226375: "PACU Crystalloid Intake",
}
ALBUMIN_ITEMIDS = {
    220862: "Albumin 25%",
    220864: "Albumin 5%",
    220861: "Albumin 20%",
    220863: "Albumin 4%",
}
RRT_ITEMIDS = {
    225441: "Hemodialysis",
    225802: "Dialysis - CRRT",
    225803: "Dialysis - CVVHD",
    225809: "Dialysis - CVVHDF",
    225955: "Dialysis - SCUF",
    225805: "Peritoneal Dialysis",
    225436: "CRRT Filter Change",
}

SOURCE_LABELS = [
    "low_systolic_bp",
    "low_map",
    "high_heart_rate",
    "high_respiratory_rate",
    "low_spo2",
    "high_fio2",
    "fever",
    "vasopressor",
    "crystalloid_fluid",
    "albumin_colloid",
    "rrt_started",
]


@dataclass(frozen=True)
class StayInfo:
    subject_id: int
    hadm_id: int
    stay_id: int
    intime: pd.Timestamp
    outtime: pd.Timestamp
    los_hours: float


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_stays(path: Path, min_los_hours: float, max_los_hours: float) -> tuple[pd.DataFrame, dict[int, StayInfo]]:
    stays = pd.read_csv(path)
    required = {"subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"}
    missing = required - set(stays.columns)
    if missing:
        raise ValueError(f"icustays missing columns: {sorted(missing)}")
    stays["intime"] = pd.to_datetime(stays["intime"])
    stays["outtime"] = pd.to_datetime(stays["outtime"])
    stays["los_hours"] = pd.to_numeric(stays["los"], errors="coerce") * 24.0
    stays = stays[
        stays["stay_id"].notna()
        & stays["intime"].notna()
        & stays["outtime"].notna()
        & stays["los_hours"].between(float(min_los_hours), float(max_los_hours), inclusive="both")
    ].copy()
    info = {
        int(row.stay_id): StayInfo(
            subject_id=int(row.subject_id),
            hadm_id=int(row.hadm_id),
            stay_id=int(row.stay_id),
            intime=row.intime,
            outtime=row.outtime,
            los_hours=float(row.los_hours),
        )
        for row in stays.itertuples(index=False)
    }
    return stays, info


def _append_events(path: Path, rows: Iterable[dict[str, Any]], *, header: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence_id", "time", "event_type", "source_table", "itemid"]
    count = 0
    mode = "w" if header else "a"
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _relative_hours(chunk: pd.DataFrame, stay_intime: dict[int, pd.Timestamp], time_col: str) -> pd.Series:
    intime = chunk["stay_id"].map(stay_intime)
    event_time = pd.to_datetime(chunk[time_col], errors="coerce")
    return (event_time - intime).dt.total_seconds() / 3600.0


def _relative_one(value: Any, intime: pd.Timestamp | None) -> float | None:
    if intime is None:
        return None
    event_time = pd.to_datetime(value, errors="coerce")
    if pd.isna(event_time):
        return None
    delta = event_time - intime
    return float(delta.total_seconds() / 3600.0)


def _valid_weight_kg(value: Any, *, source_unit: str = "kg") -> float | None:
    weight = _safe_float(value)
    if weight is None:
        return None
    if source_unit == "lb":
        weight *= 0.45359237
    if 20.0 <= weight <= 300.0:
        return float(weight)
    return None


def _valid_weight_series(values: pd.Series) -> pd.Series:
    weights = pd.to_numeric(values, errors="coerce")
    return weights.where(weights.between(20.0, 300.0))


def _status_is_usable(value: Any) -> bool:
    text = str(value or "").lower()
    bad = ("cancel", "rewritten", "not given", "stopped")
    return not any(token in text for token in bad)


def _write_dataset_info(path: Path, audit: dict[str, Any] | None = None) -> None:
    lines = [
        "# MIMIC-IV LowUrine Event Construction",
        "",
        "## Target",
        "",
        (
            "`low_urine_output` is an episode-onset target derived from ICU "
            "`outputevents`. For stay `i` with body weight `w_i`, urine records "
            "`(t_j, v_j)` are evaluated at urine chart times after the first 6 hours."
        ),
        "",
        "```text",
        "UO_rate_i(t) = sum(v_j for t - 6h < t_j <= t) / (6 * w_i)",
        "low_urine_output occurs when UO_rate_i(t) < 0.5 mL/kg/h",
        "```",
        "",
        "Only transitions from non-low to low are emitted as target events. Events after the first RRT/dialysis time are censored.",
        "",
        "## Urine Output Itemids",
        "",
    ]
    for itemid, label in sorted(URINE_OUTPUT_ITEMIDS.items()):
        lines.append(f"- `{itemid}`: {label}")
    lines += [
        "",
        "Irrigant-mixed output items are excluded from the primary target definition.",
        "",
        "## Weight",
        "",
        "- Primary: `226512` Admission Weight (Kg)",
        "- Secondary: `224639` Daily Weight",
        "- Additional fallback: `inputevents.patientweight` / `procedureevents.patientweight` when available",
        "- Valid range: 20-300 kg",
        "",
        "## Source Predicates",
        "",
        "| predicate | definition | itemids |",
        "|---|---|---|",
        f"| `low_systolic_bp` | numeric value < 90 mmHg | `{sorted(LOW_SYSBP_ITEMIDS)}` |",
        f"| `low_map` | numeric value < 65 mmHg | `{sorted(LOW_MAP_ITEMIDS)}` |",
        f"| `high_heart_rate` | numeric value > 100 bpm | `{sorted(HIGH_HR_ITEMIDS)}` |",
        f"| `high_respiratory_rate` | numeric value > 22 /min | `{sorted(HIGH_RR_ITEMIDS)}` |",
        f"| `low_spo2` | numeric value < 92% | `{sorted(LOW_SPO2_ITEMIDS)}` |",
        f"| `high_fio2` | FiO2 > 0.5 after normalizing percent values to fractions | `{sorted(HIGH_FIO2_ITEMIDS)}` |",
        f"| `fever` | numeric value >= 38.3 Celsius | `{sorted(FEVER_ITEMIDS)}` |",
        f"| `vasopressor` | selected vasoactive medication input start | `{sorted(VASOPRESSOR_ITEMIDS)}` |",
        f"| `crystalloid_fluid` | selected crystalloid fluid input start | `{sorted(CRYSTALLOID_ITEMIDS)}` |",
        f"| `albumin_colloid` | selected albumin/colloid input start | `{sorted(ALBUMIN_ITEMIDS)}` |",
        f"| `rrt_started` | selected dialysis/RRT procedure start; also used as censor time | `{sorted(RRT_ITEMIDS)}` |",
        "",
        "## Notes",
        "",
        "- ICU stay is the independent sequence unit.",
        "- Time is measured in hours from ICU `intime`.",
        "- Event log is marked by discrete predicates and is used for SCR-TPP and logical rule baselines.",
        "- EasyTPP inputs are generated from the same sequence split.",
    ]
    if audit is not None:
        lines += [
            "",
            "## Generated Audit",
            "",
            "```json",
            json.dumps(audit, indent=2),
            "```",
        ]
    path.write_text("\n".join(lines) + "\n")


def _process_chartevents(
    path: Path,
    source_events_path: Path,
    stay_info: dict[int, StayInfo],
    chunksize: int,
    max_hours: float,
) -> tuple[dict[int, list[tuple[float, float, str]]], Counter[str], dict[str, int]]:
    stay_intime = {sid: info.intime for sid, info in stay_info.items()}
    selected_ids = set().union(
        WEIGHT_ITEMIDS_KG,
        WEIGHT_ITEMIDS_LB,
        LOW_SYSBP_ITEMIDS,
        LOW_MAP_ITEMIDS,
        HIGH_HR_ITEMIDS,
        HIGH_RR_ITEMIDS,
        LOW_SPO2_ITEMIDS,
        HIGH_FIO2_ITEMIDS,
        FEVER_ITEMIDS,
    )
    weights: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    stats = {"rows_read": 0, "rows_selected": 0, "events_written": 0, "weight_records": 0}
    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    for chunk_idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
        stats["rows_read"] += int(len(chunk))
        if chunk_idx == 1 or chunk_idx % 25 == 0:
            print(f"[mimic] chartevents chunk={chunk_idx} rows_read={stats['rows_read']} events={stats['events_written']}", flush=True)
        chunk = chunk[chunk["itemid"].isin(selected_ids) & chunk["stay_id"].isin(stay_info)]
        if chunk.empty:
            continue
        stats["rows_selected"] += int(len(chunk))
        chunk = chunk.copy()
        chunk["time"] = _relative_hours(chunk, stay_intime, "charttime")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk[chunk["time"].notna() & chunk["valuenum"].notna()]
        chunk = chunk[(chunk["time"] >= 0.0) & (chunk["time"] <= float(max_hours))]
        if chunk.empty:
            continue

        weight_mask = chunk["itemid"].isin(set(WEIGHT_ITEMIDS_KG) | set(WEIGHT_ITEMIDS_LB))
        for row in chunk[weight_mask].itertuples(index=False):
            unit = "lb" if int(row.itemid) in WEIGHT_ITEMIDS_LB else "kg"
            weight = _valid_weight_kg(row.valuenum, source_unit=unit)
            if weight is not None:
                weights[int(row.stay_id)].append((float(row.time), float(weight), f"chartevents:{int(row.itemid)}"))
                stats["weight_records"] += 1

        rows: list[dict[str, Any]] = []
        non_weight = chunk[~weight_mask]
        for row in non_weight.itertuples(index=False):
            itemid = int(row.itemid)
            value = float(row.valuenum)
            label = None
            if itemid in LOW_SYSBP_ITEMIDS and value < 90.0:
                label = "low_systolic_bp"
            elif itemid in LOW_MAP_ITEMIDS and value < 65.0:
                label = "low_map"
            elif itemid in HIGH_HR_ITEMIDS and value > 100.0:
                label = "high_heart_rate"
            elif itemid in HIGH_RR_ITEMIDS and value > 22.0:
                label = "high_respiratory_rate"
            elif itemid in LOW_SPO2_ITEMIDS and value < 92.0:
                label = "low_spo2"
            elif itemid in HIGH_FIO2_ITEMIDS:
                fio2 = value / 100.0 if value > 1.5 else value
                if fio2 > 0.5:
                    label = "high_fio2"
            elif itemid in FEVER_ITEMIDS and value >= 38.3:
                label = "fever"
            if label is None:
                continue
            rows.append(
                {
                    "sequence_id": int(row.stay_id),
                    "time": round(float(row.time), 6),
                    "event_type": label,
                    "source_table": "chartevents",
                    "itemid": itemid,
                }
            )
            counts[label] += 1
        stats["events_written"] += _append_events(source_events_path, rows)
    print(f"[mimic] chartevents done rows_read={stats['rows_read']} events={stats['events_written']} weights={stats['weight_records']}", flush=True)
    return weights, counts, stats


def _process_inputevents(
    path: Path,
    source_events_path: Path,
    stay_info: dict[int, StayInfo],
    chunksize: int,
    max_hours: float,
) -> tuple[dict[int, list[tuple[float, float, str]]], Counter[str], dict[str, int]]:
    stay_intime = {sid: info.intime for sid, info in stay_info.items()}
    selected_ids = set().union(VASOPRESSOR_ITEMIDS, CRYSTALLOID_ITEMIDS, ALBUMIN_ITEMIDS)
    weights: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    stats = {"rows_read": 0, "rows_selected": 0, "events_written": 0, "weight_records": 0}
    usecols = ["stay_id", "starttime", "itemid", "amount", "rate", "patientweight", "statusdescription"]
    for chunk_idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
        stats["rows_read"] += int(len(chunk))
        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(f"[mimic] inputevents chunk={chunk_idx} rows_read={stats['rows_read']} events={stats['events_written']}", flush=True)
        chunk = chunk[chunk["stay_id"].isin(stay_info)].copy()
        if chunk.empty:
            continue
        weight_rows = chunk[["stay_id", "starttime", "patientweight"]].copy()
        weight_rows["patientweight"] = _valid_weight_series(weight_rows["patientweight"])
        weight_rows = weight_rows[weight_rows["patientweight"].notna()]
        if not weight_rows.empty:
            weight_rows["time"] = _relative_hours(weight_rows, stay_intime, "starttime")
            weight_rows = weight_rows[weight_rows["time"].notna()]
            weight_rows = weight_rows[(weight_rows["time"] >= 0.0) & (weight_rows["time"] <= float(max_hours))]
            if not weight_rows.empty:
                weight_rows["abs_time"] = weight_rows["time"].abs()
                weight_rows = weight_rows.sort_values(["stay_id", "abs_time"], kind="mergesort")
                weight_rows = weight_rows.drop_duplicates(subset=["stay_id"], keep="first")
                for row in weight_rows.itertuples(index=False):
                    weights[int(row.stay_id)].append((float(row.time), float(row.patientweight), "inputevents:patientweight"))
                    stats["weight_records"] += 1

        chunk = chunk[chunk["itemid"].isin(selected_ids)]
        if chunk.empty:
            continue
        stats["rows_selected"] += int(len(chunk))
        chunk["time"] = _relative_hours(chunk, stay_intime, "starttime")
        chunk["amount"] = pd.to_numeric(chunk["amount"], errors="coerce")
        chunk["rate"] = pd.to_numeric(chunk["rate"], errors="coerce")
        usable_status = chunk["statusdescription"].map(_status_is_usable)
        positive = (chunk["amount"].fillna(0.0) > 0.0) | (chunk["rate"].fillna(0.0) > 0.0)
        chunk = chunk[chunk["time"].notna() & usable_status & positive]
        chunk = chunk[(chunk["time"] >= 0.0) & (chunk["time"] <= float(max_hours))]
        rows: list[dict[str, Any]] = []
        for row in chunk.itertuples(index=False):
            itemid = int(row.itemid)
            if itemid in VASOPRESSOR_ITEMIDS:
                label = "vasopressor"
            elif itemid in CRYSTALLOID_ITEMIDS:
                label = "crystalloid_fluid"
            elif itemid in ALBUMIN_ITEMIDS:
                label = "albumin_colloid"
            else:
                continue
            rows.append(
                {
                    "sequence_id": int(row.stay_id),
                    "time": round(float(row.time), 6),
                    "event_type": label,
                    "source_table": "inputevents",
                    "itemid": itemid,
                }
            )
            counts[label] += 1
        stats["events_written"] += _append_events(source_events_path, rows)
    print(f"[mimic] inputevents done rows_read={stats['rows_read']} events={stats['events_written']} weights={stats['weight_records']}", flush=True)
    return weights, counts, stats


def _process_procedureevents(
    path: Path,
    source_events_path: Path,
    stay_info: dict[int, StayInfo],
    chunksize: int,
    max_hours: float,
) -> tuple[dict[int, list[tuple[float, float, str]]], dict[int, float], Counter[str], dict[str, int]]:
    stay_intime = {sid: info.intime for sid, info in stay_info.items()}
    weights: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    rrt_time: dict[int, float] = {}
    counts: Counter[str] = Counter()
    stats = {"rows_read": 0, "rows_selected": 0, "events_written": 0, "weight_records": 0}
    usecols = ["stay_id", "starttime", "itemid", "patientweight", "statusdescription"]
    for chunk_idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
        stats["rows_read"] += int(len(chunk))
        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(f"[mimic] procedureevents chunk={chunk_idx} rows_read={stats['rows_read']} events={stats['events_written']}", flush=True)
        chunk = chunk[chunk["stay_id"].isin(stay_info)].copy()
        if chunk.empty:
            continue
        weight_rows = chunk[["stay_id", "starttime", "patientweight"]].copy()
        weight_rows["patientweight"] = _valid_weight_series(weight_rows["patientweight"])
        weight_rows = weight_rows[weight_rows["patientweight"].notna()]
        if not weight_rows.empty:
            weight_rows["time"] = _relative_hours(weight_rows, stay_intime, "starttime")
            weight_rows = weight_rows[weight_rows["time"].notna()]
            weight_rows = weight_rows[(weight_rows["time"] >= 0.0) & (weight_rows["time"] <= float(max_hours))]
            if not weight_rows.empty:
                weight_rows["abs_time"] = weight_rows["time"].abs()
                weight_rows = weight_rows.sort_values(["stay_id", "abs_time"], kind="mergesort")
                weight_rows = weight_rows.drop_duplicates(subset=["stay_id"], keep="first")
                for row in weight_rows.itertuples(index=False):
                    weights[int(row.stay_id)].append((float(row.time), float(row.patientweight), "procedureevents:patientweight"))
                    stats["weight_records"] += 1

        chunk = chunk[chunk["itemid"].isin(RRT_ITEMIDS)]
        if chunk.empty:
            continue
        stats["rows_selected"] += int(len(chunk))
        chunk["time"] = _relative_hours(chunk, stay_intime, "starttime")
        usable_status = chunk["statusdescription"].map(_status_is_usable)
        chunk = chunk[chunk["time"].notna() & usable_status]
        chunk = chunk[(chunk["time"] >= 0.0) & (chunk["time"] <= float(max_hours))]
        rows: list[dict[str, Any]] = []
        for row in chunk.itertuples(index=False):
            stay_id = int(row.stay_id)
            t = float(row.time)
            rrt_time[stay_id] = min(float(rrt_time.get(stay_id, t)), t)
            rows.append(
                {
                    "sequence_id": stay_id,
                    "time": round(t, 6),
                    "event_type": "rrt_started",
                    "source_table": "procedureevents",
                    "itemid": int(row.itemid),
                }
            )
            counts["rrt_started"] += 1
        stats["events_written"] += _append_events(source_events_path, rows)
    print(f"[mimic] procedureevents done rows_read={stats['rows_read']} events={stats['events_written']} weights={stats['weight_records']}", flush=True)
    return weights, rrt_time, counts, stats


def _choose_weights(*weight_maps: dict[int, list[tuple[float, float, str]]]) -> tuple[dict[int, float], dict[int, str]]:
    records: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for weight_map in weight_maps:
        for stay_id, values in weight_map.items():
            records[int(stay_id)].extend(values)
    out: dict[int, float] = {}
    source: dict[int, str] = {}
    priority = {
        "chartevents:226512": 0,
        "chartevents:224639": 1,
        "chartevents:226531": 2,
        "inputevents:patientweight": 3,
        "procedureevents:patientweight": 4,
    }
    for stay_id, values in records.items():
        values = sorted(values, key=lambda x: (priority.get(x[2], 99), abs(float(x[0])), float(x[0])))
        if not values:
            continue
        out[stay_id] = float(values[0][1])
        source[stay_id] = str(values[0][2])
    return out, source


def _process_outputevents_low_urine(
    path: Path,
    target_events_path: Path,
    stay_info: dict[int, StayInfo],
    weights: dict[int, float],
    rrt_time: dict[int, float],
    chunksize: int,
    max_hours: float,
) -> tuple[Counter[str], dict[str, Any], dict[int, int]]:
    stay_intime = {sid: info.intime for sid, info in stay_info.items()}
    urine_by_stay: dict[int, list[tuple[float, float]]] = defaultdict(list)
    stats = {
        "rows_read": 0,
        "rows_selected": 0,
        "valid_urine_records": 0,
        "target_events_written": 0,
        "stays_with_urine_records": 0,
        "stays_with_low_urine": 0,
        "stays_skipped_no_weight": 0,
    }
    usecols = ["stay_id", "charttime", "itemid", "value", "valueuom"]
    for chunk_idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
        stats["rows_read"] += int(len(chunk))
        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(f"[mimic] outputevents chunk={chunk_idx} rows_read={stats['rows_read']} selected={stats['rows_selected']}", flush=True)
        chunk = chunk[chunk["stay_id"].isin(stay_info) & chunk["itemid"].isin(URINE_OUTPUT_ITEMIDS)].copy()
        if chunk.empty:
            continue
        stats["rows_selected"] += int(len(chunk))
        chunk["time"] = _relative_hours(chunk, stay_intime, "charttime")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk[chunk["time"].notna() & chunk["value"].notna()]
        chunk = chunk[(chunk["time"] >= 0.0) & (chunk["time"] <= float(max_hours))]
        chunk = chunk[chunk["value"] >= 0.0]
        stats["valid_urine_records"] += int(len(chunk))
        for row in chunk.itertuples(index=False):
            urine_by_stay[int(row.stay_id)].append((float(row.time), float(row.value)))

    stats["stays_with_urine_records"] = int(len(urine_by_stay))
    rows: list[dict[str, Any]] = []
    target_counts_by_stay: dict[int, int] = {}
    for stay_id, records in urine_by_stay.items():
        weight = weights.get(stay_id)
        if weight is None:
            stats["stays_skipped_no_weight"] += 1
            continue
        censor = min(float(max_hours), float(stay_info[stay_id].los_hours), float(rrt_time.get(stay_id, max_hours)))
        records = sorted((t, v) for t, v in records if 0.0 <= t <= censor)
        window: deque[tuple[float, float]] = deque()
        total_ml = 0.0
        was_low = False
        count = 0
        for t, volume in records:
            window.append((t, volume))
            total_ml += float(volume)
            while window and window[0][0] <= t - LOW_URINE_WINDOW_HOURS:
                _, old_volume = window.popleft()
                total_ml -= float(old_volume)
            if t < LOW_URINE_WINDOW_HOURS:
                was_low = False
                continue
            rate = total_ml / (LOW_URINE_WINDOW_HOURS * float(weight))
            is_low = rate < LOW_URINE_RATE_THRESHOLD_ML_PER_KG_H
            if is_low and not was_low:
                rows.append(
                    {
                        "sequence_id": stay_id,
                        "time": round(float(t), 6),
                        "event_type": LOW_URINE_LABEL,
                        "source_table": "outputevents_roll6h",
                        "itemid": 0,
                    }
                )
                count += 1
            was_low = bool(is_low)
        if count:
            target_counts_by_stay[stay_id] = count
    stats["target_events_written"] = _append_events(target_events_path, rows, header=True)
    stats["stays_with_low_urine"] = int(len(target_counts_by_stay))
    print(f"[mimic] outputevents done rows_read={stats['rows_read']} target_events={stats['target_events_written']}", flush=True)
    return Counter({LOW_URINE_LABEL: stats["target_events_written"]}), stats, target_counts_by_stay


def _load_event_log(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["sequence_id", "time", "event_type", "source_table", "itemid"])
    return pd.read_csv(path)


def _build_final_sequences(
    *,
    source_events_path: Path,
    target_events_path: Path,
    final_event_log_path: Path,
    sequence_pickle_path: Path,
    split_pickle_path: Path,
    easytpp_dir: Path,
    target_easytpp_dir: Path,
    stay_info: dict[int, StayInfo],
    rrt_time: dict[int, float],
    max_hours: float,
    split_seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_df = _load_event_log(source_events_path)
    target_df = _load_event_log(target_events_path)
    df = pd.concat([source_df, target_df], ignore_index=True)
    if df.empty:
        raise ValueError("no MIMIC events generated")
    df["sequence_id"] = pd.to_numeric(df["sequence_id"], errors="coerce").astype("Int64")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["sequence_id", "time", "event_type"])
    df["sequence_id"] = df["sequence_id"].astype(int)
    df = df[df["sequence_id"].isin(stay_info)]
    df = df[(df["time"] >= 0.0) & (df["time"] <= float(max_hours))]
    censor = df["sequence_id"].map(lambda sid: min(float(max_hours), float(stay_info[int(sid)].los_hours), float(rrt_time.get(int(sid), max_hours))))
    df = df[df["time"] <= censor]
    df = df.drop_duplicates(subset=["sequence_id", "time", "event_type"])
    labels = [LOW_URINE_LABEL] + [label for label in SOURCE_LABELS if label in set(df["event_type"])]
    labels += [label for label in sorted(set(df["event_type"]) - set(labels))]
    event_to_id = {label: idx for idx, label in enumerate(labels)}
    df["event_id"] = df["event_type"].map(event_to_id).astype(int)
    df = df.sort_values(["sequence_id", "time", "event_id"], kind="mergesort")
    final_event_log_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(final_event_log_path, index=False)

    sequences: list[dict[str, Any]] = []
    target_id = int(event_to_id[LOW_URINE_LABEL])
    for stay_id, group in df.groupby("sequence_id", sort=True):
        if len(group) < 2:
            continue
        sequences.append(
            {
                "sequence_id": str(stay_id),
                "time": [float(x) for x in group["time"].tolist()],
                "event": [int(x) for x in group["event_id"].tolist()],
            }
        )
    rng = np.random.default_rng(int(split_seed))
    idx = np.arange(len(sequences), dtype=np.int64)
    target_presence = np.array([any(int(e) == target_id for e in seq["event"]) for seq in sequences], dtype=bool)
    train_idx: list[int] = []
    dev_idx: list[int] = []
    test_idx: list[int] = []
    for flag in [True, False]:
        subset = idx[target_presence == flag]
        rng.shuffle(subset)
        n_train = int(round(len(subset) * float(train_ratio)))
        n_dev = int(round(len(subset) * float(dev_ratio)))
        train_idx.extend(int(i) for i in subset[:n_train])
        dev_idx.extend(int(i) for i in subset[n_train:n_train + n_dev])
        test_idx.extend(int(i) for i in subset[n_train + n_dev:])
    train = [sequences[i] for i in sorted(train_idx)]
    dev = [sequences[i] for i in sorted(dev_idx)]
    test = [sequences[i] for i in sorted(test_idx)]
    metadata = {
        "dataset": "mimic_iv_low_urine",
        "dim_process": int(len(event_to_id)),
        "num_types": int(len(event_to_id)),
        "event_to_id": event_to_id,
        "id_to_event": {str(v): k for k, v in event_to_id.items()},
        "target_event_label": LOW_URINE_LABEL,
        "target_event_id": target_id,
        "time_unit": "hours_since_icu_intime",
        "split_seed": int(split_seed),
        "split_strategy": "sequence_level_stratified_by_target_presence",
    }
    sequence_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with sequence_pickle_path.open("wb") as f:
        pickle.dump({"sequences": sequences, "metadata": metadata}, f)
    split_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with split_pickle_path.open("wb") as f:
        pickle.dump({"train": train, "val": dev, "test": test, "metadata": metadata}, f)

    write_gatech_pickles(train=train, dev=dev, test=test, dim_process=len(event_to_id), output_dir=easytpp_dir)
    target_train = [_target_only(seq, target_id) for seq in train]
    target_dev = [_target_only(seq, target_id) for seq in dev]
    target_test = [_target_only(seq, target_id) for seq in test]
    target_train = [seq for seq in target_train if seq is not None]
    target_dev = [seq for seq in target_dev if seq is not None]
    target_test = [seq for seq in target_test if seq is not None]
    if target_train and target_dev and target_test:
        write_gatech_pickles(
            train=target_train,
            dev=target_dev,
            test=target_test,
            dim_process=1,
            output_dir=target_easytpp_dir,
        )
        (target_easytpp_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "easytpp_gatech",
                    "evaluation_scope": "target_event_process",
                    "target_event_label": LOW_URINE_LABEL,
                    "target_event_id_in_full_process": target_id,
                    "dim_process": 1,
                    "split_sequences": {
                        "train": len(target_train),
                        "dev": len(target_dev),
                        "test": len(target_test),
                    },
                },
                indent=2,
            )
        )
    split_summary = {
        "train_sequences": len(train),
        "dev_sequences": len(dev),
        "test_sequences": len(test),
        "train_target_sequences": sum(any(int(e) == target_id for e in seq["event"]) for seq in train),
        "dev_target_sequences": sum(any(int(e) == target_id for e in seq["event"]) for seq in dev),
        "test_target_sequences": sum(any(int(e) == target_id for e in seq["event"]) for seq in test),
        "train_events": sum(len(seq["event"]) for seq in train),
        "dev_events": sum(len(seq["event"]) for seq in dev),
        "test_events": sum(len(seq["event"]) for seq in test),
    }
    return sequences, metadata, split_summary


def _target_only(seq: dict[str, Any], target_id: int) -> dict[str, Any] | None:
    target_times = [float(t) for t, e in zip(seq["time"], seq["event"]) if int(e) == int(target_id)]
    if len(target_times) < 2:
        return None
    return {
        "sequence_id": seq.get("sequence_id"),
        "time": target_times,
        "event": [0 for _ in target_times],
    }


def _event_counts_from_sequences(sequences: list[dict[str, Any]], id_to_event: dict[str, str]) -> dict[str, dict[str, int]]:
    event_counts: Counter[int] = Counter()
    seq_counts: Counter[int] = Counter()
    for seq in sequences:
        seen = set()
        for event in seq["event"]:
            event_counts[int(event)] += 1
            seen.add(int(event))
        for event in seen:
            seq_counts[int(event)] += 1
    out = {}
    for event_id, count in event_counts.most_common():
        label = id_to_event[str(event_id)]
        out[label] = {"events": int(count), "sequences": int(seq_counts[event_id])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare MIMIC-IV low urine event-log benchmark inputs.")
    ap.add_argument("--raw_dir", default="data/realworld_raw/mimic_iv")
    ap.add_argument("--prepared_dir", default="data/realworld_prepared/mimic_low_urine")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--min_los_hours", type=float, default=6.0)
    ap.add_argument("--max_los_hours", type=float, default=336.0)
    ap.add_argument("--max_event_time_hours", type=float, default=336.0)
    ap.add_argument("--split_seed", type=int, default=111)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    prepared_dir = Path(args.prepared_dir)
    derived_dir = raw_dir / "derived_low_urine"
    derived_dir.mkdir(parents=True, exist_ok=True)
    info_path = raw_dir / "mimic_low_urine_dataset_info.md"
    _write_dataset_info(info_path)

    paths = {
        "icustays": raw_dir / "icustays.csv",
        "chartevents": raw_dir / "chartevents.csv",
        "inputevents": raw_dir / "inputevents.csv",
        "procedureevents": raw_dir / "procedureevents.csv",
        "outputevents": raw_dir / "outputevents.csv",
        "d_items": raw_dir / "d_items.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing MIMIC files: {missing}")

    stays, stay_info = _read_stays(paths["icustays"], args.min_los_hours, args.max_los_hours)
    source_events_path = derived_dir / "source_events_raw.csv"
    target_events_path = derived_dir / "target_events_low_urine.csv"
    final_event_log_path = raw_dir / "events_low_urine.csv"
    _append_events(source_events_path, [], header=True)

    char_weights, char_counts, char_stats = _process_chartevents(
        paths["chartevents"],
        source_events_path,
        stay_info,
        int(args.chunksize),
        float(args.max_event_time_hours),
    )
    input_weights, input_counts, input_stats = _process_inputevents(
        paths["inputevents"],
        source_events_path,
        stay_info,
        int(args.chunksize),
        float(args.max_event_time_hours),
    )
    proc_weights, rrt_time, proc_counts, proc_stats = _process_procedureevents(
        paths["procedureevents"],
        source_events_path,
        stay_info,
        int(args.chunksize),
        float(args.max_event_time_hours),
    )
    weights, weight_source = _choose_weights(char_weights, input_weights, proc_weights)
    target_counts, target_stats, target_counts_by_stay = _process_outputevents_low_urine(
        paths["outputevents"],
        target_events_path,
        stay_info,
        weights,
        rrt_time,
        int(args.chunksize),
        float(args.max_event_time_hours),
    )
    sequences, metadata, split_summary = _build_final_sequences(
        source_events_path=source_events_path,
        target_events_path=target_events_path,
        final_event_log_path=final_event_log_path,
        sequence_pickle_path=prepared_dir / "sequences.pkl",
        split_pickle_path=prepared_dir / f"split_seed{int(args.split_seed)}.pkl",
        easytpp_dir=prepared_dir / "easytpp",
        target_easytpp_dir=prepared_dir / "easytpp_target_event",
        stay_info=stay_info,
        rrt_time=rrt_time,
        max_hours=float(args.max_event_time_hours),
        split_seed=int(args.split_seed),
        train_ratio=float(args.train_ratio),
        dev_ratio=float(args.dev_ratio),
    )
    event_counts = _event_counts_from_sequences(sequences, metadata["id_to_event"])
    target_id = int(metadata["target_event_id"])
    target_sequences = sum(any(int(e) == target_id for e in seq["event"]) for seq in sequences)
    audit = {
        "dataset": "mimic_iv_low_urine",
        "raw_dir": str(raw_dir),
        "prepared_dir": str(prepared_dir),
        "eligible_stays": int(len(stays)),
        "sequence_count": int(len(sequences)),
        "sequence_coverage_among_eligible_stays": float(len(sequences) / max(len(stays), 1)),
        "low_urine_target": {
            "definition": f"rolling {LOW_URINE_WINDOW_HOURS:g}h urine output rate < {LOW_URINE_RATE_THRESHOLD_ML_PER_KG_H:g} mL/kg/h",
            "target_event_id": target_id,
            "target_event_label": LOW_URINE_LABEL,
            "target_events": int(sum(target_counts_by_stay.values())),
            "target_stays": int(len(target_counts_by_stay)),
            "target_stay_rate_among_eligible_stays": float(len(target_counts_by_stay) / max(len(stays), 1)),
            "target_sequence_rate": float(target_sequences / max(len(sequences), 1)),
            "urine_itemids": URINE_OUTPUT_ITEMIDS,
            "target_stats": target_stats,
        },
        "weight_coverage": {
            "stays_with_weight": int(len(weights)),
            "coverage_among_eligible_stays": float(len(weights) / max(len(stays), 1)),
            "source_counts": dict(Counter(weight_source.values())),
        },
        "rrt_censor": {
            "stays_with_rrt": int(len(rrt_time)),
            "coverage_among_eligible_stays": float(len(rrt_time) / max(len(stays), 1)),
        },
        "source_predicate_counts_raw": {
            **dict(char_counts),
            **{k: int(char_counts.get(k, 0) + input_counts.get(k, 0) + proc_counts.get(k, 0)) for k in SOURCE_LABELS},
        },
        "source_predicate_frequency_final": {
            key: value for key, value in event_counts.items() if key != LOW_URINE_LABEL
        },
        "event_frequency_final": event_counts,
        "split_summary": split_summary,
        "processing_stats": {
            "chartevents": char_stats,
            "inputevents": input_stats,
            "procedureevents": proc_stats,
            "outputevents": target_stats,
        },
        "outputs": {
            "dataset_info": str(info_path),
            "source_events_raw": str(source_events_path),
            "target_events_low_urine": str(target_events_path),
            "final_event_log": str(final_event_log_path),
            "sequence_pickle": str(prepared_dir / "sequences.pkl"),
            "split_pickle": str(prepared_dir / f"split_seed{int(args.split_seed)}.pkl"),
            "easytpp_dir": str(prepared_dir / "easytpp"),
            "target_easytpp_dir": str(prepared_dir / "easytpp_target_event"),
        },
    }
    audit_path = raw_dir / "mimic_low_urine_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / "manifest.json").write_text(json.dumps(audit, indent=2))
    _write_dataset_info(info_path, audit=audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

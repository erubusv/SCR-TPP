from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from ..adapters.realworld import events_csv_to_sequences, write_sequence_pickle


DEFAULT_PHYSIONET_PROJECT = "physionet-data"
DEFAULT_HOSP_DATASET = "mimiciv_v3_1_hosp"
DEFAULT_ICU_DATASET = "mimiciv_v3_1_icu"
USD_PER_TIB = 6.25


def _require_bigquery():
    try:
        from google.cloud import bigquery  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "google-cloud-bigquery is not installed. Run: python -m pip install google-cloud-bigquery"
        ) from exc
    return bigquery


def _bytes_from_gib(value: float) -> int:
    return int(float(value) * (1024**3))


def _human_bytes(value: int | float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(value)
    for unit in units:
        if abs(x) < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"


def _worst_case_usd(processed_bytes: int) -> float:
    return (float(processed_bytes) / float(1024**4)) * USD_PER_TIB


def _table(project: str, dataset: str, name: str) -> str:
    return f"`{project}.{dataset}.{name}`"


def _load_selected_itemids(path: str | None) -> dict[str, set[int]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    out: dict[str, set[int]] = {}
    for source, values in data.items():
        out[str(source)] = {int(v) for v in values}
    return out


def _itemid_filter(alias: str, source: str, selected: dict[str, set[int]]) -> str:
    values = sorted(selected.get(source, set()))
    if not values:
        return ""
    joined = ", ".join(str(v) for v in values)
    return f"AND {alias}.itemid IN ({joined})"


def _source_list(value: str) -> list[str]:
    sources = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"outputevents", "inputevents", "procedureevents", "datetimeevents", "labevents"}
    bad = sorted(set(sources) - allowed)
    if bad:
        raise argparse.ArgumentTypeError(f"unknown source(s): {', '.join(bad)}")
    if not sources:
        raise argparse.ArgumentTypeError("at least one source is required")
    return sources


def _sanitize_dataset_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(f"unsafe BigQuery dataset id: {value!r}")
    return value


def build_event_log_query(args: argparse.Namespace) -> str:
    project = args.physionet_project
    hosp = _sanitize_dataset_id(args.hosp_dataset)
    icu = _sanitize_dataset_id(args.icu_dataset)
    selected_itemids = _load_selected_itemids(args.selected_itemids_json)
    stays_limit = ""
    if int(args.max_stays) > 0:
        stays_limit = f"QUALIFY ROW_NUMBER() OVER (ORDER BY stay_id) <= {int(args.max_stays)}"

    stays = f"""
stays AS (
  SELECT
    stay_id,
    subject_id,
    hadm_id,
    intime,
    outtime
  FROM {_table(project, icu, "icustays")}
  WHERE DATETIME_DIFF(outtime, intime, HOUR) BETWEEN {float(args.min_los_hours)} AND {float(args.max_los_hours)}
  {stays_limit}
)"""

    ctes = [stays]
    selects: list[str] = []

    if "outputevents" in args.sources:
        ctes.append(
            f"""
output_events AS (
  SELECT
    CAST(o.stay_id AS STRING) AS sequence_id,
    DATETIME_DIFF(o.charttime, s.intime, SECOND) / 3600.0 AS time,
    CONCAT(
      'output:',
      REGEXP_REPLACE(LOWER(COALESCE(d.label, CONCAT('item_', CAST(o.itemid AS STRING)))), r'[^a-z0-9]+', '_')
    ) AS event_type
  FROM {_table(project, icu, "outputevents")} AS o
  INNER JOIN stays AS s USING (stay_id)
  LEFT JOIN {_table(project, icu, "d_items")} AS d USING (itemid)
  WHERE o.charttime BETWEEN s.intime AND s.outtime
    {_itemid_filter("o", "outputevents", selected_itemids)}
)"""
        )
        selects.append("SELECT sequence_id, time, event_type FROM output_events")

    if "inputevents" in args.sources:
        ctes.append(
            f"""
input_events AS (
  SELECT
    CAST(i.stay_id AS STRING) AS sequence_id,
    DATETIME_DIFF(i.starttime, s.intime, SECOND) / 3600.0 AS time,
    CONCAT(
      'input:',
      REGEXP_REPLACE(LOWER(COALESCE(d.label, CONCAT('item_', CAST(i.itemid AS STRING)))), r'[^a-z0-9]+', '_')
    ) AS event_type
  FROM {_table(project, icu, "inputevents")} AS i
  INNER JOIN stays AS s USING (stay_id)
  LEFT JOIN {_table(project, icu, "d_items")} AS d USING (itemid)
  WHERE i.starttime BETWEEN s.intime AND s.outtime
    {_itemid_filter("i", "inputevents", selected_itemids)}
)"""
        )
        selects.append("SELECT sequence_id, time, event_type FROM input_events")

    if "procedureevents" in args.sources:
        ctes.append(
            f"""
procedure_events AS (
  SELECT
    CAST(p.stay_id AS STRING) AS sequence_id,
    DATETIME_DIFF(p.starttime, s.intime, SECOND) / 3600.0 AS time,
    CONCAT(
      'procedure:',
      REGEXP_REPLACE(LOWER(COALESCE(d.label, CONCAT('item_', CAST(p.itemid AS STRING)))), r'[^a-z0-9]+', '_')
    ) AS event_type
  FROM {_table(project, icu, "procedureevents")} AS p
  INNER JOIN stays AS s USING (stay_id)
  LEFT JOIN {_table(project, icu, "d_items")} AS d USING (itemid)
  WHERE p.starttime BETWEEN s.intime AND s.outtime
    {_itemid_filter("p", "procedureevents", selected_itemids)}
)"""
        )
        selects.append("SELECT sequence_id, time, event_type FROM procedure_events")

    if "datetimeevents" in args.sources:
        ctes.append(
            f"""
datetime_events AS (
  SELECT
    CAST(dt.stay_id AS STRING) AS sequence_id,
    DATETIME_DIFF(dt.charttime, s.intime, SECOND) / 3600.0 AS time,
    CONCAT(
      'datetime:',
      REGEXP_REPLACE(LOWER(COALESCE(d.label, CONCAT('item_', CAST(dt.itemid AS STRING)))), r'[^a-z0-9]+', '_')
    ) AS event_type
  FROM {_table(project, icu, "datetimeevents")} AS dt
  INNER JOIN stays AS s USING (stay_id)
  LEFT JOIN {_table(project, icu, "d_items")} AS d USING (itemid)
  WHERE dt.charttime BETWEEN s.intime AND s.outtime
    {_itemid_filter("dt", "datetimeevents", selected_itemids)}
)"""
        )
        selects.append("SELECT sequence_id, time, event_type FROM datetime_events")

    if "labevents" in args.sources:
        ctes.append(
            f"""
lab_events AS (
  SELECT
    CAST(s.stay_id AS STRING) AS sequence_id,
    DATETIME_DIFF(l.charttime, s.intime, SECOND) / 3600.0 AS time,
    CONCAT(
      'lab',
      IF(l.flag IS NULL OR l.flag = '', '', CONCAT('_', REGEXP_REPLACE(LOWER(l.flag), r'[^a-z0-9]+', '_'))),
      ':',
      REGEXP_REPLACE(LOWER(COALESCE(d.label, CONCAT('item_', CAST(l.itemid AS STRING)))), r'[^a-z0-9]+', '_')
    ) AS event_type
  FROM {_table(project, hosp, "labevents")} AS l
  INNER JOIN stays AS s
    ON l.hadm_id = s.hadm_id
   AND l.charttime BETWEEN s.intime AND s.outtime
  LEFT JOIN {_table(project, hosp, "d_labitems")} AS d USING (itemid)
  WHERE l.charttime IS NOT NULL
    {_itemid_filter("l", "labevents", selected_itemids)}
)"""
        )
        selects.append("SELECT sequence_id, time, event_type FROM lab_events")

    union_sql = "\nUNION ALL\n".join(selects)
    return f"""
#standardSQL
WITH
{",".join(ctes)},
all_events AS (
{union_sql}
)
SELECT
  sequence_id,
  time,
  event_type
FROM all_events
WHERE time >= 0
  AND time <= {float(args.max_event_time_hours)}
ORDER BY sequence_id, time, event_type
""".strip()


def build_catalog_query(args: argparse.Namespace) -> str:
    project = args.physionet_project
    hosp = _sanitize_dataset_id(args.hosp_dataset)
    icu = _sanitize_dataset_id(args.icu_dataset)
    return f"""
#standardSQL
SELECT
  'icu.d_items' AS source_table,
  itemid,
  label,
  category,
  unitname AS unit
FROM {_table(project, icu, "d_items")}
UNION ALL
SELECT
  'hosp.d_labitems' AS source_table,
  itemid,
  label,
  fluid AS category,
  NULL AS unit
FROM {_table(project, hosp, "d_labitems")}
ORDER BY source_table, category, label, itemid
""".strip()


def _run_dry_run(client: Any, bigquery: Any, sql: str, max_bytes_billed: int, location: str | None) -> int:
    job_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        maximum_bytes_billed=max_bytes_billed,
    )
    job = client.query(sql, job_config=job_config, location=location or None)
    return int(job.total_bytes_processed)


def _execute_to_csv(
    client: Any,
    bigquery: Any,
    sql: str,
    output_csv: Path,
    max_bytes_billed: int,
    location: str | None,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        maximum_bytes_billed=max_bytes_billed,
    )
    job = client.query(sql, job_config=job_config, location=location or None)
    rows = job.result()
    count = 0
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence_id", "time", "event_type"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence_id": row["sequence_id"],
                    "time": row["time"],
                    "event_type": row["event_type"],
                }
            )
            count += 1
    return {
        "job_id": job.job_id,
        "output_csv": str(output_csv),
        "rows_written": count,
        "total_bytes_billed": int(job.total_bytes_billed or 0),
        "total_bytes_processed": int(job.total_bytes_processed or 0),
    }


def _execute_catalog_to_csv(
    client: Any,
    bigquery: Any,
    sql: str,
    output_csv: Path,
    max_bytes_billed: int,
    location: str | None,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        maximum_bytes_billed=max_bytes_billed,
    )
    job = client.query(sql, job_config=job_config, location=location or None)
    rows = job.result()
    count = 0
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_table", "itemid", "label", "category", "unit"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_table": row["source_table"],
                    "itemid": row["itemid"],
                    "label": row["label"],
                    "category": row["category"],
                    "unit": row["unit"],
                }
            )
            count += 1
    return {
        "job_id": job.job_id,
        "output_csv": str(output_csv),
        "rows_written": count,
        "total_bytes_billed": int(job.total_bytes_billed or 0),
        "total_bytes_processed": int(job.total_bytes_processed or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Safely extract a MIMIC-IV BigQuery event log into the benchmark CSV schema. "
            "Default mode is dry-run only; pass --execute to write data."
        )
    )
    ap.add_argument("--billing_project", required=True, help="Google Cloud project charged for BigQuery jobs")
    ap.add_argument(
        "--credentials_json",
        help="Optional Google Application Default Credentials JSON path",
    )
    ap.add_argument("--physionet_project", default=DEFAULT_PHYSIONET_PROJECT)
    ap.add_argument("--hosp_dataset", default=DEFAULT_HOSP_DATASET)
    ap.add_argument("--icu_dataset", default=DEFAULT_ICU_DATASET)
    ap.add_argument("--location", default="US")
    ap.add_argument(
        "--sources",
        type=_source_list,
        default=_source_list("outputevents,inputevents,procedureevents"),
        help="comma-separated event tables: outputevents,inputevents,procedureevents,datetimeevents,labevents",
    )
    ap.add_argument(
        "--selected_itemids_json",
        help=(
            "Optional JSON map from source table to itemid list. Example: "
            "{\"outputevents\": [226559], \"inputevents\": [225158]}"
        ),
    )
    ap.add_argument("--min_los_hours", type=float, default=4.0)
    ap.add_argument("--max_los_hours", type=float, default=336.0)
    ap.add_argument("--max_event_time_hours", type=float, default=336.0)
    ap.add_argument("--max_stays", type=int, default=0, help="0 means all eligible stays")
    ap.add_argument("--max_bytes_billed_gib", type=float, default=20.0)
    ap.add_argument("--execute", action="store_true", help="Run the query after dry-run passes")
    ap.add_argument("--catalog_only", action="store_true", help="Extract only d_items/d_labitems item catalog")
    ap.add_argument("--catalog_csv", default="data/realworld_raw/mimic_iv/item_catalog.csv")
    ap.add_argument("--output_csv", default="data/realworld_raw/mimic_iv/events.csv")
    ap.add_argument("--manifest_json", default="data/realworld_raw/mimic_iv/bigquery_extract_manifest.json")
    ap.add_argument("--write_sql", default="", help="Optional path to save the generated SQL")
    ap.add_argument("--prepare_sequence_pickle", action="store_true")
    ap.add_argument("--sequence_pickle", default="data/realworld_prepared/mimic_iv/sequences.pkl")
    ap.add_argument("--top_k_event_types", type=int, default=50)
    ap.add_argument("--target_event_label", default="")
    ap.add_argument("--min_events_per_sequence", type=int, default=2)
    args = ap.parse_args()

    if args.credentials_json:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(args.credentials_json).expanduser())

    bigquery = _require_bigquery()
    client = bigquery.Client(project=args.billing_project)
    sql = build_catalog_query(args) if args.catalog_only else build_event_log_query(args)
    max_bytes_billed = _bytes_from_gib(args.max_bytes_billed_gib)

    if args.write_sql:
        sql_path = Path(args.write_sql)
        sql_path.parent.mkdir(parents=True, exist_ok=True)
        sql_path.write_text(sql + "\n")

    estimated_bytes = _run_dry_run(
        client=client,
        bigquery=bigquery,
        sql=sql,
        max_bytes_billed=max_bytes_billed,
        location=args.location,
    )
    manifest: dict[str, Any] = {
        "mode": "execute" if args.execute else "dry_run",
        "billing_project": args.billing_project,
        "physionet_project": args.physionet_project,
        "hosp_dataset": args.hosp_dataset,
        "icu_dataset": args.icu_dataset,
        "sources": args.sources,
        "catalog_only": bool(args.catalog_only),
        "selected_itemids_json": args.selected_itemids_json,
        "filters": {
            "min_los_hours": args.min_los_hours,
            "max_los_hours": args.max_los_hours,
            "max_event_time_hours": args.max_event_time_hours,
            "max_stays": args.max_stays,
        },
        "estimated_bytes_processed": estimated_bytes,
        "estimated_bytes_processed_human": _human_bytes(estimated_bytes),
        "max_bytes_billed": max_bytes_billed,
        "max_bytes_billed_human": _human_bytes(max_bytes_billed),
        "worst_case_query_cost_usd_before_free_tier": round(_worst_case_usd(estimated_bytes), 6),
        "cost_guard": "query fails if BigQuery estimates bytes above max_bytes_billed",
        "sql_path": args.write_sql,
    }
    if estimated_bytes > max_bytes_billed:
        raise SystemExit(
            json.dumps(
                {
                    **manifest,
                    "error": "estimated bytes exceed --max_bytes_billed_gib; query not executed",
                },
                indent=2,
            )
        )

    if args.execute:
        if args.catalog_only:
            execution = _execute_catalog_to_csv(
                client=client,
                bigquery=bigquery,
                sql=sql,
                output_csv=Path(args.catalog_csv),
                max_bytes_billed=max_bytes_billed,
                location=args.location,
            )
        else:
            execution = _execute_to_csv(
                client=client,
                bigquery=bigquery,
                sql=sql,
                output_csv=Path(args.output_csv),
                max_bytes_billed=max_bytes_billed,
                location=args.location,
            )
        manifest["execution"] = execution
        if args.prepare_sequence_pickle and not args.catalog_only:
            sequences, metadata = events_csv_to_sequences(
                csv_path=args.output_csv,
                sequence_col="sequence_id",
                time_col="time",
                event_col="event_type",
                top_k_event_types=args.top_k_event_types,
                target_event_label=args.target_event_label or None,
                min_events_per_sequence=args.min_events_per_sequence,
                start_at_zero=True,
            )
            metadata["bigquery_manifest"] = str(args.manifest_json)
            manifest["sequence_pickle"] = write_sequence_pickle(
                sequences=sequences,
                metadata=metadata,
                output_path=args.sequence_pickle,
            )
            manifest["sequence_metadata"] = metadata

    manifest_path = Path(args.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

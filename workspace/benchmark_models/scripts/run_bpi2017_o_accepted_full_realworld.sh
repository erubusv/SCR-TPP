#!/usr/bin/env bash
set -Eeuo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-12}"
export PYTHONPATH="${PYTHONPATH:-/workspace}"

PREPARED_DIR="${PREPARED_DIR:-data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000}"
RESULT_DIR="${RESULT_DIR:-data/realworld_results/bpi2017_o_accepted_mixed_random5000_seed111_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-111}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRID_SIZE="${GRID_SIZE:-96}"
PRED_GRID_SIZE="${PRED_GRID_SIZE:-64}"
MAX_LEN="${MAX_LEN:-64}"

RULE_MANIFEST="${PREPARED_DIR}/rule_inputs/rule_input_manifest.json"
SEQUENCE_PICKLE="${PREPARED_DIR}/sequences.pkl"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-workspace/benchmark_models/configs/baselines.official_or_faithful.yaml}"
PREDICTION_MANIFEST="${PREDICTION_MANIFEST:-workspace/benchmark_models/configs/prediction_baselines.easytpp_official.yaml}"

mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/scr_tpp"
echo "${RESULT_DIR}" > data/realworld_results/latest_bpi2017_o_accepted_full_result_dir.txt
echo "running" > "${RESULT_DIR}/STATUS"

cp "${PREPARED_DIR}/readiness_audit.json" "${RESULT_DIR}/readiness_audit.json"
cp "${PREPARED_DIR}/readiness_benchmark_audit.json" "${RESULT_DIR}/readiness_benchmark_audit.json"

python - <<'PY' "${PREPARED_DIR}" "${RULE_MANIFEST}" "${SEQUENCE_PICKLE}"
import json
import pathlib
import sys

prepared = pathlib.Path(sys.argv[1])
rule_manifest = pathlib.Path(sys.argv[2])
sequence_pickle = pathlib.Path(sys.argv[3])

checks = [
    prepared / "readiness_benchmark_audit.json",
    prepared / "readiness_audit.json",
    rule_manifest,
    sequence_pickle,
]
missing = [str(path) for path in checks if not path.exists()]
if missing:
    raise SystemExit(f"missing required inputs: {missing}")
audit = json.loads((prepared / "readiness_benchmark_audit.json").read_text())
if not audit.get("ok"):
    raise SystemExit("BPI benchmark readiness audit is not ok")
manifest = json.loads(rule_manifest.read_text())
if int(manifest["target_event_id"]) != 0 or manifest["target_event_label"] != "O_Accepted":
    raise SystemExit("unexpected BPI target event")
print(json.dumps({
    "ok": True,
    "prepared_dir": str(prepared),
    "target": manifest["target_event_label"],
    "target_event_id": manifest["target_event_id"],
    "sequence_pickle": str(sequence_pickle),
}, indent=2))
PY

python -m workspace.benchmark_models.scr_tpp.run_realworld \
  --rule_input_manifest "${RULE_MANIFEST}" \
  --output_dir "${RESULT_DIR}/scr_tpp" \
  --seeds "${SEED}" \
  --device cuda:0 \
  --sieve_batch_size 1 \
  --exact_batch_size "${EXACT_BATCH_SIZE:-1}" \
  > "${RESULT_DIR}/logs/scr_tpp.log" 2>&1

run_bg() {
  local name="$1"
  shift
  (
    set -Eeuo pipefail
    echo "[${name}] start $(date -Is)"
    "$@"
    echo "[${name}] end $(date -Is)"
  ) > "${RESULT_DIR}/logs/${name}.log" 2>&1 &
  echo "$!" > "${RESULT_DIR}/logs/${name}.pid"
}

run_bg logical_clnn_nstpp \
  python -m workspace.benchmark_models.runners.realworld_rule_benchmarks \
    --rule_input_manifest "${RULE_MANIFEST}" \
    --baseline_manifest "${BASELINE_MANIFEST}" \
    --models CLNN,NSTPP \
    --output_dir "${RESULT_DIR}/logical_gpu0" \
    --device cuda:0 \
    --seeds "${SEED}"

run_bg logical_cluster \
  python -m workspace.benchmark_models.runners.realworld_rule_benchmarks \
    --rule_input_manifest "${RULE_MANIFEST}" \
    --baseline_manifest "${BASELINE_MANIFEST}" \
    --models CLUSTER \
    --output_dir "${RESULT_DIR}/logical_gpu1" \
    --device cuda:1 \
    --seeds "${SEED}"

run_bg logical_teller \
  python -m workspace.benchmark_models.runners.realworld_rule_benchmarks \
    --rule_input_manifest "${RULE_MANIFEST}" \
    --baseline_manifest "${BASELINE_MANIFEST}" \
    --models TELLER \
    --output_dir "${RESULT_DIR}/logical_teller" \
    --device cuda:1 \
    --seeds "${SEED}"

run_bg easytpp_core_gpu0 \
  python -m workspace.benchmark_models.runners.realworld_prediction \
    --dataset generic_pickle \
    --target O_Accepted \
    --input_pickle "${SEQUENCE_PICKLE}" \
    --output_dir "${RESULT_DIR}/easytpp_core_gpu0" \
    --prepare_easytpp \
    --prediction_manifest "${PREDICTION_MANIFEST}" \
    --models RMTPP,NHP,SAHP \
    --seed "${SEED}" \
    --gpu 0 \
    --batch_size "${BATCH_SIZE}" \
    --max_epoch "${EPOCHS}" \
    --learning_rate 0.001 \
    --hidden_size 64 \
    --time_emb_size 16 \
    --num_layers 2 \
    --num_heads 2 \
    --mc_samples 20 \
    --max_len "${MAX_LEN}" \
    --target_prediction_grid_size "${PRED_GRID_SIZE}" \
    --prediction_results_jsonl "${RESULT_DIR}/easytpp_core_gpu0/prediction_results.jsonl"

run_bg easytpp_transformer_gpu1 \
  python -m workspace.benchmark_models.runners.realworld_prediction \
    --dataset generic_pickle \
    --target O_Accepted \
    --input_pickle "${SEQUENCE_PICKLE}" \
    --output_dir "${RESULT_DIR}/easytpp_transformer_gpu1" \
    --prepare_easytpp \
    --prediction_manifest "${PREDICTION_MANIFEST}" \
    --models THP,AttNHP,IntensityFree \
    --seed "${SEED}" \
    --gpu 1 \
    --batch_size "${BATCH_SIZE}" \
    --max_epoch "${EPOCHS}" \
    --learning_rate 0.001 \
    --hidden_size 64 \
    --time_emb_size 16 \
    --num_layers 2 \
    --num_heads 2 \
    --mc_samples 20 \
    --max_len "${MAX_LEN}" \
    --target_prediction_grid_size "${PRED_GRID_SIZE}" \
    --prediction_results_jsonl "${RESULT_DIR}/easytpp_transformer_gpu1/prediction_results.jsonl"

failed=0
for name in logical_clnn_nstpp logical_cluster logical_teller easytpp_core_gpu0 easytpp_transformer_gpu1; do
  pid="$(cat "${RESULT_DIR}/logs/${name}.pid")"
  if ! wait "${pid}"; then
    echo "[driver] ${name} failed; see ${RESULT_DIR}/logs/${name}.log" | tee -a "${RESULT_DIR}/logs/driver.log"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "failed" > "${RESULT_DIR}/STATUS"
  exit 1
fi

python - <<'PY' "${RESULT_DIR}"
import json
import pathlib
import sys

result_dir = pathlib.Path(sys.argv[1])
rows = []
for path in [
    result_dir / "scr_tpp" / "scr_tpp_results.jsonl",
    result_dir / "logical_gpu0" / "rule_results.jsonl",
    result_dir / "logical_gpu1" / "rule_results.jsonl",
    result_dir / "logical_teller" / "rule_results.jsonl",
]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
out = result_dir / "logical_train_results.jsonl"
with out.open("w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(json.dumps({"logical_train_results": str(out), "rows": len(rows)}, indent=2))
PY

python -m workspace.benchmark_models.evaluation.realworld_logical_prediction \
  --rule_input_manifest "${RULE_MANIFEST}" \
  --rule_results_jsonl "${RESULT_DIR}/logical_train_results.jsonl" \
  --output_jsonl "${RESULT_DIR}/logical_prediction_results.jsonl" \
  --device cuda:0 \
  --grid_size "${GRID_SIZE}" \
  --prediction_grid_size "${PRED_GRID_SIZE}" \
  > "${RESULT_DIR}/logs/logical_prediction_eval.log" 2>&1

python - <<'PY' "${RESULT_DIR}"
import json
import pathlib
import sys

result_dir = pathlib.Path(sys.argv[1])
rows = []
for path in [
    result_dir / "logical_prediction_results.jsonl",
    result_dir / "easytpp_core_gpu0" / "prediction_results.jsonl",
    result_dir / "easytpp_transformer_gpu1" / "prediction_results.jsonl",
]:
    if not path.exists():
        raise SystemExit(f"missing result file: {path}")
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("dataset") == "generic_pickle":
                    row["dataset"] = "bpi2017_o_accepted_mixed_random_5000"
                rows.append(row)
out = result_dir / "all_realworld_results.jsonl"
with out.open("w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(json.dumps({"all_realworld_results": str(out), "rows": len(rows)}, indent=2))
PY

python -m workspace.benchmark_models.reports.write_realworld_benchmark_report \
  --results_jsonl "${RESULT_DIR}/all_realworld_results.jsonl" \
  --output_md "${RESULT_DIR}/realworld_benchmark_report.md" \
  --readiness_audit_json "${RESULT_DIR}/readiness_benchmark_audit.json" \
  > "${RESULT_DIR}/logs/write_report.log" 2>&1

echo "completed" > "${RESULT_DIR}/STATUS"
echo "${RESULT_DIR}"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/workspace}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-12}"

PREPARED_DIR="${PREPARED_DIR:-data/realworld_prepared/mimic_low_urine_random_5000}"
RESULT_DIR="${RESULT_DIR:-data/realworld_results/mimic_low_urine_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-111}"
TARGET_ID="${TARGET_ID:-0}"
TARGET_LABEL="${TARGET_LABEL:-low_urine_output}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GRID_SIZE="${GRID_SIZE:-96}"
PRED_GRID_SIZE="${PRED_GRID_SIZE:-64}"

mkdir -p "${RESULT_DIR}/logs"

python -m workspace.benchmark_models.data.prepare_realworld_rule_inputs \
  --sequence_pickle "${PREPARED_DIR}/sequences.pkl" \
  --split_pickle "${PREPARED_DIR}/split_seed${SEED}.pkl" \
  --output_dir "${PREPARED_DIR}/rule_inputs" \
  --dataset_name mimic_iv_low_urine \
  --target_event_id "${TARGET_ID}" \
  --target_event_label "${TARGET_LABEL}" \
  --seeds "${SEED}" \
  > "${RESULT_DIR}/logs/prepare_rule_inputs.log" 2>&1

if [ -d "${PREPARED_DIR}/easytpp_seed${SEED}" ]; then
  rm -rf "${PREPARED_DIR}/easytpp"
  ln -s "$(realpath "${PREPARED_DIR}/easytpp_seed${SEED}")" "${PREPARED_DIR}/easytpp"
fi

python -m workspace.benchmark_models.data.audit_realworld_benchmark_readiness \
  --prepared_dir "${PREPARED_DIR}" \
  --split_pickle "${PREPARED_DIR}/split_seed${SEED}.pkl" \
  --rule_input_manifest "${PREPARED_DIR}/rule_inputs/rule_input_manifest.json" \
  --output_json "${RESULT_DIR}/readiness_audit.json" \
  > "${RESULT_DIR}/logs/readiness_audit.log" 2>&1

(
  python -m workspace.benchmark_models.scr_tpp.run_realworld \
    --rule_input_manifest "${PREPARED_DIR}/rule_inputs/rule_input_manifest.json" \
    --output_dir "${RESULT_DIR}/scr_tpp" \
    --seeds "${SEED}" \
    --device cuda:0 \
    --sieve_batch_size 1 \
    --exact_batch_size "${EXACT_BATCH_SIZE:-1}"

  python -m workspace.benchmark_models.runners.realworld_rule_benchmarks \
    --rule_input_manifest "${PREPARED_DIR}/rule_inputs/rule_input_manifest.json" \
    --baseline_manifest workspace/benchmark_models/configs/baselines.official_or_faithful.yaml \
    --models CLNN,NSTPP \
    --output_dir "${RESULT_DIR}/logical_gpu0" \
    --device cuda:0 \
    --seeds "${SEED}"
) > "${RESULT_DIR}/logs/logical_gpu0.log" 2>&1 &
PID_LOGICAL_0=$!

(
  python -m workspace.benchmark_models.runners.realworld_rule_benchmarks \
    --rule_input_manifest "${PREPARED_DIR}/rule_inputs/rule_input_manifest.json" \
    --baseline_manifest workspace/benchmark_models/configs/baselines.official_or_faithful.yaml \
    --models CLUSTER,TELLER \
    --output_dir "${RESULT_DIR}/logical_gpu1" \
    --device cuda:1 \
    --seeds "${SEED}"
) > "${RESULT_DIR}/logs/logical_gpu1.log" 2>&1 &
PID_LOGICAL_1=$!

wait "${PID_LOGICAL_0}"
wait "${PID_LOGICAL_1}"

cat \
  "${RESULT_DIR}/scr_tpp/scr_tpp_results.jsonl" \
  "${RESULT_DIR}/logical_gpu0/rule_results.jsonl" \
  "${RESULT_DIR}/logical_gpu1/rule_results.jsonl" \
  > "${RESULT_DIR}/logical_train_results.jsonl"

python -m workspace.benchmark_models.evaluation.realworld_logical_prediction \
  --rule_input_manifest "${PREPARED_DIR}/rule_inputs/rule_input_manifest.json" \
  --rule_results_jsonl "${RESULT_DIR}/logical_train_results.jsonl" \
  --output_jsonl "${RESULT_DIR}/logical_prediction_results.jsonl" \
  --device cuda:0 \
  --grid_size "${GRID_SIZE}" \
  --prediction_grid_size "${PRED_GRID_SIZE}" \
  > "${RESULT_DIR}/logs/logical_prediction_eval.log" 2>&1

python -m workspace.benchmark_models.runners.realworld_prediction \
  --dataset mimic_iv \
  --target "${TARGET_LABEL}" \
  --input_pickle "${PREPARED_DIR}/split_seed${SEED}.pkl" \
  --output_dir "${RESULT_DIR}/easytpp_gpu0" \
  --prepare_easytpp \
  --prediction_manifest workspace/benchmark_models/configs/prediction_baselines.easytpp_official.yaml \
  --models RMTPP,NHP,SAHP \
  --prediction_results_jsonl "${RESULT_DIR}/easytpp_gpu0/prediction_results.jsonl" \
  --seed "${SEED}" \
  --gpu 0 \
  --max_epoch "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --target_prediction_grid_size "${PRED_GRID_SIZE}" \
  > "${RESULT_DIR}/logs/easytpp_gpu0.log" 2>&1 &
PID_EASY_0=$!

python -m workspace.benchmark_models.runners.realworld_prediction \
  --dataset mimic_iv \
  --target "${TARGET_LABEL}" \
  --input_pickle "${PREPARED_DIR}/split_seed${SEED}.pkl" \
  --output_dir "${RESULT_DIR}/easytpp_gpu1" \
  --prepare_easytpp \
  --prediction_manifest workspace/benchmark_models/configs/prediction_baselines.easytpp_official.yaml \
  --models THP,AttNHP,IntensityFree \
  --prediction_results_jsonl "${RESULT_DIR}/easytpp_gpu1/prediction_results.jsonl" \
  --seed "${SEED}" \
  --gpu 1 \
  --max_epoch "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --target_prediction_grid_size "${PRED_GRID_SIZE}" \
  > "${RESULT_DIR}/logs/easytpp_gpu1.log" 2>&1 &
PID_EASY_1=$!

wait "${PID_EASY_0}"
wait "${PID_EASY_1}"

cat \
  "${RESULT_DIR}/logical_prediction_results.jsonl" \
  "${RESULT_DIR}/easytpp_gpu0/prediction_results.jsonl" \
  "${RESULT_DIR}/easytpp_gpu1/prediction_results.jsonl" \
  > "${RESULT_DIR}/all_realworld_results.jsonl"

python -m workspace.benchmark_models.reports.summarize_benchmark_results \
  --realworld_jsonl "${RESULT_DIR}/all_realworld_results.jsonl" \
  --realworld_summary_csv "${RESULT_DIR}/realworld_summary.csv" \
  > "${RESULT_DIR}/logs/summarize.log" 2>&1

python -m workspace.benchmark_models.reports.write_realworld_benchmark_report \
  --results_jsonl "${RESULT_DIR}/all_realworld_results.jsonl" \
  --readiness_audit_json "${RESULT_DIR}/readiness_audit.json" \
  --output_md "${RESULT_DIR}/realworld_benchmark_report.md" \
  > "${RESULT_DIR}/logs/write_report.log" 2>&1

printf '%s\n' "${RESULT_DIR}" > data/realworld_results/latest_mimic_low_urine_result_dir.txt
echo "Real-world benchmark complete: ${RESULT_DIR}"

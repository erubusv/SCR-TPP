#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"

raw_xes="${RAW_XES_GZ:-data/realworld_raw/bpi_2017/BPI_Challenge_2017.xes.gz}"
prepared_dir="${PREPARED_DIR:-data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000}"
seed="${SEED:-111}"
timestamp="$(date +%Y%m%d_%H%M%S)"
result_dir="${RESULT_DIR:-data/realworld_results/reproduce_bpi2017_o_accepted_seed${seed}_${timestamp}}"

if [[ ! -f "${raw_xes}" ]]; then
  echo "missing required BPI 2017 file: ${raw_xes}" >&2
  exit 1
fi

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  python -m workspace.benchmark_models.data.prepare_bpi2017_o_accepted \
    --raw_xes_gz "${raw_xes}" \
    --output_dir "${prepared_dir}" \
    --target_activity O_Accepted \
    --seed "${seed}" \
    --train_size 5000 \
    --val_size 1000 \
    --test_size 1000

  python -m workspace.benchmark_models.data.prepare_realworld_rule_inputs \
    --sequence_pickle "${prepared_dir}/sequences.pkl" \
    --split_pickle "${prepared_dir}/split_seed${seed}.pkl" \
    --output_dir "${prepared_dir}/rule_inputs" \
    --dataset_name bpi2017_o_accepted_mixed_random_5000 \
    --target_event_id 0 \
    --target_event_label O_Accepted \
    --seeds "${seed}"

  python -m workspace.benchmark_models.data.audit_realworld_benchmark_readiness \
    --prepared_dir "${prepared_dir}" \
    --split_pickle "${prepared_dir}/split_seed${seed}.pkl" \
    --rule_input_manifest "${prepared_dir}/rule_inputs/rule_input_manifest.json" \
    --prediction_manifest workspace/benchmark_models/configs/prediction_baselines.easytpp_official.yaml \
    --rule_baseline_manifest workspace/benchmark_models/configs/baselines.official_or_faithful.yaml \
    --output_json "${prepared_dir}/readiness_benchmark_audit.json"
fi

PREPARED_DIR="${prepared_dir}" \
RESULT_DIR="${result_dir}" \
SEED="${seed}" \
EPOCHS="${EPOCHS:-20}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
GRID_SIZE="${GRID_SIZE:-96}" \
PRED_GRID_SIZE="${PRED_GRID_SIZE:-64}" \
MAX_LEN="${MAX_LEN:-64}" \
bash workspace/benchmark_models/scripts/run_bpi2017_o_accepted_full_realworld.sh

echo "[bpi] results: ${result_dir}"

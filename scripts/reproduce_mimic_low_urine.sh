#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"

raw_dir="${RAW_DIR:-data/realworld_raw/mimic_iv}"
prepared_full="${PREPARED_FULL_DIR:-data/realworld_prepared/mimic_low_urine}"
prepared_split="${PREPARED_DIR:-data/realworld_prepared/mimic_low_urine_random_5000}"
seed="${SEED:-111}"
timestamp="$(date +%Y%m%d_%H%M%S)"
result_dir="${RESULT_DIR:-data/realworld_results/reproduce_mimic_low_urine_seed${seed}_${timestamp}}"

required=(
  "${raw_dir}/icustays.csv"
  "${raw_dir}/chartevents.csv"
  "${raw_dir}/inputevents.csv"
  "${raw_dir}/procedureevents.csv"
  "${raw_dir}/outputevents.csv"
  "${raw_dir}/d_items.csv"
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required MIMIC-IV file: ${path}" >&2
    exit 1
  fi
done

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  python -m workspace.benchmark_models.data.prepare_mimic_low_urine \
    --raw_dir "${raw_dir}" \
    --prepared_dir "${prepared_full}" \
    --chunksize "${CHUNKSIZE:-1000000}" \
    --min_los_hours 6.0 \
    --max_los_hours 336.0 \
    --max_event_time_hours 336.0 \
    --split_seed "${seed}" \
    --train_ratio 0.7 \
    --dev_ratio 0.1

  python -m workspace.benchmark_models.data.prepare_mimic_random_subset_splits \
    --source_pickle "${prepared_full}/sequences.pkl" \
    --output_dir "${prepared_split}" \
    --seeds "${seed}" \
    --train_size 5000 \
    --val_size 1000 \
    --test_size 1000
fi

PREPARED_DIR="${prepared_split}" \
RESULT_DIR="${result_dir}" \
SEED="${seed}" \
TARGET_ID=0 \
TARGET_LABEL=low_urine_output \
EPOCHS="${EPOCHS:-20}" \
BATCH_SIZE="${BATCH_SIZE:-256}" \
GRID_SIZE="${GRID_SIZE:-96}" \
PRED_GRID_SIZE="${PRED_GRID_SIZE:-64}" \
bash workspace/benchmark_models/scripts/run_mimic_low_urine_full_realworld.sh

echo "[mimic] results: ${result_dir}"

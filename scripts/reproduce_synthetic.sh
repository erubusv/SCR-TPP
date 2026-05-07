#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"

timestamp="$(date +%Y%m%d_%H%M%S)"
result_root="${RESULT_ROOT:-data/paper_suite/results/reproduce_synthetic_${timestamp}}"
config_root="${CONFIG_ROOT:-data/paper_suite/configs/hetero_source_2000_adjusted}"
dataset_root="${DATASET_ROOT:-data/paper_suite/datasets/hetero_source_2000_adjusted}"
seeds_csv="${SEEDS:-111,222,333}"
models="${BASELINE_MODELS:-TELLER,CLNN,NSTPP,CLUSTER}"
device="${BASELINE_DEVICE:-cuda:0}"
gpu_id="${CUDA_DEVICE:-0}"

mkdir -p "${result_root}/scr_tpp" "${result_root}/logs"
printf '%s\n' "${result_root}" > data/paper_suite/results/latest_synthetic_reproduce_dir.txt

IFS=',' read -r -a seeds <<< "${seeds_csv}"

if [[ "${RUN_SCR_TPP:-1}" == "1" ]]; then
  for seed in "${seeds[@]}"; do
    echo "[synthetic] SCR-TPP seed=${seed}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHON_BIN="${PYTHON_BIN:-python}" \
    CPU_THREADS="${OMP_NUM_THREADS}" \
    workspace/train/paper_benchmark_active/run_benchmarks.sh \
      --dataset_seed "${seed}" \
      --result_path "${result_root}/scr_tpp/scr_tpp_seed${seed}.json" \
      > "${result_root}/logs/scr_tpp_seed${seed}.log" 2>&1
  done

  python -m workspace.benchmark_models.runners.synthetic_rule_recovery \
    --models ours \
    --config_root "${config_root}" \
    --dataset_root "${dataset_root}" \
    --seeds "${seeds_csv}" \
    --ours_results_dir "${result_root}/scr_tpp" \
    --out_jsonl "${result_root}/ours.jsonl" \
    --out_summary_csv "${result_root}/ours_summary.csv"
fi

if [[ "${RUN_BASELINES:-1}" == "1" ]]; then
  python -m workspace.benchmark_models.runners.synthetic_rule_recovery \
    --models "${models}" \
    --config_root "${config_root}" \
    --dataset_root "${dataset_root}" \
    --seeds "${seeds_csv}" \
    --baseline_manifest workspace/benchmark_models/configs/baselines.official_or_faithful.yaml \
    --device "${device}" \
    --out_jsonl "${result_root}/baselines.jsonl" \
    --out_summary_csv "${result_root}/baselines_summary.csv"
fi

if [[ -f "${result_root}/ours.jsonl" && -f "${result_root}/baselines.jsonl" ]]; then
  cat "${result_root}/ours.jsonl" "${result_root}/baselines.jsonl" > "${result_root}/all_models.jsonl"
  python -m workspace.benchmark_models.reports.summarize_benchmark_results \
    --synthetic_jsonl "${result_root}/all_models.jsonl" \
    --synthetic_summary_csv "${result_root}/all_models_summary.csv"
fi

echo "[synthetic] results: ${result_root}"

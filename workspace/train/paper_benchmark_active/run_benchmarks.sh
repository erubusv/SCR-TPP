#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
default_python_bin="${HOME}/miniconda3/envs/lob_rep/bin/python"
python_bin="${PYTHON_BIN:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ -x "${default_python_bin}" ]]; then
    python_bin="${default_python_bin}"
  else
    python_bin="python"
  fi
fi

cpu_threads="${CPU_THREADS:-}"
if [[ -z "${cpu_threads}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    cpu_threads=$(( $(nproc) / 2 ))
  else
    cpu_threads=1
  fi
  if [[ "${cpu_threads}" -lt 1 ]]; then
    cpu_threads=1
  fi
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$cpu_threads}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

exec "${python_bin}" "${repo_root}/workspace/train/paper_benchmark_active/run_paper_benchmarks.py" \
  --cpu_threads "$OMP_NUM_THREADS" \
  --device "${BENCHMARK_DEVICE:-auto}" \
  "$@"

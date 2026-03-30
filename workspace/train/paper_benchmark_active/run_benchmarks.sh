#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/workspace
exec python /workspace/workspace/train/paper_benchmark_active/run_paper_benchmarks.py "$@"

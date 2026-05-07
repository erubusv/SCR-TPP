# Paper Benchmark Active Path

This directory contains the active SCR-TPP training and evaluation code.

## Required Files

- [run_paper_benchmarks.py](/workspace/workspace/train/paper_benchmark_active/run_paper_benchmarks.py)
  - benchmark runner and paper algorithm profile defaults
- [run_paper_benchmarks_real_world.py](/workspace/workspace/train/paper_benchmark_active/run_paper_benchmarks_real_world.py)
  - real-world SCR-TPP runner used by the benchmark harness
- [run_fixed_support_kernel_mle.py](/workspace/workspace/train/paper_benchmark_active/run_fixed_support_kernel_mle.py)
  - fixed-support final kernel-shape MLE for synthetic profile metrics
- [rule_dependent_kernel_active_set.py](/workspace/workspace/train/paper_benchmark_active/rule_dependent_kernel_active_set.py)
  - exact support refits, batched refits, canonical likelihood, and BIC utilities
- [conjunctive_rule_initializer.py](/workspace/workspace/train/paper_benchmark_active/conjunctive_rule_initializer.py)
  - strict-AND witness features and rule/kernel initialization utilities
- [runtime_resources.py](/workspace/workspace/train/paper_benchmark_active/runtime_resources.py)
  - CPU/GPU thread-resource configuration
- [run_benchmarks.sh](/workspace/workspace/train/paper_benchmark_active/run_benchmarks.sh)
  - thin shell wrapper around `run_paper_benchmarks.py`

## Algorithm

The default paper path runs:

1. all-data synthetic likelihood-context construction
2. sieve-priced column generation with deterministic `(order, sign)` strata
3. free rule-source kernel polish
4. exact pricing rescue
5. exact drop certificate plus deterministic same-sign one-source extension certificate
6. exact one-drop/one-add swap certificate
7. numerical zero cleanup and final rule/kernel recovery reporting

The default data/config suite is `hetero_source_2000_adjusted`.

## Run

From the repo root:

```bash
workspace/train/paper_benchmark_active/run_benchmarks.sh
```

Or manually:

```bash
PYTHONPATH=/workspace python workspace/train/paper_benchmark_active/run_paper_benchmarks.py
```

Synthetic dataset generation remains CPU-side; fitting uses CUDA when available.

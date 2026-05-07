# Synthetic Benchmark Reproduction Arguments

This file records the arguments and fixed parameters needed to reproduce the latest synthetic rule-recovery benchmark with this repository.

## Dataset

| Field | Value |
|---|---|
| Config root | `data/paper_suite/configs/hetero_source_2000_adjusted` |
| Dataset root | `data/paper_suite/datasets/hetero_source_2000_adjusted` |
| Seeds | `111,222,333` |
| Split policy | All generated synthetic sequences are used as train data for rule discovery. No train/validation/test split is used for synthetic rule recovery. |
| Match key | `(source set, sign, target)` |
| Secondary overlap metric | Maximum-weight one-to-one source-set Jaccard matching, plus sign-aware Jaccard |

## Synthetic Configs

| Dataset |
|---|
| `logical_clean_plus` |
| `logical_shared` |
| `logical_context` |
| `kernel_triangular` |
| `kernel_exponential` |
| `kernel_gaussian` |
| `num_predicates_10` |
| `num_predicates_20` |
| `ablation_excitation_only` |
| `ablation_inhibition_only` |
| `ablation_mixed_sign` |

## SCR-TPP Parameters

SCR-TPP synthetic runs use `workspace/train/paper_benchmark_active/run_paper_benchmarks.py`.

| Parameter | Value |
|---|---|
| `FINAL_MAX_RULE_ORDER` | `3` |
| `FINAL_SIEVE_STEPS` | `20` |
| `FINAL_SIEVE_BATCH_SIZE` | `16` |
| `_FINAL_SCORE_NUM_KNOTS` | `7` |
| `FINAL_KERNEL_SMOOTHNESS_RIDGE` | `1e-3` |
| `FINAL_SUPPORT_BETA_THRESHOLD` | `1e-4` |
| `FINAL_REPORT_KERNEL_DF` | `True` |
| Device default | `auto`, resolved to `cuda:0` if available |
| Thread default | `run_benchmarks.sh` uses `CPU_THREADS`, otherwise half of `nproc` |

## SCR-TPP Command

Run one seed at a time. Use separate GPUs by setting `CUDA_VISIBLE_DEVICES`.

```bash
PYTHONPATH=/workspace \
PYTHON_BIN=python \
CPU_THREADS=12 \
CUDA_VISIBLE_DEVICES=0 \
workspace/train/paper_benchmark_active/run_benchmarks.sh \
  --dataset_seed 111 \
  --result_path data/paper_suite/results/scr_tpp_synthetic_latest/seed111.json
```

Repeat with `--dataset_seed 222` and `--dataset_seed 333`.

Import SCR-TPP rows into the common benchmark schema:

```bash
python -m workspace.benchmark_models.runners.synthetic_rule_recovery \
  --models ours \
  --config_root data/paper_suite/configs/hetero_source_2000_adjusted \
  --dataset_root data/paper_suite/datasets/hetero_source_2000_adjusted \
  --seeds 111,222,333 \
  --ours_results_dir data/paper_suite/results/scr_tpp_synthetic_latest \
  --out_jsonl data/paper_suite/results/scr_tpp_synthetic_latest/ours.jsonl \
  --out_summary_csv data/paper_suite/results/scr_tpp_synthetic_latest/ours_summary.csv
```

## Rule Baseline Manifest

Rule baselines are defined in `workspace/benchmark_models/configs/baselines.official_or_faithful.yaml`.

| Model | Implementation type | Source | Key arguments |
|---|---|---|---|
| `TELLER` | `official_equivalent_accelerated_teller` | `https://github.com/FengMingquan-sjtu/Logic_Point_Processes_ICLR` | `--algorithm REFS`, `--time_limit 600`, `--worker_num 4`, `--num_epoch 5`, `--feature_cache_entries 200000`, `--exclude_target_from_sources` |
| `CLNN` | `restricted_paper_based_reimplementation_weighted_clock_logic_tpp` | `https://openreview.net/forum?id=YfUICnZMwk7` | `--grid_size 24`, `--epochs 120`, `--num_formulas 8`, `--max_rules 8`, `--max_rule_length 3`, `--exclude_target_from_sources` |
| `NSTPP` | `restricted_paper_based_reimplementation_neural_symbolic_tpp` | `https://proceedings.mlr.press/v235/yang24ag.html` | `--grid_size 24`, `--epochs 120`, `--refine_epochs 60`, `--search_restarts 4`, `--max_rules 8`, `--max_rule_length 3`, `--exclude_target_from_sources` |
| `CLUSTER` | `restricted_paper_based_reimplementation_latent_causal_rule_em` | `https://proceedings.mlr.press/v238/kuang24a.html` | `--grid_size 24`, `--epochs 60`, `--em_iters 3`, `--max_rules 8`, `--max_rule_length 3`, `--dummy_count 3`, `--tau 20.0`, `--tau_min 0.1`, `--softmin_rho 0.1`, `--gamma_threshold 0.0`, `--exclude_target_from_sources` |

## Rule Baseline Command

```bash
python -m workspace.benchmark_models.runners.synthetic_rule_recovery \
  --models TELLER,CLNN,NSTPP,CLUSTER \
  --config_root data/paper_suite/configs/hetero_source_2000_adjusted \
  --dataset_root data/paper_suite/datasets/hetero_source_2000_adjusted \
  --seeds 111,222,333 \
  --baseline_manifest workspace/benchmark_models/configs/baselines.official_or_faithful.yaml \
  --out_jsonl data/paper_suite/results/synthetic_baselines_latest/all_baselines.jsonl \
  --out_summary_csv data/paper_suite/results/synthetic_baselines_latest/all_baselines_summary.csv
```

## Report Command

```bash
python -m workspace.benchmark_models.reports.write_synthetic_recovery_report \
  --rows_jsonl data/paper_suite/results/synthetic_baselines_latest/all_baselines.jsonl \
  --config_root data/paper_suite/configs/hetero_source_2000_adjusted \
  --result_root data/paper_suite/results/synthetic_baselines_latest \
  --manifest workspace/benchmark_models/configs/baselines.official_or_faithful.yaml \
  --out_md workspace/train/research_docs/benchmark_results/synthetic_benchmark_latest.md
```

## Current Result Pointers

| Artifact | Path |
|---|---|
| Latest synthetic result pointer | `data/paper_suite/results/latest_synthetic_result_dir.txt` |
| Current TELLER restore run root | `data/paper_suite/results/teller_synthetic_latest_20260507` |


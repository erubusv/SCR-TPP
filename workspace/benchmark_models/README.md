# Benchmark Models

Benchmark code is grouped by role. Generated data, logs, figures, and result
tables live under `data/` and are ignored by git.

## Layout

- `scr_tpp/`: our SCR-TPP benchmark entrypoints and result importers.
- `baselines/`: TELLER, CLNN, NSTPP, CLUSTER, and EasyTPP wrappers.
- `runners/`: synthetic and real-world benchmark drivers.
- `data/`: MIMIC-IV, BPI 2017, and generic rule-input preparation scripts.
- `adapters/`: sequence, EasyTPP, and synthetic data format adapters.
- `evaluation/`: target-event NLL/MAE/RMSE evaluation for logical TPP models.
- `reports/`: CSV/JSONL aggregation and benchmark report writers.
- `core/`: shared schemas, manifest runners, and summary utilities.
- `configs/`: baseline command manifests.
- `scripts/`: end-to-end real-world shell runners.

## SCR-TPP

The model implementation used by these benchmarks is in:

- `workspace/train/paper_benchmark_active/`

The benchmark-facing SCR-TPP wrappers are:

- `scr_tpp/run_realworld.py`
- `scr_tpp/synthetic_results.py`
- `scr_tpp/plot_learned_distributions.py`

## Synthetic Rule Recovery

Import SCR-TPP synthetic outputs:

```bash
python -m workspace.benchmark_models.runners.synthetic_rule_recovery \
  --models ours \
  --ours_results_dir data/paper_suite/results/<scr_tpp_synthetic_result_dir> \
  --out_jsonl data/paper_suite/results/benchmark_synthetic_ours.jsonl \
  --out_summary_csv data/paper_suite/results/benchmark_synthetic_ours_summary.csv
```

Run logical baselines:

```bash
python -m workspace.benchmark_models.runners.synthetic_rule_recovery \
  --models TELLER,CLNN,NSTPP,CLUSTER \
  --baseline_manifest workspace/benchmark_models/configs/baselines.official_or_faithful.yaml \
  --out_jsonl data/paper_suite/results/logic_baselines.jsonl \
  --out_summary_csv data/paper_suite/results/logic_baselines_summary.csv
```

## Real-World Benchmarks

MIMIC-IV LowUrine:

```bash
bash workspace/benchmark_models/scripts/run_mimic_low_urine_full_realworld.sh
```

BPI 2017 `O_Accepted`:

```bash
bash workspace/benchmark_models/scripts/run_bpi2017_o_accepted_full_realworld.sh
```

Both scripts train SCR-TPP, logical TPP baselines, and EasyTPP baselines on the
same train/dev/test split, then write a single `all_realworld_results.jsonl`.

## Prepared Data

- Synthetic configs: `data/paper_suite/configs/hetero_source_2000_adjusted/`
- MIMIC-IV prepared split: `data/realworld_prepared/mimic_low_urine_random_5000/`
- BPI 2017 prepared split: `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/`

## Result Locations

- Synthetic SCR-TPP and baseline outputs:
  `data/paper_suite/results/`
- MIMIC-IV real-world outputs:
  `data/realworld_results/mimic_low_urine_random5000_seed111_20260506_060453/`
- BPI 2017 real-world outputs:
  `data/realworld_results/bpi2017_o_accepted_mixed_random5000_seed111_20260507_040210/`

These directories are ignored by git.

## Checks

```bash
python -m py_compile $(find workspace/benchmark_models -name '*.py' -type f | sort)
python -m workspace.benchmark_models.tests
bash -n workspace/benchmark_models/scripts/run_mimic_low_urine_full_realworld.sh \
  workspace/benchmark_models/scripts/run_bpi2017_o_accepted_full_realworld.sh
```

# MIMIC-IV LowUrine Reproduction Arguments

This file records the arguments and fixed parameters needed to reproduce the latest MIMIC-IV LowUrine real-world benchmark.

## Raw And Prepared Data

| Field | Value |
|---|---|
| Raw data root | `data/realworld_raw/mimic_iv` |
| Prepared root | `data/realworld_prepared/mimic_low_urine_random_5000` |
| Full prepared source pickle | `data/realworld_prepared/mimic_low_urine/sequences.pkl` |
| Random split pickle | `data/realworld_prepared/mimic_low_urine_random_5000/split_seed111.pkl` |
| Rule input manifest | `data/realworld_prepared/mimic_low_urine_random_5000/rule_inputs/rule_input_manifest.json` |
| Result root used for latest report | `data/realworld_results/mimic_low_urine_random5000_seed111_20260506_060453` |
| Seed | `111` |
| Split | `5000` train, `1000` validation, `1000` test sequences |
| Split policy | Sequence-level uniform random fixed-size split |
| Target event | `low_urine_output` |
| Target id | `0` |
| Time unit | Hours since ICU intime |
| Event type count | `12` |

## LowUrine Construction

The event construction script is `workspace/benchmark_models/data/prepare_mimic_low_urine.py`.

| Parameter | Value |
|---|---|
| `--raw_dir` | `data/realworld_raw/mimic_iv` |
| `--prepared_dir` | `data/realworld_prepared/mimic_low_urine` |
| `--chunksize` | `1000000` |
| `--min_los_hours` | `6.0` |
| `--max_los_hours` | `336.0` |
| `--max_event_time_hours` | `336.0` |
| `--split_seed` | `111` |
| `--train_ratio` | `0.7` |
| `--dev_ratio` | `0.1` |
| Low urine window | `6.0` hours |
| Low urine threshold | `< 0.5 ml/kg/hour` |
| RRT handling | Events after RRT/dialysis are censored for LowUrine construction |

Source predicates are fixed in `SOURCE_LABELS`:

| Source id | Source predicate |
|---:|---|
| 1 | `low_systolic_bp` |
| 2 | `low_map` |
| 3 | `high_heart_rate` |
| 4 | `high_respiratory_rate` |
| 5 | `low_spo2` |
| 6 | `high_fio2` |
| 7 | `fever` |
| 8 | `vasopressor` |
| 9 | `crystalloid_fluid` |
| 10 | `albumin_colloid` |
| 11 | `rrt_started` |

## Random 5000 Split Construction

The random split script is `workspace/benchmark_models/data/prepare_mimic_random_subset_splits.py`.

```bash
python -m workspace.benchmark_models.data.prepare_mimic_random_subset_splits \
  --source_pickle data/realworld_prepared/mimic_low_urine/sequences.pkl \
  --output_dir data/realworld_prepared/mimic_low_urine_random_5000 \
  --seeds 111 \
  --train_size 5000 \
  --val_size 1000 \
  --test_size 1000
```

## Rule Input Conversion

The benchmark script regenerates the train-only logical-rule input before running models.

| Argument | Value |
|---|---|
| Module | `workspace.benchmark_models.data.prepare_realworld_rule_inputs` |
| `--sequence_pickle` | `data/realworld_prepared/mimic_low_urine_random_5000/sequences.pkl` |
| `--split_pickle` | `data/realworld_prepared/mimic_low_urine_random_5000/split_seed111.pkl` |
| `--output_dir` | `data/realworld_prepared/mimic_low_urine_random_5000/rule_inputs` |
| `--dataset_name` | `mimic_iv_low_urine` |
| `--target_event_id` | `0` |
| `--target_event_label` | `low_urine_output` |
| `--seeds` | `111` |
| `--max_lag_days` | `30.0` |
| `--train_ratio` | `0.7` |
| `--dev_ratio` | `0.1` |

## End-To-End Benchmark Script

Use:

```bash
bash workspace/benchmark_models/scripts/run_mimic_low_urine_full_realworld.sh
```

The script default parameters are:

| Parameter | Value |
|---|---|
| `PREPARED_DIR` | `data/realworld_prepared/mimic_low_urine_random_5000` |
| `SEED` | `111` |
| `TARGET_ID` | `0` |
| `TARGET_LABEL` | `low_urine_output` |
| `EPOCHS` | `20` |
| `BATCH_SIZE` | `256` |
| `GRID_SIZE` | `96` |
| `PRED_GRID_SIZE` | `64` |
| `OMP_NUM_THREADS` | `12` |
| `MKL_NUM_THREADS` | `12` |
| `OPENBLAS_NUM_THREADS` | `12` |
| `NUMEXPR_NUM_THREADS` | `12` |

## SCR-TPP Real-World Arguments

SCR-TPP is called through `workspace.benchmark_models.scr_tpp.run_realworld`.

| Argument | Value |
|---|---|
| `--rule_input_manifest` | `data/realworld_prepared/mimic_low_urine_random_5000/rule_inputs/rule_input_manifest.json` |
| `--seeds` | `111` |
| `--device` | `cuda:0` |
| `--sieve_batch_size` | `1` |
| `--sieve_steps` | `20` |
| `--exact_batch_size` | `1` |
| Internal max rule order | `3` |
| Internal beta threshold | `1e-4` |
| Internal kernel smoothness ridge | `1e-3` |

## Logical Rule Baselines

The same manifest is used as in the synthetic benchmark: `workspace/benchmark_models/configs/baselines.official_or_faithful.yaml`.

| Model | Device in script | Key arguments |
|---|---|---|
| `CLNN` | `cuda:0` | `--grid_size 24`, `--epochs 120`, `--num_formulas 8`, `--max_rules 8`, `--max_rule_length 3`, `--exclude_target_from_sources` |
| `NSTPP` | `cuda:0` | `--grid_size 24`, `--epochs 120`, `--refine_epochs 60`, `--search_restarts 4`, `--max_rules 8`, `--max_rule_length 3`, `--exclude_target_from_sources` |
| `CLUSTER` | `cuda:1` | `--grid_size 24`, `--epochs 60`, `--em_iters 3`, `--max_rules 8`, `--max_rule_length 3`, `--dummy_count 3`, `--tau 20.0`, `--tau_min 0.1`, `--softmin_rho 0.1`, `--gamma_threshold 0.0`, `--exclude_target_from_sources` |
| `TELLER` | `cuda:1` | `--algorithm REFS`, `--time_limit 600`, `--worker_num 4`, `--num_epoch 5`, `--feature_cache_entries 200000`, `--exclude_target_from_sources` |

## EasyTPP Neural Baselines

EasyTPP models are called through `workspace.benchmark_models.runners.realworld_prediction` and `workspace/benchmark_models/configs/prediction_baselines.easytpp_official.yaml`.

| Model group | GPU | Models |
|---|---:|---|
| Core recurrent/self-attentive group | 0 | `RMTPP,NHP,SAHP` |
| Transformer/intensity-free group | 1 | `THP,AttNHP,IntensityFree` |

| Argument | Value |
|---|---|
| `--dataset` | `mimic_iv` |
| `--target` | `low_urine_output` |
| `--input_pickle` | `data/realworld_prepared/mimic_low_urine_random_5000/split_seed111.pkl` |
| `--seed` | `111` |
| `--max_epoch` | `20` |
| `--batch_size` | `256` |
| `--learning_rate` | `0.001` |
| `--hidden_size` | `64` |
| `--time_emb_size` | `16` |
| `--num_layers` | `2` |
| `--num_heads` | `2` |
| `--mc_samples` | `20` |
| `--target_prediction_grid_size` | `64` |

## Prediction Metric Arguments

Logical models and SCR-TPP are evaluated by `workspace.benchmark_models.evaluation.realworld_logical_prediction`.

| Argument | Value |
|---|---|
| `--grid_size` | `96` |
| `--prediction_grid_size` | `64` |
| `--device` | `cuda:0` |
| Metrics | Target-event `NLL`, `MAE`, `RMSE`, training time, inference time |
| Prediction rule | For each target event, fix history before the event, evaluate future target intensity on a grid, integrate cumulative hazard, compute density `lambda(t) exp(-int lambda)`, and use `E[t]` as predicted target time. |

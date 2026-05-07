# BPI 2017 O_Accepted Reproduction Arguments

This file records the arguments and fixed parameters needed to reproduce the latest BPI 2017 `O_Accepted` real-world benchmark.

## Raw And Prepared Data

| Field | Value |
|---|---|
| Raw XES file | `data/realworld_raw/bpi_2017/BPI_Challenge_2017.xes.gz` |
| Prepared root | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000` |
| Sequence pickle | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/sequences.pkl` |
| Rule input manifest | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/rule_inputs/rule_input_manifest.json` |
| Result root used for latest report | `data/realworld_results/bpi2017_o_accepted_mixed_random5000_seed111_20260507_040210` |
| Seed | `111` |
| Split | `5000` train, `1000` validation, `1000` test cases |
| Split policy | Case-level uniform random fixed-size split |
| Target event | `O_Accepted` |
| Target id | `0` |
| Time unit | Days since first selected event in case |
| Event type count | `16` |
| Lifecycle filter | `complete` |

## Data Preparation Arguments

The preparation script is `workspace/benchmark_models/data/prepare_bpi2017_o_accepted.py`.

```bash
python -m workspace.benchmark_models.data.prepare_bpi2017_o_accepted \
  --raw_xes_gz data/realworld_raw/bpi_2017/BPI_Challenge_2017.xes.gz \
  --output_dir data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000 \
  --target_activity O_Accepted \
  --seed 111 \
  --train_size 5000 \
  --val_size 1000 \
  --test_size 1000
```

`--target_positive_only` is not used. The prepared data keeps both accepted and non-accepted cases.

## Source Predicates

| Source id | Source predicate |
|---:|---|
| 1 | `O_Create Offer` |
| 2 | `O_Created` |
| 3 | `O_Sent (mail and online)` |
| 4 | `A_Validating` |
| 5 | `A_Complete` |
| 6 | `O_Returned` |
| 7 | `A_Incomplete` |
| 8 | `A_Submitted` |
| 9 | `W_Complete application` |
| 10 | `W_Validate application` |
| 11 | `W_Handle leads` |
| 12 | `W_Call incomplete files` |
| 13 | `O_Sent (online only)` |
| 14 | `W_Call after offers` |
| 15 | `W_Assess potential fraud` |

## Rule Input Conversion

The prepared BPI folder contains the train-only logical-rule input used by SCR-TPP and logical baselines.

| Argument | Value |
|---|---|
| Module | `workspace.benchmark_models.data.prepare_realworld_rule_inputs` |
| `--sequence_pickle` | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/sequences.pkl` |
| `--split_pickle` | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/split_seed111.pkl` |
| `--output_dir` | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/rule_inputs` |
| `--dataset_name` | `bpi2017_o_accepted_mixed_random_5000` |
| `--target_event_id` | `0` |
| `--target_event_label` | `O_Accepted` |
| `--seeds` | `111` |
| `--max_lag_days` | `30.0` |
| `--train_ratio` | `0.7` |
| `--dev_ratio` | `0.1` |

```bash
python -m workspace.benchmark_models.data.prepare_realworld_rule_inputs \
  --sequence_pickle data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/sequences.pkl \
  --split_pickle data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/split_seed111.pkl \
  --output_dir data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/rule_inputs \
  --dataset_name bpi2017_o_accepted_mixed_random_5000 \
  --target_event_id 0 \
  --target_event_label O_Accepted \
  --seeds 111
```

The end-to-end BPI runner also requires a benchmark readiness audit:

```bash
python -m workspace.benchmark_models.data.audit_realworld_benchmark_readiness \
  --prepared_dir data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000 \
  --split_pickle data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/split_seed111.pkl \
  --rule_input_manifest data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/rule_inputs/rule_input_manifest.json \
  --prediction_manifest workspace/benchmark_models/configs/prediction_baselines.easytpp_official.yaml \
  --rule_baseline_manifest workspace/benchmark_models/configs/baselines.official_or_faithful.yaml \
  --output_json data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/readiness_benchmark_audit.json
```

## End-To-End Benchmark Script

Use:

```bash
bash workspace/benchmark_models/scripts/run_bpi2017_o_accepted_full_realworld.sh
```

The script default parameters are:

| Parameter | Value |
|---|---|
| `PREPARED_DIR` | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000` |
| `SEED` | `111` |
| `EPOCHS` | `20` |
| `BATCH_SIZE` | `64` |
| `GRID_SIZE` | `96` |
| `PRED_GRID_SIZE` | `64` |
| `MAX_LEN` | `64` |
| `BASELINE_MANIFEST` | `workspace/benchmark_models/configs/baselines.official_or_faithful.yaml` |
| `PREDICTION_MANIFEST` | `workspace/benchmark_models/configs/prediction_baselines.easytpp_official.yaml` |
| `OMP_NUM_THREADS` | `12` |
| `MKL_NUM_THREADS` | `12` |
| `OPENBLAS_NUM_THREADS` | `12` |
| `NUMEXPR_NUM_THREADS` | `12` |

## SCR-TPP Real-World Arguments

SCR-TPP is called through `workspace.benchmark_models.scr_tpp.run_realworld`.

| Argument | Value |
|---|---|
| `--rule_input_manifest` | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/rule_inputs/rule_input_manifest.json` |
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
| `--dataset` | `generic_pickle` |
| `--target` | `O_Accepted` |
| `--input_pickle` | `data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/sequences.pkl` |
| `--seed` | `111` |
| `--max_epoch` | `20` |
| `--batch_size` | `64` |
| `--learning_rate` | `0.001` |
| `--hidden_size` | `64` |
| `--time_emb_size` | `16` |
| `--num_layers` | `2` |
| `--num_heads` | `2` |
| `--mc_samples` | `20` |
| `--max_len` | `64` |
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

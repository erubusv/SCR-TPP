# SCR-TPP

## Introduction

SCR-TPP is a signed conjunctive-rule temporal point process for interpretable event-sequence modeling. The model searches for a compact set of excitation and inhibition rules, then assigns each rule source its own temporal kernel. The synthetic experiments evaluate rule recovery. The real-world experiments evaluate target-event prediction and learned rules on MIMIC-IV LowUrine and BPI 2017 `O_Accepted`.

The artifact is intended to run through Docker. Generated datasets, logs, figures, checkpoints, and result files are not committed.

## Repository Structure

```text
SCR-TPP/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── scripts/
│   ├── reproduce_synthetic.sh              # synthetic rule-recovery experiment
│   ├── reproduce_mimic_low_urine.sh        # MIMIC-IV LowUrine experiment
│   └── reproduce_bpi2017_o_accepted.sh     # BPI 2017 O_Accepted experiment
├── workspace/
│   ├── train/
│   │   └── paper_benchmark_active/         # SCR-TPP model and training code
│   └── benchmark_models/
│       ├── baselines/                      # TELLER, CLNN, NSTPP, CLUSTER, EasyTPP wrappers
│       ├── data/                           # real-world preprocessing and audits
│       ├── evaluation/                     # target-event NLL, MAE, RMSE evaluation
│       ├── reports/                        # result aggregation and markdown reports
│       ├── runners/                        # synthetic and real-world benchmark drivers
│       ├── scr_tpp/                        # SCR-TPP benchmark-facing wrappers
│       └── reproduce/                      # exact arguments used for reported runs
├── data/
│   ├── paper_suite/
│   │   ├── configs/hetero_source_2000_adjusted/   # committed synthetic configs
│   │   ├── datasets/                              # generated, ignored by git
│   │   └── results/                               # generated, ignored by git
│   ├── realworld_raw/                             # raw MIMIC-IV/BPI files, ignored by git
│   ├── realworld_prepared/                        # prepared sequences, ignored by git
│   └── realworld_results/                         # benchmark outputs, ignored by git
└── external/
    └── official_baselines/                 # external baseline source snapshots
```

## How to Reproduce the Results

Build the Docker image:

```bash
docker compose build
```

Run a quick check:

```bash
docker compose run --rm scr-tpp bash -lc \
  "python -m py_compile \$(find workspace/benchmark_models workspace/train/paper_benchmark_active -name '*.py' -type f | sort) && python -m workspace.benchmark_models.tests"
```

Run the synthetic benchmark:

```bash
docker compose run --rm scr-tpp bash scripts/reproduce_synthetic.sh
```

Synthetic outputs are written to:

```text
data/paper_suite/results/reproduce_synthetic_<timestamp>/
```

For MIMIC-IV LowUrine, place the required raw files in:

```text
data/realworld_raw/mimic_iv/
  icustays.csv
  chartevents.csv
  inputevents.csv
  procedureevents.csv
  outputevents.csv
  d_items.csv
```

Then run:

```bash
docker compose run --rm scr-tpp bash scripts/reproduce_mimic_low_urine.sh
```

MIMIC-IV outputs are written to:

```text
data/realworld_results/reproduce_mimic_low_urine_seed111_<timestamp>/
```

For BPI 2017, place the XES file in:

```text
data/realworld_raw/bpi_2017/BPI_Challenge_2017.xes.gz
```

Then run:

```bash
docker compose run --rm scr-tpp bash scripts/reproduce_bpi2017_o_accepted.sh
```

BPI outputs are written to:

```text
data/realworld_results/reproduce_bpi2017_o_accepted_seed111_<timestamp>/
```

The exact fixed arguments used by these scripts are documented in:

```text
workspace/benchmark_models/reproduce/synthetic_reproduction_arguments.md
workspace/benchmark_models/reproduce/mimic_low_urine_reproduction_arguments.md
workspace/benchmark_models/reproduce/bpi2017_o_accepted_reproduction_arguments.md
```

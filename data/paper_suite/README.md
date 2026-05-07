# Paper Suite Layout

This directory holds reproducible synthetic benchmark configs. No generated
dataset is committed. `scripts/reproduce_synthetic.sh` regenerates the datasets
from these configs before running the models. Generated datasets, logs, and
result tables are local artifacts and are ignored by git.

## Structure

- `configs/hetero_source_2000_adjusted/`
  - main synthetic benchmark configs
  - one file per benchmark, for example `logical_context.yaml`
  - canonical log-link generation and learning
  - hetero-source rule-source kernels
- `configs/hetero_source_500_adjusted/`
  - data-size ablation configs
- `configs/hetero_source_1000_adjusted/`
  - data-size ablation configs
- `datasets/`
  - generated locally by `scripts/reproduce_synthetic.sh`
- `results/` and `logs/`
  - generated locally by benchmark runners

## Naming

- benchmark names are short and stable:
  - `logical_context`
  - `logical_clean_plus`
  - `kernel_gaussian`
  - `num_predicates_20`
- seed identity is carried by the folder:
  - `datasets/hetero_source_2000_adjusted/seed_111/...`
  - `datasets/hetero_source_2000_adjusted/seed_222/...`

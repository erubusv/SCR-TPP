# Paper Suite Layout

This directory is organized so the path itself explains both the benchmark name
and the seed.

## Structure

- `configs/final_logical_tpp/`
  - final human-readable benchmark configs
  - one file per benchmark, for example `logical_context.yaml`
  - these keep intuitive raw-scale `base_intensity`, `W_pos`, and `W_neg`
  - generation and learning are both canonical log-link with
    `synthetic_generation_model: canonical_loglink`
    and `intensity_model: canonical_loglink`
- `datasets/final_logical_tpp/seed_111/`
- `datasets/final_logical_tpp/seed_222/`
- `datasets/final_logical_tpp/seed_333/`
- `datasets/final_logical_tpp/seed_444/`
- `datasets/final_logical_tpp/seed_555/`
  - generated dataset pickles for each benchmark/seed pair
  - filenames are short benchmark names such as `logical_context.pkl`
- `results/`
  - benchmark summary outputs written by the active runner

## Naming

- benchmark names are short and stable:
  - `logical_context`
  - `logical_clean_plus`
  - `kernel_gaussian`
  - `num_predicates_20`
- seed identity is carried by the folder:
  - `datasets/final_logical_tpp/seed_111/...`
  - `datasets/final_logical_tpp/seed_222/...`

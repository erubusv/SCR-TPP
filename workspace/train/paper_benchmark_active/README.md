# Paper Benchmark Active Path

This folder groups the files that are actually needed to run the current
synthetic paper benchmark path.

## Active Files

- [run_paper_benchmarks.py](/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/run_paper_benchmarks.py)
  - benchmark runner for the final `final_logical_tpp` suite
- [rule_dependent_kernel_active_set.py](/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/rule_dependent_kernel_active_set.py)
  - retained mainline learner and final CLI
- [conjunctive_rule_initializer.py](/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/conjunctive_rule_initializer.py)
  - feature/kernel initialization utilities used by the learner

## Final Model

The final synthetic benchmark path is fixed to:

1. global source-kernel initialization
2. raw active-set proposal
3. `family_attribution_refine` with `1` pass
4. fair rule-count BIC post-prune with
   - `penalty_scale = 1.0`
   - `min_order = 1`
   - `max_drop_size = 2`
5. same-sign overlap component exact subset search

This directory no longer keeps experimental branches for:

- penalty-scale grid search
- intensity-model override at evaluation time
- cross-sign / canonical / signed-chart selector variants

## Supporting Docs

- [paper_synthetic_benchmark_plan.md](/home/yangmg1216/hnstpp/workspace/train/research_docs/paper_synthetic_benchmark_plan.md)
- [rule_dependent_kernel_research_notes.md](/home/yangmg1216/hnstpp/workspace/train/research_docs/rule_dependent_kernel_research_notes.md)

Canonical document location:

- [research_docs](/home/yangmg1216/hnstpp/workspace/train/research_docs)

## Notes

- This folder now contains the actual active benchmark code files.
- The current best synthetic benchmark path does not depend on the removed
  legacy scripts.

## Run

From anywhere:

```bash
/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/run_benchmarks.sh
```

By default, benchmark fitting now uses:

- `cuda:0` when CUDA is available
- `cpu` otherwise

Synthetic dataset generation remains CPU-side because the event simulation path
is not torch-accelerated.

Or manually:

```bash
PYTHONPATH=/home/yangmg1216/hnstpp python /home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/run_paper_benchmarks.py
```

## Paper Suite Layout

- configs:
  - [final_logical_tpp](/home/yangmg1216/hnstpp/data/paper_suite/configs/final_logical_tpp)
- datasets:
  - [final_logical_tpp](/home/yangmg1216/hnstpp/data/paper_suite/datasets/final_logical_tpp)
- results:
  - [results](/home/yangmg1216/hnstpp/data/paper_suite/results)

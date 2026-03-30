# Paper Benchmark Active Path

This folder groups the files that are actually needed to run the current
synthetic paper benchmark path.

## Active Files

- [run_paper_benchmarks.py](/workspace/workspace/train/paper_benchmark_active/run_paper_benchmarks.py)
  - benchmark runner
- [rule_dependent_kernel_active_set.py](/workspace/workspace/train/paper_benchmark_active/rule_dependent_kernel_active_set.py)
  - retained mainline learner
- [conjunctive_rule_initializer.py](/workspace/workspace/train/paper_benchmark_active/conjunctive_rule_initializer.py)
  - feature/kernel initialization utilities used by the learner

## Supporting Docs

- [paper_synthetic_benchmark_plan.md](/workspace/workspace/train/research_docs/paper_synthetic_benchmark_plan.md)
- [rule_dependent_kernel_research_notes.md](/workspace/workspace/train/research_docs/rule_dependent_kernel_research_notes.md)

Canonical document location:

- [research_docs](/workspace/workspace/train/research_docs)

## Notes

- This folder now contains the actual active benchmark code files.
- The current best synthetic benchmark path does not depend on the removed
  legacy scripts.

## Run

From anywhere:

```bash
/workspace/workspace/train/paper_benchmark_active/run_benchmarks.sh
```

Or manually:

```bash
PYTHONPATH=/workspace python /workspace/workspace/train/paper_benchmark_active/run_paper_benchmarks.py
```

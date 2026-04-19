# Rule-Dependent Kernel Project Handoff

This note is meant to let the next Codex continue work immediately without
reconstructing the full conversation history.

## 1. Non-Negotiable Ground Truth

Only the **canonical-to-canonical** setting is valid now.

- synthetic generation:
  - `lambda_T(t) = b_T * exp(E_T(t) - I_T(t))`
- learning:
  - `mu * exp(E - I)`
- activation:
  - `product_bounded`
  - source factor `1 - exp(-z)`
  - rule activation = product of source factors

Do **not** treat old mismatched experiments as valid evidence.

Invalid old line:

- generated with multiplicative/exact-log-rewrite semantics
- learned with canonical log-link

That line is now only historical diagnosis, not valid benchmark evidence.

## 2. Canonical Data / Config State

Final configs:

- `/home/yangmg1216/hnstpp/data/paper_suite/configs/final_logical_tpp`

Current stored canonical datasets:

- `/home/yangmg1216/hnstpp/data/paper_suite/datasets/final_logical_tpp/seed_111`

Benchmark family:

- 11 datasets
- 5000 sequences each
- split:
  - train 4000
  - val 500
  - test 500
- total currently materialized sequences:
  - `11 * 5000 = 55,000`

Important structural fact:

- all rules point to a single target
- the target never appears as a source
- non-target types are baseline-only

This is why the current canonical generator fast path is exact, not heuristic.

## 3. Current Data Size

Current canonical dataset directory size:

- `/home/yangmg1216/hnstpp/data/paper_suite/datasets/final_logical_tpp`
- about `163 MB`

Per-file sizes in `seed_111`:

- `ablation_excitation_only.pkl`: `17,712,892 B` (~16.9 MiB)
- `ablation_inhibition_only.pkl`: `13,333,087 B` (~12.7 MiB)
- `ablation_mixed_sign.pkl`: `15,269,580 B` (~14.6 MiB)
- `kernel_exponential.pkl`: `15,643,171 B` (~14.9 MiB)
- `kernel_gaussian.pkl`: `15,704,964 B` (~15.0 MiB)
- `kernel_triangular.pkl`: `15,269,578 B` (~14.6 MiB)
- `logical_clean_plus.pkl`: `20,223,746 B` (~19.3 MiB)
- `logical_context.pkl`: `13,166,799 B` (~12.6 MiB)
- `logical_shared.pkl`: `12,762,844 B` (~12.2 MiB)
- `num_predicates_10.pkl`: `16,019,269 B` (~15.3 MiB)
- `num_predicates_20.pkl`: `15,481,454 B` (~14.8 MiB)

Config directory:

- 11 YAML files
- each about `1.8K ~ 2.1K`

## 4. Best Valid Results So Far

### 4.1 Best confirmed general canonical branch

Directory:

- `/home/yangmg1216/hnstpp/data/paper_suite/results/canonical_profilegain_eval_20260418`

This branch changed canonical forward screening from the old quadratic
surrogate to **exact 1D profiled gain**.

Confirmed hard-case results:

- `ablation_inhibition_only`: `1.0 / 1.0`
- `ablation_excitation_only`: `1.0 / 0.75`
- `kernel_gaussian`: `1.0 / 0.6`
- `logical_context`: `1.0 / 0.7778`
- `logical_clean_plus`: `1.0 / 0.6`

Interpretation:

- recall collapse was mostly fixed
- remaining problem is mainly precision, i.e. shadow/overlap extra rules

### 4.2 Strongest positive precision result

Directory:

- `/home/yangmg1216/hnstpp/data/paper_suite/results/canonical_seedpartition_eval_20260419`

Confirmed result:

- `ablation_excitation_only`: `1.0 / 1.0`

This came from the new forward variant:

- `run_active_set_seed_partition(...)`

Key interpretation:

- at least in this benchmark, the precision bottleneck really was the first
  greedy merged seed
- exact competition over the seed-support partition family can fix it

## 5. What Has Been Shown Not To Help

These should not be resurrected casually unless there is a new mathematical
reason.

- data easing via target base intensity increase
- data easing via uniform rule-weight scaling
- full local-chart exact branch
- exact same-sign component refit as final precision fix
- cross-sign component competition as a blanket fix
- heavy fixed-feature backward exact branches
- multiplicative-inference rescue inside the canonical benchmark story

Several of these were either ineffective, too slow, or invalid under the
canonical benchmark semantics.

## 6. Current Hypothesis About the Precision Bottleneck

Current best explanation:

- the main precision bottleneck is not the last prune/component stage
- many extras are already effectively determined at the **forward support
  formation** stage
- specifically, a merged or overlap shadow seed can enter before the correct
  decomposition

Examples:

- `ablation_excitation_only`: `AB` enters before `A` and `B`
- `kernel_gaussian`: `ACD` can enter before a cleaner decomposition

So the current open question is:

- does the `seed_partition` forward repair generalize across the benchmark
  suite, or was the successful `ablation_excitation_only` result unusually
  favorable?

## 7. Current Code State

Main learner file:

- `/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/rule_dependent_kernel_active_set.py`

Main benchmark runner:

- `/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/run_paper_benchmarks.py`

Generator:

- `/home/yangmg1216/hnstpp/data/synthetic.py`

Research notes:

- `/home/yangmg1216/hnstpp/workspace/train/research_docs/rule_dependent_kernel_research_notes.md`

### Important current runner behavior

The runner now has a forward switch:

- `--forward_variant baseline`
- `--forward_variant seed_partition`

Default is still:

- `baseline`

So no accidental change to the baseline happens unless explicitly requested.

## 8. Exact, General Speed Improvements Already Applied

### Generator side

In `data/synthetic.py`:

- monotone in-place source-count advancement
- reuse of already-computed activations for auxiliary labels

These are exact.

### Learner side

In `optimize_active_set_torch(...)`:

1. validation checkpoints now compute tensor-only BIC first
   - only export NumPy arrays if BIC improves
2. rule feature assembly now uses:
   - prefetched source basis tensors
   - deterministic rule specs
   - incremental multiply instead of `stack(...).prod(...)`

These are exact:

- same objective
- same optimizer
- same checkpoint schedule
- fewer repeated allocations / fewer CPU exports

Small smoke test already passed after the learner-side speed patch.

## 9. Latest Batch Status

A full 11-benchmark `seed_partition` batch was launched and then intentionally
stopped before any benchmark completed.

Important:

- do **not** treat that directory as a completed result set
- it was an aborted run and its raw batch artifacts may be pruned during cleanup

There should currently be no active `eval_seedpart_*` tmux sessions from that
attempt.

## 10. Best Next Step

The next most valuable experiment is:

1. evaluate `seed_partition` forward on a **small but informative canonical
   subset**, not immediately full 11
2. recommended first subset:
   - `kernel_gaussian`
   - `logical_context`
   - `logical_clean_plus`
   - optionally re-check `ablation_excitation_only`
3. compare against:
   - baseline canonical `profilegain`

Why this is the best next step:

- it directly tests whether the positive `ablation_excitation_only` result is
  general
- it stays theorem-friendly
- it does not change the model class
- it targets the current best-supported bottleneck story

## 11. Guardrails For The Next Codex

- Keep updating:
  - `/home/yangmg1216/hnstpp/workspace/train/research_docs/rule_dependent_kernel_research_notes.md`
- Do not cite old mixed-semantics runs as valid benchmark performance
- Remove ineffective research branches from active code when clearly negative
- Prefer exact / deterministic / theorem-friendly changes over heuristic patching
- Be careful not to accidentally replace the default baseline runner behavior

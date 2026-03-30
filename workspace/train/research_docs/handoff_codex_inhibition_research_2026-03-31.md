# Codex Handoff: Inhibition / Family Search Research

This file summarizes the key methods tried, what was learned, the practical
state of the codebase, and the important experiment artifacts to keep.

## Current Mainline

Mainline code remains the stable baseline:

- `/workspace/workspace/train/paper_benchmark_active/rule_dependent_kernel_active_set.py`
- `/workspace/workspace/train/paper_benchmark_active/run_paper_benchmarks.py`

Shared utility cleanup completed:

- `/workspace/workspace/train/paper_benchmark_active/conjunctive_rule_initializer.py`
  now keeps only the utility/kernel-building layer used by the active runner
  and learner.
- Its old standalone family-search / local PP-model / CLI block was removed
  because it is no longer imported by the retained benchmark path.

No experimental method below has been merged into the mainline.

## Official Paper Benchmark State

Reference results:

- `/workspace/data/paper_suite/paper_benchmark_results.json`

Key outcomes:

- `logical_shared`: exact
- `logical_context`: exact
- `kernel_triangular / exponential / gaussian`: exact
- `num_predicates_10`: exact
- `num_predicates_20`: recall `5/6`, precision `1.0`, missing `DFG inh`
- `num_predicates_30`: recall `5/6`, precision `1.0`, missing `DFG inh`
- `ablation_inhibition_only` with `target=0.05`: recall `3/6`, precision `1.0`

## Most Important Conclusions

### 1. Pure inhibition is not impossible in principle

These two diagnostics show that official `inhibition_only 0.05` is not a
signal-free regime:

- `/workspace/data/paper_suite/inhibition_only_oracle_expected_support.json`
- `/workspace/data/paper_suite/inhibition_only_true_feature_observed_support.json`

In both cases, full truth support is preferred when true features are used.

Interpretation:

- the benchmark is recoverable in principle
- the observed failure is not explained by "no signal"
- the real bottleneck is `kernel learning + support search coupling`

### 2. Pair-level inhibition can be improved; triplet inhibition is still the bottleneck

The strongest non-mainline positive result was the staged hierarchical family
search on official `inhibition_only`:

- `/workspace/data/paper_suite/paper_ablation_inhibition_only_hierarchical_family_search.json`

It improved:

- baseline: recall `3/6`, precision `1.0`
- hierarchical family search: recall `4/6`, precision `1.0`

Recovered support:

- `A inh`
- `B inh`
- `CD inh`
- `EF inh`

Still missing:

- `CGH inh`
- `DFG inh`

### 3. This pair-level gain is real, but still not enough

Direct controls showed that `EF inh` was not just a trivially easy rule:

- `/workspace/data/paper_suite/paper_ablation_inhibition_only_hierarchical_controls.json`

Examples:

- `+EF` one-shot add alone did not improve the baseline
- `+DFG` alone did not improve the baseline
- simple forced family additions did not improve the baseline

So the hierarchical family search produced a real pair-level recovery effect,
not just an accidental rule that would have appeared anyway.

### 4. Oracle-free local confidence is still weak

Observable `delta=0.05` certificate:

- `/workspace/data/paper_suite/paper_ablation_inhibition_only_delta05_certificate.json`

Result:

- best and runner-up support gaps were smaller than the empirical Bernstein
  uncertainty bounds
- no candidate in the tested local neighborhood was certified as stably better

Interpretation:

- the current local candidate neighborhood is still too ambiguous
- improved support exists, but is not yet statistically robust under the tested
  oracle-free local certificate

### 5. Repeated-split stability confirms pair-level gains, not triplet recovery

Repeated random `80/20` resplits over the official `inhibition_only` train/val
pool:

- `/workspace/data/paper_suite/paper_ablation_inhibition_only_stability_selection.json`

Results over 5 repeats:

- baseline average recall / precision: `0.567 / 1.000`
- hierarchical average recall / precision: `0.733 / 1.000`

Rule frequencies:

- baseline:
  - `A inh`: `5/5`
  - `B inh`: `5/5`
  - `CD inh`: `3/5`
  - `EF inh`: `4/5`
- hierarchical:
  - `A inh`: `5/5`
  - `B inh`: `5/5`
  - `CD inh`: `5/5`
  - `EF inh`: `5/5`
  - `CGH inh`: `2/5`

At stability threshold `0.6`, both methods yield the same stable support:

- `{A, B, CD, EF}`

Interpretation:

- hierarchical family search genuinely stabilizes pair-level inhibition recovery
- it still does not stably recover either true triplet inhibition

### 6. `num_predicates_20/30` remain the hard scalability bottleneck

Ad hoc half-resource hierarchical runs:

- `/workspace/data/paper_suite/paper_num_predicates_20_hierarchical_family_search_half_resources.json`

Results:

- `num_predicates_20`
  - wall time: `504.656s`
  - no performance change
  - top triplet screening raised `DFG`, but family exact search still chose
    empty support, so no support change was accepted
- `num_predicates_30`
  - wall time before failure: `912.595s`
  - failed under half-RAM constraint with NumPy allocation error while building
    feature arrays

Interpretation:

- current hierarchical family search is too expensive for large predicate count
- even when it finishes on `20`, it does not fix the missing `DFG inh`

## Methods Tried And What They Taught Us

### Global lesson

The failure is not explained by one issue alone. The project converged on the
following decomposition:

- the objective can prefer truth in oracle feature space
- pure inhibition has weaker local separation than mixed-sign settings
- learned kernels can distort weak higher-order inhibition families
- greedy or approximate support updates then amplify that distortion

### Legacy Side Paths Not Used By The Current Best Synthetic Benchmark

Two older research directions were explored but are not
part of the current best synthetic benchmark path:

- `/workspace/workspace/train/ablate_custom_overlap.py`
- `/workspace/workspace/train/adaptive_multichannel_selector.py`

What they were for:

- `ablate_custom_overlap.py`
  - initializer-only / HNSTPP-side overlap experiments
  - depends on `wh_init` and `workspace.models.HNSTPP`
- `adaptive_multichannel_selector.py`
  - an older adaptive multichannel stagewise selector for overlap-rule recovery

What matters now:

- neither file is imported by the current paper benchmark path
- neither file is imported by the retained mainline learner
- the current benchmark/mainline dependency chain is:
  - `paper_benchmark_active/run_paper_benchmarks.py`
  - `paper_benchmark_active/rule_dependent_kernel_active_set.py`
  - `paper_benchmark_active/conjunctive_rule_initializer.py`

Practical conclusion:

- these two files represent explored but currently abandoned side directions
- they are not required to preserve the current best synthetic benchmark
  results
- they have now been removed from the working tree to keep the retained path
  smaller and less ambiguous

### Methods tried

#### 1. Rule-level one-rule fixes

Examples:

- candidate repricing
- parent/core-guided scoring
- cheap family ranking
- conditioned score tests
- sign/order stratified screening

Outcome:

- none solved triplet inhibition robustly
- most either did nothing or increased runtime without fixing recovery

#### 2. Joint sparse inhibition block

Examples:

- plain weighted `l1` joint sparse prototype
- convex family-sparse inhibition prototype

Outcome:

- both collapsed toward stronger singleton/pair surrogate inhibition
- neither recovered the true missing triplets

#### 3. Naive EM-like alternating `rule -> kernel -> rule`

Artifact:

- `/workspace/data/paper_suite/em_like_inhibition_only_results.json`

Outcome:

- quick baseline inside the prototype could reach `5/6`
- the EM-like alternation then chose a surrogate family and collapsed back to
  `3/6`

Lesson:

- naive alternation amplifies wrong support updates rather than fixing them

#### 4. Hierarchical family search

Procedure:

- current kernel fixed
- top-K family shortlist
- family-internal exact fixed-kernel subset search
- short kernel refit on top few support changes
- accept only if BIC improves

Outcome:

- real pair-level improvement on official `inhibition_only`
- too slow for large predicate settings
- still misses both true triplets

#### 5. Oracle-free confidence / stability layer

Examples:

- local `delta=0.05` certificate
- repeated-split stability selection

Outcome:

- helpful for judging whether a discovered improvement is trustworthy
- does not itself create triplet recovery
- useful as a decision-quality layer, not a direct accuracy fix

## Practical State Of The Research

What now appears most accurate:

- mixed-sign and calibrated kernel-robustness results are already strong enough
  for the main paper contribution
- pure inhibition should currently be treated as a stress / limitation regime
  unless a better higher-order inhibition method is found
- hierarchical family search is a meaningful pair-level improvement, but not yet
  a viable higher-order inhibition solution
- the next serious method step, if revisited, should focus on:
  - richer but still tractable support neighborhoods
  - better triplet-family separation under learned kernels
  - possibly bound-guided pruning or stability-aware selection

## Files To Keep

### Mainline / benchmark

- `/workspace/workspace/train/paper_benchmark_active/rule_dependent_kernel_active_set.py`
- `/workspace/workspace/train/paper_benchmark_active/run_paper_benchmarks.py`
- `/workspace/workspace/train/research_docs/paper_synthetic_benchmark_plan.md`
- `/workspace/workspace/train/research_docs/rule_dependent_kernel_research_notes.md`
- `/workspace/workspace/train/paper_benchmark_active/`
  - benchmark-only organization folder
  - contains the active benchmark code plus a benchmark runner wrapper
- `/workspace/data/paper_suite/paper_benchmark_results.json`
- all `paper_*.yaml` and `paper_*.pkl` benchmark configs/data under
  `/workspace/data/paper_suite`

### Important diagnostics

- `/workspace/data/paper_suite/inhibition_only_oracle_expected_support.json`
- `/workspace/data/paper_suite/inhibition_only_true_feature_observed_support.json`
- `/workspace/data/paper_suite/em_like_inhibition_only_results.json`
- `/workspace/data/paper_suite/paper_ablation_inhibition_only_hierarchical_family_search.json`
- `/workspace/data/paper_suite/paper_ablation_inhibition_only_hierarchical_controls.json`
- `/workspace/data/paper_suite/paper_ablation_inhibition_only_delta05_certificate.json`
- `/workspace/data/paper_suite/paper_ablation_inhibition_only_stability_selection.json`
- `/workspace/data/paper_suite/paper_num_predicates_20_hierarchical_family_search_half_resources.json`

## Cleanup Performed

The following temporary experiment scripts were removed because they are no
longer needed for the retained code path:

- `tmp_em_inhibition_family.py`
- `tmp_hierarchical_family_search.py`
- `tmp_inhibition_hierarchical_controls.py`
- `tmp_inhibition_stability_selection.py`
- `tmp_inhibition_support_certificate.py`
- `tmp_inhibition_targeted_experiments.py`

The following legacy side-path scripts were also removed because they are not
used by the current best synthetic benchmark path:

- `ablate_custom_overlap.py`
- `adaptive_multichannel_selector.py`
- `run_overlap_research.py`
- `run_rule_dependent_synthetic_experiments.py`
- `overlap_research_manifest.yaml`

Temporary `/tmp` logs from these runs were also cleaned up.

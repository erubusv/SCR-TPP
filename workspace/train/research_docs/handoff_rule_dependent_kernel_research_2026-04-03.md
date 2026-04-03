# Handoff: Rule-Dependent Kernel Research

Created: 2026-04-03  
Purpose: server migration handoff. This document is meant to be sufficient on its own.

## 1. Project Goal

We are studying rule-dependent kernels for synthetic multivariate Hawkes benchmarks.
The practical goal is to recover the true rule support with high recall and precision
while keeping runtime manageable. The paper goal has shifted from "just get better
search" to:

1. certified family construction
2. finite-sample oracle guarantee over that family
3. exact recovery only as a corollary under additional margin assumptions

The important point is that the main theorem should be about what the algorithm
itself certifies, not about a recovery result that assumes away the hard part.

## 2. Current Best Empirical Method Overall

The current best-performing method overall is still:

`frozen-kernel exact add-only correction`

Why it is still the best overall:

- it is the only method that has been verified end-to-end on the full official
  12-benchmark suite with recall `1.0` on all 12
- precision is not perfect, but no later theorem-friendly variant has beaten it
  on the full suite

Best overall evidence files:

- `data/paper_suite/tmp_exact_*.json`
- `data/paper_suite/tmp_exact_num_predicates_30.json`
- `data/paper_suite/paper_benchmark_results.json`
- `train/research_docs/rule_dependent_kernel_research_notes.md`

Add-only correction full-suite summary:

- exact `100/100`:
  - `kernel_triangular`
  - `kernel_exponential`
  - `num_predicates_10`
  - `num_predicates_20`
  - `ablation_excitation_only`
- one extra:
  - `logical_clean_plus`
  - `kernel_gaussian`
  - `num_predicates_30`
  - `ablation_inhibition_only`
  - `ablation_mixed_sign`
- two extras:
  - `logical_shared`
  - `logical_context`

Most important hard case under add-only correction:

- `num_predicates_30`
- elapsed `3734.88s`
- recall `1.0`
- precision `0.8571`
- recovered the missing true rule `D and F and G -> T : inhibition`
- kept one extra `F and N and U -> T : inhibition`
- evidence: `data/paper_suite/tmp_exact_num_predicates_30.json`

This matters because the simpler mainline benchmark result without correction was:

- recall `0.8333`
- precision `1.0`
- missing `D and F and G -> T : inhibition`
- evidence: `data/paper_suite/paper_benchmark_results.json`

## 3. Main Research Line After Add-Only Correction

The main post-add-only research line was:

`candidate-centered motif block + restricted local swap + exact support-fixed solve`

This line tried to keep the theorem-friendly parts:

- frozen kernel
- exact support-fixed refits
- small local ambiguity neighborhoods instead of global search

Why this line mattered:

- on `paper_ablation_inhibition_only`, a candidate-centered motif block gave the
  first concrete case where both runtime and exact-search accuracy improved at
  once
- that benchmark hit:
  - elapsed `100.22s`
  - recall `1.0`
  - precision `1.0`
- evidence: `data/paper_suite/tmp_block_motif_ablation_inhibition_only.json`

Important negative result:

- replacing local swap inside the motif block with block-global subset
  enumeration was much slower and worse
- on the same fast benchmark:
  - elapsed `646.11s`
  - recall `0.8333`
  - precision `1.0`
- evidence: `data/paper_suite/tmp_block_motif_exact_cert_ablation_inhibition_only.json`

Interpretation:

- the useful gain came from searching the right small ambiguity neighborhood
- not from making the local block search more exhaustive

## 4. Provable / Theorem-Friendly Direction

The current theorem-friendly direction is not the old heuristic top-k center
rule itself. The promising replacement is:

1. safe inactive superset by exact singleton-gain upper bounds
2. safe active interaction block
3. exact search only inside that certified family
4. confidence-aware familywise final selection

The key theoretical reframing is:

- do not use as the main theorem:
  - "if the truth is already in the family, the surrogate aligns with truth, and
    there is a margin, then we recover truth"
- use instead:
  - certified family construction / no-materially-better-support exclusion
  - finite-sample oracle guarantee over the certified family
  - exact recovery only as a corollary

Why this matters:

- the weak direct recovery theorem assumes away the core algorithmic challenge
- the stronger framing makes the contribution live in the constructed family and
  its selection guarantee

## 5. Latest Final-Selection Result

The latest mathematically structured final-selection rule is:

`confidence-set-restricted model selection`

Design:

- split A:
  - rank active-rule ambiguity candidates
  - build a fixed small ambiguity family
  - use split-A penalized empirical criterion for ranking
- split B:
  - exact-refit lower confidence bounds define the confidence set
- final:
  - choose the best split-A penalized model among confidence-set members

This is the first final-selection rule in the current line that removed an
extra rule without damaging recall in a nontrivial logical benchmark.

Latest results:

- `logical_shared`
  - elapsed `337.51s`
  - recall `1.0`
  - precision `1.0`
  - extra removed
  - evidence: `data/paper_suite/tmp_safe_swap_confset_select_logical_pair.json`
- `logical_clean_plus`
  - elapsed `862.96s`
  - recall `1.0`
  - precision `0.8571`
  - extra remained: `L -> T : excitation`
  - evidence: `data/paper_suite/tmp_safe_swap_confset_select_logical_pair.json`

Interpretation:

- `logical_shared` is now fixed by the confidence-set selector
- `logical_clean_plus` is not failing because search is weak
- `logical_clean_plus` is failing because the frozen-kernel final criterion
  still slightly prefers keeping `L -> T : excitation`

## 6. Very Important Audit Result: `logical_clean_plus` Was Not A Stable 100/100 Case

This was explicitly audited because there was confusion about whether
`logical_clean_plus` had previously been a fully solved easy benchmark.

Audit conclusion:

- in the current saved experimental line, `logical_clean_plus` has consistently
  been:
  - recall `1.0`
  - precision `0.8571`
  - extra `L -> T : excitation`
- the only saved run that removed this extra was the old aggressive
  confidence-prune, but that same run also dropped the true rule
  `J and K and L -> T : inhibition`

Implication:

- `logical_clean_plus` is not a new regression
- the benchmark is exposing a persistent local ambiguity under the frozen-kernel
  surrogate:
  - true inhibitory triplet: `J and K and L -> T : inhibition`
  - sticky surrogate extra: `L -> T : excitation`

## 7. Best Theorem-Friendly Full-Suite Status

The strongest theorem-friendly safe-swap line that was actually run at scale is
the combined result from:

- `data/paper_suite/tmp_safe_swap_suite_report.json`
- `data/paper_suite/tmp_safe_swap_ablation_batch_rerun.json`

Combined status:

- completed benchmarks: `11/12`
- total elapsed over completed cases: `9294.47s`
- exact `100/100` on `7/11`:
  - `ablation_excitation_only`
  - `ablation_inhibition_only`
  - `ablation_mixed_sign`
  - `kernel_exponential`
  - `kernel_triangular`
  - `num_predicates_10`
  - `num_predicates_20`
- not exact but full recall:
  - `logical_clean_plus`
  - `logical_shared`
  - `logical_context`
  - `kernel_gaussian`
- unresolved hardest case:
  - `num_predicates_30`
  - process died before result write under several worker settings
  - consistent with memory blow-up / runtime tail

This is why `num_predicates_30` became the bottleneck benchmark for the
theorem-friendly path.

## 8. Methods That Were Tried And Should Not Be Treated As The Main Path

These experiments were informative but are not the recommended default:

- root safe-screen + uncapped exact search
  - did not reduce candidates on the tested benchmark
  - made runtime worse
- block-global subset enumeration inside the motif block
  - slower and worse than restricted local swap
- old aggressive confidence-prune
  - could delete extras
  - but also deleted true rules and broke recall

These should be treated as negative results or cautionary baselines, not as the
main algorithm.

## 9. Code Map

Most important files:

- current running notes:
  - `train/research_docs/rule_dependent_kernel_research_notes.md`
- best overall empirical runner:
  - `train/paper_benchmark_active/tmp_run_exact_correction_benchmarks.py`
- active theorem-friendly runner:
  - `train/paper_benchmark_active/tmp_run_frozen_block_exact_solver_benchmarks.py`
- main official benchmark suite list:
  - `train/paper_benchmark_active/run_paper_benchmarks.py`
- main active-set / fitting utilities:
  - `train/paper_benchmark_active/rule_dependent_kernel_active_set.py`
- exact add-only support utilities:
  - `train/paper_benchmark_active/tmp_inhibition_profile_block_validate.py`

Important result files:

- best overall add-only hard-case result:
  - `data/paper_suite/tmp_exact_num_predicates_30.json`
- theorem-friendly 11/12 suite status:
  - `data/paper_suite/tmp_safe_swap_suite_report.json`
  - `data/paper_suite/tmp_safe_swap_ablation_batch_rerun.json`
- latest confidence-set final-selection result:
  - `data/paper_suite/tmp_safe_swap_confset_select_logical_pair.json`
- older prune comparisons:
  - `data/paper_suite/tmp_safe_swap_conf_prune_screened_logical_pair.json`
  - `data/paper_suite/tmp_safe_swap_budget_subset_logical_pair.json`
- latest official baseline result file:
  - `data/paper_suite/paper_benchmark_results.json`

## 10. What Was The Last Thing In Progress When Execution Stopped

This is the most important operational handoff item.

The last requested task was:

- run the current theorem-friendly method on the benchmark suite
- but because `num_predicates_30` was too slow, change the predicate-variation
  setup so that we use `10/20` rather than `20/30`
- all CPU and GPU resources were explicitly allowed for that pending run

What had been checked before the stop:

- the benchmark suite definitions still included all three predicate settings:
  - `num_predicates_10`
  - `num_predicates_20`
  - `num_predicates_30`
- this was confirmed in:
  - `train/paper_benchmark_active/run_paper_benchmarks.py`
  - `train/paper_benchmark_active/tmp_run_exact_correction_benchmarks.py`
  - `train/paper_benchmark_active/tmp_run_frozen_block_exact_solver_benchmarks.py`

What had not happened yet:

- the suite list had not yet been edited to remove or replace
  `num_predicates_30`
- no new full-suite run had been launched after that requested suite change

So the execution did **not** stop in the middle of a long benchmark run.
It stopped at the pre-run configuration stage, right after confirming that the
suite files still contained `num_predicates_30`.

## 11. Recommended Immediate Next Step On The New Server

Do this first:

1. edit the benchmark suite definition used by the intended runner so that the
   predicate variation uses only `num_predicates_10` and `num_predicates_20`
   for the next theorem-friendly run
2. rerun the current theorem-friendly method with:
   - confidence-set-restricted final selection
   - all resources enabled
3. do not claim `logical_clean_plus` solved unless the `L -> T : excitation`
   extra actually disappears without dropping `J and K and L -> T : inhibition`

Research priority after that:

1. certified family construction
2. finite-sample oracle guarantee over that family
3. exact recovery corollary only after the first two are solid

## 12. One-Sentence Bottom Line

Empirically, the best overall method is still `frozen-kernel exact add-only
correction`; theoretically, the most credible paper direction is now
`certified family construction + familywise confidence-set selection`, and the
last unfinished operational task was to rerun the theorem-friendly suite after
dropping `num_predicates_30` from the predicate-variation benchmark set.

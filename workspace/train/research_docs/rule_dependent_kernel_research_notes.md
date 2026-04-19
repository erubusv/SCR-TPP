# Rule-Dependent Kernel Research Notes

This note records the main directions tried around
[rule_dependent_kernel_active_set.py](/workspace/workspace/train/paper_benchmark_active/rule_dependent_kernel_active_set.py),
what actually helped, and what should not be revisited unless the modeling
assumptions change.

## Retained Mainline

The retained path is:

1. `run_active_set`
   - Build rule-specific piecewise-linear kernels from the start.
   - Use stale-but-cheap one-sided PP score tests for initial sparse admission.
2. `family_attribution_refine`
   - Reassign event/exposure mass inside nested active families and refit kernels.
3. `post_prune_irreducible_rules`
   - Apply exact nested PP refit with full kernel degrees of freedom.
4. `choose_post_prune_by_penalty_scale`
   - Choose the post-prune scale on held-out validation.

Current reference command:

```bash
python workspace/train/paper_benchmark_active/rule_dependent_kernel_active_set.py \
  --data <dataset.pkl> \
  --config <config.yaml> \
  --fixed_target <target> \
  --device cuda \
  --opt_steps 60 \
  --family_attribution_passes 1 \
  --post_prune_kernel_df \
  --post_prune_penalty_scale_grid 0.6,0.8,1.0
```

Reference result on the 3 active full datasets:

- `shared_conflict_full`: `7/7`, precision `7/7`
- `context_competition_full`: `7/7`, precision `7/7`
- `clean_full`: `5/5`, precision `5/5`

## Modeling Assumptions To Keep

- Kernels are rule-dependent, not source-global.
- Kernels are piecewise-linear with fixed knots, nonnegative heights, and area normalization.
- No parent/child kernel sharing.
- No child-rule complexity discount.
- No strong hierarchy constraint.

## Directions Tried And Not Kept

### 1. Global Source Kernel Only

Idea:
- Use one kernel per source for screening and selection.

Why rejected:
- This was the main mismatch against the synthetic generator, which already uses
  rule-specific source kernels.
- It systematically missed shared/context-dependent higher-order rules.

### 2. Unknown-Shape Kernel From Marginal Source-Target Histograms

Idea:
- Replace triangular/flat source kernels with piecewise-linear basis kernels
  learned from marginal lag histograms before rule selection.

Why rejected:
- This is still a source-global representation.
- It does not solve conditional overlap.
- It improved little or even hurt on harder shared/context settings.

### 3. Exact-State / Möbius Family Basis

Idea:
- Replace overlapping subset features with mutually exclusive exact-state basis
  inside each family.

Why rejected:
- It fixed some local family diagnostics but broke global consistency.
- As a global replacement it degraded overall recovery badly.

### 4. Parent/Core-Guided Higher-Order Screening

Idea:
- Judge a triplet by its gain over a lower-order parent/core.

Why rejected:
- A true higher-order rule need not have any true lower-order parent rule.
- Treating parent/core as necessary conflicts with the intended logical-TTP
  semantics.

### 5. Parent-Shared Kernel Reuse / Child DF Discount

Idea:
- Reuse parent kernels for child rules or discount child complexity.

Why rejected:
- This directly contradicts the rule-dependent kernel assumption.
- It is a post-hoc patch, not a justified model assumption.

### 6. Local Candidate Repricing

Idea:
- Re-optimize a small top-K inactive set under the current model before admission.

Observed behavior:
- Helped `pair_rules10_overlap`.
- Did not solve `gaussian_context`.
- Became too expensive or unstable under many-types candidate explosion.

Why not kept:
- It only reprices a stale shortlist.
- The shortlist itself remains contaminated.
- It adds a lot of code and runtime without becoming the new mainline.

### 7. Working-Set Admission With Per-Candidate Local Optimizers

Idea:
- Use a broader candidate pool and exact local repricing before admission.

Observed behavior:
- Better than plain stale top-K on overlap-pair settings.
- Still weak on `gaussian_context`.
- Still too slow for `num_types=30`.

Why not kept:
- Candidate pricing remained the bottleneck.
- The code became significantly more complex.

### 8. Cheap Reduced-Cost / KKT-Style Candidate Gate

Idea:
- Replace per-candidate local optimizers with cheap score/Hessian-style pricing.

Observed behavior:
- Preserved baseline accuracy on some stress cases.
- Improved overlap-pair precision/speed relative to expensive repricing.
- Still did not solve many-types candidate explosion robustly.

Why not kept:
- It still required the working-set admission branch.
- It did not become the new best path.
- It materially increased code complexity.

### 9. EBIC / Multiplicity-Aware Penalties

Idea:
- Penalize model space size as the candidate library grows.

Observed behavior:
- Preserved exact recovery on the 3 active full datasets.
- Did not solve `gaussian_context`.
- `num_types=30` remained computationally problematic even with lighter settings.

Why not kept in code:
- It did not become necessary for the current best path.
- It adds conceptual and code complexity without fixing the main remaining bottleneck.

### 10. Source-Level Singleton Score Screening With BH/FDR

Idea:
- Before building higher-order candidates, test each source with a singleton
  point-process score statistic and keep only BH/FDR-significant sources.

Observed behavior:
- In the hard many-types/high-filler setting, this kept essentially every
  source, including nuisance fillers.
- The reason is that when redundant predicates have very high base intensity,
  they also acquire marginal singleton significance under the current target
  process, so source-level marginal testing does not separate structure from
  nuisance.

Why not kept:
- It did not change the selected source set in the stress case that motivated it.
- It adds another screening branch without improving the actual bottleneck.
- It confirms that the many-types failure is not solved by marginal source
  screening; the problem is structured-vs-noise separation, not just
  marginal significance testing.

### 11. Train-Side Source Event Reweighting Via Lag-Residual Structured/Noise Split

Idea:
- Treat each source-target lag histogram as an additive decomposition
  `observed = homogeneous-noise null + nonnegative structured residual`.
- Estimate a per-lag structured posterior from the positive residual over the
  source-specific homogeneous Poisson expectation.
- Convert that lag posterior into per-source-event weights and use those
  weights when building train-side basis responses and rule-specific kernel
  initializers.

Why it looked principled:
- It explicitly models the intuition that a redundant predicate with high base
  intensity should not look important unless it leaves a lag-structured
  residual beyond its own background rate.
- It keeps the rule-dependent kernel model unchanged and only modifies the
  train-side source-event measure.

Observed behavior:
- The implementation was computationally prohibitive in the current code
  architecture.
- Even on the reduced `num_types_30_context_high_filler` benchmark, the learner
  became so slow that it was not competitive with the retained baseline.
- Because the heavy cost came from rebuilding weighted basis responses through
  the existing per-basis signal path, it was not a candidate for the stable
  mainline without a deeper rewrite of the basis cache.

Why not kept:
- The idea itself may still be valid, but the current architecture makes it
  too expensive.
- It did not produce a clean, benchmark-level improvement quickly enough to
  justify keeping the extra code.
- The code was reverted so the stable mainline stays small and exact on the
  3 active full datasets.

### 12. Non-Uniform Early-Dense Knot Grid For Kernel Variation

Idea:
- Keep the same rule-dependent piecewise-linear kernel family and the same
  number of knot heights, but replace the uniform knot grid with an
  early-dense grid.
- Motivation: kernel-variation stress cases have peaks near small lags and
  wider tails; a uniform 7-knot grid over `[0, 10]` allocates too little
  resolution near the early peak.

Observed behavior:
- On `gaussian_shared`, a fixed early-dense grid improved substantially:
  - uniform `[0,10]`: recall `4/7`, precision `4/9`
  - early-dense `[0,10]`: recall `6/7`, precision `6/10`
  - early-dense `[0,14]`: recall `6/7`, precision `6/8`
- On `gaussian_context`, the same change did **nothing**:
  - all three settings stayed at recall `5/7`, precision `5/10`
- As a fixed global change, it is not admissible because it breaks the active
  benchmark suite:
  - `shared_full` with fixed early-dense `[0,14]` dropped to recall `5/7`,
    precision `5/9`

Why not kept:
- It helps one kernel-shift regime (`gaussian_shared`) but not the harder one
  (`gaussian_context`).
- Because it regresses an active full benchmark when applied globally, it
  cannot replace the retained baseline.
- The result suggests that kernel approximation bias is only part of the
  Gaussian failure; the remaining hard case is still driven by selection under
  broad overlap, not by the knot grid alone.

### 13. Baseline-Occupancy Feature Corrections For Broad Overlap

Ideas tried:
- Per-source excess occupancy transform:
  \[
  a_s'(t)=\max\{(a_s(t)-p_s)/(1-p_s), 0\}
  \]
  where \(p_s\) is the train-grid average occupancy.
- Higher-order null-joint correction:
  keep singleton features unchanged, but for order \(\ge 2\) replace
  \[
  \rho_U(t)
  \]
  by
  \[
  \max\{\rho_U(t)-\prod_{s\in U} p_s, 0\}.
  \]

Observed behavior:
- On overlap-matched triangular stress cases, both variants reduced extras but
  also destroyed true higher-order recall.
- On `gaussian_context`, the per-source excess transform recovered some missing
  singleton inhibition but killed true higher-order rules and increased extras.

Why not kept:
- These transforms remove broad baseline co-activation, but they also remove
  true high-order signal because the current model still treats rule admission
  independently after the transform.
- They are therefore not a robust fix for overlap; they merely move the
  precision/recall trade-off.

### 14. Overlap-Aware Swap Search

Ideas tried:
- Replace the greedy add/drop neighborhood with exact swap moves in which an
  inactive candidate can replace a block of same-sign active rules whose source
  sets overlap it.
- Also tried a stronger closure-based variant, where the swap candidate pool is
  expanded to inactive rules contained in the source-union of an active
  overlap block.

Observed behavior:
- On reduced overlap stress cases this either matched baseline or became too
  slow for the amount of improvement obtained.
- It did not become a new stable mainline.

Why not kept:
- The direction is conceptually closer to the real overlap problem than stale
  top-K repricing, but in the current architecture exact swap refits are still
  too expensive.
- More importantly, the search neighborhood alone was not enough to reliably
  recover the true overlap solution.

### 15. Gaussian-Context Diagnostic: Truth Beats Baseline Under The Same Objective

Approximate exact-refit diagnostic on `kernel_gaussian_context`:
- `baseline_from_json` (the current selected 10-rule proxy solution): BIC
  about `69741.1`
- `true_rules` (the 7 true rules, same learner objective, same kernel family):
  BIC about `69311.6`
- `baseline_plus_B_E` (baseline proxies plus the missing singleton inhibitors):
  BIC about `69836.4`

Interpretation:
- The current model class is capable of preferring the true rule set on the
  Gaussian overlap case.
- The failure is therefore primarily an **inference/search problem**, not a
  fundamental mismatch of the rule-dependent piecewise-linear kernel model.
- Simply adding the missing singleton rules on top of the proxy solution is not
  enough; the proxy block must be **replaced**, not augmented.

### 16. Same-Sign Source-Overlap Component Block Search

Idea:
- Keep the current model and BIC exactly as they are.
- Replace the one-rule greedy neighborhood with a broader local search over a
  same-sign overlap component.
- Build a component from active rules whose source sets overlap.
- Hold the outside-of-component model fixed, and re-solve the component from
  scratch or via exact subset search inside the source union.

Why it looked principled:
- The Gaussian-context diagnostic shows that the true rule set already wins
  under the current objective; the learner just fails to reach it.
- A block-level replacement neighborhood is the first natural way to let the
  search replace an entire proxy block with a truth block.

Observed behavior:
- A code-path that re-solved same-sign source-overlap components from scratch
  was added behind experimental flags.
- On the reduced overlap-matched triangular context stress case, it collapsed
  to a smaller inhibitory-only model and reduced recovery rather than
  improving it.
- A more targeted external experiment that enumerated subsets of
  `active component + top missing candidates` also failed to change the reduced
  overlap context solution meaningfully.
- On full Gaussian-context, the exact subset-search variant became too slow to
  finish in a practical time budget.

Why not kept:
- The source-overlap component definition is too coarse; it opens blocks that
  are broader than the actual competitive explanation set.
- Re-solving a component from scratch can destroy good rules once the same-sign
  source union is large.
- The exact subset-search variant is computationally too expensive on the full
  Gaussian stress case.
- The code was reverted so the stable mainline stays clean.

## What The Failed Experiments Actually Taught

The unresolved bottleneck is not kernel family mismatch alone.

The main unresolved issue is:

- `inactive` candidates are not admitted using a fully current-model,
  global, cheap enough pricing rule under candidate explosion.

This matters most when:

- kernels are broad and overlapping (`gaussian_context`)
- event types are numerous (`num_types >= 20`)
- overlapping lower-order rules can be absorbed by a false higher-order proxy

## Many-Types Diagnosis To Keep In Mind

The current `num_types` stress test can easily conflate:

- more candidate predicates
- more nuisance background intensity
- sequence truncation through `max_len`

Controlled context-only diagnostics showed:

- with `20` or `30` total types but **fixed total filler base mass**, the retained
  mainline recovered the full rule set exactly
- with `30` types and **zero filler base**, it also recovered exactly
- the major collapse happened when every extra filler type inherited the same
  average non-target base intensity, which drove the total nuisance base mass
  up sharply and saturated sequence length

Interpretation:

- many-types degradation in the current synthetic suite is driven primarily by
  nuisance-event mass, not by type count alone
- a future robust-many-types fix should therefore target structured-vs-noise
  separation, not just combinatorial candidate pruning

## Next Direction To Revisit Later

If research resumes from this file, the next serious direction should be:

- global current-model pricing
- global event attribution
- working-set admission driven by a truly cheap reduced-cost / KKT rule

This should be attempted only if:

- we are ready to replace the admission logic end-to-end, not patch it locally
- and we can show it improves stress cases without regressing the 3 active full datasets

Until then, the retained mainline above is the stable baseline.

### 17. Entropy-Aware / ICL-Style Diagnostic

Idea:
- Keep the current model, kernel family, and exact-refit support evaluation.
- Add an ambiguity term on top of `BIC`:
  - `ICL := BIC + 2 * entropy`
- Here `entropy` is computed from validation-side attribution:
  - excitation uses per-event attribution entropy
  - inhibition uses exposure-weighted attribution entropy on the validation grid

Why it looked principled:
- Broad overlap may hurt practical identifiability even when the true support
  already wins in likelihood/BIC.
- `ICL`-style criteria explicitly penalize diffuse explanations, so they are a
  natural way to test whether overlap failures are mainly due to the final
  criterion rather than the search path.

Observed behavior:
- `shared_truth`:
  - `BIC = 97266.9`
  - `entropy = 21983.8`
  - `ICL = 141234.5`
- `shared_truth_plus_acd`:
  - `BIC = 97405.2`
  - `entropy = 22430.3`
  - `ICL = 142265.7`
- `clean_truth`:
  - `BIC = 90803.7`
  - `entropy = 15064.0`
  - `ICL = 120931.7`
- `clean_truth_plus_gi`:
  - `BIC = 90462.5`
  - `entropy = 16278.4`
  - `ICL = 123019.2`
- `gauss_context_truth`:
  - `BIC = 67067.7`
  - `entropy = 10404.1`
  - `ICL = 87876.0`
- `gauss_context_proxy`:
  - `BIC = 67542.1`
  - `entropy = 14579.5`
  - `ICL = 96701.1`

Interpretation:
- On `shared`, both `BIC` and `ICL` prefer truth.
- On `clean`, plain `BIC` incorrectly prefers a truth-plus-extra support, while
  `ICL` correctly prefers truth because the extra rule increases attribution
  ambiguity substantially.
- On `gaussian_context`, plain `BIC` already strongly prefers truth over the
  proxy support; `ICL` amplifies that gap further.

Why not kept:
- This diagnostic is useful, but it does not fix the main overlap failure:
  on the hard Gaussian case, the true support already wins under plain `BIC`,
  so the bottleneck is still search/inference, not the final criterion.
- `ICL`/entropy-aware scoring may still be useful later as a final pruning or
  model-selection layer for extras, but it is not the main next step for
  recovering missed rules under broad overlap.

### 18. Gaussian Overlap: Candidate Admission, Not Local Replacement

Idea:
- test whether the remaining Gaussian failure is because local overlap-aware
  replacement cannot recover the truth even if the right candidates are
  present, or because the correct missing candidates never make it into the
  local search neighborhood.

Observed diagnostics:
- On the current Gaussian-context proxy solution, inhibition-candidate ranking
  under the current one-rule gain is dominated by higher-order proxy rules:
  - top candidates included `(1,4,6)`, `(0,1,4)`, `(1,3,4)`, `(1,2,4)`, ...
  - the true missing singleton inhibitors ranked much lower:
    - `(4,)` ranked around 12th
    - `(1,)` ranked around 14th
- This means current marginal candidate pricing does **not** surface the true
  missing rules in the broad-overlap regime.

Targeted local exact-refit test:
- Fixed the already-matched support:
  - `A exc`, `ACD inh`, `BCD exc`, `EF exc`, `EFG inh`
- Then compared four local supports:
  - `fixed`
  - `fixed + B`
  - `fixed + E`
  - `fixed + B + E`

Results:
- At low optimization budget (`5` steps), the local search still preferred
  dropping the ambiguous block entirely.
- At `10` steps:
  - `fixed`: `67091.6`
  - `fixed + B`: `67081.1`
  - `fixed + E`: `67073.6`
  - `fixed + B + E`: `67067.7`
- At `20` steps:
  - `fixed`: `66562.4`
  - `fixed + B`: `66432.7`
  - `fixed + E`: `66406.9`
  - `fixed + B + E`: `66291.1`

Interpretation:
- Once the correct missing candidates `B` and `E` are admitted and the local
  refit is given enough optimization budget, the local block replacement
  **does** move toward the true support.
- Therefore the remaining bottleneck is not that local block replacement is
  inherently wrong.
- The bottleneck is that current candidate admission is still based on
  **marginal one-rule gain**, and that ranking systematically favors broad
  higher-order proxy rules over the true low-order replacements under strong
  overlap.

Conclusion:
- The next principled direction is not another criterion change.
- It is **group / set admission**:
  - admit small competing candidate sets jointly rather than one rule at a time
  - because the Gaussian-overlap failure is driven by a ranking error at the
    admission stage, not by the exact local replacement stage once the right
    candidates are present.

### 19. Immediate-Parent Restriction Lifts The True Missing Singletons

Idea:
- Instead of ranking all inactive rules, restrict candidate admission to the
  **immediate lower-order parents** of the currently active inhibition rules.
- This is a principled local-search neighborhood on the subset lattice rather
  than a heuristic candidate list.

Observed diagnostic on `gaussian_context` proxy support:
- Full-space marginal inhibition ranking had the true missing singletons very
  low:
  - `(4,)` around 12th
  - `(1,)` around 14th
- After restricting the candidate set to immediate parents of active
  inhibition rules, the ranking changed to:
  - `(3,4)` inhibition, gain `1339.3`
  - `(2,4)` inhibition, gain `1263.4`
  - `(4,)` inhibition, gain `986.2`
  - `(1,)` inhibition, gain `735.4`
  - `(0,4)` inhibition, gain `402.9`

Interpretation:
- The true missing singletons `E` and `B` move into the top 3-4 positions once
  admission is restricted to the immediate-parent neighborhood.
- This is exactly the direction suggested by the earlier local exact-refit
  result: once the right candidates are admitted, the local block replacement
  can move to the true support.

Conclusion:
- The next serious direction should be **set admission over a responsibility-
  overlap block plus its immediate lower-order closure**.
- This is more targeted and more principled than global one-rule candidate
  ranking, and it directly addresses the observed Gaussian-overlap ranking
  failure.

### 20. Exact Set Admission On The Gaussian Proxy Block

Idea:
- Take the Gaussian-context proxy solution and hold the already-matched support
  fixed:
  - `A exc`, `ACD inh`, `BCD exc`, `EF exc`, `EFG inh`
- Then do exact subset search over the ambiguous inhibition block:
  - proxy rules: `AB`, `ACE`, `ADE`, `BE`, `EG`
  - plus the true missing singleton competitors: `B`, `E`
- This is the smallest clean test of whether **set admission** can replace the
  proxy block with the truth block.

Observed behavior:
- With `10` local optimization steps per subset, the best `BIC` support was:
  - `A exc`
  - `ACD inh`
  - `B inh`
  - `BCD exc`
  - `E inh`
  - `EF exc`
  - `EFG inh`
- In other words, exact set admission over the ambiguous block recovered the
  full true support.
- Best score:
  - `BIC = 67067.687`
- By contrast, the best `ICL` subset in this exact search stayed at the
  smaller support without `B` and `E`, showing that entropy-aware scoring is
  too aggressive here if the main goal is recall recovery under overlap.

Interpretation:
- This is the strongest evidence so far that the unresolved Gaussian failure is
  **not** due to model mismatch or final criterion mismatch.
- Once the correct competing candidates are admitted jointly, the current
  model and current `BIC` can recover the true block exactly.
- Therefore the remaining missing ingredient is a principled way to form and
  admit these candidate sets automatically.

Conclusion:
- The next mainline research direction should be:
  - build a small **responsibility-overlap block**
  - augment it with its immediate lower-order closure
  - run local exact/near-exact set admission under plain `BIC`
- This is now directly supported by experiment, not just by diagnosis.

### 21. Automatic Parent-Block Extraction On Gaussian Context

Idea:
- Replace the hand-picked proxy block with an automatic one:
  1. fit the current Gaussian-context proxy support,
  2. build same-sign inhibition components using normalized responsibility
     overlap,
  3. for each component, rank its inactive immediate parents by current-model
     one-rule gain,
  4. define the candidate block as those top parent rules plus the currently
     active inhibition rules that have one of those parents.

Observed behavior:
- At overlap threshold `tau = 0.2`, the highest-scoring inhibition component
  was
  - active rules:
    - `AB`
    - `ACE`
    - `ADE`
    - `BE`
    - `EG`
  - top immediate parents:
    - `DE` with gain `1339.3`
    - `CE` with gain `1263.4`
    - `E` with gain `986.2`
    - `B` with gain `735.4`
- This automatic block is almost exactly the manually identified ambiguous
  proxy block from the earlier exact-set-admission experiment.
- Importantly, the already-matched `EFG` rule stayed outside the block under
  this construction, which keeps the local search small and semantically
  aligned with the actual ambiguity.

Interpretation:
- Responsibility-overlap plus immediate-parent closure is a workable automatic
  way to isolate the ambiguous Gaussian proxy block.
- The remaining difficulty is computational rather than conceptual: full exact
  subset search on the resulting 9-rule block is still expensive in the
  current external implementation.

Conclusion:
- Automatic block formation is no longer the missing conceptual step.
- The next gap is efficient local set search inside that automatically formed
  block.

### 22. Parent-Only Replacement Is Enough To Recover The Gaussian Truth Block

Idea:
- Use the automatically extracted immediate-parent candidates from the
  Gaussian-context inhibition block:
  - `DE`, `CE`, `E`, `B`
- Keep the already-matched outside support fixed:
  - `A exc`, `ACD inh`, `BCD exc`, `EF exc`, `EFG inh`
- Then test only **parent-only replacements**, i.e. drop the current proxy
  descendants and search over subsets of those four parent candidates.

Observed behavior:
- The best support under exact local refit was:
  - `A exc`
  - `ACD inh`
  - `B inh`
  - `BCD exc`
  - `E inh`
  - `EF exc`
  - `EFG inh`
- Best score:
  - `BIC = 66435.671`
- The best subset was just the two singleton parents:
  - `B`
  - `E`
- Adding the extra pair parents `CE` and `DE` did not survive exact refit;
  they collapsed back to the same best support.

Observed stability on the current exact-recovery full benchmarks:
- `shared_conflict_full` truth seed:
  - no positive parent-only replacement block found
  - support unchanged
- `context_competition_full` truth seed:
  - no positive parent-only replacement block found
  - support unchanged
- `clean_full` truth seed:
  - no positive parent-only replacement block found
  - support unchanged

Interpretation:
- For the unresolved Gaussian overlap failure, the real missing ingredient is
  not a richer kernel family or a different final criterion.
- It is the ability to jointly admit a small lower-order parent set in place
  of a correlated proxy descendant set.
- The fact that the same external pass leaves the existing exact-recovery
  benchmarks unchanged is the first evidence that this direction is not merely
  overfitting the Gaussian stress case.

Conclusion:
- The most credible next mainline direction is:
  - automatic explanation-overlap block extraction
  - immediate-parent candidate generation
  - local parent-set admission / descendant replacement under plain `BIC`
- This is now supported by experiment on both the Gaussian failure case and
  the existing exact-recovery benchmark suite.

### 23. End-to-End Kernel-Family Post-Pass Evaluation

Setup:
- Treat the current rule-dependent learner output as the seed support.
- Apply the new external post-pass:
  - explanation-overlap component extraction on inhibition rules,
  - immediate-parent candidate generation,
  - local **parent-only replacement** under plain `BIC`.
- For `triangular`, the seed support was the current exact-recovery support.
- For `gaussian` and `exponential`, the seed support came from the stored
  baseline predictions on the kernel-family benchmark split.

Observed results:

`triangular` full benchmarks:
- `shared_conflict_full`
  - unchanged
  - still exact
- `context_competition_full`
  - unchanged
  - still exact
- `clean_full`
  - unchanged
  - still exact

`gaussian` split benchmarks:
- `gaussian_context`
  - baseline: recall `5/7`, precision `5/10`
  - after parent-only replacement: recall `7/7`, precision `7/7`
  - BIC improved from `67542.076` to `67067.687`
- `gaussian_shared`
  - unchanged
  - stayed at recall `4/7`, precision `4/9`
- `gaussian_clean`
  - unchanged
  - stayed at recall `5/5`, precision `5/7`

`exponential` split benchmarks:
- `exponential_context`
  - unchanged
  - stayed at recall `6/7`, precision `6/8`
- `exponential_shared`
  - unchanged
  - stayed exact
- `exponential_clean`
  - unchanged
  - stayed exact

Interpretation:
- The new post-pass is **surgical rather than universal**.
- It directly fixes the broad-overlap Gaussian-context failure, where the seed
  support is missing low-order inhibitory parents and contains correlated
  proxy descendants.
- It does **not** improve Gaussian-shared or Gaussian-clean, which means those
  failures are not driven by the same parent-vs-descendant ambiguity.
- It also does not help exponential-context, whose residual error pattern is
  different from Gaussian-context.

Conclusion:
- Automatic parent-only replacement is now validated as a real solution to one
  specific overlap pathology:
  - broad-overlap, same-sign, descendant-proxy substitution.
- The next mainline step should integrate this pass into the learner without
  disturbing the already-exact triangular benchmark.

### 24. Exponential Context Needs The Same Local Replacement, But With Stronger Refit

We diagnosed `exponential_context` with a local inhibition replacement search
around the residual proxy block.

Fixed support:
- `A exc`
- `ACD inh`
- `BCD exc`
- `E inh`
- `EF exc`
- `EFG inh`

Local candidates:
- `AB inh`
- `BE inh`
- `B inh`

Observed behavior:
- At modest local refit (`steps=10`):
  - `fixed`: `73879.654`
  - `fixed + B`: `73935.638`
  - best subset under exact local search was still the proxy-free `fixed`
    support.
- At stronger local refit:
  - `steps=20`
    - `fixed`: `72015.576`
    - `fixed + B`: `71994.834`
  - `steps=30`
    - `fixed`: `71973.724`
    - `fixed + B`: `71929.334`

Interpretation:
- The same parent-replacement mechanism appears relevant for
  `exponential_context`.
- The difference from `gaussian_context` is not that the mechanism is wrong,
  but that the local exact refit must run long enough to expose the gain from
  adding the missing singleton parent `B`.
- So `exponential_context` looks like an optimization-budget / local-search
  issue, not a criterion issue.

### 25. Gaussian Clean Has A Different Failure Mode: Nested Extras

We diagnosed the `gaussian_clean` excitation tail around the exact truth block.

Fixed support:
- `A exc`
- `B inh`
- `CD exc`
- `EF inh`

Local excitation candidates:
- `GHI exc`
- `GI exc`
- `HI exc`

Results:
- Best by plain `BIC` kept the redundant subset extras:
  - support = `A`, `B`, `CD`, `EF`, `GHI`, `GI`, `HI`
  - `BIC = 151187.246`
- Best by a naive global `ICL = BIC + 2H(q)` was too aggressive:
  - best ICL dropped even the true `GHI`
  - support = `A`, `B`, `CD`, `EF`
  - `BIC = 153595.118`
  - `ICL = 178867.607`
- The next-best ICL support did keep `GHI`, but still preferred removing the
  extras:
  - support = `A`, `B`, `CD`, `EF`, `GHI`
  - `BIC = 153485.446`
  - `ICL = 180022.918`

Interpretation:
- `gaussian_clean` is not a missing-parent problem.
- It is a nested redundant-extra problem: plain `BIC` under-penalizes the
  extra subsets `GI` and `HI`, while naive global `ICL` over-penalizes and can
  also remove the true higher-order rule.
- So the right next step here is not global `ICL`, but a more selective local
  irreducibility / nested-extra criterion.

### 26. Gaussian Shared Likely Needs A Broader Sibling / Mixed-Sign Block

We probed `gaussian_shared` with a local excitation block search under a fixed
inhibition scaffold.

Fixed inhibition support:
- `A inh`
- `ACD inh`
- `BE inh`
- `BEF inh`
- `CD inh`

Excitation candidates:
- `AB exc`
- `ABD exc`
- `CE exc`
- `DEF exc`
- `C exc`
- `D exc`
- `EF exc`

Best local exact subset search (`steps=5`) returned:
- `A inh`
- `CD inh`
- `BEF inh`
- `AB exc`
- `CE exc`
- `C exc`
- `D exc`
- `EF exc`

with
- `BIC = 118440.520`

Interpretation:
- This local excitation block can recover the missing true excitations
  `C`, `D`, and `EF`.
- But it still keeps `CE exc`, which is a redundant sibling-style proxy.
- So `gaussian_shared` appears to require a broader competitive block than the
  simple parent-only closure used for `gaussian_context`.
- In particular, it points toward a mixed-sign / sibling-closure local search
  rather than a pure parent replacement pass.

### 27. Generic Ambiguity-Block Exhaustive Search Is Too Expensive Without Candidate Pruning

We attempted the next logical generalization:
- build explanation-overlap blocks automatically from the current fit,
- take the recursive lower-order closure of each block,
- run exact subset search inside that closure,
- then run a same-sign nested-family local-ICL prune.

This was implemented only as a temporary external script and **not** integrated
into the mainline.

Observed behavior:
- The very first stress case, `gaussian_shared`, already became impractical.
- The issue was not numerical failure; it was **closure explosion**:
  once an ambiguous excitation block is expanded by recursive lower-order
  closure, the exact subset-search neighborhood becomes too large to search
  cheaply.
- In other words, the direction is conceptually right, but a naive exhaustive
  implementation is not computationally viable.

Interpretation:
- The next step should still be block-structured inference.
- But it must include a **candidate-pruning layer inside each block** before
  exact local search.
- The most credible choices are:
  - score/Hessian or reduced-cost screening,
  - KKT-style pruning,
  - or a small beam/branch-and-bound that uses current-model local gains only
    to prune the closure, not as the final criterion.

Conclusion:
- `ambiguity-aware block inference` still looks like the right high-level
  direction.
- However, the exact-search version cannot become the mainline as-is.
- A practical version must be:
  - overlap block extraction,
  - **pruned** local closure,
  - exact or near-exact search only inside that reduced neighborhood.

### 28. A Generic Pruned Block Pass Still Degrades Gaussian/Exponential Context

We implemented a lighter external prototype:
- extract same-sign explanation-overlap blocks,
- add only the active block rules plus the top immediate-parent candidates
  under a current-model one-rule score,
- run exact local `BIC` search inside that pruned set,
- then run one pass of nested-family local `ICL` prune.

This was intentionally much cheaper than the exhaustive closure search.

Observed results on split kernel benchmarks:

- `gaussian_context`
  - seed: recall `5/7`, precision `5/10`
  - after generic pruned block pass:
    - recall `4/7`
    - precision `4/9`
  - `BIC` improved numerically (`68756.466 -> 66793.610`) but the selected
    structure became worse:
    - it dropped true `EF exc`
    - and still did not recover the missing singleton parents `B`, `E`
- `exponential_context`
  - seed: recall `6/7`, precision `6/8`
  - after generic pruned block pass:
    - recall `5/7`
    - precision `5/8`
  - again, a lower local `BIC` did not translate into better support recovery.
- `gaussian_clean`
  - the same generic pass was too slow once the nested-family exact search
    engaged, even under reduced local optimization settings.

Interpretation:
- A generic “same-sign overlap block + top parent candidates” rule is still
  too coarse.
- It can reduce local objective values while moving to structurally worse
  supports, because the block is not aligned tightly enough with the actual
  ambiguity class.
- So even after pruning, this generic block pass is **not** robust enough to
  serve as a mainline solution.

Conclusion:
- The successful `gaussian_context` fix remains a **targeted** parent-only
  replacement on the right ambiguity block.
- The more generic version fails because the automatically extracted blocks are
  still too broad, and the candidate pruning is still too weakly aligned with
  the true ambiguity structure.
- The next viable step is therefore **not** “more generic block search”, but
  a more selective block-definition rule, likely based on smaller ambiguity
  motifs rather than whole same-sign overlap components.

### 29. Candidate-Centered Motifs Expose The Right Missing Parents, But One-Rule Gain Still Misranks Them

We then tested an even smaller neighborhood:
- choose a single inactive candidate subset,
- collect the active same-sign rules that strictly contain it,
- run exact local replacement only inside that tiny motif.

The key question became: **how should the inactive candidate be ranked?**

Observed on `gaussian_context`:
- If the center candidate is chosen by current-model one-rule gain, the top
  inhibition candidate is *not* the true missing singleton parent.
- In a light exact test, the top-1 one-rule-gain motif only improved precision
  slightly and did not recover the missing `B`/`E`:
  - seed: recall `5/7`, precision `5/10`
  - after top-1 one-rule-gain motif:
    - recall `5/7`
    - precision `5/9`
- Looking only at **support count** inside the current inhibition seed gives a
  more meaningful ranking:
  - `E` is the top inactive lower-order candidate, supported by `5` active
    inhibition rules,
  - `B` is also near the top, supported by `2` active inhibition rules.

Interpretation:
- The right local neighborhood may indeed be a **candidate-centered motif**
  rather than a whole overlap component.
- But the admission score for the center candidate should not be the same
  one-rule gain currently used by the global learner.
- In broad-overlap regimes, that gain still over-favors local proxy subsets
  such as `DE`-type candidates and under-ranks the true shared parent.
- A more principled center score should combine:
  - how many active ambiguous rules the candidate can explain,
  - and how strongly those rules overlap under the current fit.

Conclusion:
- Candidate-centered motifs remain promising.
- The next useful refinement is not a larger search, but a **better center
  ranking rule**, based on shared-support / shared-explanation counts rather
  than marginal one-rule gain alone.

### 30. Generic Ambiguity Neighborhoods Still Did Not Move The Hard Seeds

We tested a more generic local post-pass outside the mainline:
- exact-refit the current seed support,
- build same-sign active-rule clusters from coefficient-weighted feature cosine,
- for each cluster, define a local ambiguity neighborhood as the union-based
  closure of:
  - parents (immediate subsets),
  - siblings (one-edit same-order variants),
  - immediate supersets,
- then run a small exact local hill-climb under plain `BIC`.

This was intentionally broader than the earlier parent-only family pass, but
still much cheaper than exhaustive block search.

Results:
- `triangular_context_full`
  - unchanged
  - still exact: `7/7`, precision `7/7`
- `gaussian_context`
  - unchanged
  - stayed at recall `5/7`, precision `5/10`
- `gaussian_shared`
  - unchanged
  - stayed at recall `4/7`, precision `4/9`
- `gaussian_clean`
  - unchanged
  - stayed at recall `5/5`, precision `5/7`
- `exponential_context`
  - unchanged
  - stayed at recall `6/7`, precision `6/8`

Interpretation:
- A generic neighborhood of “parents + siblings + one-step supersets” is still
  too weak or too misaligned to recover the true ambiguity blocks.
- The failure is not that local replacement is impossible; earlier targeted
  diagnostics already showed that the right local neighborhoods can recover
  `gaussian_context`.
- The real remaining issue is **automatic neighborhood discovery**:
  the generic cluster/closure rule above does not isolate the truly competing
  rules tightly enough.

Conclusion:
- Do not integrate this generic ambiguity-neighborhood pass.
- Keep the mainline unchanged.
- The next useful direction should focus on a better automatic notion of
  “shared missing parent / shared explanation support”, not a broader generic
  closure around same-sign overlap clusters.

### 31. Why High-Order Inhibition Screening Still Picks The Wrong Triplets

We revisited the `paper_ablation_inhibition_only` benchmark at the current
official setting (`target base intensity = 0.05`) and decomposed the local
inhibition screening score used by `rule_score()`:

\[
g_{\text{inh}}(U) = -\sum_i \rho_U(t_i) + \sum_m w_m \lambda_0(t_m)\rho_U(t_m),
\qquad
h_{\text{inh}}(U) = \sum_m w_m \lambda_0(t_m)\rho_U(t_m)^2,
\]

with local screening gain

\[
\Delta_{\text{inh}}(U) = \frac{1}{2}\frac{g_{\text{inh}}(U)^2}{h_{\text{inh}}(U)} - \text{penalty}(U).
\]

The key finding is that even if we rank **only triplets**, the top-ranked
triplets are still mostly surrogates:

- top ranked surrogate triplets:
  - `(4,5,6)`: gain `-64.35`
  - `(4,5,7)`: gain `-65.93`
  - `(0,4,5)`: gain `-74.28`
  - `(3,4,5)`: gain `-76.50`
  - `(2,4,5)`: gain `-77.04`
- true triplets:
  - `DFG = (3,5,6)`: rank `7`, gain `-97.68`
  - `CGH = (2,6,7)`: rank `15`, gain `-108.87`

The score decomposition explains why:

- surrogate `EFG = (4,5,6)`
  - `event_loss = 25.09`
  - `grid_gain = 31.76`
  - `g = 6.66`
  - `h = 0.474`
- true `DFG = (3,5,6)`
  - `event_loss = 25.91`
  - `grid_gain = 29.24`
  - `g = 3.33`
  - `h = 0.410`
- true `CGH = (2,6,7)`
  - `event_loss = 27.53`
  - `grid_gain = 28.85`
  - `g = 1.32`
  - `h = 0.368`

Interpretation:
- The failure is **not** that the true triplets have zero signal.
- The failure is that the local inhibition score still prefers triplets that
  contain a strong lower-order truth, especially the true pair `EF`.
- In other words, a raw triplet score based on the full conjunction feature
  still measures “how much total inhibitory footprint does this triplet have?”
  more than “is this source union best explained by a true 3-way interaction?”
- This is why both:
  - low-order family/member-sum screening, and
  - high-order-only/full-conjunction triplet screening
  still shortlist the wrong unions.

Additional evidence from earlier local-family experiments is consistent with
this diagnosis:
- when the top surrogate family around `EF` was sent to local exact search, the
  family-level refit preferred the empty/unchanged support,
- so the real failure is not the family-local exact comparison itself,
  but the **automatic family shortlist**.

Conclusion:
- The remaining bottleneck for weak higher-order inhibition is now clearly the
  union-level screening statistic.
- The next principled direction is **not** another rule-level or triplet-level
  gain tweak.
- The next useful direction is a **source-union family value**:
  for each source union `C`, score the whole family
  `{singletons, pairs, triplet}` over `C` with a small local family optimizer,
  then shortlist unions by that family value and only afterward do local exact
  BIC search.

### 32. Family Best-Support Screening Helps Ranking A Bit, But Not Enough

We tested the next-step surrogate proposed above:

\[
\hat V(C)
=
\max_{S \subseteq F(C)}
\left[
\max_{c_S \ge 0}
g_S^\top c_S - \frac12 c_S^\top H_{SS} c_S
- \lambda |S|
\right],
\]

where `F(C)` is the inhibition family over a source union `C`, `g/H` are the
current-model local score/Hessian terms, and the support search is exact
inside the family surrogate.

The first useful result is on `paper_ablation_inhibition_only` at the current
official target-base setting `0.05`.

Using a baseline refit close to the official support (with recovered rules
`A inh`, `B inh`, `CD inh`), triplet-family screening behaves as follows:

- top 6 triplet families are still all `EF`-contaminated surrogates:
  - `(0,4,5)`, `(1,4,5)`, `(2,4,5)`, `(3,4,5)`, `(4,5,6)`, `(4,5,7)`
  - each with the same best family support `{EF}`
- true `DFG` improves relative to raw triplet-only screening but still ranks
  only `7th`, with best family support `{DFG}`
- true `CGH` remains very weak:
  - rank `50`
  - best family value `0.0`
  - empty best support

We also forced local exact refits for:
- the top surrogate family support `{EF}`
- the true `DFG` family support `{DFG}`

Neither improved on the baseline:
- baseline BIC: `14913.8397`
- forced `{EF}` exact refit: `14913.8475`
- forced `{DFG}` exact refit: `14913.8934`

and both returned exactly the same support as baseline:
- `A inh`
- `B inh`
- `CD inh`

Interpretation:
- The family-value surrogate is better than rule-level triplet screening in
  one narrow sense: it can raise a true triplet family above many raw triplet
  surrogates if the family’s best support is itself the triplet.
- But the dominant `EF`-driven families still rank higher because their best
  family support is a strong true pair (`EF`), and the local surrogate value is
  still dominated by that stronger lower-order inhibition footprint.
- More importantly, even when the true `DFG` family is manually shortlisted,
  the local exact refit does **not** beat the current baseline.

Conclusion:
- The remaining bottleneck is no longer “triplets never reach the shortlist.”
- At least for `inhibition_only`, the stronger result is that **even a correct
  triplet family shortlist is not enough**: local family addition of `DFG`
  still loses to the baseline support under the current objective/search.
- This means the next useful direction must address the interaction between
  current active inhibition support and new higher-order families more globally,
  rather than only improving the shortlist score.

### 33. In The Current Inhibition-Only Regime, Even Jointly Adding Missing Rules Does Not Beat The Compact Baseline

We directly compared several fixed inhibition supports on
`paper_ablation_inhibition_only` at the official `target base = 0.05`, using
the same fixed-support refit routine for all compared supports.

Supports tested:
- compact baseline support: `{A, B, CD}`
- plus one missing rule at a time:
  - `{A, B, CD, EF}`
  - `{A, B, CD, DFG}`
  - `{A, B, CD, CGH}`
- plus two missing rules:
  - `{A, B, CD, EF, DFG}`
- full truth:
  - `{A, B, CD, EF, CGH, DFG}`

The compact fixed-support baseline was still preferred by this comparison:
- compact baseline refit: BIC `15203.93`
- `+EF`: BIC `15210.44`
- `+DFG`: BIC `15214.70`
- `+CGH`: BIC `15214.79`
- `+EF + DFG`: BIC `15221.30`
- full truth: BIC `15232.39`

Interpretation:
- This result should **not** be over-read as “the true support is impossible”;
  the main pipeline baseline still follows a richer optimize/refine/prune path
  and reaches a lower absolute BIC than the simple fixed-support refits.
- But it *does* show something important:
  under a support-comparison routine that treats every candidate support the
  same way, the current inhibition-only regime strongly prefers a compact
  subset of the true rules.
- In other words, the failure is not just that `EF`, `CGH`, and `DFG` do not
  reach the shortlist. Even when they are manually included, this low-signal
  regime does not provide a strong enough objective advantage for the larger
  support.

Conclusion:
- For `inhibition_only` at `target base = 0.05`, the remaining issue is not
  merely a screening bug.
- The benchmark is now in a regime where higher-order inhibitory truth is
  genuinely weak relative to the compact lower-order explanation.
- This means the next principled move is **not** another screening heuristic.
- The right split is:
  - keep `mixed-sign` and calibrated kernel/predicate benchmarks as the main
    method claims,
  - treat pure `inhibition-only` as a stress/limitation regime unless the
    benchmark is made more informative (higher target base, longer horizon, or
    more sequences).

### 34. Convex Family-Sparse Inhibition Prototype Also Collapses On Inhibition-Only

We finally tested the actual convex direction that had been discussed:
- fix the inhibition feature dictionary,
- optimize all inhibition coefficients jointly with a convex objective,
- add a family-structured regularizer over triplet source unions.

The prototype solved a convex objective of the form

\[
\min_{\beta \ge 0}
\;
\sum_j a_j \beta_j
\;+\;
\sum_m w_m \lambda_0(t_m)\exp\!\bigl(-(X\beta)_m\bigr)
\;+\;
\lambda_1 \sum_j w_j \beta_j
\;+\;
\lambda_2 \sum_C \sqrt{\|\beta_{F(C)}\|_2^2 + \varepsilon},
\]

with projected gradient on the nonnegative orthant. This is a smooth convex
family-sparse inhibition block.

On `paper_ablation_inhibition_only` (using the pure-inhibition constant
baseline so that no excitation approximation is involved), a representative run
with `lambda1 = 0.01`, `lambda2 = 0.03` gave:

- true triplet coefficients:
  - `DFG = 0.0`
  - `CGH = 0.0`
- surrogate triplet coefficient:
  - `EFG = 0.0`

The selected pre-refit support nevertheless collapsed to a set of spurious
surrogate pair/triplet rules such as:
- `AB`
- `ABG`
- `ABH`
- `ACD`
- `ACH`
- `ADF`
- `AEF`
- `AFG`
- `BCD`
- `BCF`
- `BEF`
- `BGH`
- `CDF`

and after exact refit the final selected model degraded to just:
- `AB -> T : inhibition`

with:
- recall `0.0`
- precision `0.0`
- BIC `15055.43`

Interpretation:
- The convex family penalty by itself does **not** rescue true higher-order
  inhibition in this low-signal regime.
- It still prefers compact surrogate structures built from stronger local
  footprints.
- In fact it can be worse than the current greedy baseline, because the convex
  block sees many correlated surrogates simultaneously and shrinks toward a
  spurious but locally coherent family.

Conclusion:
- The issue is no longer “we have not tried a convex family-structured
  inhibition solver.”
- We have, and on the current `inhibition-only` benchmark it still fails.
- This strengthens the earlier conclusion that the main problem is the regime
  itself: pure inhibition at `target base = 0.05` is too weak for reliable
  higher-order truth recovery under the current feature dictionary.

### 35. Oracle Conditional Expected-BIC Check: `inhibition-only 0.05` Actually Does Contain Enough Signal

We then checked the official `paper_ablation_inhibition_only` benchmark in a
way that removes **all target-event sampling noise**.

Setup:
- Use the realized train **source trajectories** only.
- Treat the target process as an inhomogeneous Poisson process conditional on
  those source paths.
- For each support \(S \subseteq \{A,B,CD,EF,CGH,DFG\}\), optimize the
  **conditional expected log-likelihood** over \((\mu,\beta_S)\) with the true
  rule features fixed.

For inhibition-only with fixed features,

\[
\lambda_S(t) = \mu \exp\!\Bigl(-\sum_{u \in S} \beta_u x_u(t)\Bigr),
\]

and conditional on the source paths, the expected log-likelihood is

\[
\mathbb E[\log L_S]
=
\int \lambda^\star(t)\log \lambda_S(t)\,dt
- \int \lambda_S(t)\,dt.
\]

This can be optimized exactly for each support by convex optimization. We then
compare **expected BIC** using only the support size penalty
\((1 + |S|)\log n_{\mathrm{eff}}\), because the features are fixed at the true
generator kernels.

Result on the official `target base = 0.05` benchmark:
- The **best expected log-likelihood** support is the full truth:
  - `{A, B, CD, EF, CGH, DFG}`
- The **best expected BIC** support is also the full truth:
  - `{A, B, CD, EF, CGH, DFG}`

Relevant expected-BIC values:
- full truth `{A,B,CD,EF,CGH,DFG}`: `113654.15`
- `{A,B,CD,EF,CGH}`: `113756.33`
- `{A,B,CD,EF,DFG}`: `113782.90`
- compact `{A,B,CD,EF}`: `113895.84`
- compact `{A,B,CD}`: `114508.52`

So the margin over the compact lower-order support is large:
- `114508.52 - 113654.15 = 854.37` expected-BIC points in favor of the full
  truth support.

Contrast check at `target base = 0.3` on the same source trajectories gives the
same ranking, with an even larger margin.

Interpretation:
- The earlier fixed-support empirical refit comparison was **not** measuring
  the oracle model-selection problem cleanly.
- The current official `inhibition-only 0.05` benchmark is **not** a regime
  where truth is intrinsically unidentifiable or even BIC-disfavored.
- Instead, the regime still contains enough signal that, with the true feature
  dictionary fixed and support search done correctly, the full higher-order
  inhibition support is preferred.

Updated conclusion:
- `inhibition-only 0.05` remains hard, but the failure is **not** primarily
  that the benchmark is too weak in principle.
- The main remaining bottleneck is still **support search / inference**:
  current greedy, screening-based, and convex-surrogate paths fail to reach the
  oracle-best support even though that support is favored by conditional
  expected BIC.

### 36. Even With Observed Target Events, True Features Make Full Truth Win

To separate **target-event sampling noise** from **feature/kernel learning**
effects, we ran a second oracle check:

- keep the **observed sampled target events** from the official train split,
- keep the realized source paths,
- but replace the learned rule features with the **true generator features**,
- then optimize \((\mu, \beta_S)\) exactly for every support
  \(S \subseteq \{A,B,CD,EF,CGH,DFG\}\).

This is no longer a conditional expectation over target events; it is the
ordinary sampled log-likelihood, but with the true feature dictionary fixed.

Result on the official `target base = 0.05` train split:
- best sampled-log-likelihood support: full truth
- best sampled-BIC support: full truth

Relevant sampled-BIC values:
- full truth `{A,B,CD,EF,CGH,DFG}`: `113433.09`
- `{A,B,CD,EF}`: `113654.66`
- `{A,B,CD}`: `114300.46`
- `{A,B,CD,DFG}`: `114149.27`
- `{A,B,CD,CGH}`: `114141.96`

Interpretation:
- The failure is **not** mainly due to sampled target-event noise.
- The official `inhibition-only 0.05` split already contains enough signal
  that, if the correct rule features are available, both sampled BIC and
  conditional expected BIC prefer the full higher-order truth support.
- Therefore the remaining gap must come from the actual learning pipeline:
  support search alone, or the interaction of support search with
  **learned/misshaped kernels**, is what prevents recovery.

Most likely conclusion:
- the dominant remaining bottleneck is not “inhibition-only lacks signal,”
  but the coupling between
  1. greedy / approximate support search, and
  2. rule-specific kernel learning on the wrong intermediate supports.

### 37. Naive EM-Like Alternation Still Reinforces Surrogates

We tested a minimal EM-like alternating scheme on the official
`paper_ablation_inhibition_only` benchmark (`target base = 0.05`):

1. run a quick baseline support search,
2. with kernels fixed, shortlist top triplet families and do a small local
   family support search,
3. refit kernels on the updated support,
4. repeat once.

This was meant to directly test the hypothesis suggested by the oracle checks:
the main problem may be the coupling between support search and kernel learning.

Observed result:
- quick baseline support:
  - `{A, B, CD, EF, DFG}`
  - recall `5/6`, precision `1.0`
  - missing only `CGH`
  - BIC `14683.1103`
- triplet screening under the current kernels ranked:
  - `CGH = (2,6,7)` first, then several surrogates
- but the local family step selected the surrogate triplet `(2,4,7)` instead
  of the true missing triplet `CGH`
- after kernel refit, the support collapsed to:
  - `{A, B, CD}`
  - recall `3/6`, precision `1.0`
  - BIC worsened to `14913.7147`

Interpretation:
- This is not just a screening problem: the true missing triplet `CGH` did
  appear at the top of the triplet-screening list.
- The failure happens because a naive local support step can still choose the
  wrong surrogate family, and once that happens, the following kernel refit
  reinforces the wrong support rather than correcting it.
- So a simple `rule -> kernel -> rule` alternation is not enough.

Updated conclusion:
- The oracle checks remain important: with true features, the full truth
  support is optimal.
- But a naive EM-like alternating procedure does not reliably move the current
  pipeline toward that truth support.
- The support update step itself must be much more reliable / profile-aware;
  otherwise the kernel step simply sharpens the wrong surrogate support.

### 38. Hierarchical Family Search Helps `inhibition-only`, But Is Too Slow For `20/30 predicates`

We then tested the exact staged procedure that had been discussed:

1. hold the current kernels fixed,
2. shortlist top-K triplet families,
3. within each family, enumerate all \(2^7 = 128\) subset supports and choose
   the best **fixed-kernel exact BIC** support,
4. keep only the top 1-2 support changes,
5. run short kernel refits only for those few candidates,
6. accept the move only if the refit BIC really improves.

This was implemented as an external prototype on top of the official baseline.

Result on `paper_ablation_inhibition_only`:
- official baseline:
  - support `{A, B, CD}`
  - recall `3/6`
  - precision `1.0`
  - BIC `14913.41`
- hierarchical family search:
  - support `{A, B, CD, EF}`
  - recall `4/6`
  - precision `1.0`
  - BIC `14692.21`

So this is the first generic support-search variant that produced a real
improvement on the official `inhibition-only` benchmark without changing the
data.

However, the family choices were still not the true missing triplet families.
The best short-refit candidates came from surrogate triplet unions:
- `(4,5,7)` with chosen fixed-kernel support `{F, EF, FG}`
- `(2,4,5)` with chosen fixed-kernel support `{CE, EF, CEF}`

Both short-refit candidates improved the objective mainly by recovering the
missing pair `EF`, not by recovering `CGH` or `DFG`.

Runtime result on `paper_num_predicates_20`:
- the same exact procedure did not finish within roughly 7.5 minutes on a
  single run
- `paper_num_predicates_30` was not run afterward, because it is strictly
  larger and would be even less practical

Updated conclusion:
- The staged hierarchical procedure is directionally right: unlike the earlier
  naive EM-like pass, it can improve the official `inhibition-only` result.
- But it is still not enough to recover the missing triplets, and its current
  exact family-search implementation is too expensive for the `20/30`
  predicate benchmarks.

### 39. Observable `delta = 0.05` Certificate Does Not Yet Validate The Improved Inhibition Support

We then computed an oracle-free local certificate using a finite observable
candidate set on the official `paper_ablation_inhibition_only` benchmark.

Candidate supports:
- baseline `{A, B, CD}`
- improved `{A, B, CD, EF}`
- surrogate `{A, B, CD, F}`

Protocol:
- fit each support on the train split with exact fixed-support kernel/coef
  optimization
- evaluate average holdout criterion on the validation split
- compare the best candidate against the runner-up with an empirical Bernstein
  bound at `delta = 0.05`

Result:
- holdout ordering:
  1. surrogate `{A,B,CD,F}`: `15.0861`
  2. improved `{A,B,CD,EF}`: `15.0884`
  3. baseline `{A,B,CD}`: `15.0911`
- but the gaps are tiny:
  - surrogate vs improved: gap `0.00224`, bound `0.01659`
  - surrogate vs baseline: gap `0.00501`, bound `0.02228`

So at `delta = 0.05`, **none of these pairwise orderings are certified**.

Interpretation:
- the current observable neighborhood is too ambiguous for a meaningful
  confidence certificate
- even though hierarchical family search improved the official benchmark
  support from `{A,B,CD}` to `{A,B,CD,EF}`, that improvement is not yet
  statistically stable under an oracle-free holdout certificate
- this further supports the view that the current local search neighborhood is
  still not rich and well-separated enough

### 40. Repeated-Split Stability Selection Confirms Pair-Level Gains But Not Triplet Recovery

We then tested whether the hierarchical family search produces a stable gain
under repeated resampling, without using any oracle information.

Protocol:
- pool the official `paper_ablation_inhibition_only` train/val sequences
- run 5 repeated random `80/20` train/validation resplits
- on each split:
  - run the plain baseline
  - run the hierarchical family search
- aggregate rule selection frequencies and stable supports

Result over 5 repeats:
- baseline average recall / precision:
  - `0.567 / 1.000`
- hierarchical-search average recall / precision:
  - `0.733 / 1.000`

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

At a stability threshold of `0.6` (selected in at least `3/5` runs):
- baseline stable support:
  - `{A, B, CD, EF}`
- hierarchical stable support:
  - `{A, B, CD, EF}`
- both stable supports achieve:
  - recall `4/6 = 0.667`
  - precision `1.0`

Interpretation:
- repeated-split stability shows that the hierarchical family search does have
  a real effect: it consistently stabilizes recovery of the missing pair-level
  inhibition rules (`CD`, `EF`) relative to the plain baseline
- however, the effect does **not** extend to the true triplet inhibition rules
  (`CGH`, `DFG`)
- so the current staged family search is a credible improvement for pair-level
  inhibition recovery, but not yet a solution to higher-order inhibition
  recovery in the official `target=0.05` pure-inhibition regime

### 41. Frozen-Kernel Exact Add-Only Correction Removes Missing Rules Across The Official Benchmarks, But Still Over-Adds

We then tested a broader exact support-correction prototype on top of the
official learned baseline anchor.

Protocol:
- run the official baseline through active-set fitting, family refinement, and
  post-prune
- freeze the learned kernels at that anchor
- keep the current active support fixed and correct only the **inactive
  inhibition** rules
- consider every inactive singleton/pair/triplet whose local fixed-kernel score
  is inhibitory and has positive gain
- solve the exact fixed-kernel penalized objective over supports of the form
  `A union T` with `|T| <= 4`, where `A` is the anchor support
- run a short full refit only for the chosen addition set and accept it only if
  the refit BIC improves
- do **not** use oracle rules in the search; use ground truth only for final
  evaluation

Result on the full official benchmark suite:
- all `12/12` benchmark datasets ended with:
  - empty missing-rule set
  - recall `1.0`
- exact recovery with precision `1.0` on:
  - `kernel_triangular`
  - `kernel_exponential`
  - `num_predicates_10`
  - `num_predicates_20`
  - `ablation_excitation_only`
- one extra rule with precision `0.857` on:
  - `logical_clean_plus`
  - `kernel_gaussian`
  - `num_predicates_30`
  - `ablation_inhibition_only`
  - `ablation_mixed_sign`
- two extra rules with precision `0.778` on:
  - `logical_shared`
  - `logical_context`

Important hard-case outcomes:
- `paper_num_predicates_20`:
  - recovered the missing true triplet `DFG inh`
  - final support matched the full truth exactly
- `paper_num_predicates_30`:
  - recovered the missing true triplet `DFG inh`
  - but also added the extra surrogate `FNU inh`
- `paper_ablation_inhibition_only`:
  - recovered `EF inh`, `CGH inh`, and `DFG inh`
  - but also added the extra surrogate `AEF inh`

What we learned:
- This is the first oracle-free prototype that removes **all missing rules**
  across the official benchmark suite under the current learned-kernel
  pipeline.
- So the bottleneck is no longer that true higher-order inhibition is
  completely invisible to the learned representation.
- Instead, once the missing triplets become reachable, the main remaining
  failure mode is **over-admission of extra inhibitory proxies**.
- This strongly suggests that fixed-kernel exact correction is the right
  backbone, but the current **add-only** neighborhood is too asymmetric.
- The next step should be an exact `add/drop/swap` neighborhood correction, or
  an exact post-correction prune, so that extra rules are rejected by the same
  objective rather than left to heuristic cleanup.

### 42. Frozen-Kernel Exact Block Solver Is Directionally Right, But Still Too Expensive And Not Yet Strong Enough

We then tested a stricter prototype that moves closer to the intended
heuristic-free direction:

- keep `family refine` only as a warm-start kernel stage
- remove `post-prune`
- freeze the learned kernels after the warm-start stage
- run an exact **block** support solver on the frozen features
- solve excitation and inhibition supports separately, with exact
  support-fixed block fits and exact `add/drop` neighborhood comparisons
- use the exact block-solver output itself as the final reported model, without
  a later heuristic post-processing step

This prototype is still not the final desired solver, because the current
implementation dropped `swap` moves to keep runtime manageable.

Observed benchmark results:
- `paper_ablation_inhibition_only`:
  - runtime about `223s`
  - final support recovered `5/6`
  - precision `1.0`
  - the only missing rule was `CGH inh`
- `paper_logical_context`:
  - runtime about `1096s`
  - final support recovered all true rules
  - recall `1.0`
  - precision `0.778`
  - extra rules were `AB inh` and `BE inh`

Operational result on the full suite:
- we launched the same exact block solver across all official benchmarks
- but under the available resource budget, only the above completed runs
  finished quickly enough to be informative
- the remaining benchmarks were still running after a long wall-clock time, so
  the sweep was stopped instead of continuing to burn CPU

What we learned:
- The exact block-solver direction is **real**: even without `post-prune`, it
  can improve the hard pure-inhibition case from the old `3/6` regime up to
  `5/6` while keeping precision `1.0`.
- But the current `add/drop` neighborhood is still not strong enough to finish
  the hardest pure-inhibition recovery (`CGH inh` remained missing).
- At the same time, it can still over-admit extra inhibition rules on mixed
  logical benchmarks (`logical_context`).
- Most importantly, the current exact block implementation is still too slow
  for an all-benchmark sweep under practical resource limits.

Updated conclusion:
- Replacing heuristics with frozen-kernel exact block correction is
  directionally correct.
- However, **exactness alone is not enough**: we still need a better
  neighborhood design and/or stronger safe screening to make the solver both
  accurate and computationally practical.
- The next practical target is not merely "run the exact block solver longer",
  but "make the exact solver cheaper and sharper", likely through stronger
  admissible screening and a more selective neighborhood construction.

### 43. Swap + Stronger Screening Improves Stability On 9/12 Benchmarks, But The 20/30-Predicate Tail Remains A Severe Runtime Bottleneck

We then upgraded the frozen-kernel exact block prototype to include a stronger
neighborhood:

- keep `family refine` only as a warm-start stage
- remove `post-prune`
- freeze the learned kernels after warm-start
- add exact `swap` evaluation on top of exact `add/drop`
- use stronger inhibition screening before expensive support solves
  - derivative-positive screening
  - exact 1D scalar-gain screening for inhibition additions
- keep the final reported model equal to the exact block-solver output

Implementation note:
- during the sweep we found and fixed two empty-support bugs
  - empty excitation block column construction
  - empty inhibition support passed to `scipy.optimize.minimize`
- after those fixes, the rerun produced valid outputs for 9 benchmarks
  without runtime errors

Completed benchmark results so far (`9/12`):
- `paper_ablation_excitation_only`:
  - recall `1.0`, precision `1.0`
  - no missing rules, no extra rules
- `paper_ablation_inhibition_only`:
  - recall `0.833`, precision `1.0`
  - missing `CGH inh`
  - no extra rules
- `paper_ablation_mixed_sign`:
  - recall `1.0`, precision `1.0`
  - no missing rules, no extra rules
- `paper_kernel_robustness_exponential`:
  - recall `1.0`, precision `1.0`
  - no missing rules, no extra rules
- `paper_kernel_robustness_triangular`:
  - recall `1.0`, precision `1.0`
  - no missing rules, no extra rules
- `paper_logical_clean_plus`:
  - recall `1.0`, precision `0.857`
  - extra rule `L exc`
- `paper_logical_context`:
  - recall `1.0`, precision `0.778`
  - extra rules `ABD inh`, `AEG exc`
- `paper_logical_shared`:
  - recall `1.0`, precision `0.875`
  - extra rule `ACD inh`
- `paper_num_predicates_10`:
  - recall `1.0`, precision `1.0`
  - no missing rules, no extra rules

Still running / not yet completed at the time of this note:
- `paper_kernel_robustness_gaussian`
- `paper_num_predicates_20`
- `paper_num_predicates_30`

Operational observation:
- the remaining three runs are exactly the cases we expected to be hardest
- the `20/30`-predicate runs in particular grew to very large resident memory
  usage during exact neighborhood search
- so the new solver is clearly more practical than the earlier exact
  `add/drop` prototype on many datasets, but the tail runtime is still a major
  research bottleneck

What we learned:
- adding `swap` plus stronger inhibition screening is a real improvement
  relative to the earlier strict exact-block prototype
- on the completed subset, this version is much more stable:
  - `excitation_only`, `mixed_sign`, and both completed kernel-robustness
    benchmarks are exact
  - `logical_clean_plus` and `logical_shared` keep full recall with fewer
    extras than the older heuristic-heavy pipeline
- however, the hardest pure-inhibition failure remains exactly the same:
  `CGH inh` is still missing in `inhibition_only`
- so the main remaining open problems are now sharply separated:
  - **accuracy**: recover `CGH inh` without reintroducing extras
  - **scalability**: make exact swap neighborhoods tractable for the
    `20/30`-predicate cases

### 44. Parallel Exact Support Solves Give A Real Wall-Time Win Without Changing The Solution

To test a low-risk acceleration that preserves the exact solver semantics, we
stopped the long-running benchmark jobs and modified the exact block solver so
that independent support solves inside the `add/drop/swap` neighborhood are
evaluated in parallel.

What changed:
- the solver still evaluates exactly the same candidate supports
- the candidate counts and exact neighborhood logic are unchanged
- only the expensive `fit_exc_support(...)` and `fit_inh_support(...)` calls
  are dispatched in parallel batches
- BLAS threads are clamped to `1` inside the parallel region to avoid
  oversubscription

Validation experiment:
- benchmark: `paper_ablation_inhibition_only`
- same solver, same data, same `opt_steps=40`, same `max_rounds=8`
- compare:
  - `support_workers=1`
  - `support_workers=8`

Observed result:
- `support_workers=1`:
  - elapsed `476.0s`
  - recall `0.833`
  - precision `1.0`
  - missing `CGH inh`
- `support_workers=8`:
  - elapsed `348.1s`
  - recall `0.833`
  - precision `1.0`
  - missing `CGH inh`

Speed effect:
- wall time improved from `476.0s` to `348.1s`
- this is about a `1.37x` speedup, or roughly `27%` less wall time

What we learned:
- parallel support solving is a clean first acceleration because it does not
  change the exact neighborhood being searched
- the solution quality on the tested case is effectively unchanged
- the speedup is real but moderate
- this means parallel support solving alone is not enough to fix the
  `20/30`-predicate bottleneck
- the next bigger speed win is still likely to come from reducing the number
  of exact support solves, especially on the excitation side where screening is
  still weaker than on inhibition

### 45. Batched Exact-Same Screening Adds Another Small But Clean Speedup

After parallelizing support solves, we applied another acceleration that keeps
the exact neighborhood logic unchanged:

- excitation derivative screening was vectorized in batches
- inhibition derivative screening was vectorized in batches
- inhibition exact 1D scalar-gain checks were parallelized
- no candidate acceptance criterion was changed
- the support search still visits the same logical neighborhood as before

Validation on the same fastest completed benchmark:
- benchmark: `paper_ablation_inhibition_only`
- compare:
  - `support_workers=1` baseline
  - `support_workers=8` after parallel support solves
  - `support_workers=8` + batched screening

Observed results:
- baseline (`workers=1`):
  - elapsed `476.0s`
  - recall `0.833`
  - precision `1.0`
  - missing `CGH inh`
- parallel support solves only (`workers=8`):
  - elapsed `348.1s`
  - recall `0.833`
  - precision `1.0`
  - missing `CGH inh`
- parallel support solves + batched screening:
  - elapsed `323.6s`
  - recall `0.833`
  - precision `1.0`
  - missing `CGH inh`

Speed effect:
- relative to the original `workers=1` baseline:
  - `476.0s -> 323.6s`
  - about `1.47x` faster
- relative to support-parallelization only:
  - `348.1s -> 323.6s`
  - about `1.08x` faster

What we learned:
- batched exact-same screening is worth keeping
- it gives a smaller gain than parallel support solving, but still reduces
  wall time without changing the recovered support
- the combined clean acceleration is now material on the easy benchmark
- however, the gain is still far smaller than what we need for the
  `20/30`-predicate tail, so the next major improvement still has to come from
  stronger safe pruning of the swap neighborhood itself

### 46. On The Fastest Benchmark, `support_workers=16` Was The Best Exact-Preserving Setting We Tested

We then did a small worker-count sweep on the same benchmark
`paper_ablation_inhibition_only`, keeping the exact same solver logic and the
same batched screening implementation.

Observed wall times:
- `support_workers=1`: `476.0s`
- `support_workers=8`: `323.6s`
- `support_workers=16`: `319.9s`
- `support_workers=24`: `337.8s`

All four runs had effectively identical output:
- recall `0.833`
- precision `1.0`
- same missing rule `CGH inh`
- no extra rules

What we learned:
- for this small benchmark, the best exact-preserving configuration we tested
  is `support_workers=16`
- the improvement beyond `8` workers is real but small
- pushing to `24` workers is already too much and loses time to overhead
- so the current "fastest rigorous setting" for small cases is roughly:
  - batched screening
  - parallel support solves
  - `support_workers` around `16`

### 47. Matrix-Store Reuse Was Not Worth Keeping

We also tested a more invasive but still exact-preserving optimization:

- pre-pack all rule feature arrays into dense matrices once
- replace repeated column construction from `arrays_all` with column slicing

This looked attractive because the exact solver repeatedly rebuilds support
matrices during screening and support fitting.

However, on `paper_ablation_inhibition_only` with the already-optimized exact
solver (`support_workers=16`), this change was slightly slower in wall time
than the simpler batched-screening version, while producing the same result.

What we learned:
- on the tested case, matrix-store packing overhead outweighed the savings
- we therefore reverted this optimization rather than keeping extra code
- the best exact-preserving configuration remains:
  - batched screening
  - parallel support solves
  - `support_workers` around `16`

### 48. Current Status Summary As Of 2026-04-01

Current best exact-support-search direction:
- `family refine` only as warm-start
- no `post-prune`
- freeze learned kernels after warm-start
- exact frozen-kernel block correction for excitation/inhibition
- `add/drop/swap` neighborhood
- stronger inhibition screening
- exact-preserving speedups:
  - parallel support solves
  - batched screening
  - `support_workers` around `16`

What is now established:
- `inhibition-only` is not impossible in principle
  - true fixed features still make full truth win
- the main failure is not lack of signal, but support-search failure under
  learned kernels
- frozen-kernel exact correction is the right backbone
- the stricter exact-block direction works on many official benchmarks
- on the completed benchmark subset, the solver is often exact or near-exact

Best completed benchmark status from the latest strict sweep:
- exact (`recall=1.0`, `precision=1.0`):
  - `ablation_excitation_only`
  - `ablation_mixed_sign`
  - `kernel_exponential`
  - `kernel_triangular`
  - `num_predicates_10`
- full recall but with extras:
  - `logical_clean_plus`
  - `logical_context`
  - `logical_shared`
- still missing one true triplet:
  - `ablation_inhibition_only` misses `CGH inh`

Unresolved heavy-tail runtime cases from the strict sweep:
- `kernel_gaussian`
- `num_predicates_20`
- `num_predicates_30`

Current best exact-preserving speed result on the fastest benchmark
(`paper_ablation_inhibition_only`):
- original strict solver baseline: `476.0s`
- current best exact-preserving configuration: `319.9s`
- speedup: about `1.49x`
- recovered support unchanged

What remains hardest:
- accuracy:
  - recover `CGH inh` in `inhibition-only`
  - reduce extras in the logical benchmarks
- scalability:
  - make `20/30 predicates` tractable without giving up exact/provable support
    search

Current recommendation:
- keep the present exact-support-search backbone
- keep the exact-preserving speedups that already helped
- focus next on stronger **safe pruning / admissible upper bounds** for the
  swap neighborhood, since that is the most likely source of a step-change in
  runtime while keeping the method logically rigorous

### 49. Real Rerun Of The Strict Exact-Block Solver On 2026-04-01

We reran the stricter frozen-kernel exact block solver with actual benchmark
execution again on `2026-04-01`, this time explicitly allowing full machine
resources for this one experiment.

Observed completed results from the rerun:
- `paper_logical_shared`:
  - elapsed `558.5s`
  - recall `1.0`, precision `0.875`
  - extra rule `ACD inh`
- `paper_logical_context`:
  - elapsed `996.6s`
  - recall `1.0`, precision `0.778`
  - extra rules `ABD inh`, `AEG exc`
- `paper_logical_clean_plus`:
  - elapsed `1414.3s`
  - recall `1.0`, precision `0.857`
  - extra rule `L exc`
- `paper_kernel_robustness_triangular`:
  - elapsed `638.1s`
  - recall `1.0`, precision `1.0`
  - no missing rules, no extra rules

Critical hard-case rerun outcome:
- `paper_num_predicates_30` did **not** complete successfully under the strict
  exact-block solver
  - GPU run failed with `CUDA out of memory`
  - failure happened during the anchor active-set solve, before the final exact
    support result was produced
  - a follow-up CPU-only rerun stayed alive for more than `17` minutes while
    consuming about `2000%` CPU, but still did not finish producing a result

What this rerun establishes:
- the completed cases match the earlier qualitative picture rather than
  overturning it
- this stricter exact `add/drop/swap` direction is still a valid rigorous
  backbone
- but it has **not** yet surpassed the earlier frozen-kernel exact
  add-only correction in overall benchmark accuracy
- and it still has a serious runtime / memory tail on the hardest
  `20/30`-predicate-type cases

Current best-performing method overall as of this note:
- still the earlier **frozen-kernel exact add-only correction**
- reason:
  - it is still the only oracle-free path we tested that removed **all missing
    rules** across the full official benchmark suite
  - it achieved recall `1.0` on all `12/12` official benchmarks
  - it was exact with precision `1.0` on:
    - `kernel_triangular`
    - `kernel_exponential`
    - `num_predicates_10`
    - `num_predicates_20`
    - `ablation_excitation_only`
  - its remaining problem is extra-rule over-admission, not missing true rules

Current best rigorous symmetric-search direction:
- the stricter frozen-kernel exact block solver with:
  - warm-start-only family refine
  - frozen kernels
  - exact `add/drop/swap`
  - stronger inhibition screening
- reason:
  - it is the cleanest path toward admissible bounds / certified pruning
  - but at the moment it is still worse than the add-only correction in
    end-to-end practical benchmark performance

Working rule going forward:
- every materially new experiment should be appended to this note
- every such entry should explicitly state what the **current best-performing
  method** is at that time
- code paths that are experimentally dominated and no longer useful as either
  baselines or active directions should be removed rather than left behind

### 50. Speed Comparison On The Fastest Benchmark For Safe-Screen Exact Search

We directly tested the "less heuristic but hopefully faster" variant on the
fastest benchmark we have been using for exact-search speed checks:
`paper_ablation_inhibition_only`.

Resource policy used for this comparison:
- CPU only
- `12` threads (`OMP_NUM_THREADS=12`)
- `CUDA_VISIBLE_DEVICES=0` set, but this specific script still ran on CPU
- same local dataset reused for both runs

Comparison setup:
- baseline:
  - current frozen-kernel exact add-only correction
  - heuristic candidate filter (`rule_score` sign/gain)
  - capped exact search with `max_add = 4`
- variant:
  - same anchor and same final refit
  - root safe-screening using exact singleton gain on the frozen residual
  - uncapped exact add-only search

Observed result:
- baseline (`tmp_exact_ablation_inhibition_only_compare_baseline.json`):
  - elapsed `186.28s`
  - candidate count `58`
  - search nodes `13`
  - screening `0.72s`
  - search `15.41s`
  - refit `4.42s`
  - recall `1.0`, precision `0.857`
- safe-screen + uncapped (`tmp_exact_ablation_inhibition_only_compare_safe_screen.json`):
  - elapsed `208.92s`
  - candidate count `58`
  - search nodes `49`
  - screening `4.19s`
  - search `85.41s`
  - refit `4.49s`
  - recall `1.0`, precision `0.857`

Interpretation:
- on this fastest benchmark, the rigorous root safe-screen did **not** remove
  any additional candidates beyond the existing heuristic filter
- because the candidate count stayed at `58`, the uncapped search only opened
  more nodes (`13 -> 49`) and paid a much larger search cost
  (`15.41s -> 85.41s`)
- end-to-end accuracy stayed exactly the same
- therefore this particular "safe-screen + uncapped" replacement is
  experimentally dominated on the fastest benchmark and should **not** remain
  as an active code path

Current best-performing method overall as of this note:
- still the earlier **frozen-kernel exact add-only correction**
- this speed comparison did not change that conclusion
- on `paper_ablation_inhibition_only`, the baseline remained both faster and no
  worse in recall/precision than the more rigorous candidate/search variant

### 51. Decision After The Failed Root Safe-Screen Speed Comparison

The latest speed comparison narrows the next research move substantially.

What should be dropped:
- do **not** continue the generic "root safe-screen + uncapped add-only exact
  search" path
- on the fastest benchmark it gave:
  - no candidate reduction
  - much larger search (`13 -> 49` nodes)
  - worse wall time
  - no accuracy gain

What still looks viable:
- keep the already validated exact-preserving systems speedups:
  - parallel support solves
  - batched screening
  - moderate worker count (`~16` on the small benchmark)
- shift the algorithmic focus away from root-level global candidate admission
  and toward **smaller ambiguity neighborhoods**
- inside those neighborhoods, focus on:
  - stronger admissible upper bounds for `swap` branches
  - tighter candidate pruning inside the block
  - more selective ambiguity motifs instead of whole same-sign overlap
    components

Working interpretation:
- the problem is not simply "heuristic filter vs safe filter"
- the main cost comes from searching the wrong neighborhood too broadly
- therefore the next serious direction is:
  - selective ambiguity-block definition first
  - then certified pruning / exact search inside that smaller block

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

#### 132. Prototype: Residualized Joint Rule-Block Pruning

Fast follow-up prototype after `joint rule-block drop-one`:

- same deterministic baseline start
- same joint coefficient + kernel-height re-optimization
- same full kernel-dimension BIC
- but a drop is now allowed only if the rule's current feature is mostly
  explained by active strict-subset ancestors

Operationally:

- for each active rule, compute a residualized feature ratio against active
  strict-subset ancestors
- order-1 rules are not ancestor-gated
- order-2/3 rules are only eligible for deletion if their residualized ratio is
  small

This is still only a prototype, but it directly targets the current failure
pattern:

- keep deleting obvious extras
- stop deleting true high-order interaction rules whose feature remains
  nontrivial after lower-order residualization

#### 133. Principled Next Step: Canonical Interaction-Increment Rule Blocks

The residual-ratio gate improved the over-pruning problem, but it still failed
to separate:

- false high-order compensation rules
- true high-order interaction rules

in `logical_context`.

Why:

- the current residualization is only against active strict-subset ancestors
- it uses raw conjunction feature vectors
- therefore it does not define a canonical "pure interaction" object

The mathematically cleaner next step is to replace raw conjunction rules by
**canonical interaction-increment blocks**.

Construction:

1. For each subset `U`, define the raw rule block tangent `T_U` for
   coefficient + kernel-shape contribution.
2. Let `L(U)` be the span of all proper-subset blocks `T_V`, `V ⊂ U`,
   under a fixed weighted inner product induced by the reference model
   (Fisher / expected Hessian metric).
3. Define the canonical interaction block

   - `Psi_U = Proj_{L(U)^\perp}(T_U)`

This is the rule-block analogue of Möbius / ANOVA decomposition on the subset
lattice, but in the model's local information geometry.

Interpretation:

- `Psi_U = 0` means subset `U` adds no interaction information beyond lower
  orders
- `Psi_U != 0` means subset `U` carries genuine higher-order interaction signal

Why this is better than the current residual-ratio gate:

- it residualizes against the full lower-order closure, not only whichever
  ancestors happen to be active
- it defines a canonical object for each subset, independent of the current
  greedy support path
- false extras that only compensate lower-order misfit should have small
  `||Psi_U||`
- true high-order rules should have nontrivial projected block energy

Deletion principle:

- drop rule `U` only if
  1. its canonical interaction block test is below the certified threshold
  2. the full joint profiled objective improves after deletion

This is theorem-friendly.

Possible theorem shape:

- assume support coverage by the deterministic baseline family
- assume each true rule has nonzero canonical interaction block norm bounded
  below by `gamma`
- assume each false extra rule has canonical interaction block norm at most
  `gamma_0 < gamma`
- assume uniform concentration of the empirical Fisher/block scores around their
  population values
- assume profile-objective margin for deletion decisions

Then with high probability:

- true rules are not deleted
- false extras with vanishing canonical interaction increment are deleted

This is the first direction in the project that directly targets the logical
object we actually want to recover:

- not raw conjunction usefulness
- but genuine signed higher-order interaction increments

#### 134. Partial Result: Canonical Increment Prototype On Logical Context

Current partial result for the same-sign proper-subset closure prototype:

- `logical_context`
  - baseline: recall `1.0`, precision `0.875`
  - result: recall `0.8571`, precision `1.0`
  - accepted drops:
    - `A and E and G -> T : excitation`
    - `E and F and G -> T : inhibition`

So this first canonical-closure approximation still behaves like the earlier
joint rule-block prune: it removes the false extra, but it also removes the
true high-order inhibition.

Interpretation:

- using same-sign proper-subset closure is still not the right canonical object
- the true inhibition rule's block remains too explainable by lower-order
  closure under the current metric
- a correct canonical interaction increment will likely need:
  - sign-coupled closure
  - amplitude-weighted tangent blocks
  - Fisher / score metric rather than raw feature projection alone

#### 135. Root-Cause Synthesis And The Next Mathematical Direction

The last three local deletion prototypes isolate the problem clearly.

1. `joint rule-block drop-one`
   - uses the right continuous information (coefficient + kernel height)
   - but no interaction residualization
   - result:
     - removes obvious extras
     - also removes true high-order inhibition in `logical_context`

2. `active-ancestor residualized gate`
   - closure is too weak and path-dependent
   - result:
     - protects the true high-order inhibition
     - also protects the false extra excitation

3. `same-sign proper-subset canonical closure`
   - closure is stronger, but still the wrong object
   - result:
     - again removes both the false extra and the true high-order inhibition

Therefore the real aliasing mechanism is not:

- "same-sign high-order conjunctions look like same-sign lower-order ones"

The real aliasing mechanism is:

- **signed compensation in the local intensity tangent geometry**

In `logical_context`, the false rule

- `A and E and G -> T : excitation`

survives because it can compensate residual suppression created by the active
inhibition structure. That is a cross-sign phenomenon. A same-sign raw-feature
closure cannot represent it correctly.

So the next mathematically coherent direction is:

### Signed Fisher-Mobius Interaction Blocks

For each active rule `r = (U, sign)`, define its local amplitude-weighted
tangent block in the actual model geometry:

- excitation:
  - `delta lambda_r = exp(-I_0) * delta eta_r`
- inhibition:
  - `delta lambda_r = -lambda_0 * delta I_r`

with block parameterization

- `u_r = (beta_r, beta_r * delta_r)`

so that kernel-shape perturbations remain identified under the null.

Then define the predecessor set of `r` as:

- all active / certified rules `q` such that `sources(q) ⊂ sources(r)`
- **with either sign**

and recursively orthogonalize in the Fisher metric:

- `Psi_r = T_r - Proj_{span{Psi_q : q prec r}} T_r`

This is the signed, lattice-residualized canonical interaction block.

Why this should fix the current pattern:

- false extras that only act as cross-sign compensation should lie mostly in the
  span of lower-order signed tangent blocks, so `||Psi_r||` should be small
- true high-order inhibition rules should retain a nontrivial signed interaction
  increment after projecting out all lower-order signed predecessors

Deletion principle:

- drop `r` only if
  1. the signed canonical block test for `Psi_r` is below threshold
  2. the full joint profiled objective improves after deletion

Theorem skeleton:

- deterministic baseline family contains the true support
- population signed canonical block norm of each true rule is at least `gamma`
- each false extra has signed canonical block norm at most `gamma_0 < gamma`
- empirical Fisher / score blocks concentrate uniformly around population
  blocks
- profile objective has a positive deletion margin

Then, with high probability:

- false extras are deletable
- true rules are nondeletable

This is now the cleanest theorem-friendly direction because it directly targets
the observed failure mode:

- not raw conjunction overlap
- not same-sign lower-order explainability
- but signed local compensation in the true model geometry

#### 130. Result: Joint Rule-Block Drop-One Pruning On 3 Hard Cases

Prototype:

- deterministic baseline support from `safe-swap confset`
- joint re-optimization of active coefficients + rule-specific kernel heights
- full kernel-dimension BIC
- exact drop-one neighborhood
- fixed-point deletion

Results:

- `logical_clean_plus`
  - baseline: recall `1.0`, precision `0.8571`
  - result: recall `1.0`, precision `1.0`
  - accepted drop:
    - `L -> T : excitation`

- `kernel_gaussian`
  - baseline: recall `1.0`, precision `0.75`
  - result: recall `1.0`, precision `1.0`
  - accepted drops:
    - `B and C and D -> T : inhibition`
    - `G and H -> T : inhibition`

- `logical_context`
  - baseline: recall `1.0`, precision `0.875`
  - result: recall `0.8571`, precision `1.0`
  - accepted drops:
    - `A and E and G -> T : excitation`
    - `E and F and G -> T : inhibition`

Interpretation:

- using joint coefficient + kernel-height information at deletion time is
  genuinely powerful
- it fixes the two clean extra-rule cases (`logical_clean_plus`,
  `kernel_gaussian`)
- but in `logical_context` it over-corrects: once the false excitation is
  removed, the same criterion still prefers dropping the true high-order
  inhibition as well

So this prototype is not yet the final method, but it is the strongest direct
evidence so far that **final deletion must use joint rule-block information**
rather than scalar frozen-feature criteria.

#### 131. Next Direction: Hierarchically Residualized Rule-Block Deletion

The `joint rule-block drop-one` result suggests the right next step:

- keep using joint coefficient + kernel-height information
- but do **not** test raw active rule blocks directly
- instead test each active rule through its **pure interaction increment**

Why this is needed:

- `logical_clean_plus` and `kernel_gaussian` show that raw joint rule-block
  deletion is strong enough to remove obvious extras
- `logical_context` shows that raw joint rule-block deletion is still too
  aggressive against a true high-order inhibition rule
- this means the deletion statistic still conflates:
  - lower-order effects already explained by simpler rules
  - genuine higher-order interaction information

The correct refinement is therefore hierarchical residualization.

For each active rule `r` with source set `U_r`, define an ancestor nuisance set
`A(r)` containing active rules whose source sets are strict subsets of `U_r`
and/or overlap in the same local interaction region. Then:

- build the local amplitude-weighted block parameter
  - `u_r = (beta_r, beta_r * delta_r)`
- build the corresponding local tangent block for coefficient + kernel shape
- project that block onto the orthogonal complement of the nuisance span
  generated by `A(r)`

This produces a **pure interaction block** for rule `r`.

Deletion rule:

- only allow dropping rule `r` if
  1. the residualized block test fails to show nonzero interaction increment
  2. the full joint profiled objective also improves after dropping `r`

Why this should fix the current pattern:

- false extras like `A and E and G -> T : excitation` are likely acting mostly
  as lower-order compensation terms, so their residualized pure-interaction
  block should be weak
- true rules like `E and F and G -> T : inhibition` should still retain a
  nontrivial residualized interaction block after accounting for `E` and
  `E and F`

This is mathematically cleaner than another raw BIC tweak because it explicitly
tests the object we actually care about:

- not "is this raw conjunction useful somewhere?"
- but "does this rule contribute unique higher-order interaction information
  beyond its lower-order ancestors?"

This is also still computationally feasible:

- each rule only projects against a small local ancestor block
- deletion remains drop-one / local-neighborhood
- no global subset enumeration is required

#### 129. New Prototype: Joint Rule-Block Pruning Around The Deterministic Baseline

Current fast prototype:

- start from deterministic `safe-swap confset` baseline support
- keep the support fixed initially, but then re-optimize the whole active set
  with **joint coefficient + rule-specific kernel-height optimization**
- score supports with **full kernel-dimension BIC**
- evaluate the exact drop-one neighborhood of the active support
- accept the best deletion if and only if it improves the full profiled
  objective
- repeat to a fixed point

Why this direction is worth testing:

- it directly uses the information that the current retained selector throws
  away at the last stage
- it treats each active rule as a block of continuous parameters rather than as
  a scalar coefficient only
- it is still much cheaper than global subset search, because only the
  drop-one neighborhood is exactly re-fit

Hard-case benchmark set for this prototype:

- `logical_clean_plus`
- `logical_context`
- `kernel_gaussian`

#### 124. Logical Context: Why The Extra Excitation Persists

Ground-truth rules in `paper_logical_context.yaml` are:

- `A -> T : excitation`
- `B -> T : inhibition`
- `A and C and D -> T : inhibition`
- `B and C and D -> T : excitation`
- `E -> T : inhibition`
- `E and F -> T : excitation`
- `E and F and G -> T : inhibition`

In the deterministic baseline and in the current efficient-score closed-family
prototype, the extra rule is:

- `A and E and G -> T : excitation`

This is not a new artifact introduced by the efficient-score selector. The
selector inherits this rule from the deterministic baseline support and then
fails to remove it.

What the current prototype does on `logical_context`:

- baseline support:
  - excitation: `A`, `E and F`, `A and E and G`, `B and C and D`
  - inhibition: `B`, `E`, `A and C and D`, `E and F and G`
- active-rule necessity test marks **all 8 active rules as core**
- in particular, `A and E and G -> T : excitation` gets a very large keep-stat
  (`117.645...`) with essentially zero p-value
- therefore the local family never even evaluates supports that drop this extra
  excitation

Interpretation:

- the extra excitation is being treated as a statistically indispensable
  conditional effect **given the current active support**
- because the active support already contains both `E -> T : inhibition` and
  `E and F and G -> T : inhibition`, the rule `A and E and G -> T : excitation`
  can act like a context-specific compensation term inside the inhibited region
- the current active-side efficient-score test is therefore still too local:
  it certifies indispensability relative to the current nuisance support, not
  relative to the true logical decomposition

So the precise failure mode is:

- the baseline search introduces `A and E and G -> T : excitation`
- the current projected keep-test then over-protects it
- the final family search cannot remove it because it is placed in `core`

This means the next fix should not focus on additional add-candidates for
`logical_context`; it must instead weaken or replace the **active-rule core
certificate** so that false active excitation rules are allowed back into the
variable family.

#### 125. Why The Extra Excitation Is Hard To Distinguish

The `logical_context` extra rule is not surviving because it is globally
better-understood logical structure. It survives because the current model and
selector only ask a **conditional incremental-fit** question.

Let the intensity be

`lambda(t) = (mu + E(t)) exp(-I(t))`.

For `logical_context`, the relevant true rules around the `E/G` region are:

- `E -> T : inhibition`
- `E and F -> T : excitation`
- `E and F and G -> T : inhibition`

The extra rule is:

- `A and E and G -> T : excitation`

Why this can look real under the current criterion:

1. Product-bounded conjunction features overlap strongly.
   - each rule feature is a product of bounded source activities
   - `A*E*G` is not orthogonal to the span created by `A`, `E`, `EF`, `EFG`
   - after kernel smoothing, these supports overlap in time rather than being
     disjoint logical atoms

2. Excitation can compensate for inhibition misfit.
   - because excitation sits in the additive part and inhibition sits in the
     multiplicative exponent, a false excitation can act as a local
     compensation term for over-suppression created by the current inhibition
     fit
   - in other words, `AEG excitation` can partially repair residual error in
     the same region where the model is trying to explain `E` and `EFG`
     inhibition

3. The current keep-test is conditional, not structural.
   - the efficient-score keep statistic asks:
     "given the current active support, is this coefficient still nonzero?"
   - it does **not** ask:
     "is this rule part of the minimal logical interaction decomposition?"
   - once the baseline introduces `AEG excitation`, the projected keep-test can
     certify it as indispensable even when it is only a surrogate for a better
     redistribution among true rules

So the failure is a form of finite-sample / finite-family aliasing:

- the extra rule has enough residualized predictive power to look necessary
- but that necessity is only relative to the current parameterization
- it is not the same as logical identifiability

This means the correct next direction is not another stronger local p-value.
It is a selector built on a representation that separates:

- lower-order main effects
- genuine higher-order interaction increments

more cleanly than raw conjunction features do.

#### 126. Why Later Fine-Tuning Does Not Automatically Remove Extra Rules

It is tempting to hope that an extra rule found during support construction will
disappear later when coefficients and kernels are re-optimized. In the current
pipeline, that is generally false.

Reason:

- the later optimization stage is a **fixed-support refit**, not a support
  selection stage
- once a rule enters the chosen support, the refit only adjusts continuous
  parameters while keeping that support fixed

In code:

- `refit_fixed_support_pair(...)` repeatedly calls `fit_exc_support(...)` and
  `fit_inh_support(...)` with the same `exc_support` / `inh_support`
- excitation coefficients are optimized under nonnegativity bounds
- inhibition coefficients are optimized under nonnegativity bounds

So the refit can:

- shrink a coefficient
- redistribute weight across active rules
- re-balance excitation and inhibition

but it does **not**:

- remove a rule from support
- apply a sparsity penalty that would force exact zeros
- revisit the combinatorial support decision unless an explicit prune/search
  stage is run again

This is especially important for false active rules like
`A and E and G -> T : excitation` in `logical_context`:

- once the baseline includes the rule
- later fixed-support refits may actually stabilize it, because the extra rule
  can absorb residual misfit left by the current true-rule parameterization

Therefore:

- post-hoc coefficient/kernel fine-tuning is not a reliable mechanism for
  removing false logical rules
- if we want extra rules to disappear, the algorithm needs an explicit
  support-level removal criterion or a support-changing sparse regularization
  stage

#### 127. What Is Actually Learned Well In The Current Retained Baseline

For the current retained theorem-friendly baseline, it is important to separate
two different questions:

1. are continuous parameters optimized well **conditional on a fixed support**?
2. does the full pipeline recover the correct logical decomposition?

The answer is:

- conditional coefficient refitting is generally well-posed and stable
- full logical decomposition is still imperfect
- temporal kernels are **not** jointly re-learned in the final exact support
  refit stage of the retained baseline

More precisely:

- `refit_fixed_support_pair(...)` keeps `exc_support` / `inh_support` fixed and
  alternates `fit_exc_support(...)` with `fit_inh_support(...)`
- these solves are continuous nonnegative constrained optimizations
- but they operate on fixed precomputed feature arrays `arrays_all`

So in the current retained baseline:

- `mu` and rule coefficients are re-fit for the chosen support
- temporal kernel shapes are inherited from the earlier anchor / feature
  construction stage, not freshly optimized at final selection time

Implication:

- if support is already close to correct, coefficient fitting can look good and
  the matched rules can be represented well
- but if support contains an extra rule, the refit can assign that rule a
  meaningful positive coefficient instead of driving it away, because the rule
  is allowed to absorb residual misfit
- therefore "coefficients look nontrivial after refit" is not evidence that the
  logical rule is genuine

So the honest summary is:

- current coefficient learning: often numerically good conditional on support
- current temporal-kernel learning at final selection: not really happening in
  the retained theorem-friendly line
- current support identification: still the real bottleneck

#### 128. A Principled Rule-Block Test Using Both Coefficient And Kernel Shape

Yes: in principle, the information used to fit rule coefficients and
rule-specific kernel shape can be turned into an explicit mathematical device
for dropping extra rules.

What is missing in the current retained selector:

- coefficient information is used only conditionally on a fixed support
- kernel-shape information is used upstream to build rule arrays, but not as a
  joint identifiability object at the final support decision

The right object is a **rule block**, not a scalar coefficient.

For an active rule `r`, let:

- `beta_r` be the rule coefficient
- `eta_r` be the free rule-specific kernel-height coordinates

Directly testing `(beta_r, eta_r)` is awkward because `eta_r` is not identified
when `beta_r = 0`. The fix is to use an amplitude-weighted local
reparameterization:

- `u_r = ( beta_r, beta_r * delta_r )`
- where `delta_r = eta_r - eta_r^0` is the deviation from a reference kernel
  shape `eta_r^0`

Then, after first-order expansion around `eta_r^0`, the contribution of rule
`r` becomes locally linear in `u_r`. Under the null "rule absent", we simply
have:

- `H0: u_r = 0`

and the null is identified.

This enables a support-deletion test built from:

- the block score for `u_r`
- the nuisance-projected block Fisher / Schur complement after removing the
  tangent span of all other active rules

Concretely, the deletion statistic should be of the form:

- `T_r = S_eff(r)^T I_eff(r)^{-1} S_eff(r)`

or the corresponding block Wald version from the fitted `u_r`.

Why this should help for false extras such as
`A and E and G -> T : excitation`:

- if the rule is only acting as a compensation surrogate, most of its local
  tangent directions lie inside the nuisance span of the true active rules
- after nuisance projection, its effective block information should collapse
- then the full rule block fails the keep-test, even if the scalar coefficient
  alone is positive

Why this is better than current scalar keep-tests:

- scalar keep-tests can protect a false rule whose coefficient is useful for
  residual correction
- block tests ask whether the **entire local rule manifold** carries unique
  information beyond the current active support

This is the cleanest mathematically coherent route to using both coefficient
and kernel-shape learning information for extra-rule deletion.

### 98. The Remaining Higher-Order Inhibition Misses Come From Sign-Asymmetric Geometry, So The Next Theorem-Friendly Direction Should Use Canonical Tangent-Intensity Features

Date:
- 2026-04-07

Question:
- after the refined structural-risk selector also failed on several
  higher-order inhibition cases, what is the right next direction if we want
  a mathematically justified method that compares excitation and inhibition on
  the same scale?

Observation:

The benchmark generator and the current theorem-friendly learner both use the
same nonlinear intensity family

\[
\lambda(t) = (\mu + E(t)) \exp(-I(t)).
\]

So the sign asymmetry is not an implementation accident. It is built into the
model class itself.

For a rule activation feature \(p_r(t)\), the pathwise intensity derivatives
are:

\[
\frac{\partial \lambda(t)}{\partial w_r^{\mathrm{exc}}}
= p_r(t)\exp(-I(t)),
\]

\[
\frac{\partial \lambda(t)}{\partial w_r^{\mathrm{inh}}}
= -p_r(t)(\mu + E(t))\exp(-I(t))
= -p_r(t)\lambda(t).
\]

This immediately explains the new failure pattern:

- excitation is measured on the additive-intensity scale
  - derivative magnitude is proportional to \(\exp(-I)\)
- inhibition is measured on the attenuation scale
  - derivative magnitude is proportional to \(\lambda\)
- therefore, in low-intensity regimes or in regions already suppressed by
  lower-order inhibition, an additional true higher-order inhibition rule has
  very small local leverage even when it is genuinely part of the truth

That is exactly the regime where the recent selectors fail:

- `logical_context`
- `kernel_exponential`
- `num_predicates_20`
- especially `ablation_inhibition_only`

So the problem is now best understood as a **sign-asymmetric local geometry
problem**, not primarily as a search problem.

#### 98.1. Why Raw Coefficient Comparison Is The Wrong Scale

The current family selection directions still compare supports through
parameters that live in different coordinates:

- excitation coefficients change the additive term \(\mu + E\)
- inhibition coefficients change the multiplicative attenuation term \(I\)

Even if the final objective is the correct nonlinear likelihood, finite-step
profile fitting and local screening operate in these unequal coordinates.

As a result, higher-order inhibition is repeatedly under-valued because:

1. its feature \(p_r\) is already sparse as a conjunction,
2. it often becomes active exactly where the current intensity is small,
3. the derivative scale for inhibition contains the extra factor \(\lambda\).

So the next method should not try to make the raw nonlinear coefficients look
similar. Instead it should compare both signs in a **common intensity unit**.

#### 98.2. Canonical Tangent-Intensity Features

The natural common unit is the signed first-order intensity effect itself.

Fix a reference model \(\theta_0\) with intensity \(\lambda_0(t)\). For each
candidate rule \(r\), define the canonical tangent-intensity feature
\(\xi_r(t)\) by the Gateaux derivative of \(\lambda\) at \(\theta_0\):

\[
\xi_r(t)
= \left.\frac{\partial \lambda_\theta(t)}{\partial \theta_r}\right|_{\theta=\theta_0}.
\]

Concretely:

- excitation:
  \[
  \xi_r^{\mathrm{exc}}(t)=p_r(t)\exp(-I_0(t))
  \]
- inhibition:
  \[
  \xi_r^{\mathrm{inh}}(t)=-p_r(t)\lambda_0(t)
  \]

Now both signs are represented in the **same output space**:

- both are infinitesimal changes in intensity
- both can be measured with the same empirical KL / Fisher metric
- both can be screened and selected with the same structural-risk machinery

This is the right sense in which excitation and inhibition can be compared
“equally” without changing the true nonlinear model.

#### 98.3. The Next Method: Tangent-Intensity Family Selection

The next theorem-friendly direction should therefore be:

1. keep the current best baseline
   - `safe-swap confset`
2. let \(\hat S\) be its active support
3. build the finite family \(F(\hat S)=\{S: S\subseteq \hat S\}\)
4. choose a reference model \(\theta_0\)
   - either the baseline fit itself
   - or a mean-field reference using empirical \(\bar E,\bar I,\Lambda\)
5. build tangent-intensity features \(\{\xi_r\}_{r\in \hat S}\)
6. for each candidate support \(S\), solve the convex profiled surrogate

\[
\tilde \lambda_S(t)=\lambda_0(t)+\sum_{r\in S}\alpha_r \xi_r(t),
\qquad \alpha_r \ge 0,
\]

with positivity constraints on the event/grid design points
7. score each support by

\[
\hat Q_n^{\mathrm{tan}}(S)
= \hat L_n^{\mathrm{tan}}(S)+\lambda_n d(S)
\]

8. choose the minimizer over the finite family
9. refit the original nonlinear model only once on the chosen support

This has three advantages:

- excitation and inhibition are compared on the same canonical scale
- the inner support evaluation becomes convex and much faster
- the final theorem is cleaner because the selection problem is a finite-family
  convex structural-risk problem plus a controlled nonlinear remainder

#### 98.4. Why This Direction Is Proof-Friendly

Let \(Q^\star(S)\) be the population structural risk of the true nonlinear
model, and let \(Q^{\mathrm{tan}}(S)\) be the population structural risk of the
tangent surrogate.

The key decomposition is:

\[
Q^\star(S)-Q^\star(S^\star)
= \bigl(Q^{\mathrm{tan}}(S)-Q^{\mathrm{tan}}(S^\star)\bigr)
+ \mathrm{Rem}(S)-\mathrm{Rem}(S^\star),
\]

where \(\mathrm{Rem}(S)\) is the nonlinear linearization remainder.

So the exact-recovery route is:

1. finite-family concentration for \(\hat Q_n^{\mathrm{tan}}(S)\),
2. uniform bound on the linearization remainder over \(F(\hat S)\),
3. positive population margin for \(S^\star\) inside the certified family.

Then, if

\[
\inf_{S\neq S^\star}
\bigl(Q^{\mathrm{tan}}(S)-Q^{\mathrm{tan}}(S^\star)\bigr)
>
2\Delta_n + 2\varepsilon_{\mathrm{lin}},
\]

where:

- \(\Delta_n\) is the empirical concentration error
- \(\varepsilon_{\mathrm{lin}}\) is the uniform linearization remainder

the selected support is exactly \(S^\star\) with high probability.

So this direction is theorem-friendly in a way the recent local heuristics were
not:

- the family is explicit
- the selector is exact over that family
- the inner support solve is convex
- the remaining approximation error is isolated into one explicit remainder
  term

#### 98.5. Why This Should Be Faster

The recent failures all used repeated nonlinear support-specific profile refits.

That is expensive because every support evaluation re-optimizes:

- the base rate
- rule coefficients
- often kernel-height geometry indirectly

The tangent-intensity selector would instead make the inner support solve a
convex affine-intensity problem over a fixed design dictionary.

So speed should improve through:

- one reference fit
- one tangent-dictionary construction
- fast convex support evaluation over \(F(\hat S)\)
- only one final nonlinear refit after support selection

#### 98.6. Updated Research Direction

The most principled next step is now:

- **certified finite-family selection over canonical tangent-intensity
  features**

This keeps the current best theorem-friendly search (`safe-swap confset`) and
changes only the final support-comparison geometry.

What should be implemented next:

1. construct tangent-intensity features for the baseline active support
2. evaluate the hard cases in that canonical surrogate family
3. verify whether the higher-order inhibition rules stop being systematically
   under-valued

No new dead-end code was kept in this step.

### 99. First Empirical Check Setup For Canonical Tangent-Intensity Family Selection

Date:
- 2026-04-07

The first implementation test of the new direction is restricted to the three
most diagnostic cases:

- `logical_clean_plus`
- `logical_context`
- `ablation_inhibition_only`

The construction is:

1. run the current best theorem-friendly baseline
   - `safe-swap confset`
2. take its final active support `\hat S`
3. rebuild a common reference fit on that support
4. define canonical tangent-intensity features
   - excitation: `p_r * exp(-I_0)`
   - inhibition: `-p_r * lambda_0`
5. form the affine surrogate family over all subsets of `\hat S`
6. solve each support exactly inside that affine family with nonnegative
   coefficients and positivity constraints
7. select by validation BIC inside the tangent family
8. refit the original nonlinear model once on the chosen support

Important scope note:

- the outer support family is explicit and non-heuristic
- the inner tangent-family solve is convex
- the only remaining numerical approximation is the constrained optimizer used
  for each convex support problem

So this experiment directly tests whether the sign-symmetric tangent-intensity
geometry fixes the repeated higher-order inhibition under-valuation.

#### 99.1. First Run Was Invalid Because The Baseline Settings Did Not Exactly Match `safe-swap confset`

The first tangent-intensity run should **not** be interpreted.

Reason:

- the wrapper that called the baseline exact solver accidentally used
  `inh_block_swap_only = False`
- but the validated `safe-swap confset` baseline uses
  `inh_block_swap_only = True`

Empirical symptom:

- on `ablation_inhibition_only`, the tangent run baseline started from a
  5-rule inhibition support
- while the validated saved baseline for the same benchmark is exact with 6
  inhibition rules

So the baseline supplied to the tangent selector was weaker than the true
reference method.

Conclusion:

- the first tangent-intensity run was invalid as a baseline-controlled
  comparison
- the run was discarded
- the wrapper was corrected before rerunning

### 100. `inh_block_swap_only` Is Not “No Add”; It Only Prevents Accepting Pure Inhibition Drops Inside The Local Inhibition Neighbor Step

Date:
- 2026-04-07

Clarification:

The option name `inh_block_swap_only` is easy to misread.

It does **not** globally forbid adding inhibition rules.

In the local inhibition neighbor routine:

- inhibition adds are still screened and evaluated
- inhibition swaps are still screened and evaluated
- what changes is only whether a **pure drop** candidate is allowed to win
  immediately inside that local neighbor step

Code-level effect:

- add states are always evaluated
- swap states are always evaluated
- but when `inh_block_swap_only = True`, the dropped support by itself is not
  accepted as the current best local move

So the operational meaning is:

- “do not let the local inhibition block step shrink support by pure deletion;
  only let it replace inhibition structure at the same local stage”

Why this was introduced:

- inhibition rules overlap heavily under
  `\lambda(t) = (\mu + E(t)) \exp(-I(t))`
- local pure deletions can look too attractive because they reduce model
  complexity immediately
- swap comparisons are cleaner because they keep local support size fixed and
  ask a more structural question:
  - “is candidate rule `r'` a better replacement for current rule `r`?”

This gives a limited mathematical rationale:

- same-cardinality local comparisons reduce the confounding between
  - support-complexity penalty
  - local likelihood geometry
- in other words, the local step becomes closer to a conditional replacement
  test rather than a shrinkage test

However, this is **not** currently backed by a standalone theorem.

Current theorem status:

- `inh_block_swap_only` itself is still a search-control heuristic
- the mathematically clean object is the final finite-family selector, not this
  particular local-move switch

So for paper-level theory:

- do not present `inh_block_swap_only` as a proved principle
- present it only as part of the engineering baseline that builds the
  high-recall certified active family

### 101. Corrected Tangent-Intensity Rerun: `logical_context` Improved, But The Baseline Itself Drifted Across Reruns

Date:
- 2026-04-07

After fixing the invalid wrapper setting mismatch, the tangent-intensity
selector was rerun on:

- `logical_clean_plus`
- `logical_context`
- `ablation_inhibition_only`

Corrected rerun results:

- `logical_clean_plus`
  - baseline: recall `1.0`, precision `0.8571`
  - tangent selector: recall `1.0`, precision `0.8571`
  - no improvement
- `logical_context`
  - baseline: recall `1.0`, precision `0.875`
  - tangent selector: recall `1.0`, precision `1.0`
  - exact recovery on this rerun
- `ablation_inhibition_only`
  - baseline: recall `0.8333`, precision `1.0`
  - tangent selector: recall `0.8333`, precision `1.0`
  - no improvement

So on the corrected rerun, the tangent-intensity selector only helped on
`logical_context`.

#### 101.1. The More Important New Finding: Baseline Reruns Are Not Stable Enough Yet

The baseline supports seen inside this rerun do **not** match the previously
saved best baseline artifacts:

- previous saved `safe-swap confset` result on `ablation_inhibition_only` was
  exact with 6 inhibition rules
- current rerun baseline only kept 5 inhibition rules and missed
  `C and G and H -> T : inhibition`
- previous saved `logical_context` extra was
  `A and B and D -> T : inhibition`
- current rerun baseline extra became
  `A and E and G -> T : excitation`

So there is a broader reproducibility issue:

- the baseline exact solver is currently not stable enough across reruns

The most likely reason is the active-set anchor / kernel optimization stage:

- no explicit `torch.manual_seed` or deterministic control is set in the
  theorem-friendly active-set path
- support construction uses GPU-based Adam optimization

This means current rerun-to-rerun changes can mix together:

- method improvement
- baseline drift

#### 101.2. Immediate Implication

Before trusting further selector comparisons, the next mandatory step is:

- make the baseline theorem-friendly pipeline reproducible across reruns

That means:

1. explicit seeding
2. deterministic backend configuration where possible
3. fixed rerun verification on the key benchmarks

Only after that should further selector comparisons be treated as decisive.

### 102. Deterministic Execution Would Fix Reproducibility Drift, But Not The Underlying Model-Selection Problem

Date:
- 2026-04-07

Question:
- if we switch the theorem-friendly pipeline to deterministic execution, does
  that remove the problem?

Answer:

- it removes one problem
  - rerun-to-rerun drift
- it does **not** remove the deeper problem
  - whether the selected support is actually the right one

These must be separated.

#### 102.1. What Deterministic Execution Would Fix

The current baseline path uses GPU-based Adam optimization without explicit
seed or deterministic backend control.

So deterministic execution would help with:

- same code + same data + same hardware giving the same active support
- stable comparison between
  - baseline
  - new selector variants
- ruling out “it changed because of rerun noise”

Concretely this means setting all of:

- `random.seed(...)`
- `numpy.random.seed(...)`
- `torch.manual_seed(...)`
- `torch.cuda.manual_seed_all(...)`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- deterministic cuBLAS workspace config
- single-thread or fully controlled BLAS / OpenMP settings where needed

#### 102.2. What Deterministic Execution Would Not Fix

Even with fully deterministic execution:

- the active-set solver is still nonconvex
- the final theorem-friendly search path is still greedy
- `inh_block_swap_only` is still a heuristic search control
- objective mismatch, if present, remains objective mismatch
- higher-order inhibition can still be systematically under-valued

So determinism makes the pipeline **repeatable**, not automatically **correct**.

#### 102.3. Why Determinism Is Still Mandatory

Even though it does not solve the model-selection problem, it is still a
mandatory next step because without it we cannot tell whether a new direction:

- truly helped
- or only looked better because the baseline drifted

So the right sequence is:

1. make the baseline deterministic
2. verify that key reruns are stable
3. only then compare new selectors

#### 102.4. Practical Caveat

Deterministic execution may cost:

- more runtime
- possible fallback away from some fast GPU kernels
- occasional implementation adjustments if a used CUDA op has no deterministic
  path

But this cost is acceptable because current research conclusions are otherwise
not trustworthy enough.

### 103. Removing The Heuristic Layers, The Core Reason Higher-Order Inhibition Is Under-Valued Is A Built-In Sign-Asymmetric Gain Geometry

Date:
- 2026-04-07

Goal:
- explain the higher-order inhibition misses without appealing to the heuristic
  search controls

So for this diagnosis, ignore the following engineering layers:

- `safe_swap_superset`
- `active_interaction_block`
- `inh_block_swap_only`
- `confidence_screen_top_k`
- `scalar_warm_reuse`
- motif / center-block restrictions

Those can change *which* candidates are visited, but they do not explain the
systematic sign bias by themselves.

The root cause already appears in the non-heuristic score formulas.

#### 103.1. Excitation And Inhibition Are Scored On Different Local Scales

From the active-set screening score:

\[
g_{\mathrm{exc}}
= \sum_i \frac{f_r(t_i)}{\eta(t_i)}
- \sum_m w_m f_r(t_m)e^{-I(t_m)},
\]

\[
g_{\mathrm{inh}}
= -\sum_i f_r(t_i)
+ \sum_m w_m f_r(t_m)\lambda_0(t_m),
\]

where:

- `f_r` is the rule feature
- `\eta = \mu + E`
- `\lambda_0 = \eta e^{-I}`

So:

- excitation event evidence is weighted by `1 / eta`
- inhibition grid evidence is weighted by `lambda_0`

These are qualitatively different.

#### 103.2. Low Intensity Helps Excitation But Hurts Inhibition

When the current intensity is already small:

- for excitation:
  - `1 / eta` becomes large
  - event evidence is amplified
- for inhibition:
  - `lambda_0` becomes small
  - grid-side evidence is suppressed

This is the first core asymmetry.

So the same low-intensity region has opposite effects:

- it makes missed excitation easier to justify
- it makes extra inhibition harder to justify

#### 103.3. Higher-Order Inhibition Is Hit Twice

For a higher-order conjunction rule:

- the feature `f_r` is already sparse because all sources must align
- in many realistic cases it becomes active exactly where lower-order
  inhibition has already pushed the intensity down

Then the inhibition gain term

\[
\sum_m w_m f_r(t_m)\lambda_0(t_m)
\]

is doubly reduced:

1. by conjunction sparsity in `f_r`
2. by low current intensity in `\lambda_0`

That means a true higher-order inhibition rule can have only a very small
incremental local gain even when it is genuinely part of the data-generating
support.

This is exactly the observed failure mode:

- `logical_context`
- `kernel_exponential`
- `num_predicates_20`
- `ablation_inhibition_only`

#### 103.4. The Inhibition-Only Scalar Gain Shows The Same Bias

The same structure appears in the exact 1D inhibition addition score:

\[
g_0 = \langle \text{residual}, x_r \rangle - a_r,
\]

where:

- `x_r` is the inhibition grid feature
- `a_r` is the event-side activation sum
- `residual = w \cdot \lambda_0`

So the sufficient evidence for adding inhibition is controlled by the current
intensity-weighted residual mass.

Again, if a true higher-order inhibition rule mainly lives where intensity is
already low, then:

- the residual term is small
- the gain can easily fall below threshold

This is not a heuristic artifact. It is already in the exact scalar
optimization formula.

#### 103.5. Why This Does Not Symmetrically Affect Excitation

The exact excitation scalar gain uses the derivative

\[
g_{\mathrm{exc}}(0)
= \sum_m w_m f_r(t_m)
- \sum_i \frac{f_r(t_i)}{\eta(t_i)}.
\]

So when `eta` is small, excitation receives stronger event-side leverage
through `1 / eta`.

This means the current model geometry is not sign-balanced:

- excitation is locally rewarded in low-intensity regimes
- inhibition is locally penalized in low-intensity regimes

That is the fundamental reason “higher-order inhibition” is more fragile than
the comparable excitation rules.

#### 103.6. Research Consequence

The next non-heuristic selector should therefore not compare raw excitation and
inhibition coefficients directly.

It should compare them only after mapping both signs into a common signed
intensity-effect space, e.g. the canonical tangent-intensity features:

- excitation: `p_r e^{-I_0}`
- inhibition: `-p_r \lambda_0`

So the current best explanation is:

- higher-order inhibition is under-valued mainly because of a built-in
  sign-asymmetric local gain geometry of the nonlinear intensity model
- heuristic search controls can worsen or mask this, but they are not the root
  cause

### 104. Deterministic Side-By-Side Check Should Record Baseline And Explicit Family Selector In The Same Run

Date:
- 2026-04-07

To avoid another comparison mismatch, the next verification run should be:

- deterministic
- same fixed seed
- same benchmark-local dataset
- baseline and explicit family selector recorded in the same output artifact

So the chosen protocol is:

1. run the deterministic `safe-swap confset` baseline inside the explicit
   family-selector wrapper
2. store both
   - baseline result
   - explicit family-selector result
   in the same JSON per benchmark
3. restrict to the three diagnostic benchmarks:
   - `logical_clean_plus`
   - `logical_context`
   - `ablation_inhibition_only`

This ensures the comparison is:

- same seed
- same runtime environment
- same wrapper
- same benchmark instance

So any remaining difference can be attributed to the selector stage rather than
to rerun drift or configuration mismatch.

### 105. Deterministic Side-By-Side Results: The Tangent-Intensity Selector Only Helped `logical_context`

Date:
- 2026-04-07

The deterministic `seed=0` side-by-side run completed on:

- `logical_clean_plus`
- `logical_context`
- `ablation_inhibition_only`

Results:

- `logical_clean_plus`
  - baseline: recall `1.0`, precision `0.8571`
  - tangent selector: recall `1.0`, precision `0.8571`
  - unchanged
- `logical_context`
  - baseline: recall `1.0`, precision `0.875`
  - tangent selector: recall `1.0`, precision `1.0`
  - improved to exact recovery
- `ablation_inhibition_only`
  - baseline: recall `0.8333`, precision `1.0`
  - tangent selector: recall `0.8333`, precision `1.0`
  - unchanged

So under deterministic comparison:

- the tangent-intensity selector does help the mixed logical hard case
  `logical_context`
- but it does **not** fix the persistent pure higher-order inhibition miss

#### 105.1. What This Means

The canonical tangent-intensity geometry is a real improvement in at least one
mixed-sign hard case, so the direction itself remains meaningful.

But the pure inhibition miss remains, which means:

- sign-symmetric tangent comparison alone is not sufficient
- the hardest remaining failure is now the pure inhibition family itself

So the next analysis should focus on:

- why `CGH -> inhibition` is already absent from the deterministic baseline
  active family on `ablation_inhibition_only`
- whether the miss is caused by
  - active-family construction
  - complexity penalty
  - or the current inhibition feature / residual geometry even before final
    family selection

### 106. Why The Miss Is So Specific To Higher-Order Inhibition

Date:
- 2026-04-07

The repeated pattern is now too specific to be accidental:

- higher-order excitation can survive
- lower-order inhibition can survive
- but higher-order inhibition is repeatedly the one that gets dropped

The reason is a combination of three effects that all point in the same
direction.

#### 106.1. Conjunction Sparsity

A higher-order rule feature is already sparse because all of its sources must
co-activate.

So before sign even matters:

- singleton / pair rules fire often
- triplet inhibition rules fire much less often

This means higher-order inhibition starts with a weaker raw feature footprint.

#### 106.2. Inhibition Is Scored On The Current Intensity Scale

For inhibition, the core gain terms are weighted by the current intensity
through `lambda_0`.

So a candidate inhibition rule gets rewarded mainly when it can suppress
regions where the model currently still has substantial intensity mass.

But a true higher-order inhibition rule often becomes active exactly in regions
where:

- lower-order inhibition is already on
- the current intensity is already low

Then its incremental score is small, because there is not much remaining
intensity left to suppress.

So higher-order inhibition is not just sparse. It is sparse **in the worst
possible places for the current score geometry**.

#### 106.3. Existing Lower-Order Inhibition Cannibalizes Its Marginal Value

This is the most important point.

The selector does not ask:

- “is this higher-order inhibition rule true in the data-generating model?”

It asks:

- “given the lower-order rules already in the current model, how much extra
  objective improvement do I still get by adding this rule?”

For higher-order inhibition, those are very different questions.

If lower-order inhibition already explains most of the suppression pattern,
then the higher-order rule looks like only a small residual correction.

That is why the miss is so systematic:

- the current optimization keeps the broad lower-order suppression footprint
- then the true higher-order inhibition rule looks like a tiny incremental
  bonus
- the bonus often fails to beat the complexity cost or the screening threshold

#### 106.4. Why The Same Logic Does Not Kill Higher-Order Excitation As Often

Excitation is also conjunction-sparse, but it does not suffer the same degree
of marginal-value collapse.

Reason:

- when intensity is low, missed excitation gets stronger event leverage through
  the `1 / eta` type weighting
- when intensity is low, missed inhibition gets weaker leverage because the
  available suppressible mass is already small

So the current local geometry is not merely “sparse rules are hard.”

It is specifically:

- sparse excitation: hard but still locally amplified in low-intensity regions
- sparse inhibition: hard and locally damped in low-intensity regions

#### 106.5. Practical Interpretation

So the full causal chain is:

1. higher-order conjunction makes the inhibition feature sparse
2. lower-order inhibition already removes much of the relevant intensity mass
3. the current score evaluates only the remaining incremental suppression
4. therefore the true higher-order inhibition rule looks weak

That is why the failure is so concentrated on higher-order inhibition rather
than appearing symmetrically across all rule types.

### 107. The Current Pipeline Is Also Not Order-Balanced: Higher-Order Rules Start With A Multiplicative Feature-Scale Disadvantage

### 108. Diagnostic For Whether Higher-Order Inhibition Is Losing Because Of Order-Unbalanced Ranking

To separate search failure from ranking failure, run a deterministic diagnostic
on three inhibition-sensitive benchmarks:

- `logical_context`
- `kernel_exponential`
- `ablation_inhibition_only`

The goal is not yet to change the selector, but to inspect where the missing
true inhibition rule sits among all inactive inhibition candidates under two
different local comparison scales:

1. current exact scalar inhibition gain
2. Fisher-normalized score

For every inactive inhibition candidate, record:

- `g0 = residual · x - a`
- `h0 = residual · x^2`
- `fisher_z = g0 / sqrt(h0)`
- quadratic approximation gain
- exact 1D scalar inhibition gain

Then, for each missing true inhibition rule, record:

- global rank by exact gain
- global rank by Fisher-normalized score
- order-restricted rank by exact gain
- order-restricted rank by Fisher-normalized score

Interpretation target:

- if the missing true high-order inhibition rule has poor rank under both
  scales, the issue is likely already in candidate construction / residual
  geometry
- if it ranks poorly under exact gain but much better under Fisher-normalized
  score, then current local ranking is sign-balanced but not order-balanced
- if it ranks well under both, then the miss occurs later in neighborhood
  search / active-family construction rather than in first-order scoring

### 109. Honest Assessment Of What Is Currently Publishable And Provable

Current status should be separated into three layers.

Layer 1: the model class itself

- the logical TPP model with rule-dependent kernels and signed
  excitation/inhibition is interesting
- the sign asymmetry is mathematically real because the current intensity takes
  the form
  `lambda(t) = (mu + E(t)) exp(-I(t))`
- under this model, higher-order inhibition can be structurally under-valued
  because its marginal gain is measured on already-suppressed intensity mass

Layer 2: what is currently theorem-friendly

- finite-family selection over a certified active-support family is provable in
  principle
- the clean theorem target is:
  1. certified family construction
  2. finite-family oracle inequality / confidence guarantee
  3. exact recovery corollary under a family-level margin condition
- this part is mathematically credible

Layer 3: what is still heuristic / not yet theorem-ready

- active-support construction still depends on local search controls such as
  inhibition neighborhood design
- deterministic execution fixes reproducibility drift but does not remove that
  heuristic dependence
- current inner optimization is still numerical rather than certified
- current best theorem-friendly empirical result is still the deterministic
  `safe-swap confset` line, not a fully closed exact-recovery method

Publication assessment:

- as-is, this is probably not yet strong enough for a NeurIPS main-track paper
- the strongest current contribution is not “we solved exact recovery
  end-to-end”, but rather:
  - a new interpretable rule-dependent logical TPP model
  - a diagnosis of sign/order asymmetry causing systematic high-order
    inhibition under-selection
  - a certified finite-family model-selection view that could support a clean
    oracle theorem

What would likely make the contribution NeurIPS-level:

1. a theorem whose main object is a certified finite model family rather than a
   heuristic search trajectory
2. an end-to-end selector whose proof assumptions are explicit and modest
3. a convincing empirical fix for the inhibition asymmetry on the full suite
4. preferably at least one non-toy or semi-real benchmark beyond the current
   synthetic suite

### 110. Safer Literature Positioning Relative To TLPP / CLNN / Recent Neuro-Symbolic TPP Work

The strongest broad claim is **not**:

- “no prior TPP paper models excitation and inhibition with logic rules”

That claim is too strong because at least CLNN / weighted clock logic point
process explicitly states that some events can promote or inhibit others.

Safer positioning is:

- prior logical or neuro-symbolic TPP papers do **not simultaneously** provide:
  1. explicit signed rule support with separate excitation and inhibition rule
     identities
  2. strict conjunctive (`AND`) rule semantics as the main recovered object
  3. rule-dependent kernel recovery / selection at the support level
  4. a theorem-friendly finite-family model-selection formulation

Representative prior-work split:

- TLPP:
  - uses first-order temporal logic rules
  - strong structured modeling contribution
  - not the same as explicit strict higher-order conjunction recovery

- CLNN / weighted clock logic point process:
  - explicitly allows promote / inhibit interpretations
  - but uses smooth activations and continuous relaxation over weighted clock
    logic formulas
  - so it is better described as logic-inspired / neuro-symbolic rather than
    strict discrete support recovery of conjunctive rules

- recent neuro-symbolic / RL TPP rule-learning papers:
  - learn rule embeddings or neural-search-generated rule content
  - emphasize efficient induction
  - still differ from exact signed conjunctive support selection with
    rule-dependent kernels

Therefore the safest main-contribution statement is:

- this work targets the gap between interpretable logical TPP modeling and
  explicit signed strict-conjunction rule recovery, especially for higher-order
  inhibition

### 111. Deterministic Order-Balance Diagnostic: The Pure Inhibition Failure Is Not A Ranking Failure

Deterministic diagnostic run:

- `logical_context`
- `kernel_exponential`
- `ablation_inhibition_only`

Outputs stored in:

- `tmp_ord_bal_context_20260407.json`
- `tmp_ord_bal_kernel_exp_20260407.json`
- `tmp_ord_bal_inhibition_20260407.json`

Main finding:

- in `ablation_inhibition_only`, the missing true rule
  `C and G and H -> T : inhibition`
  is:
  - global rank `1` by exact scalar inhibition gain
  - global rank `1` by Fisher-normalized score
  - order-3 rank `1` by exact gain
  - order-3 rank `1` by Fisher-normalized score

Therefore this benchmark's miss is **not** explained by:

- exact local inhibition gain ranking
- Fisher-normalized order-balanced re-ranking

Instead, the failure must happen later, in one of:

1. neighborhood proposal / active-family construction
2. support-update path dependence
3. local search acceptance logic

This sharply narrows the next theorem-relevant problem:

- the central remaining issue is not merely score asymmetry
- it is that the current certified active family may fail to contain the true
  higher-order inhibition support even when the missing rule is locally the
  strongest inactive inhibition candidate

### 112. Next Diagnostic Must Decompose Final-State Failure Into Filter Failure Vs. Refit-Acceptance Failure

Given the result above, the next deterministic diagnostic should focus on the
remaining hard benchmark `ablation_inhibition_only`.

For the missing rule `C and G and H -> T : inhibition`, inspect the final
baseline inhibition state and separate two possibilities:

1. filter failure:
   - the rule is excluded by `safe_inh_swap_superset` / candidate gating before
     exact support evaluation
2. refit-acceptance failure:
   - the rule survives candidate screening, but the add-support refit does not
     improve the inhibition-stage validation BIC under the current acceptance
     logic

Necessary quantities to log:

- whether the rule is in `positive_inh_adds`
- whether the rule is in the safe superset
- whether it survives the final screened add-candidate list
- its rank in the unrestricted list and in the safe-superset-restricted list
- direct add-only inhibition-support BIC change
- coupled add + alternating refit BIC change after re-optimizing the paired
  excitation block

This diagnostic determines whether the remaining miss is caused primarily by:

- heuristic family construction
- or a mismatch between scalar local gain and support-level acceptance

### 113. Deterministic Trace Result: `CGH inhibition` Survives Every Filter But Fails At Support-Level Refit

Deterministic trace file:

- `tmp_inh_family_trace_20260407.json`

Benchmark:

- `ablation_inhibition_only`

Key result for the missing true rule
`C and G and H -> T : inhibition`:

- `in_derivative_ok = True`
- `in_safe_superset = True`
- `in_unrestricted_add_candidates = True`
- `in_restricted_add_candidates = True`
- unrestricted add rank = `1`
- restricted add rank = `1`

So the rule is **not** lost by:

- derivative screening
- safe-swap superset gating
- final add-candidate screening

However, after actual support evaluation:

- add-only inhibition support has worse BIC:
  - `delta_bic = +1.7269`
- coupled alternating refit also remains worse:
  - inhibition-stage `delta_bic = +0.0404`
  - excitation-stage `delta_bic = +0.0628`

Therefore the miss is not a candidate-generation failure. It is a
support-level acceptance failure:

- scalar local inhibition gain says the rule is strongly favorable
- but once the rule is added and the support is re-optimized, the current
  validation BIC criterion still prefers the smaller support

This means the remaining theorem-relevant gap is now sharper:

- the bottleneck is the mismatch between local scalar gain and the support-level
  structural criterion used for acceptance
- not the absence of the true rule from the deterministic active family

### 114. Mathematical Form Of The Mismatch And The Right Non-Heuristic Repair

For an inactive inhibition rule `r`, the current scalar screening score solves a
one-dimensional **frozen-support** problem:

- fix the current support `S` and its fitted parameters
- vary only the new inhibition coefficient `beta_r >= 0`

The scalar objective is:

- `Delta_frozen_tr(S, r)`
  = train-NLL change from adding `r` while keeping the old support frozen,
    except for the 1D optimization over `beta_r`

In code this is the exact 1D scalar gain in
`tmp_inhibition_profile_block_validate.py::scalar_gain`.

The search score is therefore:

- `G(S, r) = - Delta_frozen_tr(S, r) - lambda_n`

where `lambda_n = 0.5 log n_eff`.

By contrast, the actual support-acceptance decision compares the
**profiled validation BIC**:

- `BIC(T) = 2 * NLL_val(theta_hat_T) + |T| log n_eff`

where `theta_hat_T` is the support-level refit on training data for support
`T`.

So the search score and the acceptance score are different objects:

- search uses frozen local training gain
- acceptance uses profiled validation risk

The difference can be written as:

- `Delta_BIC_val(S, S U {r})`
  = `2 * Delta_frozen_tr(S, r) + log n_eff`
  + `2 * [Delta_profile_tr(S, r) - Delta_frozen_tr(S, r)]`
  + `2 * [Delta_val_profile(S, r) - Delta_tr_profile(S, r)]`

Interpretation:

1. frozen-vs-profile term
   - support refit can redistribute mass across all active coefficients
2. train-vs-validation term
   - high-order sparse inhibition can have large local train gain but weak or
     unstable validation gain

In the traced failure `ablation_inhibition_only`, the missing true rule
`C and G and H -> T : inhibition` has strongly positive scalar gain, but after
support-level refit the validation BIC is still slightly worse. So the miss is
not from candidate generation. It is a support-acceptance failure caused by the
fact that local scalar gain is only a proposal score, not a correct support
selection criterion.

Non-heuristic repair:

- stop using strict greedy improvement as the family-construction rule
- instead build a **confidence-admissible support family**

For each current support `S` and neighbor `T`, estimate the validation excess
loss difference:

- `d_hat(S, T) = mean(loss_T - loss_S)`

with an empirical-Bernstein radius `rad_n(S, T)`.

Then classify:

- if `d_hat + rad_n < 0`, `T` is certified better
- if `d_hat - rad_n > 0`, `T` is certified worse
- otherwise `S` and `T` are statistically indistinguishable, so keep **both**
  in the active family

This repairs the exact failure mode above:

- `CGH inhibition` would not be discarded just because its one-step profiled
  validation BIC is slightly worse
- if that difference is within uncertainty, the enlarged family keeps the rule
  alive for later exact family selection

This is theorem-friendly because:

1. the family is finite and explicitly defined
2. each neighbor comparison is evaluated exactly
3. the expansion rule is based on confidence intervals, not ad-hoc thresholds
4. final selection can then be done over the resulting certified family

### 115. Prototype Experiment: Confidence-Admissible Inhibition Add-Family Around The Deterministic Baseline

Prototype selector for inhibition-sensitive benchmarks:

- keep the deterministic validated baseline as the current support
- enumerate all positive single-add inhibition neighbors
- for each add-neighbor `T`, compare validation excess loss against the current
  support `S`
- if `T` is certified worse, discard it
- otherwise keep it as confidence-admissible

Then construct the finite family:

- all subsets of the retained inhibition add-neighbors on top of the baseline
  support

Finally:

- refit each family member exactly
- choose the smallest pair-validation-BIC support among that explicit family

This is still only a prototype, because family growth is currently restricted
to inhibition add-neighbors around the deterministic baseline support, not a
full recursive family expansion over the entire search trajectory. But it is a
direct test of the mathematically motivated repair for the observed
`CGH inhibition` failure mode.

### 116. Prototype Result: Confidence-Admissible Add-Family Alone Does Not Fix The Remaining Miss

Deterministic prototype results:

- `logical_context`
- `kernel_exponential`
- `ablation_inhibition_only`

Files:

- `tmp_conf_adm_context_20260407.json`
- `tmp_conf_adm_kernel_exp_20260407.json`
- `tmp_conf_adm_inhibition_20260407.json`

Observed outcomes:

- `logical_context`:
  - candidate count `2`
  - confidence-kept count `2`
  - family size `4`
  - final choice stays at the baseline support
  - no precision improvement

- `kernel_exponential`:
  - candidate count `0`
  - no change

- `ablation_inhibition_only`:
  - candidate count `2`
  - confidence-kept count `2`
  - family size `4`
  - the missing true rule `C and G and H -> T : inhibition` remains in the
    family, but the best family member is still the baseline support

Conclusion:

- confidence-admissible family expansion successfully prevents premature
  rejection of near-tie add-neighbors
- but the current pair-validation-BIC selector still prefers the smaller
  baseline support
- therefore the remaining problem is no longer family construction alone
- it is the final support-level structural criterion itself

### 117. Mathematically Correct Repair: Replace Single-Rule Add Acceptance By A One-Sided Nested Profile Test, Not By BIC

The latest diagnostics show:

- the missing high-order inhibition rule survives proposal filtering
- it survives the confidence-admissible family construction
- but it still loses under the final pair-validation-BIC selector

This means the problem is no longer family construction. It is that BIC is the
wrong mathematical object for deciding a **single nested rule addition**.

For a current support `S` and an inactive inhibition rule `r`, the scientific
question is:

- `H0: beta_r = 0`
- `H1: beta_r > 0`

inside the nested model pair:

- support `S`
- support `S U {r}`

Under the current frozen-kernel theorem-friendly formulation, this nested pair
is well-defined because adding `r` only introduces one new nonnegative
coefficient.

Therefore the correct acceptance device is a one-sided nested test:

- profile likelihood-ratio test
- or equivalent score / Wald test under the nonnegative boundary

not BIC.

Concretely, define

- `Lambda_hat(S, r) = 2 * [ell(theta_hat_{S U {r}}) - ell(theta_hat_S)]`

with the likelihood evaluated on an independent validation split or by
cross-fitting.

Because `beta_r >= 0` and the null is on the boundary, the asymptotic null law
is the standard one-sided boundary mixture:

- `0.5 * chi^2_0 + 0.5 * chi^2_1`

So a mathematically principled add-rule decision is:

- accept `r` if the one-sided p-value is below the familywise threshold
- reject `r` otherwise

Why this matches the observed failure:

- BIC asks whether the larger support wins after a generic complexity penalty
- the nested add-rule question asks whether the new coefficient is detectably
  positive conditional on the current support

These are different questions.

High-order inhibition can easily fail BIC but pass a one-sided nested test:

- effect is real but concentrated and sparse
- validation BIC favors the smaller support when the net global improvement is
  small
- yet the coefficient can still be significantly positive conditional on the
  current support

The theorem-friendly path is therefore:

1. deterministic baseline support search
2. for each inactive candidate rule, perform a one-sided nested profile test
3. enlarge the active family by all non-rejected additions
4. do exact familywise selection over that certified family

This is cleaner than BIC-at-every-step because:

- each add-rule decision corresponds to a precise hypothesis test
- familywise error can be controlled explicitly
- exact recovery can be stated under a beta-min / noncentrality margin
- the computation is still local and much cheaper than global full-family
  search

### 118. Prototype Experiment: One-Sided Nested Inhibition Add Test With Holm Control

Prototype:

- keep the deterministic baseline support
- enumerate screened positive inhibition add-candidates
- for each add-rule `r`, fit the nested pair:
  - current support `S`
  - enlarged support `S U {r}`
- compute the one-sided boundary likelihood-ratio statistic on training data
- convert it to a p-value using the `0.5 * chi^2_0 + 0.5 * chi^2_1` null law
- apply Holm familywise correction
- accept all significant inhibition add-rules
- refit the final enlarged support once

This prototype directly tests whether the remaining high-order inhibition miss
is caused by using BIC for nested add-rule acceptance rather than an explicit
one-sided boundary test.

### 119. Why The One-Sided Add-Test Prototype Was A Return To An Already-Failing Path

The latest one-sided nested inhibition-add test gave:

- `ablation_inhibition_only`:
  - baseline `recall=0.8333`, `precision=1.0`
  - add-test result `recall=1.0`, `precision=0.8571`
  - true missing `CGH inhibition` was recovered
  - but false positive `AEF inhibition` was also accepted

This means the prototype reproduced the same qualitative failure mode as the
earlier add-only repair directions:

- raise recall by accepting additional inhibition rules
- lose precision because there is no equally strong non-heuristic deletion /
  exclusion principle at the same stage

So, conceptually, this was indeed a return to an already explored path:

- monotone one-sided add acceptance
- good for rescuing missing true rules
- insufficient for exact support recovery because it does not simultaneously
  control false additions

Important lesson:

- the right final object cannot be “accept all significant add-rules”
- it must be a symmetric signed-support selection principle that can both
  include and exclude rules inside one mathematically unified family decision

Therefore this one-sided add-test prototype should be treated as a diagnostic,
not as a viable final method.

### 120. Next Direction: Symmetric Local Closed Testing Around The Deterministic Baseline

The next non-heuristic local repair should be symmetric.

Object:

- start from the deterministic validated baseline support `S0`
- construct a local augmented support by testing **both** excitation and
  inhibition single-add neighbors against `S0`
- then, inside the augmented support, test every active rule for necessity by a
  nested one-sided boundary test against its deletion

So the local pipeline is:

1. deterministic baseline support `S0`
2. add-phase:
   - for every screened positive inactive rule `r`, test `S0` vs `S0 U {r}`
3. augmented support `U`
4. delete-phase:
   - for every active rule `r in U`, test `U` vs `U \\ {r}`
5. final refit on the kept signed support

This is different from the failed add-only path because:

- additions and exclusions are handled by the same mathematical device
- the final object is not monotone in support size
- false-positive add-rules can be removed in the delete phase

The current full-suite prototype should therefore test whether this symmetric
local closed-testing repair can simultaneously:

- recover missing higher-order inhibition rules
- remove spurious extras
- avoid the recall/precision imbalance of earlier one-sided add repairs

### 121. Full-Suite Result: Symmetric Local Closed Testing Still Over-Accepts And Is Not Viable

Full deterministic 11-benchmark prototype:

- `tmp_sym_closed_gpu0_20260407.json`
- `tmp_sym_closed_gpu1_20260407.json`
- `tmp_sym_closed_gpu2_20260407.json`
- `tmp_sym_closed_gpu3_20260407.json`

Main result:

- recall stayed at `1.0` on all 11 benchmarks
- precision collapsed on 9/11 benchmarks
- exact recovery only on:
  - `kernel_exponential`

Representative failures:

- `logical_clean_plus`:
  - baseline precision `0.8571`
  - prototype precision `0.5`
- `logical_shared`:
  - baseline precision `0.875`
  - prototype precision `0.6364`
- `logical_context`:
  - baseline precision `0.875`
  - prototype precision `0.7`
- `ablation_inhibition_only`:
  - baseline recall `0.8333`, precision `1.0`
  - prototype recall `1.0`, precision `0.8571`

Interpretation:

- the symmetric local closed-testing idea avoided the monotone-add-only
  pathology in form
- but in practice the boundary LRT add phase massively over-accepted local
  candidates
- the subsequent delete phase could not remove enough of them

So this direction still suffers from the same core issue:

- local nested tests are too liberal as a support-construction mechanism
- even when made symmetric, they do not define a stable enough final family

Conclusion:

- local closed testing should not be the main paper method
- the remaining promising route is to return to the best deterministic
  theorem-friendly baseline (`safe-swap confset`) and redesign only the final
  support-level selection object, not the whole local family-construction rule

### 122. Most Promising Next Direction: Nuisance-Projected Efficient-Score Family On The Boolean Interaction Lattice

The repeated failures of:

- raw scalar inhibition gain
- pair-validation-BIC
- one-sided nested add-LRT
- symmetric local closed testing with raw nested LRT

all point to the same structural issue:

- candidate rules are being judged by statistics that do not sufficiently
  partial out the nuisance effect of the current active support, especially for
  nested conjunctions on the Boolean subset lattice

This suggests the next object should be the **efficient score** for a candidate
rule after projecting out the current support tangent space.

For current support `S` and candidate rule `r`, let:

- `U_r` = score for the new coefficient at the constrained null
- `I_rr` = candidate self-information
- `I_rS`, `I_Sr`, `I_SS` = cross-information / nuisance information blocks

Then define the nuisance-projected efficient score:

- `U_eff(r | S) = U_r - I_rS I_SS^{-1} U_S`

and the efficient information:

- `I_eff(r | S) = I_rr - I_rS I_SS^{-1} I_Sr`

The one-sided boundary statistic is then:

- `T_eff(r | S) = max(U_eff(r | S), 0)^2 / I_eff(r | S)`

Under the null `beta_r = 0`, this has the usual one-sided chi-bar-square law
(`0.5 * chi^2_0 + 0.5 * chi^2_1`) in the regular boundary case.

Why this is the right next direction:

1. it directly addresses the main observed pathology
   - false add-rules such as `AEF inhibition` may have large raw gain or raw
     nested-LRT signal because they are correlated with the current support
   - after nuisance projection, only the **incremental information not already
     explained by the active support** remains

2. it is mathematically aligned with strict conjunction recovery
   - on the subset lattice, this is the local analogue of moving from raw
     conjunction features to pure interaction increments
   - i.e. a score-level Möbius / orthogonalized interaction test

3. it stays theorem-friendly
   - finite local family
   - explicit test statistic
   - explicit null law
   - familywise correction by Holm / closed testing

4. it is computationally feasible
   - the difficult quantity is the score/Fisher block for a small local
     candidate family around the deterministic baseline support
   - no global combinatorial search is required

Most plausible final selector based on this idea:

1. run deterministic `safe-swap confset` baseline
2. build a small local candidate family around the baseline support
3. use nuisance-projected efficient-score tests for:
   - inactive candidate additions
   - active rule deletions
4. form a local certified interval family:
   - lower core = active rules whose deletion is rejected
   - upper envelope = lower core plus candidates whose addition remains
     admissible after nuisance projection
5. run exact familywise selection only inside that interval family

This is different from the failed local closed-testing prototype because the
test statistic itself now changes:

- from raw nested add/delete LRT
- to nuisance-adjusted efficient-score / Schur-complement tests

So the next experiment should not be another generic local test. It should be
the first true **efficient-score closed family** experiment.

### 123. Prototype Experiment: Efficient-Score Interval Family Around The Deterministic Baseline

First implementation choice:

- benchmark set:
  - `logical_clean_plus`
  - `logical_context`
  - `kernel_gaussian`

Procedure:

1. run deterministic `safe-swap confset` baseline
2. compute active-rule necessity statistics by sign-specific Schur-complement
   / nuisance-projected information
3. compute inactive add-rule efficient-score statistics by the same
   nuisance-projected logic
4. define:
   - lower core = active rules that remain significant
   - upper envelope additions = inactive rules that pass the projected add test
5. enumerate the finite interval family between lower core and
   `core ∪ optional-active ∪ accepted-adds`
6. perform exact support-fixed refits over that family and choose the best
   pair-validation-BIC support

This is the first concrete test of whether nuisance-projected local statistics
can reduce the false-add explosion of raw local tests while still recovering
the missing true high-order inhibition rules.

Date:
- 2026-04-07

Question:
- is the current theorem-friendly pipeline too unfavorable to higher-order
  rules even before the sign-specific inhibition issue?

Answer:
- yes

The current rule feature construction already gives higher-order rules a strong
scale disadvantage.

#### 107.1. The Rule Feature Is A Product Of Bounded Source Activities

The rule feature used by the active-set learner is

\[
f_U(t)=\prod_{s\in U} z_s(t),
\qquad 0 \le z_s(t) \le 1,
\]

because the implementation explicitly multiplies bounded per-source activities
across all sources in the subset.

So if the average per-source activity level is below `1`, then the typical
feature scale decays multiplicatively with order:

\[
\mathbb{E}[f_U] \approx \prod_{s\in U}\mathbb{E}[z_s].
\]

That means:

- singleton: scale roughly `m`
- pair: scale roughly `m^2`
- triplet: scale roughly `m^3`

for some average activity level `m < 1`.

So a triplet rule can easily start an order of magnitude smaller than a pair
rule even before any sign-specific scoring enters.

#### 107.2. There Is No Compensating Order Normalization In The Current Screening Score

The active-set screening scores use the raw feature vectors:

- `feat_event`
- `feat_grid`

without any order-dependent normalization.

So higher-order rules enter screening with:

- smaller raw event mass
- smaller raw grid mass
- but essentially the same per-rule complexity cost

under the current theorem-friendly baseline search.

So the pipeline does not explicitly say “triplets cost more,” but it still
creates an **implicit order penalty** because the gain shrinks rapidly with
order while the rule-level penalty stays roughly constant.

#### 107.3. Inhibition Makes This Worse, Not Better

For higher-order inhibition, this order disadvantage is compounded by the
sign-asymmetric intensity weighting discussed above.

So the full effect is:

1. feature magnitude shrinks with conjunction order,
2. lower-order inhibition already suppresses the relevant intensity mass,
3. inhibition gain is evaluated only on the remaining suppressible mass.

This is why the failure is especially concentrated on higher-order inhibition.

#### 107.4. Updated Interpretation

The current theorem-friendly pipeline is not:

- sign-balanced
- or order-balanced

It is therefore structurally biased against higher-order inhibition through two
independent mechanisms:

- multiplicative feature shrinkage with order
- low-intensity suppression of inhibition gains

So the next principled method should eventually correct both:

1. sign balance:
   - compare excitation and inhibition in a common tangent-intensity unit
2. order balance:
   - replace raw product-feature gains with an order-normalized or
     Fisher-normalized comparison that measures distinguishability rather than
     raw feature mass

### 84. First Empirical Check Of Quantitative Residual Information Core Certificates

The next non-heuristic direction keeps the `safe-swap confset` search exactly as
the baseline and changes only the final precision-repair stage.

The key change is:

- do **not** use residual rank
- instead use **quantitative residual information**
- define each active rule by its full tangent block:
  - coefficient direction
  - source-kernel height tangent directions under area normalization
- residualize that block against the span of all competing rule tangent blocks
- certify protection only when the residual information has a positive lower
  confidence bound

Concretely, for rule `r`, let `T_r` be its constrained tangent block and let
`T_{-r}` be the concatenated tangent blocks of the remaining active rules. The
protection diagnostic is based on the residual information left after
projecting `T_r` onto `span(T_{-r})`, evaluated on a holdout split with a lower
confidence bound on the residual energy.

This is the right conceptual correction to the failed residual-rank attempt:

- residual rank was too binary and protected even false extras
- logical TPP needs a **magnitude** notion of indispensability, not a bare
  independence test

Operational test setup:

- benchmark set:
  - `logical_clean_plus`
  - `logical_context`
  - `kernel_gaussian`
- execution:
  - separate `tmux` sessions
  - one GPU per benchmark
  - exact ambiguity-family selection only after the residual-information core
    split

This run is only a first diagnostic. The criterion for keeping this direction
alive is strict:

- it must improve precision on the three hard cases
- it must preserve recall `1.0`
- and it must do so without any dataset-specific exception or heuristic cutoff

### 85. Quantitative Residual Information Improved One Hard Case, Helped Another, And Still Failed The Theorem-Friendly Standard

The first `quantitative residual information` run on the three precision-hard
benchmarks produced:

- `logical_clean_plus`:
  - exact recovery
  - recall `1.0`, precision `1.0`
- `kernel_gaussian`:
  - partial improvement
  - recall `1.0`, precision improved from `0.75` to `0.8571`
  - one extra remained
- `logical_context`:
  - regression
  - recall and precision both fell to `0.8571`
  - one true inhibition rule was deleted while the old extra remained

So this direction is **not** better than the existing `safe-swap confset`
baseline.

#### 85.1. Why It Failed

The failure is structural, not just a tuning issue.

The certificate was built from the current active support `\hat S` by asking,
for each rule `r`:

- does the full tangent block of `r`
- after projection onto the span of the other active-rule tangent blocks
- still retain a positive lower-confidence residual energy?

This quantity answers a local question:

- is `r` locally indispensable around the current parameterization of `\hat S`?

But support recovery needs a different question:

- is `r` globally indispensable after support-level reoptimization over rival
  supports?

Those are not the same.

False extras can be **locally indispensable** for two reasons.

First:
- the current overfit support may allocate part of the signal to the extra rule
  in a way that is not reproduced by a first-order projection onto the other
  rules at the same parameter point

Second:
- the real replacement of a false extra may require a joint nonlinear
  re-equilibration of several surviving true rules, not a single tangent-space
  projection around the current fit

This is exactly what happened in `logical_context`.

- the extra inhibition rule received a positive residual-information lower bound
  and became protected
- the true 3-way inhibition rule did not receive that protection
- once the extra was locked into the core, the exact ambiguity-family search no
  longer contained the correct support

So the selector failed before the exact family stage even started.

#### 85.2. Why Higher-Order Inhibition Is Especially Fragile Under This Criterion

Higher-order inhibition rules are sparse and highly interaction-dependent.

That creates two simultaneous effects.

- their tangent energy is concentrated on fewer validation locations
- after residualization against simpler inhibitory rules, the remaining signal
  can be small even when the rule is truly necessary at the support level

So split-sample lower confidence bounds tend to be conservative exactly where
the true high-order inhibition rules live.

This explains the asymmetry we repeatedly observed:

- false simple or medium-order extras can still look locally indispensable
- true high-order inhibition rules can look locally replaceable

#### 85.3. Negative Conclusion

Rule-level protected-core certificates based on local differential
indispensability are too brittle for final support selection in logical TPP,
even when they use full tangent blocks and quantitative residual information.

The core issue is:

- support recovery is a **profile-family comparison** problem
- not a local tangent-independence problem

### 86. Next Direction: Exact Profile Family Selection Over The Entire Baseline Active Support

The next direction should stop certifying individual rules altogether.

Instead:

1. keep the current `safe-swap confset` search exactly as the baseline
2. let `\hat S` be the resulting active support
3. define the final candidate family as **all subsets of `\hat S`**
4. for each candidate support `S \subseteq \hat S`, compute a support-specific
   local profile refit
5. use split-sample familywise confidence screening over that finite family
6. choose the smallest admissible support, with the profile objective only as a
   tie-break

This is mathematically cleaner than every protected-core attempt so far.

There is:

- no heuristic `k`
- no hand-designed block rule
- no protected-core preclassification that can accidentally remove the true
  support from the final family

#### 86.1. Why This Directly Targets The Observed Failures

This family automatically contains all supports obtainable by deleting any
combination of active rules.

So:

- `logical_context` is no longer blocked by a misprotected extra
- `logical_clean_plus` and `kernel_gaussian` can still benefit from
  support-specific local kernel reoptimization
- the final decision is made at the support level, where the real problem lives

#### 86.2. Why This Is Still Theorem-Friendly

Let `\mathcal F(\hat S) = \{ S : S \subseteq \hat S \}` be the finite family of
active-support subsets.

For each `S \in \mathcal F(\hat S)`, define the profiled empirical risk
`\hat R(S)` by exact or numerically exact local refit over the parameter space
associated with `S`.

Then the final selector is:

- confidence family:
  - `\mathcal C = \{ S \in \mathcal F(\hat S) : \hat R_B(S) - \hat R_B(\hat S) \le \mathrm{rad}_\delta(S) \}`
- selected support:
  - the inclusion-minimal member of `\mathcal C`
  - with profiled risk as the tie-break

This immediately supports the right theorem shape.

- familywise finite-sample oracle guarantee over `\mathcal F(\hat S)`
- exact recovery as a corollary under:
  - coverage `S^\star \subseteq \hat S`
  - uniqueness of the minimal admissible true support
  - a validation-risk margin separating `S^\star` from every other admissible
    subset

#### 86.3. Why It Can Also Be Fast Enough

The mistake in the previous exact-closure attempts was not the family size by
itself; it was the expensive candidate evaluation.

In the current benchmark line, active supports are small enough that the whole
subset family is realistic if profile evaluation is engineered properly.

The right speedups are exact-preserving:

- enumerate supports in Gray-code or parent-child order
- warm-start each candidate from its nearest already-solved parent support
- do closed-form coefficient refits before kernel updates
- use a small number of damped Newton or quasi-Newton kernel steps instead of
  long Adam runs
- prune candidates early when a lower bound already fails the confidence screen
- batch multiple candidates on GPU

So the next research direction is:

- **not** another rule-level core certificate
- but an exact, support-level, profiled confidence-family selector over the
  full baseline active support

### 87. Process Rule Going Forward

From this point on, every research iteration should end with both:

- an appended note in this research file
- cleanup of dead-end temporary code and dead-end result artifacts

Only the currently useful baseline/theorem-friendly scripts and result files
should remain in the workspace.

### 88. First Empirical Test Setup For Exact Profile Family Selection Over The Full Baseline Active Support

The next experiment removes every rule-level protected-core preclassification.

For each benchmark:

1. take the `safe-swap confset` active support `\hat S`
2. build the full subset family:
   - `\mathcal F(\hat S) = \{ S : S \subseteq \hat S \}`
3. for each `S \in \mathcal F(\hat S)`:
   - perform a support-specific local profile refit
   - allow rule-specific kernel heights to readjust inside `S`
4. compare every candidate to the profiled full-support reference on the
   validation split using the same empirical-Bernstein familywise screen
5. select an inclusion-minimal admissible support, with profiled validation
   loss only as a tie-break

This is the first fully support-level test of the new direction.

Important properties:

- no heuristic `k`
- no heuristic ambiguity block
- no protected-core pre-decision that can remove the true support from the
  family
- exact finite family defined solely by the baseline active support

Speed strategy for this first run:

- run only the three precision-hard benchmarks first
- enumerate supports in decreasing size
- warm-start each candidate from its best already-solved immediate superset
- use moderate local profile-refit steps on GPU

The direction should only be kept if it beats the `safe-swap confset`
baseline on the hard cases without breaking recall.

### 89. Full Baseline-Active-Support Profile Family Selection Still Over-Pruned The Logical Cases

The first profile-family run over the entire baseline active support produced:

- `logical_clean_plus`:
  - baseline: recall `1.0`, precision `0.8571`
  - profile family: recall `0.8333`, precision `1.0`
  - deleted the false extra, but also deleted the true rule
    `J and K and L -> T : inhibition`
- `logical_context`:
  - baseline: recall `1.0`, precision `0.875`
  - profile family: recall `0.7143`, precision `1.0`
  - deleted the extra, but also deleted the two true inhibition rules
    `A and C and D -> T : inhibition`
    `E and F and G -> T : inhibition`
- `kernel_gaussian`:
  - baseline: recall `1.0`, precision `0.75`
  - profile family: recall `1.0`, precision `1.0`
  - exact recovery

So this direction also fails the theorem-friendly target.

#### 89.1. Why It Failed

The problem is no longer a rule-level certificate mistake. The failure moved to
the final ordering rule over the support family.

The implemented selector used:

- the confidence family relative to the full profiled support
- then **inclusion-minimal admissible support**
- with profiled validation loss only as a tie-break

That ordering is too aggressive.

If several smaller supports remain statistically admissible relative to the
full active support, the selector always chooses the smallest one, even when
that smaller support drops true higher-order inhibition rules whose loss gap is
still inside the confidence radius.

This is exactly what happened in the two logical benchmarks:

- the confidence family still contained over-pruned models
- minimality then forced deletion beyond the true support

So the failure is not:

- the family
- nor support-specific local profiling

It is the current **selection order inside the admissible family**.

#### 89.2. The Correct Theorem-Friendly Fix

The selector should not choose the smallest admissible support directly.

Instead, the correct finite-family target is:

1. define the admissible confidence family over `\mathcal F(\hat S)`
2. choose the member with the best profiled objective inside that family
3. use model size only as a secondary tie-break

In other words:

- confidence set = statistical validity filter
- profiled objective = primary model-selection criterion
- size = secondary regularizer

This is the exact finite-family analogue of an oracle selector.

Theorem shape:

- if `S^\star \subseteq \hat S`
- if the profiled empirical family uniformly concentrates around the
  population family
- and if `S^\star` uniquely minimizes the profiled population objective over
  `\mathcal F(\hat S)`

then the confidence-restricted profile selector recovers `S^\star` with high
probability.

#### 89.3. Speed Implication

The family itself is still small enough on the current hard cases.

The expensive part is candidate profiling, not family size. So the exact speed
agenda remains:

- parent warm starts
- short local profile updates
- GPU batching
- early screening once confidence failure is certain

The next experiment should therefore keep the same profile-family construction
but change only the final choice rule:

- **best profiled objective inside the admissible family**
- size only as tie-break

### 90. The Next Principled Direction Is A Profiled Structural-Risk Selector Over The Certified Active-Support Family

The last two support-level experiments identified the two real failure modes.

Failure mode A:
- `logical_clean_plus`, `kernel_gaussian`
- support-specific kernels were being compared with too weak a complexity term
- this is the original criterion-mismatch problem

Failure mode B:
- `logical_clean_plus`, `logical_context`
- even after profiling over the full active-support family, the selector chose
  the **smallest** admissible support
- this is an ordering problem inside the admissible family

So the next direction must fix both at once.

#### 90.1. Correct Selection Object

Let `\hat S` be the active support returned by the current theorem-friendly
baseline:

- `safe inactive superset`
- `safe active interaction block`
- `restricted local swap`
- `confidence-set-restricted final selection`

Define the final candidate family by:

- `\mathcal F(\hat S) = \{ S : S \subseteq \hat S \}`

For each `S \in \mathcal F(\hat S)`, define the profiled empirical criterion

```text
\hat Q_n(S) = \hat R_n(S) + \lambda_n d(S)
```

where:

- `\hat R_n(S)` is the support-specific profiled validation risk
- `d(S)` is the full model dimension
- `\lambda_n` is the explicit complexity scale

The correct dimension is not rule count. It is:

```text
d(S) = 1 + \sum_{r \in S} [ 1 + |U_r|(K - 1) ]
```

where:

- `1` is the base rate
- each rule contributes one coefficient
- each source in the rule contributes `K-1` free kernel-height parameters after
  normalization

This is exactly the quantity that matched the earlier direct-support audit.

#### 90.2. Correct Final Selector

The final selector should no longer be:

- smallest support inside a confidence family

Instead it should be:

```text
\hat S_final = argmin_{S \in \mathcal F(\hat S)} \hat Q_n(S)
```

or, if we want an explicit confidence version,

1. build a familywise confidence set over `\mathcal F(\hat S)` using the
   profiled validation losses
2. select the member of that confidence family with the smallest
   `\hat Q_n(S)`
3. use support size only as a tie-break

This fixes both previous errors:

- the full dimension penalty addresses criterion mismatch
- the primary ordering is the profiled penalized objective, not support size

#### 90.3. Why This Is The Right Theorem Shape

This is now a standard finite-family structural-risk problem.

Assume:

- coverage:
  - `S^\star \subseteq \hat S`
- uniform profile control:
  - `sup_{S \in \mathcal F(\hat S)} | \hat R_n(S) - R(S) |` is small with high
    probability
- uniqueness:
  - the true support `S^\star` uniquely minimizes the population penalized
    criterion

```text
Q(S) = R(S) + \lambda_n d(S)
```

- margin:
  - `Q(S) - Q(S^\star)` is uniformly positive for every `S != S^\star`

Then the empirical profiled selector recovers `S^\star` with high probability.

This is more honest than every recent protected-core direction:

- no heuristic `k`
- no dataset-specific block rule
- no local indispensability surrogate
- exact finite family
- explicit complexity accounting

#### 90.4. Why This Can Still Be Fast

The family size itself is not the main bottleneck on the present benchmark
supports.

The true cost is candidate profiling. So speed should come from exact-preserving
numerical design, not from changing the mathematical selector.

The right speed agenda is:

- parent-to-child warm starts over the subset lattice
- Gray-code or Hasse-diagram traversal
- short local kernel refits rather than long Adam runs
- coefficient refit before kernel steps
- lower bounds for branch-and-bound pruning
- GPU batching for sibling supports

So the next implementation target is:

- keep the certified active-support family
- use the profiled penalized objective with full kernel dimension
- remove size-first admissibility as the primary decision rule

This is currently the cleanest path that is:

- theorem-friendly
- non-heuristic
- and consistent with the empirical failure analysis so far

### 91. First Empirical Test Setup For The Profiled Structural-Risk Selector

The next experiment implements the selector implied by the previous section
directly.

Family:

- `\mathcal F(\hat S) = \{ S : S \subseteq \hat S \}`

Candidate score:

```text
\hat Q_n(S) = \hat R_n(S) + \lambda_n d(S)
```

implemented as the support-specific validation BIC with:

- full profiled local kernel refit
- full kernel-dimension penalty

Selection rule:

- choose the support with the smallest profiled structural-risk score
- use support size only as a tie-break

This first check is run on the three precision-hard benchmarks only:

- `logical_clean_plus`
- `logical_context`
- `kernel_gaussian`

Speed design for this first run:

- full support gets a longer profile fit
- smaller supports inherit warm starts from the best already-solved parent
- each benchmark runs in its own `tmux` session on its own GPU

This is the first experiment whose final selector matches the actual theorem
shape we now believe is correct.

### 92. The First Profiled Structural-Risk Selector Fixed `kernel_gaussian` But Still Over-Pruned Logical Inhibition

The first run of the profiled structural-risk selector produced:

- `logical_clean_plus`:
  - baseline: recall `1.0`, precision `0.8571`
  - structural-risk selector: recall `0.8333`, precision `1.0`
  - deleted the extra `L -> T : excitation`
  - but also deleted the true rule `J and K and L -> T : inhibition`
- `logical_context`:
  - baseline: recall `1.0`, precision `0.875`
  - structural-risk selector: recall `0.8571`, precision `1.0`
  - deleted the extra `A and B and D -> T : inhibition`
  - but also deleted the true rule `E and F and G -> T : inhibition`
- `kernel_gaussian`:
  - baseline: recall `1.0`, precision `0.75`
  - structural-risk selector: recall `1.0`, precision `1.0`
  - exact recovery

So this direction is still not good enough to replace the current
`safe-swap confset` baseline.

#### 92.1. What Changed Relative To The Previous Full-Family Experiment

The previous full-family experiment failed because:

- it selected the smallest admissible support inside the confidence family

This new experiment removed that ordering error and instead selected the member
with the best support-specific profiled criterion using:

- support-specific local kernel refit
- full kernel-dimension penalty

So the remaining failure is more informative.

#### 92.2. What The Remaining Failure Means

At this point the issue is no longer:

- rule-level indispensability certificates
- structural ambiguity blocks
- or size-first selection order

The remaining issue is that the current support-specific local profile refit is
still too favorable to simpler inhibitory structures on the logical benchmarks.

Empirically, the selector now does the right thing on the criterion-mismatch
case `kernel_gaussian`, but it still drops true higher-order inhibition rules on
the logical benchmarks.

So the problem has narrowed to:

- support-specific local profiling is still not close enough to the true
  profiled population criterion on those logical inhibition cases

#### 92.3. The Next Principled Correction

The next theorem-friendly direction should therefore keep the same final
selection principle:

- finite certified support family
- support-specific structural-risk objective
- full kernel-dimension penalty

but improve the **candidate profiling accuracy**, not the selector itself.

Concretely, the next exact-preserving direction is:

- use the same family `\mathcal F(\hat S)`
- keep the same full-dimension structural-risk ordering
- upgrade candidate evaluation from short local refits to a tighter
  support-specific profiling routine
- use parent warm starts and aggressive numerical reuse so that profiling gets
  better without becoming exponentially slower in practice

This is now the cleanest remaining decomposition:

- selector principle:
  - appears conceptually right
- numerical profile approximation:
  - still too weak on higher-order logical inhibition

### 93. The Remaining Problem Is Now A Certified Inner-Optimization Problem, Not A Search Problem

At this point the outer selection principle is mostly settled.

The mathematically correct target is:

```text
\hat Q_n(S) = \inf_{\theta \in \Theta(S)} \hat L_n(\theta) + \lambda_n d(S)
```

where:

- `S` is a support inside the certified active-support family
- `\Theta(S)` is the support-specific parameter space
- `\hat L_n` is the empirical validation loss
- `d(S)` is the full kernel-dimension complexity

The remaining empirical failures show that the current implementation is not
ranking supports by `\hat Q_n(S)` itself. It is ranking them by a **support-
dependent approximation** `\tilde Q_n(S)` produced by short local profiling.

That is the residual mismatch.

#### 93.1. Why This Creates The Current Failure Pattern

For simpler supports, short local profiling is often already close to the local
optimum.

For supports containing true higher-order inhibition rules, short local
profiling can remain far from the correct profiled optimum because:

- the kernel-height landscape is flatter
- the useful parameter couplings are more nonlinear
- good warm starts matter more

So the current algorithm can systematically favor simpler false supports even
when the *true* profiled objective would not.

This explains the observed pattern:

- `kernel_gaussian` can already be fixed
- logical higher-order inhibition still gets under-profiled and then deleted

#### 93.2. The Right Mathematical Fix

The next non-heuristic step is to make the inner profiling **certified**.

For each support `S`, we should compute deterministic bounds

```text
\underline Q_n(S) \le \hat Q_n(S) \le \overline Q_n(S)
```

where:

- `\overline Q_n(S)` comes from a feasible support-specific profiled fit
- `\underline Q_n(S)` comes from a valid lower bound on the same profiled
  structural-risk objective

Then the selector works on these certified bands rather than on a single
uncertified point estimate.

Selection rule:

1. maintain the current best support `S_best` with upper bound
   `\overline Q_n(S_best)`
2. discard any support `T` once

```text
\underline Q_n(T) > \overline Q_n(S_best)
```

3. continue refining only the unresolved supports until one support is
   certified optimal

This turns the problem into a branch-and-bound structural-risk solver over the
finite certified support family.

#### 93.3. What Can Actually Be Proved

If:

- `S^\star \subseteq \hat S`
- the true target support is inside the certified active-support family
- every support has valid optimization bands
  - `\underline Q_n(S) \le \hat Q_n(S) \le \overline Q_n(S)`
- empirical structural-risk uniformly concentrates around population
  structural-risk
- the population gap satisfies

```text
\inf_{S \neq S^\star} \{ Q(S) - Q(S^\star) \}
>
2 \Delta_n + 2 \varepsilon_n
```

where:

- `\Delta_n` is the statistical uniform concentration error
- `\varepsilon_n = \sup_S [\overline Q_n(S) - \underline Q_n(S)]`

then the certified selector recovers `S^\star` with high probability.

This is the cleanest theorem shape reached so far, because it separates:

- statistical error
- optimization error
- structural complexity

#### 93.4. Practical Consequence

The next implementation should not invent another selector heuristic.

It should keep:

- the same certified active-support family
- the same full kernel-dimension structural-risk objective

and only upgrade candidate evaluation into:

- a certified upper/lower-bound profiling routine
- with warm starts, bound reuse, and branch-and-bound pruning

So the remaining problem is no longer “how do we search better?”

It is:

- “how do we compute support-specific profiled structural risk with explicit
  optimization error control?”

### 94. First Empirical Check Of Tighter Support-Specific Profiling Inside The Same Structural-Risk Selector

Before implementing a full certified upper/lower-bound branch-and-bound solver,
the next empirical check should isolate the inner-optimization bottleneck more
directly.

Keep fixed:

- the same certified active-support family `\mathcal F(\hat S)`
- the same full kernel-dimension structural-risk objective
- the same final support ordering by profiled structural risk

Change only:

- make each support-specific profiling call materially tighter

The right first test is:

1. support-specific local kernel/coef fit
2. support-specific family-attribution refinement
3. a second support-specific full structural-risk polish

This still does **not** change the theorem object. It only reduces the
optimization error inside the same objective.

So this run is the first empirical check of the claim:

- the remaining logical failures come from under-profiled higher-order
  inhibition supports
- not from the structural-risk selector principle itself

### 95. Tighter Support-Specific Profiling Improved The Structural-Risk Selector To 2/3 Exact On The Hard Cases

The refined profiling run kept the same structural-risk selector and changed
only the support-specific inner optimization by using:

1. a first support-specific structural-risk fit
2. support-specific `family_attribution_refine`
3. a second support-specific structural-risk polish

Results:

- `logical_clean_plus`:
  - baseline: recall `1.0`, precision `0.8571`
  - previous structural-risk run: recall `0.8333`, precision `1.0`
  - refined profiling run: recall `1.0`, precision `1.0`
  - exact recovery
- `logical_context`:
  - baseline: recall `1.0`, precision `0.875`
  - previous structural-risk run: recall `0.8571`, precision `1.0`
  - refined profiling run: recall `0.8571`, precision `1.0`
  - still misses:
    - `E and F and G -> T : inhibition`
- `kernel_gaussian`:
  - baseline: recall `1.0`, precision `0.75`
  - previous structural-risk run: recall `1.0`, precision `1.0`
  - refined profiling run: recall `1.0`, precision `1.0`
  - exact recovery

So the refined profiling run is the strongest support-level theorem-friendly
signal so far after the baseline:

- `2/3` hard cases exact
- and the remaining failure is now isolated to `logical_context`

#### 95.1. What This Means

This result supports the previous diagnosis.

The outer selector principle:

- certified active-support family
- support-specific structural-risk objective
- full kernel-dimension penalty

does look substantially better once candidate profiling is tightened.

So the current bottleneck is even narrower than before:

- not the support family
- not the structural-risk penalty
- not the size-first ordering error
- but the remaining profiling inaccuracy on the hardest logical inhibition case

#### 95.2. Practical Interpretation

`logical_context` is now the diagnostic benchmark.

The next improvement should focus only on making the support-specific profile
evaluation tighter on this case, while keeping the same outer selector.

That is exactly consistent with the certified-bounds direction:

- same finite support family
- same structural-risk objective
- tighter control of inner optimization error

### 96. Full Remaining-Suite Check For The Refined Structural-Risk Selector

After the hard-case run reached:

- exact on `logical_clean_plus`
- exact on `kernel_gaussian`
- one remaining miss on `logical_context`

the next step is to run the same selector on the remaining benchmark suite:

- `logical_shared`
- `kernel_triangular`
- `kernel_exponential`
- `num_predicates_10`
- `num_predicates_20`
- `ablation_excitation_only`
- `ablation_inhibition_only`
- `ablation_mixed_sign`

This run keeps exactly the same method as the hard-case test:

- same finite family over the baseline active support
- same full kernel-dimension structural-risk objective
- same refined support-specific profiling routine

No new selector heuristic is introduced in this run. The only caveat remains
the same as before:

- inner support-specific profiling is still numerical rather than fully
  certified

So this run is a pure generalization check, not a method change.

### 97. Full Remaining-Suite Run Showed That The Refined Structural-Risk Selector Is Still Not A Full Replacement For `safe-swap confset`

The remaining eight benchmarks were run with the same refined selector used on
the three hard cases.

Results on the remaining eight:

- exact:
  - `logical_shared`
  - `kernel_triangular`
  - `num_predicates_10`
  - `ablation_excitation_only`
  - `ablation_mixed_sign`
- failures:
  - `kernel_exponential`
    - missing `D and F and G -> T : inhibition`
    - recall `0.8333`
  - `num_predicates_20`
    - missing `D and F and G -> T : inhibition`
    - recall `0.8333`
  - `ablation_inhibition_only`
    - only kept `A -> T : inhibition`, `B -> T : inhibition`
    - recall `0.3333`

Combining these with the earlier hard-case run gives:

- exact `7/11`
- mean recall `0.8961`
- mean precision `1.0`

So the refined structural-risk selector is **not** a better full theorem-
friendly replacement than the current `safe-swap confset` baseline.

#### 97.1. What This New Failure Pattern Means

The remaining problem is no longer isolated to `logical_context`.

The full-suite run shows a broader pattern:

- the refined structural-risk selector still systematically under-values some
  true inhibitory higher-order rules
- the strongest evidence is `ablation_inhibition_only`, where the selector
  collapses to a very small inhibition support

So the issue is more structural than a single benchmark-specific profiling
failure.

#### 97.2. Updated Conclusion

The refined structural-risk direction improved the three hard cases, but it is
still not stable enough across the full suite to keep as the main candidate.

Therefore:

- as a full method, this direction should be treated as a dead end
- the current best theorem-friendly baseline remains `safe-swap confset`

What survives conceptually is narrower:

- support-specific profiling matters
- full kernel-dimension accounting matters

But the exact refined-selector implementation itself should not be kept as the
current front-runner.

### 70. Protected-Core + Tiny-Ambiguity Enumeration: First Positive Signal

Date:
- 2026-04-06

Goal:
- stay on top of the best theorem-friendly baseline
  - `safe inactive superset + safe active interaction block + restricted local swap + confidence-set-restricted final selection`
- change only the final precision-repair stage
- avoid global deletion pressure
- keep the exact search space tiny

Construction:

1. Start from the baseline active support `\hat S`.
2. For each active rule, test whether the single-drop candidate remains inside
   the split confidence set.
3. Rules with no admissible single-drop witness are declared `protected core`.
4. Rules that do admit a single-drop witness seed a tiny `ambiguity set`.
5. Close that ambiguity set only under local overlap / subset / sign-conflict
   relations.
6. Enumerate all subsets only inside that ambiguity set.
7. Keep the protected core fixed.
8. Allow local kernel refit only for rules inside the ambiguity subset.
9. Refit coefficients exactly on `core + ambiguity-subset`, then select the
   best confidence-admissible candidate.

Why this differs from the failed directions:

- Unlike global full-DF closure, it does **not** expose every true rule to
  global deletion pressure.
- Unlike anchor-fixed deletion, it does allow local kernel reshaping where the
  ambiguity actually lives.
- Unlike the previous local ambiguity-block witness run, the selector is now
  centered on a **protected core + tiny ambiguous block** rather than on a
  deletion-first local sweep over the whole active support.

Interim result on `logical_clean_plus`:

- baseline:
  - recall `1.0`
  - precision `0.8571`
  - extra:
    - `L -> T : excitation`
- protected-core selector:
  - recall `1.0`
  - precision `1.0`
  - missing:
    - none
  - extra:
    - none
  - elapsed:
    - `491.86s`

Internal structure on this case:

- protected core size:
  - `5`
- ambiguity set size:
  - `2`
- ambiguous rules:
  - `exc:11`
  - `inh:297`
- chosen ambiguity support:
  - keep only `inh:297`
  - drop `exc:11`

Interpretation:

- This is the first positive signal that the precision failure in
  `logical_clean_plus` was truly localized.
- The extra excitation rule and the true `J and K and L -> T : inhibition`
  rule form a tiny local ambiguity block.
- Once all other rules are protected, the exact local family search can remove
  the extra rule without collapsing recall.

Current status:

- `logical_clean_plus` finished successfully and became exact.
- `logical_context` and `kernel_gaussian` are still running under the same
  selector, so the direction is still provisional.

Update:

- `kernel_gaussian` also finished successfully and became exact.

Details on `kernel_gaussian`:

- baseline:
  - recall `1.0`
  - precision `0.75`
  - extra:
    - `B and C and D -> T : inhibition`
    - `G and H -> T : inhibition`
- protected-core selector:
  - recall `1.0`
  - precision `1.0`
  - missing:
    - none
  - extra:
    - none
  - elapsed:
    - `2871.34s`

Internal structure on this case:

- protected core size:
  - `2`
- ambiguity set size:
  - `6`
- current ambiguous support:
  - excitatory:
    - `21`
    - `81`
  - inhibitory:
    - `1`
    - `35`
    - `57`
    - `85`
- chosen ambiguity support:
  - excitatory:
    - `21`
    - `81`
  - inhibitory:
    - `1`
    - `85`

Interpretation:

- This is important because `kernel_gaussian` was one of the two cases that
  the anchor-fixed final selector could not fix at all.
- The new protected-core construction seems to be doing the right thing:
  it shields the indisputable rules, then resolves only the genuinely coupled
  local family where the extra inhibitory explanations live.
- The ambiguity family is still small enough to enumerate exactly here
  (`2^6 = 64`), so the approach remains much faster than global closure over
  the full active support.

Revised status:

- exact on `logical_clean_plus`
- exact on `kernel_gaussian`
- `logical_context` still running

If `logical_context` also becomes exact, then this will be the first
theorem-friendly precision-repair direction that fixes **all three** hard
precision failures while preserving recall on the cases tested so far.

### 71. Protected-Core Selector on All Three Hard Precision Cases

Date:
- 2026-04-06

Final hard-case results:

- `logical_clean_plus`
  - baseline:
    - recall `1.0`
    - precision `0.8571`
  - protected-core selector:
    - recall `1.0`
    - precision `1.0`
    - exact recovery
  - elapsed:
    - `491.86s`

- `kernel_gaussian`
  - baseline:
    - recall `1.0`
    - precision `0.75`
  - protected-core selector:
    - recall `1.0`
    - precision `1.0`
    - exact recovery
  - elapsed:
    - `2871.34s`

- `logical_context`
  - baseline:
    - recall `1.0`
    - precision `0.875`
  - protected-core selector:
    - recall `0.8571`
    - precision `1.0`
  - missing:
    - `E and F and G -> T : inhibition`
  - extra:
    - none
  - elapsed:
    - `6936.86s`

Bottom line:

- exact on `2 / 3`
- precision failure is removed on all three
- but recall is preserved only on two of the three
- so this is **not yet** the final theorem-friendly fix

Why `logical_context` failed:

- On this case, the protected core collapsed to size `0`.
- The ambiguity set became the entire active support of size `8`.
- So the selector effectively lost the main protection mechanism and reverted
  to a near-global local enumeration problem over the whole support family.
- Under that configuration, the selector again preferred deleting the true
  higher-order inhibition rule `E and F and G -> T : inhibition`.

Interpretation:

- The idea is still materially better than the previous dead-end directions:
  it fixed both `logical_clean_plus` and `kernel_gaussian` without hurting
  recall.
- But the current definition of `protected core`
  - "no single-drop confidence-admissible witness"
  is too weak.
- In `logical_context`, that test certifies nothing, so the method degenerates.

What this teaches us:

- the right direction is probably still
  - baseline `safe-swap confset`
  - plus a **localized exact ambiguity repair**
- but the localization cannot depend only on single-drop admissibility
- it must certify a nonempty stable core even when several rules are jointly
  entangled

This suggests the next refinement should be a stronger, still non-heuristic
core definition, for example a `k`-stable protected core or a witness notion
based on grouped deletions rather than only single deletions.

### 72. Recommended Next Direction: `k`-Stable Protected Core + Exact Local Ambiguity Repair

Date:
- 2026-04-06

Current evidence now points to one main conclusion:

- do **not** change the search stage
- keep the best theorem-friendly baseline:
  - `safe inactive superset + safe active interaction block + restricted local swap + confidence-set-restricted final selection`
- change only the final precision-repair layer

Why this is the right place to intervene:

- the baseline already gives full recall on the full theorem-friendly suite
- global full-closure methods destroyed recall
- anchor-fixed deletion was too weak
- single-drop protected-core repair fixed `logical_clean_plus` and
  `kernel_gaussian`, but failed on `logical_context`
- the failure mode in `logical_context` was exactly:
  - protected core collapsed to size `0`
  - ambiguity set expanded to the entire active support

So the next step should be:

- replace **single-drop protected core**
- with a stronger **group-stable protected core**

Proposed object:

- `k`-stable protected core, starting with `k = 2`

Definition sketch:

1. Let `\hat S` be the baseline active support.
2. Build local overlap blocks on `\hat S`.
3. For each block `B`, enumerate all deletions of size at most `k` inside `B`
   while holding all rules outside `B` fixed.
4. For each candidate:
   - perform exact support-fixed coefficient refit
   - optionally allow kernel refit only inside `B`
   - test confidence-set admissibility
5. Let `\mathcal F_k(B)` be the finite family of confidence-admissible,
   objective-undominated candidates.
6. Define:
   - protected core:
     - `C_k(B) = intersection_{S in \mathcal F_k(B)} S`
   - ambiguity set:
     - `A_k(B) = union_{S in \mathcal F_k(B)} S \ C_k(B)`
7. Keep `C_k(B)` fixed.
8. Perform exact enumeration only on `A_k(B)`.

Why this is better than the current protected-core definition:

- the current rule
  - "a rule is protected if no single-drop candidate is admissible"
  is too weak
- it cannot certify jointly entangled true rules
- `logical_context` is the direct counterexample
- a grouped witness can protect a rule that is individually droppable but not
  jointly removable with its coupled neighbors

Why this remains non-heuristic:

- the family is finite and explicitly defined
- every local candidate is exactly evaluated
- the core and ambiguity sets are defined by exact set operations on that
  family
- the final selector is still exact over a certified finite family

Why this should still be fast:

- screening cost becomes:
  - `sum_B sum_{j=0}^k C(|B|, j)`
- not global `2^{|\hat S|}`
- if `k = 2` and blocks stay small, this is cheap
- the expensive exact enumeration is then reserved only for the reduced
  ambiguity sets `A_k(B)`

Why this is the best next theorem direction:

- It keeps the only full-recall theorem-friendly baseline intact.
- It explains the two successful repairs
  - `logical_clean_plus`
  - `kernel_gaussian`
  as genuinely local ambiguity resolution.
- It directly targets the only remaining failure:
  - empty protected core under single-drop admissibility.

Theorem shape to aim for:

Assumptions:

- coverage:
  - true support lies inside the baseline certified family
- local `k`-stability:
  - every true core rule survives all confidence-admissible deletions of size
    at most `k` in its local block
- local ambiguity bound:
  - false extras are confined to finite small ambiguity sets
- local margin:
  - inside each ambiguity set, the true restricted support uniquely minimizes
    population risk after the allowed local refit

Conclusion:

- the `k`-stable protected core contains all true indispensable rules
- extras lie only inside the ambiguity sets
- exact blockwise ambiguity selection recovers the true support

Recommended next experiment order:

1. implement `k = 2` stable-core screening on top of the current baseline
2. rerun the three hard precision cases only
3. only if all three become exact, expand to the full 11-benchmark suite

### 73. Clarification: A Fixed `k` Is Heuristic Unless It Is Fully Certified

Date:
- 2026-04-06

Important correction:

- a rule like
  - "choose `k = 2`"
  is **heuristic** if `k` is selected for convenience or because it empirically
  looks good
- that is not the right final theorem direction

So the previous `k`-stable-core proposal should be interpreted only as a
diagnostic stepping stone, not as the final mathematically clean method.

What is truly non-heuristic:

- once a finite family is explicitly fixed, exact selection over that family is
  non-heuristic
- therefore the real question is not
  - "which `k` should we choose?"
- but instead
  - "how do we define the finite family itself without an ad hoc truncation
    parameter?"

The cleaner direction is:

- **certified ambiguity block + exact block family selection**

Formal sketch:

1. Start from the baseline active support `\hat S`.
2. Partition `\hat S` into deterministic structural blocks using only
   rule-level objects already present in the model:
   - source overlap
   - subset/superset relation
   - sign conflict
3. For each block `B`, define the local family exactly as:
   - all supports obtained by keeping `\hat S \setminus B` fixed
   - and allowing any subset of `B`
4. Evaluate every support in that local family exactly:
   - exact admissibility test
   - exact local refit under the allowed model class
   - exact objective
5. Define:
   - protected core in block:
     - intersection of all admissible-undominated local supports
   - ambiguity set in block:
     - union minus intersection
6. Perform final exact selection only over the ambiguity sets.

Why this removes the heuristic `k`:

- no truncation level is chosen
- no neighborhood radius is tuned
- every subset inside the certified local block is considered
- the only reduction comes from the deterministic block construction itself

Where the proof burden moves:

- the theorem must now justify **block localization**
- not a user-chosen deletion depth `k`

That is much cleaner:

- if all false extras are confined to their structural ambiguity blocks
- and true indispensable rules belong to the intersection of admissible local
  supports in those blocks
- then exact blockwise selection recovers the true support

Speed implication:

- this is fast only if the certified blocks are small
- so the main algorithm-design problem is now:
  - build a deterministic block definition that stays small on the real hard
    cases

Revised conclusion:

- fixed-`k` stable-core is useful as an exploratory lens
- but the non-heuristic target should be:
  - deterministic ambiguity blocks
  - exact subset family inside each block
  - core/ambiguity defined by exact set operations over that full block family

### 74. Research Constraint: Every Step Must Be Mathematically Justified and Non-Dataset-Specific

Date:
- 2026-04-06

From this point on, the research program should obey the following hard
constraint:

- every algorithmic step must have a mathematical justification
- and that justification must be stated at the level of the model/statistical
  problem
- not at the level of a particular benchmark or dataset idiosyncrasy

This means the following are **not acceptable** as final ingredients:

- choosing a deletion depth because it worked on the current hard cases
- hand-tuning thresholds per benchmark
- treating `logical_context` differently because we saw it fail
- adding special handling for 3-way inhibition only because it appears in the
  present suite
- any repair rule whose definition depends on the names or empirical quirks of
  current datasets

This also means the following **are acceptable**:

- deterministic constructions from the model class
  - rule overlap graph
  - subset/superset relations
  - sign-conflict relations
  - kernel-interaction operators defined from the feature map
- exact finite-family procedures
  - exact subset enumeration over a formally defined family
  - confidence-set admissibility
  - objective minimization over that family
- data-dependent quantities that come from a theorem
  - empirical risk
  - empirical Bernstein radii
  - split confidence sets
  - penalties derived from parameter dimension and sample size

So the next method must be of the following form:

1. define a finite candidate family using only structural properties of the
   model and the certified baseline output
2. prove that the true support belongs to that family under explicit
   assumptions
3. define the final selector exactly over that family
4. prove finite-sample control for the selector
5. state exact recovery only as a corollary under an additional margin /
   identifiability condition

In particular, the block construction itself must be justified at the model
level.

The right target is therefore not merely:

- "small blocks that work well empirically"

but:

- "blocks induced by a mathematically defined interaction relation for which
  cross-block effects are either zero or uniformly controllable"

The theorem program should look like this:

Definitions:

- define a rule-interaction graph `G`
- define its connected components as ambiguity blocks
- define the certified local family for each block
- define core and ambiguity sets as exact intersection / union operations over
  admissible-undominated local supports

Lemmas:

- structural localization lemma:
  - false alternatives that can compete with the truth must remain inside the
    ambiguity block containing their interacting rules
- cross-block stability lemma:
  - modifying one block perturbs the score of another block only through a term
    that is either zero or bounded by an explicit radius
- core-retention lemma:
  - truly indispensable rules lie in the intersection of all admissible
    undominated supports in their block

Main theorem:

- exact blockwise family selection enjoys finite-sample oracle control over the
  certified family

Corollary:

- if each block satisfies an explicit population margin / identifiability
  condition, then the selector achieves exact support recovery

Practical consequence:

- future experiments should be presented only as tests of these mathematically
  defined objects
- not as benchmark-specific tricks

### 75. First Run Of Deterministic Structural-Block Family Selection

Date:
- 2026-04-06

Goal:

- test the first fully block-based version of the non-heuristic direction
- keep the best theorem-friendly baseline fixed
- replace the final repair stage by:
  - deterministic structural block decomposition
  - exact subset family inside each block
  - blockwise admissible-undominated family reduction
  - exact selection over the reduced global family

Method instantiated for this run:

1. Start from the `safe-swap confset` active support.
2. Build the structural interaction graph on active rules using only:
   - source overlap
   - subset/superset relation
   - sign-conflict overlap
3. Take connected components as structural ambiguity blocks.
4. For each block:
   - enumerate every subset of that block
   - keep all rules outside the block fixed
   - allow local kernel refit for the kept rules inside the block
   - perform exact support-fixed coefficient refit
   - test confidence-set admissibility against the baseline active support
   - retain the admissible-undominated local family
5. Build the reduced global family as the product of the retained local block
   families.
6. Select exactly over that reduced global family.

Why this is the right next experiment:

- no ad hoc depth parameter like `k`
- no benchmark-specific threshold
- every candidate family is explicitly defined
- every candidate in that family is explicitly evaluated
- the only reduction step is exact dominance pruning inside a deterministic
  structural block family

Current run configuration:

- target cases:
  - `logical_clean_plus`
  - `logical_context`
  - `kernel_gaussian`
- device:
  - 3-way GPU parallel
- local kernel optimization steps:
  - `30`
- purpose:
  - fast first signal on whether deterministic block-family localization can
    keep the two positive fixes and avoid the `logical_context` recall drop

Files:

- script:
  - `workspace/train/paper_benchmark_active/tmp_run_structural_block_family_selector.py`
- outputs:
  - `data/paper_suite/tmp_structural_block_clean_plus_20260406.json`
  - `data/paper_suite/tmp_structural_block_context_20260406.json`
  - `data/paper_suite/tmp_structural_block_kernel_gaussian_20260406.json`

### 76. Structural Overlap Blocks Were Too Weak

Date:
- 2026-04-06

Outcome:

- this direction was stopped as a dead end and the experiment code was removed

Reason:

- the first completed case, `logical_clean_plus`, showed **no improvement**
- baseline:
  - recall `1.0`
  - precision `0.8571`
- structural-block family selector:
  - recall `1.0`
  - precision `0.8571`
  - extra:
    - `L -> T : excitation`

What still looked good:

- the decomposition was fast
- `logical_clean_plus` was partitioned into block sizes:
  - `[1, 2, 1, 1, 1, 1]`
- the reduced global family size became only:
  - `2`

Why this is still a failure:

- the whole point of localization was to preserve the two successful fixes from
  the previous protected-core experiment while keeping the method exact and
  fast
- here, the localization became cheap, but it lost the ability to remove the
  extra rule even on the cleanest positive case

What this teaches us:

- **structural overlap alone is not the right interaction notion**
- source overlap / subset relation / sign conflict are not enough to define the
  true statistical ambiguity classes
- two rules may be structurally connected but still not compete in the right
  objective geometry
- conversely, two rules may compete because their score directions are locally
  close even when the crude combinatorial graph is too weak or too coarse

So the next direction must replace:

- combinatorial structural blocks

with:

- **certified score/Fisher interaction blocks**

### 77. Next Direction: Certified Tangent-Interaction Core + Exact Ambiguity Family

Date:
- 2026-04-06

This is now the best mathematically grounded next direction.

Core idea:

- the final selector should not be based on raw rule overlap
- it should be based on the **local geometry of the risk / likelihood**

Model-level object:

- for each active rule `r`, define its local tangent block
  - coefficient derivative
  - kernel-height derivatives
- evaluate these derivatives at the certified baseline fit
- this gives a rule-specific tangent subspace `T_r`

Then define a rule-interaction graph by a mathematically justified criterion:

- connect `r` and `s` if their residualized tangent blocks are not
  certifiably orthogonal / separable
- equivalently:
  - the off-diagonal Fisher or Hessian block between `T_r` and `T_s` is larger
    than its confidence radius
  - or the distance of one tangent block to the span of the others cannot be
    certified away

Why this is better:

- it is derived from the model likelihood geometry
- not from benchmark-specific patterns
- not from a hand-chosen depth parameter
- not from a combinatorial proxy alone

Definitions for the next method:

1. Build the certified tangent-interaction graph on active rules.
2. Take connected components as ambiguity blocks.
3. Inside each block, form the exact local family of support subsets with the
   allowed local kernel refit.
4. Use exact admissibility and exact objective evaluation inside that block.
5. Define:
   - local core:
     - rules whose tangent contribution remains indispensable across the
       admissible-undominated local family
   - local ambiguity set:
     - the remaining rules inside the block
6. Perform exact selection only over the product of the local ambiguity
   families.

The key new certificate:

- a rule should be declared protected **not** because single deletion fails
- and **not** because it sits in an isolated overlap component
- but because its tangent block has certified distance from the span of the
  competing local alternatives

This directly targets the previous failures:

- `logical_context`:
  - single-drop protection failed because true rules were jointly entangled
  - tangent-space indispensability is the right stronger notion
- `logical_clean_plus`:
  - crude structural blocks localized the pair but still did not encode the
    right competition geometry
  - tangent-space competition should
    distinguish a truly redundant extra explanation from a necessary rule
- `kernel_gaussian`:
  - the ambiguity seems genuinely local, so a score/Fisher interaction block
    should still stay small

Why this remains non-heuristic:

- the graph is defined by a theorem-backed interaction quantity
- confidence radii come from concentration bounds, not hand tuning
- the candidate family is finite and explicit
- the final selector is exact over that family

Why this can still be fast:

- the expensive part moves to linear algebra on the tangent Gram / Fisher
  blocks
- exact subset enumeration is then needed only inside the small certified
  ambiguity blocks
- this is much cheaper than global closure and more principled than overlap-only
  localization

Theorem path:

1. define empirical tangent blocks and the empirical interaction matrix
2. prove a uniform concentration bound to the population interaction matrix
3. prove a structural localization lemma:
   - supports outside the connected interaction block are separable up to an
     explicit remainder
4. prove a core-indispensability lemma:
   - rules with tangent blocks separated from the competing span by a margin
     stay in the local core
5. prove exact blockwise family selection over the certified ambiguity family
6. derive exact recovery as a corollary under a local identifiability margin

Recommended next implementation:

- build the tangent / Fisher interaction matrix on the baseline active support
- inspect its connected components on the three hard cases
- only then implement the exact block-family selector on those certified
  components

### 78. First Full-Suite Run Of Score-Interaction Block Family Selection

Date:
- 2026-04-06

This run is the first implementation of the tangent/Fisher-block direction in a
fast experimental form.

Implemented approximation:

- the interaction graph is built from **rule-level coefficient score
  directions** at the certified baseline fit
- for each active rule:
  - excitatory direction:
    - event contribution `-f_r / eta`
    - grid contribution `f_r * w * exp(-inh)`
  - inhibitory direction:
    - event contribution `g_r`
    - grid contribution `-g_r * w * eta * exp(-inh)`
- these score directions are normalized to unit empirical RMS
- two rules are connected if their empirical score correlation is not
  certifiably zero under an empirical Bernstein bound

This is not yet the full tangent-block theorem target because:

- the current graph uses coefficient score directions only
- it does not yet include the kernel-height tangent coordinates

But it is still a model-based, non-dataset-specific interaction definition and
is the fastest first full-suite test of the score/Fisher direction.

Selection procedure in this run:

1. start from the `safe-swap confset` active support
2. build score-interaction connected components
3. inside each component, enumerate every local subset
4. hold all rules outside the component fixed
5. allow local kernel refit for the kept subset
6. keep every confidence-admissible local candidate
7. form the reduced global family as the product of those local confident
   families
8. perform exact global selection over that reduced family

Speed settings:

- 4-way GPU parallel sharding
- `local_steps = 20`
- target:
  - fast first signal on all 11 theorem-friendly benchmarks

Shard layout:

- GPU 0:
  - `logical_context`
  - `logical_shared`
  - `ablation_inhibition_only`
- GPU 1:
  - `kernel_gaussian`
  - `kernel_exponential`
  - `num_predicates_20`
- GPU 2:
  - `logical_clean_plus`
  - `kernel_triangular`
  - `num_predicates_10`
- GPU 3:
  - `ablation_excitation_only`
  - `ablation_mixed_sign`

Files:

- script:
  - `workspace/train/paper_benchmark_active/tmp_run_score_interaction_block_family_selector.py`
- shard outputs:
  - `data/paper_suite/tmp_score_interaction_suite_gpu0_20260406.json`
  - `data/paper_suite/tmp_score_interaction_suite_gpu1_20260406.json`
  - `data/paper_suite/tmp_score_interaction_suite_gpu2_20260406.json`
  - `data/paper_suite/tmp_score_interaction_suite_gpu3_20260406.json`

### 79. Full-Suite Result Of Score-Interaction Block Family Selection

Date:
- 2026-04-06

Final full-suite summary:

- completed:
  - `11 / 11`
- exact:
  - `7 / 11`
- mean recall:
  - `0.8961`
- mean precision:
  - `1.0`

Per-benchmark result:

- exact:
  - `logical_clean_plus`
  - `logical_shared`
  - `kernel_gaussian`
  - `kernel_triangular`
  - `num_predicates_10`
  - `ablation_excitation_only`
  - `ablation_mixed_sign`

- recall loss:
  - `logical_context`
    - missing:
      - `E and F and G -> T : inhibition`
    - recall:
      - `0.8571`
  - `kernel_exponential`
    - missing:
      - `D and F and G -> T : inhibition`
    - recall:
      - `0.8333`
  - `num_predicates_20`
    - missing:
      - `D and F and G -> T : inhibition`
    - recall:
      - `0.8333`
  - `ablation_inhibition_only`
    - missing:
      - `C and D -> T : inhibition`
      - `C and G and H -> T : inhibition`
      - `D and F and G -> T : inhibition`
      - `E and F -> T : inhibition`
    - recall:
      - `0.3333`

Important positive signals:

- `logical_clean_plus` was fixed exactly:
  - precision repaired to `1.0`
  - recall preserved
- `kernel_gaussian` was fixed exactly:
  - both extra inhibitory rules removed
  - recall preserved
- `logical_shared` also remained exact under this selector

Important failure pattern:

- the selector still over-prunes true inhibition rules in several benchmarks
- the repeated victim is again:
  - `D and F and G -> T : inhibition`
- the most dramatic failure is:
  - `ablation_inhibition_only`
  - where the interaction graph produced one large 6-rule block with
    `58 / 64` confidence-admissible local candidates, and the final selector
    collapsed to only the two singleton inhibitions

Interpretation:

- the score-interaction graph is **strictly better** than crude structural
  overlap
  - because it exactly fixes `logical_clean_plus`
  - and exactly fixes `kernel_gaussian`
- but coefficient-score interactions alone are still too weak to protect the
  higher-order inhibition rules in the hard negative geometry
- in particular, the current graph does not yet encode enough notion of
  indispensability for purely inhibitory higher-order rules

What this means for the theorem program:

- the score/Fisher direction remains promising
- but the current implementation is not sufficient
- the missing piece is likely the **full tangent block**, not only the
  coefficient-score direction
- kernel-height tangent directions and blockwise span-separation certificates
  still need to be included

Updated bottom line:

- `source-overlap blocks` were too crude
- `coefficient-score blocks` are better, but still not enough
- the next mathematically clean target should be:
  - full tangent-block interaction
  - span-separation / indispensability certificates
  - exact ambiguity family only after that stronger core certificate

### 80. Why Score-Only Interaction Still Failed

Date:
- 2026-04-06

The score-interaction experiment clarified one precise point:

- coefficient-score directions alone do not carry enough information to protect
  higher-order inhibition rules

Model-level reason:

- for an excitatory rule with feature `f_r`, the coefficient score direction is
  essentially
  - event part:
    - `- f_r / eta`
  - grid part:
    - `+ w * exp(-I) * f_r`
- for an inhibitory rule with feature `g_r`, the coefficient score direction is
  essentially
  - event part:
    - `+ g_r`
  - grid part:
    - `- w * eta * exp(-I) * g_r`

So the coefficient-score geometry only sees the **current rule feature**.

What it does not see:

- how the feature itself changes when the rule-specific kernel shape changes
- whether a higher-order inhibition rule has a kernel-height tangent direction
  that cannot be reproduced by lower-order or singleton inhibitory rules

This explains the repeated failure:

- in pure or mostly inhibitory settings, several inhibition rules can have
  coefficient-score directions that are too aligned
- then the interaction graph merges them into one large block
- and the final selector still prefers smaller supports
- so true higher-order inhibition rules are deleted

Therefore the missing object is not another better combinatorial block rule.

It is:

- the **full tangent block** of each rule

### 81. Next Direction: Full Tangent-Block Indispensability Certificate

Date:
- 2026-04-06

This is now the recommended research target.

Rule parameter block:

- for each active rule `r`, define its parameter block
  - coefficient:
    - `a_r` for excitation or `b_r` for inhibition
  - kernel-height coordinates:
    - the free simplex coordinates of each source-specific kernel

At the baseline fit `\hat theta`, define the tangent block `T_r` as the matrix
whose columns are the loss-direction derivatives with respect to that entire
parameter block.

Concretely:

- coefficient directions are the score directions above
- kernel-height directions come from the chain rule:
  - derivative of the rule feature with respect to each free kernel-height
    coordinate
  - multiplied by the corresponding coefficient effect on the loss

Important consequence:

- two rules that look similar at the coefficient-score level can still be
  distinguishable once their kernel-height tangent directions are included
- this is exactly what the current failed score-only graph was unable to detect

Protected-core definition should then be:

- a rule is protected if its tangent block has certified separation from the
  span of the tangent blocks of the competing alternatives

Mathematically clean versions of that certificate:

- Schur-complement information lower bound
- minimal singular value of the residualized tangent block
- or a blockwise partial-Fisher norm bounded away from zero

In words:

- after projecting out the tangent directions of the rival rules, this rule
  still has a nontrivial direction that the rivals cannot imitate

Why this is the right object:

- it is model-based
- it is not benchmark-specific
- it does not require a heuristic depth parameter
- it directly encodes indispensability rather than hoping deletion tests will
  discover it

Resulting selector:

1. keep the current `safe-swap confset` search unchanged
2. fit the baseline active support
3. build the full tangent block for every active rule
4. compute the empirical block-Fisher / block-Gram matrix
5. declare a rule protected when its residualized block information exceeds its
   confidence radius
6. build ambiguity blocks only among the non-protected rules whose residualized
   interaction is not certifiably zero
7. enumerate the exact family only inside those ambiguity blocks

Why this should be faster than the failed global methods:

- the expensive step is a one-time block linear algebra computation
- exact family search is postponed until after the core certificate removes the
  indisputable rules
- so the exhaustive part should be much smaller than global closure

Theorem program:

1. concentration of the empirical block-Fisher matrix around the population
   block-Fisher matrix
2. block-separation lemma:
   - if the residualized population information of rule `r` is bounded below by
     a margin, then `r` is protected with high probability
3. block-localization lemma:
   - only rules with non-negligible residualized interaction can belong to the
     same ambiguity block
4. exact ambiguity-family oracle bound
5. exact recovery corollary under local identifiability margins

Practical implication:

- the next actual implementation should not be another selector first
- it should first compute:
  - full tangent blocks
  - residualized block-Fisher certificates
  - ambiguity blocks induced by those certificates

Only after that should exact family selection be reintroduced.

### 82. First Run Of Full-Tangent Protected-Core Selector

Date:
- 2026-04-06

This is the first implementation that actually uses the full tangent block
idea, rather than only coefficient-score directions.

Implemented object:

- start from the `safe-swap confset` active support
- fit that support with the frozen-kernel baseline state used by the previous
  theorem-friendly selectors
- for each active rule, build its **full tangent block** on the validation loss
  vector:
  - coefficient direction
  - kernel-height directions for every source-specific kernel free coordinate
    under the area-normalization constraint
- orthonormalize that rule block
- residualize it against the span of all other rule blocks
- declare a rule protected if its residualized tangent block keeps full rank
- define the ambiguity set as the remaining active rules
- perform exact familywise selection over all subsets of the ambiguity set
  with local kernel refit only inside that ambiguity set

Why this is more faithful to the intended theorem direction:

- it uses the whole rule parameter block
  - not just source-overlap structure
  - not just coefficient-score directions
- kernel-shape directions are now explicitly part of the protection test
- the only exact enumeration happens after the tangent-based core reduction

Current run setup:

- target cases:
  - `logical_clean_plus`
  - `logical_context`
  - `kernel_gaussian`
- parallelization:
  - 3-way GPU `tmux`
- local kernel optimization steps:
  - `30`

Files:

- script:
  - `workspace/train/paper_benchmark_active/tmp_run_full_tangent_protected_core_selector.py`
- outputs:
  - `data/paper_suite/tmp_full_tangent_clean_plus_20260406.json`
  - `data/paper_suite/tmp_full_tangent_context_20260406.json`
  - `data/paper_suite/tmp_full_tangent_kernel_gaussian_20260406.json`

### 83. Full-Tangent Protected-Core First Attempt Failed In The Opposite Direction

Date:
- 2026-04-06

Outcome:

- this first full-tangent implementation was a dead end and was removed

Hard-case result:

- `logical_clean_plus`
  - unchanged from baseline
  - recall `1.0`
  - precision `0.8571`
- `logical_context`
  - unchanged from baseline
  - recall `1.0`
  - precision `0.875`
- `kernel_gaussian`
  - unchanged from baseline
  - recall `1.0`
  - precision `0.75`

What happened:

- in all three hard cases:
  - protected core size = active support size
  - ambiguity size = `0`
  - family size = `1`
- so the final selector did nothing at all

Why this happened:

- the implemented protection test was:
  - "a rule is protected if its full tangent block keeps full rank after
    projection against the span of all other rule blocks"
- this criterion is far too strong in the wrong direction:
  - kernel-height tangent directions are numerous
  - and generically linearly independent
- so almost every rule automatically looks indispensable
- including the known false extras

This means:

- full tangent information is still the right kind of object
- but **raw residual rank is not the right certificate**

What the certificate must measure instead:

- not whether a rule has any formally independent tangent direction
- but whether it has a direction that matters at the relevant statistical
  scale for support selection

So the next protected-core certificate cannot be:

- residual rank of the tangent block

It must be something like:

- residualized block information magnitude
- or a Schur-complement energy / smallest nonzero singular value
- compared against an explicit confidence radius

In other words:

- linear independence is too cheap
- we need **quantitative** rather than merely **algebraic** indispensability

Updated lesson:

- `source-overlap`:
  - too weak
- `coefficient-score interaction`:
  - better, but still drops true higher-order inhibition
- `full tangent residual rank`:
  - too strong, protects everything

Therefore the next step should be:

- keep the full tangent block
- replace the binary rank test by a quantitative residual information test
  with a confidence threshold

### 67. Global Full-DF Closure And Anchor-Fixed Exact Closure Both Failed; The Next Direction Should Be Local Ambiguity-Block Witness Selection

#### 67.1. What The Two New Exact Selectors Actually Did

I tested two exact final-selection upgrades beyond the current theorem-friendly
search.

Experiment A:
- global deletion-family exact selector
- support-specific joint kernel refit on every deletion candidate
- full kernel-dimension penalty

Experiment B:
- global deletion-family exact selector
- anchor kernel fixed
- coefficient-only exact refit on every deletion candidate

These two directions were both mathematically clean, but they failed in
different ways.

#### 67.2. Why The Global Full-DF Joint-Kernel Closure Failed

Empirical behavior:

- it removed extras aggressively
- but it also deleted several true higher-order inhibition rules
- recall dropped below `1.0` on many benchmarks
- runtime was also too large because the selector evaluated the whole deletion
  family and performed a fresh joint kernel refit on every support

Representative numbers from the partial run:

- `logical_context`:
  - family size: `1024`
  - elapsed: about `4388 s`
  - precision improved, but this global criterion was too aggressive overall
- `kernel_gaussian`:
  - family size: `512`
  - elapsed: about `3865 s`
- even benchmarks with only `64` candidates were still nontrivial because each
  candidate required a fresh kernel optimization

Conclusion:

- allowing global support-specific kernel refits inside the final selector is
  too expressive
- once the selector is allowed to globally reattribute kernel shape and then
  charge the full dimension penalty, it can prefer underspecified supports and
  sacrifice recall

So this direction is both:

- too slow
- too recall-destructive

#### 67.3. Why The Anchor-Fixed Exact Closure Also Failed

I then tested a lighter exact selector:

- same deletion-family exact closure
- but with anchor kernels fixed
- and only coefficient refits allowed for each candidate support

This was cleaner computationally and did fix the one-pass pruning failure:

- `logical_context` became `recall = 1.0`, `precision = 1.0`

But it did **not** fix the criterion-mismatch failures:

- `logical_clean_plus` stayed at:
  - `recall = 1.0`
  - `precision = 0.8571`
  - extra rule:
    - `L -> T : excitation`
- `kernel_gaussian` stayed at:
  - `recall = 1.0`
  - `precision = 0.75`
  - extra rules:
    - `B and C and D -> T : inhibition`
    - `G and H -> T : inhibition`

Conclusion:

- exact fixed-point deletion is necessary
- but if the final selector uses only a globally frozen kernel surrogate, then
  it is still too weak to remove the hard extras

So the two negative lessons are now sharp:

1. global full joint-kernel final selection is too strong and harms recall
2. global frozen-kernel final selection is too weak and cannot recover
   precision on the hard cases

#### 67.4. The Structural Reason Both Extremes Fail

The hard extras in `logical_clean_plus` and `kernel_gaussian` are not global
support errors. They are **local ambiguity errors**.

That means:

- each extra rule competes only with a small set of nearby rules
- the ambiguity is caused by overlapping source sets, sign conflicts, or
  surrogate kernel attribution inside that local neighborhood
- unrelated distant true rules should not be involved in the final decision

Therefore the correct selector should have:

- more flexibility than a fully frozen global surrogate
- less flexibility than a global support-specific joint-kernel selector

This suggests the final selector should be **local in model flexibility**.

#### 67.5. New Direction: Certified Local Ambiguity-Block Witness Selection

The next direction should be:

- keep the current theorem-friendly high-recall search that builds a certified
  superset `S+`
- replace the global final selector by an exact **blockwise witness selector**

Core idea:

1. build an ambiguity graph on the active support `S+`
2. connect two rules if they are logically capable of competing:
   - their source sets overlap
   - one source set contains the other
   - or they reuse a source with opposite sign
3. take connected components `B_1, ..., B_m`
4. for each block `B_j`, keep all rules outside `B_j` fixed
5. inside `B_j`, run exact local model comparison
6. allow local coefficient refits and, if needed, local kernel refits only
   **inside that block**
7. delete a rule only if there exists a confidence-set-admissible local
   alternative that certifiably improves penalized risk
8. iterate blockwise deletions to a true fixed point

This is not heuristic if:

- the ambiguity graph is defined deterministically
- each block family is finite and explicitly enumerated
- each block candidate is exactly refit/evaluated
- deletion happens only under a valid confidence-set certificate
- the algorithm stops only at a true fixed point

#### 67.6. Why This New Direction Can Potentially Get `100/100` And Stay Fast

This direction attacks all three requirements at once.

Requirement A: `recall = 1.0`

- unlike the global full-DF selector, the block selector cannot remove a true
  rule because of a penalty tradeoff involving distant unrelated rules
- a true rule is challenged only by alternatives inside its own ambiguity block
- that is exactly the right locality for preserving recall

Requirement B: `precision = 1.0`

- unlike the fully frozen selector, the block selector is allowed to locally
  reattribute mass/kernel shape inside the ambiguous neighborhood
- this is exactly the extra flexibility needed for `logical_clean_plus` and
  `kernel_gaussian`

Requirement C: speed

- global exact closure has complexity on the order of `2^{|S+|}`
- blockwise exact closure has complexity on the order of
  `sum_j 2^{|B_j|}`
- if ambiguity blocks stay small, this is exponentially faster

This is the first direction so far that is simultaneously:

- theorem-compatible
- plausibly exact-recovery capable
- and computationally realistic

#### 67.7. The Theorem Shape

Exact recovery still cannot be claimed without assumptions. The honest theorem
shape is:

Assumptions:

- Coverage:
  - the true support `S*` belongs to the certified family generated by the
    search stage
- Block separability:
  - population risk decomposes across ambiguity blocks up to a small remainder
    `eps_sep`
- False-rule witness condition:
  - for every extra rule `e not in S*`, there exists a local block alternative
    that drops `e` and improves the penalized population risk by at least
    `gamma_del > 0`
- True-rule stability:
  - for every true rule `t in S*`, every local block alternative that drops `t`
    increases penalized population risk by at least `gamma_keep > 0`
- Uniform estimation control:
  - empirical blockwise risk errors and confidence radii are smaller than
    `0.5 * min(gamma_del, gamma_keep) - eps_sep`

Then:

- the exact blockwise witness selector returns the true support `S*`
- hence both `recall = 1.0` and `precision = 1.0`

This gives a much cleaner theorem target than either failed global selector.

#### 67.8. What Should Be Implemented Next

The next implementation should **not** be another global family experiment.

It should be:

1. exact fixed-point deletion retained
2. ambiguity graph construction added
3. local block family enumeration added
4. local blockwise exact refit/evaluation added
5. full kernel refit permitted only inside the active ambiguity block
6. outside-block rules and kernels held fixed

The first diagnostic to run is not the whole suite. It is:

- compute ambiguity blocks for
  - `logical_clean_plus`
  - `logical_context`
  - `kernel_gaussian`
- inspect whether the hard extras lie in small blocks

If the hard extras sit inside small blocks, this direction has both:

- the right theorem structure
- and the right speed structure

### 68. Full-Suite Local Ambiguity-Block Witness Selection Finished Fast, But It Collapsed Recall

I ran the local ambiguity-block witness selector on the full 11-benchmark suite
using 4-GPU shard parallelism.

Execution summary:

- all 4 shards finished successfully
- merged result file:
  - `data/paper_suite/tmp_local_ambiguity_fullsuite_parallel_20260405.json`
- launcher reported no shard failures

This was substantially faster operationally than the previous global exact
selectors because the suite was split across GPUs and benchmark shards.
But the statistical outcome was bad.

#### 68.1. Full-Suite Outcome

Aggregate:

- mean recall: `0.7944`
- mean precision: `1.0`
- exact (`recall = 1`, `precision = 1`): `2 / 11`

Per benchmark:

- `logical_context`:
  - recall: `0.7143`
  - precision: `1.0`
  - missing:
    - `A and C and D -> T : inhibition`
    - `E and F and G -> T : inhibition`
- `ablation_excitation_only`:
  - recall: `1.0`
  - precision: `1.0`
- `ablation_inhibition_only`:
  - recall: `0.1667`
  - precision: `1.0`
  - missing five true inhibition rules
- `kernel_exponential`:
  - recall: `0.8333`
  - precision: `1.0`
  - missing:
    - `D and F and G -> T : inhibition`
- `kernel_gaussian`:
  - recall: `1.0`
  - precision: `1.0`
- `ablation_mixed_sign`:
  - recall: `0.8333`
  - precision: `1.0`
  - missing:
    - `D and F and G -> T : inhibition`
- `logical_clean_plus`:
  - recall: `0.8333`
  - precision: `1.0`
  - missing:
    - `J and K and L -> T : inhibition`
- `logical_shared`:
  - recall: `0.8571`
  - precision: `1.0`
  - missing:
    - `B and E and F -> T : inhibition`
- `num_predicates_10`:
  - recall: `0.8333`
  - precision: `1.0`
  - missing:
    - `D and F and G -> T : inhibition`
- `kernel_triangular`:
  - recall: `0.8333`
  - precision: `1.0`
  - missing:
    - `D and F and G -> T : inhibition`
- `num_predicates_20`:
  - recall: `0.8333`
  - precision: `1.0`
  - missing:
    - `D and F and G -> T : inhibition`

#### 68.2. What This Means

This direction did not solve the main problem. It simply flipped the failure
mode.

Before:

- recall was often `1.0`
- precision failed because a few extras survived

Now:

- precision became `1.0`
- but the selector deleted true rules aggressively
- especially higher-order inhibition rules

The repeated failure pattern was striking:

- `D and F and G -> T : inhibition` disappeared across several unrelated
  benchmarks
- higher-order inhibition rules were systematically more vulnerable than the
  corresponding excitation rules

So the local ambiguity-block witness selector in its current form is still too
aggressive against true inhibition structure.

#### 68.3. Why This Version Failed

The problem was not runtime anymore. The problem was the criterion.

Even though refits were local rather than global:

- the selector still used a deletion-favoring local comparison
- local full-dimension credit remained large enough to prefer smaller block
  supports
- true higher-order inhibition rules often lost because the local block
  criterion still over-rewarded simplification

So this experiment shows:

- locality alone is not sufficient
- if the local witness rule is still formulated as a one-sided deletion test
  with strong dimension credit, it can destroy recall

#### 68.4. Updated Negative Conclusion

The correct final selector cannot be based on:

- global frozen deletion only
- global joint-kernel deletion only
- local ambiguity-block deletion only

All three have now failed in different ways.

What is missing is a selector that can both:

- delete false extras
- and explicitly protect true higher-order rules from one-sided deletion bias

#### 68.5. Updated Direction

The next direction should no longer be deletion-first.

Instead, the selector should be based on **certified equivalence classes /
confidence sets**, not forced support minimization.

Honest theorem target:

1. produce a certified confidence family containing the true support
2. identify an equivalence class of statistically indistinguishable local
   representations
3. report either:
   - the whole certified family, or
   - a representative selected by a criterion that is provably
     recall-safe under an explicit stability assumption

If single-support exact recovery is still desired, it likely needs an
additional theorem ingredient stronger than deletion witness logic, such as:

- a true-rule stability certificate
- or a blockwise irreducibility condition that protects each true rule against
  deletion inside its local equivalence class

So the new research direction is not:

- stronger deletion

It is:

- certified ambiguity representation first
- exact single-support recovery only under an extra stability theorem

### 69. Return To The `safe-swap confset` Baseline; The Next Credible Direction Is A Protected-Core Plus Tiny-Ambiguity Selector

After the failed global and local deletion-style selectors, the only
theorem-friendly line that still deserves to be the baseline is:

- `safe inactive superset + safe active interaction block + restricted local
  swap + confidence-set-restricted selection`

Its empirical status remains the strongest among theorem-friendly methods:

- `11 / 11` completed
- full recall `= 1.0` on all benchmarks
- exact `= 8 / 11`
- only 3 precision failures:
  - `logical_clean_plus`
  - `logical_context`
  - `kernel_gaussian`

This means the next method should not replace that line globally. It should be
a **small final-stage upgrade** on top of that baseline.

#### 69.1. What Must Be Preserved

The new method must preserve three things from the baseline.

1. high-recall certified search
   - the current safe-swap search is already doing the hard part:
     finding the true support family without missing true rules

2. no global deletion pressure
   - all failed follow-ups became too eager to simplify the model and destroyed
     higher-order inhibition rules

3. small final-stage workload
   - the next selector cannot evaluate a global `2^{|S|}` family or run fresh
     kernel optimization over large support families

So the correct next target is:

- **do almost nothing to the current search**
- modify only the tiny part responsible for the remaining precision failures

#### 69.2. The Right Structural Picture

The current failures suggest the active support after the safe-swap search
contains two qualitatively different subsets:

- a large **protected core** of rules that are clearly real
- a very small **ambiguity set** of rules that are statistically entangled

The evidence is empirical:

- most benchmarks are already exact
- even the failing benchmarks have only one or two extra rules
- the global support is not wrong; only a tiny residual ambiguity remains

Therefore the next selector should not search over all subsets of the active
support. It should search only over the ambiguity set.

#### 69.3. Proposed Next Method: Protected-Core Certified Ambiguity Enumeration

The method should be:

1. run the current `safe-swap confset` baseline and obtain final active support
   `S_hat`
2. classify each rule in `S_hat` as either:
   - protected core
   - ambiguous
3. freeze the protected core
4. enumerate exact candidates only on the ambiguous subset `A`
5. allow richer support-specific comparison only inside `A`
6. choose among those candidates using an exact confidence-set rule

The key is that `|A|` should be tiny.

If this succeeds, complexity changes from:

- bad:
  - `2^{|S_hat|}`

to:

- good:
  - `2^{|A|}`

with `|A| << |S_hat|`.

#### 69.4. How To Define The Protected Core Non-Heuristically

The protected core cannot be defined by an ad hoc score threshold.
It has to be certificate-based.

A rule `r in S_hat` should be declared protected if every admissible local
deletion candidate that removes `r` fails one of the following:

- confidence admissibility
- risk improvement
- kernel-stability compatibility

Equivalent formulation:

- `r` is protected if no certified alternative in the allowed family can remove
  it without violating the confidence/risk conditions

This is mathematically clean because protection is defined by a finite exact
test over an explicit family, not by a heuristic cutoff.

#### 69.5. How To Keep Precision Improvement Without Killing Recall

The new selector should not ask:

- “which rules can I delete?”

It should ask:

- “which non-protected rules remain unresolved inside the certified ambiguity
  family?”

That changes the logic completely.

Old failed logic:

- deletion-first
- smaller model receives strong dimension credit
- true higher-order inhibition rules get deleted

New desired logic:

- protect everything that has no certified deletion witness against it
- only compare models inside the tiny unresolved ambiguity family
- choose a representative only within that ambiguity family

This keeps recall safe by construction for the protected core.

#### 69.6. Where The Extra Flexibility Should Live

The global frozen selector was too rigid.
The global joint-kernel selector was too flexible.

The right compromise is:

- keep the protected core fully frozen
- allow support-specific coefficient refit on the ambiguity subset
- allow support-specific kernel refit only for the ambiguous rules, not for the
  whole model

This should be enough to remove `logical_clean_plus` and `kernel_gaussian`
type extras without letting the entire model reconfigure itself.

So the flexibility is:

- not global
- not local-by-deletion
- but **ambiguity-restricted**

#### 69.7. Theorem Shape

This gives a plausible exact-recovery theorem with clearly separated pieces.

Assumptions:

- Search coverage:
  - the safe-swap baseline outputs a support containing the true support
- Core protection:
  - every true rule outside the ambiguity set has no admissible deletion
    witness in the certified family
- Ambiguity localization:
  - all false extras lie inside a small ambiguity subset `A`
- Local ambiguity margin:
  - inside `A`, the true restricted support beats every false restricted
    support by a margin larger than the empirical estimation error
- Restricted kernel compatibility:
  - support-specific kernel perturbations needed for the true restricted model
    live inside the allowed ambiguity-restricted kernel family

Then:

- every true rule in the protected core is retained
- the ambiguity selector recovers the correct restricted support on `A`
- hence the full recovered support is exact

This is much more realistic than trying to prove exact recovery from a single
global deletion criterion.

#### 69.8. Why This Is Also The Fastest Credible Direction

This direction is likely the fastest path that still has a theorem story.

Reason:

- search stage stays unchanged
- protected-core testing can be cached and reused
- exact enumeration is only over a tiny ambiguity family
- any kernel refit is ambiguity-restricted rather than global

So the computational target is no longer:

- expensive global exact selection

It is:

- cheap global search
- tiny exact final ambiguity resolution

#### 69.9. Immediate Next Diagnostic

Before implementing another selector, the next diagnostic should be:

1. run the current `safe-swap confset` baseline
2. identify which final rules are non-protected under exact certified deletion
   checks
3. measure the size of the residual ambiguity subset on the 3 hard cases

If the ambiguity subset sizes are, say, `1` to `3`, then this direction is
both:

- statistically credible
- and computationally feasible

### 64. Standalone Server-Migration Handoff Document Created

Because the server became unstable, a standalone handoff document was created:

- `train/research_docs/handoff_rule_dependent_kernel_research_2026-04-03.md`

That document is intended to be sufficient on its own and includes:

- current best-performing method overall:
  - **frozen-kernel exact add-only correction**
- the theorem-friendly line status
- the latest confidence-set selection result
- the `logical_clean_plus` audit
- the recommended theorem framing
- the exact last stopped task before migration

Last stopped task recorded in the handoff:

- the next requested run was to execute the current theorem-friendly method on
  the benchmark suite
- but with predicate variation changed so that we use `10/20` rather than
  `20/30`, because `num_predicates_30` was too slow
- before any edit or run, it was confirmed that the current suite definitions
  in:
  - `run_paper_benchmarks.py`
  - `tmp_run_exact_correction_benchmarks.py`
  - `tmp_run_frozen_block_exact_solver_benchmarks.py`
  still contained `num_predicates_30`
- no new full-suite run had started after that request

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 65. The Current "True Recovery Under Margin" Statement Is Too Weak To Be The Main Contribution

Important correction after reflecting on the theorem target:

The previously sketched theorem shape
- truth in candidate family
- exact support-fixed refits
- uniform concentration over the family
- positive population margin
- surrogate alignment

is **too assumption-heavy** to serve as the main scientific contribution.

Why it is too weak as a main theorem:
- `truth in family` already assumes away the hard candidate-generation problem
- `positive population margin` is a standard identifiability assumption
- `surrogate alignment` essentially assumes that the frozen-kernel criterion
  already prefers the truth
- taken together, these assumptions almost collapse the theorem into
  "if the optimization target already favors the truth and the truth is in the
  searched family, then the method recovers the truth"

That statement is still valid as a corollary, but it is not strong enough to
carry the paper.

What should be the real contribution instead:

1. **Certified family construction / no-materially-better-support exclusion**
- prove that the safe block construction does not exclude any support whose
  improvement over the incumbent exceeds an explicit threshold
- this is algorithmic, data-dependent, and not merely an existential
  assumption

2. **Finite-sample oracle guarantee over the certified family**
- main theorem should be an oracle inequality or confidence-set selection
  guarantee relative to the best support in the constructed family
- this avoids assuming the truth is already the optimizer of the surrogate

3. **Exact recovery only as a corollary**
- if the truth lies in the certified family and enjoys a sufficient margin, then
  the oracle theorem implies exact recovery with high probability

This is a much stronger framing because:
- the main theorem is about what the algorithm itself guarantees
- the recovery result becomes a standard consequence under additional
  identifiability assumptions
- the paper contribution then lives in:
  - the certified family reduction
  - the exact local search inside that family
  - the confidence-aware familywise selection rule

Recommended theorem stack:
- Theorem A:
  - safe/certified family construction
- Theorem B:
  - finite-sample oracle inequality for the final selected model over that
    family
- Corollary C:
  - exact true-rule recovery under margin and family-containment assumptions

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 64. What Can Actually Be Proved About True-Rule Recovery In The Current Structure

If the final scientific goal is:
- "recover the true rule set with a mathematical confidence bound"

then the current structure is **not** yet enough for an unconditional theorem,
but it is already close enough for a **conditional high-probability recovery
theorem**.

Most realistic theorem shape in the current structure:
- use the present pipeline only as:
  - exact-search generator of a small ambiguity family
  - plus confidence-set-restricted model selection on that family
- then prove recovery of the true rule set **inside that family** under a small
  number of explicit assumptions

Minimal assumption set that looks sufficient:

1. **Family containment**
- the true support `S*` belongs to the final candidate family `F`
- this is the most important assumption
- without it, no familywise confidence bound can recover the truth

2. **Support-fixed exactness**
- for every candidate support `U in F`, the refit stage returns the global
  optimum of the frozen-kernel support-fixed objective
- in the current code this is already close to true, because the support-fixed
  excitation/inhibition subproblems are convex

3. **Uniform concentration on the validation criterion**
- the empirical validation loss uniformly concentrates over the finite family
  `F`
- because `F` is small and fixed before the final confidence stage, a union
  bound / empirical-Bernstein argument is feasible

4. **Population margin / identifiability**
- the true support is separated from every other support in `F`
- e.g.
  - for all `U != S*` in `F`,
    `R(U) - R(S*) >= gamma > 0`
- this is the usual recovery condition
- without a positive margin, exact recovery cannot be guaranteed

5. **Surrogate alignment**
- the frozen-kernel population criterion ranks `S*` above the competing family
  members
- this is exactly the assumption that fails on the sticky `logical_clean_plus`
  case
- it is weaker than "the whole model is correctly specified", but stronger
  than mere search correctness

What this means in plain terms:
- with assumptions `1–4`, we can prove recovery **relative to the frozen-kernel
  target family**
- to prove recovery of the actual true rules, we also need `5`
- so the cleanest current theorem is:
  - "if the truth is in the candidate family and the frozen-kernel population
    criterion has a positive margin in favor of the truth, then the
    confidence-set selector recovers the true support with probability
    at least `1-delta`"

So how many assumptions are really needed?
- the answer is roughly **4 core assumptions + 1 alignment assumption**
- equivalently:
  - `4` if we define the target as the frozen-kernel population optimum
  - `5` if we want to claim recovery of the actual true rule set

Why this is scientifically useful:
- it is already strong enough for a theorem section
- it separates:
  - search failure
  - statistical selection failure
  - surrogate-model mismatch
- and it explains exactly why `logical_shared` can now be fixed while
  `logical_clean_plus` still resists recovery

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 63. Historical Audit: `logical_clean_plus` Has Not Been 100/100 In The Current Saved Experimental Line

After the latest confusion around `logical_clean_plus`, we audited the saved
result files in `data/paper_suite`.

Observed pattern across the current saved runs:
- `swap_screen_logical_clean_plus.json`
  - recall `1.0`, precision `0.857`
  - extra `L -> T : excitation`
- `tmp_exact_logical_clean_plus.json`
  - recall `1.0`, precision `0.857`
  - extra `L -> T : excitation`
- `tmp_frozen_block_batch1.json`
  - recall `1.0`, precision `0.857`
  - extra `L -> T : excitation`
- `tmp_safe_swap_suite_report.json`
  - recall `1.0`, precision `0.857`
  - extra `L -> T : excitation`
- `tmp_safe_swap_plus_interaction_logical_pair.json`
  - recall `1.0`, precision `0.857`
  - extra `L -> T : excitation`
- `tmp_safe_swap_conf_prune_screened_logical_pair.json`
  - recall `1.0`, precision `0.857`
  - extra `L -> T : excitation`
- `tmp_safe_swap_budget_subset_logical_pair.json`
  - recall `1.0`, precision `0.857`
  - extra `L -> T : excitation`
- `tmp_safe_swap_confset_select_logical_pair.json`
  - recall `1.0`, precision `0.857`
  - extra `L -> T : excitation`
- only the old aggressive confidence-prune removed the extra, but it also
  dropped the true rule `J and K and L -> T : inhibition`

So, within the currently saved experiment line:
- `logical_clean_plus` has been consistently **full-recall but one-extra**
- not a repeatedly solved `100/100` case

What this means scientifically:
- the issue is not that a previously easy exact solution suddenly regressed
- the issue is that the present frozen-kernel surrogate has a persistent local
  ambiguity:
  - true rule: `J and K and L -> T : inhibition`
  - surrogate extra: `L -> T : excitation`
- this ambiguity is stable across many different exact-search variants

Why this particular extra is sticky:
- `L` appears inside the true inhibitory triplet
- under the frozen-kernel objective, a singleton positive effect on `L` remains
  slightly useful even after the true triplet inhibition is present
- therefore search is not the bottleneck here:
  - the exact-search path already finds all true rules
  - the final objective keeps assigning a small but nonzero advantage to the
    extra singleton

Evidence from the latest confidence-set model-selection run:
- the extra-only candidate on `logical_clean_plus` was inside the confidence
  set
- but its split-A penalized score was still positive:
  - total loss increase `7.14`
  - penalty credit `5.58`
  - final score `+1.56`
- so under the current final criterion, the extra is still preferred to stay

Bottom line:
- `logical_clean_plus` is not currently failing because it became hard to
  search
- it is failing because the present frozen-kernel final objective still
  prefers keeping `L -> T : excitation`

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 62. Confidence-Set-Restricted Model Selection Finally Removed The Extra On `logical_shared`, But Not On `logical_clean_plus`

We then implemented the more faithful version of the proposed
confidence-set-based final selection:

- split A:
  - rank active-rule ambiguity candidates
  - build a fixed top-`k` ambiguity family (`k = 2`)
  - use the exact-refit candidate's empirical penalized criterion on split A
    for model selection
- split B:
  - use exact-refit lower confidence bounds to define the confidence set
  - only candidates not significantly worse than the current support are
    eligible
- final selection:
  - among confidence-set members, choose the candidate with the best
    split-A penalized criterion

This is different from the previous BIC-budgeted prune:
- before:
  - exact `UCB <= budget` was required for deletion
- now:
  - exact split-B confidence set only constrains admissible candidates
  - split-A penalized empirical criterion performs the final ranking inside
    that admissible set

Actual run:
- benchmarks:
  - `logical_clean_plus`
  - `logical_shared`
- resources:
  - all available machine resources
  - `24` CPU threads
  - `support_workers = 24`
  - `device = auto`
- output:
  - `tmp_safe_swap_confset_select_logical_pair.json`

Observed result on `logical_clean_plus`:
- elapsed `862.96s`
- recall `1.0`
- precision `0.857`
- extra remained:
  - `L -> T : excitation`

Why `logical_clean_plus` still did not prune:
- the extra-only candidate was inside the confidence set
- but its split-A penalized selection score was still positive:
  - exact total diff on split A: `7.14`
  - selection penalty: `5.58`
  - final selection score: `+1.56`
- so even inside the confidence set, the smaller support was not preferred

Observed result on `logical_shared`:
- elapsed `337.51s`
- recall `1.0`
- precision `1.0`
- previous extra removed:
  - `A and C and D -> T : inhibition`

Why `logical_shared` did prune successfully:
- the extra-only candidate was inside the confidence set
- its split-A penalized selection score was negative:
  - exact total diff on split A: `5.21`
  - selection penalty: `5.60`
  - final selection score: `-0.39`
- the larger two-drop subset was not even in the confidence set
  - its split-B lower confidence bound was positive

Comparison against earlier variants:
- `logical_clean_plus`
  - 3-lever exact-search path:
    - `1015.95s`
    - recall `1.0`, precision `0.857`
  - screened one-shot UCB prune:
    - `960.67s`
    - recall `1.0`, precision `0.857`
  - confidence-set model selection:
    - `862.96s`
    - recall `1.0`, precision `0.857`
- `logical_shared`
  - 3-lever exact-search path:
    - `492.54s`
    - recall `1.0`, precision `0.875`
  - aggressive old confidence-prune:
    - `645.49s`
    - recall `1.0`, precision `1.0`
  - BIC-budgeted subset prune:
    - `336.07s`
    - recall `1.0`, precision `0.875`
  - confidence-set model selection:
    - `337.51s`
    - recall `1.0`, precision `1.0`

Interpretation:
- this is the first mathematically structured final-selection rule in the
  current line that:
  - keeps recall
  - removes the logical extra on `logical_shared`
  - and does so without the earlier aggressive over-pruning
- but it still fails on `logical_clean_plus`
  - there, even the exact-refit extra-only candidate is still too expensive on
    the selection split under the current frozen-kernel surrogate

Meaning:
- the remaining failure is now very specific
  - not search
  - not runtime
  - not generic confidence-prune instability
- it is a remaining criterion mismatch on `logical_clean_plus`

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 61. BIC-Budgeted Ambiguity-Subset Prune Preserved Recall And Improved Speed, But Still Did Not Remove The Logical Extras

We implemented the more rigorous variant discussed after the confidence-prune
analysis:
- stage A:
  - rank active rules by frozen-drop validation UCB on split A
- stage B:
  - take the top-`k` ambiguous rules (`k = 2`)
  - enumerate all non-empty subsets of that fixed ambiguity set
  - exact-refit each subset candidate
  - accept a smaller support only if its exact one-sided UCB on split B is
    below a BIC-derived budget
    - `budget = |drop_set| * log(n_eff) / (2 n_eff)`

This is stricter than the earlier aggressive prune:
- fixed candidate family before the exact stage
- simultaneous exact-stage testing over that finite family
- nested-support budget derived directly from the BIC penalty scale

Actual run:
- benchmarks:
  - `logical_clean_plus`
  - `logical_shared`
- resources:
  - all available machine resources
  - `24` CPU threads
  - `support_workers = 24`
  - `device = auto`
- output:
  - `tmp_safe_swap_budget_subset_logical_pair.json`

Observed result on `logical_clean_plus`:
- elapsed `863.52s`
- recall `1.0`
- precision `0.857`
- extra remained:
  - `L -> T : excitation`

Observed exact-stage candidates on `logical_clean_plus`:
- drop extra only:
  - exact UCB `4.02e-4`
  - budget `8.00e-5`
- drop true inhibition only:
  - exact UCB `2.48e-3`
  - budget `8.00e-5`
- drop both:
  - exact UCB `2.54e-3`
  - budget `1.60e-4`

Observed result on `logical_shared`:
- elapsed `336.07s`
- recall `1.0`
- precision `0.875`
- extra remained:
  - `A and C and D -> T : inhibition`

Observed exact-stage candidates on `logical_shared`:
- drop extra only:
  - exact UCB `3.54e-4`
  - budget `7.72e-5`
- second inhibition candidate only:
  - exact UCB `2.45e-3`
  - budget `7.72e-5`
- drop both:
  - exact UCB `2.57e-3`
  - budget `1.54e-4`

Comparison against previous logical runs:
- `logical_clean_plus`
  - 3-lever exact-search path:
    - `1015.95s`
    - recall `1.0`, precision `0.857`
  - screened one-shot UCB prune:
    - `960.67s`
    - recall `1.0`, precision `0.857`
  - new BIC-budgeted subset prune:
    - `863.52s`
    - recall `1.0`, precision `0.857`
- `logical_shared`
  - 3-lever exact-search path:
    - `492.54s`
    - recall `1.0`, precision `0.875`
  - screened one-shot UCB prune:
    - `440.91s`
    - recall `1.0`, precision `0.875`
  - new BIC-budgeted subset prune:
    - `336.07s`
    - recall `1.0`, precision `0.875`

Interpretation:
- this variant does what the stricter theory predicts
  - it does not over-prune
  - it keeps recall on both logical benchmarks
  - it is faster than the earlier logical confidence-prune variants
- but the BIC-derived average-risk budget is much smaller than the exact UCB of
  the observed extras
- so the method is still too conservative to actually remove those extras

Meaning:
- the remaining precision problem is no longer a runtime issue
- it is now a **criterion mismatch** issue:
  - under the present frozen-kernel surrogate and strict BIC-derived budget,
    the logical extras are not statistically cheap enough to prune
- if we want a mathematically rigorous prune that actually removes them, the
  next step is not "more search" but a different justified budget or a
  different final objective

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 60. One-Shot Screened UCB Prune Was Faster And Safe On Recall, But Too Conservative To Remove The Extras

We replaced the earlier aggressive multi-round confidence prune with a
screened one-shot version:
- stage A:
  - evaluate all active rules by a cheap frozen-drop validation UCB
  - keep only the single best candidate (`top_k = 1`)
- stage B:
  - run one exact leave-one-out refit only for that selected candidate
  - drop it only if the one-sided exact UCB satisfies `UCB <= 0`

This removes the earlier wrong monotonicity and cuts the exact-prune cost
substantially.

Actual run:
- benchmarks:
  - `logical_clean_plus`
  - `logical_shared`
- resources:
  - CPU only
  - `12` threads / `support_workers = 12`
- output:
  - `tmp_safe_swap_conf_prune_screened_logical_pair.json`

Observed result on `logical_clean_plus`:
- elapsed `960.67s`
- recall `1.0`
- precision `0.857`
- extra remained:
  - `L -> T : excitation`

Observed prune behavior on `logical_clean_plus`:
- screen stage correctly ranked the excitation extra first
  - frozen-drop UCB `6.60e-4`
- exact stage tested only that one candidate
  - exact mean diff `7.92e-5`
  - exact radius `2.66e-4`
  - exact UCB `3.45e-4`
- since `UCB > 0`, the rule was not dropped

Observed result on `logical_shared`:
- elapsed `440.91s`
- recall `1.0`
- precision `0.875`
- extra remained:
  - `A and C and D -> T : inhibition`

Observed prune behavior on `logical_shared`:
- screen stage correctly ranked the inhibition extra first
  - frozen-drop UCB `3.57e-4`
- exact stage tested only that one candidate
  - exact mean diff `8.15e-5`
  - exact radius `2.12e-4`
  - exact UCB `2.93e-4`
- again `UCB > 0`, so the rule was not dropped

Comparison against previous variants:
- `logical_clean_plus`
  - 3-lever exact-search path:
    - `1015.95s`
    - recall `1.0`, precision `0.857`
  - aggressive old confidence-prune:
    - `1112.71s`
    - recall `0.833`, precision `1.0`
  - new screened one-shot UCB prune:
    - `960.67s`
    - recall `1.0`, precision `0.857`
- `logical_shared`
  - 3-lever exact-search path:
    - `492.54s`
    - recall `1.0`, precision `0.875`
  - aggressive old confidence-prune:
    - `645.49s`
    - recall `1.0`, precision `1.0`
  - new screened one-shot UCB prune:
    - `440.91s`
    - recall `1.0`, precision `0.875`

Interpretation:
- the new screened UCB prune does exactly what it should mathematically:
  - it no longer over-prunes
  - it preserves recall on both logical benchmarks
  - it is much faster than the earlier exact-all-rules prune
- but with strict `epsilon = 0`, it is too conservative to actually remove the
  weak extra rules on these two datasets

Meaning:
- the speed problem of confidence pruning is largely solved by screening
- the remaining issue is now the statistical decision threshold, not runtime
- the next rigorous variant should be an explicit `epsilon`-noninferiority
  prune rather than strict `UCB <= 0`

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 59. Predicted Speedup From A Rigorous Two-Stage Confidence Prune

No new run was made here. This is a runtime forecast based on the completed
`logical_clean_plus` and `logical_shared` confidence-prune runs.

Observed current costs:
- `logical_clean_plus`
  - 3-lever exact-search path without confidence prune:
    - `1015.95s`
  - current confidence-prune path:
    - `1112.71s`
  - observed prune overhead:
    - about `96.76s`
  - tested leave-one-out deletions:
    - `18`
- `logical_shared`
  - 3-lever exact-search path without confidence prune:
    - `492.54s`
  - current confidence-prune path:
    - `645.49s`
  - observed prune overhead:
    - about `152.95s`
  - tested leave-one-out deletions:
    - `15`

Forecast if we switch to:
- cheap fixed-candidate frozen-drop UCB screening
- exact refit only for ambiguous survivors
- same exact search path as before

Expected prune-stage reduction:
- likely from `15–18` exact deletion tests down to about `2–4`
- expected prune-stage speedup:
  - about `3x–6x`

Forecasted total wall time:
- `logical_clean_plus`
  - expected total:
    - roughly `1035s – 1065s`
  - relative to current confidence-prune:
    - about `1.04x – 1.08x` faster
  - relative to the no-prune 3-lever path:
    - still probably slightly slower unless the prune exact solves are also
      accelerated
- `logical_shared`
  - expected total:
    - roughly `515s – 555s`
  - relative to current confidence-prune:
    - about `1.16x – 1.25x` faster
  - still likely a bit slower than the no-prune 3-lever path

Forecast if we also add exact Newton / factor reuse inside support solves:
- each exact support solve may plausibly get another `1.5x – 2.5x` speedup on
  these small-support problems
- then the combined total-wall-time forecast becomes:
  - `logical_clean_plus`
    - roughly `820s – 980s`
  - `logical_shared`
    - roughly `390s – 500s`

Main takeaway:
- redesigning prune alone helps, but not dramatically on benchmarks where the
  search phase already dominates total runtime
- the larger win needs:
  - cheap UCB screening to avoid most exact prune refits
  - exact Newton / low-rank reuse to shrink the remaining solve cost

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 58. Exact-Preserving Speedups Should Target The Prune Design, Not Just More Threads

After examining the current confidence-prune implementation, the runtime
problem is structural.

Why the current prune is slow:
- `confidence_calibrated_exact_prune` loops over every active rule and tests
  leave-one-out deletion one by one
- each tested deletion calls `refit_fixed_support_pair`
- `refit_fixed_support_pair(cycles=2)` performs:
  - one excitation solve
  - one inhibition solve
  - two more alternating excitation/inhibition cycles
  - one final excitation solve
- so each tested deletion costs **7 convex support solves**

Implication on the logical runs:
- `logical_clean_plus`
  - prune tested `7 + 6 + 5 = 18` deletions
  - this means about `18 * 7 = 126` support solves inside prune alone
- `logical_shared`
  - prune tested `8 + 7 = 15` deletions
  - this means about `15 * 7 = 105` support solves inside prune alone

Therefore the main speed issue is not lack of CPU threads by itself.
The main issue is:
- too many expensive exact re-solves are being used just to decide whether a
  single rule is dispensable

Rigor-preserving acceleration route:

1. **Use a fixed-candidate confidence screen before exact refits**
- build a fixed candidate family from the already trained support
- for each active rule `j`, define a cheap "frozen-drop" candidate by setting
  only coefficient `j` to zero and leaving all other trained parameters fixed
- these candidates are fixed before holdout evaluation, so a one-sided
  validation-risk UCB test on this family is clean
- cost:
  - only `O(|S| * n_val)` loss evaluations
  - no optimization loop per candidate

2. **Exact-refit only the ambiguous rules**
- after the cheap UCB screen, most rules should fall into:
  - definitely keep
  - definitely droppable within `epsilon`
- only the narrow ambiguous middle set needs expensive exact leave-one-out
  refits
- runtime then changes from roughly
  - `O(|S| * C_fit)`
  to
  - `O(|S| * n_val + |A_amb| * C_fit)`
  where `|A_amb| << |S|`

3. **If full finite-sample validity is required after adaptive screening,
   use a second calibration split**
- split the old validation/holdout part into:
  - calibration A for cheap frozen-drop screening
  - calibration B for final exact-refit prune on the ambiguous survivors
- this avoids adaptive reuse of the same holdout in both stages

4. **Inside each exact support solve, replace L-BFGS-B with exact Hessian
   Newton / Cholesky updates**
- both excitation and inhibition support-fixed problems are convex
- their Hessians are available in closed form
- support sizes are small, so Newton with line search should converge in a few
  steps and preserve the same optimum/KKT target
- for swap/drop neighbors, use superset block matrices and low-rank factor
  updates instead of rebuilding support matrices from scratch every time

Bottom line:
- the right way to speed this path up while keeping mathematical rigor is
  **not** "run the same prune with more hardware"
- it is:
  - cheap fixed-candidate UCB screening first
  - exact refits only for ambiguous survivors
  - exact Newton / factor reuse inside the remaining convex solves

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 57. The First Confidence-Prune Rule Was Statistically The Wrong Monotone Direction

After inspecting the implemented confidence gate, the main issue is now clear.

Implemented rule:
- for rule drop candidate `j`, let
  - `Delta_j = R(S \ {j}) - R(S)` be the population validation-risk increase
    from dropping the rule
  - `hat Delta_j` be its empirical estimate
  - `r_j` be the empirical-Bernstein radius
- the code dropped `j` whenever
  - `hat Delta_j <= r_j`

Why this is not a safe pruning rule:
- if the concentration statement is
  - `|Delta_j - hat Delta_j| <= r_j`
  then `hat Delta_j <= r_j` only implies
  - `Delta_j <= hat Delta_j + r_j <= 2 r_j`
- this does **not** certify `Delta_j <= 0`
- it only says the true loss increase might still be positive, as large as
  roughly `2 r_j`

This creates the wrong monotonicity:
- smaller sample size or smaller `delta` makes `r_j` larger
- under the implemented rule, larger uncertainty makes pruning **easier**
- that is the opposite of what a conservative safe-pruning rule should do

What happened on the two logical benchmarks:
- `logical_clean_plus`
  - true inhibition rule `J and K and L -> T : inhibition`
  - `hat Delta = 8.48e-4`
  - `r = 1.04e-3`
  - implemented rule dropped it because `hat Delta <= r`
  - but the valid upper bound is
    - `UCB = hat Delta + r = 1.89e-3 > 0`
  - so there was **no** rigorous basis to certify the rule as dispensable
- `logical_shared`
  - extra inhibition rule `A and C and D -> T : inhibition`
  - `hat Delta = 7.34e-5`
  - `r = 1.75e-4`
  - `UCB = 2.48e-4`
  - this is much smaller, so this rule really does look much easier to prune

What the mathematically correct safe-prune form should be:
- if the goal is exact `0`-excess-risk safe pruning, drop only when
  - `hat Delta_j + r_j <= 0`
- if the goal is `epsilon`-noninferiority pruning, drop only when
  - `hat Delta_j + r_j <= epsilon`
- this has the correct monotonicity:
  - larger uncertainty makes pruning harder, not easier

Implication for the current pair:
- the current rule was indeed **too aggressive** in the only sense that matters
  for safe pruning
- more precisely:
  - it treated "failure to prove necessity" as if it were
    "proof of dispensability"
- the correct rigorous interpretation is:
  - dropping is justified only by a one-sided upper confidence bound on the
    excess loss of the smaller support

Additional rigor caveat:
- even after fixing the sign of the test, a full theorem still needs
  - correction across adaptive prune rounds, not just within-round Bonferroni
  - a dependence-aware concentration argument for validation losses if the
    holdout contributions are not i.i.d.

Most useful next step:
- replace the current rule with `UCB <= epsilon` pruning
- choose `epsilon` explicitly as the allowable excess validation risk
- then the method gives a clean finite-sample noninferiority statement instead
  of the current overly aggressive elimination rule

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 56. Confidence-Calibrated Exact Prune Fixed `logical_shared` But Broke Recall On `logical_clean_plus`

We implemented a post-selection `confidence-calibrated exact prune` on top of
the current theorem-friendly exact-search path:
- exact block solver with frozen kernel
- excitation safe swap superset
- active interaction block
- scalar warm reuse
- then an exact leave-one-rule-out refit with an empirical-Bernstein
  confidence gate on validation loss difference

Rule-retention test:
- for current support `S` and one-rule-smaller support `S \ {j}`,
  keep rule `j` only if the empirical validation loss increase is larger than
  the confidence radius
- otherwise drop it

Actual run:
- benchmarks:
  - `logical_clean_plus`
  - `logical_shared`
- resources:
  - CPU only
  - `support_workers = 16`
  - unrestricted machine resources were allowed for this run
- output:
  - `tmp_safe_swap_conf_prune_logical_pair.json`

Observed result on `logical_clean_plus`:
- elapsed `1112.71s`
- recall `0.833`
- precision `1.0`
- dropped the true rule `J and K and L -> T : inhibition`
- removed the previous extra `L -> T : excitation`

Comparison on `logical_clean_plus`:
- earlier safe-block baseline:
  - `1147.41s`
  - recall `1.0`, precision `0.857`
- earlier 3-lever speed variant:
  - `1015.95s`
  - recall `1.0`, precision `0.857`
- new confidence-prune variant:
  - `1112.71s`
  - recall `0.833`, precision `1.0`

Observed result on `logical_shared`:
- elapsed `645.49s`
- recall `1.0`
- precision `1.0`
- removed the previous extra `A and C and D -> T : inhibition`

Comparison on `logical_shared`:
- earlier safe-block baseline:
  - `495.66s`
  - recall `1.0`, precision `0.875`
- earlier 3-lever speed variant:
  - `492.54s`
  - recall `1.0`, precision `0.875`
- new confidence-prune variant:
  - `645.49s`
  - recall `1.0`, precision `1.0`

What the prune logs say:
- `logical_shared` behaved as intended
  - exactly one inhibition rule was judged statistically droppable in prune
    round 1
  - after removing it, no more rules passed the confidence gate
- `logical_clean_plus` did not behave as intended
  - prune round 1 judged two rules droppable:
    - one excitation extra
    - one inhibition rule
  - prune round 2 still judged that inhibition rule droppable
  - that inhibition rule was the true `J and K and L -> T : inhibition`

Interpretation:
- the confidence gate is **not yet safe enough** as currently instantiated
- it can improve precision on some ambiguity cases, but on
  `logical_clean_plus` it over-pruned a true inhibitory rule under the frozen
  surrogate objective
- this is therefore **not** ready to become the default final-selection rule
- the current failure mode is consistent with the main concern:
  - the validation-loss confidence test is still tied to the frozen-kernel
    surrogate, so statistical non-separation does not imply structural
    dispensability of the rule

Decision:
- keep this as an experimental branch only
- do not make confidence-pruned selection the default path yet

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 57. Full-Suite Run Of The Provable Safe-Superset Restricted Local-Swap Variant: 11/12 Completed, Hardest Case Still Failed

On `2026-04-01` we ran the theorem-friendly variant discussed above:
- inhibition side uses the safe inactive-superset screen
- neighborhood stays `restricted local swap`
- support-fixed solves remain exact
- fast path is still exact-preserving:
  - all-core CPU execution
  - parallel support solves
  - no approximate pruning

Primary result files:
- `tmp_safe_swap_full_suite.json`
- `tmp_safe_swap_ablation_batch_rerun.json`
- consolidated report: `tmp_safe_swap_suite_report.json`

Resource settings actually used:
- full-suite pass for the first `8` benchmarks:
  - CPU only
  - `OMP_NUM_THREADS=24`
  - `support_workers=16`
- `num_predicates_30` retries:
  - `support_workers=16`, then `8`, then `4`
  - all three runs died before writing a benchmark result
- final ablation batch:
  - CPU only
  - `OMP_NUM_THREADS=24`
  - `support_workers=16`

Completed benchmark results:
- `logical_clean_plus`
  - `1147.4s`
  - recall `1.0`
  - precision `0.857`
  - extra: `L -> T : excitation`
- `logical_shared`
  - `495.7s`
  - recall `1.0`
  - precision `0.875`
  - extra: `A and C and D -> T : inhibition`
- `logical_context`
  - `569.8s`
  - recall `1.0`
  - precision `0.700`
  - extras:
    - `A and B and D -> T : inhibition`
    - `A and E and G -> T : excitation`
    - `E and G -> T : inhibition`
- `kernel_triangular`
  - `419.6s`
  - recall `1.0`
  - precision `1.0`
- `kernel_exponential`
  - `369.5s`
  - recall `1.0`
  - precision `1.0`
- `kernel_gaussian`
  - `1737.8s`
  - recall `1.0`
  - precision `0.667`
  - extras:
    - `A and B -> T : inhibition`
    - `B and C and D -> T : inhibition`
    - `G and H -> T : inhibition`
- `num_predicates_10`
  - `745.3s`
  - recall `1.0`
  - precision `1.0`
- `num_predicates_20`
  - `1890.5s`
  - recall `1.0`
  - precision `1.0`
- `ablation_excitation_only`
  - `996.2s`
  - recall `1.0`
  - precision `1.0`
- `ablation_inhibition_only`
  - `236.9s`
  - recall `1.0`
  - precision `1.0`
- `ablation_mixed_sign`
  - `685.8s`
  - recall `1.0`
  - precision `1.0`

`num_predicates_30` outcome:
- not completed
- the process died before JSON write in all three attempts:
  - `support_workers=16`
  - `support_workers=8`
  - `support_workers=4`
- practical interpretation:
  - the current provable safe-superset restricted local-swap path still has a
    severe memory / runtime tail on the hardest benchmark

Aggregate picture over the completed `11` benchmarks:
- recall stayed `1.0` on all completed runs
- exact `100/100` recovery on `7/11`
  - `kernel_triangular`
  - `kernel_exponential`
  - `num_predicates_10`
  - `num_predicates_20`
  - `ablation_excitation_only`
  - `ablation_inhibition_only`
  - `ablation_mixed_sign`
- average precision over the completed set: `0.918`
- total wall time over the completed set: `9294.5s` (`~154.9 min`)

Main conclusion:
- this theorem-friendly variant is **not** yet a replacement for the current
  best empirical method
- it does show that the safe-superset / restricted-swap idea can recover the
  truth exactly on many benchmarks
- but it is still too expensive, and its precision is clearly worse on
  `logical_clean_plus`, `logical_shared`, `logical_context`, and
  `kernel_gaussian`
- most importantly, it is still not practical on `num_predicates_30`

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**
- reason:
  - it is still the only method verified end-to-end on the full official
    `12/12` suite with recall `1.0`

Updated judgment on the provable direction:
- scientifically useful as a theorem-oriented reference point
- not yet a competitive end-to-end solver
- next bottlenecks are now obvious:
  - excitation-side global swap cost
  - memory growth on `num_predicates_30`
  - precision failure on ambiguity-heavy logical and gaussian cases

### 58. Adding All Three Practical Levers Reduced Some Search Cost, But Did Not Remove The Logical Extras

We then applied the three concrete levers discussed after the full-suite run:
- add an excitation-side safe swap superset
- add an interaction-based active drop block
- reuse scalar screening optima as child warm starts

Implementation summary:
- new excitation exact singleton screening / safe superset
- active interaction block for both excitation and inhibition
- scalar warm reuse for both excitation and inhibition swap children
- all changes stayed exact-preserving at the support-fixed solve level

Target benchmarks:
- `logical_clean_plus`
- `logical_shared`

Run settings:
- CPU only
- `OMP_NUM_THREADS=24`
- `support_workers=16`
- flags:
  - `--inh_safe_swap_superset`
  - `--exc_safe_swap_superset`
  - `--active_interaction_block`
  - `--scalar_warm_reuse`
  - `--inh_block_swap_only`
- output:
  - `tmp_safe_swap_plus_interaction_logical_pair.json`

Baseline comparison source:
- the earlier theorem-friendly safe-superset run from
  `tmp_safe_swap_suite_report.json`

Observed result on `logical_clean_plus`:
- baseline:
  - `1147.4s`
  - recall `1.0`
  - precision `0.857`
  - extra: `L -> T : excitation`
  - round-1 excitation stats:
    - add candidates `203`
    - swap evals `1128`
    - support evals `1336`
- new 3-lever run:
  - `1015.9s`
  - recall `1.0`
  - precision `0.857`
  - extra still `L -> T : excitation`
  - round-1 excitation stats:
    - safe superset candidates `291`
    - interaction drop candidates `4`
    - add candidates `5`
    - swap evals `850`
    - support evals `860`

Observed result on `logical_shared`:
- baseline:
  - `495.7s`
  - recall `1.0`
  - precision `0.875`
  - extra: `A and C and D -> T : inhibition`
  - round-1 excitation stats:
    - add candidates `3`
    - swap evals `95`
    - support evals `103`
- new 3-lever run:
  - `492.5s`
  - recall `1.0`
  - precision `0.875`
  - extra still `A and C and D -> T : inhibition`
  - round-1 excitation stats:
    - safe superset candidates `33`
    - interaction drop candidates `4`
    - add candidates `2`
    - swap evals `79`
    - support evals `86`

Interpretation:
- the excitation safe screen and scalar warm reuse did lower some local search
  cost
- the effect was real on `logical_clean_plus`
  - wall time down by about `11%`
  - excitation support evaluations down from `1336` to `860`
- the effect was much smaller on `logical_shared`
  - wall time changed only marginally
- most importantly:
  - **the extras did not disappear in either benchmark**

Why these three levers were not enough:
- the excitation safe superset was still too loose in `logical_clean_plus`
  - `291` excitation-safe candidates remained
- the interaction block did not materially shrink the active excitation side in
  either case
  - interaction drop candidates stayed `4`
- the surrogate extra rules are still locally admissible under the frozen-kernel
  `BIC` objective, so better screening alone is not enough to force them out

Practical conclusion:
- these three levers are worth keeping as optional speed-oriented knobs
- but they are **not** the missing fix for the logical precision failures
- the next precision-focused step must attack the objective / ambiguity itself,
  not just the local search mechanics

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 56. Swap-Evaluation Speed Bottleneck: GPU Alone Is Not The Main Idea

The current exact-support-search implementation spends most of its heavy local
search time in repeated support-fixed solves, especially for swap evaluation in
the strict exact-block solver.

Important implementation fact:
- the present support-fixed exact solves are NumPy/SciPy CPU solves, not tensor
  GPU solves
- so simply turning on GPU does not accelerate the current swap path

What looks genuinely promising:
- **batched exact screening**
  - already validated as useful on CPU
  - should also transfer naturally to GPU if needed
- **batched support-fixed swap solves**
  - many swap candidates share the same dropped-parent support and differ by
    only one added rule
  - this structure should allow:
    - batched gradient/Hessian construction
    - batched small dense linear solves
- **rank-1 / low-rank updates**
  - likely even more important than GPU
  - for swap candidates, the Hessian and feature matrix differ by one column, so
    Cholesky or Woodbury-style updates may avoid solving each support from
    scratch
- **hybrid execution**
  - use GPU for screening and batched tensor algebra
  - use exact CPU solves only for a small finalist set if needed

What is less promising:
- naive GPU migration of the current SciPy-per-support loop
- this would keep the same algorithmic structure and therefore preserve most of
  the Python / per-support overhead

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 52. Candidate-Centered Inhibition Motif Block Was A Real Win On The Fastest Benchmark

After dropping the failed root-level safe-screen idea, we tried a much more
selective local neighborhood on the fastest benchmark
`paper_ablation_inhibition_only`.

New variant:
- keep the same frozen-kernel exact block backbone
- keep the same support-fixed exact inhibition refits
- but replace the global inhibition `drop/swap` neighborhood with a tiny
  candidate-centered motif block
- center ranking:
  - inactive inhibition candidates ranked by how many current active inhibition
    rules support them as strict containers or one-edit same-order neighbors
- neighborhood:
  - keep only the top `6` centers with support count at least `2`
  - only active inhibition rules touched by those centers can be reconsidered
  - only swaps inside that motif block are allowed
  - no global pure-drop move outside the block

Actual run on `2026-04-01`:
- benchmark: `paper_ablation_inhibition_only`
- resources:
  - CPU only
  - `12` threads
  - `support_workers = 12`
- output: `tmp_block_motif_ablation_inhibition_only.json`

Observed result:
- elapsed `100.22s`
- recall `1.0`
- precision `1.0`
- no missing rules
- no extra rules

Round-1 neighborhood statistics:
- inhibition block centers: `6`
- inhibition drop candidates actually evaluated: `4`
- inhibition add candidates after block restriction: `0`
- inhibition swap evaluations: `5`
- inhibition support evaluations: `10`
- solver stopped after round 1 with no further change

Comparison against earlier strict exact-support-search results on the same
benchmark:
- older exact `add/drop` block prototype:
  - `222.6s`
  - recall `0.833`, precision `1.0`
- older exact `add/drop/swap` prototype:
  - `452.8s`
  - recall `0.833`, precision `1.0`
- best later exact-preserving speed setting we had recorded:
  - `319.9s`
  - recall `0.833`, precision `1.0`
- new candidate-centered motif block:
  - `100.2s`
  - recall `1.0`, precision `1.0`

Interpretation:
- this is the first concrete evidence in the current codebase that **smaller
  ambiguity neighborhoods** can improve both accuracy and runtime at once
- the win came from changing *which* local neighborhood is searched, not from
  a broader global safe-screen
- on this benchmark, the motif block prevented the harmful global drop that had
  been removing `CGH inh`, while also collapsing the swap search cost

Current best-performing method overall as of this note:
- still **frozen-kernel exact add-only correction** across the full official
  benchmark suite
- reason:
  - it is still the only method verified end-to-end on all `12/12` official
    benchmarks with recall `1.0`

Current most promising next exact-search direction:
- candidate-centered ambiguity blocks with exact local swap search
- reason:
  - unlike the failed root safe-screen idea, this one already produced a real
    speed and accuracy gain in an actual benchmark run

### 53. Block-Global Certificate Search Inside The Motif Was Slower And Worse On The Fastest Benchmark

We then tested the more certificate-oriented variant on the same benchmark
`paper_ablation_inhibition_only`.

Variant:
- keep the same candidate-centered inhibition motif block definition
- but instead of local `swap` search, enumerate **all subsets** inside that
  block exactly
- outside-block inhibition rules stay fixed
- this gives a valid certificate only for the chosen motif block, not for the
  full library

Actual run:
- benchmark: `paper_ablation_inhibition_only`
- resources:
  - CPU only
  - `12` threads
  - `support_workers = 12`
- output: `tmp_block_motif_exact_cert_ablation_inhibition_only.json`

Observed result:
- elapsed `646.11s`
- recall `0.833`
- precision `1.0`
- still missed `CGH inh`
- no extra rules

Block-search statistics:
- round 1:
  - inhibition block universe size `10`
  - exact subsets evaluated `1024`
  - inhibition support solves `1024`
- round 2:
  - inhibition block universe size `9`
  - exact subsets evaluated `512`
  - inhibition support solves `512`

Interpretation:
- exact subset enumeration inside the motif block was **not** a good practical
  move on this benchmark
- it was much slower than the motif `swap-only` version (`646.1s` vs `100.2s`)
- and it was also worse in accuracy, because the block-global `BIC` optimum
  inside that heuristic block preferred dropping `CGH inh`
- so this confirms an important point:
  - a certificate on the wrong block is not enough
  - the useful improvement came from the selective motif neighborhood plus the
    restricted local move set, not from exhaustive subset search inside the
    block

Decision:
- do **not** keep the exact-subset motif-certificate path as an active code path
- keep the simpler candidate-centered local swap motif version instead

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

Current best exact-search direction remains:
- candidate-centered ambiguity blocks with restricted exact local swap search

### 55. A Theorem-Friendly Safe Block Construction Looks Plausible, But It Must Replace The Current Top-K Center Heuristic

The current candidate-centered motif rule itself is still heuristic and should
not be the direct theorem target.

A more promising provable block-selection route is:
- define an inactive safe superset using exact singleton-gain upper bounds
- then define the active part of the block as the interaction component touched
  by that inactive safe set

Why the inactive side looks provable:
- in the frozen-kernel inhibition objective, the benefit of adding a set of
  inactive rules is subadditive because
  - `1 - exp(-(u+v)) <= (1 - exp(-u)) + (1 - exp(-v))`
- therefore the exact gain of any inactive set can be upper-bounded by the sum
  of exact singleton gains at the current residual
- this means a candidate family can be safely excluded if the total remaining
  singleton-gain upper bound is below the incumbent improvement gap

Why the active side may also be made provable:
- if an active rule has zero or sufficiently small interaction with every
  candidate in the inactive safe set, then swapping it with those candidates
  cannot materially improve the objective
- under an exact separability condition, only the connected interaction
  component needs to be searched
- under an approximate separability condition, the same statement may hold with
  an explicit error bound

Practical implication:
- the theorem target should not be the present support-count / top-K center
  ranking
- instead it should be:
  - safe inactive superset by exact gain bounds
  - safe active ambiguity component by interaction graph
  - exact search only inside that certified superset block

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 54. What Looks Provable From The Current Motif-Block Direction

After the latest motif-block experiments, the theorem-friendly parts are now
clearer.

Most promising to make provable:
- **recoverability condition**
  - this is the strongest candidate for a real theorem
  - likely form:
    - assume the true support differs from the current support only inside a
      localized ambiguity block
    - assume every outside-block inactive rule has non-positive reduced cost or
      a valid safe upper bound below the incumbent gap
    - assume a positive margin in penalized population risk between the true
      block support and every other block support
    - then certified search inside that block recovers the true support
- **restricted swap**
  - this looks provable only under a structural condition
  - likely form:
    - if the truth can be reached from the current support by exchanging one
      ambiguous proxy block with one truth block, and every improving move is
      representable by those swaps, then swap-restricted local search is
      sufficient
  - so this is probably a conditional theorem, not a universal one

Only partially promising:
- **block selection**
  - the current candidate-centered ranking itself is still heuristic
  - proving the exact current scoring rule is probably not the right target
  - the more realistic theorem path is:
    - define a larger safe superset block by reduced-cost / KKT screening
    - then prove the true ambiguity set is contained in that block
  - so the provable object is likely a **safe block construction**, not the
    current heuristic top-K center ranking

Bottom-line assessment:
- best theorem target: `recoverability condition`
- second-best theorem target: `restricted swap under localized ambiguity`
- weakest current theorem target: the exact current `block selection` heuristic

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 66. Precision Failure Below `1.0` Came From Two Different Mechanisms, Not One

After the theorem-friendly rerun without `num_predicates_30`, we analyzed the
three remaining non-exact cases:

- `logical_clean_plus`
- `logical_context`
- `kernel_gaussian`

Evidence files:

- full rerun:
  - `data/paper_suite/tmp_safe_swap_confset_suite_10_20_rerun_20260403.json`
- direct support-comparison audit:
  - `data/paper_suite/tmp_precision_failure_quick_analysis_20260404.json`

The important comparison was:

1. frozen-kernel exact support-fixed refit on the final predicted support
2. frozen-kernel exact support-fixed refit on the true support
3. joint kernel refit with the **current** rule-count-only penalty
4. joint kernel refit with **full kernel dimension** penalty
5. direct deletion of the remaining extra rules from the final support

Main finding:
- the precision failures do **not** all come from the same cause
- there are two distinct mechanisms:
  - `criterion mismatch / under-penalization`
  - `one-pass final-selection closure failure`

#### 66.1. `logical_clean_plus` Is A Criterion Mismatch, Not A Search Failure

Observed numbers:

- frozen support comparison:
  - true-minus-predicted BIC gap: `+13.53`
- joint kernel refit with current rule-count penalty:
  - true-minus-predicted BIC gap: `+13.26`
- joint kernel refit with full kernel-dimension penalty:
  - true-minus-predicted BIC gap: `-57.82`

Direct check on the remaining extra:

- extra:
  - `L -> T : excitation`
- exact delete candidate from the final support:
  - inside confidence set: `yes`
  - selection score: `+1.57`

Interpretation:

- under the **current** final criterion, the extra-support model is genuinely
  preferred over the true support
- therefore this case cannot be fixed by making search more exhaustive while
  keeping the same objective
- the bottleneck is not support discovery
- the bottleneck is that the present rule-count-based selector under-penalizes
  support-specific kernel flexibility

#### 66.2. `logical_context` Is Not A Criterion Mismatch Under The Frozen Final Selector; It Is A Closure Failure

Observed numbers:

- frozen support comparison:
  - true-minus-predicted BIC gap: `-0.65`
- joint kernel refit with current rule-count penalty:
  - true-minus-predicted BIC gap: `+24.08`
- joint kernel refit with full kernel-dimension penalty:
  - true-minus-predicted BIC gap: `-188.28`

Direct check on the remaining extra:

- extra:
  - `A and B and D -> T : inhibition`
- exact delete candidate from the final support:
  - inside confidence set: `yes`
  - selection score: `-1.61`

Interpretation:

- under the current frozen-kernel final selector, this extra **should** be
  removable
- the reason it remained is that the final confidence-set deletion was applied
  only once
- after the support moved, a newly admissible improving deletion remained, but
  the pipeline stopped before taking that next exact step

So this case is qualitatively different from `logical_clean_plus`:

- `logical_context` is fixable by making the final selection stage exact over
  the deletion closure / fixed point
- it does **not** require a different criterion to remove the last extra

#### 66.3. `kernel_gaussian` Is Also A Criterion Mismatch, Not A One-More-Pass Failure

Observed numbers:

- frozen support comparison:
  - true-minus-predicted BIC gap: `+42.66`
- joint kernel refit with current rule-count penalty:
  - true-minus-predicted BIC gap: `+62.48`
- joint kernel refit with full kernel-dimension penalty:
  - true-minus-predicted BIC gap: `-295.12`

Direct checks on the remaining extras:

- extras:
  - `B and C and D -> T : inhibition`
  - `G and H -> T : inhibition`
- exact delete candidates from the final support:
  - drop `BCD inh` only:
    - inside confidence set: `yes`
    - selection score: `+0.82`
  - drop `GH inh` only:
    - inside confidence set: `yes`
    - selection score: `+4.96`
  - drop both extras:
    - inside confidence set: `yes`
    - selection score: `+6.61`

Interpretation:

- even after reaching the final support, the current selector still prefers
  keeping the extra rules
- so this is not a one-pass closure issue
- this case, like `logical_clean_plus`, requires a criterion upgrade rather
  than a search upgrade

#### 66.4. Important Negative Conclusion

It is mathematically impossible to fix `logical_clean_plus` or
`kernel_gaussian` by changing **only** the search procedure while keeping the
same current final criterion.

Reason:

- in both cases, the true support loses to the predicted support under the
  current criterion
- an exact algorithm that faithfully optimizes that criterion must therefore
  keep the extras

This means:

- a stronger local search theorem will not solve these cases
- a larger certified family alone will not solve these cases
- exact recovery cannot be a theorem consequence of the current selector on
  these benchmarks

#### 66.5. What The Non-Heuristic Fix Must Be

There are two corresponding non-heuristic fixes.

Fix A:
- replace one-pass final deletion with **exact fixed-point familywise
  selection**
- target failure type:
  - `logical_context`

Rigorous form:

1. define a finite certified deletion family over the active support
2. perform exact support-fixed refits over that family
3. choose the best confidence-set-admissible candidate
4. repeat until no admissible improving deletion remains

This is not heuristic if:

- the family is explicitly defined
- every family member is exactly evaluated
- stopping occurs only at a true fixed point

Fix B:
- replace rule-count-only complexity with **full kernel-dimension complexity**
- target failure type:
  - `logical_clean_plus`
  - `kernel_gaussian`

The natural parameter dimension is already present in code:

- `model_param_dim = 1 + sum_r [1 + |U_r|(K-1)]`

where:

- `1` is the base rate
- each rule contributes one coefficient
- each source in each rule contributes `K-1` free kernel-height degrees of
  freedom after normalization

Empirical consequence of that penalty upgrade:

- once support-specific kernel refit is scored with full kernel dimension, the
  true support beats the extra-support model on **all three** hard cases

#### 66.6. The Paper-Theorem Implication

If kernels are support-specific at the final stage, then the correct selection
object is no longer support alone. It is:

- a finite certified family of `(support, kernel)` models

So the strongest theorem direction is:

1. certified family construction
2. exact familywise selection over that family
3. explicit complexity accounting for support-specific kernels
4. finite-sample oracle guarantee over the family

Exact recovery should then appear only as a corollary under an additional
margin assumption inside that family.

#### 66.7. Bottom Line

Precision `< 1.0` is not coming from one single issue.

- `logical_context`:
  - final selection stops too early
- `logical_clean_plus`, `kernel_gaussian`:
  - the current rule-count-based selector genuinely prefers the extra-support
    model because it under-penalizes rule-specific kernel flexibility

Therefore the non-heuristic path forward is now clear:

- exact fixed-point familywise deletion for the final prune stage
- support-specific model selection with full kernel complexity accounting

Current best-performing method overall remains:
- **frozen-kernel exact add-only correction**

### 136. Signed Fisher-Mobius Full-Suite Prototype (2026-04-08)

Given the failure of raw joint block deletion, active-ancestor residualization, and same-sign proper-subset closure, the next prototype uses a signed lower-order closure over the local rule tangent geometry.

Prototype design:

- keep deterministic `safe-swap confset` as the baseline active-support constructor
- jointly refit active rules with coefficient + rule-specific kernel-height parameters
- for each active rule `r`, build a signed local tangent block using:
  - excitation event term scaled by `1 / sqrt(mu + E)`
  - excitation grid term scaled by `sqrt(exp(-I))`
  - inhibition grid term scaled by `sqrt(lambda)`
  - coefficient amplitude and kernel-height deviation from the global source kernel
- recursively residualize this block against the canonical blocks of all active lower-order predecessor rules, regardless of sign
- allow a deletion only if:
  - the empirical canonical block energy is below the local BIC-style complexity threshold
  - and the full joint kernel-df BIC improves after dropping the rule

Motivation:

- same-sign closure was insufficient because the dominant aliasing mechanism appears to be cross-sign local compensation inside
  `lambda = (mu + E) exp(-I)`
- this prototype therefore uses a signed lower-order closure rather than a same-sign closure
- the gate is still coupled to the successful full joint kernel-df deletion objective, rather than replacing that objective

Experiment protocol:

- evaluate the new prototype on the full 11-benchmark suite with deterministic seed and tmux sharding
- compare against the retained deterministic `safe-swap confset` baseline recorded in each result file

### 137. Signed Fisher-Mobius Full-Suite Results (2026-04-08)

Full 11-benchmark deterministic run completed.

Result summary:

- exact recovery: `6 / 11`
- mean recall: `0.9719`
- mean precision: `0.9075`
- accepted deletion occurred only once across the full suite
  - `logical_clean_plus`: dropped `L -> T : excitation`

Benchmark-level outcomes:

- exact:
  - `ablation_inhibition_only`
  - `num_predicates_20`
  - `kernel_triangular`
  - `kernel_exponential`
  - `num_predicates_10`
  - `ablation_mixed_sign`
- unchanged failure:
  - `logical_shared`
  - `logical_context`
  - `kernel_gaussian`
  - `ablation_excitation_only`
- partial improvement only:
  - `logical_clean_plus`
    - baseline: recall `0.8333`, precision `0.7143`
    - result: recall `0.8333`, precision `0.8333`
    - dropped extra `L -> T : excitation`
    - but still missed true `J and K and L -> T : inhibition`
    - and still retained extra `C and K and L -> T : inhibition`

Most important diagnostic:

- in the overwhelming majority of active rules, the signed canonical residual ratio stayed at `1.0`
- predecessor sets were often empty in the final active supports
- therefore the signed lower-order closure gate almost never opened
- as a result, the prototype behaved nearly identically to the deterministic baseline on most benchmarks

Interpretation:

- the failure is not simply that same-sign closure was too narrow
- even with cross-sign lower-order closure, the final active supports often do not contain the lower-order predecessors needed for a useful Mobius-style residualization
- this means the key bottleneck is still the active-support path itself: the right compensating lower-order signed rules are frequently absent by the time the final deletion stage is reached
- consequently, a final-stage-only canonical block gate is too late to resolve the aliasing

Conclusion:

- signed Fisher-Mobius residualization, applied only as a final deletion gate on the active support, is not sufficient
- the next mathematically coherent direction must move the signed-canonical interaction idea earlier, into candidate-family construction rather than only final deletion

### 138. Why Precision Breaks In The Best Baseline And What The Next Correct Direction Must Be (2026-04-08)

Reference baseline:

- deterministic best baseline snapshot remains
  `tmp_safe_swap_confset_suite_10_20_rerun_20260403.json`
- failures are precision-only on exactly three benchmarks:
  - `logical_clean_plus`
  - `logical_context`
  - `kernel_gaussian`

#### 138.1. The Three Precision Failures Are Not The Same Failure Mode

`logical_context`:

- this is the fixed-point deletion failure already identified earlier
- the best baseline keeps one extra inhibition rule
  - `A and B and D -> T : inhibition`
- in the earlier direct-drop analysis, the support obtained by deleting that rule was already confidence-admissible and preferred by the current selector
- therefore this case is not fundamentally a criterion-mismatch case
- it is an exact-final-selection / stopping-too-early case

`logical_clean_plus` and `kernel_gaussian`:

- these are true criterion-mismatch cases
- under the frozen or support-only final criterion, the extra-support model is genuinely preferred
- once joint coefficient + kernel-shape refit is scored with full kernel-dimension complexity, the extra rules become deletable
- therefore these cases require a model-pair criterion, not merely a stronger search

So the baseline precision failures split cleanly into:

- failure type A: exact fixed-point deletion missing (`logical_context`)
- failure type B: support-only complexity under-penalizes rule-specific kernel flexibility (`logical_clean_plus`, `kernel_gaussian`)

#### 138.2. Why The Recent Final-Stage Canonical Directions Failed

The recent signed-canonical / Mobius-style deletion prototypes failed for a structural reason:

- they attempted to use signed lower-order residualization only after the baseline active support had already been fixed
- by that point, the relevant lower-order signed predecessor rules were often absent from the active support
- therefore the canonical residualization had nothing useful to project against
- empirically this showed up as canonical residual ratios staying near `1.0` and predecessor sets being empty

Hence the signed-canonical idea was not wrong in substance.
The mistake was applying it too late.

#### 138.3. The Next Correct Direction

The next mathematically coherent direction is:

**Signed-Canonical Certified Family Construction + Exact Model-Pair Fixed-Point Selection**

This combines the successful insights from the earlier analyses.

Stage 1: signed-canonical candidate-family construction

- work over the full candidate rule universe up to the allowed order, not only the active support
- for each signed rule `r`, build a local amplitude-weighted tangent block in the intensity geometry
- recursively orthogonalize it against the signed lower-order closure over the full candidate universe
- use the resulting canonical block score, together with concentration radii, to build a deterministic certified candidate family
- this removes the path-dependence problem that broke the final-stage-only canonical gates

Stage 2: exact model-pair family selection

- over the certified family from Stage 1, define the selection object as a finite family of `(support, kernel)` models
- for each candidate model, perform support-specific joint coefficient + kernel-shape refit
- score candidates with full kernel-dimension complexity
- perform exact fixed-point familywise deletion / exchange until a true fixed point is reached

This is the first direction that directly matches both failure types simultaneously:

- it fixes `logical_context` by requiring exact fixed-point final selection
- it fixes `logical_clean_plus` and `kernel_gaussian` by upgrading the selection object from support-only to support-kernel pairs with full kernel complexity

#### 138.4. Why This Direction Is The Most Defensible Theorem-Wise

A clean theorem skeleton is now available.

Assumptions:

- the true signed support belongs to the signed-canonical certified family
- true rules have canonical block norm bounded below by `gamma`
- false extras have canonical block norm bounded above by `gamma_0 < gamma`
- empirical block scores and empirical Fisher blocks concentrate uniformly over the finite family
- inside the certified model-pair family, the true `(support, kernel)` model uniquely minimizes the population penalized objective with positive margin

Then:

- Stage 1 retains all true rules and rejects sufficiently spurious compensating extras with high probability
- Stage 2 selects the true `(support, kernel)` pair with high probability
- exact recovery is a corollary of the finite-family oracle guarantee plus the family margin assumption

#### 138.5. Why This Is Better Than The Directions Already Tried

Compared with raw/final local deletion:
- it is not path-dependent on the final active support alone

Compared with add-only or local closed testing:
- it is symmetric and does not monotonically inflate the support

Compared with support-only structural-risk selection:
- it uses the correct model-pair object when kernels are support-specific

Compared with final-stage-only signed canonical gates:
- it uses signed-canonical interaction information early enough to matter

Bottom line:

- the next serious direction should no longer be a final-prune patch
- it should be a two-stage method:
  - signed-canonical certified family construction first
  - exact support-kernel pair selection second

### 139. Canonical Increment Alone Is Not Enough; Self-Normalization Is Needed (2026-04-08)

One more correction is needed before the signed-canonical direction becomes the right theorem target.

Observation:

- strict higher-order AND rules are intrinsically rarer than lower-order rules
- canonical interaction increments remove lower-order explainability
- but canonical residualization alone does **not** normalize for the smaller effective sample size of higher-order rules

Therefore:

- raw canonical block norm is still not an order-fair statistic
- a true higher-order rule can remain disadvantaged simply because it fires rarely

The correct quantity should be a self-normalized block statistic:

- build the signed canonical increment block `Psi_r`
- compute its efficient score block `S_r`
- compute its efficient Fisher block `I_r`
- use a whitened / self-normalized statistic such as
  `T_r = S_r^T I_r^dagger S_r`

Interpretation:

- canonicalization removes lower-order signed compensation
- Fisher normalization removes the frequency disadvantage of rare higher-order rules

Hence the most defensible next theorem-friendly direction is not:

- raw canonical increment selection

but rather:

- signed canonical increment
- self-normalized efficient-score block testing
- certified family construction
- exact support-kernel pair selection

### 140. Signed Canonical Model-Pair Full-Suite Prototype (2026-04-08)

Working prototype for the currently preferred final direction:

- Stage 1: signed canonical family construction over the full signed rule universe
  - build score-contribution vectors for every signed rule under the deterministic baseline geometry
  - recursively residualize each rule against all lower-order signed predecessors
  - use self-normalized one-sided score statistics with Holm correction to define:
    - protected active rules
    - certified inactive add rules
- Stage 2: exact model-pair family selection
  - enumerate the interval family between protected active rules and certified add rules
  - for each candidate support, jointly refit coefficient + kernel-height parameters
  - score candidates with full kernel-dimension BIC
  - repeat family construction / model-pair selection to a fixed point for a small number of rounds

The key difference from the failed final-stage-only canonical gates is that canonicalization now happens over the full signed candidate universe before exact model-pair selection.

### 141. How To Reduce Computation Without Losing The Core Mathematical Direction (2026-04-08)

The full signed-canonical model-pair prototype is too expensive because it combines:

- full signed-universe canonical screening
- interval-family construction
- repeated joint support-kernel refits
- multiple fixed-point rounds

across all 11 benchmarks at once.

However, the core direction does **not** require keeping all of that expense.

The core ideas that must be preserved are:

1. signed canonicalization happens before final selection
2. higher-order rarity is handled by self-normalized score / Fisher scaling
3. the final selection object is a finite family of `(support, kernel)` models
4. the final choice over that family is exact

What can be reduced without changing the theorem target:

#### 141.1. Keep Full-Universe Screening, But Make It One-Shot

- compute signed canonical self-normalized scores over the full rule universe only once per outer round
- do **not** refit a candidate model for every rule during screening
- screening should remain a score/Fisher calculation only

This preserves the certified-family logic while keeping screening cheap.

#### 141.2. Exact Search Should Be Over A Small Certified Interval Family, Not The Whole Universe

Define:

- `C`: protected active core from delete-side canonical tests
- `A`: certified inactive add rules from add-side canonical tests
- `V = (S0 \\ C) union A`

Then exact search is performed only over the interval family

- `{ C union W : W subset of V }`

This still gives an explicit finite family and exact family selection, but avoids full-universe subset search.

#### 141.3. Use Exact Branch-And-Bound Rather Than Flat Enumeration

Exactness does not require enumerating every family member to completion.

It is enough to maintain:

- an upper bound from the best fully refit candidate so far
- a lower bound for each partial branch

and prune any branch whose lower bound already exceeds the current best upper bound.

This preserves exact family optimization while dramatically reducing candidate refits.

#### 141.4. Use Certified Coarse-To-Fine Candidate Evaluation

For each candidate support:

- first compute a cheap upper bound via warm-started short refit
- then compute a lower bound or certified dominance check
- only ambiguous candidates receive a longer joint refit

This keeps the selection exact as long as pruning only uses valid lower bounds and incumbent upper bounds.

#### 141.5. Reuse Parent Candidate State

Within the interval family, neighboring supports differ by one rule.
Therefore:

- warm-start coefficients from the parent support
- warm-start rule-specific kernel heights from the parent support
- cache candidate evaluations by support key

This is computational optimization only; it does not change the selection object.

#### 141.6. The Right Reduced-Cost Algorithm

The reduced-cost theorem-friendly algorithm should therefore be:

1. deterministic baseline `S0`
2. one-shot full-universe signed canonical self-normalized screening
3. build a small certified interval family between protected active rules and certified add rules
4. exact branch-and-bound support-kernel pair selection on that family
5. optional one more outer round only if the chosen support differs from `S0`

This keeps the mathematical core intact while removing the main computational blow-up from full flat enumeration with repeated heavy refits.

### 142. Reduced-Cost Single-Benchmark Smoke For The Shortest Dataset (2026-04-08)

Before re-running the full suite, run the reduced-cost signed-canonical model-pair prototype on the shortest benchmark only:

- benchmark: `logical_shared`
- rationale: smallest signed rule universe among the paper suite benchmarks
- settings:
  - `family_rounds = 1`
  - reduced optimization steps
  - tmux single-session smoke run

Goal:

- measure actual wall-clock
- inspect resulting `core_count`, `accepted_add_count`, `variable_count`, and `family_size`
- verify whether the reduced-cost protocol is computationally viable before scaling back to all 11 benchmarks

### 143. Even The Reduced-Cost Model-Pair Prototype Is Still Too Slow; The Next Cut Must Remove Candidate-Wise Heavy Refits (2026-04-08)

Empirical update:

- even the single-benchmark smoke run on `logical_shared` remained too slow
- after more than 20 minutes, it still had not completed one benchmark
- therefore the main remaining bottleneck is not full-suite parallelism, but candidate-wise repeated joint support-kernel refits inside the exact family search

This implies the next computational reduction must be stricter:

Preserve the mathematical core:
- signed canonical self-normalized family construction
- finite certified family
- final exact selection over `(support, kernel)` models

But remove candidate-wise heavy refits from the search phase:

1. screening remains one-shot and full-universe
2. family construction produces a small certified interval family
3. branch-and-bound uses only cheap valid bounds during most of the search
4. full joint support-kernel refit is executed only for:
   - the incumbent best candidate
   - candidates whose lower bound does not certify inferiority
5. only one outer round should be used in the first scalable version

This keeps the theorem target intact because exactness is preserved by valid lower/upper bounds, not by flat exhaustive refitting of every family member.

### 144. Exact Branch-And-Bound Lower-Bound Prototype (2026-04-08)

Refined the signed-canonical model-pair selector so that computational reduction no longer depends on flat family enumeration.

New exact search principle:

- build the same signed-canonical certified interval family
- search that family with exact branch-and-bound
- for each branch `(included, undecided)`, compute a valid lower bound by:
  - exactly refitting the full support `included union undecided`
  - subtracting the penalty of that full support
  - adding back the penalty of the minimal support `included`
- prune the branch only when this lower bound exceeds the current incumbent upper bound

This keeps exactness because pruning uses only valid lower bounds and incumbent exact candidate values.

### 145. Invalid First Smoke Run Of Exact BB Prototype (2026-04-08)

The first single-benchmark smoke run of the exact branch-and-bound prototype on
`logical_shared` was invalid due to an implementation bug.

Bug:

- canonical screening was supposed to run over the full signed rule universe
- but the implementation accidentally used `arrays_out` from the active-support
  joint refit
- that dictionary contains only active rules, so screening an inactive rule
  raised `KeyError`

Consequence:

- the failed smoke run is not informative about the method itself
- the run must be ignored and repeated after rebuilding full-universe feature
  arrays from the updated rule heights

### 146. Valid Smoke Result For Exact BB Signed-Canonical Selector On `logical_shared` (2026-04-08)

The corrected exact branch-and-bound smoke run on `logical_shared` completed successfully.

Observed outcome:

- elapsed time: about `790s`
- baseline support was already exact
- final result remained exact and identical to the baseline support

Important diagnostics:

- `core_count = 1`
- `accepted_add_count = 16`
- `variable_count = 22`
- branch-and-bound stats:
  - `visited = 37`
  - `pruned = 19`
  - `support_evals = 20`

Interpretation:

- exact branch-and-bound dramatically reduced the number of heavy candidate evaluations compared with flat family enumeration
- so the computational direction is materially better
- however, signed-canonical screening still accepted far too many inactive add rules on this benchmark
- therefore the next bottleneck is no longer exact search itself, but over-expansive certified family construction

This means the next refinement should target the certification statistic, not the branch-and-bound machinery.

### 148. Cross-Sign Compensation Aliasing: The Right Mathematical Fix (2026-04-08)

The remaining hard case `logical_context` suggests that the main unresolved
failure is not merely same-sign higher-order aliasing, but cross-sign
compensation aliasing.

Observed phenomenon:

- a false excitation rule can survive because it compensates residual
  over-suppression created by the current inhibition structure
- equivalently, a false excitation and a missing/underfit inhibition can induce
  a nearly cancelling direction in intensity space
- ordinary BIC or dimension-only penalization cannot distinguish this well,
  because they count parameters but do not penalize local non-identifiability

This suggests that the correct selection object is not

- support size only
- or even support plus kernel dimension only

but rather a Laplace-style support-kernel evidence criterion that includes local
 curvature / identifiability.

Proposed exact criterion on a certified finite family:

- for each candidate `(S, K)` after exact joint refit, define
  `Q_n(S, K) = 2 * NLL_val(theta_hat_{S,K}) + d(S,K) * log n - log pdet(I_can(theta_hat_{S,K})))`
- here `I_can` is the observed or empirical Fisher matrix written in the
  signed-canonical rule-block coordinates
- `pdet` is the pseudo-determinant on the canonical active subspace

Interpretation:

- `2 * NLL_val` measures fit
- `d log n` is the usual complexity penalty
- `- log pdet(I_can)` is an exact Occam / curvature correction

Why this directly targets cross-sign compensation aliasing:

- if a false excitation survives only because it compensates inhibition error,
  then the associated signed-canonical active block has a nearly redundant
  direction
- this makes the canonical Fisher matrix close to singular
- hence `pdet(I_can)` becomes small
- therefore the Laplace-corrected criterion penalizes that false compensatory
  model more strongly than ordinary BIC

By contrast, a true higher-order inhibition rule should contribute a genuine
 canonical interaction increment, so after exact refit its canonical Fisher
 block should remain well-conditioned.

This gives a mathematically coherent next algorithm:

1. deterministic safe-swap confset baseline
2. full-universe signed-canonical self-normalized screening
3. certified finite interval family construction
4. exact branch-and-bound over `(support, kernel)` candidates
5. final selection by Laplace-corrected canonical evidence, not dimension-only
   BIC

The theorem target is clean:

- true `(S*, K*)` lies in the certified family
- false compensatory models have either larger risk or smaller canonical
  pseudo-determinant
- empirical risk and canonical Fisher concentrate uniformly over the family
- then the exact finite-family selector based on `Q_n` recovers `(S*, K*)` with
  high probability under a positive evidence margin

This is preferable to heuristic exchange or ad-hoc deletion rules because the
entire method remains:

- finite-family
- exact over the family
- based on explicit statistical functionals
- directly tied to local identifiability in the signed logical TPP model

### 149. First Laplace-Corrected Exact BB Implementation With Exact Speed Patches (2026-04-08)

Implemented the next mathematically aligned prototype on top of the signed
 canonical exact branch-and-bound selector.

Criterion change:

- the final support score is no longer plain support-kernel BIC
- for each exact candidate support, use
  `criterion = base_bic + canonical_correlation_penalty`
- `canonical_correlation_penalty = - log det(C_can)`
- `C_can` is the correlation matrix of active signed-canonical rule vectors
- this penalty is nonnegative and grows when cross-sign compensatory models
  become nearly redundant in canonical interaction space

Why this is mathematically useful:

- false compensatory excitation/inhibition combinations should produce nearly
  collinear canonical active directions
- the determinant of the canonical correlation matrix then shrinks
- therefore the selector penalizes locally non-identifiable compensatory models
  more strongly than dimension-only BIC

Exact speed patches added at the same time:

1. cached-super-support warm starts for exact support-kernel refits
2. two-tier exact lower bounds:
   - first use the best cached super-support lower bound
   - only if needed, evaluate the branch full support exactly
3. branch pruning still uses only valid lower bounds, so exactness is preserved

These are exact accelerations, not heuristic truncations.

### 150. Partial Hard-Case Result Of Laplace-Corrected Exact BB Selector (2026-04-08)

Ran the Laplace-corrected exact branch-and-bound selector with exact speed
 patches on the three hard cases.

Current status:

- `logical_clean_plus`: finished
- `logical_context`: finished
- `kernel_gaussian`: still running

Observed finished results:

- `logical_clean_plus`
  - baseline in this deterministic rerun already missed
    `J and K and L -> T : inhibition`
  - result was identical to that baseline
  - `core_count = 1`, `accepted_add_count = 60`, `variable_count = 64`
  - branch-and-bound immediately pruned at the root after only two exact
    support evaluations

- `logical_context`
  - baseline in this deterministic rerun had precision failure with two extra
    inhibition rules:
    - `A and E and G -> T : inhibition`
    - `B and E -> T : inhibition`
  - result was identical to that baseline
  - `core_count = 2`, `accepted_add_count = 18`, `variable_count = 25`
  - branch-and-bound again pruned at the root after only two exact support
    evaluations

Interpretation:

- the speed patches worked in one narrow sense: exact branch-and-bound did very
  little exact search
- however, the certified family is still far too expansive
- more importantly, the incumbent baseline support is so strong under the new
  criterion that the root lower bound already certifies the whole family as not
  better
- therefore the current Laplace-style canonical correlation penalty is not yet
  separating the false compensatory models from the incumbent support

This means the main bottleneck remains the certification / evidence statistic,
not the exact search backend.

### 151. Active-Support Projected Canonical Efficient-Score Family Construction (2026-04-08)

The next mathematically coherent refinement is to tighten the certified family
 construction itself.

Reason:

- the previous signed-canonical family statistic only removed lower-order signed
  structure in the full rule universe
- but it did not condition on the current active support tangent space
- therefore many compensatory inactive rules still looked significant and the
  certified family remained too large

New family-construction principle:

1. compute full-universe signed-canonical interaction blocks as before
2. let the current deterministic baseline active support define the nuisance
   tangent space
3. for each inactive candidate, project its signed-canonical block onto the
   orthogonal complement of the active-support tangent span
4. use the resulting self-normalized one-sided statistic for certified add
   decisions
5. for each active rule, project against the other active rules and use the
   analogous statistic for protected-core decisions

This directly targets cross-sign compensation aliasing:

- if a false excitation or inhibition is only compensating error already spanned
  by the current active support, its projected canonical score should collapse
- a genuine missing interaction should retain projected signal beyond the active
  nuisance span

This refinement is still non-heuristic because:

- the family is defined by explicit projected canonical score statistics
- multiplicity correction remains Holm on a finite family
- the exact branch-and-bound model-pair selector is unchanged
- all speedups remain exact lower-bound or warm-start reuse, not approximate
  truncation

### 152. Deterministic Hard-Case + Full-Suite Run For Projected Canonical Family (2026-04-08)

Launched the first deterministic experiment for the projected-canonical
 efficient-score family construction.

Protocol:

- keep the exact signed-canonical model-pair branch-and-bound backend
- replace the certified family statistic by:
  - full-universe signed-canonical increment construction
  - projection onto the orthogonal complement of the current active-support
    tangent span
  - self-normalized one-sided score statistics
  - Holm correction for active-core and inactive-add decisions

Runs launched:

- hard-case suite:
  - `logical_clean_plus`
  - `logical_context`
  - `kernel_gaussian`
- deterministic full suite shards over the 11 benchmarks

Rationale:

- this is the first version that directly attacks cross-sign compensation at the
  family-construction stage while preserving exact finite-family model-pair
  selection
- if it works, the expected signal is:
  - much smaller `accepted_add_count`
  - improved hard-case precision
  - reduced branch-and-bound search cost relative to the previous
    non-projected canonical family

### 153. Current Deterministic Baseline Snapshot From The Running Projected-Canonical Experiments (2026-04-08)

The projected-canonical experiments record the deterministic baseline together
with the exact selector result in the same JSON files.

Among the 9 benchmarks already finished in the deterministic `seed=0` runs, the
baseline currently is:

- exact on 7/9
- mean recall `0.9815`
- mean precision `0.9753`

Finished deterministic baseline outcomes:

- exact:
  - `logical_shared`
  - `kernel_triangular`
  - `kernel_exponential`
  - `num_predicates_10`
  - `num_predicates_20`
  - `ablation_inhibition_only`
  - `ablation_mixed_sign`
- non-exact:
  - `logical_clean_plus`
    - missing `J and K and L -> T : inhibition`
  - `logical_context`
    - extras `A and E and G -> T : inhibition`, `B and E -> T : inhibition`

Interpretation:

- the deterministic baseline used inside the new exact-selector experiments is
  not reproducing the older saved best-baseline snapshot
- therefore comparisons should distinguish:
  - the older retained best baseline snapshot
  - the current deterministic rerun baseline embedded in the new experiments

### 154. Why The Projected-Canonical Direction Degraded Performance, Especially For Inhibition (2026-04-08)

The projected-canonical family construction degraded performance because the new
 statistic tested the wrong object.

What it effectively tested:

- for each rule, keep only the component of its signed-canonical block that is
  orthogonal to the current active-support tangent span
- then use a self-normalized one-sided score on that projected residual

This sounds principled, but in pure or inhibition-dominated regimes it is too
 strong. A true high-order inhibition rule may be:

- jointly necessary for the correct decomposition
- but not individually orthogonal to the rest of the active inhibition support

So the projection removes not only false compensatory directions, but also much
of the true higher-order inhibition signal.

The clearest example is `ablation_inhibition_only`.

Observed diagnostics:

- deterministic baseline was exact with inhibition support
  `{A, B, CD, CGH, DFG, EF}`
- after projected-canonical family construction:
  - `core_count = 0`
  - only one inactive add was certified:
    `A and E and F -> inhibition`
  - the final exact selector chose only `{A, B}`
- the active projected statistics for true rules were all weak:
  - `CGH inhibition`: stat about `1.86`
  - `DFG inhibition`: stat about `1.22`
  - `CD inhibition`: stat about `0.21`
  - `EF inhibition`: stat about `0.20`
- meanwhile the strongest inactive projected statistic was the false rule
  `AEF inhibition` with stat about `37.9`

Interpretation:

- in a same-sign inhibition system, the active-support nuisance span is itself
  dominated by inhibition directions
- projecting a true higher-order inhibition onto the orthogonal complement of
  that span removes much of its real signal, because the rule is not
  "orthogonally unique" even though it is jointly indispensable
- therefore the new statistic confuses:
  - "not individually orthogonal to the active inhibition basis"
  with
  - "not truly needed"

This explains why inhibition collapsed much more severely than excitation:

- same-sign inhibitory rules are much more mutually collinear in the
  multiplicative suppression geometry
  `lambda = (mu + E) exp(-I)`
- after conditioning on the whole active support, genuine inhibition increments
  can look weak
- self-normalization then favors small-variance spurious directions such as
  compensatory inactive inhibition rules

So the failure is not caused by exact search or branch-and-bound.
It is caused by an over-conditioned family-construction statistic.

Conclusion:

- the correct object is not
  "signal orthogonal to the entire current active support"
- but rather
  "signal orthogonal only to the nuisance subspace generated by structurally
   lower-order ancestors (and possibly sign-linked compensators), while keeping
   jointly necessary same-order inhibitory structure intact"

### 155. Is Excitation-Inhibition Entanglement Fundamental To TPPs? (2026-04-08)

Important distinction:

- some excitation-inhibition entanglement is fundamental whenever we try to
  recover multiple signed latent effects from a single observed point-process
  intensity
- but the *severity and form* of that entanglement is highly model-dependent

What is fundamentally unavoidable:

- if two different signed decompositions induce the same intensity trajectory
  `lambda_theta(t)`, then no estimator can distinguish them from event times
  alone
- more generally, near-equal intensity trajectories imply weak local
  identifiability and poor finite-sample recovery

So full disentanglement is not guaranteed for arbitrary TPP models.

What is specific to the current model:

- in the current signed logical TPP,
  `lambda(t) = (mu + E(t)) exp(-I(t))`
- inhibition does not merely subtract a signed contribution; it rescales the
  entire excitation-plus-baseline channel
- therefore higher-order inhibition can suppress the very evidence used to
  detect itself
- this is a stronger asymmetry than in a model with an additive canonical
  predictor

Examples of models where the entanglement is milder:

1. additive canonical-parameter models

- define a signed predictor
  `eta(t) = b + sum_r s_r beta_r phi_r(t)`
  and intensity `lambda(t) = psi(eta(t))`
- here excitation and inhibition are symmetric in `eta`
- compensation can still happen through correlated features, but not through the
  same multiplicative suppression geometry as `(mu + E) exp(-I)`

2. models with extra observed channels or states

- if excitation-like and inhibition-like mechanisms affect different observed
  marks, latent states, or side information, then identifiability can improve
  dramatically

Hence:

- cross-sign aliasing is not "the destiny of all TPP models" in the same
  severity
- but some identifiability tension is unavoidable for single-stream signed TPP
  recovery
- our current failures come from both:
  - the unavoidable single-stream ambiguity
- and an especially harsh multiplicative excitation/inhibition coupling in the
  current parameterization

### 156. Clarifying The Two Alternative Ways To Reduce Sign Entanglement (2026-04-08)

Important clarification about the two broad alternatives:

1. additive canonical-parameter signed TPP
2. extra observed channels / states

On 1:

- a naive additive intensity
  `lambda(t) = mu + E(t) - I(t)`
  is not acceptable because positivity can fail
- therefore positivity must be enforced through a monotone link:
  - `lambda(t) = softplus(eta(t))`
  - or `lambda(t) = exp(eta(t))`
  with
  `eta(t) = b + sum_r s_r beta_r phi_r(t)`

Tradeoff:

- yes, this introduces a nonlinearity
- but the current model is already highly nonlinear because
  `(mu + E) exp(-I)` couples excitation and inhibition multiplicatively and also
  uses nonlinear conjunctive features with learned kernel shapes
- so the real comparison is not
  "linear vs nonlinear"
  but
  "asymmetric multiplicative nonlinearity vs symmetric linked canonical
   nonlinearity"

Why the linked canonical form is still attractive:

- excitation and inhibition enter symmetrically in `eta`
- identifiability analysis can be done in the canonical predictor geometry
- positivity is automatic
- with bounded features and compact parameter sets, links such as softplus can
  be controlled by derivative bounds in proofs

On 2:

- the idea is not to add arbitrary hidden variables
- the point is that identifiability improves if the data contain more than a
  single scalar event stream

More detailed examples:

a. marked target events

- instead of observing only event times `{t_i}`, observe `(t_i, m_i)` where
  `m_i` is a mark
- if excitation and inhibition affect the mark law differently, then two models
  with the same intensity trajectory but different sign decompositions may
  induce different joint distributions of `(t_i, m_i)`
- then sign disentanglement can improve

b. multitype target channels

- observe several target-like outputs rather than one scalar target stream
- excitation may increase one channel while inhibition suppresses another or
  changes relative mark probabilities
- then inference uses the joint multivariate intensity, not one scalar stream

c. observed refractory / suppression state

- suppose there is an observed covariate or state process `x(t)` that records
  whether the target is locally suppressed, saturated, or context-gated
- inhibition can then be identified not only through fewer events, but through
  its systematic effect on `x(t)`

d. context covariates that expose conjunction activation

- if one observes side information making conjunction activation more explicit,
  high-order rules become less sparse/ambiguous
- then a true high-order inhibition need not compete only through the scalar
  event-intensity residual

Takeaway:

- alternative (2) is powerful for identifiability, but it changes the learning
  problem because the data are richer than a single-stream TPP
- so it is conceptually important, but probably not the right immediate move
  if the paper's object remains single-stream strict logical TPP recovery

Hence the practically relevant comparison for the current project is:

- stay with the current asymmetric multiplicative model and repair its
  statistical selection geometry
  versus
- move to a symmetric canonical-parameter signed TPP with a positive link and
  redo the recovery theory in that model class

### 147. Exact Speed-Ups That Preserve The Mathematical Core (2026-04-08)

To make the signed-canonical exact selector fast enough without introducing
heuristics, the only admissible accelerations are ones that do not change:

- the certified family definition
- the exact `(support, kernel)` objective
- the pruning rule "prune only when a valid lower bound exceeds the incumbent"

The mathematically safe accelerations are:

1. recursive canonicalization on the subset lattice

- current screening recomputes many least-squares projections rule-by-rule
- instead, build canonical predecessor bases recursively and reuse them
- this keeps the same canonical statistic, but avoids repeated dense solves

2. two-tier exact lower bounds

- keep the current exact branch-and-bound search
- but first use a cheap global lower bound inherited from an already refit
  super-support
- only compute the tighter branch-specific lower bound if the cheap bound
  cannot prune
- this preserves exactness because both bounds are valid lower bounds

3. monotone super-support cache reuse

- if a branch support is contained in a previously refit super-support, the
  super-support objective gives a valid lower bound for the branch
- therefore branch search can reuse super-support evaluations instead of
  refitting every branch full support from scratch

4. exact incumbent warm starts

- warm starts may be used only as a numerical accelerator for solving the same
  exact support-kernel optimization problem
- they do not change the candidate family or pruning logic
- therefore they affect runtime, not the theorem object

5. deterministic parallel sibling evaluation

- sibling branches or benchmark shards may be evaluated in parallel
- this changes only wall-clock time, not the algorithmic object

The main point is:

- speed must be improved by algebraic reuse, valid bounds, caching, and
  deterministic parallelism
- not by candidate caps, top-k truncation, ad-hoc thresholds, or approximate
  family restriction

### 157. Observed-State Detour Removed As Invalid Branch (2026-04-09)

The observed-state branch was removed entirely after user clarification.

- that branch changed the data object by adding an extra observed channel
- but the intended direction was to change the intensity formula itself
- therefore the observed-state diagnostics, scripts, generated datasets, and
  outputs were all deleted and are not part of the active research path

The corrected next step is:

- build a hardest-case synthetic dataset under a sign-symmetric positive-link
  intensity model
- keep the strict AND rule semantics
- evaluate exact local support selection under the matching intensity model

### 158. Corrected Intensity-Change Hard-Case Experiment (2026-04-09)

To align with the intended direction, the active branch is now:

- hardest case: `logical_context`
- new intensity model:
  `lambda_k(t) = exp(log b_k + sum_r s_r w_r p_r(t))`
  where:
  - `s_r = +1` for excitation rules
  - `s_r = -1` for inhibition rules
  - `p_r(t)` keeps the same strict AND `product_bounded` activation
- therefore the change is in the intensity formula itself, not in the
  observation model

Current diagnostic object:

- materialize a new synthetic dataset
  `paper_logical_context_loglink`
- compare a finite exact local family under the matching log-link model:
  - true support
  - true support plus the historical precision-failure competitor
    `A and B and D -> T : inhibition`

The purpose of this diagnostic is:

- not to solve the full learning stack yet
- but to test whether the sign-symmetric positive-link intensity alone removes
  the original compensatory extra-rule preference in the hardest precision
  failure case

### 159. Log-Link Logical-Context Hard-Case Result (2026-04-09)

The single hard-case diagnostic completed for `logical_context` under the
sign-symmetric positive-link model

`lambda_k(t) = exp(log b_k + sum_r s_r w_r p_r(t))`

with the same strict AND `product_bounded` rule activation.

Compared candidates:

- true support
- true support plus historical extra
  `A and B and D -> T : inhibition`

Result:

- true-support BIC: `6817.6810`
- competitor BIC: `6824.1274`
- `delta_bic_competitor_minus_true = 6.4464`
- winner: `true_support`

Important detail:

- the competitor fit pushed the extra rule coefficient to `0.0`
- so under the log-link model, the extra inhibition rule was not supported as a
  useful compensatory term in this local exact comparison

Interpretation:

- in the original asymmetric multiplicative model, the extra rule could survive
  as a compensatory fit artifact
- in the sign-symmetric positive-link model, the same local ambiguity is
  resolved in favor of the true support

This is only a local hardest-case result, not a full recovery result, but it
is the first direct evidence that changing the intensity formula itself may fix
the precision failure mechanism instead of only patching the selector.

### 160. End-to-End Log-Link Learner On The Same Hard Case Failed Trivially (2026-04-09)

After adding `canonical_loglink` branches into the current active-set learner,
the same `logical_context_loglink` hard case was run end-to-end.

Outcome:

- selected rule count: `0`
- predicted support: empty
- recall: `0.0`
- precision: undefined / effectively no recovered rules
- validation BIC of the empty model: `7031.2847`

Interpretation:

- the local exact comparison showed that the log-link intensity can resolve the
  `true` vs `true + extra` ambiguity once the relevant family is given
- but the current active-set search/screening path, even after a first
  log-link patch, is still calibrated to the old multiplicative geometry
- in particular, the one-step candidate score and the sparse search path are
  now too conservative and reject every rule at the screening stage

Therefore the present conclusion is:

- changing the intensity formula itself is promising
- but the current end-to-end learner is not yet mathematically aligned with
  that new model class
- the next step must redesign the active-set score test and sparse search path
  specifically for the canonical log-link predictor, rather than only swapping
  the NLL/BIC formulas

### 161. Best-Matched Additive-Model Direction (2026-04-09)

Given the current research knowledge, the best mathematically aligned path for
the additive / canonical-predictor model is:

- fixed-kernel strict-AND feature construction
- one-sided canonical score screening under the log-link null
- familywise correction (Holm)
- exact convex support selection over the screened finite family

Why this direction fits the additive model best:

- with fixed rule features, the log-link negative log-likelihood is convex in
  the nonnegative rule coefficients
- excitation and inhibition enter symmetrically as signed columns in the
  canonical predictor
- one-sided score tests are the natural local optimality conditions for adding
  positive coefficients
- exact family selection over the screened family is theorem-friendly and avoids
  the path dependence of the previous multiplicative active-set search

This direction was implemented on the `logical_context_loglink` hardest case as
the next experiment after the failed log-link patch of the old active-set
learner.

### 162. Full Log-Link Canonical-Family Suite Launch (2026-04-09)

After the single-case `logical_context_loglink` experiments, the same
mathematically aligned additive-model direction was generalized to the full
benchmark suite.

The suite runner:

- materializes a log-link version of each benchmark config
- keeps strict AND rule semantics and fixed kernel dictionaries
- computes one-sided canonical score statistics
- applies Holm familywise correction
- performs exact convex family selection over the screened finite family

The suite was launched in deterministic sharded form via `tmux`:

- `loglink_suite_gpu0_20260409`
- `loglink_suite_gpu1_20260409`
- `loglink_suite_gpu2_20260409`
- `loglink_suite_gpu3_20260409`

with partial JSON outputs written after each completed benchmark so that the
suite can be monitored incrementally instead of only at the end.

### 163. Certified Approximate Inner Solves While Preserving Exact Model Selection (2026-04-09)

The current bottleneck of the additive / log-link suite is not the screening
 statistic itself but the exact family evaluation stage:

- Holm-selected candidate count `m` can still be moderate
- the family size is then `2^m`
- each subset is currently solved by a fresh convex optimization

This suggests the right acceleration principle is **not** heuristic family
 truncation, but certified approximation of the inner convex subproblems.

The mathematically clean direction is:

- keep the same finite screened family
- keep the same exact support-selection objective
- replace exact inner solves by primal/dual certified solves that provide
  `L(S) <= Q(S) <= U(S)` for each support `S`
- prune supports or branch-and-bound nodes only using valid lower bounds
- terminate with exact optimality when the incumbent upper bound is below every
  remaining lower bound
- or terminate with `epsilon`-optimality when the global gap is at most
  `epsilon`

This preserves theorem-friendly exactness at the model-selection level while
 turning the expensive convex refits into anytime certified numerical solves.

In particular, for a true model-selection margin `Delta`, any `epsilon` such
 that `2 epsilon < Delta` still yields exact selected support despite not
 solving every inner convex problem to machine precision.

### 164. Certified Approximate Inner Solve Worked On The Log-Link Local Hard Case (2026-04-09)

A local hard-case diagnostic was run on `logical_context_loglink` using the
historical precision-failure support and its single-deletion neighborhood.

Method:

- exact outer selection over the 9-support local family
- certified approximate inner solve for each support
- profile log-link convex objective on the training split
- primal upper bound from the current fitted support objective
- dual lower bound from entropy-projected feasible moment matching
- exact winner declared when one support's upper bound is below every other
  support's lower bound

Result:

- certified exact winner: `drop_inh_29`
- this is exactly the true support, i.e. the historical extra
  `A and B and D -> T : inhibition` is deleted
- resulting local metrics are `recall=1.0`, `precision=1.0`

The run finished in about `83.45s`, which is dramatically faster than the
previous exact family implementation while preserving a certified
interval-based decision rule at the inner-solve level.

### 165. Full Benchmark Launch For The Certified Local-Family Additive Direction (2026-04-09)

To make the additive / log-link direction practical at benchmark scale without
 introducing heuristic truncation, the current full-suite experiment uses:

- deterministic log-link dataset materialization
- Holm-screened support
- exact outer selection over the deterministically defined local family
  consisting of the screened support and all single-deletion neighbors
- certified approximate inner solves via primal/dual interval bounds

This is not yet the full screened-family exact outer selection, but it keeps
 the same theorem-friendly additive model and replaces the previous
 combinatorial bottleneck by a tractable certified local exact-selection stage.

The full 11-benchmark suite was launched with one dataset per `tmux` session
 so that all currently free local CPU resources can be used in parallel while
 avoiding the GPU currently occupied by other users.

Correction:

- this branch changed the outer selection object from the intended
  full screened-family exact outer selection to a local screened-support plus
  single-deletion neighborhood exact selection
- therefore it is not the same algorithmic object and cannot be used as the
  main theorem-aligned benchmark result

Action taken:

- the local-family 11-benchmark run was stopped
- the suite wrapper for that branch was removed
- only the single-case certified-inner prototype was retained as an inner-solve
  development artifact

### 166. Quick Hierarchical-Branch Screening Diagnostic Was Far Too Loose (2026-04-09)

A fast structural diagnostic was run on `logical_context_loglink` to estimate
 how much a naive branch-level hierarchical screening would shrink the screened
 family before exact outer selection.

Diagnostic choice:

- tree over strict-AND subsets up to order 3
- separate excitation / inhibition branches
- conservative branch-level group score test under the null
- recurse into children whenever the parent branch is rejected

Outcome on `logical_context_loglink`:

- tested branches: `126`
- surviving leaf rules: `110`
- implied exact outer family size: `2^110`

Interpretation:

- this naive branch-screening design does **not** solve the family explosion
- in fact it is even looser than the flat Holm screen for this hard case
- therefore simply replacing flat screening by a branch score test is not
  enough; the branch statistic itself must be made much more selective

Consequence:

- under this naive branch construction, exact outer selection is still
  computationally impossible
- the next viable direction must strengthen the structured screening object
  rather than only reorganize the same weak score test onto a tree

### 167. Projected-Branch Screening Was Still Far Too Loose (2026-04-09)

To test whether the branch statistic only failed because it used a raw group
 score, a stronger single-case diagnostic was run on `logical_context_loglink`.
 This version projected each branch statistic against strict lower-order signed
 subset nuisance before forming the branch score.

Diagnostic choice:

- same strict-AND subset tree up to order 3
- separate excitation / inhibition branches
- branch score formed after projecting out lower-order signed subset nuisance
- recurse into children whenever the projected branch statistic is rejected

Outcome on `logical_context_loglink`:

- tested branches: `126`
- surviving leaf rules: `94`
- implied exact outer family size: `2^94`

Interpretation:

- nuisance projection did reduce the survivor count from `110` to `94`
- however the screen is still far too loose to make full exact outer selection
  feasible
- therefore the bottleneck is not just flat-vs-tree organization; even the
  stronger projected branch null is still too weak as a structured screening
  object for the hard case

Consequence:

- exact outer selection remains computationally impossible at this screening
  strength
- the next structured screen must test something more selective than branch
  existence, e.g. genuinely new canonical interaction information rather than
  broad branch-level non-nullness

### 168. Exact Outer Selection Need Not Enumerate Every Subset (2026-04-09)

It is not necessary to solve the inner convex fit for every one of the `2^m`
 screened subsets in order to preserve exact outer-model selection.

Exact alternative:

- keep the same screened family and the same support-level objective
- replace flat subset enumeration by exact branch-and-bound over inclusion
  variables
- at each node, compute a valid lower bound on all descendants by solving a
  convex relaxation
- keep an incumbent exact feasible support as an upper bound
- prune any node whose lower bound is already worse than the incumbent

Why this still preserves exactness:

- the outer optimization problem is unchanged
- no heuristic truncation is used
- pruning is based only on mathematically valid lower/upper certificates
- therefore the final selected support is identical to what exhaustive exact
  enumeration would have returned

Interpretation:

- the current bottleneck is not that exactness requires explicit evaluation of
  all subsets
- the real question is whether the lower bounds are strong enough to prune most
  of the screened lattice early
- for the additive/log-link setting, this is the most promising non-heuristic
  path because the inner problem is convex and naturally admits certified
  primal/dual bounds

### 169. Strict Full-Family Exact Outer Selection Was Switched To Exact Branch-And-Bound (2026-04-09)

To preserve the intended model-selection object while avoiding explicit
 evaluation of all `2^m` screened subsets, a new suite wrapper was created for
 the additive/log-link setting:

- screened family is unchanged
- outer problem is still exact support selection over the full screened family
- exhaustive enumeration is replaced by exact branch-and-bound
- support-level convex fits use certified primal/dual interval solves
- pruning uses only mathematically valid lower/upper objective certificates

This keeps the outer exactness target intact while moving all approximation into
 the inner convex solves, where certified gaps can be monitored.

Correction:

- the first full-suite launch of this strict branch-and-bound wrapper was
  invalid because empty-support evaluation still called the generic L-BFGS-B
  path with an empty bound list
- this caused most runs to terminate immediately before producing any usable
  result
- the wrapper was patched with an exact closed-form empty-support solve and the
  failed logs/json files were discarded before rerunning

### 170. Concrete Bottlenecks Of The Strict Log-Link Branch-And-Bound Wrapper (2026-04-09)

Single-case profiling on `paper_logical_shared_loglink` showed three separate
 bottlenecks.

Measured preprocessing cost:

- `materialize_loglink_dataset`: about `0.009s` when cached files already exist
- `setup_problem`: about `8.5s`
- `compute_rule_feature_arrays`: about `11.8s`
- therefore each benchmark pays about `20s` of deterministic setup cost before
  branch-and-bound even starts

Measured screened-family size:

- `logical_shared_loglink`: Holm survivor count `34`
- `logical_context_loglink`: Holm survivor count `41`

Measured support-evaluation cost before speed patches:

- empty support: about `0.001s`
- 4-rule support: about `1.64s`
- 34-rule support: about `12.6s`

Profiling showed that the main per-support bottleneck was
 `certified_fit_support`, especially repeated calls to
 `repair_dual_distribution`.

Exactness-preserving speed patches applied:

- dual repair now uses warm starts rather than zero starts
- the certified inner solver now uses a cheaper default of one primal round
  plus one dual repair
- the previous iterative deletion incumbent was removed because it introduced an
  `O(m^2)` support-evaluation bottleneck before branch-and-bound

Measured support-evaluation cost after the inner-solver patch:

- 34-rule support with `1` primal round and `25` iterations: about `1.93s`
- corresponding certified BIC interval width on this case: about `9.41`

Interpretation:

- the largest per-node cost was genuinely reduced by more than a factor of `6`
  relative to the original `12.6s` full-support evaluation
- however the remaining bottleneck is still the combination of
  `20s` preprocessing plus node explosion from screened-family sizes in the
  `34-41` range
- therefore the next exactness-preserving improvement must strengthen
  branch-level pruning or reduce survivor count via a mathematically stronger
  screening object

### 171. Active-Set Dual Repair Fix: The Idea Was Right, The Tolerance Was Wrong (2026-04-09)

To reduce the dominant inner-solver cost without changing the exact outer
 object, the dual repair step was rewritten from:

- full-dimensional nonnegative L-BFGS-B over all screened coordinates

to:

- active-set dual repair over only the currently violated dual constraints

This is still the same dual feasibility problem. The only change is that the
 repair now solves it in the smallest active coordinate set and expands that set
 only when new violated constraints appear.

Initial result looked bad:

- the first active-set implementation made 34-rule support evaluation jump back
  to about `25.4s`

Root cause:

- after the first reduced 3-dimensional repair solve, the minimum dual moment
  residual was already only about `-1.2e-9`
- but the repair used `feas_tol = 1e-10`
- so the code kept re-solving the **same 3-dimensional reduced problem**
  roughly 30 more times with `nit = 0`

Concrete diagnostic on `paper_logical_shared_loglink` full screened support:

- initial violated constraints from primal-implied distribution: `3`
- initial minimum residual: about `-1.95e-5`
- reduced active-set after first repair solve:
  - active size `3`
  - optimizer iterations `9`
  - minimum residual improved to about `-1.2e-9`

Fix:

- set the numerical dual-feasibility tolerance to `1e-8`
- stop immediately if the violating set is unchanged, the reduced optimizer
  returns `nit = 0`, and the minimum residual no longer improves

Measured outcome after the fix:

- 34-rule support evaluation returned to about `1.98s`
- full preprocessing + full-support single evaluation on
  `paper_logical_shared_loglink` is about `23.75s`
- certified BIC interval width remains about `9.41`

Interpretation:

- active-set dual repair is still the right exactness-preserving direction
- the slowdown came from an over-strict numerical feasibility threshold, not
  from the active-set idea itself
- after correcting the tolerance, inner support evaluations are again in the
  intended `~2s` range

Current strict branch-and-bound status on `paper_logical_shared_loglink` after
 the fix:

- progress output now appears normally
- after a few minutes the solver had already visited hundreds of nodes and
  started pruning regularly
- so the main remaining bottleneck is no longer the dual repair itself, but the
  strength of the branch lower bound / survivor count

### 172. Gurobi Installation Succeeds, But The Direct Nonlinear Formulation Is Still Too Heavy (2026-04-09)

`gurobipy` was installed successfully in the `lob_rep` environment, and the
 machine has a working restricted non-production license:

- version: `13.0.1`
- license expiry reported by Gurobi: `2027-11-29`

A small sanity-check model solved correctly, so the solver itself is usable.

Then a direct continuous log-link formulation was tested on the real
 `paper_logical_shared_loglink` screened support:

- variables:
  - nonnegative coefficient vector `a in R_+^34`
  - scalar `u`
- nonlinear constraint:
  - `u = log(sum_i w_i exp(x_i^T a))`
- linear objective:
  - `const + n * u - event_sum^T a`

The real case size was:

- training events: `7455`
- training grid points: `960000`
- screened support dimension: `34`

Important outcome:

- Gurobi accepted the model and the license without error
- however the direct nonlinear expression with `960000` exponential terms was
  already too heavy at model build / early solve time
- after roughly two minutes, the process was still consuming CPU heavily but
  had not yet reached useful presolve / progress output

Interpretation:

- installing an exact nonlinear solver was the right thing to test
- however, for the current full-grid direct expression, the bottleneck is no
  longer the Python branch-and-bound only; the nonlinear expression itself is
  already huge
- therefore `Gurobi + direct full-grid nonlinear expression` is not yet a
  realistic path to "minutes per case"

Most important lesson:

- Gurobi is available and may still be valuable
- but to get a practical exact-solver route, the formulation must be made
  solver-friendly first
- the next mathematically coherent options are:
  - use Gurobi only for smaller certified inner subproblems
  - or derive an exact reformulation / compression that does not expose all
    `960000` grid terms directly to the solver

### 173. Parallel Exact Branch-And-Bound Works, But Alone Is Not Enough (2026-04-09)

A new wrapper partitioned the same exact outer branch-and-bound tree into
 `2^3 = 8` disjoint subtrees and solved them in parallel worker processes.

Important details:

- outer optimization object was unchanged
- each worker solved its subtree exactly with the same certified inner solver
- no heuristic truncation was introduced
- a shared global incumbent upper bound was added so that workers could prune
  against the best solution discovered anywhere in the tree

Case tested:

- `paper_logical_shared_loglink`
- workers: `8`
- split depth: `3`

Observed outcome:

- all 8 workers were fully active and consuming CPU as intended
- however, after about `9` minutes of worker runtime, the solve was still not
  complete and no final JSON had been written

Interpretation:

- parallel exact BB is valid and does reduce wall-clock relative to a purely
  serial traversal
- but it does **not** by itself bring the problem into the desired "few minutes
  per case" regime
- the dominant issue is still structural: each subtree remains too large when
  survivor count is in the `30+` range

Consequence:

- simple exact branch parallelization is not enough
- the next mathematically coherent speed direction must reduce effective search
  size further, e.g. via much stronger theorem-backed screening or much stronger
  exact branch lower bounds

### 174. Quick Timing Decomposition After Active-Set Dual Repair (2026-04-09)

Re-measured `paper_logical_shared` under the current strict
`log-link + exact branch-and-bound + certified inner solve` pipeline.

Deterministic timing breakdown:

- `materialize_loglink_dataset`: `0.009s`
- `setup_problem`: `9.07s`
- `compute_rule_feature_arrays`: `12.35s`
- `build_train_feature_cache`: `0.77s`
- `candidate_scores + Holm`: `0.35s`
- `screened_count`: `34`
- `evaluate_support(full screened support)`: `4.01s`
  - this currently triggers `support_evals=2` because of the coarse/refined
    certified fit stages

So one case already pays about `22.6s` of fixed preprocessing before the outer
tree really starts.

The large remaining cost is still outer node explosion:

- the latest `logical_shared` progress log reached
  `visited_nodes=2020`, `support_evals=1194`, `cache_hits=1008`
  without finishing

Interpretation:

- recent exactness-preserving inner patches do help substantially
- but wall-clock is now dominated by the **number of evaluated supports**, not
  by raw per-support convex optimization alone

### 175. Exact Perspective-Relaxation Node Bound Is Too Expensive As A Speed Patch (2026-04-09)

Tried the most rigorous next bound candidate for additive/log-link exact BB:

- for each screened variable, compute an exact incumbent-based coefficient cap
  by convex feasibility / bisection
- then use those caps inside a perspective relaxation to strengthen the branch
  lower bound

This is mathematically clean, but the quick root-only diagnostic on
`logical_shared_loglink` was already too expensive:

- ran with `8` worker processes
- after about `8` minutes it still had not finished the root-only diagnostic
- no JSON result had been written yet

Interpretation:

- the idea is rigorous, but exact cap computation itself is too heavy to serve
  as the practical speed patch
- so, in the current implementation regime, stronger exact node bounds of this
  type are not the fastest way to reach "minutes per case"

Consequence:

- the next realistic exactness-preserving speed direction should focus on
  reducing survivor count *before* BB, not on expensive exact per-node bound
  tightening

### 176. Quick BIC-Aligned Root Screening Diagnostic Is Not Strong Enough (2026-04-09)

Tried a very cheap screening diagnostic aligned with the exact outer objective:

- for each signed candidate under the root null,
  `z^2 = g^2 / h` was treated as an upper bound on possible reduction in
  `2 * NLL` from a one-coordinate add
- kept a candidate only if `z^2 > log(n_eff)`, i.e. only if even the one-step
  upper bound could pay the BIC complexity price

This is much more objective-aligned than flat Holm p-values and is extremely
cheap to evaluate, so it was a natural candidate for fast survivor reduction.

Observed counts:

- `logical_shared_loglink`
  - Holm survivors: `34`
  - BIC-aligned survivors: `28`
  - true-rule recall under screen: unchanged at `5/7`
- `logical_context_loglink`
  - Holm survivors: `41`
  - BIC-aligned survivors: `38`
  - true-rule recall worsened from `4/7` to `3/7`

Interpretation:

- this root-null BIC-aligned screen is too weak to shrink the family enough
- and, worse, it starts to lose true context-dependent rules
- therefore the next viable speed direction cannot rely on root-only marginal
  score filtering; it must use stronger structured / conditional screening

### 177. Order-By-Order Conditional Screening Is Very Aggressive But Not Truth-Preserving (2026-04-09)

Tested a stronger structured screen for the additive/log-link model:

- go order by order (`1 -> 2 -> 3`)
- at each order, fit the current accepted support
- compute one-sided **conditional efficient-score** tests for candidates at the
  next order
- enforce strong heredity: a higher-order candidate can only be tested if its
  immediate same-sign lower-order subsets have already been accepted
- apply Holm within each order

This is mathematically coherent and very fast to evaluate, but the first
diagnostic shows it is too aggressive:

- `logical_shared_loglink`
  - Holm survivors: `34`
  - stagewise conditional survivors: `7`
  - recall drops from `5/7` to `4/7`

- `logical_context_loglink`
  - Holm survivors: `41`
  - stagewise conditional survivors: `3`
  - recall drops from `4/7` to `3/7`

Interpretation:

- this style of screening *does* solve the runtime side by collapsing survivor
  counts drastically
- but it implicitly imposes a strong-heredity structure that is too strong for
  the current logical benchmarks
- in particular, true contextual rules whose lower-order same-sign ancestors are
  absent or weak get blocked before they can be tested

Consequence:

- structured screening must be **branch-aware** without imposing strong
  same-sign heredity as a hard gate
- the right next object is not "test only descendants of already accepted
  same-sign ancestors", but rather a branch test that keeps a whole ambiguous
  interaction branch alive whenever lower-order evidence is insufficient to
  certify it null

### 178. Branch-Aware Conditional Screening Preserves More Truth But Expands The Leaf Family (2026-04-09)

Tested the next structured screen:

- use a canonical prefix tree over sorted source subsets
- for each signed prefix branch, test the whole descendant branch by a
  conditional group score test
- condition on currently active singleton support
- use hierarchical branch pruning (a child branch is tested only if its parent
  branch survived)

This was designed to avoid the too-strong heredity gate of the previous
order-by-order exact-rule screening.

Observed results:

- `logical_shared_loglink`
  - Holm survivors: `34`
  - branch-leaf survivors: `52`
  - recall improves from `5/7` to `6/7`

- `logical_context_loglink`
  - Holm survivors: `41`
  - branch-leaf survivors: `50`
  - recall improves from `4/7` to `5/7`

Interpretation:

- branch-aware conditional screening does what the previous stagewise screen
  could not: it protects more true contextual / higher-order rules
- but it fails at the runtime objective because the induced leaf family is even
  larger than the flat Holm family

So we now have two complementary failure modes:

- root / flat screens are too loose
- strong-heredity stagewise screens are too aggressive
- branch-aware conditional screens preserve truth better, but are still too
  loose for runtime

Consequence:

- the next screening object must be **between** exact-rule strong-heredity and
  prefix-branch existence
- concretely, it should certify *small ambiguous blocks* rather than whole
  branches or individual rules

### 179. Naive Fisher-Block Graph Collapses To A Giant Component (2026-04-09)

To test the proposed block-factorization direction as quickly as possible, a
diagnostic was run on the additive/log-link model using:

- Holm survivor family as the coarse universe
- full-survivor log-link fit
- observed Fisher matrix on the survivor coordinates
- normalized pairwise couplings
  `|H_ij| / sqrt(H_ii H_jj)`
- edge threshold given by a bounded-feature concentration radius
  `rho_n = sqrt(2 log(p^2 / delta) / n_eff)`
- ambiguity blocks defined as connected components of the resulting graph

Results:

- `logical_shared_loglink`
  - Holm survivors: `34`
  - block count: `1`
  - block sizes: `[34]`

- `logical_context_loglink`
  - Holm survivors: `41`
  - block count: `1`
  - block sizes: `[41]`

Interpretation:

- the **direction** (decompose via Fisher geometry rather than by heuristic
  support gating) still looks conceptually right
- but the naive realization `pairwise Fisher edge -> connected components`
  does not work in these hard cases
- all survivors collapse into a single giant component, so there is no runtime
  benefit

Most likely reason:

- pairwise observed Fisher couplings are too dense once evaluated on the full
  survivor fit
- they do not isolate the truly irreducible ambiguity structure
- transitive chaining then merges everything into one component

Consequence:

- the next viable block-factorization object must be stronger than raw pairwise
  Fisher adjacency
- likely options are:
  - conditional / Schur-complement couplings
  - block graph built after removing low-rank global directions
  - or certified sparse precision-graph style ambiguity diagnostics

### 180. Schur-Complement Coupling With Certified Singleton Core Still Gives A Giant Component (2026-04-09)

Tested the strongest immediate refinement of the Fisher-block idea:

- coarse universe: Holm survivors
- certified core `C`: singleton survivors from the same screen
- ambiguity set `A = survivors \\ C`
- fit the full survivor support under the additive/log-link model
- compute the survivor Fisher matrix `H`
- form the Schur complement
  `H_{A|C} = H_{AA} - H_{AC} H_{CC}^{-1} H_{CA}`
- build the ambiguity graph on `A` using normalized conditional couplings
  `|(H_{A|C})_{ij}| / sqrt((H_{A|C})_{ii}(H_{A|C})_{jj})`
- threshold edges by the same bounded-feature concentration radius

Observed results:

- `logical_shared_loglink`
  - Holm survivors: `34`
  - certified singleton core size: `6`
  - ambiguity size: `28`
  - raw ambiguity graph block sizes: `[28]`
  - Schur-complement graph block sizes: `[28]`

- `logical_context_loglink`
  - Holm survivors: `41`
  - certified singleton core size: `3`
  - ambiguity size: `38`
  - raw ambiguity graph block sizes: `[38]`
  - Schur-complement graph block sizes: `[38]`

Interpretation:

- the user's proposed direction is still conceptually correct: nuisance
  projection is the right next object to test
- however, in the most direct implementation, Schur-complementing out the
  certified singleton core does **not** break the giant component
- the ambiguity structure remains fully dense even after conditioning on the
  obvious low-order core

Consequence:

- `pairwise conditional Fisher + connected components` is still too weak as the
  actual decomposition object
- the next viable refinement must go beyond Schur-on-singleton-core, e.g.:
  - conditional precision graph rather than conditional covariance/Fisher graph
  - low-rank global-mode deflation before graph construction
  - or a small-block construction based on sparse local ambiguity neighborhoods

Clarification:

- the normalized Schur-complement coupling used here is already the same object
  as a **partial-correlation / conditional-precision graph** built from
  `H_{A|C}`
- therefore the non-heuristic precision-graph check has, in effect, already
  been performed
- its negative result means the next step must go beyond pairwise conditional
  precision itself, not merely rename the same graph object

### 181. Canonical Interaction Basis Screening Diagnostic Looks Promising (2026-04-09)

Tested a different line after the Fisher/precision graph decomposition branch
failed:

- keep the additive/log-link model
- replace the raw strict-AND dictionary `phi_U` by a **weighted canonical
  interaction basis** obtained by triangular projection against strict
  lower-order subsets
- then rerun the same root candidate-score + Holm screening diagnostic on the
  transformed basis

Implementation notes:

- the basis was built in subset-size / lexicographic order
- each raw column was projected out of the previously constructed canonical
  columns under the weighted grid inner product
- this was only a **quick proxy diagnostic**: screening labels were still read
  back against the original raw subset/sign identifiers

Observed results:

- `logical_shared_loglink`
  - raw Holm survivors: `34`
  - canonical-basis Holm survivors: `16`
  - raw recall proxy: `5/7 = 0.7143`
  - canonical recall proxy: `6/7 = 0.8571`
  - raw missing proxy:
    - `B and E and F -> T : inhibition`
    - `C and D -> T : inhibition`
  - canonical missing proxy:
    - `B and E and F -> T : inhibition`

- `logical_context_loglink`
  - raw Holm survivors: `41`
  - canonical-basis Holm survivors: `14`
  - raw recall proxy: `4/7 = 0.5714`
  - canonical recall proxy: `7/7 = 1.0`
  - raw missing proxy:
    - `A and C and D -> T : inhibition`
    - `B and C and D -> T : excitation`
    - `E and F -> T : excitation`
  - canonical missing proxy: none

Interpretation:

- unlike the graph-decomposition line, the **basis change itself** appears to
  remove a large amount of the global coupling
- survivor counts dropped sharply (`34 -> 16`, `41 -> 14`) while the recall
  proxy improved rather than degraded
- this is the first fast diagnostic in the log-link setting that simultaneously
  improved both "search size" and "truth coverage"

Important caution:

- this is not yet a full theorem-level result
- the current check is a **proxy** because screening is still read against the
  original raw rule labels after transforming the basis
- the next step is to formalize the canonical basis itself as the selection
  object, then state how raw-rule recovery follows from the triangular inverse
  map under a no-cancellation / margin condition

### 182. Fast Canonical Sparse GLM Prototype Is Computationally Good But Too Conservative (2026-04-12)

Tested the fastest plausible replacement for exponential-size exact subset search:

- keep the additive/log-link model
- use an orthonormalized canonical interaction basis
- fit a **single global convex sparse GLM**
  `avg_NLL + lambda_n * ||alpha||_1`
  with nonnegative signed coefficients
- choose
  `lambda_n = sqrt(2 log(2p) / n_seq_train)`
  using the number of iid training sequences as the sample size

Quick run on `logical_shared_loglink`:

- runtime: `26.86s`
- `n_seq_train = 4000`
- `lambda_n = 0.05050`
- optimizer converged cleanly
- selected active count: `0`
- raw support proxy:
  - `recall = 0.0`
  - `precision = 0.0`

Interpretation:

- this is the first path that is genuinely **fast** without any combinatorial
  outer loop
- however, the naive universal penalty on the current orthonormal canonical
  basis is too conservative in this case and collapses to the empty model
- therefore the "canonical basis + one global convex sparse estimator" line
  remains promising computationally, but it still needs a more faithful
  theorem-backed penalty / normalization / debiasing design before it can serve
  as the main method

### 183. Screened-Canonical Sparse GLM Is Faster But Still Too Conservative On A Hard Case (2026-04-12)

Ran the fastest mixed prototype on `logical_context_loglink`:

- start from canonical Holm survivors only (`14` signed variables)
- fit a single global nonnegative sparse GLM on that screened set
- use the same universal sequence-level penalty
  `lambda_n = sqrt(2 log(2p) / n_seq_train)`

Observed result:

- active count: `2`
  - `A -> T : excitation`
  - `B -> T : inhibition`
- support proxy:
  - `recall = 2/7 = 0.2857`
  - `precision = 1.0`
- optimizer converged cleanly

Interpretation:

- shrinking the optimization domain from all canonical variables to the
  screened canonical set helps computationally, but it does **not** solve the
  main statistical issue
- the current theory penalty still keeps only the strongest low-order signals
  and removes the contextual / higher-order rules
- therefore the next version must change the statistical object (penalty,
  weighting, refit, or de-biasing), not just the search backend

### 184. Mixed Canonical Sparse Fit + Efficient-Score Add-Back Compresses The Hard Case, But Exact Correction Still Dominates Runtime (2026-04-12)

Implemented a mixed prototype:

- canonical Holm screening
- one global penalized canonical sparse fit
- exact refit on the stage-1 active set
- efficient-score add-back over the inactive screened variables
- exact search only on the resulting small union family

Also found an important systems issue:

- the earlier direct runs were silently using multithreaded BLAS and could
  consume dozens of CPU cores for a single case
- capping
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
  made the wall-clock behavior much more interpretable without changing the
  statistical object

Quick diagnostic on `logical_context_loglink` under thread-capped execution:

- canonical Holm screened count: `14`
- stage-1 sparse active count: `2`
  - `A -> T : excitation`
  - `B -> T : inhibition`
- efficient-score add-back count: `11`
- exact-correction union count: `13`
- time to reach this union summary: `36.03s`

Interpretation:

- this is the first mixed selector that compresses the hard case to a
  genuinely modest ambiguity family without any heuristic cap
- the front half of the pipeline is now computationally plausible
- however, even on this reduced family, the final exact-correction stage still
  costs on the order of minutes rather than seconds when implemented naively

Further exactness-preserving implementation updates were attempted:

- replaced the final exhaustive subset enumeration with exact branch-and-bound
- ordered the branch-and-bound variables as
  `stage1_active first + add-back evidence order`
- initialized the exact correction from the stage-1 exact refit incumbent

These changes are correctness-preserving and improve the algorithmic setup, but
they did not yet bring the end-to-end hard-case runtime down to the desired
"few minutes or less" regime in the quick tests run on April 12, 2026.

Current conclusion:

- the mixed line is **much more promising** than raw exact search because it
  creates a small ambiguity family on hard cases
- the remaining bottleneck is now sharply localized to the final exact
  correction over that ambiguity family
- the next speed work should target that exact-correction layer specifically
  rather than the front-end screening/compression stages

### 185. Exact-Correction Speed Patches Still Do Not Close The Hard-Case Loop (2026-04-14)

Re-tested the mixed additive/log-link selector on the hardest currently useful
case, `logical_context_loglink`, with additional exactness-preserving
implementation patches:

- BLAS thread caps
- exact branch-and-bound instead of naive subset enumeration
- ordering the exact-correction variables as
  `stage1-active first + add-back evidence order`
- stage-1 exact refit as the initial incumbent
- single-deletion incumbent tightening on the full union
- child-level lower-bound pruning
- parallel exact branch-and-bound over the final ambiguity family

Measured result:

- front-half compression remained the strongest positive signal
  - canonical screened count: `14`
  - stage-1 active count: `2`
  - add-back count: `11`
  - final exact-correction union count: `13`
  - time to reach this union summary: `36.03s`

But the end-to-end hard-case loop still did not complete quickly:

- the older serial exact-correction version on this `13`-variable union still
  ran for multiple minutes without completing
- the newer stronger-incumbent / stronger-pruning variant did **not** improve
  the wall-clock enough; in direct observation it could remain in the
  pre-parallel tightening phase for about two minutes without reaching the
  worker-parallel correction stage

Interpretation:

- the main gain is real: the mixed selector converts a hopeless global exact
  search into a modest ambiguity family on the hard case
- however, the exact-correction layer remains the bottleneck
- some seemingly principled exactness-preserving tightenings (especially
  aggressive pre-BB incumbent tightening) can actually hurt runtime because
  they enlarge the deterministic pre-search constant before parallel work even
  begins

Current takeaway:

- the mixed selector is statistically more promising than the sparse-only line
- but speed work must now target the **constant factor and repeated support
  evaluation cost** inside exact correction, not merely add more root-level
  exact tightenings

### 186. Literature-Guided Model Choice: Keep Log-Link As The Main Theorem Line, Keep The Old Asymmetric Model Only As A Baseline/Proposal Generator (2026-04-14)

After re-reading the full local notebook and comparing it against primary-source
literature on nonlinear Hawkes processes, point-process GLMs, sparse Poisson /
count regression, and canonical interaction decompositions, the most defensible
mainline choice is:

- **keep the additive / log-link signed model as the main theorem target**
- **do not revert to the old asymmetric multiplicative model as the main
  research line**
- **retain the old asymmetric model only as**
  - an empirical baseline
  - an initialization / proposal generator
  - an ablation showing the value of the new model class

Why this is the right split:

1. Local evidence in our own notebook favors the log-link geometry on the key
   structural ambiguity.

- `### 159` showed that on the hardest known local precision failure
  (`logical_context`), the log-link model chose the true support over
  `true + historical extra`, and the extra inhibition coefficient shrank to
  zero.
- By contrast, the old asymmetric model repeatedly allowed compensatory extras
  to survive.

2. The literature strongly supports nonlinear / canonical-link point-process
   models as the natural setting for inhibition.

- Chen, Shojaie, Shea-Brown, Witten (2017/2019) explicitly study multivariate
  Hawkes processes **beyond mutual excitation**, emphasizing that inhibitory
  relationships and nonlinear links are the right generalization target.
- Sulem, Rivoirard, Rousseau (2021/2023) prove posterior concentration and graph
  consistency for **nonlinear Hawkes** models that include inhibition.
- Bonnet, Martinez Herrera, Sangnier (2022/2023) give identifiability and exact
  frequentist inference for exponential Hawkes models with inhibition.

3. The literature on efficient sparse estimation is much more mature for convex
   GLM / Poisson / point-process likelihoods than for our old custom
   multiplicative parameterization.

- Paninski (2004) and the broader PP-GLM literature support canonical-link
  positive-intensity models with convex likelihood structure.
- Hansen, Reynaud-Bouret, Rivoirard (2015) and Ivanoff, Picard, Rivoirard
  (2016) show how concentration-based weighted Lasso / group-Lasso can be made
  theorem-friendly for counting-process / Poisson settings.
- This matches the current computational need: we need a sparse, theorem-backed
  front end that avoids exponential raw subset search.

4. The old asymmetric model keeps one practical advantage: it is empirically
   search-friendly because it suppresses many candidate rules. But this
   sparsifying effect is partly an artifact of the geometry itself.

- As analyzed in `### 155`, the old model
  `lambda(t) = (mu + E(t)) exp(-I(t))`
  makes inhibition suppress the evidence for its own detection.
- So the fact that it leaves smaller candidate families is not pure good news:
  it also bakes in the very under-selection / compensation pathology that drove
  the model change.

5. The most promising new ingredient is not graph decomposition but the
   canonical interaction basis.

- `### 181` is the first fast diagnostic that simultaneously reduced family size
  and improved truth coverage.
- This aligns well with the interaction-purification / functional-ANOVA
  literature, where pure interaction components are used to remove lower-order
  confounding before interpretation or selection.

Recommended research program from here:

Stage A: formalize the **canonical interaction basis** as the primary selection
object under the log-link model.

- Selection is performed on canonical coefficients, not raw strict-AND
  coefficients.
- Raw-rule recovery is then phrased through the triangular inverse map and a
  no-cancellation / margin condition.

Stage B: replace pure subset search by a **theorem-derived sparse convex front
end** on the canonical basis.

- Use concentration-based weighted penalties or debiased one-step inference,
  rather than ad hoc tuning.
- The goal is not to make the sparse fit itself the final answer, but to
  produce:
  - certified core
  - certified zero
  - small ambiguity set

Stage C: keep exactness only where it is affordable.

- Run exact correction only on the ambiguity set.
- This preserves a rigorous finite-family exact-selection step without forcing
  the entire problem through a global `2^p` search.

Stage D: keep the old asymmetric model only as a supporting line.

- It remains useful as:
  - a benchmark to beat
  - a warm-start / incumbent source
  - a demonstration that faster search can come from biased geometry, not from
    better identifiability

Bottom line:

- the old asymmetric model is still valuable operationally
- but the additive / log-link model is the better theorem target
- and the next serious mainline should be
  **canonical-basis sparse front end + certified core/zero/ambiguity split +
  exact ambiguity correction**

### 187. Weighted Self-Normalized Canonical Sparse Front End: Strong New Signal, But Same-Sample Active-Core Lock-In Fails On The Hard Case (2026-04-14)

To make the canonical sparse front end less conservative without introducing
heuristic tuning, tested a weighted nonnegative `L1` fit on the canonical basis
using coordinate-wise penalties

`pen_j = sqrt(2 log(2p)) * sqrt(H_jj(null)) / n_seq_train`

where `H_jj(null)` is the null Fisher / score-variance diagonal for canonical
coordinate `j`.

This is a natural self-normalized, theorem-friendly alternative to the earlier
uniform sequence-level penalty.

#### 187.1. Hard-Case Front-Half Result (`logical_context_loglink`)

Observed quick diagnostic:

- canonical Holm screened count: `14`
- weighted sparse stage-1 active count: `12`
- stage-1 proxy metrics:
  - `recall = 1.0`
  - `precision = 0.5833`
- inactive add-back count after efficient-score Holm: `0`

This is the first sparse front end that:

- remains very fast
- keeps **all** true rules in the hard case at stage 1
- and removes the need for inactive add-back on that case

So the weighted self-normalized penalty is a real statistical improvement over
the earlier uniform penalty line (`### 182` / `### 183`).

#### 187.2. Active-Side Wald / Holm Split

Using an exact refit on the `12` active rules and one-sided Wald p-values:

- Holm-certified active core size: `9`
- active ambiguity size: `3`

The three weak active coordinates were:

- `A and C and D -> T : inhibition`
- `B and C -> T : excitation`
- `E and F and G -> T : inhibition`

This means the active side can, in principle, be compressed from `12` to only
`3` combinatorial degrees of freedom after a same-sample refit.

#### 187.3. Exact Correction On The 3-Variable Ambiguity Family

Then fixed the `9` active-core coordinates, allowed only the `3` ambiguous
actives to vary, and exactly enumerated the resulting `2^3 = 8` support family.

Result:

- exact-correction family size: `8`
- support evaluations: `16`
- elapsed: about `168.1s`
- final metrics:
  - `recall = 0.8571`
  - `precision = 0.5455`

Final support still kept multiple extras and dropped one true inhibition rule.

#### 187.4. Interpretation

This reveals an important distinction:

1. the **weighted self-normalized sparse front end is genuinely better**
   - it fixes the earlier empty/underfit stage-1 problem
   - it produces a fast high-recall active set

2. but **same-sample active-core certification is not safe enough**
   - several false/proxy-extra active coordinates receive extremely large Wald
     z-scores on the same data used to construct the active set
   - once these are locked into the core, the final exact correction can no
     longer repair them

Current takeaway:

- the weighted canonical sparse front end should be kept as a promising new
  ingredient
- but active-core certification must move to a more conservative object:
  likely
  - sample-split / holdout certification
  - or another exact objective-aligned active-core test
- otherwise the method becomes fast but locks the wrong actives too early

### 188. Holdout-Certified Active Core Avoids Lock-In But Becomes Too Conservative On The Hard Case (2026-04-14)

Tested the next strict idea immediately after `### 187`:

- use the weighted self-normalized canonical sparse front end on train
- but certify the active core on **holdout / validation data** in the raw
  support space
- then run exact correction only over the holdout-certified active ambiguity
  family

This removes the same-sample inflation that made the active core too sticky in
`### 187`.

Observed result on `logical_context`:

- train canonical screened count: `14`
- train weighted stage-1 active count: `12`
- holdout active core count: `3`
  - `C -> T : inhibition`
  - `B -> T : inhibition`
  - `E and F -> T : excitation`
- holdout active ambiguity count: `9`
- exact correction family size: `2^9 = 512`
- support evaluations: `1024`
- elapsed: about `238.2s`
- final metrics:
  - `recall = 0.5714`
  - `precision = 0.8`

Interpretation:

- holdout certification does fix the earlier "false active core gets locked in"
  problem
- but on this hard case it swings too far in the other direction:
  - the active core becomes very small
  - the ambiguity family becomes much larger again
  - several true contextual / higher-order rules are no longer stably carried
    through to the final solution

So we now have a clean contrast:

- same-sample active-core certification:
  - fast ambiguity family
  - but false core lock-in
- holdout active-core certification:
  - less lock-in
  - but ambiguity explodes and recall drops

Current takeaway:

- the weighted self-normalized canonical front end is still the best new
  ingredient discovered on April 14
- but "certify active core first, then exact-correct the rest" is not yet the
  right decomposition, either in same-sample or holdout form
- the next exactness-preserving step should likely avoid a hard active-core
  split and instead seek a **small ambiguity set directly**, without freezing a
  large active block too early

### 189. A Two-Stage Adaptive Canonical Sparse Fit Improves Precision Slightly, But Not Nearly Enough (2026-04-14)

As a faster theorem-friendly alternative to the stalled exact-correction layer,
I tested a pure convex two-stage sparse procedure on the `logical_context`
hard case:

1. stage 1: weighted self-normalized canonical sparse fit
2. stage 2: adaptive lasso on the stage-1 active coordinates only, using
   weights proportional to `base_weight / max(alpha_stage1, 1e-6)`

This keeps the same additive/log-link and canonical-basis line, and replaces
the expensive combinatorial correction by one more convex fit.

Observed result:

- elapsed: about `83.76s`
- stage-1 active count: `12`
  - metrics:
    - `recall = 1.0`
    - `precision = 0.5833`
- stage-2 active count: `11`
  - metrics:
    - `recall = 1.0`
    - `precision = 0.6364`

Concretely, the second stage removed one false extra
(`C -> T : inhibition`) but left four other extras:

- `A and B -> T : inhibition`
- `A and B and D -> T : excitation`
- `A and F -> T : excitation`
- `B and C -> T : excitation`

Interpretation:

- the weighted canonical front end still looks like the right backbone
- a second adaptive convex stage does help a little
- but on the hardest context case, it is still far from enough to reach
  exact recovery
- so the remaining problem is not "sparse fit vs exact correction" in the
  abstract; it is that the current stage-2 penalty geometry still does not
  separate true contextual interactions from compensatory extras sharply
  enough

Current takeaway:

- pure convex `stage1 + adaptive stage2` is much faster than exact correction
- but it is still not accurate enough
- the next promising direction is likely a **sample-split / holdout version of
  stage 2**, or a lighter exact local certificate layer built only around the
  stage-2 ambiguity coordinates

### 190. Holdout Adaptive Stage-2 Is Too Conservative On The Hard Case (2026-04-14)

To test the most natural sample-split variant of `### 189`, I kept:

- train stage 1: weighted self-normalized canonical sparse fit

and then compared two second stages on the stage-1 active coordinates:

1. same-sample adaptive lasso on train
2. holdout adaptive lasso on validation, using the same stage-1 support but
   fitting in the raw validation space

Observed result on `logical_context`:

- same-sample adaptive stage 2:
  - active count: `11`
  - `recall = 1.0`
  - `precision = 0.6364`
- holdout adaptive stage 2:
  - active count: `6`
  - `recall = 0.5714`
  - `precision = 0.6667`

So the holdout version does remove several extras, but it also drops true
contextual interactions too aggressively:

- missing:
  - `A and C and D -> T : inhibition`
  - `B and C and D -> T : excitation`
  - `E and F -> T : excitation`

Interpretation:

- sample splitting by itself is not the missing ingredient
- on this hard case, adaptive shrinkage on holdout data becomes too
  conservative
- this reinforces the earlier picture:
  - pure convex sparse stages are fast
  - but they still do not separate contextual / compensatory coordinates
    sharply enough to replace a final exact or certified-local correction

Current takeaway:

- same-sample adaptive stage 2 is the better of the two convex-only variants
  tried so far
- but even that only reaches `precision = 0.6364`
- so the next step should not be "more shrinkage", but a **small,
  objective-aligned local certificate / correction layer** on top of the
  strong weighted canonical front end

### 191. Local Certified Correction Around Stage-2 Support Still Fails To Separate Truth From Extras Cleanly (2026-04-14)

I pushed one level deeper and tested three "small local correction" ideas on
top of the same-sample adaptive stage-2 support for `logical_context`.

Base support going into local correction:

- 11 active rules
- `recall = 1.0`
- `precision = 0.6364`

#### 1. Holdout local certified single-deletion scan

Using the validation raw objective with certified BIC intervals, I checked
whether any single deletion was *provably* better than the current support.

Result:

- elapsed: about `91.7s`
- certified beneficial deletions: `0`
- final support unchanged
- final metrics unchanged:
  - `recall = 1.0`
  - `precision = 0.6364`

So the coarse local certificate was too weak to remove any extras.

#### 2. Refined holdout deletion scan

When the same deletion scan is tightened with refined certified fits, the
result swings too far in the other direction:

- elapsed: about `95.7s`
- certified beneficial deletions: `8`
- but these include several **true** contextual rules:
  - `E and F -> T : excitation`
  - `A and C and D -> T : inhibition`
  - `B and C and D -> T : excitation`
  - `E and F and G -> T : inhibition`

So holdout local deletion is not safely aligned with exact recovery here.

#### 3. Refined train deletion scan

On the train raw objective, the behavior is the opposite:

- elapsed: about `243.4s`
- certified beneficial deletions: `0`

Thus:

- train local deletion is too optimistic and removes nothing
- holdout local deletion is too pessimistic and wants to remove true rules

#### 4. Holdout active-side Wald/Holm on the 11-rule stage-2 support

I also tested a holdout Wald/Holm filter directly on the 11 active
coefficients:

- elapsed: about `84.8s`
- accepted active rules: `3`
- final metrics:
  - `recall = 0.4286`
  - `precision = 1.0`

Again, the holdout side is too conservative for contextual / higher-order
truths.

Current takeaway:

- same-sample stage 2 is still the best *fast* support compressor found so far
- but neither local holdout deletion nor holdout Wald/Holm can safely be used
  as the final selector
- the most natural next theorem-friendly direction is now a
  **cross-fit ambiguity band**:
  - train-side strong evidence keeps recall high
  - holdout-side weakness prevents false actives from being frozen as core
  - disagreement is sent to a small ambiguity set, instead of being
    immediately dropped or immediately certified

### 192. Cross-Fit Ambiguity Band + Pooled Exact Selection Finally Solves The Hard Case (2026-04-14)

I implemented the next natural step directly:

1. train-side weighted canonical sparse front end
2. same-sample adaptive stage-2 on train
3. holdout Wald/Holm only to define the **strong core**
4. ambiguity set = train stage-2 support minus holdout-certified core
5. final exact selection is run only over this small ambiguity family

Two final objectives were compared on the same band:

- holdout-only exact selection
- pooled train+holdout exact selection

Observed result on `logical_context`:

- train stage-2 support:
  - `11` rules
  - `recall = 1.0`
  - `precision = 0.6364`
- holdout-certified core:
  - `3` rules
  - ambiguity size: `8`

Final exact selection over the `2^8 = 256` ambiguity family:

- holdout-only exact selection:
  - `recall = 0.4286`
  - `precision = 1.0`
- pooled exact selection:
  - `recall = 1.0`
  - `precision = 1.0`
  - exact final support:
    - `A -> T : excitation`
    - `E and F -> T : excitation`
    - `B and C and D -> T : excitation`
    - `B -> T : inhibition`
    - `E -> T : inhibition`
    - `A and C and D -> T : inhibition`
    - `E and F and G -> T : inhibition`

Timing / search stats:

- total elapsed: about `314.4s`
- pooled exact BB stats:
  - `visited_nodes = 119`
  - `pruned_nodes = 124`
  - `support_evals = 154`
  - `cache_hits = 189`

Interpretation:

- this is the first theorem-friendly line on April 14 that simultaneously:
  - preserves recall
  - restores precision to `1.0`
  - and keeps the final exact search on a genuinely small family
- the key was not "train only" or "holdout only"
- the key was:
  - use train to keep the contextual truths alive
  - use holdout only to prevent false actives from becoming permanent core
  - then run exact selection on the resulting small ambiguity family using the
    pooled exact objective

Current best direction:

- keep `log-link + canonical basis`
- keep the weighted self-normalized stage-1 front end
- keep adaptive stage-2 as a fast support compressor
- define a **cross-fit ambiguity band**
- run exact selection only inside that small band, with pooled exact
  objective on the final family

### 193. The Cross-Fit Ambiguity Band Generalizes Unevenly Across Hard Cases (2026-04-14)

I next checked how the same structure behaves beyond `logical_context`.

#### `logical_context`

- cross-fit band + pooled exact selection still succeeds
- final metrics:
  - `recall = 1.0`
  - `precision = 1.0`
- ambiguity size: `8`

So this hard case is now genuinely in reach with the current line.

#### `kernel_gaussian`

A lighter stage-only diagnostic (before the final exact search) shows why the
same method is still slow there:

- screened count: `78`
- stage-2 active count: `22`
- stage-2 metrics:
  - `recall = 1.0`
  - `precision = 0.2727`
- holdout-certified core count: `6`
- ambiguity count: `16`

Interpretation:

- the front end still keeps the truth
- but it leaves far too many compensatory extras alive
- the ambiguity band is not yet small enough for the final exact stage to be
  fast or reliable

#### `logical_clean_plus`

Even the lighter stage-only diagnostic remained slow enough that it was not
worth waiting for completion in the same iteration. This suggests that, on
that case, the bottleneck may occur even earlier in the weighted sparse front
end rather than only in the final exact correction.

Current takeaway:

- the new line is now **validated on the hardest contextual case**
- but it has not yet generalized cleanly to the other hard failures
- the next bottleneck to solve is no longer "can the ambiguity-band idea
  work?"; it is "how do we shrink the band much more aggressively on the large
  kernel / clean-plus failures without losing theorem-friendly coverage?"

### 194. Symmetric Cross-Fit Stage-2 Union + Pooled Strong Core Is Not Better (2026-04-14)

I tested a more symmetric split-based variant:

- run the weighted canonical stage-1 plus adaptive stage-2 on **train**
- run the same weighted canonical stage-1 plus adaptive stage-2 on
  **holdout**
- define the candidate family by the stage-2 **union**
- define the strong core by a **pooled Wald/Holm** screen on that union
- send only the remaining ambiguity set to the final exact layer

This keeps the procedure theorem-friendly in spirit:

- no heuristic thresholds
- fixed split symmetry
- a pooled formal testing step for the strong core

#### `logical_context`

The pre-exact diagnostic was:

- train stage-2 count: `11`
- train stage-2 metrics:
  - `recall = 1.0`
  - `precision = 0.6364`
- holdout stage-2 count: `3`
- holdout stage-2 metrics:
  - `recall = 0.2857`
  - `precision = 0.6667`
- stage-2 union count: `12`
- pooled strong core count: `4`
- ambiguity count: `8`

Interpretation:

- this does **not** improve on the earlier cross-fit ambiguity-band line
- the ambiguity size remains `8`
- holdout stage-2 is still too conservative on the contextual case
- so the extra split symmetry does not buy us a smaller exact family

#### `kernel_gaussian`

The same diagnostic behaved even worse:

- train stage-2 count: `22`
- train stage-2 metrics:
  - `recall = 1.0`
  - `precision = 0.2727`
- holdout stage-2 count: `22`
- holdout stage-2 metrics:
  - `recall = 0.8333`
  - `precision = 0.2273`
- stage-2 intersection count: `17`
- stage-2 union count: `27`
- pooled strong core count: `9`
- ambiguity count: `18`

Interpretation:

- the symmetric split variant is actually **worse** than the earlier line on
  `kernel_gaussian`
- the stage-2 union inflates the candidate family from the earlier ambiguity
  `16` to `18`
- so this variant should **not** be pursued as the main line

Current conclusion:

- the best line remains:
  - train weighted stage-1
  - train adaptive stage-2
  - holdout only for strong-core certification
  - pooled exact selection on the remaining ambiguity band
- the new symmetric split-stage2 union + pooled-core idea is a useful
  negative result, but it does not improve either the ambiguity size or the
  final search cost

### 195. Train Stage-2 + Pooled Strong Core Is Better Than Holdout Core (2026-04-14)

I next removed the extra split-stage2 symmetry and instead asked a simpler
question:

- keep the original strong train-side support compressor
- keep the candidate family equal to the **train stage-2 support**
- define the strong core by a **pooled** Wald/Holm test on that train stage-2
  support
- send only the remaining active coordinates to the exact final stage

This is still theorem-friendly in the same general sense:

- no heuristic thresholds
- one fast front-end support compressor
- a formal pooled inferential step for the strong core
- exact search only on the remaining ambiguity

#### `logical_context`

Stage-only diagnostic:

- screened count: `14`
- train stage-2 active count: `11`
- train stage-2 metrics:
  - `recall = 1.0`
  - `precision = 0.6364`
- pooled strong core count: `4`
  - `A -> T : excitation`
  - `E and F -> T : excitation`
  - `B -> T : inhibition`
  - `E -> T : inhibition`
- ambiguity count: `7`

This improves on the earlier holdout-core ambiguity band:

- old ambiguity: `8`
- new ambiguity: `7`

So on the contextual hard case, pooled strong-core certification appears to be
strictly better than holdout-only strong-core certification.

#### `kernel_gaussian`

Stage-only diagnostic:

- screened count: `78`
- train stage-2 active count: `22`
- train stage-2 metrics:
  - `recall = 1.0`
  - `precision = 0.2727`
- pooled strong core count: `11`
- ambiguity count: `11`

This is the first useful reduction on the Gaussian hard case within the new
line:

- earlier holdout-core ambiguity: `16`
- new pooled-core ambiguity: `11`

So the pooled-core variant is materially better on the Gaussian case as well.

#### Remaining issue

When I tried the exact final search on the new `logical_context` pooled-core
band, the ambiguity was smaller but the exact process still did not finish
quickly, and the current worker-spawning wrapper left behind too many forked
processes before cleanup.

Interpretation:

- **front-end compression is improving in the right direction**
- the new pooled strong-core idea appears better than the earlier holdout-core
  idea on both `logical_context` and `kernel_gaussian`
- but the final exact executor itself still needs a cleaner and cheaper
  implementation before this line can be used end-to-end at the target speed

Current best research direction after this update:

- keep `log-link + canonical basis`
- keep train weighted stage-1
- keep train adaptive stage-2
- replace holdout-only strong core by **pooled strong core on the train
  stage-2 support**
- then focus optimization effort on the exact ambiguity solver itself

### 196. Small-Family Exactness Is No Longer a Search-Tree Problem (2026-04-14)

I followed the new best front end one step further and explicitly changed the
final exact layer for small ambiguity families.

#### Change

For small ambiguity sets (`m <= 12`), I replaced the previous fork-heavy exact
branch-and-bound wrapper by a **serial small-family exact enumerator**:

- enumerate the entire ambiguity family exactly
- use descending-size order so each support can warm start from a cached
  superset
- add a simple certification loop:
  - coarse certified fits for all supports
  - then refine only the support with best upper bound and any remaining
    unresolved competitors whose lower bound is still below that best upper

This keeps the outer selection theorem-friendly:

- no heuristic truncation
- full finite-family exact comparison
- only certificate-driven refinement of ambiguous competitors

#### Additional implementation finding

The first serial runs were still misleadingly slow because NumPy / BLAS was
quietly using many cores inside a single process. Re-running with

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

removed that artifact and confirmed that the final layer was genuinely serial.

#### `logical_context`

Using the improved front end:

- pooled strong core count: `4`
- ambiguity count: `7`

So the final exact family size is only:

- `2^7 = 128`

This is already a meaningful reduction from the earlier ambiguity `8`.

However, even with:

- the smaller family
- a single clean serial process
- the new small-family exact enumerator
- BLAS thread caps

the exact run still did **not** finish quickly enough; after more than
`5 minutes`, the final JSON had still not been written.

Interpretation:

- the remaining bottleneck is **no longer the combinatorial search tree**
- the remaining bottleneck is now the **per-support certified fit itself**
- in particular, the repeated dual-repair / certified optimization inside
  `evaluate_support(...)` is still too expensive even once the ambiguity band
  is very small

This is actually a useful narrowing of the problem:

- front-end compression has improved substantially
- exact family size has improved substantially
- the next speed bottleneck is now sharply localized to the certified support
  solver, not to the ambiguity-band idea itself

Current next direction:

- keep the new front end (`train stage2 + pooled strong core`)
- stop spending research time on alternative ambiguity constructions for now
- instead, redesign the **certified support solver** so that:
  - most supports get a much cheaper certificate
  - only a tiny number of near-incumbent supports pay for the more expensive
    dual-repair refinement

### 197. Exact-Layer Bottleneck Is the Certified Support Solver, Not the Ambiguity Family (2026-04-14)

I pushed this one step further on `logical_context` with the new best front
end:

- train weighted stage-1
- train adaptive stage-2
- pooled strong core on the train stage-2 support

This gives:

- ambiguity count: `7`
- exact family size: `2^7 = 128`

I then made two implementation-side improvements to the final exact layer:

1. **small-family exact enumeration**
   - for `m <= 12`, switch from the old branch-and-bound wrapper to a serial
     exact finite-family search
   - keep exactness by enumerating the whole family
   - use cached superset warm starts

2. **cheaper coarse certificates**
   - separate coarse and refined dual-repair budgets
   - lower the minimum active-set optimization budget inside
     `repair_dual_distribution`
   - force `OMP/MKL/OPENBLAS/NUMEXPR = 1` so the exact layer is truly
     single-process and easier to profile

What happened:

- the process became clean and interpretable:
  - one single exact process
  - no runaway fork explosion
  - no hidden multi-core BLAS oversubscription
- but the final exact run still did not finish within a few minutes

Interpretation:

- once the ambiguity family is as small as `128` supports, the remaining wall
  clock is no longer being driven by the combinatorial search space itself
- it is being driven by the **cost of the per-support certified fit**
- in particular, `evaluate_support(...)` / `certified_fit_support(...)`
  remains the dominant bottleneck even after:
  - ambiguity compression
  - exact-family simplification
  - and thread-cap cleanup

So the current problem has become much sharper:

- **family construction is no longer the main blocker**
- **search-tree design is no longer the main blocker**
- the next speed breakthrough must come from a cheaper theorem-friendly
  certificate for individual supports

Concretely, the next promising research direction is:

- keep the present front end unchanged
- replace the current support certifier by a two-tier certificate:
  - an ultra-cheap valid lower certificate for most supports
  - a stronger dual-repair certificate only for the tiny unresolved set near
    the incumbent

### 198. On `kernel_gaussian`, Hard-Fixing a Core Blocks Exact Recovery (2026-04-14)

While prioritizing accuracy first, the structural issue on
`kernel_gaussian` became much clearer.

The true support on this case is:

- excitation:
  - `A -> T : excitation`
  - `C and D -> T : excitation`
  - `C and G and H -> T : excitation`
- inhibition:
  - `B -> T : inhibition`
  - `E and F -> T : inhibition`
  - `D and F and G -> T : inhibition`

But both currently tested "strong core" definitions include false extras.

#### Holdout core

The holdout-certified core contains:

- `A -> T : excitation`
- `C and D -> T : excitation`
- `C and G -> T : excitation`
- `G and H -> T : excitation`
- `B -> T : inhibition`
- `F -> T : inhibition`

So the holdout core already freezes false rules such as:

- `C and G -> T : excitation`
- `G and H -> T : excitation`
- `F -> T : inhibition`

#### Pooled core

The pooled-certified core is even less safe:

- `A -> T : excitation`
- `C -> T : excitation`
- `A and D -> T : excitation`
- `C and D -> T : excitation`
- `C and G -> T : excitation`
- `C and H -> T : excitation`
- `G and H -> T : excitation`
- `B -> T : inhibition`
- `E -> T : inhibition`
- `F -> T : inhibition`
- `E and F -> T : inhibition`

This freezes many more false extras.

#### Consequence

So the current architecture

- compress support with the front end
- hard-fix a "strong core"
- exact-search only on the remaining ambiguity

is fundamentally unsafe on `kernel_gaussian`.

Even a perfect final exact solver could not recover `100/100` once false
extras have been frozen into the core.

Current implication:

- the immediate accuracy blocker is **false core fixation**
- not merely a slow exact solver

So the next theorem-friendly direction must be:

- use the front end only to define a small candidate family
- but treat current core-like rules as **reviewable**
  unless they pass a much stronger non-removability certificate

In short:

- on `kernel_gaussian`, the next accuracy step is **not** "speed up exact
  search on the current core"
- it is "stop freezing unsafe core rules"

### 199. Single-Rule Removability Certificates Are Too Weak; Ancestor Blocks Show the Right Failure Mode (2026-04-14)

To move beyond unsafe hard-fixed cores while staying theorem-friendly, I
tested a new "branch-removable" idea on the current best front end:

- `log-link + canonical basis`
- `train weighted stage1`
- `train adaptive stage2`
- no hard-fixed core in the deletion diagnostic itself

The object was:

- build the current active stage-2 support
- fit the pooled objective on that support
- use the pooled fit's dual lower value to define a valid branch lower bound
- test whether deleting a rule (or structured block) yields an upper BIC
  below that branch lower bound

This is theorem-friendly in spirit because it tries to certify **safe
removability**, not heuristic pruning.

#### Stage-2-only single-rule deletion: `logical_context`

Using the saved stage-2 support from
`tmp_train_stage2_pooled_core_logical_context_stageonly_20260414.json`:

- active stage-2 support size: `11`
- stage-2 metrics: `recall=1.0`, `precision=0.6364`
- elapsed for coarse-only deletion scan: `159.66s`
- branch lower include: `-inf`
- cert removable count: `0`

So on `logical_context`, the line fails immediately because the full
stage-2 support does not yield a finite dual lower certificate. Without a
finite lower bound, the branch-removable test cannot certify anything.

#### Stage-2-only single-rule deletion: `kernel_gaussian`

Using the saved stage-2 support from
`tmp_train_stage2_pooled_core_kernel_gaussian_20260414.json`:

- active stage-2 support size: `22`
- stage-2 metrics: `recall=1.0`, `precision=0.2727`
- elapsed for coarse-only deletion scan: `687.39s`
- branch lower include: `100295.95`
- cert removable count: `0`

So even when the full support has a finite lower certificate, **single-rule
deletion** is still too weak:

- every one-rule deletion remained unresolved
- no active rule, true or false, could be safely removed by this test

This is a strong sign that the false active rules are not acting
independently. They survive as a **coupled cluster**, so removing only one
of them at a time does not create enough objective improvement to certify
safe deletion.

#### Structured block diagnostic: proper-ancestor blocks on `kernel_gaussian`

To check whether the failure is genuinely cluster-like, I ran a structured
block diagnostic on the same stage-2 support:

- block family: for each active rule, remove its active same-sign proper
  ancestors as a block
- candidate block count: `12`
- elapsed: `315.23s`
- cert removable count: `0`

So the theorem-grade certificate still did not fire. However, one block
gave an important objective signal:

- deleting the block `{C -> exc, H -> exc}` reduced pooled upper BIC by
  about `45.45`

while preserving all currently active true stage-2 rules outside that
block. This means:

- the current false extras are indeed behaving as a **small ancestor
  clutter cluster**
- the right object is not "single rule deletability"
- it is "small structured block removability"

#### Consequence

This is the first clear evidence that the next accuracy-oriented exact
layer should be:

- not hard-fixed cores
- not single-rule removability
- but **reviewable structured blocks**, especially ancestor-style blocks in
  the canonical lattice

The right next theorem-friendly direction is therefore:

- build a finite family of reviewable blocks from the active canonical
  support
- certify safe block removals when possible
- send only the unresolved blocks to the final exact selection layer

In short:

- **single deletions are too weak**
- **false extras are coupled**
- **small structured block removability is the next serious candidate**

### 200. Ancestor-Block Families Reduce Search Size but Still Miss the True Support; Naive Closure-State Families Also Fail Coverage (2026-04-14)

I pushed the new structured-block line one step further by checking whether
the finite family induced by the tested blocks actually contains the truth.

#### `kernel_gaussian`: ancestor-block family

Using the `proper_ancestor` block family from
`tmp_branch_removable_kernel_gaussian_ancestorblocks_coarse_20260414.json`:

- stage-2 active support size: `22`
- candidate block count: `12`
- distinct supports induced by removing any subset of blocks: `91`

So the family is computationally attractive.

However, the crucial combinatorial fact is:

- **the exact true support is not in this 91-support family**

Even worse, among all supports in this family with full recall, the best
precision is only about `0.4286`.

So although ancestor blocks capture some real clutter geometry, the current
block family is still too coarse: it does not include the actual oracle
support.

#### Why it fails

The current ancestor blocks remove the wrong objects together.

Example:

- false higher-order clutter such as `A and C and D -> excitation` should be
  removable
- but the corresponding ancestor block also removes true rules such as
  `A -> excitation` or `C and D -> excitation`

So the family is finite and small, but it is **not truth-preserving**.

#### Naive closure-state family is not the answer either

I also checked a more ambitious representation idea:

- for each active maximal rule, enumerate antichain states inside its active
  downward closure
- then take unions of these local closure states

The quick combinatorial scan showed:

- `logical_context`: raw product state count `972`
  - dedup family `756`
  - **true support not in family**
- `kernel_gaussian`: raw product state count already `1,270,080`
  - too large even before deduplication

So the most naive "closure-state" representation fails in two ways:

- it still misses the truth on `logical_context`
- and it explodes combinatorially on `kernel_gaussian`

#### Consequence

This rules out two tempting next steps:

- ancestor-block exact family, as currently defined
- naive maximal-closure antichain state family

The next direction should instead be:

- keep the strong front end
- but generate a **small deterministic support family from a regularization
  path**, not from arbitrary subset search or naive closure states
- then perform exact pooled selection over that finite path family

Why this is promising:

- it stays theorem-friendly as a finite-family selector
- the path family can be made deterministic by using a fixed grid of theory
  penalty multipliers
- it avoids unsafe hard-fixed cores
- and it avoids the combinatorial explosion of raw subset search

### 201. Deterministic Path Families Need Enrichment; One-Flip Enrichment Helps Coverage but Still Misses Truth, and Path-Intersection Cores Are Unsafe on `kernel_gaussian` (2026-04-14)

I tested two deterministic path-family follow-ups.

#### 1. Deterministic path family alone

Using a fixed penalty grid and taking the distinct train/holdout stage-2
supports as a finite family:

- `logical_context`
  - result file:
    `tmp_stage2_path_family_logical_context_20260414.json`
  - family size: `3`
  - elapsed: `165.33s`
  - best metrics: `recall=0.7143`, `precision=0.8333`
- `kernel_gaussian`
  - result file:
    `tmp_stage2_path_family_kernel_gaussian_20260414.json`
  - family size: `5`
  - elapsed: `403.70s`
  - best metrics: `recall=1.0`, `precision=0.4`

So path-family selection alone is computationally attractive, but the family
coverage is too weak.

#### 2. Path family + deterministic local Holm add/drop band on `logical_context`

I then enriched each base support deterministically:

- weak active rules = not Holm-accepted among active coefficients
- add candidates = Holm-accepted one-sided pooled score additions

First I tried a one-flip neighborhood (drop one weak active, add one
candidate, or swap one-for-one). That still failed:

- best result had `recall=0.8571`, `precision=1.0`
- the missing true rule was
  `A and C and D -> T : inhibition`

Then I widened the local family to the **full local band**:

- for each base support, fix the strong active set
- let the uncertain band be
  `weak_active ∪ add_candidates`
- include every support of the form
  `strong_active ∪ subset(uncertain_band)`

On `logical_context` this gave:

- initial path family size: `4`
- enriched family size: `46`
- elapsed: `608.67s`
- best metrics: `recall=0.8571`, `precision=1.0`

So the richer local band improved coverage relative to the raw path family,
but still missed the same true inhibition rule.

#### Why this still fails

The failure mode is now clear:

- the local band still fixes some active rules too early
- in particular, a false excitation such as
  `A and F -> T : excitation`
  can remain fixed inside the local family around the one base support that
  also exposes the missing true inhibition
- therefore the true support is still absent from the family

So:

- local deterministic enrichment helps
- but **strong/weak active classification is still too local**

#### Path-band exact selector

I also checked a simpler deterministic exact selector:

- path supports from the fixed penalty grid
- `core = intersection(path supports)`
- `ambiguity = union(path supports) \\ core`
- exact selection over that finite path band

On `logical_context` the path-band geometry was:

- path family count: `4`
- core count: `4`
- ambiguity count: `7`

This is computationally attractive and truth-covering in principle, but the
exact run still spent many minutes in the certified support solver.

More importantly, on `kernel_gaussian`, the path-family supports from
`tmp_stage2_path_family_kernel_gaussian_20260414.json` imply:

- path family count: `5`
- path intersection core count: `13`

and that intersection core already contains many false extras, such as:

- `C -> T : excitation`
- `A and D -> T : excitation`
- `C and G -> T : excitation`
- `C and H -> T : excitation`
- `G and H -> T : excitation`
- `A and E and F -> T : inhibition`

So the naive `intersection(path supports)` core is not generally safe.

#### Consequence

This rules out the next two naive path-based variants as main lines:

- path family alone
- path-intersection core + exact ambiguity search

and weakens the fully local strong/weak band idea.

The next deterministic theorem-friendly direction should therefore be:

- use the path family only as a **proposal family**
- but define reviewable coordinates using a **global path-stability object**
  rather than per-support local strong/weak decisions
- in particular, avoid hard-fixing any rule just because it is stable inside
  one local path support or inside the raw path intersection

### 202. Simplification Pass: Strip Back to the Smallest Line That Still Preserves The Good Signals (2026-04-14)

After revisiting the full chain of experiments, the right simplification is now
quite clear.

#### What should be kept

Only a few ingredients have consistently produced useful signals:

- `log-link + canonical interaction basis`
  - this is still the cleanest theorem-friendly model class
  - it fixes the old cross-sign compensation pathology better than the
    asymmetric `(\mu + E) exp(-I)` line
- `train weighted stage-1`
  - this is the strongest general high-recall front end we have found
- `train adaptive stage-2`
  - this is the fastest useful compressor after stage-1
- `pooled exact selection` on a **small finite family**
  - this remains the only place where exact `100/100` has actually been
    recovered on `logical_context`

#### What should be removed from the main line

Several layers added complexity without becoming reliable general components:

- hard-fixed `holdout core`
- hard-fixed `pooled strong core`
- single-rule removable certificates as a main selector
- ancestor-block exact families
- path-intersection core
- per-support local strong/weak classification
- deterministic path family as a stand-alone selector

These objects may still be diagnostically useful, but they should no longer be
treated as main-line components.

#### Why this simplification is justified

The repeated failure mode is always the same:

- if a layer is too conservative, it drops true high-order/context rules
- if a layer is too aggressive, it freezes false extras as a core
- once a false rule is hard-fixed, the final exact stage cannot recover

So the general lesson is:

- use early stages only to create a **small reviewable candidate family**
- do **not** let early stages permanently decide inclusion unless the
  certificate is truly theorem-grade and obviously safe

#### The next main line

The simplified main line should now be:

1. `log-link + canonical basis`
2. `train weighted stage-1`
3. `train adaptive stage-2`
4. build a **global reviewable band** from deterministic path-stability /
   pooled diagnostics
5. run pooled exact selection only on that reviewable band

with the crucial restriction:

- no hard-fixed core unless backed by an unusually strong removability /
  non-removability certificate

In other words, path information should be used as a **proposal / stability
summary**, not as a direct inclusion rule.

#### Practical goal

The right target is no longer:

- "make the front end fully decide the support"

but instead:

- "make the front end produce a small, truth-covering, reviewable family"

Once that family is small enough, the exact pooled selector can finish the job
without heuristic pruning.

### 203. Global Path Band Simplifies the Line but Still Fails to Generalize to `kernel_gaussian` (2026-04-14)

I tested the simplified "global reviewable band" idea directly:

- keep `log-link + canonical basis`
- keep `train weighted stage-1`
- build a small deterministic path family from a fixed penalty grid
- define the reviewable set globally from:
  - the path union
  - plus any globally pooled add-candidates
- avoid hard-fixing a large core

#### `logical_context`

This simplification behaves reasonably:

- screened count: `14`
- path family count: `3`
- path union count: `11`
- stable active count: `3`
- global band count: `11`
- path union / band metrics:
  - `recall = 1.0`
  - `precision = 0.6364`

Crucially, it no longer hard-fixes obvious false rules such as
`A and F -> excitation`.

So on `logical_context`, the simplified global-band line appears compatible
with the previous successful exact correction story.

#### `kernel_gaussian`

The same idea fails to generalize:

- screened count: `78`
- path family count: `5`
- path union count: `25`
- stable active count: `7`
- global add-union count: `20`
- global band count: `37`
- global band metrics:
  - `recall = 1.0`
  - `precision = 0.1622`

The issue is structural:

- the path family is still too stable around many false Gaussian clutter rules
- some false rules remain stable active across the whole path, e.g.
  - `C and G -> excitation`
  - `C and H -> excitation`
  - `G and H -> excitation`
- pooled add-union then explodes the band further

#### Consequence

This means:

- the simplification itself is good
- but **penalty-path stability alone is not a sufficient global review object**
  for the Gaussian clutter case

The next main-line question is no longer "how to make the path family richer"
but rather:

- how to generate support diversity that actually breaks these false stable
  Gaussian clutter motifs

This suggests the next direction should likely move away from ever-more-elaborate
path/core logic and toward a more structural source of diversity, most likely
through the kernel side or another deterministic global object that is not just
the penalty path.

### 204. Direct Kernel Alternation Is Plausible, But Only After a Truth-Covering Support Band (2026-04-14)

I revisited the user suggestion of:

- initializing the bar-height kernels from the data
- then alternating between rule updates and kernel-height updates

This is a reasonable theorem-friendly direction **if** the function class is
fixed in advance:

- fixed knot grid
- fixed bar/interpolation basis
- data-driven initialization only
- alternating convex / block-convex updates thereafter

So the idea itself is not the problem.

#### What failed immediately

Running the existing end-to-end rule-dependent kernel learner directly on the
`kernel_gaussian_loglink` hard case collapsed to the empty model:

- selected rule count: `0`
- `recall = 0.0`
- `precision = 0.0`

So "full-universe direct alternating active-set" is too brittle / conservative
as a first-stage solver.

#### What also became clear

I then tried the more constrained idea:

- keep a small deterministic support family
- for each support in that family, refit the bar-height kernels by alternating
  optimization
- select the best refit support

But here the real limitation is structural:

- the deterministic path family on `kernel_gaussian` is still too weak
- even if kernel refitting reorders the candidate supports, it cannot recover
  `100/100` unless the true support is already inside that finite family

So kernel alternation can improve scoring, but it cannot repair a support
family with insufficient coverage.

#### Consequence

The right combined line is therefore:

1. first produce a **truth-covering small support band**
2. then run kernel-height alternating refits only inside that band
3. then finish with exact finite-family selection

In short:

- direct full-universe alternating = too unstable
- kernel alternation on a weak support family = too limited
- kernel alternation **inside a good reviewable support band** remains the
  right next synthesis

### 205. Deterministic Pilot-Kernel Diversity Is Conceptually Right, But The Current Support-Family Builder Is Still Too Expensive (2026-04-14)

I tried to operationalize the next step as:

- keep the `log-link + canonical basis` main line
- keep the function class fixed (`fixed knots + bar/interpolation basis`)
- generate a small deterministic pilot-kernel family by fixed transforms of the
  initialized bar heights
- use those pilots only to create support diversity, not as ad-hoc hand-tuned
  kernels

The intended theorem-friendly shape was:

1. fixed pilot family (for example, fixed power transforms of the initialized
   bar heights)
2. deterministic support families under each pilot
3. finite support-kernel family exact selection

#### What was learned immediately

- A naive full-screened-universe multikernel probe is still too expensive.
  Even stage-only versions are dominated by rebuilding the support-family
  machinery, especially on `kernel_gaussian`.
- I also caught an implementation mistake: one active-universe pilot probe was
  initially using the transformed raw cache directly, instead of re-entering the
  canonical basis before path construction. That is not the right main-line
  object and should not be used to draw conclusions.

#### The updated interpretation

This does **not** mean pilot-kernel diversity is the wrong idea.

It means that, with the current implementation:

- support-family generation is still too expensive even before exact selection
- the next probe must stay aligned with the canonical main line
- pilot-kernel diversity should be tested in the smallest possible conditional
  setting first, rather than by repeatedly rerunning the full screened-family
  machinery

So the next question is no longer:

- "can we just bolt pilot kernels onto the existing heavy pipeline?"

but instead:

- "what is the cheapest canonical diagnostic that can tell whether pilot
  kernels actually separate true rules from Gaussian clutter before we pay for
  a full family build?"

### 206. The Immediate Bottleneck In Pilot-Kernel Diversity Is Repeated Rule-Source Feature Construction, Not The Selector Itself (2026-04-14)

After attempting several progressively cheaper pilot-kernel probes, the next
diagnostic lesson is clear:

- the main cost is not yet the final exact selector
- the main cost is not even the path family itself
- the main cost is the repeated construction of transformed rule-source feature
  arrays under each pilot kernel

Concretely, the current probes rebuild, for each pilot and for each
`(rule, source)` pair:

- `normalized_kernel_response(basis_matrix, heights)`
- then bounded activity
- then subset products

This repeats large dense matrix-vector work many times before the support-family
stage can even start.

#### Consequence

This means pilot-kernel diversity should not be judged by the current wall-clock
alone. The current implementation is paying a large avoidable fixed cost.

#### The right next systems/algorithmic synthesis

The next theorem-friendly implementation should:

1. keep the fixed knot/bar basis and deterministic pilot family
2. group transformed rule-source kernels **by source**
3. evaluate all pilot-specific source responses in batched matrix-matrix form
   rather than repeated per-rule matvecs
4. compose subset products lazily from those cached source-level responses

This does not change the model class or the selector. It only removes a large
amount of duplicated linear algebra.

So the next question is no longer:

- "is pilot-kernel diversity itself too slow?"

but rather:

- "can we make pilot-kernel diversity cheap enough by batching source-level
  transformed responses so that the real support-quality question can finally
  be measured?"

### 207. Batched Source-Level Cache Was Implemented, But `kernel_gaussian` Is Still Dominated By Upstream Setup And Dense Stage-1 Solves (2026-04-14)

I implemented the first concrete systems change toward pilot-kernel diversity:

- in the active-universe Gaussian probe, transformed kernel responses are now
  batched source-by-source
- each source performs one matrix-matrix multiply against all pilot-specific
  height vectors for the rules that use that source
- subset products are then composed from those cached source-level responses

This removes a large amount of duplicated rule-wise matvec work inside the
pilot-specific cache builder.

#### What the follow-up diagnostic showed

Even after this change, the cheapest active-universe pilot probes remained
slower than desired on `kernel_gaussian`.

The evidence points to two remaining bottlenecks ahead of any final selector:

1. `materialize_loglink_dataset + setup_problem`
   - this is still a large fixed cost before any pilot family is built
2. dense canonical stage-1 solves on the active universe
   - even after source-level batching, the weighted canonical sparse fit itself
     remains expensive on the Gaussian active universe

#### Consequence

So the next implementation step is not another selector tweak.

It should be:

- persist / cache the expensive `setup_problem` outputs for the hard-case
  diagnostics
- and make the pilot diagnostic cheaper than a full stage-1/2 path, likely via
  a lighter canonical score-based or one-step proxy before paying for full
  sparse solves

In short:

- batched source-level caching was the right first systems fix
- but it is not enough by itself
- the remaining bottleneck has moved upstream to problem setup and dense
  canonical stage-1 optimization

### 208. A Real Input-Resolution Bug Was Making Gaussian Log-Link Diagnostics Look Much Slower Than They Really Were (2026-04-14)

While resuming the Gaussian hard-case line, I found a concrete implementation
bug in several diagnostic scripts:

- they were passing an already-materialized `_loglink` config into
  `materialize_loglink_dataset(...)`
- which means the helper tried to create a second-level
  `..._loglink_loglink.{yaml,pkl}` pair instead of using the existing
  `paper_kernel_robustness_gaussian_loglink.{yaml,pkl}`

This was not a theorem issue; it was simply the wrong input-resolution path.

I patched:

- `tmp_profile_kernel_gaussian_setup.py`
- `tmp_probe_gaussian_active_universe_holm_pilots.py`
- `tmp_probe_gaussian_active_universe_kernel_pilots.py`

to recognize an already-materialized `_loglink` config and use the existing
dataset directly.

#### Clean setup timing after the bug fix

Using the existing Gaussian log-link dataset and dumping the cached `problem`
object to:

- `tmp_problem_kernel_gaussian_loglink_cache_20260414.pkl`

the actual `setup_problem` timing is:

- `materialize_loglink_dataset`: essentially `0s`
- `load_dataset`: `0.083s`
- `build_seq_event_arrays`: `0.265s`
- `estimate_source_kernels`: `2.469s`
- `collect_events_and_grids`: `2.632s`
- `build_global_activity`: `11.543s`
- `build_event_lag_bin_cache`: `13.662s`
- `initialize_rule_specific_heights`: `13.811s`
- total cache dump complete: `14.078s`

So the corrected picture is:

- the Gaussian hard-case fixed setup cost is large but manageable (`~14s`)
- the dominant setup subroutine is `build_global_activity`
- the previous "many-minutes-before-anything-happens" interpretation was partly
  polluted by the `_loglink_loglink` bug

### 209. Deterministic Pilot-Kernel Holm On The 22-Rule Gaussian Review Universe Does Not Solve The Clutter Problem (2026-04-14)

With the corrected cached Gaussian `problem`, I ran the cheapest canonical
pilot-kernel diagnostic:

- active universe = current 22-rule review universe from
  `tmp_train_stage2_pooled_core_kernel_gaussian_20260414.json`
- pilot transforms = `alpha in {0.5, 1.0, 2.0}`
- object = canonical feature cache under that pilot, followed by one-sided Holm
  screening inside the fixed active universe

Result file:

- `tmp_gaussian_active_universe_holm_pilots_20260414.json`

#### Outcome

For all three pilot transforms, the result is essentially the same:

- `kept_count = 19, 19, 20`
- recall only `5/6 = 0.8333`
- precision only `0.25 ~ 0.263`

The same true rule is always missing:

- `D and F and G -> T : inhibition`

And the same false Gaussian clutter remains:

- `C and G -> T : excitation`
- `C and H -> T : excitation`
- `G and H -> T : excitation`
- plus related extras such as `C -> T : excitation`, `H -> T : excitation`,
  `D and F -> T : inhibition`, etc.

#### Important per-rule score diagnosis

For the missing true rule:

- `D and F and G -> T : inhibition`

the marginal canonical score is actually **negative** under every tested pilot:

- `alpha=0.5`: `z = -1.0`, event sum `-1.436`
- `alpha=1.0`: `z = -1.0`, event sum `-1.868`
- `alpha=2.0`: `z = -1.0`, event sum `-1.308`

Meanwhile, the false clutter rules remain strongly positive:

- `C and G -> exc`: `z ≈ 5.77, 6.48, 6.59`
- `C and H -> exc`: `z ≈ 3.99, 4.80, 4.92`
- `G and H -> exc`: `z ≈ 2.87, 3.56, 4.04`
- `D and F -> inh`: `z ≈ 6.06, 6.09, 5.64`

#### Interpretation

This is decisive:

- simple deterministic pilot-kernel diversity does **not** fix the Gaussian
  hard case if the band/object is still based on marginal canonical Holm
- the missing true inhibition is not "weak but positive"; it is marginally
  wrong-signed under every tested pilot
- so any theorem-friendly band generator built on marginal add/keep scores will
  remain unsafe on this case

### 210. The Gaussian Hard Case Is Conditional, Not Marginal: The True Rule Reappears Once The Current Support Is Conditioned On (2026-04-14)

I then took the current 22-rule Gaussian review support and ran the direct
conditional add-back diagnostic:

- remove one rule from the current support
- refit the reduced support
- compute the efficient one-sided add-back score for the removed rule

This shows something crucial.

For the previously missing true rule:

- `D and F and G -> T : inhibition`

the conditional add-back score is **strongly positive**:

- `z = 7.86`
- one-sided `p ≈ 1.9e-15`

So the Gaussian hard case is not that the true rule has no signal.
It is that the signal is **conditional** and therefore invisible to the
marginal Holm-style diagnostics above.

However, single-rule conditional add-back is still not enough as a selector,
because several false clutter rules are also conditionally strong once they are
embedded inside the large current support:

- `C and G -> exc`: `z = 15.06`
- `C and H -> exc`: `z = 6.99`
- `G and H -> exc`: `z = 9.74`
- `C and G and H -> exc`: `z = 8.61`

#### Interpretation

This gives the right structural reading:

- the Gaussian problem is neither a pure marginal screening failure
- nor something that single-rule conditional correction alone can fix
- it is a **group / surrogate-support phenomenon**

So:

- marginal review bands are too weak
- single-rule local certificates are too weak
- the next valid objects must be small **structured support families** or
  kernel-adaptive support comparisons

### 211. Fixed-Kernel Exact BIC Prefers Gaussian Clutter Blocks, But Kernel-Refit Comparisons Reverse Part Of That Conclusion (2026-04-14)

Using the 22-rule Gaussian review support under the canonical cache:

- fixed-kernel exact evaluation of the full support gives
  `upper_bic = 91051.62`
- removing the pair clutter block
  `{C and G, C and H, G and H}` made the support **look much worse** under the
  fixed kernel:
  `upper_bic = 91976.87` (worse by `+925.24`)

This is exactly why the support-only line kept getting trapped:

- under fixed kernels, the clutter block behaves like a genuinely useful
  surrogate support

But when I moved to **support-fixed kernel-height refits** with
`optimize_active_set_torch(..., intensity_model='canonical_loglink')`, the
picture changed.

Using a 30-step kernel/coef refit:

- `FULL` 22-rule support:
  - refit BIC `57902.53`
- `pair_clutter_CGH` removed:
  - refit BIC `57754.97`

So once the kernel is allowed to move, removing the clutter block is no longer
harmful; it is actually slightly beneficial.

#### Meaning

This is strong evidence that the Gaussian clutter is not an immutable support
fact. It is at least partly a **support-only artifact of frozen or poorly
matched kernels**.

So the next main line should not be:

- "keep refining support-only exact selectors under fixed kernels"

It should be:

- "build a truth-covering finite support family, then compare those supports
  after band-internal kernel refit"

### 212. The True Gaussian Support Becomes Competitive Only After Sufficient Kernel Refitting, And Complexity Penalties Matter (2026-04-14)

I also evaluated the true Gaussian support under support-fixed kernel refitting.

For the exact true support:

- 30-step refit + one family-attribution pass:
  - `bic_refine ≈ 19052.84`
- 60-step refit + one family-attribution pass:
  - `bic_refine ≈ 13567.52`
- 100-step refit + one family-attribution pass:
  - `bic_refine ≈ 12518.41`

So the earlier poor true-support BIC (`~62913` at the shallow 30-step raw fit)
was a misleading under-optimized value. The true support becomes far more
competitive once the kernel update is actually allowed to settle.

#### Another implementation issue discovered

During this comparison, I found a real bug in
`family_attribution_refine(...)` for `canonical_loglink`:

- excitation-side attribution was still using the multiplicative-model formula
  `coef * feature / eta_ev`
- but `eta_ev` is not even defined in the canonical log-link branch

I patched this so that canonical excitation attribution uses the direct
event-side contribution instead.

I also enabled `family_attribution_refine` to optionally pass through
`penalize_kernel_df=True`, so kernel-adaptive comparisons can use a more
appropriate complexity penalty.

#### Kernel-DF-penalized comparison

With `penalize_kernel_df=True` and 30-step refit + one attribution pass:

- `TRUE_SUPPORT`:
  - `bic_refine ≈ 19895.77`
- `FULL` 22-rule support:
  - `bic_refine ≈ 16619.01`
- `pair_clutter_CGH` removed:
  - `bic_refine ≈ 16183.46`

So kernel-DF penalty helps in the right direction:

- large cluttered supports are penalized more strongly
- the pair-clutter drop becomes preferable to the full cluttered support

But the true support still does **not** yet win automatically at shallow
refits. That means the remaining challenge is not only support coverage;
it is also:

- better kernel-adaptive optimization on small supports
- and a principled complexity penalty strong enough to keep surrogate supports
  from dominating

### 213. Updated Main Research Direction After These Diagnostics (2026-04-14)

The cumulative evidence now points to a much cleaner next line.

What is no longer plausible as the main line:

- marginal Holm / marginal add-back review bands
- fixed-kernel support-only exact selection
- single-rule conditional certificates

What now looks most promising:

1. keep `log-link + canonical basis`
2. keep a theorem-friendly high-recall front end
3. build a **finite, truth-covering structured support family**
   rather than a single hard band
4. for each candidate support in that family, run **support-fixed kernel-height
   refit**
5. compare those candidates with a complexity-aware criterion
   (`kernel_df`-aware BIC / oracle bound)

In short:

- the Gaussian hard case is not mainly a "search tree" problem anymore
- it is a **support-family + kernel-adaptation + complexity-penalty** problem
- and the right next step is to design the smallest theorem-friendly support
  family whose members can be fairly compared only **after** kernel refitting

### 214. Releasing The Core Is Necessary, But Singleton-Only Profiled Deletion Is Still Not Enough (2026-04-15)

I tested the most stripped-down version of the new profile-selector idea:

- keep the current stage-2 active support
- treat **every** active rule as reviewable (`all_reviewable`)
- build the smallest deterministic family consisting of:
  - the full support
  - every single-rule deletion
- compare these candidates only after support-fixed kernel refitting with
  `penalize_kernel_df=True`

This removes the earlier hard-core fixation problem as cleanly as possible while
staying theorem-friendly.

#### `logical_context`

The resulting family size was:

- active support size: `11`
- singleton-delete profile family size: `12`

At shallow profiled refit (`steps=15`), the first few candidates behaved badly:

- full support profiled `bic_refine ≈ 8888.29`
- dropping `A -> T : excitation` improved this to `≈ 8805.08`
- dropping `E and F -> T : excitation` improved it further to `≈ 8735.99`

So the profiled criterion, at this optimization depth, is already preferring to
delete true low-order excitation rules on the context hard case.

#### `kernel_gaussian`

The corresponding singleton-delete profile family was:

- active support size: `22`
- singleton-delete family size: `23`

The full-support profiled refit alone gave:

- `bic_refine ≈ 16827.98`

and each candidate refit was still expensive enough that the run was stopped
once the `logical_context` failure mode became clear.

#### Interpretation

This is an important negative result:

- releasing the unsafe core **is necessary**
- but "all-reviewable + singleton profile deletions" is still too weak

The reason is now clearer:

- if the support family is too local (only singleton deletions),
  profiled selection can still improve its criterion by removing genuinely
  useful low-order rules before it has the right mixed add/drop support family
  available
- so the next family must be **mixed add/drop and structured**, not just
  drop-only

In short:

- hard-fixed cores were indeed a real blocker
- but simply freeing all rules and allowing only singleton deletions does **not**
  yet produce a truth-preserving profiled selector

### 215. Root Cause Distillation: We Have Been Generating Support Families Under The Wrong Objective (2026-04-15)

After stripping back the recent experiments, the root cause now looks much
cleaner than before.

#### What is the real structural mistake?

Almost every failed line has the same hidden pattern:

1. build a support family under a **frozen-kernel or shallow-kernel surrogate**
2. then hope that a later exact / profiled step will repair the mistakes

But the Gaussian diagnostics show that this ordering is wrong.

- A true rule like
  `D and F and G -> inhibition`
  is marginally wrong-signed under the frozen-kernel review objects, yet becomes
  strongly positive conditionally once the support is conditioned on.
- A false clutter block like
  `{C and G, C and H, G and H}`
  looks useful under fixed-kernel exact BIC, but becomes removable once the
  kernel heights are refit.
- Even the new all-reviewable singleton-profile family can improve its profiled
  criterion by deleting true low-order excitations, because the family itself is
  still too local and was generated before the correct mixed add/drop profiled
  comparisons were made available.

So the real issue is **not** just:

- search tree size
- bad core freezing
- or insufficiently rich local bands

The real issue is:

- **support families are being generated under the wrong objective**

We are still using support generators that are aligned with a frozen or shallow
kernel surrogate, while the true comparison object is the **profiled support
criterion** after kernel refitting.

#### Consequence

This means the next main line should not be:

- more exact search on frozen-kernel support families
- more local add/drop heuristics around fixed-kernel supports
- more hand-built block grammars layered on top of a surrogate support family

It should be:

- build the **support path / review family directly under the profiled
  objective**

#### Updated main algorithmic direction

Keep:

- `log-link + canonical interaction basis`
- fixed knot / bar kernel sieve
- theorem-friendly weighted penalties

Change the key object:

- for each penalty level `lambda` on a deterministic grid,
  solve the **profiled penalized problem**
  over canonical support coefficients and support-fixed kernel heights
- take the resulting profiled support path as the deterministic proposal family
- enrich only with a small profiled KKT / near-threshold review band if needed
- then run final finite-family profiled BIC / LM-BIC selection

In symbols, the proposal path should come from

`(theta_hat_lambda, h_hat_lambda) = argmin_{theta, h in sieve} -ell_n(theta, h)
 + lambda * weighted_L1(theta) + pen_kernel(h)`

not from a frozen-kernel surrogate.

#### Why this is a real simplification

If this is right, then many recent layers can be dropped entirely:

- path families built from frozen-kernel stage-2 fits
- path-intersection cores
- global add-union under frozen kernels
- structured deletion families built on top of surrogate supports

All of these were trying to patch a support family that was already born under
the wrong objective.

#### Theorem-friendly shape

This line also gives a cleaner proof story:

1. fixed kernel sieve class
2. profiled penalized path over a deterministic lambda grid
3. finite profiled support family
4. final profiled BIC / LM-BIC selection over that family

Then the theorem target becomes:

- if the true support appears somewhere on the profiled path family
- and the profiled criterion approximates the oracle support score uniformly
- and the support margin is positive

then finite-family profiled selection recovers the true support.

This is much cleaner than trying to prove correctness for frozen-kernel local
add/drop enrichments.

### 216. First profiled-path probe: correct objective, insufficient support diversity

We implemented a first `profiled canonical path` probe on top of the current
reviewable universes:

- script:
  [tmp_probe_loglink_profiled_path_family.py](/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/tmp_probe_loglink_profiled_path_family.py)
- runs:
  - [tmp_profiled_path_family_logical_context_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_path_family_logical_context_20260415.json)
  - [tmp_profiled_path_family_kernel_gaussian_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_path_family_kernel_gaussian_20260415.json)

Setup:

- fixed active universe from the latest `train_stage2_pooled_core` review set
- deterministic lambda grid `{0.75, 1.0, 1.25}`
- for each lambda:
  - solve penalized canonical coefficient step on the active universe
  - then run support-fixed kernel-height refit
  - repeat for two outer iterations

Observed results:

- `logical_context`
  - elapsed `261.81s`
  - support family collapsed to a single support (`family_count = 1`)
  - best support size `11`
  - `recall = 1.0`, `precision = 0.6364`
- `kernel_gaussian`
  - elapsed `417.71s`
  - support family also collapsed to a single support (`family_count = 1`)
  - best support size `18`
  - `recall = 1.0`, `precision = 0.3333`

Interpretation:

- this is the first direct evidence that the **profiled objective is the right
  objective**, in the sense that both hard cases retained full recall without
  reverting to frozen-kernel support logic;
- however, the current profiled-path implementation does **not generate enough
  support diversity** to separate truth from clutter.

So the root problem has moved:

- before: the support family was being generated under the wrong
  frozen-kernel/shallow-kernel surrogate objective;
- now: the support family is being generated under the right profiled
  objective, but the deterministic path itself is too coarse/stable and
  collapses to one support.

This is still progress, because it means the next theorem-friendly step should
*not* be another frozen-kernel enrichment layer. Instead it should be a
`profiled local family` around the profiled path:

- profiled path support(s) as anchors
- deterministic one-step or small-block add/drop candidates evaluated under the
  same profiled objective
- final finite-family profiled BIC / LM-BIC selection on that enriched profiled
  family

In short:

- `profiled path` fixed the objective mismatch,
- but by itself it is too low-entropy to reach `100/100`,
- so the next line should be **profiled family enrichment**, not a return to
  frozen-kernel support logic.

### 217. Why extras do not drop cleanly: kernel-profile flexibility can also absorb true low-order rules

We next probed a `profiled local family` around the profiled-path anchor:

- script:
  [tmp_probe_loglink_profiled_local_family.py](/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/tmp_probe_loglink_profiled_local_family.py)
- family:
  - anchor
  - all singleton drops
  - all singleton adds from the review universe
  - same-sign overlap pair drops

The first useful diagnostic came from `logical_context`
([log](/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_local_family_logical_context_20260415.log)).

Early profiled local results:

- anchor refit:
  - support size `11`
  - `bic_refine ≈ 8994.32`
- singleton drops:
  - `drop A -> excitation`: `bic_refine ≈ 8912.40`
  - `drop A and F -> excitation`: `bic_refine ≈ 8842.21`
  - `drop B and C -> excitation`: `bic_refine ≈ 8842.21`
  - `drop E and F -> excitation`: `bic_refine ≈ 8842.21`
  - `drop A and B and D -> excitation`: `bic_refine ≈ 8772.01`

Interpretation:

- under the current profiled criterion, extras are not the only terms that look
  removable;
- even true low-order excitation rules can look *more removable* than some
  clutter terms.

So the reason extras do not “obviously” drop from the fitted intensity is not
that they are harmless in isolation. The deeper issue is that the current
kernel-profile nuisance class is flexible enough to **re-distribute intensity
mass after a drop**, sometimes making the model prefer a simpler but wrong
support.

This is a more principled diagnosis than “search is weak”:

- the local family is already being evaluated under the right profiled
  objective;
- yet the criterion still over-rewards support simplification, including
  dropping true low-order rules.

Therefore the next theorem-friendly fix should not be “more local drop
candidates” alone. It should be to **reduce nuisance flexibility in a
structured, deterministic way**, so that kernel refits cannot freely mask
missing true rules while still allowing clutter to be removed.

The cleanest next direction is:

- keep `log-link + canonical support`
- keep a fixed kernel sieve
- but replace raw bar-height flexibility by a more controlled
  `orthogonal/low-dimensional kernel basis` or shared-shape kernel sieve
  criterion before final finite-family selection

In short:

- extras survive because the profiled nuisance layer can compensate for both
  extras and some true low-order rules;
- this is now pointing to a **kernel-side identifiability problem**, not just a
  support-family problem.

### 218. Simple kernel-side restrictions are not enough: ridge and source-sign tying leave the drop ordering unchanged

We tested two theorem-friendly deterministic restrictions on the profiled local
family around the current `logical_context` anchor:

- global-source kernel anchor ridge
- shared kernel heights by `(source, sign)` instead of full `(rule, source)`

Scripts:

- [tmp_probe_loglink_profiled_local_family.py](/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/tmp_probe_loglink_profiled_local_family.py)

Main result:

- neither restriction materially changed the local ranking
- the same singleton drops still dominated:
  - `drop A and B and D -> excitation`
  - `drop A and F -> excitation`
  - `drop B and C -> excitation`
- and dropping a true low-order rule like `A -> excitation` still looked better
  than keeping the anchor support.

Representative run:

- [tmp_profiled_local_family_logical_context_tiesrcsign_pooled_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_local_family_logical_context_tiesrcsign_pooled_20260415.json)
- `kernel_tie_mode = source_sign`
- `family_count = 6`
- anchor:
  - pooled BIC `≈ 67984.43`
  - `recall=1.0`, `precision=0.6364`
- best singleton:
  - `drop A and B and D -> excitation`
  - pooled BIC `≈ 66202.85`
  - `recall=1.0`, `precision=0.7`
- but `drop A -> excitation` still also improves pooled BIC

Interpretation:

- this is important because it rules out a simple explanation that “the kernel
  nuisance is just too flexible because it is fully rule-specific.”
- even after deterministic kernel simplifications, the local criterion still
  prefers support simplification broadly.

So the next fix should not be another kernel-only restriction of the same type.
The more likely issue is that the current anchor support contains a
`surrogate-equivalent excitation clutter set`, and singleton diagnostics cannot
separate:

- “drop only the false extras”
from
- “drop a true low-order excitation and re-balance elsewhere.”

This pushes the next theorem-friendly direction toward:

- `profiled multi-drop / swap families`
- over a small deterministic structured group dictionary
- evaluated by pooled profiled BIC / LM-BIC

rather than more singleton-local kernel penalties.

### 219. Exact root cause on Gaussian: true high-order rules are masked by surrogate neighborhoods; deterministic centered purges can recover `100/100`

We pushed the diagnosis further in two directions.

#### 219.1. Cross-fit / pooled criterion disagreement on Gaussian

For `kernel_gaussian`, we directly compared the current profiled-path anchor to
the *true support* under support-fixed kernel refits.

Key comparisons:

- shallow/cold refits made the true support look much worse than the anchor;
- with deeper refits and better warm starts, the gap shrank dramatically;
- importantly, the **validation-side profiled criterion** and the **pooled
  profiled criterion** disagreed.

Example:

- anchor at `40` steps:
  - pooled BIC `≈ 119831.11`
  - refine/validation criterion `≈ 15727.41`
- true support at `40` steps, cold:
  - pooled BIC `≈ 146045.63`
  - refine `≈ 17072.89`
- true support at `40` steps, warm-started from anchor heights:
  - pooled BIC `≈ 131183.88`
  - refine `≈ 15448.85`

So Gaussian has **both**:

- an optimization-basin issue (warm starts matter a lot), and
- a criterion issue (pooled profile still over-rewards clutter relative to
  validation-side profile).

#### 219.2. Singleton deletion paths are the wrong geometry

We then ran a deterministic profiled deletion path on the Gaussian anchor:

- script:
  [tmp_probe_profiled_deletion_path.py](/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/tmp_probe_profiled_deletion_path.py)
- result:
  [tmp_profiled_deletion_path_kernel_gaussian_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_deletion_path_kernel_gaussian_20260415.json)

This path immediately removed the true high-order inhibition
`D and F and G -> inhibition` at the first deletion step and never recovered it.
The path kept peeling easier lower-order surrogates and eventually over-pruned
to a wrong sparse support.

This identifies the exact local geometry problem:

- a true high-order rule can be **locally less preferred than its surrogate
  lower-order/neighbor clutter** under singleton deletion;
- therefore singleton add/drop or greedy deletion paths do not generate a
  truth-covering family, even when the final criterion is improved.

#### 219.3. Deterministic centered neighborhood purges reveal the right block structure

We then tested a fully deterministic `rule-centered surrogate neighborhood
purge` diagnostic on the Gaussian anchor.

For a chosen active rule `r`, define its same-sign surrogate neighborhood as:

- active same-sign strict subsets of `r`
- active same-sign same-order neighbors sharing `|r|-1` sources

Then evaluate supports obtained by *keeping the center rule* and *dropping its
surrogate neighborhood*.

Representative diagnostics:

- `purge around C and G and H -> excitation`
  - `recall=1.0`, `precision≈0.4615`
- `purge around D and F and G -> inhibition`
  - `recall=1.0`, `precision≈0.4615`
- `purge around E and F -> inhibition`
  - `recall=1.0`, `precision≈0.4000`
- combined purge around
  - `C and G and H -> excitation`
  - `D and F and G -> inhibition`
  - `E and F -> inhibition`
  gave
  - `recall=1.0`, `precision≈0.8571`
  - only extra left: `A and D -> excitation`
- adding one more deterministic center,
  `C and D -> excitation`,
  produced the exact true support:
  - rules:
    - `A -> T : excitation`
    - `C and D -> T : excitation`
    - `C and G and H -> T : excitation`
    - `B -> T : inhibition`
    - `E and F -> T : inhibition`
    - `D and F and G -> T : inhibition`
  - `recall=1.0`
  - `precision=1.0`

Interpretation:

- the real problem is **not** “extras do not perturb the intensity enough”;
- the real problem is that extras live in **surrogate neighborhoods centered on
  true high-order rules**;
- dropping rules one-by-one cannot expose the true rule because the surrogate
  neighborhood rebalances around it;
- but dropping the whole deterministic surrogate neighborhood around a true
  center does.

This is the clearest root-cause statement so far.

#### 219.4. Research direction update

The next theorem-friendly line should therefore be:

- keep `log-link + canonical support`
- keep profiled kernel refits
- stop relying on singleton local families or greedy deletion paths
- build a finite family from **deterministic centered neighborhood purge
  blocks**
- and select among that family using a validation-side / cross-fit profiled
  criterion rather than pooled profile alone

In short:

- `context` failed mainly because singleton families did not include the right
  multi-drop block;
- `gaussian` failed because true high-order rules are masked by deterministic
  surrogate neighborhoods, which must be purged as blocks.

### 220. Heuristic-free runtime cleanup: shared profiled-support cache and warm-started family evaluation (2026-04-15)

This round did **not** change the selector family or criterion. It only tightened
the implementation so the *same* profiled family is evaluated more cheaply and
more consistently.

#### 220.1. Implementation fixes

- Created a shared helper module:
  - `workspace/train/paper_benchmark_active/tmp_profiled_support_common.py`
- `fit_profiled_support(...)` now reuses:
  - cached `n_events_train`
  - cached CPU `TorchBasisCache`
  - cached dataset/config resolution
- `family_attribution_refine(...)` had a real objective mismatch:
  its final refine pass was not forwarding
  - `kernel_anchor_heights`
  - `kernel_anchor_ridge`
  - `kernel_tie_mode`
  even when the earlier optimize steps were. This is now fixed.
- `family_attribution_refine(...)` can optionally return final `arrays_out`, so
  pooled-profile criteria can reuse already-computed rule arrays instead of
  recomputing them.
- structured profiled family evaluation now:
  - fits the anchor support once
  - reuses anchor kernel heights / coefficients / `mu`
  - warm-starts nearby candidate supports from the anchor fit

#### 220.2. Focused hard-case microbenchmarks

Representative `logical_context` support fit:

- anchor support (11 rules):
  - `43.16s`
  - `bic_refine ≈ 8714.64`
- candidate support, cold start (7 rules):
  - `13.62s`
  - `bic_refine ≈ 7965.79`
- same candidate, warm start from anchor:
  - `13.48s`
  - `bic_refine ≈ 7855.25`

Representative `kernel_gaussian` support fit:

- anchor support (18 rules):
  - `51.61s`
  - `bic_refine ≈ 15931.37`
- candidate support, cold start (6 rules):
  - `6.77s`
  - `bic_refine ≈ 19895.77`
- same candidate, warm start from anchor:
  - `6.77s`
  - `bic_refine ≈ 13537.29`

Interpretation:

- the main runtime win is not from changing the family, but from:
  - *not* rebuilding the anchor refit over and over
  - *not* rebuilding CPU basis/tensor caches over and over
- warm starts do not dramatically shrink wall-clock on very small candidate
  supports, but they **do** materially improve the profiled objective reached,
  especially on `kernel_gaussian`

#### 220.3. Early full-family speed signal

Re-running the existing structured-block family with the same `steps=30` but the
new shared-cache/warm-start implementation gave the following early progress:

- `logical_context`:
  - `3 / 13` candidates in about `210s`
  - about `70s` per candidate so far
  - old full run average had been roughly `103s` per candidate
- `kernel_gaussian`:
  - anchor candidate still about `200s` scale

So the cleanup is real, but it is **not** the whole answer. The remaining
bottleneck is still support-fixed kernel refit on medium/large supports.

#### 220.4. Research direction update

- These changes preserve the current theorem-friendly object:
  - same support family
  - same profiled objective
  - same complexity penalty
- They should be treated as *required infrastructure* before any further family
  design work.
- Remaining runtime breakthroughs will likely need:
  - cheaper medium-support kernel refits
  - or a smaller truth-covering profiled family
  rather than more caching alone.

### 221. Next runtime line: deterministic candidate family + candidate-level parallelism + certified approximate inner optimization (2026-04-15)

The next runtime reduction should **not** change the selector family or its
finite-family theorem object.

Keep fixed:

- `log-link + canonical support`
- deterministic profiled candidate family
- same profiled criterion for final comparison

Change only the evaluation strategy:

1. `candidate-level parallelism`

- evaluate different candidate supports concurrently
- this changes wall-clock only, not the argmin
- deterministic execution still requires controlled seeds / thread caps

2. `certified approximate inner optimization`

- do **not** replace the criterion with a heuristic score
- instead, for each candidate support `S`, compute:
  - approximate profiled objective `J_hat(S)`
  - certified optimization error radius `eps(S)`
- use coarse approximate fits for all candidates
- then refine only near-tie candidates whose error bands overlap

Safe comparison rule:

- if for the current best candidate `S*` and another candidate `T`,
  `J_hat(S*) + eps(S*) < J_hat(T) - eps(T)`,
  then `T` cannot beat `S*` under the exact profiled objective

This preserves the theorem-friendly finite-family selector structure:

- family generation remains deterministic
- final decision can still be justified as exact over the finite family once all
  unresolved near-ties are refined

Current runtime expectation from existing measurements:

- current patched implementation:
  - `logical_context`: about `10~20 min`
  - `kernel_gaussian`: about `40~90 min`
- with candidate-level parallelism plus certified approximate inner solves:
  - `logical_context`: about `3~8 min`
  - `kernel_gaussian`: about `15~35 min`

Interpretation:

- parallelism alone is exact and safe
- approximate inner optimization is only safe when coupled to
  candidate-specific error control / selective exact refinement
- the next implementation target is therefore:
  deterministic family + parallel candidate evaluation + coarse-to-exact
  certified support refits

#### 221.1. First implementation result: parallel candidate evaluation works, but gains are modest so far

Implemented:

- shared profiled-support helper:
  - `tmp_profiled_support_common.py`
- anchor-fit reuse across family candidates
- warm-started candidate refits from the anchor support
- `spawn`-based candidate-level multiprocessing in
  - `tmp_run_loglink_structured_block_kernel_family.py`

Focused measurements:

- representative microbenchmarks already showed:
  - `logical_context`
    - 11-rule anchor fit: `~43.2s`
    - 7-rule candidate fit: `~13.5s`
  - `kernel_gaussian`
    - 18-rule anchor fit: `~51.6s`
    - 6-rule candidate fit: `~6.8s`

- `logical_context` singleton-only parallel smoke test
  - family size: `8`
  - workers: `2`
  - after about `300s`, `3 / 8` candidates had completed
  - best profiled BIC had improved from
    - full support: `~8785.43`
    - to `~8452.12`

Interpretation:

- candidate-level parallelism is operational and exactness-preserving
- but the wall-clock gain is still limited because medium-support profiled kernel
  refits dominate runtime
- in other words, the main bottleneck has moved from
  *family enumeration* to *per-candidate kernel refit cost*

Current practical takeaway:

- parallelism should remain in the final implementation
- but it is not enough by itself
- the next required acceleration is still on the inner profiled kernel refit
  path, ideally via certified coarse-to-exact optimization schedules

#### 221.2. First coarse-to-exact inner-optimization probe: better objective, worse wall-clock

We tested a simple deterministic inner schedule:

- run a short coarse profiled fit
- then warm-start a second profiled fit with fewer “exact” steps

This preserves the same candidate family and same final profiled criterion, but
is only useful if the two-stage schedule beats a single warm-started fit in wall
clock.

Representative `logical_context` candidate support:

- one-shot warm fit:
  - `steps=30`
  - `~13.95s`
  - `bic_refine ≈ 7855.25`
- coarse/exact:
  - `(4,12)`:
    - `~23.07s`
    - `bic_refine ≈ 7848.51`
  - `(6,10)`:
    - `~22.90s`
    - `bic_refine ≈ 7848.49`
  - `(6,12)`:
    - `~22.80s`
    - `bic_refine ≈ 7848.51`

Representative `kernel_gaussian` candidate support:

- one-shot warm fit:
  - `steps=30`
  - `~7.64s`
  - `bic_refine ≈ 13537.29`
- coarse/exact:
  - `(4,12)`:
    - `~10.66s`
    - `bic_refine ≈ 13120.58`
  - `(6,10)`:
    - `~10.58s`
    - `bic_refine ≈ 13122.09`
  - `(6,12)`:
    - `~10.72s`
    - `bic_refine ≈ 13095.19`

Interpretation:

- the naive two-stage “coarse then exact” schedule does improve the profiled
  objective slightly
- but it does **not** reduce wall-clock in the current implementation
- for the candidates tested, it is consistently slower than a single warm-started
  fit

So the next inner-optimization line should **not** be “stack two full profiled
fits.” The more promising directions are:

- true early-stopping / tolerance-aware approximate solves
- certified coarse scores with selective exact refinement
- or a cheaper medium-support kernel refit backend

#### 221.3. Approximate profiled path family run: preserves recall, does not fix precision

We also tested the same deterministic profiled-path family with a lighter inner
optimization budget (`outer_iters=2`, `kernel_steps=12`) to see whether
general, heuristic-free approximate inner solves could cut runtime without
materially hurting support quality.

`logical_context`
- result file:
  - `/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_path_family_logical_context_approx12_20260415.json`
- total elapsed: `~387.49s`
- family count: `1`
- best support metrics:
  - `recall = 1.0`
  - `precision = 0.6364`
- missing rules: none
- extra rules:
  - `A and B -> T : inhibition`
  - `A and B and D -> T : excitation`
  - `A and F -> T : excitation`
  - `B and C -> T : excitation`

`kernel_gaussian`
- result file:
  - `/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_path_family_kernel_gaussian_approx12_20260415.json`
- total elapsed: `~624.91s`
- family count: `2`
- best support metrics:
  - `recall = 1.0`
  - `precision = 0.3333`
- missing rules: none
- extra rules:
  - `A and D -> T : excitation`
  - `A and F and G -> T : inhibition`
  - `C -> T : excitation`
  - `C and F and G -> T : inhibition`
  - `C and G -> T : excitation`
  - `C and H -> T : excitation`
  - `D and F -> T : inhibition`
  - `E -> T : inhibition`
  - `F -> T : inhibition`
  - `F and G and H -> T : inhibition`
  - `G and H -> T : excitation`
  - `H -> T : excitation`

Interpretation:
- the lighter profiled inner solve preserves high recall in both hard cases
- but it does **not** improve precision enough to solve the benchmark
- and, in the current implementation, it is still not clearly faster than the
  stronger profiled path baseline
- so “approximate inner optimization” only helps if it becomes a genuinely
  cheaper *single-pass* solver or is paired with selective exact refinement
  near ties

### 222. Structural diagnosis of false extras: same-sign surrogate neighborhoods

Looking across the current best profiled-path runs, the dominant false extras are
not arbitrary. They are almost all **same-sign local surrogate neighbors** of a
true rule.

`logical_context` false extras from the profiled-path run:
- `A and B -> inhibition` behaves like a same-sign extension/surrogate around
  true inhibitory centers `B` and `A,C,D`
- `A and B and D -> excitation` is a same-sign near-neighbor of true
  `B,C,D -> excitation`
- `A and F -> excitation` is a same-sign surrogate of true `E,F -> excitation`
- `B and C -> excitation` is the lower-order shadow of true
  `B,C,D -> excitation`

`kernel_gaussian` false extras are even more systematic:
- `C`, `C,G`, `C,H`, `G,H`, `H` are all lower-order / nearby same-sign shadows of
  true `C,G,H -> excitation`
- `D,F`, `C,F,G`, `A,F,G`, `F,G,H` are same-sign neighbors of true
  `D,F,G -> inhibition`
- `E`, `F` are lower-order shadows of true `E,F -> inhibition`

Interpretation:
- the precision problem is **not** mainly “bad individual rules survive”
- it is that each true center rule generates a small same-sign surrogate
  neighborhood that can trade off against the center under the profiled
  kernel nuisance
- so singleton add/drop is fundamentally misaligned with the local geometry
- the right theorem-friendly object is a deterministic **surrogate-neighborhood
  family** around each active/proposed center, with final selection acting on
  these local structured neighborhoods rather than on isolated rules

This supports the next line: focus on distinguishing **center rules vs their
local same-sign surrogate neighborhood**, not “important vs unimportant rules” in
isolation.

### 223. First general deterministic surrogate-neighborhood purge run (`logical_context`, K=1)

We implemented a genuinely general, deterministic surrogate-neighborhood family
around the profiled-path anchor support. For each active anchor rule `r`, the
neighborhood `N(r)` is defined by a fixed grammar within the anchor support:
- same-sign strict subsets differing by one literal
- same-sign strict supersets differing by one literal
- same-sign same-order neighbors with overlap `|U ∩ V| = |U|-1`

Candidate supports are `S0 \ N(r)` for all centers `r`, plus the anchor itself.
This is a theorem-friendly finite family; no rule names are hand-picked.

Run:
- script:
  - `/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/tmp_probe_loglink_profiled_surrogate_family.py`
- result:
  - `/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_surrogate_family_logical_context_k1_20260415.json`

Observed performance (`logical_context`):
- total elapsed: `~378.24s`
- family size: `7`
- anchor metrics:
  - `recall = 1.0`
  - `precision = 0.6364`
- best surrogate-purge candidate:
  - `name = purge::exc:43`
  - purged rules: `exc:13`, `exc:29`
  - support size: `9`
  - `recall = 1.0`
  - `precision = 0.7778`
  - missing rules: none
  - extra rules:
    - `A and B -> T : inhibition`
    - `A and F -> T : excitation`

Interpretation:
- the general surrogate-neighborhood family does improve precision without
  hurting recall
- this supports the structural diagnosis that false extras are local same-sign
  surrogate neighbors of true centers
- however, `K=1` is not enough to reach `100/100`; some residual extras require
  coordinated multi-center purges or swap-style candidates


Note on expected runtime for the deterministic surrogate-neighborhood family:
- `logical_context`, `K=1` run finished in `~378.24s` with family size `7`
- the same anchor support has `21` distinct purge supports for `K<=2`, so the
  practical family size would be about `22` including the anchor
- with observed per-candidate times around `45s~56s`, the current serial
  implementation suggests roughly `15~25 minutes` for `K=2` on `logical_context`
- this is a scaling estimate, not yet a completed end-to-end run

### 224. Dynamic `K` and GPU: when they stay theorem-friendly

`K` in the surrogate-neighborhood family denotes the maximum number of centers
whose neighborhoods are purged jointly. A naive data-dependent choice of `K`
(“increase until the result looks good”) would be heuristic and should not be a
mainline method.

A theorem-friendly dynamic-`K` strategy is still possible, but only if the
stopping/expansion rule is itself deterministic and certificate-based. The right
shape is:
- evaluate the full deterministic family for `K=1`
- maintain the incumbent support and criterion value
- expand to `K=2` only if the unresolved higher-order neighborhood family cannot
  be safely ruled out by a valid bound
- otherwise stop

So dynamic `K` is acceptable only as **certificate-driven family expansion**, not
as manual or visual tuning.

GPU use is also theorem-friendly. It changes neither the family nor the
objective, only the hardware backend for the same tensor program. In the current
implementation, however, `fit_profiled_support` still hardcodes `device="cpu"`,
so GPU acceleration requires a plumbing change rather than a methodological one.

Expected practical effect:
- dynamic `K` with valid stopping can save substantial time on cases that are
  already resolved at `K=1`
- GPU should mostly help the medium-support profiled kernel refits that dominate
  wall-clock, with no accuracy loss
- the strongest wall-clock improvement will come from combining
  certificate-driven `K` expansion with GPU-backed candidate refits

### 225. Root-cause diagnosis: the bottleneck is now the selector criterion, not search or optimization

The recent dynamic-`K` + GPU surrogate-neighborhood runs clarify the current
failure mode. The main problem is no longer family construction or raw runtime;
it is the **selection criterion** used to rank structured supports under a
profiled kernel nuisance.

Evidence from `logical_context`:
- the profiled-path anchor has `recall=1.0`, `precision=0.6364`
- deterministic surrogate-neighborhood purge with `K=1` improves to
  `recall=1.0`, `precision=0.7778`
- but dynamic `K<=2` under pooled profiled BIC moves to a support with
  `recall=0.8571`, `precision=0.75`, dropping the true center
  `B,C,D -> excitation`
- so the issue is not that the family lacks the true support; the criterion is
  willing to over-purge a true center when the profiled kernel can compensate

Evidence from `kernel_gaussian`:
- the profiled-path anchor has `recall=1.0`, `precision=0.3333`
- in the dynamic-`K` run, essentially every `K=1` purge candidate improves the
  pooled profiled criterion relative to the anchor
- therefore the current certificate-driven expansion rule cannot prune: the
  pooled criterion treats many surrogate-purge moves as favorable
- this means the bottleneck is not GPU speed or family size; it is that the
  pooled profiled score does not sharply distinguish true centers from their
  local surrogate neighborhood

Root cause:
- profiled in-sample / pooled BIC is too permissive when the kernel nuisance is
  flexible enough to absorb intensity mismatches created by deleting a true rule
- the same criterion simultaneously rewards deleting local surrogate clutter and
  deleting some true centers, because both moves can be partially compensated by
  refitting nearby rules and kernel heights
- hence the current selector is misaligned with the center-vs-surrogate decision
  geometry

Implication:
- further search-tree or GPU optimization will not fix `100/100` by itself
- the next theorem-friendly breakthrough has to come from replacing the current
  pooled selector criterion with a criterion that better separates true centers
  from same-sign surrogate neighborhoods, likely via cross-fit / validation-side
  profiled scoring or an orthogonalized local center-vs-neighborhood score

### 226. Refined root cause for `logical_context`: family miss dominates selector miss

A deeper check on `logical_context` resolves an ambiguity in the diagnosis. We
compared three quantities directly under the same profiled criterion:
- the current best `K<=2` surrogate-family candidate chosen by the selector
- the better high-recall/high-precision `K<=2` candidates already present in the
  family
- the actual true support, fit directly under the same profiled objective

Findings:
- among `K<=2` candidates, the current selector still prefers a support with
  `recall=0.8571`, `precision=0.75` over candidates with
  `recall=1.0`, `precision=0.875`
- however, when the actual true support is fit directly, its profiled criterion
  is **better than the selected `K<=2` candidate**:
  - true support, `steps=40`: `bic_refine ≈ 8028.98`, `recall=1.0`, `precision=1.0`
  - selected `K<=2` candidate, `steps=40`: `bic_refine ≈ 8115.25`, `recall=0.8571`, `precision=0.75`

Interpretation:
- the selector is not the whole problem here
- the more fundamental issue is that the deterministic surrogate family with
  `K<=2` does **not contain the true support**
- in `logical_context`, reaching the true support requires a coordinated purge of
  multiple overlapping surrogate neighborhoods (effectively a `K=3` move)

So for `logical_context`, the primary failure mode is now:
- **family miss first**, selector miss second

This also explains why one-step add-back checks on the selected `K<=2` support do
not rescue it: the local family simply does not yet span the jointly cleaned
true-support candidate.

### 227. Cluster family fixes coverage for `logical_context`, but raw profiled BIC still prefers a smaller surrogate support

To repair the `K<=2` family miss without hand-picking centers, we replaced
independent center subsets by **overlapping surrogate-neighborhood clusters**.
Each active anchor rule induces a deterministic same-sign surrogate neighborhood
(`strict subset +/-1 literal`, `strict superset +/-1 literal`, or same-order
overlap `|U∩V|=|U|-1`). Centers whose neighborhoods overlap are merged into a
connected component; the family then takes the cartesian product of one local
purge option per component.

For `logical_context` this gave:
- components:
  - `{exc:0, exc:11, exc:25}`
  - `{exc:13, exc:29, exc:43}`
  - `{inh:1, inh:7}`
- family size: `27`
- result file:
  [tmp_profiled_cluster_family_logical_context_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_profiled_cluster_family_logical_context_20260415.json)

Most importantly, the family **does contain the exact true support**:
- `cluster::purge::exc:0|purge::exc:43|purge::inh:1`
- `recall=1.0`, `precision=1.0`
- `bic_refine ≈ 8110.08`

However, raw profiled BIC still selected
`cluster::purge::exc:11|purge::exc:43|purge::inh:1`
with:
- `recall=0.7143`, `precision=0.8333`
- `bic_refine ≈ 8027.52`

So after coverage is fixed, the remaining problem is genuinely the **selector**:
the current profiled BIC prefers a smaller surrogate-compressed support even
when the exact true support is already in the finite family.

### 228. Why raw profiled BIC misranks: the truth has slightly better likelihood but loses on per-rule kernel DF

We compared the exact true-support row against the current BIC winner under the
same `steps=40` profiled refit:
- true support:
  - `bic_refine ≈ 8028.98`
  - `ll_train ≈ -27353.30`
  - `ll_val ≈ -3476.27`
  - `df = 92`
- surrogate-compressed winner:
  - `bic_refine ≈ 7947.28`
  - `ll_train ≈ -27353.95`
  - `ll_val ≈ -3476.37`
  - `df = 85`

Interpretation:
- the exact truth has **better train and validation log-likelihood**
- but it pays for one extra rule block in the current `kernel_df`-based BIC
- that raw per-rule kernel complexity penalty is large enough to reverse the
  ordering

This means the current failure is not that the truth fits worse. The truth fits
*better*, but loses because the selector penalizes additional center rules more
than the finite surrogate family geometry warrants.

### 229. Single-rule and block add-back do not by themselves fix the selector

We checked local add-back behavior on the key `logical_context` candidates:
- wrong surrogate-compressed winner:
  `cluster::purge::exc:11|purge::exc:43|purge::inh:1`
- exact truth row:
  `cluster::purge::exc:0|purge::exc:43|purge::inh:1`

Single-rule add-back:
- on the wrong winner, adding back missing true rules (`A`, `E and F`) improves
  `bic_refine` strongly
- but on the exact truth, adding back omitted extras (`A and F`, `B and C`,
  `A and B`, `A and B and D`) also improves `bic_refine` strongly

Block add-back by purged surrogate neighborhood shows the same pattern:
- wrong winner: re-adding the purged `exc:11` block `{A, E and F}` helps
- exact truth: re-adding the purged `exc:0` block `{A and F}` also helps

So the local issue is not “the wrong winner violates one-step local optimality
while the truth does not.” Under the current profiled/BIC criterion, *both*
supports admit locally favorable add-backs. The selector miss is global and
criterion-level, not a simple local-certificate failure.

### 230. A promising theorem-friendly selector shape: coarse family generation by BIC, deep re-ranking by total profiled likelihood

Although raw profiled BIC misranks the exact truth, it still keeps the truth
very near the top of the deterministic cluster family:
- in the `27`-support cluster family, the true support ranks:
  - `#2` by coarse `bic_refine`
  - only `#13` by coarse total profile likelihood

This suggests a practical selector decomposition:
1. use coarse profiled BIC to generate / prune a small near-tie shortlist
2. refit only that shortlist more deeply
3. re-rank the shortlist by **total profiled likelihood**
   (equivalently, a uniform finite-family penalty rather than raw per-rule BIC)

We tested this on the top-`5` coarse-BIC `logical_context` candidates:
- script output:
  [tmp_probe_logical_context_coarse_to_refine_selector_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_probe_logical_context_coarse_to_refine_selector_20260415.json)
- selector:
  - coarse stage: top-`5` by `bic_refine`
  - refine stage: `steps=40`
  - final ranking: smallest deep `neg2ll_total`

Result:
- best candidate became
  `cluster::purge::exc:0|purge::exc:43|purge::inh:1`
- `recall=1.0`
- `precision=1.0`

So the current best theorem-friendly story is:
- deterministic structured family for coverage
- coarse BIC only as a shortlist generator
- deeper re-ranking by total profiled likelihood on the shortlist

This is not yet a full general theorem, but it is the clearest current route to
exact synthetic recovery without heuristic center-picking.

### 231. `kernel_gaussian` diagnosis: current cluster-purge family fails by coverage, not selector

Running the same `cluster family + coarse/deep selector` line on
`paper_kernel_robustness_gaussian_loglink.yaml` showed that the current
structured family is still not expressive enough.

Results:
- coarse cluster family:
  [tmp_probe_loglink_cluster_family_val_selector_kernel_gaussian_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_probe_loglink_cluster_family_val_selector_kernel_gaussian_20260415.json)
- top-`5` coarse-to-refine reranking:
  [tmp_probe_cluster_coarse_to_refine_selector_kernel_gaussian_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_probe_cluster_coarse_to_refine_selector_kernel_gaussian_20260415.json)

Best observed family member by coarse BIC:
- `cluster::purge::exc:25|purge::inh:85`
- `recall=0.6667`, `precision=0.5`

Best observed family member with full recall:
- `cluster::purge::exc:21|purge::inh:85`
- `recall=1.0`, `precision=0.6`

Most importantly, the exact Gaussian truth is **not** in the current
`cluster-purge` family:
- true excitation: `{A, C,D, C,G,H}`
- true inhibition: `{B, E,F, D,F,G}`
- check result: `true_support_in_family = []`

So unlike `logical_context`, the main issue here is not yet selector misranking.
The problem is earlier: the current family grammar does not cover the exact
truth.

### 232. Over-merging is the specific family-grammar bug; direct-conflict keep-family repairs coverage

The current cluster family merges centers whenever their surrogate neighborhoods
overlap or share center-neighbor incidence. This creates large components and
then restricts each component to a single local purge choice. In Gaussian this
is too coarse.

We diagnosed a finer deterministic graph:
- vertices: active centers (rules with non-empty surrogate neighborhood)
- undirected edge `c -- d` iff `d in N(c)` or `c in N(d)`
  (direct surrogate/conflict only)
- candidate support from a chosen center set:
  `fixed_noncenters + selected_centers`

This yields the **direct-conflict keep-family**.

Diagnostics:
- `logical_context`
  - centers: `8`
  - noncenters: `3`
  - independent-set family size: `75`
  - maximal independent sets: `8`
  - exact truth is included, and even appears among maximal independent sets
- `kernel_gaussian`
  - centers: `17`
  - noncenters: `1`
  - independent-set family size: `1617`
  - maximal independent sets: `110`
  - exact truth is included in the full independent-set family
  - exact truth is **not** included among maximal independent sets

Interpretation:
- the cluster family failed because it over-merged neighborhoods and forced
  one purge option per coarse component
- the direct-conflict keep-family is much more faithful: it covers the exact
  truth in both `logical_context` and `kernel_gaussian`
- however, `kernel_gaussian` requires the **full independent-set family**, not
  only maximal independent sets, because the true center set is not maximal

This gives a much cleaner next line:
- keep the same log-link + canonical basis
- replace overlap-cluster purge grammar by a deterministic direct-conflict
  keep-family
- then solve selection over that keep-family (likely via exact / weighted
  independent-set style search rather than brute-force full enumeration)

### 233. `kernel_gaussian` also has a strong optimization-basin issue: truth is excellent under warm-started profiled refit

The `kernel_gaussian` diagnosis sharpened further when we fit the exact true
support under the current profiled objective using an **anchor warm start**
(coefficients and kernel heights initialized from the profiled-path anchor).

File:
- [tmp_probe_gaussian_warm_truth_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_probe_gaussian_warm_truth_20260415.json)

Results for the exact truth:
- `steps=8`: `bic_refine ≈ 14104.75`, `recall=1.0`, `precision=1.0`
- `steps=20`: `bic_refine ≈ 14082.65`, `recall=1.0`, `precision=1.0`
- `steps=40`: `bic_refine ≈ 13722.01`, `recall=1.0`, `precision=1.0`
- `steps=80`: `bic_refine ≈ 12651.36`, `recall=1.0`, `precision=1.0`

For comparison, the best current cluster-family candidate was only:
- `bic_refine ≈ 14380.70`
- `recall=0.6667`, `precision=0.5`

Interpretation:
- `kernel_gaussian` is **not** failing because the profiled objective dislikes
  the exact truth
- instead, two things are simultaneously wrong in the current main line:
  1. the overlap-cluster family misses the truth
  2. cold / shallow optimization badly underestimates the truth when it is fit
     directly

So the next general theorem-friendly direction is no longer just “better
family.” It is:
- deterministic direct-conflict keep-family for coverage
- **anchor-warm profiled refit** for each candidate support
- then selection on that warm-started family

This is a much cleaner decomposition:
- coverage problem: solved by the direct-conflict keep-family
- optimization basin problem: handled by deterministic anchor warm-start
- selector problem: revisit only after these two are fixed

### 234. `logical_context`: direct-conflict keep-family fixes coverage, but profiled BIC again misranks badly

We ran the full warm-started direct-conflict keep-family on
`logical_context`.

File:
- [tmp_direct_keep_family_logical_context_20260415.json](/home/yangmg1216/hnstpp/data/paper_suite/tmp_direct_keep_family_logical_context_20260415.json)

Summary:
- elapsed `≈ 515.07s`
- family size `75`
- exact truth is in the family:
  - `keep::exc:0|exc:25|exc:43|inh:1`
  - `recall=1.0`, `precision=1.0`
  - `bic_refine ≈ 7935.32`

But the raw warm-started profiled `BIC` selector still chose:
- `keep::noncenters`
- support:
  - `E -> T : inhibition`
  - `A and C and D -> T : inhibition`
  - `E and F and G -> T : inhibition`
- `recall=0.4286`, `precision=1.0`
- `bic_refine ≈ 7530.83`

Interpretation:
- the direct-conflict keep-family does repair **coverage**
- but `logical_context` confirms that even after coverage and warm-start are
  fixed, the current profiled `BIC` can still strongly prefer an underspecified
  surrogate support
- so the next bottleneck is not family construction alone; it is again the
  **selection criterion** over the corrected family

### 235. The raw vs `canonical_loglink` performance gap is also a data-regime gap, not just a selector/model gap

We audited the paper-suite configs and the generated datasets themselves.

Config-level conclusion:
- for nearly all raw/loglink pairs, the configs are identical except for:
  - `intensity_model`
  - `path.output_path`
- only `paper_logical_context_loglink.yaml` additionally changes the seed
  (`303 -> 0`)
- rules, base intensities, kernels, and sample counts are otherwise the same

But the generated data regime changes drastically.

Empirical target-event counts per sequence:
- `logical_context`
  - raw: mean target events `≈ 26.21`
  - loglink: `≈ 1.23`
- `logical_shared`
  - raw: `≈ 48.73`
  - loglink: `≈ 1.87`
- `logical_clean_plus`
  - raw: `≈ 38.95`
  - loglink: `≈ 1.86`
- `kernel_gaussian`
  - raw: `≈ 60.32`
  - loglink: `≈ 2.93`
- `kernel_exponential`
  - raw: `≈ 39.88`
  - loglink: `≈ 1.63`
- `kernel_triangular`
  - raw: `≈ 41.02`
  - loglink: `≈ 1.77`

So the new loglink suite is not merely “the same benchmark under a better
model”; it is effectively a much sparser target-event regime.

Code-level reason:
- raw generator:
  - `lambda_k(t) = (b_k + E_k(t)) exp(-I_k(t))`
- loglink generator:
  - `eta_k(t) = log b_k + sum_r s_r w_r p_r(t)`
  - `lambda_k(t) = exp(eta_k(t))`

With `product_bounded` activations, `p_r(t) in [0, 1]`, and the same
`W_pos/W_neg` reused in both generators:
- multiplicative positive rules add linearly to the target rate
- loglink positive rules only multiply the baseline rate

For the paper configs, target baselines are tiny (`0.01` in most suites), so
the same nominal weights induce very different effective rates:
- multiplicative example: `b=0.01`, `W_pos=3.8`, `p=1` gives
  `lambda ≈ 3.81`
- loglink example: `b=0.01`, `w=3.8`, `p=1` gives
  `lambda ≈ 0.01 * exp(3.8) ≈ 0.447`

Hence the raw/loglink comparison is confounded by a major **rate mismatch**.

Interpretation:
- part of the old raw-model advantage came from a structurally easier data
  regime with far more target events
- therefore, raw vs loglink accuracy comparisons are not fully fair unless we
  also rate-match the loglink generator

### 236. 2026-04-15 22:53 KST — Config-only rate matching for canonical_loglink synthetic suite

To remove the major target-rate confound without paying the full regeneration
cost up front, I rewrote the `paper_*_loglink.yaml` configs in a deterministic
config-only pass:

- keep the current canonical loglink intensity form unchanged
- keep all rules / kernels / sample sizes unchanged
- set `seed_loglink := seed_raw`
- set
  `base_intensity[target]_loglink := base_intensity[target]_old * (mean_target_raw / mean_target_current_loglink)`

Because the target type is not used as a source in the paper suites, this is
the cleanest deterministic calibration knob. The rewrite was applied to:

- `paper_ablation_inhibition_only_loglink.yaml`
- `paper_ablation_mixed_sign_loglink.yaml`
- `paper_kernel_robustness_exponential_loglink.yaml`
- `paper_kernel_robustness_gaussian_loglink.yaml`
- `paper_kernel_robustness_triangular_loglink.yaml`
- `paper_logical_clean_plus_loglink.yaml`
- `paper_logical_context_loglink.yaml`
- `paper_logical_shared_loglink.yaml`
- `paper_num_predicates_10_loglink.yaml`
- `paper_num_predicates_20_loglink.yaml`

Key updated target baselines, rounded to human-readable config values:

- `ablation_inhibition_only`: `0.05 -> 0.05`
- `ablation_mixed_sign`: `0.01 -> 0.23`
- `kernel_exponential`: `0.01 -> 0.25`
- `kernel_gaussian`: `0.01 -> 0.21`
- `kernel_triangular`: `0.01 -> 0.23`
- `logical_clean_plus`: `0.01 -> 0.21`
- `logical_context`: `0.01 -> 0.21`
- `logical_shared`: `0.01 -> 0.26`
- `num_predicates_10`: `0.01 -> 0.23`
- `num_predicates_20`: `0.01 -> 0.18`

Clarification:

- the config still stores `base_intensity` on the natural rate scale
- there is **no need** to write log-scaled values into the yaml
- the canonical loglink generator internally uses `log(base_intensity[target])`
  when constructing `eta`, but the config should remain human-readable

Important scope note:

- this was **config-only**
- existing `*_loglink.exact_block.local.pkl` datasets have **not** yet been
  regenerated under the new rate-matched configs
- the previous attempt to regenerate every suite in parallel was too slow
  because each run invoked the full Python/Ogata loglink generator for 5000
  sequences

Result:

- the suite configs are now aligned for a fairer rate-matched loglink
  benchmark
- the next safe step is to regenerate only the most important cases first
  (e.g. `logical_context`, `kernel_gaussian`) rather than all suites at once

Validation check:

- across the 10 rate-matched `paper_*_loglink.yaml` configs, the only numeric
  change in `base_intensity` is the **target** entry
- all non-target `base_intensity` entries are unchanged
- all rule definitions (`id`, `target`, `W_pos`, `W_neg`, `bias`, `condition`)
  are unchanged
- `paper_logical_context_loglink.yaml` also restores the raw seed
  (`0 -> 303`)
- a small smoke run with the updated configs still passes through
  `create_rules_from_config -> generate_loglink_data` without schema/runtime
  errors in the research environment

### 237. 2026-04-16 00:xx KST — Cleaner resolution: keep raw-scale configs and move the rewrite into the generator

The config-only target-base recalibration above was only a temporary stopgap.
The cleaner fix is:

- keep the `paper_*_loglink.yaml` configs on the same intuitive raw scale as
  the original multiplicative configs
- keep all `base_intensity`, `W_pos`, `W_neg`, `bias`, and `condition`
  exactly equal to the raw suite
- set `seed_loglink := seed_raw`
- add
  `synthetic_generation_model: exact_log_multiplicative_rewrite`
  to the loglink configs
- let the synthetic generator interpret this as
  `lambda(t) = exp(log(b + E(t)) - I(t))`

Implementation details:

- added `generate_exact_log_rewrite_data(...)` in [data/synthetic.py] which
  intentionally reuses the battle-tested multiplicative generator because
  `exp(log(b+E)-I)` is algebraically identical to `(b+E)exp(-I)`
- updated
  [tmp_run_loglink_certified_branchbound_suite.py] to route generation through
  `synthetic_generation_model`
- rewrote all 10 `paper_*_loglink.yaml` configs so that:
  - non-target and target `base_intensity` exactly match the raw config
  - all rules exactly match the raw config
  - only `path.output_path`, `intensity_model: canonical_loglink`, and
    `synthetic_generation_model: exact_log_multiplicative_rewrite` differ

Validation:

- across the 10 loglink configs, `changed_base == []` relative to raw
- all rule lists are byte-for-byte equal to raw
- smoke generation passes for
  - `paper_logical_context_loglink.yaml`
  - `paper_kernel_robustness_gaussian_loglink.yaml`

This is the final clean direction:

- configs stay human-readable
- attribution/rate regime are preserved exactly at generation time
- theorem-friendly downstream fitting can still use the separate
  `intensity_model: canonical_loglink` flag

### 238. 2026-04-16 11:xx KST — After fixing the synthetic generation mismatch, the active benchmark algorithm can likely be simplified substantially

The regenerated `canonical_loglink_exact_rewrite` hard cases changed the
picture materially:

- `logical_context_loglink` is now recovered at `100/100` by the current
  active benchmark path
- `kernel_gaussian_loglink` is also recovered at `100/100`

This strongly suggests that much of the earlier algorithmic complexity was
compensating for a **data-generation mismatch**, not a fundamental deficiency
of the canonical log-link learner.

Current active path in
[run_paper_benchmarks.py](/home/yangmg1216/hnstpp/workspace/train/paper_benchmark_active/run_paper_benchmarks.py):

1. estimate source-global kernels
2. initialize rule-specific heights from those global kernels
3. run greedy active-set growth
4. run `family_attribution_refine(...)`
5. run `choose_post_prune_by_penalty_scale(...)` over a small scale grid

Two pieces now look like the main remaining sources of avoidable complexity:

- `family_attribution_refine(...)`
  - useful implementation-wise, but not obviously essential to the theorem
    story
- `choose_post_prune_by_penalty_scale(...)`
  - currently selects over a data-dependent penalty-scale grid
  - this is practical, but less theorem-clean than a single deterministic
    penalty rule

Most of the previously explored machinery now looks unnecessary as a main
paper algorithm:

- surrogate-neighborhood purge families
- dynamic `K` expansion
- cluster/direct-conflict keep-family selectors
- profiled shortlist / deep reranking selectors

Those should be retained only as diagnosis history, not as the main method.

Most promising simplification direction:

- keep the current **global-kernel -> rule-height init -> active-set ->
  post-prune** backbone
- make `family_attribution_refine` optional / implementation-only
- replace penalty-scale grid search with a **single deterministic
  kernel-DF-aware penalty**
- state the main theorem for the simpler backbone

This would make the paper algorithm both:

- simpler to explain and reproduce
- more theorem-friendly

while preserving the empirical `100/100` regime on the corrected synthetic
data, assuming the fixed-penalty version remains competitive.

### 239. 2026-04-16 12:xx KST — Minimal theorem-friendlier path after the synthetic fix

I parameterized the active runner so we can cleanly compare:

- `family_attribution_passes = 0` vs `1`
- `post_prune_kernel_df = False` vs `True`
- fixed post-prune scale `0.6` / `1.0`

Results on the corrected hard cases:

1. `logical_context_loglink`
- no refine + fixed `0.6`: `recall=0.8571`, `precision=1.0`
- no refine + fixed `1.0`: `recall=0.8571`, `precision=1.0`
- refine + no post-prune: `recall=1.0`, `precision=0.7`
- refine + fixed `0.6`: `recall=1.0`, `precision=1.0`
- refine + fixed `1.0`: `recall=0.8571`, `precision=1.0`

2. `kernel_gaussian_loglink`
- no refine + fixed `0.6`: `recall=1.0`, `precision=1.0`
- no refine + fixed `1.0`: `recall=1.0`, `precision=1.0`
- refine + no post-prune: `recall=1.0`, `precision=1.0`
- refine + fixed `0.6`: `recall=1.0`, `precision=1.0`
- refine + fixed `1.0`: `recall=1.0`, `precision=1.0`

Main conclusion:

- the adaptive penalty-scale grid is **not needed** for the corrected
  synthetic suite
- `logical_context` still needs both:
  - one family-attribution refinement pass
  - one post-prune step
- a single fixed penalty constant `0.6` works on both hard cases

So the current best paper line is now much simpler:

1. global kernel estimate
2. rule-height initialization
3. greedy active-set growth
4. **one** family-attribution refinement pass
5. **one fixed** kernel-DF-aware post-prune with scale `0.6`

This is cleaner than the earlier adaptive grid and gives a more theorem-friendly
main selector:

- no data-dependent penalty-grid search
- deterministic finite computation graph
- easier to describe as a fixed-step block-coordinate / forward-backward
  procedure

Practical update:

- `run_paper_benchmarks.py` now exposes these switches explicitly
- the default active path is set to the fixed post-prune scale `0.6`

### 240. The real issue was BIC sample-size scaling, not the coefficient

I revisited the fixed post-prune coefficient question because `0.6` had no
clean theorem story. The key observation is that our BIC penalty had been using

- `n_eff = (# validation target events) + (# quadrature grid points)`

which is much too aggressive for this point-process setting. The quadrature grid
is only a numerical approximation of the integral term; it is not a set of
additional independent observations. The natural independent unit in this suite
is the **sequence**, since the dataset consists of i.i.d. synthetic
trajectories.

So I replaced the BIC sample size with

- `n_eff = # validation sequences`

while keeping the kernel-DF coefficient fixed at `1.0`.

Code change:

- `rule_dependent_kernel_active_set.py`
  - added `bic_sample_size(num_sequences)`
  - switched all BIC / penalty uses from event+grid counts to sequence count
  - threaded `num_val_sequences` through `run_active_set`,
    `family_attribution_refine`, `post_prune_irreducible_rules`, and
    `choose_post_prune_by_penalty_scale`
- `run_paper_benchmarks.py`
  - passes `len(val)` into the active-set / refine / prune stack

Results on the corrected exact-rewrite synthetic data:

1. `logical_context_loglink`
- file: `ctx_ref1_fixed10_seqbic_20260416.json`
- refine `= 1`, kernel-DF post-prune `= True`, fixed coefficient `= 1.0`
- `recall = 1.0`
- `precision = 1.0`

2. `kernel_gaussian_loglink`
- file: `gauss_ref1_fixed10_seqbic_20260416.json`
- refine `= 1`, kernel-DF post-prune `= True`, fixed coefficient `= 1.0`
- `recall = 1.0`
- `precision = 1.0`

So the previous need for `0.6` was not evidence that we needed a mysterious
empirical coefficient. It was evidence that the BIC sample-size term was
mis-scaled.

This gives a much cleaner theorem-friendly main line:

1. global kernel estimate
2. rule-height initialization
3. greedy active-set growth
4. one family-attribution refinement pass
5. one kernel-DF-aware post-prune with **fixed coefficient `1.0`**
6. BIC sample size based on the number of independent sequences

This is a better story than “`0.6` just works”, because it ties the selector to
the actual independence structure of the benchmark.

### 241. Failure cases split cleanly into two post-prune regimes

After the sequence-based BIC fix, the full 11-dataset benchmark still has four
non-`100/100` cases:

- `paper_ablation_excitation_only_loglink`
- `paper_ablation_inhibition_only_loglink`
- `paper_logical_clean_plus_loglink`
- `paper_num_predicates_20_loglink`

I ran two targeted diagnostics on exactly these cases:

1. `refine = 1`, `no post-prune`
2. `refine = 0`, `fixed post-prune scale = 1.0`

Results:

1. `paper_ablation_inhibition_only_loglink`
- baseline (`refine=1`, post-prune): `recall=0.6667`, `precision=1.0`
- `refine=1`, no-prune: `recall=1.0`, `precision=1.0`
- `refine=0`, fixed prune: `recall=0.6667`, `precision=1.0`

2. `paper_num_predicates_20_loglink`
- baseline (`refine=1`, post-prune): `recall=0.8333`, `precision=1.0`
- `refine=1`, no-prune: `recall=1.0`, `precision=1.0`
- `refine=0`, fixed prune: `recall=0.8333`, `precision=1.0`

So for these two cases:

- `family_attribution_refine` is needed to recover the true high-order
  inhibition rule(s)
- the current kernel-DF post-prune then removes those true inhibition rules

This is not a family-coverage failure anymore. It is a **post-prune
over-pruning** problem for high-order inhibition.

For the other two:

3. `paper_ablation_excitation_only_loglink`
- baseline (`refine=1`, post-prune): `recall=1.0`, `precision=0.8571`
- `refine=1`, no-prune: `recall=1.0`, `precision=0.6`
- `refine=0`, fixed prune: `recall=1.0`, `precision=0.8571`

4. `paper_logical_clean_plus_loglink`
- baseline (`refine=1`, post-prune): `recall=1.0`, `precision=0.75`
- `refine=1`, no-prune: `recall=1.0`, `precision=0.6667`
- `refine=0`, fixed prune: `recall=1.0`, `precision=0.75`

So for these two cases:

- `refine` is not the determining factor
- post-prune helps, but not enough
- the remaining errors are false singleton / low-order **excitation shadows**

This cleanly splits the remaining benchmark gap into two theorem-relevant
problems:

1. **high-order inhibition preservation**
   - refine finds the rule
   - post-prune removes it

2. **low-order excitation shadow suppression**
   - post-prune removes some clutter
   - but not enough to eliminate all singleton excitation extras

So the next selector work should focus on the post-prune stage, not on
active-set family generation:

- make post-prune less aggressive against high-order inhibition
- make post-prune more aggressive against singleton excitation shadows

### 242. Semiparametric rule-count BIC fixes inhibition misses but not excitation shadows

To make the selector fairer, I changed the benchmark runner so that post-prune
can still run even when `post_prune_kernel_df=False`. This lets us treat the
kernel shapes as profiled nuisance parameters and penalize only the finite rule
support size.

Interpretation:

- `post_prune_kernel_df=True`:
  kernel-DF-aware BIC
- `post_prune_kernel_df=False`:
  semiparametric / rule-count BIC

I then reran only the four failing datasets with:

- `family_attribution_passes = 1`
- `fixed_post_prune_scale = 1.0`
- `post_prune_kernel_df = False`

Results:

1. `paper_ablation_inhibition_only_loglink_rulebic_eval_20260416.json`
- `recall = 1.0`
- `precision = 1.0`

2. `paper_num_predicates_20_loglink_rulebic_eval_20260416.json`
- `recall = 1.0`
- `precision = 1.0`

3. `paper_logical_clean_plus_loglink_rulebic_eval_20260416.json`
- `recall = 1.0`
- `precision = 0.75`
- unchanged from before

4. `paper_ablation_excitation_only_loglink_rulebic_eval_20260416.json`
- `recall = 1.0`
- `precision = 0.6667`
- worse than the kernel-DF-aware selector

So the fair semiparametric penalty resolves the two **miss-type** failures:

- inhibition-only
- num-predicates-20

This confirms that those errors were caused by over-penalizing high-order
inhibition structure via naive kernel DF.

But the same change does **not** solve the two **extra-type** failures:

- excitation-only
- logical-clean-plus

and in excitation-only it actually gets worse.

This means the remaining excitation-shadow problem is not mainly a DF-scaling
issue. It is a likelihood-side indistinguishability issue: some false
singleton/low-order excitation rules still look genuinely useful after refit.

Current picture:

- miss-type failures: solved by fairer post-prune criterion
- extra-type failures: still need a stronger but rule-symmetric selector

So the next theorem-friendly step should not be a sign-specific hack. It should
be a rule-symmetric criterion that compares each rule by the same profiled
drop-in-likelihood score, but with a better local keep/drop decision rule than
plain BIC alone.

### 243. All-rule single/pair exact drop improves 3 of the remaining 4 cases

To make post-prune genuinely rule-symmetric, I changed it so that:

- all active rules are eligible for pruning (`min_order = 1`)
- candidate drops are evaluated by exact refit
- both single drops and pair drops are considered (`max_drop_size = 2`)

I then reran the four previously failing datasets using:

- `family_attribution_passes = 1`
- `post_prune_kernel_df = False`
- `fixed_post_prune_scale = 1.0`
- `post_prune_min_order = 1`
- `post_prune_max_drop_size = 2`

Results:

1. `fair2_inhib_20260416.json`
- `recall = 1.0`
- `precision = 1.0`

2. `fair2_np20_20260416.json`
- `recall = 1.0`
- `precision = 1.0`

3. `fair2_clean_20260416.json`
- `recall = 1.0`
- `precision = 0.8571`
- improved from `0.75`
- remaining extra: `L -> T : excitation`

4. `fair2_excit_20260416.json`
- `recall = 1.0`
- `precision = 0.6667`
- still poor

Interpretation:

- allowing singleton pruning and pair pruning solves the remaining miss-type
  cases and improves one excitation-shadow case
- `logical_clean_plus` still has one persistent singleton excitation shadow
- `ablation_excitation_only` remains difficult even under fair rule-count BIC
  and all-rule single/pair exact drop

So the remaining unsolved case is no longer “the selector is asymmetric”. The
selector is now symmetric, exact over local single/pair drops, and still leaves
excitation-only surrogates alive. This suggests the last gap is due to
likelihood-level indistinguishability among purely excitatory shadows, not just
DF accounting or order-restricted pruning.

### 244. Component exact subset search resolves excitation-only shadows

I then implemented a more topological selector:

- build same-sign overlap components on the active support
- for each component, evaluate **all subsets** under the same profiled score
- refit the full support after component updates

This is different from the previous single/pair local move:

- it does not force any raw subset/superset exclusivity
- if both `AB` and `ABC` are truly needed, the component subset search can keep
  both because they are just one candidate subset among many
- the change is in the support-space topology, not in a sign-specific rule

Implementation:

- `build_overlap_components(...)`
- `component_subset_search(...)`
- runner flags:
  - `component_subset_search=True`
  - `component_same_sign_only=True`

Rerun setting:

- `family_attribution_passes = 1`
- `post_prune_kernel_df = False`
- `fixed_post_prune_scale = 1.0`
- `post_prune_min_order = 1`
- `post_prune_max_drop_size = 2`
- `component_subset_search = True`

Results:

1. `comp_inhib_20260416.json`
- `recall = 1.0`
- `precision = 1.0`

2. `comp_np20_20260416.json`
- `recall = 1.0`
- `precision = 1.0`

3. `comp_clean_20260416.json`
- `recall = 1.0`
- `precision = 0.8571`
- remaining extra: `L -> T : excitation`

4. `comp_excit_20260416.json`
- `recall = 1.0`
- `precision = 1.0`

This is important:

- `excitation_only`, which was the hardest pure-positive surrogate case, is now
  solved
- so the previous failure really was a support-topology problem, not merely a
  bad coefficient or an asymmetric selector

Current remaining gap:

- only `logical_clean_plus` still has one persistent singleton excitation shadow

So the main story is now:

1. fix synthetic regime mismatch
2. use sequence-based BIC
3. treat kernels as profiled nuisance for post-prune
4. allow all-rule pruning
5. replace Hamming-local support moves with component exact subset search

At this point the benchmark gap is down to one residual excitation singleton.

### 245. Cross-Sign Component Search And Canonicalized Component Scoring On `logical_clean_plus` (2026-04-16)

Goal:

- test whether the remaining `logical_clean_plus` failure is still a same-sign
  topology issue, or whether the problem is already deeper than component
  search

Two deterministic follow-up runs were checked on
`paper_logical_clean_plus_loglink`:

1. cross-sign overlap components
  - exact same component subset search as `### 244`
  - but with `component_same_sign_only = False`
2. cross-sign overlap components + canonicalized component scoring
  - before component subset scoring, transform the current active-support
    arrays by a descending Mobius-style residualization over strict supersets
  - use these canonicalized arrays only for the component subset score
  - final support is still refit by the same profiled optimizer

Result files:

- `cleanplus_crosssign_20260416.json`
- `cleanplus_crosssign_canonical_20260416.json`

Observed result:

- both runs stayed at
  - `recall = 1.0`
  - `precision = 0.8571`
  - remaining extra: `L -> T : excitation`

Interpretation:

- forcing cross-sign overlap rules into the same component was not enough
- even a deterministic Mobius-style residualization inside the final component
  selector was not enough
- therefore the last `logical_clean_plus` error is no longer a final-stage
  topology/search issue
- it is a deeper active-support / raw-feature geometry issue:
  - by the time final selection runs, the support and current kernel geometry
    already make `L -> excitation` look genuinely useful next to
    `JKL -> inhibition`

Practical conclusion:

- component topology changes solved `excitation_only`
- but `logical_clean_plus` requires moving the signed-canonical idea earlier
  than final-stage subset search
- the next real fundamental direction is not more local pruning
  - it is signed-canonical candidate construction or a true canonical
    interaction basis earlier in the active-set path

### 246. Active-Set Candidate Scoring With Global Exact-State Canonicalization Fails (2026-04-16)

Goal:

- move the canonical idea earlier than the final-stage selector
- test whether active-set growth itself should score candidates on a global
  Mobius / exact-state basis instead of raw strict-AND features

Implementation:

- added `canonicalize_arrays_by_subset(...)`
- built `active_score_canonicalize=True` option in
  `run_paper_benchmarks.evaluate_config(...)`
- active-set candidate gain was computed from the globally canonicalized
  `init_arrays`, while support refit still used the original profiled optimizer

Main runs:

- `cleanplus_activecanonical_20260416.json`
- `excit_activecanonical_20260416.json`

Results:

- `logical_clean_plus`
  - `recall = 0.6667`
  - `precision = 0.6667`
  - missing true inhibition rules:
    - `E and F -> T : inhibition`
    - `J and K and L -> T : inhibition`
  - extra inhibition rules:
    - `D and E and F -> T : inhibition`
    - `E and F and H -> T : inhibition`
- `ablation_excitation_only`
  - collapsed to the empty model
  - `recall = 0.0`
  - `precision = 0.0`

Interpretation:

- global exact-state canonicalization is too aggressive when used directly as
  the active-set score basis
- it removes too much lower-order mass from the candidate features before the
  support path is stabilized
- under finite-sample synthetic counts, this causes genuine rules to lose their
  screening gain and lets nearby higher-order alternatives distort the path
- therefore the correct “more fundamental” direction is **not** to replace the
  whole active-set score by a global exact-state basis

Conclusion:

- final-stage-only canonicalization was too late
- but global exact-state active scoring is too early / too aggressive
- the promising middle ground is now:
  - canonicalization only inside local ambiguity blocks / conflict families, or
  - signed-canonical candidate augmentation rather than full candidate-score
    replacement

### 247. Blockwise Canonical Exact Search Preserves `excitation_only` But Still Does Not Fix `logical_clean_plus` (2026-04-16)

Goal:

- avoid the failure of global active-score canonicalization
- keep the successful raw active-set path
- but perform an exact deterministic search over local ambiguity blocks, where
  each block compares:
  - raw subset family
  - canonicalized subset family
  under the same profiled score

Implementation:

- added `feature_override_arrays` to `optimize_active_set_torch(...)`
  so fixed transformed local feature families can be refit exactly with the
  same likelihood/BIC objective
- added `blockwise_canonical_exact_search(...)`
  - deterministic overlap components
  - exact subset enumeration inside each component
  - for every subset, compare both raw and locally canonicalized feature
    representations
  - then refit the selected support back in the original raw model

Runs:

- `cleanplus_blockcanon_20260416.json`
- `excit_blockcanon_20260416.json`

Results:

- `logical_clean_plus`
  - `recall = 1.0`
  - `precision = 0.8571`
  - remaining extra unchanged:
    - `L -> T : excitation`
- `ablation_excitation_only`
  - `recall = 1.0`
  - `precision = 1.0`

Interpretation:

- this is an important negative result
- the “middle ground” blockwise canonical exact search is stable and does not
  destroy the successful excitation-only benchmark
- but even with exact local block family search, `logical_clean_plus` still
  prefers retaining `L -> excitation`

Conclusion:

- the final remaining `logical_clean_plus` ambiguity is not just:
  - same-sign topology
  - cross-sign topology
  - approximate local subset scoring
- it persists even under exact local raw-vs-canonical block comparison
- therefore the root issue is deeper:
  - the current raw active support and kernelized likelihood geometry make the
    lower-order opposite-sign shadow genuinely competitive
- the next fundamental direction must move beyond final local block repair,
  toward earlier signed-canonical candidate construction / support path design

### 248. `logical_clean_plus` Is A Representation / Selection-Object Failure, Not A Search Failure (2026-04-16)

Direct support-pair comparison was run on the regenerated
`paper_logical_clean_plus_loglink` dataset:

- exact true support
- exact true support plus the persistent extra `L -> T : excitation`

Both supports were refit directly under the current profiled optimizer.

Result file:

- `cleanplus_support_compare_20260416.json`
- `cleanplus_support_compare_kerneldf_20260416.json`

Observed scores:

- rule-count BIC
  - true support: `67806.52`
  - true + `L`: `67662.58`
- kernel-df BIC
  - true support: `68253.97`
  - true + `L`: `68147.32`

Interpretation:

- the extra-support model is preferred not only by the current search path
- it is preferred even under direct fixed-support refit
- and this remains true under both
  - rule-count BIC
  - kernel-df BIC

Therefore:

- the remaining failure is not a local-search failure
- not a component-family coverage failure
- and not merely a too-weak final-stage pruning family

The missing piece is more fundamental:

- in the current raw strict-AND representation, the support-kernel model class
  does not identify the intended “pure interaction” object strongly enough
- `L -> excitation` plus `JKL -> inhibition` is a genuinely better fit under the
  current representation and selection object than the intended pure
  `JKL -> inhibition` model

Consequence:

- no amount of search strengthening alone can guarantee `100/100`
- the next general theorem-friendly step must change the representation /
  selection object itself, not merely the search procedure

Direct representation check:

- `cleanplus_repr_compare_20260416.json`

Scores:

- true support, raw representation: `67806.52`
- true + `L`, raw representation: `67662.58`
- true support, naive exact-state canonical override: `70254.37`
- true + `L`, naive exact-state canonical override: `70107.91`

Interpretation:

- naive global exact-state canonicalization does not fix the issue either
- it makes both models worse, but still prefers the extra-support model
- therefore the missing piece is not “just use canonical features”
- the required change must be a more structured signed-canonical estimand, not
  a raw global feature replacement

### 249. Inference-Link Mismatch Is Real But Not The Whole Story (2026-04-16)

Hypothesis:

- because the regenerated `canonical_loglink_exact_rewrite` datasets are drawn
  from the exact multiplicative rewrite, perhaps the remaining gap is mainly a
  link mismatch:
  - generation: exact multiplicative rewrite
  - inference: canonical log-link

To test this, `evaluate_config(...)` was extended with a deterministic
`intensity_model_override` so the same dataset can be fit under a different
inference model without changing configs.

Representative runs:

- `cleanplus_multinfer_20260416.json`
- `context_multinfer_20260416.json`
- `gauss_multinfer_20260416.json`

Results:

- `logical_clean_plus`, multiplicative inference
  - `recall = 1.0`
  - `precision = 0.8571`
  - same remaining extra:
    - `L -> T : excitation`
- `logical_context`, multiplicative inference
  - `recall = 1.0`
  - `precision = 0.7778`
  - worse than the current canonical-loglink path
- `kernel_gaussian`, multiplicative inference
  - `recall = 1.0`
  - `precision = 1.0`

Interpretation:

- inference-link mismatch is real for some cases
  - especially `kernel_gaussian`
- but it is not the full explanation
  - `logical_clean_plus` does not improve
  - `logical_context` degrades

Conclusion:

- there is no single global “just switch the link” fix
- the remaining issue is more structural than a mere mismatch between
  generation and inference link
- this supports the view that the missing piece is a better signed interaction
  estimand / representation, not just a different global link choice

### 250. Signed Closure Chart Exact Search Is Theoretically Cleaner But Computationally Explodes (2026-04-16)

To test a more theorem-friendly representation-level fix without adding more
ad-hoc local heuristics, a deterministic `signed_chart_exact_search(...)` was
implemented.

Idea:

- keep the raw active-set path as a recall-preserving proposal stage
- build deterministic signed closure charts around active high-order rules
- inside each chart, enumerate a ternary family:
  - off
  - excitation
  - inhibition
- refit each chart state exactly with the same profiled score

Important implementation detail:

- `optimize_active_set_torch(...)` was extended with
  `feature_override_arrays` so chart-local canonicalized arrays can be used
  without rewriting the global support path

Smoke runs were launched for:

- `logical_clean_plus`
- `ablation_excitation_only`

However, both runs failed to finish in a practical time window.

Observed runtime signal:

- both processes were still running after roughly 30 minutes
- no result JSON had been written yet

Structural reason:

- for both supports, the deterministic signed closure charts were:
  - `num_charts = 4`
  - chart sizes = `[3, 3, 7, 7]`
- this implies
  - total ternary chart states
  - `sum_j 3^{|chart_j|} = 4428`
- each chart state triggers an exact profiled refit, so the current
  implementation is computationally prohibitive even on just the remaining
  hard cases

Interpretation:

- the representation idea may still be directionally correct
- but the present exact chart-state implementation is too expensive to be the
  main practical algorithm
- this is now a genuine complexity barrier, not just an implementation glitch

Conclusion:

- deterministic signed closure chart selection is cleaner than the earlier
  patchwork search refinements
- but exact ternary chart enumeration with full refits is not practical
- the next viable step must preserve the representation insight while avoiding
  `3^m` exact state evaluation for every closure chart

### 251. Dead Canonicalization Branches Removed From The Active Code Path (2026-04-17)

Decision:

- the following branches are now considered abandoned and were removed from the
  active benchmark code:
  - active-score canonicalization
  - component canonicalized scoring
  - blockwise canonical exact search
  - signed closure chart exact search

Code cleanup:

- `run_paper_benchmarks.py`
  - removed the associated experimental flags and dispatch paths
- `rule_dependent_kernel_active_set.py`
  - removed the experimental canonical/search helpers and the temporary
    `feature_override_arrays` path in `optimize_active_set_torch(...)`

Current baseline we are keeping:

- raw active-set proposal
- `family_attribution_refine` (1 pass)
- fair rule-count BIC post-prune
- small exact component subset search over same-sign overlap components

Reason:

- this line remains simple enough to reason about
- it already gives best-known performance on the regenerated synthetic suite
- it avoids carrying dead experimental branches that did not improve the final
  benchmark

### 252. Literature-Backed Diagnosis Of The Remaining `logical_clean_plus` Failure (2026-04-17)

Relevant literature points to the same diagnosis from several directions.

1. Interaction identifiability:

- Lengerich et al. (2020) and Hooker (2007) emphasize that lower-order and
  higher-order effects are not identifiable in a raw additive interaction basis
  unless one works with a canonical decomposition relative to an input law.
- In our setting, the logical indicators are strongly dependent and nested:
  `phi_JKL <= phi_L` pointwise.
- Therefore `L` and `JKL` are not orthogonal coordinates for a signed effect.

2. Why hierarchy is not the right repair:

- Bien et al. (2013) and Taylor-Rodriguez et al. (2015) use strong/weak
  heredity constraints to stabilize interaction selection.
- Those assumptions are helpful in many regression settings, but they are not
  appropriate as truth assumptions for our benchmark because the synthetic truth
  allows a higher-order interaction without a lower-order main effect.

3. Why point-process penalties matter:

- Hansen, Reynaud-Bouret, and Rivoirard (2015) show that in multivariate point
  processes, penalty weights should track the martingale variance of each
  dictionary element rather than use a uniform penalty.
- This matches our failure pattern:
  a frequent lower-order shadow like `L`
  can carry a large empirical score fluctuation and absorb residual mismatch
  from the profiled nuisance fit, while a fixed rule-count penalty treats it
  the same as a much rarer high-order rule.

Mathematical diagnosis for the remaining `L` shadow:

- compare the true support `S` against `S' = S ∪ {L}`.
- the selected model minimizes a penalized empirical criterion, so `S'` wins
  whenever `2(ell_n(S') - ell_n(S)) > Delta_pen`.
- the local score of `L` at the smaller support is
  `U_n(L | S) = sum_i X_L(t_i) - ∫ X_L(t) lambda_S(t) dt`,
  which is a martingale term plus approximation/profiling bias.
- because `X_L` is active on a much larger set than `X_JKL`, its predictable
  variation is much larger.
- a uniform rule-count BIC ignores this variance scale, so a frequent lower-
  order shadow can beat the fixed penalty even when the structural truth has no
  pure `L` effect.

Conclusion:

- the last failure is best viewed as a score-normalization / estimand problem,
  not a search problem
- the most promising next theorem-friendly direction is not more local search,
  but a self-normalized or variance-adaptive rule score / penalty derived from
  point-process martingale concentration

### 253. A Naive Self-Normalized Rule Penalty Does Not Fix The Remaining Failure (2026-04-17)

Tested idea:

- keep the same active-set / refine / component-subset-search pipeline
- replace the uniform rule-count BIC in post-prune / component selection by a
  deterministic self-normalized rule penalty derived from validation-side
  predictable variation

Prototype weight:

- for each active rule `r`,
  `w_r = sqrt(2 V_r log n) + (B_r log n) / 3`
- where `V_r` is estimated from
  `∫ X_r(t)^2 lambda_hat(t) dt`
  on the validation grid, and `B_r = ||X_r||_∞`

Representative results:

- `cleanplus_selfnorm_20260417.json`
  - `recall = 0.8333`
  - `precision = 0.4545`
  - missed the true `JKL -> inhibition`
  - introduced many new false rules
- `excit_selfnorm_20260417.json`
  - `recall = 1.0`
  - `precision = 0.5455`
  - introduced several false inhibition triplets

Interpretation:

- the high-level diagnosis from the literature was useful
- but this direct translation into a weighted L0-style support score is too
  naive
- the variance term alone does not distinguish genuine signed interactions from
  badly oriented lower-order projections
- in practice it over-penalizes the true sparse rule while leaving room for
  structurally wrong supports

Decision:

- this self-normalized penalty branch is not retained in the active code
- the active implementation is reverted to the previous best-known baseline

### 254. clean_plus Shows A Genuine Lower-Order Projection Effect For `L` (2026-04-17)

Question:

- is the last `logical_clean_plus` failure just a search/pruning miss, or does
  the raw basis itself make `L -> excitation` look statistically useful?

Cheap geometry diagnosis:

- file: `cleanplus_activation_overlap_20260417.json`
- used the *true* generator-side `product_bounded` activations for:
  - singleton source activity from `L`
  - true inhibitory triplet activity from `JKL`
- compared how often these activations are present on:
  - a uniform validation grid
  - actual validation target event times

Key numbers:

- at threshold `0.01`
  - `grid_L_active_frac = 0.4300`
  - `event_L_active_frac = 0.4594`
  - enrichment `= 1.0684`
  - `grid_JKL_active_frac = 0.0799`
  - `event_JKL_active_frac = 0.0582`
  - enrichment `= 0.7283`
  - `grid_(L and not JKL)_active_frac = 0.3500`
  - `event_(L and not JKL)_active_frac = 0.4012`
  - enrichment `= 1.1461`

Interpretation:

- `JKL` behaves exactly like a true inhibitory interaction should:
  it is *depleted* at target event times relative to the background grid
- but the raw lower-order feature `L` is active on a much broader set, and
  the subset where `L` is active while `JKL` is inactive is actually
  *enriched* at target event times
- therefore, in the raw rule dictionary, `L` is not merely a harmless shadow;
  it has a positive projected association with the target process

Consequence:

- the last `logical_clean_plus` failure is not well described as
  "the algorithm failed to find the true support"
- rather, under the current raw-basis selection object, the benchmark truth
  `JKL -> inhibition` and the statistical projection target are misaligned
- this strongly supports the view that the remaining gap is an
  estimand/representation problem, not a search-radius problem

Research implication:

- further local search or penalty tuning is unlikely to solve the last failure
  in a principled way
- the next theorem-friendly step should be to formalize the distinction
  between:
  - benchmark truth: raw config support
  - statistical truth: nonredundant / projection-stable support under the
    current model class

### 255. The Remaining clean_plus Failure Is Not Seed-Stable At Fixed Sample Size (2026-04-17)

Question:

- is the last `logical_clean_plus` failure just an unlucky sample at the
  benchmark seed, or does the current score systematically prefer
  `true + (L -> excitation)`?

Experiment:

- regenerated `logical_clean_plus` at the same sample size `n = 5000`
- seeds: `111, 211, 311, 411`
- for each seed, compared two exact supports by direct refit:
  - `S_true`
  - `S_true_plus_L = S_true ∪ {L -> excitation}`
- result file: `cleanplus_support_compare_multiseed_20260417.json`

Results:

- seed `111`
  - `BIC(true) = 61134.61`
  - `BIC(true+L) = 61116.19`
  - prefers `true+L`
- seed `211`
  - `BIC(true) = 62004.25`
  - `BIC(true+L) = 62047.32`
  - prefers `true`
- seed `311`
  - `BIC(true) = 61198.21`
  - `BIC(true+L) = 61265.16`
  - prefers `true`
- seed `411`
  - `BIC(true) = 60401.81`
  - `BIC(true+L) = 60447.66`
  - prefers `true`

Summary:

- only `1 / 4` seeds preferred the extra-`L` model
- the original benchmark seed `111` is exactly that exceptional case

Interpretation:

- the remaining `clean_plus` failure is **not** a stable structural property of
  the current score at this sample size
- there is still a meaningful projection effect in the raw basis
  (see `### 254`)
- but the actual benchmark failure appears to depend materially on finite-sample
  fluctuation / seed realization

Consequence:

- it is no longer accurate to describe the last failure as purely structural
- the most honest diagnosis is:
  - raw-basis projection creates vulnerability
  - finite-sample randomness determines whether that vulnerability actually
    flips the support comparison

### 256. Final Code Path Cleanup For The Paper Benchmark Active Folder

We performed a final cleanup pass on the active benchmark code to keep only the
theorem-friendly mainline.

Kept as the final path:

- global source-kernel initialization
- raw active-set proposal
- `family_attribution_refine(..., passes=1)`
- `post_prune_irreducible_rules(...)` with
  - rule-count BIC
  - `penalty_scale = 1.0`
  - `min_order = 1`
  - `max_drop_size = 2`
- same-sign overlap `component_subset_search(...)`

Removed from the active path:

- penalty-scale grid search
- evaluation-time intensity override
- cross-sign toggle as a top-level benchmark option
- stale CLI branches that no longer match the final benchmark path

Code-level audit notes:

- `validation_bic_from_arrays(...)` had an inconsistent empty-support branch
  that did not match `fit_constant_bic(...)`; it was replaced with an explicit
  error since the final active path never evaluates empty support through this
  helper
- `run_paper_benchmarks.py` now defaults to the final
  `canonical_loglink_exact_rewrite` 11-benchmark suite
- `paper_benchmark_active/README.md` now documents only the retained final path

### 257. Five-Seed Collapse Is Primarily A Link-Mismatch Problem

After renaming and regenerating the final synthetic suite across seeds
`111/222/333/444/555`, we ran the retained final benchmark path on all 55
seed-benchmark pairs. The result was a broad precision collapse:

- overall mean recall `~0.929`
- overall mean precision `~0.543`

The important pattern was not random variation but repeated extra rules across
seeds and across multiple benchmark families. For example:

- `ablation_excitation_only` repeatedly added the same false inhibition triples
- `kernel_*` and `num_predicates_*` repeatedly added the same `ACD/ACG/ACH`
  inhibition shadows plus lower-order excitation shadows

This strongly suggested a structural issue rather than a search-radius issue.

The key audit finding is that the final synthetic configs currently combine:

- `synthetic_generation_model: exact_log_multiplicative_rewrite`
- `intensity_model: canonical_loglink`

So the data are generated from multiplicative semantics

`lambda(t) = (b + E_+(t)) exp(-I_-(t))`

while the benchmark learner fits a log-link model

`lambda(t) = exp(log b + E_+(t) - I_-(t)) = b exp(E_+(t) - I_-(t))`.

These are not equivalent. When `E_+(t)` is not tiny relative to `b`, the
log-link learner can compensate for misspecification by adding opposite-sign
shadow rules. This matches the repeated extra-rule patterns observed in the
five-seed run.

Representative re-runs with multiplicative inference on the same datasets gave
strong evidence that this is the main failure mode:

- `ablation_excitation_only`, seed 222:
  - canonical-loglink inference: `1.0 / 0.5455`
  - multiplicative inference: `1.0 / 1.0`
- `kernel_exponential`, seed 222:
  - canonical-loglink inference: `1.0 / 0.5455`
  - multiplicative inference: `1.0 / 1.0`
- `logical_clean_plus`, seed 444:
  - canonical-loglink inference: `0.6667 / 0.3333`
  - multiplicative inference: `1.0 / 1.0`
- `logical_context`, seed 111:
  - canonical-loglink inference: `0.8571 / 0.6`
  - multiplicative inference: `1.0 / 0.875`

Updated interpretation:

- the current five-seed collapse is primarily a model-class mismatch problem
- earlier efforts that focused on search/pruning/partial canonicalization were
  mostly trying to repair consequences of this mismatch
- the simplest theorem-friendly fix is to align inference with the actual final
  synthetic data-generating semantics, i.e. use multiplicative inference for
  the final synthetic suite

### 258. Final Synthetic Suite Was Re-Centered On True Canonical Log-Link (2026-04-18)

We audited the original modeling goal again and confirmed that the intended
paper direction is not multiplicative recovery. The actual goal is a signed
logical TPP with canonical log-link dynamics, i.e.

`lambda_k(t) = b_k exp(E_k(t) - I_k(t))`.

That choice was originally motivated by the desire to let excitation and
inhibition compete on the same log-scale, rather than keeping inhibition in the
separate multiplicative damping term of

`lambda_k(t) = (b_k + E_k(t)) exp(-I_k(t))`.

So the final synthetic suite was rebuilt to restore end-to-end semantic
consistency:

- `synthetic_generation_model = canonical_loglink`
- `intensity_model = canonical_loglink`

The old mixed configuration

- `synthetic_generation_model = exact_log_multiplicative_rewrite`
- `intensity_model = canonical_loglink`

was discarded as an incorrect detour.

We also re-audited the activation semantics:

- generation uses `product_bounded`
- learning uses the same bounded source activity
  `a_s(t) = 1 - exp(-z_s(t))`
- conjunction rule features are products of these source activities on both
  sides

so after this correction the generator and learner are aligned both at the
intensity level and at the rule-feature level.

### 259. Canonical Generator Was Rewritten To Exploit The True Benchmark Structure (2026-04-18)

Once we restored the true canonical generator, the naive all-type thinning path
became too slow. But the actual benchmark structure is much simpler than a
general multivariate nonlinear TPP:

- all rules point to a single target event type
- non-target source types are not themselves targets of rules
- therefore all non-target source processes are just independent homogeneous
  Poisson processes with their baseline rates
- conditional on those source histories, the target process is a 1D
  inhomogeneous Poisson process with canonical log-link intensity

This means the final synthetic family admits an exact factorization:

1. generate all non-target source event streams independently from their
   baseline Poisson laws
2. condition on those source histories
3. generate the target event stream from the resulting 1D canonical intensity

This change is exact for the current benchmark family, not a heuristic
approximation. The canonical fast path now lives in `data/synthetic.py`.

After this rewrite the seed-111 canonical suite was generated successfully, and
the resulting benchmark-specific target means were:

- `ablation_excitation_only`: `69.8304`
- `ablation_inhibition_only`: `3.2662`
- `ablation_mixed_sign`: `29.6946`
- `kernel_exponential`: `34.8158`
- `kernel_gaussian`: `37.9278`
- `kernel_triangular`: `29.6946`
- `logical_clean_plus`: `25.8520`
- `logical_context`: `22.1886`
- `logical_shared`: `39.1638`
- `num_predicates_10`: `29.7414`
- `num_predicates_20`: `21.2120`

This confirmed that the canonical suite is not globally too sparse. Most
benchmarks remain in a moderate-count regime.

### 260. Canonical Seed-111 Failures Are No Longer A Mismatch Problem (2026-04-18)

Running the retained benchmark pipeline on the corrected canonical seed-111 data
produced:

- mean recall `0.7749`
- mean precision `0.6629`
- kernel mean `L1 = 0.2927`
- kernel mean `L2 = 0.1626`

The failure pattern changed qualitatively relative to the previous mixed
generator/inference setting:

- the previous mixed-semantics benchmark over-selected many extra rules
- the corrected canonical benchmark instead exhibits several cases of severe
  under-selection or empty-model collapse

The two strongest failures are:

- `ablation_excitation_only`: `0.0 / 0.0`
- `kernel_gaussian`: `0.0 / 0.0`

This is especially important because neither dataset is impossibly sparse:

- `ablation_excitation_only` target mean: `69.83`
- `kernel_gaussian` target mean: `37.93`

So the remaining failure cannot be explained by event scarcity alone. The
problem moved into the learner itself.

### 261. Current Canonical Add-Step Fails Because The Screening Surrogate Is Geometrically Unstable (2026-04-18)

After restoring true canonical generation, we re-diagnosed the learner and the
main issue is now the first candidate-add stage in `run_active_set(...)`.

The active path currently works as follows:

1. build candidate subset features from the rule-specific bounded activity basis
2. score each inactive rule with `rule_score(...)`
3. select the best candidate and sign
4. run exact refit via `optimize_active_set_torch(...)`
5. keep the candidate only if the refit BIC improves

For the corrected canonical suite, the failure is not that the learner finds no
high-gain candidates. The opposite happens in the hardest cases:

- broad-kernel or highly overlapping positive shadows receive extremely large
  one-step surrogate gains
- those candidates then enter exact canonical refit with inflated coefficient
  initializations
- the subsequent exact fit becomes numerically pathological or produces a much
  worse BIC
- because the first add fails, the active set can terminate at the empty model

This explains why `kernel_gaussian` and `ablation_excitation_only` can collapse
even though their target counts are more than sufficient.

We tested one mathematically cleaner branch: replace the canonical surrogate
candidate score with a more exact profiled one-rule likelihood-ratio criterion.
This was appealing because it would compare each inactive rule under a common
profiled canonical objective instead of the current local quadratic surrogate.

However, in direct targeted runs this branch was too expensive to be practical
in the current implementation. Even when restricted to a handful of failing
benchmarks and throttled to small thread counts, it did not finish in a
reasonable amount of time. Since it did not produce usable benchmark outputs,
it was removed from the active code path.

So the current best diagnosis is:

- the corrected canonical synthetic data are not the main bottleneck
- the retained canonical learner is failing because the current
  score-and-refit interaction is numerically and geometrically unstable at the
  candidate addition stage
- an all-candidate exact profiled forward step is principled but too expensive
  in naive form

Most plausible theorem-friendly next direction:

- keep the fast proposal step
- make the canonical refit itself separation-safe / bias-reduced
- then re-evaluate whether the catastrophic first-add failures disappear

This points toward bias-reduced canonical fitting (e.g. Jeffreys/Firth-type
stabilization) rather than increasingly expensive exact candidate search.

### 262. Jeffreys/Firth-Type Bias Reduction Did Not Fix The Canonical Failures (2026-04-18)

The next attempt was exactly the bias-reduced canonical refit suggested above:
add a Jeffreys/Firth-style information penalty inside
`optimize_active_set_torch(...)` for the canonical branch, so that canonical
refits become less prone to separation-like coefficient explosions after
admitting a strong excitation candidate.

This was tested on the main canonical failures:

- `ablation_excitation_only`
- `ablation_inhibition_only`
- `kernel_gaussian`
- `logical_context`

Observed outcome:

- `ablation_excitation_only` stayed at `0 / 0`
- `kernel_gaussian` stayed at `0 / 0`
- `logical_context` stayed at `0.857 / 0.75`
- `ablation_inhibition_only` stayed at `0.833 / 0.833`

So the canonical problem is not just a refit-bias issue. The wrong candidates
are already being favored before that stabilization can help.

This branch was removed from code.

### 263. Empty-Support Exact Profile Initialization Was The First Real Canonical Improvement (2026-04-18)

The next clean correction was much smaller: only when the active set is empty,
initialize the first candidate coefficient by solving the exact
one-dimensional profiled canonical problem rather than using the old local
quadratic coefficient surrogate.

This keeps the model class unchanged and solves an exact scalar subproblem
under the current canonical likelihood.

Observed outcome:

- `ablation_excitation_only`: `0 / 0 -> 1.0 / 1.0`
- `ablation_inhibition_only`: `0.833 / 0.833 -> 1.0 / 1.0`
- `kernel_gaussian`: `0 / 0 -> 1.0 / 0.667`
- `logical_context`: unchanged at `0.857 / 0.75`

Interpretation:

- the first canonical add really was partly failing because the old surrogate
  coefficient initialization was too crude;
- however, this alone does not solve the nested/context ambiguity cases.

### 264. Global Coefficient Warm Starts Helped Some Cases But Were Not A Stable Mainline (2026-04-18)

I then tested a more aggressive but still structured stabilization:

- before the full joint kernel+coefficient optimization,
- solve a canonical fixed-feature coefficient-only subproblem,
- then warm start the full refit from that solution.

Observed outcome:

- `kernel_gaussian`: improved to `1.0 / 0.75`
- `logical_context`: improved to `1.0 / 0.875`
- `ablation_inhibition_only`: improved to `1.0 / 1.0`
- but `ablation_excitation_only` degraded relative to the empty-support-only
  improvement, down to about `0.667 / 0.444`

Interpretation:

- coefficient warm starts do improve canonical numerical stability;
- but applying them globally is not yet a clean net improvement because the
  pure excitation case regresses.

This branch was therefore removed from the active code.

### 265. Exact Component Refit Search Was Too Expensive To Keep (2026-04-18)

To improve the remaining precision errors after admission, I replaced the cheap
component subset comparison with exact subset refits inside
`component_subset_search(...)`.

This direction is principled:

- same candidate family
- same finite support search
- exact refit for each subset rather than frozen-array comparison

But in practice it was too expensive:

- targeted runs became materially heavier,
- benchmark outputs did not arrive quickly enough to justify the added code,
- and it risked turning the canonical branch into another computational dead
  end.

So this branch was removed again.

### 266. Current Active Direction: Exact One-Dimensional Profile Gain For Canonical Screening (2026-04-18)

The current branch under test is the cleanest compromise found so far:

- keep the candidate family unchanged
- keep the canonical model unchanged
- replace the old quadratic canonical screening surrogate by the exact
  one-dimensional profiled gain under the current fixed-feature support

For a candidate feature `f` and sign `s in {+1, -1}`, this evaluates the
profiled canonical gain

- after profiling out `mu` exactly,
- without doing a full exact refit for every inactive rule.

This is attractive because it is:

- much more faithful than the old quadratic screening surrogate
- much cheaper than all-candidate exact forward refits
- still theorem-friendly and sign-symmetric

This branch is now the main thing being tested.

Interim results on the canonical seed-111 failures:

- `ablation_inhibition_only`: `1.0 / 1.0`
- `logical_context`: `1.0 / 0.7778`
- `kernel_gaussian`: `1.0 / 0.6`

relative to the current canonical baseline:

- `ablation_inhibition_only`: `0.833 / 0.833`
- `logical_context`: `0.857 / 0.75`
- `kernel_gaussian`: `0 / 0`

So the direction is clearly helping:

- it removes the catastrophic empty-model collapse in `kernel_gaussian`
- it restores full recall in `logical_context`
- it fully fixes `ablation_inhibition_only`

The remaining issue is precision:

- `kernel_gaussian` still adds several broad/overlap shadows
- `logical_context` still keeps two extra excitation shadows

This suggests that the screening surrogate really was one of the main
canonical bottlenecks, but not yet the only one.

### 267. Fixed-Feature Canonical Warm Starts And Fixed-Feature Subset Refits Were Not Kept (2026-04-18)

After the exact profile-gain improvement, I tried two follow-up refinements:

1. a small fixed-feature canonical coordinate-ascent initialization before the
   full joint refit
2. a heavier fixed-feature canonical support scorer inside component subset
   search

These were both logically clean, but neither became a stable mainline:

- the coordinate-ascent initialization had not produced a decisive,
  benchmark-level gain before the next cleaner direction appeared;
- the fixed-feature support scorer inside component search was too heavy
  relative to its likely benefit.

Both branches were removed rather than leaving extra machinery in the active
path.

### 268. Current Simpler Next Step: Sign-Agnostic Overlap Competition (2026-04-18)

The current question is whether the remaining extras are partly an artifact of
forcing overlap competition to happen only within the same sign.

The active retained profile-gain branch still leaves:

- same-sign overlap shadows (`A and B`, `C`, `H`, ...)
- some cross-sign overlap shadows (`E`, `F`, or an opposite-sign triple)

So the next clean structural change under test is:

- keep the exact profile-gain screening,
- keep the model class unchanged,
- but let overlapping rules compete in the same component **regardless of sign**.

This is attractive because it changes only the support-comparison topology, not
the model itself.

### 269. Exact Profile-Gain Screening Is The Current Best Canonical Mainline (2026-04-18)

The profile-gain branch completed on the main canonical trouble cases with the
following results:

- `ablation_excitation_only`: `1.0 / 0.75`
- `ablation_inhibition_only`: `1.0 / 1.0`
- `kernel_gaussian`: `1.0 / 0.6`
- `logical_context`: `1.0 / 0.7778`
- `logical_clean_plus`: `1.0 / 0.6`

Compared with the corrected canonical baseline:

- `ablation_excitation_only`: `0 / 0 -> 1.0 / 0.75`
- `ablation_inhibition_only`: `0.833 / 0.833 -> 1.0 / 1.0`
- `kernel_gaussian`: `0 / 0 -> 1.0 / 0.6`
- `logical_context`: `0.857 / 0.75 -> 1.0 / 0.7778`
- `logical_clean_plus`: `1.0 / 0.75 -> 1.0 / 0.6`

Interpretation:

- the catastrophic under-selection / empty-model failures are largely fixed;
- the remaining problem is now almost entirely **precision**;
- the extras are mostly overlap/shadow rules, e.g.:
  - `A and B`
  - `C`
  - `H`
  - `A and C and D`
  - `E`, `F`, `H`, `I`

So the canonical bottleneck has shifted from *screening collapse* to
*shadow-rule pruning / irredundant support definition*.

### 270. Sign-Agnostic Overlap Competition Did Not Obviously Improve The Completed Cases (2026-04-18)

I then removed the same-sign restriction inside overlap component comparison so
that overlapping rules could compete regardless of sign.

On the completed cases, this made no visible difference:

- `ablation_excitation_only`: unchanged at `1.0 / 0.75`
- `kernel_gaussian`: unchanged at `1.0 / 0.6`
- `logical_context`: unchanged at `1.0 / 0.7778`

This suggests that the remaining shadow rules are not primarily surviving
because overlap competition is sign-restricted.

### 271. Current Next Target: Exact Fixed-Feature Backward Pruning (2026-04-18)

The current remaining issue is precision after support recovery, so the next
clean direction is a final deterministic backward prune:

- keep the learned kernels fixed
- profile `mu` and rule coefficients exactly in the induced fixed-feature
  canonical model
- drop any rule whose removal improves the exact profiled support criterion

This preserves the model class and only sharpens the definition of
support-wise irreducibility.

### 272. Exact Fixed-Feature Backward Pruning Was Too Heavy To Keep (2026-04-18)

I implemented a fixed-feature backward selector that:

- froze the learned rule features,
- profiled `mu` and rule coefficients exactly in that induced canonical model,
- and greedily removed rules whose drop improved the profiled support score.

This direction was mathematically clean, but in practice it was too expensive:

- the profiling step itself was slow even on a single screened support;
- the full benchmark reruns did not return results in a reasonable time;
- the cost came from repeatedly solving a multi-rule coordinate profile problem
  before any final exact refit.

So I removed this branch rather than leaving a slow, unconfirmed path in the
active code.

### 273. Simpler Replacement: Exact Profile-Gain Backward Pruning (2026-04-18)

The next simplification is to use the same exact 1D canonical profile gain that
fixed the forward screening collapse, but now in reverse:

- take the current support recovered by the profile-gain forward path;
- for each active rule, compare `S \\ {r}` against `S` using the exact
  sign-fixed profiled gain of `r` conditional on the rest of the current
  support;
- drop a rule only if its conditional profiled gain is non-positive after the
  same BIC-style complexity penalty;
- only after the dropping phase finishes, do one final exact joint refit of the
  remaining support.

This is much simpler than the fixed-feature multi-rule profile branch:

- it keeps the model class unchanged;
- it uses the same exact profile criterion in both forward and backward
  selection;
- it avoids repeated full refits after every tentative drop;
- it is still deterministic and theorem-friendly as a finite-step local support
  irreducibility procedure.

The current experiment under test is whether this exact profile backward prune
can remove the remaining overlap/shadow extras in:

- `ablation_excitation_only`
- `kernel_gaussian`
- `logical_context`
- `logical_clean_plus`

### 274. Cross-Sign Competition Was Reverted; Current Active Branch Is Profile Forward + Profile Backward (2026-04-18)

Since sign-agnostic overlap competition did not improve the completed cases, I
reverted the mainline overlap selector back to:

- same-sign overlap competition only.

The current active canonical path is now:

1. exact profiled forward screening (`profilegain`)
2. same-sign overlap component selection
3. exact profile-based backward pruning

This keeps the support notion simple:

- forward: add a rule only when its exact conditional profiled gain is large
  enough;
- backward: keep a rule only when its exact conditional profiled gain remains
  positive given the rest of the selected support.

### 275. Benchmark-Level Difficulty Comparison Against Local Baseline Papers (2026-04-19)

I compared the current matched canonical synthetic suite against the local
benchmark papers in `benchmarks/`.

Current canonical suite:

- 11 benchmarks
- 5000 sequences each
- 7 to 20 event types
- 6 or 7 ground-truth rules
- rule orders 1 to 3
- exact signed canonical log-link generation/inference
- target mean counts roughly 3.27 to 69.83 per sequence
- average total sequence length roughly 161 to 253 events

The nearest local rule-discovery baselines are TELLER, CLNN, NSTPP, CLUSTER,
and OGEM/GCH.

Relative to TELLER / CLNN:

- those synthetic setups are mostly small-to-moderate label spaces (4 to 5
  labels in the paper examples),
- with fewer ground-truth rules (often around 3),
- and typically 600 to 2400 sequences for TELLER or 1000 streams for CLNN.

So our current suite is larger in support size and label space, and harder as
an exact rule-support recovery problem, even though the process architecture is
still simplified by using a single target event per benchmark.

Relative to NSTPP / CLUSTER:

- those papers use much larger candidate predicate spaces (for example 30 body
  predicates) and larger sample counts (5000 to 20000 or more),
- but their synthetic construction is easier to disentangle because each sample
  or event is usually explained by at most one rule / latent rule assignment.

So they are combinatorially larger, but our suite is more entangled
statistically because multiple signed rules can compete simultaneously inside
the same canonical log-intensity.

Relative to GCH / OGEM:

- those works focus on pairwise Hawkes-style graph / impact-function recovery,
- synthetic examples are much smaller (for example GCH uses 500 sequences and 5
  event types),
- and the structural target is graph or impact-function recovery, not exact
  higher-order conjunctive signed rule recovery.

So our suite is harder in support structure, even though it is still easier
than a fully coupled multivariate logical TPP because only one target type is
dynamically modulated.

Relative to NHP / RMTPP / THP:

- these papers are mainly predictive neural TPP baselines,
- their synthetic sections are not exact rule-support recovery benchmarks,
- and the main task is likelihood / next-time / next-type prediction.

Therefore raw dataset size comparisons to them are misleading. Our suite is
strictly harder if the metric is exact interpretable rule recovery, but easier
if the metric is only predictive likelihood because the benchmark family is
single-target and interpretable by construction.

Overall conclusion:

- the current canonical synthetic suite is not a toy benchmark;
- it is moderate in event density but hard in identifiability;
- it is harder than classic TELLER / CLNN-style toy rule-recovery setups;
- it is different from NSTPP / CLUSTER, which are larger in candidate set size
  but easier in per-sample rule disentanglement;
- and it is not directly comparable to NHP / RMTPP / THP except to note that
  those models solve a substantially easier objective than exact rule-support
  recovery.

### 276. If We Slightly Ease the Canonical Data, the Cleanest Knob Is Target Base Intensity (2026-04-19)

To make the canonical synthetic suite slightly easier without changing the
ground-truth support, the cleanest first knob is **target base intensity only**,
not rule weights.

Reason:

- in the current benchmark family, every benchmark has a single dynamically
  modulated target;
- that target is never used as a source for any other rule;
- therefore, in the canonical model
  `lambda_T(t) = b_T * exp(E_T(t) - I_T(t))`,
  multiplying `b_T` by a factor `c > 0` scales the target intensity uniformly
  without changing the source process or the normalized signed rule geometry.

So target-base scaling is a mathematically clean way to increase effective
sample size while preserving the structural truth.

By contrast, scaling rule weights changes the normalized competition among rules
and shadows, so it should be treated as a second-stage knob only if base-only
recalibration is still insufficient.

Practical complication:

- some benchmarks already sit close to `max_len = 256`, especially
  `logical_clean_plus` and, to a lesser extent, `ablation_excitation_only`;
- so fully matching the old easier target counts via base scaling alone would
  push mean sequence length beyond the current truncation ceiling.

Therefore the clean deterministic recommendation is:

1. keep rule weights unchanged;
2. rescale only target base intensity;
3. move only partway toward the old easier target counts (for example halfway),
   and cap the multiplier so the expected mean sequence length does not exceed
   the current `max_len` budget.

This gives a theorem-friendly “same truth, more samples” adjustment rather than
a qualitative change in the support-recovery problem itself.

### 277. Half-Step Target-Base Easing Did Increase Counts, But Did Not Fix Precision (2026-04-19)

I tested the cleanest easing variant first:

- keep the canonical model unchanged;
- keep all ground-truth rules and rule weights unchanged;
- rescale only the target base intensity by the previously derived half-step
  multiplier, then round to one decimal place.

For the two hardest current canonical cases, this yielded:

- `ablation_excitation_only`: target base `0.23 -> 0.3`
- `kernel_gaussian`: target base `0.21 -> 0.3`

The data did become easier in raw count terms:

- `ablation_excitation_only`: mean target count `69.83 -> 93.66`
- `kernel_gaussian`: mean target count `37.93 -> 59.03`

However, the current best confirmed canonical learner (`profilegain`) did **not**
improve in the way we wanted:

- `ablation_excitation_only`: `1.0 / 0.75 -> 1.0 / 0.5`
- `kernel_gaussian`: `1.0 / 0.6 -> 1.0 / 0.6`

Interpretation:

- increasing target base intensity does raise effective sample size;
- but the remaining failure is not primarily low-count recall collapse anymore;
- it is a precision problem driven by overlap / shadow supports.

So the sample-size knob is useful, but it does not by itself solve the current
identifiability issue. This is an important diagnostic: the present bottleneck is
support competition, not merely event scarcity.

### 278. After the Base-Only Test, Uniform Rule-Weight Scaling Looks Like the Cleaner Easing Knob (2026-04-19)

The half-step target-base experiment clarified something important:

- base-only easing increases target counts;
- but it does not materially improve the current precision bottleneck;
- in one hard case (`ablation_excitation_only`) it even worsened precision.

So if we want to make the canonical benchmark slightly easier while keeping the
truth support unchanged, the next cleaner knob is **uniform rule-weight scaling**
rather than further base scaling.

The clean version of this adjustment is:

- keep base intensities fixed;
- keep rule signs fixed;
- keep all within-benchmark relative weight ratios fixed;
- multiply every nonzero rule magnitude by a single benchmark-level scalar
  `alpha_b > 1`;
- optionally round the resulting weights to one decimal place for readability.

This is still structurally simple:

- same target;
- same support;
- same rule ordering by strength;
- same sign symmetry between excitation and inhibition if the same `alpha_b` is
  applied to both positive and negative rules.

Unlike base scaling, this directly increases the contrast between rule-on and
rule-off histories, so it targets identifiability rather than merely increasing
event counts.

### 279. Uniform Weight Scaling (`alpha = 1.1`, One-Decimal Rounding) Also Failed To Improve the Hard Canonical Cases (2026-04-19)

I reverted the temporary half-step target-base experiment entirely:

- removed the temporary configs;
- removed the temporary datasets;
- removed the temporary result directory.

Then I tested the next clean easing variant:

- keep the canonical generator and canonical learner unchanged;
- keep target base intensity unchanged;
- multiply all nonzero rule magnitudes in a benchmark by a common factor
  `alpha = 1.1`;
- round the resulting weights to one decimal place.

I ran this only on the two hardest current canonical cases:

- `ablation_excitation_only`
- `kernel_gaussian`

Results:

- `ablation_excitation_only`
  - mean target count: `69.83 -> 80.36`
  - rule recovery: `1.0 / 0.75 -> 1.0 / 0.545`
  - extra rules:
    - `A and B -> T : excitation`
    - `A and B and D -> T : inhibition`
    - `A and C and F -> T : excitation`
    - `A and D and H -> T : excitation`
    - `B and D -> T : excitation`
  - kernel mean `L1`: `0.342 -> 0.325`

- `kernel_gaussian`
  - mean target count: `37.93 -> 43.29`
  - rule recovery: `1.0 / 0.6 -> 1.0 / 0.545`
  - extra rules:
    - `A and C and D -> T : excitation`
    - `C -> T : excitation`
    - `D -> T : excitation`
    - `G -> T : excitation`
    - `H -> T : excitation`
  - kernel mean `L1`: `0.423 -> 0.441`

Interpretation:

- uniform weight scaling did increase target counts, but much less dramatically
  than the base-only test;
- more importantly, it did **not** improve the rule-selection bottleneck;
- in both hard cases the surviving failure mode remained shadow / overlap
  over-selection, and precision was slightly worse than the current canonical
  `profilegain` baseline.

So this is another useful negative result:

- the remaining difficulty is not cured by mildly increasing global signal
  amplitude;
- the present bottleneck is still the *support competition geometry*, not merely
  low signal strength.

### 280. The Remaining Precision Errors Have a Clear Local-Closure Structure (2026-04-19)

After reverting the temporary data-easing experiments and re-reading the current
best valid canonical results (`canonical_profilegain_eval_20260418`), the
remaining precision errors are not arbitrary. They fall into a few recurring
local patterns:

1. **same-sign supersets of true singleton rules**
   - `ablation_excitation_only`: `A and B`
   - `logical_context`: `A and B`, `A and E`

2. **same-sign subsets of true higher-order rules**
   - `kernel_gaussian`: `C`
   - `logical_clean_plus`: `H`, `I`

3. **mixed overlap composites that borrow support from multiple true rules**
   - `ablation_excitation_only`: `B and C and G`
   - `kernel_gaussian`: `A and C and D`, `A and D and F`, `D and H`

This matters because it narrows the problem:

- the current best canonical forward step (`exact 1D profiled gain`) already
  fixes the large recall collapse;
- the remaining failure is almost entirely a **local support competition**
  problem inside small overlap / closure neighborhoods.

This also explains why simple easing failed:

- increasing target base intensity did not help;
- mildly increasing all rule weights did not help;
- both changed sample size / signal amplitude but not the underlying local
  closure ambiguity.

So the next theorem-friendly direction should not be “more data” or “stronger
global shrinkage.” It should be a **local exact competition rule** for selected
supports, applied after the successful exact-profile forward path.

Crucially, standard strong-hierarchy constraints are not appropriate here:

- our truth frequently contains higher-order rules without the corresponding
  lower-order rules;
- so importing a hierarchical interaction constraint would solve the wrong
  problem by changing the benchmark truth.

The clean target is instead:

- keep the current forward `profilegain` screening;
- then enforce a *closure-irredundant* support notion under the same exact
  profile criterion.

### 281. Exact Local Component Refit Did Not Change the Hard Canonical Supports (2026-04-19)

I implemented and tested a stronger local competition rule:

- keep the current canonical `profilegain` forward path;
- keep same-sign overlap components;
- but replace the cheap fixed-parameter component search with **exact subset
  refits inside each selected component**.

This was the cleanest local finite-family test of the “closure-irredundant
support” idea.

Result on the four current hard canonical cases:

- `ablation_excitation_only`: unchanged at `1.0 / 0.75`
- `kernel_gaussian`: unchanged at `1.0 / 0.6`
- `logical_context`: unchanged at `1.0 / 0.7778`
- `logical_clean_plus`: unchanged at `1.0 / 0.6`

The selected extra rules were exactly the same as in the current `profilegain`
baseline. Only BIC shifted slightly in a couple of cases, but the support did
not improve.

Interpretation:

- the current remaining extras are not artifacts of the old cheap
  `component_subset_search`;
- even under exact local finite-family competition inside same-sign overlap
  components, those shadow rules still survive;
- therefore the present precision bottleneck lies **earlier** than the final
  local subset search.

This is another strong negative result:

- precision is not failing because the final component search is too cheap;
- it is failing because the selected support that arrives at that stage already
  makes the shadow rules look genuinely useful under the current objective.

Following the “remove ineffective code” rule, I discarded this branch from the
active learner after the experiment.

### 282. Stagewise Diagnosis: The Remaining Precision Loss Is Already Present After Forward Selection (2026-04-19)

I decomposed the current canonical `profilegain` path into:

- forward
- refine
- post-prune
- component search

for the hard canonical cases.

The key pattern is now very clear:

- `kernel_gaussian`: the forward stage already lands at the final
  `1.0 / 0.6`; refine, post-prune, and component search do not change the
  support.
- `logical_clean_plus`: the forward stage already lands at the final
  `1.0 / 0.6`; the later stages again do not change the support.
- `ablation_excitation_only`: post-prune removes one obviously bad extra rule
  (`B and E and F`), improving `1.0 / 0.6667 -> 1.0 / 0.75`, but the main
  extras (`A and B`, `B and C and G`) are already forward-stage artifacts.

So the present precision bottleneck is not primarily:

- family attribution refinement,
- post-prune weakness, or
- final component search weakness.

It is overwhelmingly an **earlier support-formation problem**.

### 283. Forward-Path Diagnosis: The First Greedy Entry Is the Main Failure Mode (2026-04-19)

I then traced the actual forward acceptance path under canonical
`profilegain`.

Two especially important cases:

1. `ablation_excitation_only`

   - step 0 accepts `A and B -> T : excitation`
   - only later does it accept the true singleton rules `A` and `B`

   So the first shadow seed enters *before* the simpler true decomposition.

2. `kernel_gaussian`

   - step 0 accepts `A and C and D -> T : excitation`
   - only later does it accept the simpler true pieces such as `A` and
     `C and D`

Again, the first greedy seed is a merged / shadow explanation that wins under
the current exact 1D profiled gain.

This means the remaining precision problem is not “insufficient cleanup after a
mostly correct forward path.” The problem is that the **top-1 greedy entry rule
is often the wrong local chart representative**.

### 284. Exact Local-Closure Family Search Is Directionally Right but Computationally Too Heavy (2026-04-19)

I implemented a deterministic local-closure forward branch:

- take the top seed rule,
- form a local closure universe from its support plus active lower-order /
  minimal super-order context,
- exact-search the whole finite local family by refit.

This was theorem-friendly in spirit:

- deterministic finite family,
- exact refit criterion,
- no hand-tuned shortlist threshold.

However, in practice it was too expensive:

- even two hard cases (`ablation_excitation_only`, `kernel_gaussian`) with the
  rest of the pipeline fixed became too slow to use as the next mainline;
- even a one-step diagnostic with the full local family was already much
  heavier than acceptable for iterative research.

So the lesson is:

- the idea of replacing a shadow seed with a better local representation is
  probably correct,
- but the local family must be **much smaller** than the full closure family.

### 285. Next Reduction: Seed-Support Partition Competition (2026-04-19)

The most natural reduction is to focus only on the support of the top seed
rule itself.

If the seed support is:

- `{A, B}`, compare exact finite alternatives:
  - `{AB}`
  - `{A, B}`

If the seed support is:

- `{A, C, D}`, compare:
  - `{ACD}`
  - `{A, CD}`
  - `{C, AD}`
  - `{D, AC}`
  - `{A, C, D}`

This is appealing because it directly targets the observed pathology:

- a merged shadow seed enters before a simpler decomposition on the same local
  variable set.

And it stays theorem-friendly:

- the comparison family is deterministic,
- finite,
- and exact under the same profiled criterion.

It is also dramatically smaller than the earlier closure family:

- Bell number 2 for support size 2,
- Bell number 5 for support size 3.

So the current working hypothesis is:

- the right next forward-step repair is not full closure search,
- but **exact partition competition on the top seed support**.

### 286. Full Local-Chart Code Removed; Only the Smaller Seed-Partition Direction Remains (2026-04-19)

After the negative computational result above, I removed the inactive
full local-chart implementation from the learner code.

What remains in code is only the smaller deterministic helper:

- `support_partitions(...)`
- `exact_seed_partition_step(...)`

This matches the current research conclusion:

- full local closure search is too heavy for the next mainline,
- but exact partition competition on the top seed support is still a clean,
  theorem-friendly candidate for the next forward-step repair.

### 287. Implemented the Smaller Forward Variant: Seed-Partition Forward Selection (2026-04-19)

I implemented a new experimental forward variant:

- `run_active_set_seed_partition(...)`

It differs from the current `profilegain` forward step in exactly one place:

- after choosing the top seed candidate by the usual exact profiled gain,
- it does **not** add that single rule directly;
- instead it calls `exact_seed_partition_step(...)` and exact-compares the
  finite partition family of the seed support.

So this is a much smaller alternative to the abandoned local-chart branch:

- still deterministic,
- still finite-family,
- still exact under the same canonical refit objective,
- but now with only a Bell-number-size local family.

This is the current best theorem-friendly repair candidate for the forward
selection bottleneck.

### 288. First Seed-Partition Forward Result: `ablation_excitation_only` Reaches 100/100 (2026-04-19)

I ran the new `run_active_set_seed_partition(...)` forward variant on the
current canonical `ablation_excitation_only` benchmark.

Result:

- baseline canonical `profilegain`: `1.0 / 0.75`
- seed-partition forward: `1.0 / 1.0`

So the two persistent extras

- `A and B -> T : excitation`
- `B and C and G -> T : excitation`

were completely removed.

This is the strongest positive signal so far for the current bottleneck story:

- the remaining precision loss really does appear to come from the first greedy
  seed being a merged shadow,
- and exact competition over the seed-support partition family can correct that
  without changing the model class or introducing heuristic thresholds.

Important caveat:

- this is only one benchmark so far;
- the next critical check is whether the same mechanism also helps on
  `kernel_gaussian`, where the first shadow seed is broader and the geometry is
  less singleton-friendly.

### 289. Consistency and Speed Audit: Generator/Learner Still Match Canonical Log-Link (2026-04-19)

I re-checked the current final synthetic configs and the active learner path.

For all 11 final configs:

- `intensity_model = canonical_loglink`
- `synthetic_generation_model = canonical_loglink`
- `activation_mode = product_bounded`

So the intended semantics are consistently:

- data generation:
  - `lambda_T(t) = b_T * exp(E_T(t) - I_T(t))`
- learning:
  - `mu * exp(E - I)` under the same bounded product activation features.

I also verified again that the current synthetic fast path is exact for the
benchmark family:

- all rules point to a single target,
- that target never appears as a source,
- all non-target types are baseline-only.

So the generator’s source-first / target-conditional decomposition is not a
heuristic; it is an exact factorization of the current synthetic benchmark
structure.

### 290. Exact Speed Improvements Applied (2026-04-19)

I applied two exact speed/consistency improvements.

1. Generator fast-path optimization in `data/synthetic.py`

Previously, the canonical fast generator recomputed source-event counts at each
candidate time by scanning all source arrays with `searchsorted`, and then
recomputed the rule activations again for the auxiliary target labels.

Now:

- source-event counts are advanced monotonically in-place with
  `_advance_past_event_counts_in_place(...)`,
- accepted-event labels reuse the already computed rule activations through
  `precomputed_activations`.

This does not change the generated law at all; it only removes redundant work.

2. Runner cleanup in `run_paper_benchmarks.py`

The benchmark runner still had an extra canonical
`canonical_profile_backward_prune(...)` stage even though that branch was not
the current best confirmed mainline.

I removed that extra stage from the runner, so the default evaluation path is
again aligned with the actual confirmed canonical baseline:

- forward selection
- family attribution refine
- post-prune
- same-sign component search

This improves both:

- logical consistency of what we call the “baseline,” and
- runtime, by removing an extra exact canonical pass that was not part of the
  validated best path.

### 291. Small Exact Learning-Side Speed Improvement Applied (2026-04-19)

For the new `seed_partition` branch, I added memoization to the deterministic
partition family:

- `support_partitions(...)` now uses an unbounded LRU cache.

This is exact and theorem-neutral:

- the partition family is deterministic,
- caching only avoids recomputing the same finite family for repeated supports,
- it does not alter the comparison criterion or selected support.

### 292. Exact Learner-Side Validation Export Gating (2026-04-19)

I tightened the canonical learner loop without changing the objective or the
checkpoint schedule.

Inside `optimize_active_set_torch(...)`, validation checkpoints previously
called `current_model(emit_numpy=True)` every time, even when the current
checkpoint did not improve the best validation BIC. That forced unnecessary:

- CPU transfers,
- NumPy materialization of `coef_out`,
- NumPy materialization of `heights_out`,
- NumPy materialization of `arrays_out`.

Now the checkpoint logic is:

1. call `current_model(emit_numpy=False)` to compute validation BIC purely on
   tensors,
2. only if the BIC improves, call `current_model(emit_numpy=True)` once to
   snapshot the best state.

This is exact:

- same optimizer,
- same losses,
- same checkpoint times,
- same selected `best_state` criterion,
- strictly less redundant export work.

### 293. Exact Learner-Side Rule Feature Assembly Reuse (2026-04-19)

I also reduced repeated work inside `optimize_active_set_torch(...)` by
changing how rule features are assembled at each optimizer step.

Before:

- each rule repeatedly looked up its source basis tensors inside the inner
  loop,
- source factor vectors were accumulated into Python lists,
- then `torch.stack(...).prod(...)` was used to form the bounded conjunction.

Now:

- unique source basis tensors are prefetched once per optimization call,
- rule/source structure is precompiled into deterministic `rule_specs`,
- rule features are assembled by incremental multiplication of source factors.

This is still exactly the same computation:

- same bounded source factor `1 - exp(-z)`,
- same product conjunction,
- same canonical likelihood,
- same optimizer path,
- just fewer temporary tensors and fewer repeated lookups.

I validated this patch with a small smoke test on canonical
`ablation_inhibition_only`:

- one GT rule,
- CPU,
- `steps = 5`,
- `optimize_active_set_torch(...)` completed normally in about `3.57s`,
- returned a finite BIC, finite `mu`, correct sign-specific parameter map, and
  nonempty feature arrays.

### 294. Runner-Level Forward Variant Switch Added (2026-04-19)

I added a very small runner-facing switch in
`workspace/train/paper_benchmark_active/run_paper_benchmarks.py`:

- `--forward_variant baseline`
- `--forward_variant seed_partition`

Important points:

- the default remains `baseline`,
- this does not change the default benchmark behavior,
- it only makes the currently best forward-repair candidate (`seed_partition`)
  reproducible at the runner level without maintaining a separate ad hoc script.

This keeps the research line cleaner:

- baseline remains stable,
- experimental forward variants can be evaluated under the exact same
  downstream pipeline,
- comparisons are easier to reproduce and audit.

### 295. Full-11 Seed-Partition Attempt Was Launched Then Intentionally Stopped (2026-04-19)

I launched a full 11-benchmark canonical evaluation using the new
`seed_partition` forward variant, distributed across 5 workers over 4 GPUs,
with dataset reuse enabled.

However, this run was stopped intentionally before any benchmark completed.

So at this point:

- there is **no valid full-11 result** for `seed_partition`,
- only the earlier single-benchmark result on
  `ablation_excitation_only` is confirmed,
- the current open question remains whether the forward repair generalizes
  beyond that benchmark.

### 296. Artifact Cleanup Policy Applied (2026-04-19)

I cleaned the paper-suite artifact tree so that only files still needed for the
current canonical research line remain.

Kept:

- current final canonical configs in `configs/final_logical_tpp/`
- current canonical baseline full-suite result:
  - `final_logical_tpp_seed111_eval_20260418/`
- current canonical dataset frequency summary:
  - `final_logical_tpp_seed111_frequency_summary_20260418.json`
- current best confirmed general canonical branch:
  - `canonical_profilegain_eval_20260418/`
- current strongest positive seed-partition result:
  - `canonical_seedpartition_eval_20260419/`

Removed:

- legacy `configs/multiplicative/`
- old multiplicative / mixed-semantics result artifacts
- negative and superseded canonical experiment artifacts whose conclusions are
  already summarized in these notes
- aborted full-batch seed-partition artifacts
- bulky log directories not needed for current research continuity

This keeps the filesystem aligned with the actual active research story and
reduces the chance that future work accidentally uses invalid or superseded
artifacts as evidence.

This matters because the next Codex should not accidentally treat the aborted
full-batch directory as evidence of completed performance.

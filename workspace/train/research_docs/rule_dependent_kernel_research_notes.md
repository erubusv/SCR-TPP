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

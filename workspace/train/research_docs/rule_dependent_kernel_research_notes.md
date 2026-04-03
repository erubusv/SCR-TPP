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

# Paper Synthetic Benchmark Plan

This document defines a synthetic benchmark suite that is aligned with the
current learner and supports a clean paper narrative:

1. `Excitation/Inhibition logical rule recovery`
2. `Temporal kernel robustness`
3. `Robustness to number of predicates`

The key design choice is to separate:
- logical ambiguity,
- temporal-kernel family shift,
- nuisance predicate count / filler noise.

That avoids the previous issue where the Gaussian benchmark was also an extreme
overlap stress test rather than a clean kernel-family comparison.

## 1. Excitation/Inhibition Logical Rule Recovery

Use a triangular-kernel logical recovery suite as the main exact-recovery
benchmark.

Keep the current exact `shared/context` full suites, and add a new paper-only
`clean+` suite that also contains a triplet inhibition rule.

Configs:
- [paper_logical_clean_plus.yaml](/workspace/data/paper_suite/paper_logical_clean_plus.yaml)
- [paper_logical_shared.yaml](/workspace/data/paper_suite/paper_logical_shared.yaml)
- [paper_logical_context.yaml](/workspace/data/paper_suite/paper_logical_context.yaml)

Ground-truth rules:
- `clean+`
  - `A -> T : excitation`
  - `B -> T : inhibition`
  - `CD -> T : excitation`
  - `EF -> T : inhibition`
  - `GHI -> T : excitation`
  - `JKL -> T : inhibition`
- `shared`
  - `A -> T : inhibition`
  - `AB -> T : excitation`
  - `BEF -> T : inhibition`
  - `C -> T : excitation`
  - `CD -> T : inhibition`
  - `D -> T : excitation`
  - `EF -> T : excitation`
- `context`
  - `A -> T : excitation`
  - `B -> T : inhibition`
  - `ACD -> T : inhibition`
  - `BCD -> T : excitation`
  - `E -> T : inhibition`
  - `EF -> T : excitation`
  - `EFG -> T : inhibition`

Role in paper:
- This is the main “can the model recover mixed-sign logical rules across
  singleton/pair/triplet orders?” section.
- Keep it triangular and exact-recovery oriented.

## 2. Temporal Kernel Robustness

Use one fixed **moderate-difficulty** logical rule set and vary only kernel
family:
- triangular
- exponential
- gaussian

We intentionally use a *moderate-overlap* Gaussian and Exponential regime
rather than the earlier extreme one.

### Moderate rule set

Target is the last type.

Rules:
- `A -> T : excitation`
- `B -> T : inhibition`
- `CD -> T : excitation`
- `EF -> T : inhibition`
- `CGH -> T : excitation`
- `DFG -> T : inhibition`

Properties:
- mixed order: singleton / pair / triplet
- mixed sign: excitation and inhibition
- partial source reuse:
  - `C` shared between `CD` and `CGH`
  - `D` shared between `CD` and `DFG`
  - `F` shared between `EF` and `DFG`
  - `G` shared between `CGH` and `DFG`
- but no exact parent-child nesting

This is harder than `clean` but avoids the most pathological overlap motif from
the old Gaussian context case.

### Kernel calibration

Use order-dependent peaks:
- singleton: peak `0.8`
- pair: peak `1.0`
- triplet: peak `1.2`

Use:
- triangular widths: `2.4 / 3.2 / 4.0`
- gaussian sigmas: `1.2 / 1.6 / 2.0`
- exponential taus: `1.2 / 1.6 / 2.0`

This makes Gaussian/Exponential noticeably broader than triangular, but not
the earlier extreme regime where Gaussian support was about `3.3x` wider.

Evaluation:
- rule recovery: recall / precision / sign accuracy
- kernel recovery:
  - `L1` distance between true and learned normalized kernel
  - peak-location error
  - effective-support error

Configs:
- [paper_kernel_robustness_triangular.yaml](/workspace/data/paper_suite/paper_kernel_robustness_triangular.yaml)
- [paper_kernel_robustness_exponential.yaml](/workspace/data/paper_suite/paper_kernel_robustness_exponential.yaml)
- [paper_kernel_robustness_gaussian.yaml](/workspace/data/paper_suite/paper_kernel_robustness_gaussian.yaml)

## 3. Robustness To Number Of Predicates

Keep the same moderate true rule set fixed and increase only the total number
of predicate types:
- `10`
- `20`
- `30`

Important design choice:
- keep the **total non-target base-intensity budget fixed**
- scale **all non-target predicates together** as predicate count grows

This isolates the effect of predicate count from nuisance-event explosion.

Rule set:
- same as temporal-kernel robustness set
- kernel family fixed to triangular for this experiment

Predicate-count design:
- true source ids remain `0..7`
- target is the last id
- all remaining non-rule predicates are filler predicates
- start from one common template:
  - true-source bases: `0.18, 0.18, 0.17, 0.17, 0.17, 0.16, 0.16, 0.16`
  - filler template base per predicate: `0.08`
- then scale **all** non-target bases by one global factor so that the total
  non-target base-intensity sum matches the `10-predicate` case

Meaning:
- when predicate count increases, true-source predicates are *also* scaled down
- so the model is compared under the same total event-rate budget, distributed
  across more predicates

This is fairer than keeping true-source predicates unchanged while only
redistributing filler mass.

Concretely:
- `10 predicates`
  - total non-target budget: `1.43`
  - scale: `1.0`
- `20 predicates`
  - unscaled budget would be `2.23`
  - apply global scale `1.43 / 2.23 ≈ 0.6413`
- `30 predicates`
  - unscaled budget would be `3.03`
  - apply global scale `1.43 / 3.03 ≈ 0.4719`

So this experiment measures:
- same overall event budget
- more predicates competing for that budget
- more difficult source/rule recovery because *all* sources become effectively
  weaker as the predicate vocabulary expands

Configs:
- [paper_num_predicates_10.yaml](/workspace/data/paper_suite/paper_num_predicates_10.yaml)
- [paper_num_predicates_20.yaml](/workspace/data/paper_suite/paper_num_predicates_20.yaml)
- [paper_num_predicates_30.yaml](/workspace/data/paper_suite/paper_num_predicates_30.yaml)

Role in paper:
- this is the clean “dimension scaling under fixed event budget” experiment
- a separate robustness section should vary noise/filler intensity explicitly
  while keeping predicate count fixed

## Recommended Additional Experiments For A Top-Tier Submission

### A. Stress-vs-standard regime split

Have two benchmark tiers:
- `standard`
  - the suite above
- `stress`
  - old high-overlap Gaussian
  - high filler-mass many-predicate setting

This lets the paper claim:
- strong performance in a calibrated standard regime
- clear analysis of failure modes in stress regimes

### B. Ablation on rule-dependent kernels

Show:
- global kernel only
- post-hoc rule-specific refinement
- full rule-dependent kernels from the start

This directly supports the temporal-kernel contribution.

### C. Formal Ablation On Inhibition

Make inhibition a first-class benchmark rather than a paragraph-level ablation.

Use one fixed logical topology and create three formal settings:
- excitation-only
- inhibition-only
- mixed-sign

Recommended configs:
- [paper_ablation_excitation_only.yaml](/workspace/data/paper_suite/paper_ablation_excitation_only.yaml)
- [paper_ablation_inhibition_only.yaml](/workspace/data/paper_suite/paper_ablation_inhibition_only.yaml)
- [paper_ablation_mixed_sign.yaml](/workspace/data/paper_suite/paper_ablation_mixed_sign.yaml)

Use the same condition sets in all three:
- `A`
- `B`
- `CD`
- `EF`
- `CGH`
- `DFG`

Only the sign pattern changes. This isolates the value of inhibition handling
from the value of rule-shape search itself.

### D. Runtime / scaling

Report:
- wall time vs number of candidate rules
- wall time vs number of predicates
- GPU vs CPU path if relevant

### E. Multi-seed stability

For each standard benchmark:
- run 3-5 random seeds
- report mean/std of recall, precision, and kernel-recovery metrics

This is important for credibility.

### F. Separate Noise-Robustness Experiment

Do **not** use the predicate-count benchmark itself to also claim robustness to
changing noise mass.

Instead, add a separate experiment:
- fix predicate count, e.g. `30`
- fix the same true rule set
- vary a global non-target scale or filler-template strength

Example:
- `noise_scale = 1.0`
- `noise_scale = 0.75`
- `noise_scale = 0.5`

This keeps the interpretation clean:
- predicate-count robustness = vocabulary size effect
- noise robustness = nuisance-rate effect

## Summary

The clean paper story should be:

1. exact logical-rule recovery on the triangular logical suite,
2. kernel-family robustness on a calibrated moderate-overlap rule set,
3. predicate-count robustness under a fixed global non-target event budget,
4. formal sign ablation showing what inhibition modeling adds,
5. separate noise-robustness experiments and stress appendices showing where
   overlap and nuisance mass break identifiability/search.

That is much cleaner and more defensible than using the old extreme Gaussian
regime as the main kernel-family comparison.

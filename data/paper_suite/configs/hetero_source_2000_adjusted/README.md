# Hetero-Source 2000 Synthetic Suite

This suite is the 2000-sample rule-recovery benchmark with heterogeneous
source kernels inside each multi-source rule.

## Kernel Design

The logical rule support, signs, targets, canonical log-link intensity, and
strict product-max-witness activation follow the final logical TPP synthetic
template.
Most rule weights are unchanged.  Several inhibition weights were raised
after the shadow-margin audit showed that the generated data made the true
high-order rule too close to its low-order shadow:

```text
ablation_mixed_sign: E and F -> I inhibition, W_neg 3.2 -> 3.8
ablation_mixed_sign: D and F and G -> I inhibition, W_neg 3.6 -> 6.2
kernel_exponential: D and F and G -> I inhibition, W_neg 3.6 -> 4.2
logical_clean_plus: J and K and L -> M inhibition, W_neg 4.0 -> 7.2
num_predicates_20: E and F -> T inhibition, W_neg 3.2 -> 3.8
num_predicates_20: D and F and G -> T inhibition, W_neg 3.6 -> 12.0
```

Only the per-source kernel parameters inside each rule condition are changed.
The refreshed configs are produced by:

```text
python data/paper_suite/scripts/refresh_identifiable_hetero_source_2000_adjusted.py
```

Sources are assigned one-decimal kernel parameters deterministically from rule
rank, source id, rule id, and same-source occurrence index.  For each
multi-source rule, sources are sorted by source id and assigned increasing
peak-lag bands.  The bands are deliberately closer and wider than the first
identifiability refresh, because the first refresh made several strict
product-AND high-order rules too rare to observe reliably:

```text
triangular peak bands   = rank0 around 0.5-2.4, rank1 around 2.2-3.5, rank2 around 3.8-5.1
gaussian peak bands     = rank0 around 1.2-1.8, rank1 around 2.2-3.0, rank2 around 3.0-3.2
exponential peak bands  = rank0 around 0.5-1.3, rank1 around 1.1-1.8, rank2 around 1.6-2.0
```

A larger peak means that the source tends to occur farther before the target.
Thus the sources inside a rule carry an identifiable temporal relation by
comparing source-specific kernels.  The rank bands are ordered but not fully
disjoint: strict product-AND rules need enough temporal overlap to be observed
in finite samples.  The values still vary by rule and source, so the benchmark
does not assume one shared temporal template.

The refreshed suite additionally enforces an identifiability audit:

```text
same-source normalized kernel correlation <= 0.90
no duplicated (peak, width) pair inside a benchmark config
```

This condition targets the previous failure mode where reused sources had
nearly identical rule-specific kernels, making true high-order rules and proxy
rules statistically hard to separate.

If a deterministic rank/lane assignment accidentally violates the same-source
correlation bound, the refresh script moves only the later rule-source kernel
by one decimal step until the audit passes.  This preserves the benchmark
semantics while avoiding hand-edited benchmark-specific exceptions.

Rule-source diagrams are available in:

```text
data/paper_suite/configs/hetero_source_2000_adjusted/rule_source_diagrams.md
data/paper_suite/configs/hetero_source_2000_adjusted/diagrams/*.svg
```

## Base Intensities

Base intensities are simple and slightly lower than the original
suite.  Standard source rates use clean values around `0.14`, `0.15`, and
`0.16`.  `kernel_exponential` uses `0.12` and `0.20` source rates to keep its
long-tailed source process stable.  `num_predicates_20` uses only `0.10` for
active sources, `0.05` for distractor sources, and `0.01` for the target
background.  The target background remains `0.01` rather than one decimal
place because rounding it to `0.0` would remove almost all target events from
the canonical log-link generator.  The pure inhibition benchmark keeps a
higher but lowered target background of `0.20`.

## Generated Data

The generated datasets are stored under:

```text
data/paper_suite/datasets/hetero_source_2000_adjusted/seed_111/
data/paper_suite/datasets/hetero_source_2000_adjusted/seed_222/
data/paper_suite/datasets/hetero_source_2000_adjusted/seed_333/
```

Each synthetic rule-discovery dataset uses all 2000 sequences for support
recovery.  The pickle keeps `train`/`val`/`test` keys for loader
compatibility, but the refreshed files store all sequences in `train` and
leave `val` and `test` empty.  The active benchmark runner also merges
`train + val + test` when it reads older split files, so synthetic rule
discovery always uses the full 2000 sequences.  Predictive real-world
experiments should still use a proper train/validation/test split.

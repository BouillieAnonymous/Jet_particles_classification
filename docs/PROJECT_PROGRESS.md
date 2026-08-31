# Jet Quark/Gluon Classification — Project Progress

> **Canonical progress tracker.** Update this document whenever a material
> experiment finishes, a scientific conclusion changes, or the next decision
> gate changes. Keep claims proportional to the evidence.

**Last updated:** 2026-08-31
**Current phase:** Stage 5 — information-source analysis
**Current research question:**

> Under what conditions, if any, does dynamic graph rewiring add value in
> quark–gluon classification, and where does the discriminating information
> actually come from?

## Current status at a glance

- The data/experiment pipeline is reproducible and event-safe.
- Fixed and Dynamic EdgeConv now form a controlled comparison.
- The full-feature paired 5-seed benchmark is complete.
- No consistent dynamic-graph advantage was observed in the current
  full-feature, `k=8`, approximately 14k-jet setup.
- The project is now prioritising a traditional physics baseline and controlled
  feature ablations over complex rewiring interpretation.

## How the project evolved

### 1. Working GNN project

The project began as a standard quark/gluon jet-classification exercise. Each
jet was represented as a graph of particle-flow constituents, using features
such as `pT`, `dEta`, and `dPhi`. The initial objective was to establish basic
engineering feasibility and build the necessary physical understanding.

### 2. Experiment engineering

To move beyond manually edited notebooks, the project added:

- unified ROOT data loading;
- composite-event-safe train/validation/test splitting;
- training-only feature normalisation;
- a command-line experiment runner;
- metrics logging and per-run artifacts;
- best-validation checkpoints and early stopping;
- reproducible training seeds;
- unit tests.

The objective of this phase was to make experimental results repeatable rather
than dependent on one successful notebook execution.

### 3. Controlled-comparison correction

An audit found that the original Fixed/Dynamic comparison had confounders: the
initial graph semantics, feature preprocessing, and training protocols were not
fully aligned.

The controlled definition is now:

```text
E_geo = kNN(pos)
pos   = (dEta, dPhi)       # fixed geometric coordinates
x     = selected features # message-passing inputs
```

Both models use `E_geo` for the first EdgeConv layer. Their only architectural
difference is the graph used by the second layer:

```text
Fixed:   EdgeConv2 uses E_geo
Dynamic: EdgeConv2 uses E_latent = kNN(h1)
```

Hidden dimensions, EdgeConv MLPs, aggregation, pooling, classifier, dropout,
optimiser, loss, class weighting, training schedule, and parameter count are
matched.

### 4. Testing whether dynamic rewiring adds value

A paired 5-seed benchmark was run with:

```text
features     = full
k            = 8
split_seed   = 42
max_epochs   = 50
patience     = 8
selected jets = 13,912
train/validation/test = 9,732 / 2,102 / 2,078
```

Results:

| Model | Test ROC-AUC (mean ± std) | Balanced accuracy (mean ± std) |
|---|---:|---:|
| Fixed EdgeConv | 0.83042 ± 0.00313 | 0.75679 ± 0.00415 |
| Dynamic EdgeConv | 0.83187 ± 0.00073 | 0.75574 ± 0.00227 |

For each seed, define:

```text
Delta AUC = Dynamic test AUC - Fixed test AUC
```

The paired result was:

```text
Delta AUC = +0.00146 ± 0.00266
positive seeds = 3/5
negative seeds = 2/5
```

All ten runs converged under the fixed early-stopping protocol; no run required
retraining. Full artifacts are recorded in [`../results/metrics.csv`](../results/metrics.csv)
and `../results/runs/`.

The correct conclusion is:

> Across five paired training seeds, no consistent dynamic-graph advantage was
> observed.

This does **not** prove that Fixed and Dynamic are equivalent. The five seeds
measure training stochasticity on the same split and test set; they are not five
independent datasets.

### 5. Information-source analysis — current phase

The original candidate story—“Dynamic rewiring improves classification; now
explain why”—is not supported and has been dropped.

The project now asks:

> What information makes quark and gluon jets distinguishable, and how much is
> contributed by the input representation versus graph architecture?

The controlled feature sets are:

```text
geometry:
    dEta, dPhi

geometry_pt:
    dEta, dPhi, log_pT

full:
    dEta, dPhi, log_pT, log_pT_over_jetPt
```

For every feature set, `pos=(dEta,dPhi)` and the initial graph remain unchanged.

The traditional physics baseline will initially consider:

- constituent multiplicity (`QG_mult`);
- jet girth (`jetGirth`);
- `pT^D` (`QG_ptD`);
- second principal axis (`QG_axis2`).

## Next planned work

1. Implement and evaluate the event-safe traditional physics baseline.
2. Run paired Fixed/Dynamic 5-seed benchmarks for `geometry`.
3. Run paired Fixed/Dynamic 5-seed benchmarks for `geometry_pt`.
4. Compare all regimes with the existing `full` results.
5. Decide whether any feature regime justifies targeted rewiring analysis.
6. Consider dataset-size scaling only after feature ablation, because adding
   more ROOT files has higher engineering cost.

No hyperparameter tuning or architecture expansion is planned during these
controlled comparisons.

## Decision gates

- **Fixed approximately equals Dynamic for every feature set:** stop treating
  dynamic rewiring as the main paper direction.
- **Dynamic is consistently better only with weak inputs:** investigate whether
  learned topology compensates for missing explicit physical information.
- **Only small, inconsistent differences remain:** retain Dynamic as an
  engineering comparison, not a scientific headline.
- **Physics baseline approaches GNN performance:** reassess the marginal value
  of constituent-level graph representations.
- **Physics baseline is substantially weaker:** focus the project narrative on
  constituent-level information rather than dynamic rewiring.

## Maintenance rule

After every material milestone, update at least:

1. `Last updated` and `Current phase`;
2. the relevant completed-stage evidence and result table;
3. the strongest scientifically justified conclusion;
4. `Next planned work` and the active decision gate;
5. the change log below.

Do not overwrite negative results or earlier decisions. Record why the route
changed so the research history remains auditable.

## Change log

| Date | Milestone | Decision or consequence |
|---|---|---|
| 2026-08-31 | Created canonical progress tracker after the full-feature paired 5-seed benchmark | Dropped the assumed “Dynamic is better” narrative; moved to physics baseline and feature-ablation analysis |

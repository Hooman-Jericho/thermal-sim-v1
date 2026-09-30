# Day 19 — Feature Engineering on the Thermal Dataset

Answers the 8 problems found in the Sep-27 submission (`thermal_ml_project_v2.zip`,
graded 4/10). Every number below came out of an actual run
(`PYTHONPATH=. python -m experiments.baseline_dynamics_learning.run_feature_study`,
production config, ~100 s), not from reading the code.

## New files

```
experiments/baseline_dynamics_learning/
    features.py            # feature registry, leakage guard, episode-aware lags, redundancy audit
    estimators.py           # target-mode factory (absolute/delta/physics_residual), target scaling
    config.py               # base config + feature_study.yaml overrides, deep-merge, up-front validation
    feature_study.yaml      # overrides: only what the study changes vs. baseline config.yaml
    run_feature_study.py    # ties it together: ablations, safety agreement, manifest, W&B
tests/
    test_features.py         (23 tests)
    test_estimators.py       (20 tests)
    test_config.py           (13 tests)
    test_run_feature_study.py (10 tests, end-to-end on a fast config)
```

## Files that already exist in the repo and are MODIFIED, not new

- **`experiments/baseline_dynamics_learning/data_generation.py`** — added
  `DisturbanceProcess` (optional AR(1) load) and a `disturbance_autocorr` field
  on `DataGenConfig`. `rho=0` (the default) reproduces every dataset generated
  before this change bit-for-bit — nothing about `train_baselines.py` changes
  unless its config opts in.
- **`experiments/baseline_dynamics_learning/metrics.py`** — appended
  `persistence_prediction`, `physics_prediction`, `oracle_prediction`,
  `regression_scores`, `safety_agreement_metrics`. Nothing existing was edited
  or removed; `physics_informed_metrics` (used by `train_baselines.py`) is untouched.
- **`experiments/baseline_dynamics_learning/config.yaml`** — `models.svm.C`
  1.0 -> 10.0, `epsilon` 0.1 -> 0.01 (see problem 6 below; comment explains why,
  in-file). `train_baselines.py` runs `GridSearchCV` regardless, so this default
  is never actually used there — it only matters for the new feature study.

All 82 previously-passing tests still pass unmodified after these edits.

## Why a study, not a script

The Sep-27 file reported one number (R2 = 0.9997) for "the engineered
features." That framing was the root problem: it never asked *compared to
what*, and it changed the features and the target formulation in the same
commit, so there was no way to tell which one (if either) did anything.
`run_feature_study.py` runs two independent ablations —
**feature ablation** (target fixed, inputs vary) and **target ablation**
(inputs fixed, target formulation varies) — against three stated references
that require no learning at all: persistence, deterministic physics, and an
oracle that is told the true disturbance (not deployable, just a floor).

## The 8 problems, and what running the fix actually shows

**1 & 4 — Redundant / leaky features (`Delta_T_amb`, `T_error` were copies of
`T_current`; the same failure mode nearly reappeared).**
`find_redundant_pairs` runs over every shipped feature set on real generated
data before any model is trained. It's not a one-time manual check: while
building the registry, it caught a *second* instance of the same mistake —
`T_lag_1` correlates 0.9999 with `T_current` (the plant is slow) and equals
`T_current - dT_prev` exactly, so the lag was redundant with a feature already
in the set. It was removed; the final `redundancy_report.json` is empty.
`assert_no_leakage` separately blocks `d`, any `debug_*` column, and `T_next`
from ever being selected as a model input — checked by name, not by
convention, and covered by 5 tests including one that every named feature set
passes it.

**2 — The features didn't help (measured: RF got slightly worse).**
Feature ablation, target held at `delta` (see problem 6 for why `delta`, not
raw `T_next`), CV MSE, held-out test episodes:

| model | raw | physics | history | observer | full |
|---|---|---|---|---|---|
| linear | 0.00335 | 0.00335 | **0.00068** | 0.00068 | 0.00068 |
| random_forest | 0.00638 | 0.00494 | 0.00162 | **0.00094** | 0.00103 |
| svm | 0.00386 | 0.00399 | **0.00082** | 0.00086 | 0.00089 |

`history`/`observer`/`full` (lag and disturbance-observer features) cut MSE
4-7x versus `raw` for every model. `physics` alone (the energy-balance feature
with no history) barely moves linear/SVM and makes Random Forest *worse* —
consistent with `dT_phys` being a fixed deterministic function of columns a
tree already has; it adds a split option, not new information, and costs
splits on nearly-collinear noise. The lag/observer features are the ones that
carry a genuinely new signal: the load disturbance in this dataset is
autocorrelated (`rho=0.9`; `disturbance_autocorr` in `feature_study.yaml`),
so its recent past is informative about its next value — which is exactly
what `d_hat_lag1` (an observer solved from the energy balance, not measured)
and `dT_prev` extract.

**3 — R2 = 0.9997 was misleading (persistence alone scores ~0.998).**
`regression_scores` reports `R2_T_next` alongside `Skill_vs_persistence` and
`Skill_vs_physics` (fraction of the *persistence-relative* or
*physics-relative* error eliminated; 0 = no better than the reference, 1 =
perfect). `reference_floors.csv` states the reference numbers plainly:
persistence MSE 0.0163, physics(d=0) MSE 0.0040, and a disturbance-oracle
floor of 0.0007 that no deployable model can beat (it is handed the true,
unmeasured `d`) — every model number above is judged against these, not in
isolation.

**5 — Constraint Violation Rate was 100% for every model (uninformative).**
Replaced with `safety_agreement_metrics`, evaluated on the dedicated
sustained-actuation stress test (already built for Day 18; reused here):

| recipe | true unsafe | predicted unsafe | missed violations | false alarms |
|---|---|---|---|---|
| v1_reproduced (raw target) | 47.7% | 47.6% | 0.14% | 0.0% |
| best_available (physics_residual target) | 47.7% | 47.6% | 0.14% | 0.0% |

`missed_violation_pct` is the number that matters for a safety argument: of
the steps that are truly unsafe, how many does the model call safe. 0.14%
(both recipes, here) is what "this model's safety verdict is trustworthy"
looks like, as opposed to the old metric's 100%/100% telling you nothing.

**6 — The SVM number was a scaling artifact (R2 = 0.44 on an unscaled 0-580 C
target with C=1).** Target ablation, features held at `full`:

| model | absolute_unscaled | absolute (scaled) | delta | physics_residual |
|---|---|---|---|---|
| linear | 0.000676 | 0.000676 | 0.000676 | 0.000676 |
| random_forest | 0.1651 | 0.1644 | 0.00103 | **0.00072** |
| svm | 0.1434 | 0.1172 | 0.00089 | **0.00075** |

Scaling the target alone (`absolute`) barely helps RF/SVM — the real fix is
the target *formulation*: `delta`/`physics_residual` ask the model to predict
the small part that persistence/physics don't already explain, which is ~150x
smaller in MSE. Linear Regression is invariant to all four (it's already
exact on affine reparametrisations of the same target), which is itself a
useful check: if linear had also moved 150x, the ablation code would be
suspect, not the finding.

**7 — Code reverted to the pre-fix version (`d` as a feature, `u_max=5000`,
one continuous trajectory, shuffled split, global seed).**
`features.py` is built on top of the already-fixed `data_generation.py`
(episode-based, `u_max=300`, `GroupKFold` split by `episode_id`, per-episode
seeded RNGs) — there was no separate copy to drift out of sync with. The lag
leak across episode boundaries (mentioned as a risk in the Sep-27 review) is
closed by construction: every lag in `features.py` groups by `episode_id`
before shifting (`_lag`), and `test_lag_features_do_not_cross_episode_boundaries`
checks it directly, not just the fix that caused it.

**8 — Not reviewed in depth on Sep 27 (`wandb.init` with no `mode`; outputs
dropped in the working directory; no tests, no README).**
`run_feature_study.py` reads `ml_pipeline.wandb_mode` from config (same
pattern as `train_baselines.py`) and writes everything under
`outputs/feature_study/`. 66 new tests (`test_features.py`,
`test_estimators.py`, `test_config.py`, `test_run_feature_study.py`) — this
file is the README.

## Honest limitations

- **Autocorrelation is fixed at 0.9.** The observer features would show
  smaller gains for a less predictable load and no gain at all for white
  noise (`rho=0`, the original default) — that boundary isn't swept here.
  Worth a follow-up ablation over `rho` if the real load's autocorrelation is
  ever estimated from data instead of assumed.
- **Hyperparameters are fixed, not tuned**, deliberately, so the ablations
  isolate features/target rather than confounding them with per-cell tuning.
  This means the reported numbers are not each model's best possible score —
  compare shapes across the ablation, not absolute numbers against literature.
- **`find_redundant_pairs` only catches pairwise affine collinearity.** A
  feature that is redundant only in combination with two or more others (e.g.
  a sum of three existing columns) would pass silently. Worth a VIF-based
  check if the feature set grows past the current 7.

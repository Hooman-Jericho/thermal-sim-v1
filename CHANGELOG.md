# CHANGELOG — `thermal-ml-v1` (Sep 26 submission → this revision)

## Role in the defense line
This experiment is a **baseline / motivation study**, not a competitor to the
thesis's RL contribution. It shows that generic, physics-agnostic ML models
(Linear Regression, Random Forest, SVR) trained to predict the plant's next
temperature either (a) don't know when they're wrong, or (b) don't know
they're extrapolating — and in the worst case, are *silently* both. That is
exactly the failure mode Paper 2's CBF-QP safety layer exists to eliminate.
This experiment produces the **"before"** numbers that Paper 2's results get
compared against. It is also the first user of `src/plant.py`, the actuated
plant class both this experiment and the future `thermal_env.py` (RL) train
against — so nothing here gets thrown away when the RL environment is built.

## Fixes (each maps to a specific problem in the Sep 26 review)

| # | Problem in v1 | Fix |
|---|---|---|
| 1 | `d` (commented "Unmeasured thermal load noise") was used as a model **input feature** — a model trained on it could never be deployed | `d` is generated and recorded (`debug_d`) for diagnostics only; `FEATURE_COLUMNS = ["T_current", "u"]` never includes it. `test_d_is_never_a_feature_column` locks this in. |
| 2 | `train_test_split(shuffle=True)` on autocorrelated time-series rows → train/test leakage | Data is now generated as **independent episodes** (`src/plant.py` + `data_generation.py`), and `episode_train_test_split` holds out **entire episodes**, never splits within one. Hyperparameter tuning (`GridSearchCV`) uses `GroupKFold` on the same episode groups, so tuning doesn't leak either. |
| 3 | "Physics-informed" metric was only a bound check | Bound check is kept (`Constraint_Violation_Rate_pct` — it's the exact quantity the CBF-QP layer targets) **and** a second, independent `Energy_Residual` metric is added: compares the model's prediction against the plant's own deterministic (`d=0`) energy balance, catching e.g. a model that ignores the actuator entirely. |
| 4 | Bound check was uninformative in practice | Discovered while re-running the pipeline: with v1's physics params (`u_max=5000`, `k_loss=5`), steady-state temperature is ~1025°C — **95% of all data** was out of bounds regardless of model quality. Fixed `u_max` to 300 W so steady-state at full actuation (85°C) sits just above `T_max` (80°C) — genuinely marginal, not trivially always-violated. |
| 5 | Control signal `u` was i.i.d. noise redrawn every `dt` | Not a realistic actuator signal and under-excites the plant's slower dynamics (τ = mCp/k_loss = 100 s). Replaced with a held, piecewise-constant signal (`u_hold_steps`, a discretized PRBS) — standard system-identification practice. |
| 6 | Random-excitation test data almost never visits the unsafe region, making CVR ≈ 0% and uninformative | Added a dedicated **safety stress test** (`generate_stress_test_episodes`): sustained near-max actuation, held out of training *and* the nominal test set, used only to compute `Stress_Test_CVR_pct`. |
| 7 | New ODE re-implemented from scratch, ignoring the existing `ThermalSystem` ABC / `SystemState` architecture from `thermal-sim-v1` | New `src/plant.py::ActuatedThermalPlant` **subclasses `ThermalSystem`** (same `reset()`/seeded-`rng` contract as `NewtonianCoolingSystem`/`HeatExchangerSystem`), adding only the control input `u` and disturbance `d` the passive systems don't have. |
| 8 | No connection to the actual thesis method (RL) | Documented explicitly (see "Role in the defense line" above and `train_baselines.py` module docstring) — this is the classical-ML "before" baseline, not a parallel track. |
| 9 | `wandb.init(reinit=True)` — deprecated pattern | `reinit="finish_previous"` + explicit `run.finish()` per model. |
| 10 | No hyperparameter tuning — RF/SVM compared at arbitrary defaults | `GridSearchCV` with `GroupKFold` (episode-grouped) added, config-driven (`tune_hyperparameters`, `param_grid`), toggleable per model. |
| 11 | No plots / no visual diagnostics | Parity plots (predicted vs. actual) per model, logged to W&B and saved under `outputs/`. |
| 12 | Single continuous trajectory (`T[0] = T_amb` once) → low state-space diversity | 15 independent episodes (matches the thesis Super Planner's seed count), each with its own random initial temperature and disturbance realization. |

## A genuine finding, not a paper exercise
Once the stress test was in place, running it surfaced something worth
putting directly in the defense:

```
Model              Nominal-Test CVR   Stress-Test CVR   (Ground truth in stress region: 50.5% unsafe)
Linear_Regression         0.0%              50.5%
Random_Forest             0.0%               0.0%
SVM                       0.0%               0.0%
```

Random Forest and SVR report **zero** predicted constraint violations under
sustained high actuation where the plant is **actually unsafe half the
time** — they don't extrapolate past their training distribution and
default to "safe-looking" predictions instead. Linear Regression, whose
inductive bias happens to match the plant's true (linear) physics, tracks
the real unsafe region correctly but has no mechanism to *know* it's a
constraint. Neither behavior is acceptable for a safety-critical control
loop with no other safeguard — which is precisely the argument for the
CBF-QP safety layer as a **hard constraint enforced regardless of what any
learned model predicts**, not a metric reported after the fact.

## Files added (drop into `thermal-sim-v1/`, nothing existing is modified)
```
src/plant.py                                        # NEW
experiments/baseline_dynamics_learning/
    config.yaml
    data_generation.py
    metrics.py
    train_baselines.py
    utils.py
tests/test_plant.py                                  # NEW
tests/test_data_generation.py                         # NEW
tests/test_metrics.py                                 # NEW
requirements-additions.txt   -> append into requirements.txt
```
`src/core.py`, `src/systems.py`, `tests/test_core.py` are untouched — every
pre-existing `test_core.py` test plus all 16 new tests pass together (30
total in this sandbox run), confirming nothing already-approved broke.

## Verified, not asserted
Ran end-to-end locally (`WANDB_MODE=offline`) before delivering this:
- All new tests pass (`test_plant.py`, `test_data_generation.py`, `test_metrics.py`), alongside the untouched `test_core.py` suite.
- Full pipeline runs start-to-finish and produces the results table above.

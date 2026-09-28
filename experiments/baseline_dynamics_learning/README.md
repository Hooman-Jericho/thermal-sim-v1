# baseline_dynamics_learning

Classical ML baselines (Linear Regression, Random Forest, SVR) for one-step
thermal dynamics prediction on `src/plant.py::ActuatedThermalPlant`.

**Purpose in the thesis:** motivate the CBF-QP safety layer (Paper 2) by
showing what generic ML models do — and fail to do — near the plant's safe
operating limits, before any physics-informed constraint is added. See
`CHANGELOG.md` (repo root) for the full "why" and the fixes applied to the
Sep 26 submission.

## Run

```bash
pip install -r requirements.txt   # after appending requirements-additions.txt
export WANDB_MODE=offline          # or "online" with a configured API key
python -m experiments.baseline_dynamics_learning.train_baselines
```

Outputs land in `outputs/`: `thermal_dataset.csv`, one `.pkl` per model,
one parity plot per model, and `results_summary.csv`.

## What's evaluated and why

| Metric | Computed on | What it tells you |
|---|---|---|
| MSE / MAE / R2 | Nominal held-out episodes (random excitation) | Predictive accuracy under typical operating conditions |
| `Constraint_Violation_Rate_pct` | Nominal held-out episodes | How often the model's own prediction breaks `[T_min, T_max]` under typical conditions (usually near 0% — see below) |
| `Stress_Test_CVR_pct` | Dedicated sustained-high-actuation episodes, held out of training | How the model behaves where the plant is genuinely at risk — the number that matters for the safety argument |
| `Energy_Residual_Mean/Std` | Nominal held-out episodes | Whether the model's prediction agrees with the plant's own (disturbance-free) energy balance — catches a model that ignores the actuator |

Run `tests/test_plant.py`, `tests/test_data_generation.py`,
`tests/test_metrics.py` (repo root) to verify the physics, the
leakage-free split, and the metrics themselves before trusting any of the
above on new data.

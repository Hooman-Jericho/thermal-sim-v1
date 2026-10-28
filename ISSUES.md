# Issue log -- Day 41 bug-fix pass

Scope: issues found by the 2026-10-26/27 audit of this repository against the six
recurring bug patterns (see the planner/project instructions) and the new Day 37-44
code. It is **not** the tracker of last week's open issues: that list was not
available to the audit. Planner Day 41 Definition of Done: every issue is
**closed** or **deferred with a written reason**.

| ID | Finding (verified by running it) | Status | Evidence / reason |
|---|---|---|---|
| I-1 | `plant.reset()` returns a fresh state but does **not** reset the plant (`t` stayed 10 s after `reset()`). A Gymnasium `reset()` written on top of it would silently continue the old episode. | **Closed (documented + pinned by test)** | The contract is now stated in `ThermalSystem.reset`; `reseed()` is the real restart (`test_reset_is_a_factory_and_reseed_is_the_restart`). `thermal_env.reset(seed=)` must call `reseed()` (Day 71-72). |
| I-2 | `ActuatedThermalPlant.simulate()` ran with `u = d = 0` without saying so (60 -> 46.2 deg C pure cooling). | **Closed** | `simulate(duration, u=0, d=0)` takes explicit inputs; default behaviour unchanged (`test_simulate_*`). |
| I-3 | `generate_dataset()` wrote `thermal_dataset.csv` into the current directory when called without `output_path`. | **Closed** | Default is now `None` (no file); `test_generate_dataset_writes_no_file_unless_asked`. |
| I-4 | `wandb.init(mode="online")` with no credentials crashed a fresh machine (the Day 20 failure class). | **Closed in the new code** | `wandb_utils.resolve_mode` downgrades to offline with a warning; `safe_log`/`finish` never raise. The older pipelines already default to offline. |
| I-5 | `disturbance_std` is per-step white noise, so `Var(T)` scales with `dt`. | **Fixed in Plant v2** | OU load (`src/disturbance.py`), exactly discretised; Var(T) is dt-invariant (tested). Legacy config unchanged. |
| I-6 | Training data never visited the safety bounds; the lower bound (20 C) was below ambient. | **Fixed in Plant v2** | `T_min = 40 C`, wide initial temperatures, lower-bound stress set; the heater-off test shows the bound binds. |
| I-7 | The thesis says **nonlinear**; the plant was linear. | **Fixed in Plant v2** | `beta = 0.005` loss nonlinearity, control-affine so CBF-QP stays an interval projection. Caveat: the effect is below the noise floor for one-step learning (see mlp_dynamics README). |
| I-8 | Disturbance model disagreed across the project. | **Fixed in Plant v2** | One model: OU, sigma 30 W, tau 10 s, in `configs/plant_v2.yaml`. Planner Day 74 should say "OU(30 W, 10 s)"; domain randomization covers mCp +/-10 %, k_loss +/-15 %. |
| I-9 | GATE M2 "loss decreasing 20+ epochs" is unrealistic as a strict streak. | **Fixed (decision Oct 28, option a)** | New rule: train loss fell AND last-5-epoch validation loss within 5 % of the theoretical noise floor (`converged_to_noise_floor`). Measured: v2 ratio 0.943, legacy 0.941 -> PASS on both. Caveat: with 1500 validation samples the realised floor itself fluctuates about 4 %, so 5 % is a tight band; the old block-mean rule is still logged as `gate_m2_block_rule_pass` (False on both). |

Checks that found **nothing**: no row-level shuffled split anywhere in `src/` or
`experiments/`; `d`/`T_next` are blocked as features (`assert_no_leakage`); no
affine-copy pair among the `full` features (`find_redundant_pairs` at 0.999 is empty);
every model's target is scaled.

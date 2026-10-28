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
| I-5 | `disturbance_std` is per-step white noise, so `Var(T)` scales with `dt` (measured: 0.00503 at dt=1, 0.00251 at dt=0.5). | **Deferred** to the env design (Day 72-74) | Changing it now would alter every dataset. Written down in `docs/PHYSICS_SPEC.md` section 3. |
| I-6 | Training data never leaves `[25.7, 74.5]` deg C: episodes start in `[T_amb, T_amb+20]`, so neither safety bound is visited outside the stress set. The lower bound (20 deg C) is below ambient and can hardly be reached at all. | **Deferred** to the env design (Day 72-75) | The data generator is for system identification; the RL env defines its own initial states. Decision recorded in the spec, section 2. |
| I-7 | The thesis calls the plant **nonlinear**; `ActuatedThermalPlant` is **linear** in `T`, `u`, `d`. | **Open -- needs your decision** before the Day 88 freeze | `docs/PHYSICS_SPEC.md` section 7 lists the two options. |
| I-8 | Disturbance model disagrees across the project: white 5 W (baseline config), AR(1) rho=0.9 / 30 W (feature study), `N(0, sigma_d)` (planner Day 74). | **Open -- needs your decision** | Section 3 of the spec. |
| I-9 | GATE M2 says "loss decreasing 20+ consecutive epochs". Measured on the 60-epoch curve: longest strictly decreasing run is **3** epochs (train) and 2 (validation); on a 5-epoch moving average 7 and 5. The loss does fall overall (0.491 -> 0.208, best validation at epoch 49), but mini-batch noise makes a strict 20-epoch streak unrealistic. | **Open -- wording needs a decision** | Read the criterion as a trend (e.g. mean of epochs 1-10 > 11-20 > 21-30 ...) and write that rule down; both numbers are logged in the manifest. |

Checks that found **nothing**: no row-level shuffled split anywhere in `src/` or
`experiments/`; `d`/`T_next` are blocked as features (`assert_no_leakage`); no
affine-copy pair among the `full` features (`find_redundant_pairs` at 0.999 is empty);
every model's target is scaled.

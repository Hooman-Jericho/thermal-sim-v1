# Physics model specification (Planner Day 47)

Status: written 2026-10-27 from the code and configs as they are today.
Items marked **DECISION NEEDED** are not settled by the code or the planner;
they must be settled before the environment and reward freeze on Day 88.

## 1. The model

One lumped body (single state `T`), integrated with forward Euler:

    mCp * dT/dt = u - k_loss * (T - T_amb) - d

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| `mCp` | thermal capacity | 500 J/K | `config.yaml: physics.mCp` |
| `k_loss` | heat-loss coefficient | 5 W/K | `physics.k_loss` (the planner calls it "Q_loss") |
| `T_amb` | ambient temperature | 25 deg C | `physics.T_amb` |
| `dt` | Euler step | 1 s | `physics.dt` |
| `u` | heater power (the RL action) | 0 .. 300 W | `data.u_max` |
| `d` | unmeasured load disturbance | see section 3 | `data.disturbance_*` |

Derived (checked in `tests/test_plant_physics.py`, Day 48):
time constant `tau = mCp / k_loss = 100 s`;
steady state `T_ss = T_amb + (u - d) / k_loss`;
at full actuation `T_ss(300 W) = 85 deg C`.
Euler is stable because `dt / tau = 0.01 << 2`; it stays stable over the
planned randomization range (worst case `1.15 * 5 / (0.9 * 500) * 1 = 0.013`).

## 2. State, initial conditions, bounds

* State: `T` (deg C). Observation for the policy: `T` (plus whatever the env adds).
* Initial condition in the data generator: `T0 ~ U(T_amb, T_amb + 20)` per episode.
  **DECISION NEEDED** for the Gymnasium env: the safe band is `[20, 80]`, so
  episodes should also start near both edges (the dataset never does; see ISSUES.md I-6).
* Actuator bound: `u in [0, 300] W`, enforced by clipping in the env (and
  re-checked after the QP solve, Day 84). There is **no cooling actuator**.
* Safety bounds: `T in [20, 80] deg C`.
  * Upper bound: reachable (`T_ss(u_max) = 85 > 80`), margin 5 deg C. Good: the filter has work to do.
  * Lower bound: `T_min = 20 < T_amb = 25`, and with `u >= 0` the plant relaxes to
    `T_amb` at best, so the lower bound can only be crossed by a sustained
    disturbance `d > u + k_loss * (T_amb - T_min) = u + 25 W`. In the nominal
    training data the minimum temperature is 25.7 deg C. **DECISION NEEDED**:
    either accept that the lower barrier is practically inactive (and say so in
    Ch.4), or move `T_min` above `T_amb`, or add a cooling action.

## 3. Disturbance model d(t)

Two different models are implemented today and they do not agree:

| Where | Model | Parameters |
|---|---|---|
| `baseline_dynamics_learning/config.yaml` | white Gaussian, fresh every step | std 5 W |
| `feature_study.yaml` (used by the Day 19/20/43 experiments) | AR(1) `d_k = rho d_{k-1} + sqrt(1-rho^2) std eps` | rho 0.9, std 30 W |
| Planner Day 74 (domain randomization) | `d ~ N(0, sigma_d)` | sigma_d not fixed |

**DECISION NEEDED:** pick one for `thermal_env.py`. Two facts to base it on:
(1) the AR(1) load is the only one for which a disturbance-observer feature can
help, and real loads (occupancy, solar) are autocorrelated; (2) with the
per-step white model the temperature variance scales with `dt`
(`Var(T) ~ dt * sigma^2 / (2 k_loss mCp)`, measured in Day 48), so `sigma_d` has
no physical meaning independent of `dt`. If the planner's `N(0, sigma_d)` is
kept, state `sigma_d` together with `dt`. Recommendation: keep the planner's
white model as the nominal setting (E0/E1) and use the AR(1) load as an extra
robustness scenario, so no hypothesis depends on an unstated choice.

## 4. How the physics enters the thesis

Exactly as stated in the planner (Day 47), and nothing more:

1. **(a)** through the physics-informed **reward-shaping term**;
2. **(b)** through the **ODE-based transition dynamics** used for **domain randomization**
   (`mCp` +/-10 %, `k_loss` +/-15 %, `d ~ N(0, sigma_d)`; justified as engineering
   tolerances and tested by the Ch.5 sensitivity analysis).

It does **not** enter as a physics-based loss on the policy network.
The Day 43 MLP is a *dynamics model* experiment, unrelated to the policy.

## 5. Hypotheses (copied from the planner; Section 0 and Day 47)

| ID | Statement | Tested by |
|---|---|---|
| H1 | The physics-reward term improves energy efficiency without raising the constraint-violation rate. | Ablation 1 on E1 (Day 107) |
| H2 | Domain randomization improves generalization to out-of-distribution disturbances (performance degradation on held-out disturbance draws outside the training range). | Ablation 3 on E2 (Day 109) |
| H3 | Under in-range parameter mismatch, SAC + CBF achieves lower mean episodic cost than the nominal-model MPC. | SAC vs MPC-nominal on E1 (Day 110) |
| H4 | The CBF filter eliminates observed violations on the 20 stress rollouts at <= 5 % performance cost. | Day 84 criteria + Ablation 2 (Day 108) |

Copy H1-H4 and the Section 0 contribution statement into the Ch.3 outline.

## 6. Validation status

Day 48 (`make validate-physics`): the simulator equals closed-form forward Euler
to 1e-9; the error against the exact ODE is first order in `dt` (observed 1.005);
the noise variance matches AR(1) theory within about 1 %; energy balance closes
in joules.

## 7. Open question for the advisor: the plant is linear

The thesis is described as controlling a **nonlinear** thermal system. The model
above is **linear** in `T`, `u` and `d` (the only nonlinearity is the actuator
clip, applied outside the plant, and the CBF constraint). A reviewer will notice.
**DECISION NEEDED before the Day 88 freeze**, one of:

* keep the plant and drop the word "nonlinear" from the contribution statement; or
* add a real nonlinearity to the ODE, e.g. temperature-dependent loss
  `k_loss(T) = k0 (1 + beta (T - T_amb))`, a radiative term
  `eps sigma (T^4 - T_amb^4)`, or an actuator lag / dead time.
  Any of these changes the closed-form checks of Day 48 (they stay valid for `beta = 0`).

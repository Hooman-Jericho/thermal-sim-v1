# thermal-sim-v1

A minimal, extensible object-oriented simulation of thermal systems,
built as the foundational codebase for a physics-informed,
safety-constrained reinforcement learning thesis on thermal energy
optimization.

This repository is intentionally small and physics-correct rather than
feature-rich. It exists to be **grown**, not rewritten: the class
hierarchy here is designed to become the RL environment
(`thermal_env.py`) used later in the project, with the reward
function, action space, and safety layer (Control Barrier Functions)
added on top of the same `step()` / `reset()` interface.

## Why this design

- **`ThermalSystem` (abstract base class)** owns the generic
  simulation loop (`step`, `simulate`) using fixed-timestep forward
  Euler integration — the same discrete-time convention a Gymnasium
  RL environment needs.
- **Every subclass only implements physics** (`_dynamics`), never
  the loop. Adding a new thermal system means writing one method, not
  copying simulation code.
- **State is explicit and named** (`SystemState.temperatures` is a
  dict, not a bare float), so a two-body system and a ten-body system
  share exactly the same interface.
- **Every system takes an optional `seed`** and exposes a seeded
  `self.rng` (`np.random.Generator`), plus a `reseed(seed)` method
  that re-seeds and resets in one call -- the same shape as
  Gymnasium's `Env.reset(seed=...)`. Neither concrete system is
  stochastic yet, but this means adding domain randomization later
  (randomizing `k`, `k_exchange`, initial temperatures) only touches
  `reset()`/`_dynamics()`, not every call site.

## Systems included

| Class | Physics | Purpose |
|---|---|---|
| `NewtonianCoolingSystem` | `dT/dt = -k(T - T_ambient)` | Simplest possible case; has a closed-form solution, used to validate the integrator. |
| `HeatExchangerSystem` | Two coupled bodies exchanging heat with each other and with ambient | Structurally closer to the actual thermal plant this thesis targets. |

## Project structure

```
thermal-sim-v1/
├── src/
│   ├── core.py       # ThermalSystem (ABC) + SystemState — the generic engine
│   └── systems.py    # Concrete physics: NewtonianCoolingSystem, HeatExchangerSystem
├── tests/
│   └── test_core.py  # Validates simulation against analytical solutions
├── simulate.py        # Runs both systems, produces outputs/thermal_sim_v1_demo.png
├── requirements.txt
└── outputs/            # Generated plots (git-ignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the simulation

```bash
python simulate.py
```

Produces `outputs/thermal_sim_v1_demo.png` — a two-panel plot showing
both systems converging to a physically sane equilibrium.

## Run the tests

```bash
PYTHONPATH=. pytest tests/ -v
```

The tests don't just check that the code runs — they check the
**physics** is right, by comparing the numerical simulation against
the closed-form analytical solution for Newton's Law of Cooling
(`T(t) = T_ambient + (T0 - T_ambient) * exp(-k*t)`), and by checking
energy conservation in the two-body exchanger when ambient loss is
disabled.

## Roadmap (this thesis's Day 6 → later milestones)

- [ ] Wrap `HeatExchangerSystem` as a Gymnasium `Env` → `thermal_env.py`
- [ ] Add a reward function for energy-optimal control
- [ ] Add a Control Barrier Function (CBF) safety layer
- [ ] Train SAC / PPO / TD3 baselines against the environment
- [ ] Compare against classical PID / MPC controllers

## License

MIT

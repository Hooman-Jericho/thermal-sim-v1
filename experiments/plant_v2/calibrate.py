"""Reproducible calibration of the Plant v2 numbers (``configs/plant_v2.yaml``).

The nonlinearity ``beta``, the actuator limit ``u_max`` and the lower safety
bound ``T_min`` are DESIGN choices, so they come from stated rules rather than
from a number that "looked fine":

* **R1 (lower bound must be able to bind).** ``T_min = T_amb + 15 K``. With a
  heater-only plant the temperature relaxes to ``T_amb`` at best, so a bound at
  or below ``T_amb`` can never be crossed (the legacy ``T_min = 20`` was dead).
* **R2 (upper bound must be reachable by every plant in the randomization
  range).** ``u_max`` is the smallest multiple of 10 W for which the steady state
  at full power exceeds ``T_max`` by at least 2 K even for the stiffest corner
  (``k_loss`` + 15 %). Otherwise some randomized plants could never be unsafe.
* **R3 (but not absurdly).** At the nominal plant the steady state at full power
  exceeds ``T_max`` by at most 15 K.
* **R4 (a safe action always exists).** At every corner the band edges can be held
  against a load of +/-4 sigma_d (``holdable_disturbance_range``).
* **R5 (the lower bound is dynamically relevant).** Natural cooling from ``T_max``
  to ``T_min`` takes between 0.5 and 5 local time constants.
* **R6 (Euler is stable at every corner).**

``beta = 0.005 / K`` gives a heat loss 27 % above the linear one at ``dT = 55 K``,
the right order of magnitude for natural convection plus radiation; it is an
engineering assumption (like the randomization ranges), not a fit to data.

Run::

    PYTHONPATH=. python -m experiments.plant_v2.calibrate
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from src.analytical import (
    holdable_disturbance_range,
    local_time_constant,
    ode_reference,
    steady_state,
)
from src.plant import PlantConfig

# Inputs of the calibration (everything except u_max and T_min).
MCP, K_LOSS, BETA, T_AMB, DT = 500.0, 5.0, 0.005, 25.0, 1.0
T_MAX, SIGMA_D = 80.0, 30.0
MCP_REL, K_REL = 0.10, 0.15
MARGIN_MIN_K, MARGIN_MAX_K, LOAD_SIGMAS = 2.0, 15.0, 4.0


def corners(u_max: float) -> list[PlantConfig]:
    """Return the 4 extreme plants of the randomization box (plus nominal)."""
    out = [PlantConfig(MCP, K_LOSS, T_AMB, DT, BETA)]
    for m, k in itertools.product((1 - MCP_REL, 1 + MCP_REL), (1 - K_REL, 1 + K_REL)):
        out.append(PlantConfig(MCP * m, K_LOSS * k, T_AMB, DT, BETA))
    return out


def pick_u_max() -> float:
    """Return the smallest multiple of 10 W satisfying R2 (see module doc)."""
    for u in np.arange(100.0, 1000.0, 10.0):
        if min(steady_state(c, u) for c in corners(u)) >= T_MAX + MARGIN_MIN_K:
            return float(u)
    raise RuntimeError("no u_max below 1000 W satisfies R2")


def check(u_max: float, T_min: float) -> dict[str, float | bool]:
    """Evaluate R1-R6 for a candidate ``(u_max, T_min)``; return the evidence."""
    cs = corners(u_max)
    nominal = cs[0]
    margins = [steady_state(c, u_max) - T_MAX for c in cs]
    holdable_ok = True
    worst_reserve = np.inf
    for c in cs:
        lo, hi = holdable_disturbance_range(c, u_max, T_min, T_MAX)
        holdable_ok &= lo <= -LOAD_SIGMAS * SIGMA_D and hi >= LOAD_SIGMAS * SIGMA_D
        worst_reserve = min(worst_reserve, hi - LOAD_SIGMAS * SIGMA_D)
    t = np.linspace(0.0, 4000.0, 4001)
    cool = ode_reference(nominal, t, T_MAX, 0.0, 0.0)
    t_cool = float(t[np.argmax(cool <= T_min)])
    tau_mid = local_time_constant(nominal, 0.5 * (T_min + T_MAX))
    return {
        "R1_lower_bound_above_ambient": T_min >= T_AMB + 15.0 - 1e-9,
        "R2_min_corner_margin_K": min(margins),
        "R2_ok": min(margins) >= MARGIN_MIN_K,
        "R3_nominal_margin_K": margins[0],
        "R3_ok": margins[0] <= MARGIN_MAX_K,
        "R4_ok": bool(holdable_ok),
        "R4_worst_heater_reserve_W": float(worst_reserve),
        "R5_cooling_time_s": t_cool,
        "R5_cooling_time_in_tau": t_cool / tau_mid,
        "R5_ok": 0.5 <= t_cool / tau_mid <= 5.0,
        "R6_ok": True,  # PlantConfig validates Euler stability on construction
    }


def main() -> int:
    """Print the calibration; exit non-zero if a rule fails."""
    T_min = T_AMB + 15.0
    u_max = pick_u_max()
    print(f"R1: T_min = T_amb + 15 K = {T_min:g} deg C")
    print(
        f"R2: smallest u_max (step 10 W) with margin >= {MARGIN_MIN_K} K: {u_max:g} W"
    )
    rep = check(u_max, T_min)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    ok = all(v for k, v in rep.items() if k.endswith("_ok") or k.startswith("R1"))
    print("ALL RULES PASS" if ok else "A RULE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

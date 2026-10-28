"""Compare the simulator with closed-form solutions and save a figure.

Run from the repository root::

    python -m experiments.physics_validation.run_validation

Writes ``docs/physics_validation.png`` and prints the numbers quoted in the
README / thesis (Euler error vs ``dt``, observed order, noise variance).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analytical import (  # noqa: E402
    continuous_response,
    euler_stationary_variance,
    steady_state,
    time_constant,
)
from src.plant import ActuatedThermalPlant, PlantConfig  # noqa: E402

MCP, K_LOSS, T_AMB = 500.0, 5.0, 25.0
U, D, T0, HORIZON = 200.0, 10.0, 25.0, 300.0
OUT = Path("docs/physics_validation.png")


def _run(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the step response and return ``(times, temperatures)``."""
    cfg = PlantConfig(mCp=MCP, k_loss=K_LOSS, T_amb=T_AMB, dt=dt)
    plant = ActuatedThermalPlant(cfg, initial_temp=T0, seed=0)
    n = int(round(HORIZON / dt))
    temps = [T0] + [plant.step(u=U, d=D).temperatures["T"] for _ in range(n)]
    return np.arange(n + 1) * dt, np.array(temps)


def _noise_variance(dt: float, sigma: float, n: int = 200_000) -> float:
    """Return the sample variance of T under white disturbance noise."""
    cfg = PlantConfig(mCp=MCP, k_loss=K_LOSS, T_amb=T_AMB, dt=dt)
    plant = ActuatedThermalPlant(cfg, initial_temp=T_AMB, seed=1)
    burn = int(10 * time_constant(cfg) / dt)
    xs = []
    for i in range(burn + n):
        plant.step(u=0.0, d=plant.sample_disturbance(sigma))
        if i >= burn:
            xs.append(plant.state.temperatures["T"])
    return float(np.var(xs))


def main() -> None:
    """Print the validation numbers and save the figure."""
    ref = PlantConfig(mCp=MCP, k_loss=K_LOSS, T_amb=T_AMB, dt=1.0)
    print(f"tau = {time_constant(ref):.1f} s, T_ss = {steady_state(ref, U, D):.2f} C")

    dts = [4.0, 2.0, 1.0, 0.5, 0.25]
    errs = []
    for dt in dts:
        cfg = PlantConfig(mCp=MCP, k_loss=K_LOSS, T_amb=T_AMB, dt=dt)
        t, sim = _run(dt)
        errs.append(float(np.max(np.abs(sim - continuous_response(cfg, t, T0, U, D)))))
    slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
    print("dt [s]   max |Euler - exact| [C]")
    for dt, e in zip(dts, errs, strict=True):
        print(f"{dt:6.2f}   {e:.5f}")
    print(f"observed order of convergence = {slope:.3f} (theory: 1)")

    print("noise: sigma_d = 5 W, u = 0")
    for dt in (1.0, 0.5):
        cfg = PlantConfig(mCp=MCP, k_loss=K_LOSS, T_amb=T_AMB, dt=dt)
        th, sm = euler_stationary_variance(cfg, 5.0), _noise_variance(dt, 5.0)
        print(
            f"dt={dt}: theory Var(T)={th:.6f}  simulated={sm:.6f}  ({sm / th - 1:+.1%})"
        )

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    t, sim = _run(1.0)
    a.plot(t, continuous_response(ref, t, T0, U, D), "k-", label="exact ODE")
    a.plot(t[::10], sim[::10], "o", ms=4, label="simulator (dt = 1 s)")
    a.axhline(steady_state(ref, U, D), ls=":", c="gray")
    a.axvline(time_constant(ref), ls=":", c="gray")
    a.set(xlabel="time [s]", ylabel="T [deg C]", title="Step response, u = 200 W")
    a.legend()
    b.loglog(dts, errs, "o-", label="max error")
    b.loglog(dts, [errs[2] * dt for dt in dts], "k--", label="slope 1")
    b.set(xlabel="dt [s]", ylabel="max |error| [deg C]", title="Euler convergence")
    b.legend()
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

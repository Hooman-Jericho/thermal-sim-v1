"""
simulate.py
-----------
Entry point for Day 6's deliverable: run both thermal systems and
produce a plot showing a sane, converging temperature curve.

Usage:
    python simulate.py
Output:
    outputs/thermal_sim_v1_demo.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.systems import HeatExchangerSystem, NewtonianCoolingSystem


def run_newtonian_cooling() -> tuple[list[float], list[float]]:
    """Simulate a single hot body cooling toward ambient."""
    system = NewtonianCoolingSystem(
        initial_temp=350.0,  # ~77 C
        ambient_temp=293.15,  # 20 C
        k=0.02,
        dt=1.0,
    )
    trajectory = system.simulate(duration=300.0)  # 5 minutes
    times = [s.t for s in trajectory]
    temps = [s.temperatures["body"] for s in trajectory]
    return times, temps


def run_heat_exchanger() -> tuple[list[float], list[float], list[float]]:
    """Simulate two bodies exchanging heat and equilibrating."""
    system = HeatExchangerSystem(
        hot_initial_temp=360.0,  # ~87 C
        cold_initial_temp=290.0,  # ~17 C
        ambient_temp=293.15,
        k_exchange=0.05,
        k_loss=0.01,
        dt=1.0,
    )
    trajectory = system.simulate(duration=300.0)
    times = [s.t for s in trajectory]
    hot = [s.temperatures["hot"] for s in trajectory]
    cold = [s.temperatures["cold"] for s in trajectory]
    return times, hot, cold


def main() -> None:
    t1, temps1 = run_newtonian_cooling()
    t2, hot, cold = run_heat_exchanger()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(t1, temps1, color="tab:red", linewidth=2)
    axes[0].axhline(293.15, color="gray", linestyle="--", label="Ambient (293.15 K)")
    axes[0].set_title("NewtonianCoolingSystem")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t2, hot, color="tab:red", linewidth=2, label="Hot body")
    axes[1].plot(t2, cold, color="tab:blue", linewidth=2, label="Cold body")
    axes[1].axhline(293.15, color="gray", linestyle="--", label="Ambient (293.15 K)")
    axes[1].set_title("HeatExchangerSystem")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("thermal-sim-v1 — Day 6 convergence plot", fontsize=13)
    fig.tight_layout()

    out_path = "outputs/thermal_sim_v1_demo.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    print(f"Newtonian final temp: {temps1[-1]:.2f} K (ambient: 293.15 K)")
    print(f"Exchanger final hot/cold: {hot[-1]:.2f} K / {cold[-1]:.2f} K")


if __name__ == "__main__":
    main()

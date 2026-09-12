"""
test_core.py
------------
These tests don't just check that the code runs -- they check that
the PHYSICS is correct, by comparing the numerical (forward-Euler)
simulation against the closed-form analytical solution.

For Newton's Law of Cooling:
    dT/dt = -k(T - T_ambient)
the analytical solution is:
    T(t) = T_ambient + (T0 - T_ambient) * exp(-k * t)

A small dt should make forward-Euler converge to this closed form.
This is the standard way to validate a numerical integrator: check it
against a case where you already know the right answer.
"""

import math

import pytest

from src.core import SystemState, ThermalSystem
from src.systems import HeatExchangerSystem, NewtonianCoolingSystem


class TestThermalSystemABC:
    """The base class must not be directly instantiable."""

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ThermalSystem(dt=1.0)  # type: ignore[abstract]

    def test_rejects_nonpositive_dt(self):
        with pytest.raises(ValueError):
            NewtonianCoolingSystem(dt=0.0)
        with pytest.raises(ValueError):
            NewtonianCoolingSystem(dt=-1.0)

    def test_seed_produces_reproducible_rng_stream(self):
        """Two systems built with the same seed must draw identical
        numbers from ``self.rng`` -- the property this needs once
        domain randomization actually starts using it."""
        a = NewtonianCoolingSystem(seed=42)
        b = NewtonianCoolingSystem(seed=42)
        assert a.rng.random() == b.rng.random()

    def test_reseed_resets_state_and_rng(self):
        system = NewtonianCoolingSystem(initial_temp=350.0, seed=1)
        system.step()
        system.step()
        assert system.state.t == pytest.approx(2.0)

        state = system.reseed(seed=7)
        assert state.t == 0.0
        assert state.temperatures["body"] == 350.0
        assert system.seed == 7


class TestNewtonianCoolingSystem:
    def test_reset_returns_initial_state(self):
        system = NewtonianCoolingSystem(initial_temp=350.0, dt=1.0)
        state = system.reset()
        assert isinstance(state, SystemState)
        assert state.t == 0.0
        assert state.temperatures["body"] == 350.0

    def test_single_step_matches_euler_formula(self):
        system = NewtonianCoolingSystem(
            initial_temp=350.0, ambient_temp=293.15, k=0.02, dt=1.0
        )
        expected_dT = -0.02 * (350.0 - 293.15)
        expected_T = 350.0 + expected_dT * 1.0

        new_state = system.step()
        assert new_state.t == pytest.approx(1.0)
        assert new_state.temperatures["body"] == pytest.approx(expected_T)

    def test_converges_toward_ambient(self):
        system = NewtonianCoolingSystem(
            initial_temp=350.0, ambient_temp=293.15, k=0.02, dt=1.0
        )
        trajectory = system.simulate(duration=600.0)  # 10 minutes
        final_temp = trajectory[-1].temperatures["body"]
        assert final_temp == pytest.approx(293.15, abs=1.0)

    def test_matches_analytical_solution(self):
        """The core correctness test: compare against the closed form."""
        T0, T_amb, k, dt = 350.0, 293.15, 0.02, 0.1  # small dt for accuracy
        system = NewtonianCoolingSystem(
            initial_temp=T0, ambient_temp=T_amb, k=k, dt=dt
        )
        duration = 100.0
        trajectory = system.simulate(duration=duration)

        analytical_T = T_amb + (T0 - T_amb) * math.exp(-k * duration)
        numerical_T = trajectory[-1].temperatures["body"]

        # Forward Euler has O(dt) error; dt=0.1 should be within 1%.
        assert numerical_T == pytest.approx(analytical_T, rel=0.01)

    def test_dynamics_output_keys_match_state(self):
        system = NewtonianCoolingSystem(dt=1.0)
        rates = system._dynamics(system.state)
        assert set(rates.keys()) == set(system.state.temperatures.keys())


class TestHeatExchangerSystem:
    def test_reset_returns_both_bodies(self):
        system = HeatExchangerSystem(
            hot_initial_temp=360.0, cold_initial_temp=290.0, dt=1.0
        )
        state = system.reset()
        assert state.temperatures["hot"] == 360.0
        assert state.temperatures["cold"] == 290.0

    def test_hot_cools_and_cold_warms(self):
        system = HeatExchangerSystem(
            hot_initial_temp=360.0,
            cold_initial_temp=290.0,
            ambient_temp=293.15,
            k_exchange=0.05,
            k_loss=0.0,  # isolate pure exchange, no ambient leakage
            dt=1.0,
        )
        state = system.step()
        assert state.temperatures["hot"] < 360.0
        assert state.temperatures["cold"] > 290.0

    def test_energy_conserved_without_ambient_loss(self):
        """With k_loss=0, hot+cold average temperature must be constant
        (energy can only move BETWEEN the bodies, not leave the system)."""
        system = HeatExchangerSystem(
            hot_initial_temp=360.0,
            cold_initial_temp=290.0,
            k_exchange=0.05,
            k_loss=0.0,
            dt=0.5,
        )
        initial_avg = (
            system.state.temperatures["hot"] + system.state.temperatures["cold"]
        ) / 2
        trajectory = system.simulate(duration=200.0)
        final_avg = (
            trajectory[-1].temperatures["hot"] + trajectory[-1].temperatures["cold"]
        ) / 2
        assert final_avg == pytest.approx(initial_avg, abs=0.05)

    def test_converges_to_thermal_equilibrium_with_ambient(self):
        system = HeatExchangerSystem(
            hot_initial_temp=360.0,
            cold_initial_temp=290.0,
            ambient_temp=293.15,
            k_exchange=0.05,
            k_loss=0.01,
            dt=1.0,
        )
        trajectory = system.simulate(duration=1000.0)
        final = trajectory[-1]
        # Both bodies should settle near ambient once enough time passes.
        assert final.temperatures["hot"] == pytest.approx(293.15, abs=1.0)
        assert final.temperatures["cold"] == pytest.approx(293.15, abs=1.0)

    def test_dynamics_output_keys_match_state(self):
        system = HeatExchangerSystem(dt=1.0)
        rates = system._dynamics(system.state)
        assert set(rates.keys()) == set(system.state.temperatures.keys())

import numpy as np
import pytest

from experiments.baseline_dynamics_learning.metrics import physics_informed_metrics


COMMON = dict(dt=1.0, mCp=500.0, k_loss=5.0, T_amb=25.0, t_min=20.0, t_max=80.0)


def test_perfect_prediction_gives_zero_error_metrics():
    T_current = np.array([30.0, 40.0, 50.0])
    u = np.array([100.0, 100.0, 100.0])
    deterministic_next = T_current + COMMON["dt"] * (
        u - COMMON["k_loss"] * (T_current - COMMON["T_amb"])
    ) / COMMON["mCp"]

    m = physics_informed_metrics(
        y_true=deterministic_next, y_pred=deterministic_next,
        T_current=T_current, u=u, **COMMON,
    )
    assert m["MSE"] == pytest.approx(0.0, abs=1e-9)
    assert m["Energy_Residual_Mean"] == pytest.approx(0.0, abs=1e-9)
    assert m["Energy_Residual_Std"] == pytest.approx(0.0, abs=1e-9)


def test_constraint_violation_rate_counts_out_of_bounds():
    y_true = np.array([50.0, 50.0, 50.0, 50.0])
    y_pred = np.array([50.0, 90.0, 10.0, 50.0])  # 2 of 4 outside [20, 80]
    T_current = np.array([50.0, 50.0, 50.0, 50.0])
    u = np.zeros(4)
    m = physics_informed_metrics(y_true=y_true, y_pred=y_pred, T_current=T_current, u=u, **COMMON)
    assert m["Constraint_Violation_Rate_pct"] == pytest.approx(50.0)


def test_energy_residual_detects_ignored_actuator():
    """A model that ignores u entirely should show a large, biased residual."""
    T_current = np.array([30.0, 30.0, 30.0])
    u = np.array([0.0, 500.0, 1000.0])  # model below ignores this
    y_pred_ignoring_u = np.full(3, 30.0)  # predicts "no change" regardless of u
    m = physics_informed_metrics(
        y_true=y_pred_ignoring_u, y_pred=y_pred_ignoring_u,
        T_current=T_current, u=u, **COMMON,
    )
    # Residual vs. the TRUE energy balance (which depends on u) should be
    # large for the high-u rows, since y_pred_ignoring_u doesn't move.
    assert abs(m["Energy_Residual_Mean"]) > 0.1

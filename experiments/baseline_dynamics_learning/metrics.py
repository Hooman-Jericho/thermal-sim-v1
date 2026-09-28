"""
metrics.py
----------
Physics-informed evaluation metrics.

The v1 submission's only "physics-informed" metric was a bound check
(``Constraint_Violation_Rate``). That is kept here (it is a real and
useful safety-relevant number -- it is exactly the quantity the
thesis's CBF-QP safety layer exists to drive to zero), but it is not
sufficient on its own to justify the "physics-informed" label, so a
second, independent check is added: ``Energy_Residual``.

Energy_Residual asks a narrower, model-agnostic question: *ignoring
the unmeasured disturbance d (which is zero-mean over many samples),
does the model's predicted one-step temperature change agree with
what the plant's own energy balance predicts from u alone?* A model
that has learned the right physics should have a residual close to
the disturbance's own standard deviation's contribution -- not
dramatically larger or systematically biased.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def physics_informed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    T_current: np.ndarray,
    u: np.ndarray,
    dt: float,
    mCp: float,
    k_loss: float,
    T_amb: float,
    t_min: float,
    t_max: float,
) -> dict[str, Any]:
    """Standard regression metrics + two physics-informed checks.

    Parameters
    ----------
    y_true, y_pred : predicted/actual T_next for the test set.
    T_current, u : the corresponding input features (needed to compute
        the deterministic, disturbance-free energy-balance prediction).
    dt, mCp, k_loss, T_amb : the plant's own physical parameters
        (``PlantConfig``), so this check uses the SAME physics the
        data was generated from -- not a re-derived approximation.
    t_min, t_max : safety bounds (``config.physics.T_min/T_max``).
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # --- Constraint Violation Rate: % of predictions outside safe bounds.
    violations = np.sum((y_pred < t_min) | (y_pred > t_max))
    cvr = float(violations / len(y_pred) * 100.0)

    # --- Energy-balance residual: deterministic (d=0) prediction from
    # the plant's OWN physics, compared to the model's prediction.
    # Because d is zero-mean and exogenous, this residual's mean
    # should be close to 0 and its spread should track the
    # disturbance's own variance -- a model that ignores the actuator
    # entirely, or gets the sign/scale of k_loss wrong, shows up here
    # as a large or systematically biased residual, REGARDLESS of MSE.
    deterministic_next = T_current + dt * (u - k_loss * (T_current - T_amb)) / mCp
    energy_residual = y_pred - deterministic_next
    energy_residual_mean = float(np.mean(energy_residual))
    energy_residual_std = float(np.std(energy_residual))

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "R2_Score": float(r2),
        "Constraint_Violation_Rate_pct": cvr,
        "Energy_Residual_Mean": energy_residual_mean,
        "Energy_Residual_Std": energy_residual_std,
    }

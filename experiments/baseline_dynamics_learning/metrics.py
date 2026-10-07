"""Physics-informed evaluation metrics for dynamics models.

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
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def physics_informed_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    T_current: NDArray[np.float64],
    u: NDArray[np.float64],
    dt: float,
    mCp: float,
    k_loss: float,
    T_amb: float,
    t_min: float,
    t_max: float,
) -> dict[str, Any]:
    """Compute standard regression metrics plus two physics-informed checks.

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


# ---------------------------------------------------------------------------
# Day 19 additions: honest scores + safety agreement.
# ``physics_informed_metrics`` above is unchanged (train_baselines.py uses it).
# ---------------------------------------------------------------------------


def persistence_prediction(T_current: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the persistence reference: predict that nothing changes."""
    return np.asarray(T_current, dtype=float)


def physics_prediction(
    T_current: NDArray[np.float64],
    u: NDArray[np.float64],
    dt: float,
    mCp: float,
    k_loss: float,
    T_amb: float,
) -> NDArray[np.float64]:
    """One Euler step of the energy balance with the unmeasured load set to 0."""
    T_current, u = np.asarray(T_current, dtype=float), np.asarray(u, dtype=float)
    return T_current + dt * (u - k_loss * (T_current - T_amb)) / mCp


def oracle_prediction(
    y_physics: NDArray[np.float64],
    d_prev: NDArray[np.float64],
    dt: float,
    mCp: float,
    rho: float,
) -> NDArray[np.float64]:
    """ORACLE reference, not a model: physics plus the best guess of the next load.

    For an AR(1) load, E[d_k | d_{k-1}] = rho * d_{k-1}. This knows the true
    previous disturbance (a simulation-only quantity) and therefore marks the
    lowest MSE ANY deployable model can reach; its irreducible error is
    ``(dt/mCp)^2 * (1 - rho^2) * std^2``.
    """
    return (
        np.asarray(y_physics, dtype=float)
        - dt * rho * np.asarray(d_prev, dtype=float) / mCp
    )


def _skill(mse: float, mse_reference: float) -> float:
    """1 = perfect, 0 = no better than the reference, < 0 = worse."""
    return float("nan") if mse_reference <= 0 else 1.0 - mse / mse_reference


def regression_scores(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    T_current: NDArray[np.float64],
    y_persistence: NDArray[np.float64],
    y_physics: NDArray[np.float64],
) -> dict[str, float]:
    """Scores that do not flatter a model for the part of the answer that is free.

    ``R2_T_next`` is kept only to show how misleading it is: ``T_next`` is
    almost ``T_current``, so *doing nothing* scores ~0.998. ``R2_delta`` and
    the two skill scores measure what the model adds beyond that.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    T_current = np.asarray(T_current, float)
    mse = float(mean_squared_error(y_true, y_pred))

    delta_true = y_true - T_current
    ss_tot = float(np.sum((delta_true - delta_true.mean()) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))  # same residual, in delta space
    return {
        "MSE": mse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2_T_next": float(r2_score(y_true, y_pred)),
        "R2_delta": float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        "Skill_vs_persistence": _skill(
            mse, float(mean_squared_error(y_true, y_persistence))
        ),
        "Skill_vs_physics": _skill(mse, float(mean_squared_error(y_true, y_physics))),
    }


def safety_agreement_metrics(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], t_min: float, t_max: float
) -> dict[str, float]:
    """Measure whether the model's *safety verdict* agrees with reality.

    The Sep-27 ``Constraint_Violation_Rate`` counted predictions outside the
    band and returned 100 % for every model, because 99.5 % of the *true*
    labels were outside it too -- it separated no model from another. The
    question that matters for a safety layer is different: of the steps that
    really are unsafe, how many does the model call safe?

    * ``missed_violation_pct`` -- truly unsafe, predicted safe (dangerous
      optimism). NaN when the data contains no true violation.
    * ``false_alarm_pct``      -- truly safe, predicted unsafe (conservative).
    * ``violation_gap_pct``    -- predicted minus true violation rate.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    unsafe_true = (y_true < t_min) | (y_true > t_max)
    unsafe_pred = (y_pred < t_min) | (y_pred > t_max)
    n_unsafe, n_safe = int(unsafe_true.sum()), int((~unsafe_true).sum())
    return {
        "true_violation_pct": 100.0 * float(unsafe_true.mean()),
        "pred_violation_pct": 100.0 * float(unsafe_pred.mean()),
        "violation_gap_pct": 100.0 * float(unsafe_pred.mean() - unsafe_true.mean()),
        "missed_violation_pct": float("nan")
        if n_unsafe == 0
        else 100.0 * float((unsafe_true & ~unsafe_pred).sum() / n_unsafe),
        "false_alarm_pct": float("nan")
        if n_safe == 0
        else 100.0 * float((~unsafe_true & unsafe_pred).sum() / n_safe),
    }

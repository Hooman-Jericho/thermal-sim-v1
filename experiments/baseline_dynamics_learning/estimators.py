"""Build the models compared in the feature study.

Builds the models compared in the feature study.

Two ideas live here, both fixing problems in the Sep-27 draft:

**Target modes.** What a model is asked to predict matters as much as what it
is given. ``T_next`` is almost equal to ``T_current`` (persistence already
scores R2 ~ 0.998), so predicting it directly wastes the model's capacity on
the part that is trivial.

===================  =======================================  ============
``target_mode``      the model learns ...                     prediction
===================  =======================================  ============
absolute_unscaled    T_next, raw units (the Sep-27 setup)      model(X)
absolute             T_next, target standardised               model(X)
delta                T_next - T_current                        T + model(X)
physics_residual     T_next - T_phys  (what physics misses)    T_phys + model(X)
===================  =======================================  ============

All modes are converted back to a ``T_next`` prediction, so their metrics are
directly comparable.

**Target scaling for every model.** The draft fed raw temperatures (up to
~580 C) to an RBF-SVR with ``epsilon = 0.1`` and ``C = 1``: with such a
target the model cannot fit and reported R2 = 0.44 -- a scaling artefact
presented as a model comparison. Every mode except the deliberately
reproduced ``absolute_unscaled`` standardises the target inside the model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.plant import PlantConfig

MODEL_NAMES = ("linear", "random_forest", "svm")
TARGET_MODES = ("absolute_unscaled", "absolute", "delta", "physics_residual")


class PersistenceBaseline:
    """Predicts 'the temperature stays where it is'."""

    def __call__(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Return ``T_current`` as the next-step temperature prediction.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature table with a ``T_current`` column (deg C).

        Returns
        -------
        numpy.ndarray
            Predicted next temperature per row (deg C).
        """
        out: NDArray[np.float64] = X["T_current"].to_numpy(dtype=float)
        return out


class PhysicsBaseline:
    """One forward-Euler step of the plant's energy balance with d = 0.

    Uses only ``T_current`` and ``u``, so it is available for every feature
    set (including ``raw``) and never touches the unmeasured disturbance.
    """

    def __init__(self, plant: PlantConfig) -> None:
        """Store the plant constants used for the Euler step.

        Parameters
        ----------
        plant : PlantConfig
            Plant constants (``mCp``, ``k_loss``, ``T_amb``, ``dt``).
        """
        self.plant = plant

    def __call__(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Return one disturbance-free Euler step from ``(T_current, u)``.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature table with ``T_current`` (deg C) and ``u`` (heater input).

        Returns
        -------
        numpy.ndarray
            Predicted next temperature per row (deg C).
        """
        p = self.plant
        T = X["T_current"].to_numpy(dtype=float)
        u = X["u"].to_numpy(dtype=float)
        pred: NDArray[np.float64] = T + p.dt * (u - p.k_loss * (T - p.T_amb)) / p.mCp
        return pred


# scikit-learn ships no type stubs, so its base classes resolve to Any.
class ResidualRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Learn ``y - baseline(X)`` and add the baseline back at prediction time.

    ``X`` must be a DataFrame: the baseline reads named columns, so a model
    that silently reordered or dropped columns would be caught, not ignored.
    """

    def __init__(
        self,
        estimator: Any,
        baseline: Callable[[pd.DataFrame], NDArray[np.float64]],
    ) -> None:
        """Store the wrapped estimator and the baseline function.

        Parameters
        ----------
        estimator : sklearn estimator
            Unfitted regressor trained on the residual ``y - baseline(X)``.
        baseline : callable
            Maps a feature DataFrame to the baseline prediction.
        """
        self.estimator = estimator
        self.baseline = baseline

    def _base(self, X: pd.DataFrame) -> NDArray[np.float64]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ResidualRegressor needs a pandas DataFrame with named columns."
            )
        return np.asarray(self.baseline(X), dtype=float)

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> ResidualRegressor:
        """Fit the wrapped estimator on the residual ``y - baseline(X)``.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features with named columns.
        y : array-like
            Training targets (``T_next``).

        Returns
        -------
        ResidualRegressor
            The fitted regressor (``self``).
        """
        self.estimator_ = clone(self.estimator).fit(
            X, np.asarray(y, dtype=float) - self._base(X)
        )
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Predict ``T_next`` as the baseline plus the learned residual.

        Parameters
        ----------
        X : pandas.DataFrame
            Features with the same named columns as in ``fit``.

        Returns
        -------
        numpy.ndarray
            Predicted next temperature per row.
        """
        pred: NDArray[np.float64] = self.estimator_.predict(X) + self._base(X)
        return pred


def _base_model(name: str, models_cfg: dict[str, Any], seed: int) -> Any:
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        rf = models_cfg["random_forest"]
        return RandomForestRegressor(
            n_estimators=rf["n_estimators"],
            max_depth=rf.get("max_depth"),
            random_state=seed,
            n_jobs=1,
        )
    if name == "svm":
        sv = models_cfg["svm"]
        return SVR(kernel="rbf", C=sv["C"], epsilon=sv["epsilon"])
    raise ValueError(f"Unknown model '{name}'. Known: {MODEL_NAMES}")


def make_model(
    name: str,
    target_mode: str,
    plant: PlantConfig,
    models_cfg: dict[str, Any],
    seed: int,
) -> Any:
    """Build an unfitted estimator for ``(model, target_mode)``.

    Hyper-parameters are fixed.

    Hyper-parameters are deliberately NOT tuned per feature set: the study
    isolates the effect of the inputs and the target, and per-set tuning
    would confound the two.
    """
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unknown target_mode '{target_mode}'. Known: {TARGET_MODES}")

    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", _base_model(name, models_cfg, seed))]
    )
    if target_mode == "absolute_unscaled":
        return pipeline
    scaled = TransformedTargetRegressor(
        regressor=pipeline, transformer=StandardScaler()
    )
    if target_mode == "absolute":
        return scaled
    baseline = (
        PersistenceBaseline() if target_mode == "delta" else PhysicsBaseline(plant)
    )
    return ResidualRegressor(scaled, baseline)

"""
features.py
-----------
Feature engineering for one-step thermal dynamics prediction (Day 19).

Design rules -- each one exists because the previous version broke it:

1. **Only deployable information.** A feature may use what a controller can
   measure *at prediction time*: ``T_current``, ``u``, and their own past.
   The unmeasured disturbance ``d`` (stored as ``debug_d``) and the target
   ``T_next`` are forbidden by ``assert_no_leakage`` -- a guard that runs on
   every feature request, not a convention.

2. **Episode-aware.** Lags are computed *within* each episode. A plain
   ``df.shift(1)`` on a multi-episode table copies the last row of episode
   *e* into the first row of episode *e+1*.

3. **Common support.** Every registered feature is computed once, and the
   first ``WARMUP_ROWS`` rows of every episode are dropped for *all* feature
   sets alike. Otherwise a set that needs a lag would be evaluated on
   fewer rows than one that does not, and the comparison would be confounded.

4. **Causal.** Feature values at row *k* depend on rows <= *k* only
   (property-tested by truncating episodes).

5. **Every feature must carry information.** The Sep-27 draft shipped
   ``T - T_amb`` and ``T_ref - T``: affine copies of ``T_current`` (|corr| = 1).
   ``find_redundant_pairs`` detects that class of mistake automatically -- and
   it caught a second one while this module was written: ``T_lag_1`` correlates
   0.9999 with ``T_current`` (the state is slow), and equals
   ``T_current - dT_prev``, so the *difference* is the feature, not the lag.

The physically meaningful features come from the plant's own energy balance

    mCp * dT/dt = u - k_loss * (T - T_amb) - d

* ``dT_phys``   -- the temperature change that balance predicts with d = 0.
* ``d_hat_lag1``-- the previous step's disturbance, *reconstructed* from
  measured quantities by solving the balance for d. It is a disturbance
  observer. It only helps when the load is autocorrelated (see
  ``DisturbanceProcess``); for white noise it is exactly uninformative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import pandas as pd

from experiments.baseline_dynamics_learning.data_generation import GROUP_COLUMN, TARGET_COLUMN
from src.plant import PlantConfig

STEP_COLUMN = "step"
BASE_COLUMNS = (GROUP_COLUMN, STEP_COLUMN, "T_current", "u")
DEBUG_PREFIX = "debug_"
FORBIDDEN_FEATURES = frozenset({"d", TARGET_COLUMN})


@dataclass(frozen=True)
class FeatureDef:
    name: str
    group: str            # "base" | "physics" | "history" | "observer"
    max_lag: int          # rows of history the feature needs inside an episode
    description: str
    fn: Callable[[pd.DataFrame, PlantConfig], pd.Series]


def _lag(df: pd.DataFrame, column: str, k: int = 1) -> pd.Series:
    """Per-episode shift: the value ``k`` steps earlier *in the same episode*."""
    return df.groupby(GROUP_COLUMN, sort=False)[column].shift(k)


def _t_current(df, p):
    return df["T_current"]


def _u(df, p):
    return df["u"]


def _dT_phys(df, p):
    return p.dt * (df["u"] - p.k_loss * (df["T_current"] - p.T_amb)) / p.mCp


def _u_lag_1(df, p):
    return _lag(df, "u")


def _dT_prev(df, p):
    return df["T_current"] - _lag(df, "T_current")


def _d_hat_lag1(df, p):
    # Solve  mCp*(T_k - T_{k-1})/dt = u_{k-1} - k_loss*(T_{k-1} - T_amb) - d_{k-1}  for d_{k-1}.
    T_prev, u_prev = _lag(df, "T_current"), _lag(df, "u")
    return u_prev - p.k_loss * (T_prev - p.T_amb) - p.mCp * (df["T_current"] - T_prev) / p.dt


_DEFS: Sequence[FeatureDef] = (
    FeatureDef("T_current", "base", 0, "Current temperature (measured).", _t_current),
    FeatureDef("u", "base", 0, "Current heater power (the action).", _u),
    FeatureDef("dT_phys", "physics", 0,
               "Temperature change predicted by the energy balance with d = 0.", _dT_phys),
    FeatureDef("u_lag_1", "history", 1, "Heater power one step ago.", _u_lag_1),
    FeatureDef("dT_prev", "history", 1, "Last observed temperature change.", _dT_prev),
    FeatureDef("d_hat_lag1", "observer", 1,
               "Previous step's unmeasured load, reconstructed from T and u only.", _d_hat_lag1),
)
FEATURES: dict[str, FeatureDef] = {f.name: f for f in _DEFS}

# Rows dropped at the start of EVERY episode, for every feature set (common support).
WARMUP_ROWS = max(f.max_lag for f in _DEFS)

#: The ablation ladder. Each set adds one *idea* to ``raw``.
FEATURE_SETS: dict[str, list[str]] = {
    "raw": ["T_current", "u"],
    "physics": ["T_current", "u", "dT_phys"],
    "history": ["T_current", "u", "u_lag_1", "dT_prev"],
    "observer": ["T_current", "u", "dT_phys", "d_hat_lag1"],
    "full": ["T_current", "u", "dT_phys", "u_lag_1", "dT_prev", "d_hat_lag1"],
}


def assert_no_leakage(names: Iterable[str]) -> None:
    """Raise if any requested feature is unmeasured, a target, or unknown."""
    for name in names:
        if name in FORBIDDEN_FEATURES or name.startswith(DEBUG_PREFIX):
            raise ValueError(
                f"'{name}' is not deployable: it is the unmeasured disturbance, a debug "
                f"column, or the prediction target, and must never be a model input."
            )
        if name not in FEATURES:
            raise ValueError(f"Unknown feature '{name}'. Known: {sorted(FEATURES)}")


def resolve_feature_set(feature_set: str | Sequence[str]) -> list[str]:
    """Turn a set name (or an explicit list) into a validated list of feature names."""
    if isinstance(feature_set, str):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set '{feature_set}'. Known: {sorted(FEATURE_SETS)}")
        names = list(FEATURE_SETS[feature_set])
    else:
        names = list(feature_set)
    assert_no_leakage(names)
    return names


def _validate_frame(df: pd.DataFrame) -> None:
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Frame is missing required columns: {missing}")
    steps = df.groupby(GROUP_COLUMN, sort=False)[STEP_COLUMN].diff().dropna()
    if not (steps == 1).all():
        raise ValueError(
            "Rows must be ordered by consecutive 'step' within each episode; "
            "lags are meaningless otherwise."
        )


def build_features(df: pd.DataFrame, plant: PlantConfig) -> pd.DataFrame:
    """Compute EVERY registered feature and drop the common warm-up rows.

    Returns a copy of ``df`` (all original columns, including ``debug_*``,
    are kept for diagnostics) with one extra column per feature that is not
    already present. Downstream code must select model inputs through
    ``resolve_feature_set``, never by column position.
    """
    _validate_frame(df)
    out = df.copy()
    for feat in FEATURES.values():
        if feat.name not in out.columns:
            out[feat.name] = feat.fn(df, plant)
    position = out.groupby(GROUP_COLUMN, sort=False).cumcount()
    out = out.loc[position >= WARMUP_ROWS].reset_index(drop=True)
    model_columns = [f for f in FEATURES]
    if out[model_columns].isna().any().any():  # pragma: no cover - defensive
        raise RuntimeError("NaNs remain after warm-up removal; a feature needs more history than WARMUP_ROWS.")
    return out


def find_redundant_pairs(X: pd.DataFrame, threshold: float = 0.999) -> list[tuple[str, str, float]]:
    """Feature pairs whose absolute Pearson correlation reaches ``threshold``.

    Catches *univariate affine copies* such as ``T - 25`` or ``50 - T`` (|corr| = 1).
    It deliberately does not flag a feature that is a combination of *several*
    others (``dT_phys`` is a linear function of ``T`` and ``u``): that is
    redundant for a linear model but not for a tree, which is exactly what the
    ablation measures.
    """
    corr = X.corr().abs()
    pairs = []
    cols = list(X.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.loc[a, b] >= threshold:
                pairs.append((a, b, float(corr.loc[a, b])))
    return sorted(pairs, key=lambda t: -t[2])

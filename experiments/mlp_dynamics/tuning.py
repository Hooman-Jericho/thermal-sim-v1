"""Optuna hyper-parameter search for the MLP (Day 38).

What makes this search honest (each point is a bug this project had):

* **Episode-grouped folds.** Validation folds are whole episodes
  (``GroupKFold`` on ``episode_id``); a row-level split would leak across time.
* **Test set untouched.** Only the TRAIN partition is used here.
* **Early stopping decides the epoch count.** Each trial stops when its
  validation loss stops improving; the final model is later trained for the
  *mean best epoch* of the winning trial, with no validation data needed.
* **Same target as the final model.** The residual target
  (``delta`` / ``physics_residual``) is built exactly as in ``make_model``.
* **Budget is fixed in the config** (trials, folds, cap on epochs, pruner),
  never "run until it looks good".
* **Seeded.** Sampler and network seeds come from the config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from experiments.baseline_dynamics_learning.data_generation import (
    GROUP_COLUMN,
    TARGET_COLUMN,
)
from experiments.baseline_dynamics_learning.estimators import (
    PersistenceBaseline,
    PhysicsBaseline,
)
from src.nn.mlp import TrainConfig, train_mlp
from src.plant import PlantConfig

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def baseline_for(df: pd.DataFrame, target_mode: str, plant: PlantConfig) -> np.ndarray:
    """Return the part of ``T_next`` the model does NOT have to learn.

    ``absolute`` -> 0, ``delta`` -> ``T_current``, ``physics_residual`` -> one
    disturbance-free Euler step. Mirrors ``estimators.make_model``.
    """
    if target_mode == "absolute":
        return np.zeros(len(df))
    if target_mode == "delta":
        return np.asarray(PersistenceBaseline()(df), dtype=float)
    if target_mode == "physics_residual":
        return np.asarray(PhysicsBaseline(plant)(df), dtype=float)
    raise ValueError(f"unsupported target_mode for the MLP search: {target_mode!r}")


def suggest(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    """Draw one hyper-parameter set from the search space in the config."""
    lo, hi = space["hidden_layers"]
    n_layers = trial.suggest_int("hidden_layers", lo, hi)
    width = trial.suggest_int("width", *space["width"], log=True)
    return {
        "hidden": tuple([width] * n_layers),
        "activation": trial.suggest_categorical("activation", space["activation"]),
        "lr": trial.suggest_float("lr", *space["lr"], log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", *space["weight_decay"], log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
    }


@dataclass
class TuneResult:
    """Outcome of a search."""

    best_params: dict[str, Any]
    best_value: float
    n_epochs: int
    trials: pd.DataFrame

    def model_cfg(self) -> dict[str, Any]:
        """Hyper-parameters in the form ``models.mlp`` / ``make_model`` expects."""
        p = self.best_params
        return {
            "hidden": list(p["hidden"]),
            "activation": p["activation"],
            "lr": p["lr"],
            "weight_decay": p["weight_decay"],
            "batch_size": p["batch_size"],
            "max_epochs": self.n_epochs,
        }


def tune_mlp(
    train_df: pd.DataFrame,
    columns: list[str],
    target_mode: str,
    plant: PlantConfig,
    optuna_cfg: dict[str, Any],
    on_trial: Any = None,
) -> TuneResult:
    """Run the grouped-CV Optuna search and return the best settings.

    Parameters
    ----------
    train_df : pandas.DataFrame
        TRAIN partition only (never the test set).
    columns : list of str
        Feature columns.
    target_mode : str
        ``absolute``, ``delta`` or ``physics_residual``.
    plant : PlantConfig
        Plant constants (for the physics baseline).
    optuna_cfg : dict
        The ``optuna`` section of the config.
    on_trial : callable, optional
        ``on_trial(trial_number, value, params)`` after each finished trial
        (used for W&B logging).
    """
    n_folds = optuna_cfg["n_folds"]
    seed = optuna_cfg["seed"]
    cap, patience = optuna_cfg["max_epochs_cap"], optuna_cfg["patience"]
    warmup = optuna_cfg["pruner_warmup_epochs"]

    X = train_df[columns].to_numpy(dtype=float)
    base = baseline_for(train_df, target_mode, plant)
    y = train_df[TARGET_COLUMN].to_numpy(dtype=float) - base
    folds = list(
        GroupKFold(n_splits=n_folds).split(X, y, groups=train_df[GROUP_COLUMN])
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, optuna_cfg["space"])
        cfg = TrainConfig(max_epochs=cap, patience=patience, seed=seed, **params)
        best_vals, best_epochs = [], []
        for fold_i, (tr, va) in enumerate(folds):

            def report(
                epoch: int, _tr: float, val: float | None, fi: int = fold_i
            ) -> None:
                if fi == 0 and val is not None:  # prune on the first fold only
                    trial.report(val, epoch)
                    if epoch >= warmup and trial.should_prune():
                        raise optuna.TrialPruned()

            _, hist = train_mlp(X[tr], y[tr], cfg, X[va], y[va], on_epoch=report)
            best_vals.append(min(hist.val_loss))
            best_epochs.append(hist.best_epoch)
        trial.set_user_attr("mean_best_epoch", float(np.mean(best_epochs)))
        return float(np.mean(best_vals))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=warmup),
    )

    def _cb(_study: optuna.Study, frozen: optuna.trial.FrozenTrial) -> None:
        if on_trial is not None and frozen.value is not None:
            on_trial(frozen.number, frozen.value, frozen.params)

    study.optimize(objective, n_trials=optuna_cfg["n_trials"], callbacks=[_cb])

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError("Optuna finished with no completed trial; raise n_trials")
    best = study.best_trial
    # Re-create the typed parameter dict for the winning trial.
    p = best.params
    best_params = {
        "hidden": tuple([p["width"]] * p["hidden_layers"]),
        "activation": p["activation"],
        "lr": p["lr"],
        "weight_decay": p["weight_decay"],
        "batch_size": p["batch_size"],
    }
    n_epochs = max(1, int(round(best.user_attrs["mean_best_epoch"])))
    logger.info(
        "Optuna: %d complete / %d pruned; best CV val loss %.5f, epochs %d, %s",
        len(done),
        len(study.trials) - len(done),
        best.value,
        n_epochs,
        best_params,
    )
    assert best.value is not None
    return TuneResult(
        best_params, float(best.value), n_epochs, study.trials_dataframe()
    )

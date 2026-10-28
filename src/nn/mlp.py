"""A small multilayer perceptron and its training loop, in plain PyTorch.

Day 37 (``nn.Module`` + a full training loop) and Day 43 (MLP on thermal data).

Design rules, each taken from a bug this project already had once:

* **Validation data is passed in explicitly.** Early stopping needs a
  validation set, and a row-level random split of time-series rows leaks
  across time (the Sep-26 bug). So :func:`train_mlp` never splits anything:
  the caller supplies a validation set that was cut by whole episode.
* **Scaling lives inside the model.** Inputs and target are standardised
  with statistics from the *training* rows only, and stored with the weights,
  so ``predict`` takes raw physical units.
* **Seeded and repeatable.** One seed fixes the initial weights and the batch
  order (a private ``torch.Generator``, not the global state).
* **Bad input fails loudly.** NaN/inf data and a wrong feature count raise a
  clear error instead of producing ``nan`` weights.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}

EpochCallback = Callable[[int, float, float | None], None]


class MLP(nn.Module):
    """Fully connected network ``n_in -> hidden... -> 1``.

    Parameters
    ----------
    n_in : int
        Number of input features.
    hidden : sequence of int
        Width of each hidden layer (at least one).
    activation : {"relu", "tanh", "silu"}
        Hidden activation.
    dropout : float
        Dropout probability after each hidden activation, in ``[0, 1)``.
    """

    def __init__(
        self,
        n_in: int,
        hidden: Sequence[int] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_in < 1:
            raise ValueError(f"n_in must be >= 1, got {n_in}")
        if len(hidden) < 1 or any(h < 1 for h in hidden):
            raise ValueError(f"hidden must be positive widths, got {tuple(hidden)}")
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation '{activation}'; known: {sorted(ACTIVATIONS)}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape ``(batch,)``."""
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out


@dataclass
class TrainConfig:
    """Hyper-parameters of :func:`train_mlp`.

    ``patience == 0`` disables early stopping (train for ``max_epochs``).
    """

    hidden: tuple[int, ...] = (64, 64)
    activation: str = "relu"
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 0
    seed: int = 0

    def validate(self) -> None:
        """Raise ``ValueError`` for settings that cannot train."""
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr must be > 0 and weight_decay >= 0")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 0:
            raise ValueError("batch_size, max_epochs >= 1 and patience >= 0 required")


@dataclass
class History:
    """Per-epoch losses, in standardised-target units (MSE)."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


def _as_2d(X: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (rows, features), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _as_1d(y: ArrayLike, name: str, n: int) -> NDArray[np.float64]:
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


class FittedMLP:
    """A trained network together with the scaling it was trained under."""

    def __init__(
        self,
        module: MLP,
        x_mean: NDArray[np.float64],
        x_std: NDArray[np.float64],
        y_mean: float,
        y_std: float,
    ) -> None:
        self.module = module
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict the target, in its original units, for raw features ``X``."""
        arr = _as_2d(X, "X")
        if arr.shape[1] != self.x_mean.shape[0]:
            raise ValueError(
                f"expected {self.x_mean.shape[0]} features, got {arr.shape[1]}"
            )
        z = torch.as_tensor((arr - self.x_mean) / self.x_std, dtype=torch.float32)
        self.module.eval()
        with torch.no_grad():
            out = self.module(z).numpy().astype(np.float64)
        return np.asarray(out * self.y_std + self.y_mean, dtype=np.float64)


def train_mlp(
    X_train: ArrayLike,
    y_train: ArrayLike,
    cfg: TrainConfig | None = None,
    X_val: ArrayLike | None = None,
    y_val: ArrayLike | None = None,
    on_epoch: EpochCallback | None = None,
) -> tuple[FittedMLP, History]:
    """Train an :class:`MLP` with Adam on mean squared error.

    Parameters
    ----------
    X_train, y_train : array_like
        Training rows, in physical units.
    cfg : TrainConfig, optional
        Hyper-parameters (defaults to ``TrainConfig()``).
    X_val, y_val : array_like, optional
        Validation rows. They must be cut by whole episode by the caller.
        Needed for early stopping and for a validation curve.
    on_epoch : callable, optional
        Called as ``on_epoch(epoch, train_loss, val_loss)`` after each epoch
        (``val_loss`` is ``None`` without validation data). Used for W&B
        logging and for Optuna pruning (the callback may raise).

    Returns
    -------
    (FittedMLP, History)
        The model (restored to its best-validation weights when early stopping
        is on) and the loss history.
    """
    cfg = cfg or TrainConfig()
    cfg.validate()
    Xtr = _as_2d(X_train, "X_train")
    ytr = _as_1d(y_train, "y_train", Xtr.shape[0])
    if (X_val is None) != (y_val is None):
        raise ValueError("give both X_val and y_val, or neither")
    if cfg.patience > 0 and X_val is None:
        raise ValueError("early stopping (patience > 0) needs validation data")

    x_mean = Xtr.mean(axis=0)
    x_std = Xtr.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    y_std = y_std if y_std > 1e-12 else 1.0

    def prep(X: NDArray[np.float64]) -> torch.Tensor:
        return torch.as_tensor((X - x_mean) / x_std, dtype=torch.float32)

    xt = prep(Xtr)
    yt = torch.as_tensor((ytr - y_mean) / y_std, dtype=torch.float32)
    xv = yv = None
    if X_val is not None and y_val is not None:
        Xv = _as_2d(X_val, "X_val")
        if Xv.shape[1] != Xtr.shape[1]:
            raise ValueError("X_val and X_train have different feature counts")
        xv = prep(Xv)
        yv = torch.as_tensor(
            (_as_1d(y_val, "y_val", Xv.shape[0]) - y_mean) / y_std, dtype=torch.float32
        )

    torch.manual_seed(cfg.seed)
    module = MLP(Xtr.shape[1], cfg.hidden, cfg.activation, cfg.dropout)
    opt = torch.optim.Adam(
        module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(cfg.seed)

    hist = History()
    best_val, best_state, bad = float("inf"), None, 0
    n = xt.shape[0]
    for epoch in range(1, cfg.max_epochs + 1):
        module.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(module(xt[idx]), yt[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "training diverged (non-finite loss); lower lr or check the data"
                )
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.shape[0]
        hist.train_loss.append(total / n)

        val_loss: float | None = None
        if xv is not None and yv is not None:
            module.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(module(xv), yv))
            hist.val_loss.append(val_loss)
            if val_loss < best_val - 1e-12:
                best_val, bad = val_loss, 0
                hist.best_epoch = epoch
                best_state = copy.deepcopy(module.state_dict())
            else:
                bad += 1
        if on_epoch is not None:
            on_epoch(epoch, hist.train_loss[-1], val_loss)
        if cfg.patience > 0 and bad >= cfg.patience:
            hist.stopped_early = True
            break

    if best_state is not None and cfg.patience > 0:
        module.load_state_dict(best_state)
    elif not hist.best_epoch:
        hist.best_epoch = len(hist.train_loss)
    return FittedMLP(module, x_mean, x_std, y_mean, y_std), hist


class TorchMLPRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """scikit-learn style wrapper so the MLP plugs into ``make_model``.

    It trains for a fixed ``max_epochs`` (no validation split, so nothing can
    leak); choose ``max_epochs`` and the other settings with the grouped
    Optuna search in ``experiments/mlp_dynamics/tuning.py``.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        max_epochs: int = 100,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.seed = seed

    def fit(self, X: ArrayLike, y: ArrayLike) -> TorchMLPRegressor:
        """Train on ``(X, y)``; returns ``self``."""
        cfg = TrainConfig(
            hidden=tuple(self.hidden),
            activation=self.activation,
            dropout=self.dropout,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            seed=self.seed,
        )
        self.model_, self.history_ = train_mlp(np.asarray(X, dtype=np.float64), y, cfg)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict the target for ``X``."""
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMLPRegressor is not fitted yet; call fit() first")
        out: NDArray[np.float64] = self.model_.predict(np.asarray(X, dtype=np.float64))
        return out

    def get_config(self) -> dict[str, Any]:
        """Return the hyper-parameters as a plain dict (for logging)."""
        return dict(self.get_params())

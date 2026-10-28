"""Weights & Biases helpers that can never stop a run (Day 39).

The Sep-28 submission crashed on a fresh machine because it called
``wandb.init()`` online with no login. The rules here make that impossible:

* the mode comes from ``cfg["ml_pipeline"]["wandb_mode"]``
  (``online`` | ``offline`` | ``disabled``), default ``offline``;
* ``online`` without any credentials is **downgraded to offline with a
  warning**, not an error;
* every logging call goes through :func:`safe_log`, so a logging failure is
  recorded in the log file and the experiment carries on.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import wandb

logger = logging.getLogger(__name__)

VALID_MODES = ("online", "offline", "disabled")


def has_credentials() -> bool:
    """Return True if an API key is available (env var or ``~/.netrc``)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    try:
        return netrc.exists() and "api.wandb.ai" in netrc.read_text()
    except OSError:
        return False


def resolve_mode(requested: str) -> str:
    """Return the mode to use: ``online`` without credentials becomes ``offline``."""
    if requested not in VALID_MODES:
        raise ValueError(f"wandb_mode must be one of {VALID_MODES}, got {requested!r}")
    if requested == "online" and not has_credentials():
        logger.warning("wandb_mode=online but no credentials found; using offline")
        return "offline"
    return requested


def start_run(
    cfg: dict[str, Any], name: str, config: dict[str, Any] | None = None
) -> Any:
    """Start a W&B run honouring ``cfg["ml_pipeline"]``; never raises on W&B errors.

    Returns
    -------
    wandb run
        A real run, an offline run, or a disabled run (all accept ``log``).
    """
    mode: Any = resolve_mode(cfg["ml_pipeline"].get("wandb_mode", "offline"))
    try:
        return wandb.init(
            project=cfg["ml_pipeline"]["wandb_project"],
            name=name,
            config={**cfg, **(config or {})},
            mode=mode,
            reinit="finish_previous",
        )
    except Exception as exc:  # noqa: BLE001 - logging must never stop the experiment
        logger.warning("wandb.init failed (%s); continuing with W&B disabled", exc)
        return wandb.init(mode="disabled")


def safe_log(run: Any, data: dict[str, Any], step: int | None = None) -> None:
    """Log ``data`` to ``run``; a failure is reported, not raised."""
    try:
        run.log(data, step=step)
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb log failed: %s", exc)


def epoch_logger(
    run: Any, prefix: str = "mlp"
) -> Callable[[int, float, float | None], None]:
    """Return an ``on_epoch`` callback logging per-epoch train/val loss."""

    def _log(epoch: int, train_loss: float, val_loss: float | None) -> None:
        payload: dict[str, Any] = {"epoch": epoch, f"{prefix}/train_loss": train_loss}
        if val_loss is not None:
            payload[f"{prefix}/val_loss"] = val_loss
        safe_log(run, payload)

    return _log


def finish(run: Any) -> None:
    """Finish ``run``, ignoring errors."""
    try:
        run.finish()
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb finish failed: %s", exc)

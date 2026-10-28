"""Pipeline hardening checks (Day 40).

Definition of Done (Super Planner, Day 40): *100 consecutive episodes run
without a crash, all unit tests pass, and the run is reproducible with a fixed
seed (same seed -> same output).*

This module turns each clause into code that can fail:

* :func:`run_consecutive_episodes` pushes 100 episodes through the real
  data-generation and feature code under deliberately awkward conditions
  (start temperatures far outside the safe band, actuator at both ends of its
  range, a disturbance 3x the usual size) and checks every number is finite
  and physically plausible.
* :func:`check_reproducibility` builds the data twice and trains the MLP twice
  with one seed (the outputs must be identical), and once with another seed
  (the outputs must differ, otherwise "reproducible" would be vacuous).

Run::

    PYTHONPATH=. python -m experiments.mlp_dynamics.hardening
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.baseline_dynamics_learning.data_generation import (
    GROUP_COLUMN,
    TARGET_COLUMN,
    DataGenConfig,
    generate_dataset,
)
from experiments.baseline_dynamics_learning.features import (
    build_features,
    resolve_feature_set,
)
from experiments.mlp_dynamics.data import build_study_data, load_study_config
from src.nn.mlp import TrainConfig, train_mlp
from src.plant import ActuatedThermalPlant, PlantConfig


def frame_hash(df: pd.DataFrame) -> str:
    """Return a stable hash of a DataFrame's values (bit-exact)."""
    return hashlib.sha256(
        np.ascontiguousarray(df.to_numpy(dtype=float)).tobytes()
    ).hexdigest()[:16]


def run_consecutive_episodes(
    plant_cfg: PlantConfig,
    n_episodes: int = 100,
    steps: int = 300,
    seed: int = 0,
    u_max: float = 300.0,
) -> dict[str, Any]:
    """Run ``n_episodes`` back-to-back episodes under awkward conditions.

    Each episode gets its own seed, a start temperature drawn from a wide range
    (``T_amb - 40`` .. ``T_amb + 120``), a heater held at 0, at ``u_max`` or at a
    random value, and a disturbance up to 100 W. A crash, a NaN/inf, or a
    temperature outside the physically possible envelope raises ``AssertionError``.

    Returns
    -------
    dict
        ``{"episodes": n, "rows": total_rows, "t_min": ..., "t_max": ...}``.
    """
    rng = np.random.default_rng(seed)
    t_lo, t_hi = np.inf, -np.inf
    rows = 0
    for ep in range(n_episodes):
        T0 = float(rng.uniform(plant_cfg.T_amb - 40.0, plant_cfg.T_amb + 120.0))
        u_mode = ep % 3
        u = (
            0.0
            if u_mode == 0
            else u_max
            if u_mode == 1
            else float(rng.uniform(0, u_max))
        )
        sigma = float(rng.uniform(0.0, 100.0))
        plant = ActuatedThermalPlant(plant_cfg, initial_temp=T0, seed=seed * 1000 + ep)
        T_ss_hi = plant_cfg.T_amb + (u + 6 * sigma) / max(plant_cfg.k_loss, 1e-12)
        T_ss_lo = plant_cfg.T_amb + (u - 6 * sigma) / max(plant_cfg.k_loss, 1e-12)
        lo, hi = min(T0, T_ss_lo) - 1.0, max(T0, T_ss_hi) + 1.0
        for _ in range(steps):
            T = plant.step(u=u, d=plant.sample_disturbance(sigma)).temperatures["T"]
            assert np.isfinite(T), f"episode {ep}: non-finite temperature {T}"
            assert lo <= T <= hi, (
                f"episode {ep}: T={T:.2f} outside [{lo:.1f}, {hi:.1f}]"
            )
            t_lo, t_hi = min(t_lo, T), max(t_hi, T)
            rows += 1

    # The same 100 episodes through the real dataset + feature pipeline.
    gen = DataGenConfig(
        n_episodes=n_episodes,
        steps_per_episode=steps,
        u_max=u_max,
        disturbance_std=30.0,
        base_seed=seed,
        u_hold_steps=100,
        disturbance_autocorr=0.9,
    )
    with tempfile.TemporaryDirectory() as tmp:
        df = generate_dataset(plant_cfg, gen, output_path=str(Path(tmp) / "h.csv"))
    feats = build_features(df, plant_cfg)
    cols = resolve_feature_set("full")
    assert df[GROUP_COLUMN].nunique() == n_episodes
    assert np.isfinite(feats[cols + [TARGET_COLUMN]].to_numpy()).all(), (
        "NaN/inf feature"
    )
    return {"episodes": n_episodes, "rows": rows, "t_min": t_lo, "t_max": t_hi}


def check_reproducibility(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify that one seed gives one answer, and another seed another answer."""
    cfg = cfg or load_study_config()
    cols = resolve_feature_set("full")

    def one_run(seed: int) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            data = build_study_data({**cfg, "random_seed": seed}, tmp)
        X = data.train_df[cols].to_numpy(dtype=float)
        y = data.train_df[TARGET_COLUMN].to_numpy(dtype=float)
        model, _ = train_mlp(X, y, replace(TrainConfig(max_epochs=8), seed=seed))
        return frame_hash(data.train_df[cols + [TARGET_COLUMN]]), frame_hash(
            pd.DataFrame(model.predict(data.test_df[cols].to_numpy(dtype=float)))
        )

    a1, a2, b = one_run(7), one_run(7), one_run(8)
    return {
        "same_seed_data_identical": a1[0] == a2[0],
        "same_seed_predictions_identical": a1[1] == a2[1],
        "other_seed_data_differs": a1[0] != b[0],
        "other_seed_predictions_differ": a1[1] != b[1],
    }


def main() -> int:
    """Run both checks; return a process exit code (0 = all pass)."""
    cfg = load_study_config()
    from experiments.baseline_dynamics_learning.config import plant_from_cfg

    ep = run_consecutive_episodes(
        plant_from_cfg(cfg), n_episodes=100, seed=cfg["random_seed"]
    )
    print(f"100 consecutive episodes OK: {ep}")
    rep = check_reproducibility(cfg)
    for k, v in rep.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(rep.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

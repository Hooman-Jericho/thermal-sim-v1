"""Generate the episode dataset the baseline ML models train on.

Generates the dataset the baseline ML models train on, using
``src.plant.ActuatedThermalPlant`` -- the SAME plant class the RL
policy will later be trained against -- instead of a standalone ODE
re-implementation.

Two correctness properties this module exists to guarantee (both were
bugs in the v1 submission):

1. **``d`` is never a feature.** ``d`` is the plant's UNMEASURED load
   disturbance (see ``ActuatedThermalPlant``). It is recorded per-row
   for diagnostics/plots only, under a column prefixed ``debug_`` --
   never included in ``FEATURE_COLUMNS`` -- because a model trained on
   it could never be deployed (the real plant does not expose it).

2. **Independent episodes, not one long trajectory.** Each episode is
   a fresh ``reset()`` with its own seed, its own initial temperature,
   and its own disturbance realization. Consecutive rows *within* an
   episode are still autocorrelated (T_next depends on T_current), so
   splitting must happen *between* episodes (see
   ``episode_train_test_split``), never by shuffling individual rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.plant import ActuatedThermalPlant, PlantConfig

FEATURE_COLUMNS = [
    "T_current",
    "u",
]  # NOTE: 'd' deliberately excluded -- see module docstring
TARGET_COLUMN = "T_next"
GROUP_COLUMN = "episode_id"


@dataclass
class DataGenConfig:
    """Settings for generating the episodic dataset.

    Attributes
    ----------
    n_episodes : int
        Number of independent episodes.
    steps_per_episode : int
        Control steps per episode.
    u_max : float
        Maximum heater input magnitude (same units as the plant's ``u``).
    disturbance_std : float
        Marginal standard deviation of the unmeasured load disturbance.
    base_seed : int
        Base seed; each episode derives its own seed from it.
    u_hold_steps : int
        Number of steps each sampled ``u`` is held for.
    disturbance_autocorr : float
        AR(1) coefficient rho in [0, 1); 0 means i.i.d. (legacy).
    """

    n_episodes: int
    steps_per_episode: int
    u_max: float
    disturbance_std: float
    base_seed: int
    u_hold_steps: int = 1
    disturbance_autocorr: float = 0.0


class DisturbanceProcess:
    """Stationary AR(1) load disturbance, drawn from a plant's OWN seeded RNG.

        d_k = rho * d_{k-1} + sqrt(1 - rho^2) * std * eps_k ,   eps_k ~ N(0, 1)

    The marginal std is ``std`` for every rho, so changing rho changes only how
    *predictable* the load is from its past, never how large it is. Real
    thermal loads (occupancy, solar gain) are strongly autocorrelated, and
    that is the only situation in which the past carries information about
    the next unmeasured load -- i.e. the only situation where a disturbance
    *observer* feature can help.

    ``rho == 0`` calls ``plant.sample_disturbance(std)`` directly, so every
    dataset generated before this option existed is reproduced bit-for-bit.
    """

    def __init__(
        self, plant: ActuatedThermalPlant, std: float, rho: float = 0.0
    ) -> None:
        if not (0.0 <= rho < 1.0):
            raise ValueError(
                f"disturbance autocorrelation must be in [0, 1), got {rho}"
            )
        if std < 0:
            raise ValueError(f"disturbance std must be >= 0, got {std}")
        self._plant = plant
        self.std = std
        self.rho = rho
        self._prev: float | None = None

    def next(self) -> float:
        """Draw the next disturbance value.

        Returns
        -------
        float
            Load disturbance for the next step (same units as ``std``).
        """
        if self.rho == 0.0:
            return self._plant.sample_disturbance(self.std)
        # standard normal from the plant's RNG
        eps = self._plant.sample_disturbance(1.0)
        if self._prev is None:
            d = self.std * eps  # start in the stationary distribution
        else:
            d = self.rho * self._prev + math.sqrt(1.0 - self.rho**2) * self.std * eps
        self._prev = d
        return d


def _run_one_episode(
    episode_id: int,
    plant_cfg: PlantConfig,
    gen_cfg: DataGenConfig,
) -> pd.DataFrame:
    """Simulate one independent episode and return its rows as a DataFrame."""
    # Each episode gets its own seed derived from base_seed, so the
    # whole dataset is reproducible from a single number, but no two
    # episodes share a disturbance stream or an initial temperature.
    episode_seed = gen_cfg.base_seed + episode_id
    init_rng = np.random.default_rng(episode_seed)
    initial_temp = float(init_rng.uniform(plant_cfg.T_amb, plant_cfg.T_amb + 20.0))

    plant = ActuatedThermalPlant(
        plant_cfg, initial_temp=initial_temp, seed=episode_seed
    )
    # A fresh RNG for the *control* signal, independent of the plant's
    # own disturbance RNG, so u and d are never correlated by
    # construction (both should be exogenous to each other).
    u_rng = np.random.default_rng(episode_seed + 10_000)
    dist = DisturbanceProcess(
        plant, gen_cfg.disturbance_std, gen_cfg.disturbance_autocorr
    )

    rows = []
    u = 0.0
    for k in range(gen_cfg.steps_per_episode):
        # Piecewise-constant control: redrawn every u_hold_steps, not
        # every dt. i.i.d.-per-step noise (the v1 approach) is not a
        # physically realistic actuator signal and under-excites the
        # plant's slower dynamics; a held random signal is the
        # standard system-identification input (a discretized PRBS).
        if k % gen_cfg.u_hold_steps == 0:
            u = float(u_rng.uniform(0.0, gen_cfg.u_max))
        T_current = plant.state.temperatures["T"]
        d = dist.next()

        plant.step(u=u, d=d)
        T_next = plant.state.temperatures["T"]

        rows.append(
            {
                GROUP_COLUMN: episode_id,
                "step": k,
                "T_current": T_current,
                "u": u,
                TARGET_COLUMN: T_next,
                "debug_d": d,  # kept for diagnostics/plots only, never a feature
            }
        )
    return pd.DataFrame(rows)


def generate_dataset(
    plant_cfg: PlantConfig,
    gen_cfg: DataGenConfig,
    output_path: str = "thermal_dataset.csv",
) -> pd.DataFrame:
    """Simulate ``gen_cfg.n_episodes`` independent episodes and concatenate them."""
    episodes = [
        _run_one_episode(episode_id, plant_cfg, gen_cfg)
        for episode_id in range(gen_cfg.n_episodes)
    ]
    df = pd.concat(episodes, ignore_index=True)
    df.to_csv(output_path, index=False)
    return df


def generate_stress_test_episodes(
    plant_cfg: PlantConfig,
    n_episodes: int,
    duration_steps: int,
    u_level: float,
    disturbance_std: float,
    base_seed: int,
    disturbance_autocorr: float = 0.0,
) -> pd.DataFrame:
    """Generate episodes under SUSTAINED near-maximal actuation.

    Random-excitation episodes (``generate_dataset``) are what a
    model should be *accurate* on -- they represent typical operating
    conditions and rarely visit the plant's unsafe region, which is
    exactly why ``Constraint_Violation_Rate`` on that data alone is
    close to 0% and not very informative (see CHANGELOG.md). This
    function instead holds the heater at a sustained high level for
    long enough (``duration_steps`` >> the plant's thermal time
    constant) to approach the unsafe steady state on purpose. It is
    the targeted, worst-case-adjacent scenario the thesis's CBF-QP
    safety layer needs to be evaluated against -- this dataset is
    that comparison's "before" baseline, deliberately kept OUT of
    both training and the nominal test set.
    """
    episode_dfs = []
    for i in range(n_episodes):
        seed = base_seed + 90_000 + i
        plant = ActuatedThermalPlant(plant_cfg, initial_temp=plant_cfg.T_amb, seed=seed)
        dist = DisturbanceProcess(plant, disturbance_std, disturbance_autocorr)
        rows = []
        for k in range(duration_steps):
            T_current = plant.state.temperatures["T"]
            d = dist.next()
            plant.step(u=u_level, d=d)
            rows.append(
                {
                    GROUP_COLUMN: f"stress_{i}",
                    "step": k,
                    "T_current": T_current,
                    "u": u_level,
                    TARGET_COLUMN: plant.state.temperatures["T"],
                    "debug_d": d,
                }
            )
        episode_dfs.append(pd.DataFrame(rows))
    return pd.concat(episode_dfs, ignore_index=True)


def episode_train_test_split(
    df: pd.DataFrame,
    n_test_episodes: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole episodes for testing -- never split within an episode.

    This is the fix for the v1 bug: ``sklearn.train_test_split`` with
    its default row-level shuffling leaks information between
    train/test because adjacent rows within an episode are strongly
    autocorrelated (T_next[k] ~= T_current[k+1]). Holding out entire
    episodes tests genuine generalization to unseen trajectories,
    which is the property that actually matters for deployment.
    """
    episode_ids = df[GROUP_COLUMN].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    if n_test_episodes >= len(episode_ids):
        raise ValueError(
            f"n_test_episodes ({n_test_episodes}) must be < total episodes "
            f"({len(episode_ids)})"
        )

    test_ids = set(episode_ids[:n_test_episodes])
    train_df = df[~df[GROUP_COLUMN].isin(test_ids)].reset_index(drop=True)
    test_df = df[df[GROUP_COLUMN].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df

"""The thesis plant as ONE validated object (Plant v2).

``configs/plant_v2.yaml`` is the single source of truth for the physics, the
actuator, the safety band, the load and the randomization box. Everything that
needs them (experiments now, ``thermal_env.py`` later) loads a :class:`PlantSpec`
instead of copying numbers, and records :meth:`PlantSpec.fingerprint` in its run
manifest, which is exactly what the Day-88 environment freeze needs.

Validation encodes the design lessons of this project as errors, so they cannot
come back silently:

* the lower safety bound must be *reachable* (the legacy ``T_min = 20`` lay below
  the ambient temperature of a heater-only plant and never bound);
* the upper bound must be reachable by EVERY plant of the randomization box;
* a safe action must exist at both band edges for loads up to ``4 * sigma_d``
  at every corner (otherwise "the filter keeps it safe" is not even possible);
* forward Euler must be stable at every corner.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from src.analytical import holdable_disturbance_range, steady_state
from src.disturbance import OUDisturbance
from src.plant import PlantConfig

DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[1] / "configs" / "plant_v2.yaml"
LOAD_SIGMAS = 4.0  # loads up to this many sigma_d must be holdable


@dataclass(frozen=True)
class PlantSpec:
    """All plant-level numbers of the thesis environment."""

    mCp: float
    k_loss: float
    beta: float
    T_amb: float
    dt: float
    u_min: float
    u_max: float
    T_min: float
    T_max: float
    dist_std: float
    dist_tau: float
    mCp_rel: float
    k_loss_rel: float

    def __post_init__(self) -> None:
        """Validate on construction (see :meth:`validate`)."""
        self.validate()

    # -- constructors ------------------------------------------------------
    def plant_config(
        self, mCp_scale: float = 1.0, k_loss_scale: float = 1.0
    ) -> PlantConfig:
        """Return the plant, optionally with scaled ``mCp`` / ``k_loss``."""
        return PlantConfig(
            mCp=self.mCp * mCp_scale,
            k_loss=self.k_loss * k_loss_scale,
            T_amb=self.T_amb,
            dt=self.dt,
            beta=self.beta,
        )

    def disturbance(self) -> OUDisturbance:
        """Return a fresh load process with this spec's ``sigma_d``, ``tau``, ``dt``."""
        return OUDisturbance(self.dist_std, self.dist_tau, self.dt)

    def corners(self) -> list[PlantConfig]:
        """Return the nominal plant, then the 4 corners of the randomization box."""
        out = [self.plant_config()]
        for m, k in itertools.product(
            (1 - self.mCp_rel, 1 + self.mCp_rel),
            (1 - self.k_loss_rel, 1 + self.k_loss_rel),
        ):
            out.append(self.plant_config(m, k))
        return out

    # -- checks ------------------------------------------------------------
    def design_report(self) -> dict[str, Any]:
        """Return the evidence for each design rule (see the module docstring)."""
        cs = self.corners()
        margins = [steady_state(c, self.u_max) - self.T_max for c in cs]
        load = LOAD_SIGMAS * self.dist_std
        reserves = []
        for c in cs:
            lo, hi = holdable_disturbance_range(c, self.u_max, self.T_min, self.T_max)
            reserves.append(min(hi - load, -lo - load))
        return {
            "lower_bound_reachable": self.u_min < 0 or self.T_min > self.T_amb,
            "upper_margin_nominal_K": margins[0],
            "upper_margin_worst_corner_K": min(margins),
            "safe_action_exists_at_all_corners": min(reserves) >= 0.0,
            "worst_holdable_reserve_W": min(reserves),
        }

    def validate(self) -> None:
        """Raise ``ValueError`` naming the broken rule."""
        if not self.u_min < self.u_max:
            raise ValueError(f"need u_min < u_max, got {self.u_min}, {self.u_max}")
        if not self.T_min < self.T_max:
            raise ValueError(f"need T_min < T_max, got {self.T_min}, {self.T_max}")
        if self.dist_std < 0 or self.dist_tau <= 0:
            raise ValueError("need disturbance std >= 0 and tau > 0")
        if not (0 <= self.mCp_rel < 1 and 0 <= self.k_loss_rel < 1):
            raise ValueError("randomization fractions must be in [0, 1)")
        rep = self.design_report()  # PlantConfig also checks Euler stability per corner
        if not rep["lower_bound_reachable"]:
            raise ValueError(
                f"T_min={self.T_min} <= T_amb={self.T_amb} with a heater-only plant: "
                "the lower bound can never bind; raise T_min above the ambient"
            )
        if rep["upper_margin_worst_corner_K"] <= 0:
            raise ValueError(
                "the upper bound is unreachable for some randomized plant "
                f"(steady state at u_max is {rep['upper_margin_worst_corner_K']:.2f} K "
                "below T_max at the stiffest corner); raise u_max"
            )
        if not rep["safe_action_exists_at_all_corners"]:
            raise ValueError(
                f"no safe action at a band edge for a load of {LOAD_SIGMAS:g} "
                f"sigma_d (reserve {rep['worst_holdable_reserve_W']:.1f} W); "
                "raise u_max or lower sigma_d"
            )

    def fingerprint(self) -> str:
        """Return a stable 16-hex-digit hash of every number in the spec."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_plant_spec(path: Path | str = DEFAULT_SPEC_PATH) -> PlantSpec:
    """Load and validate a plant spec from YAML."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        ph, ac, sf = raw["physics"], raw["actuator"], raw["safety"]
        di, rn = raw["disturbance"], raw["randomization"]
        return PlantSpec(
            mCp=float(ph["mCp"]),
            k_loss=float(ph["k_loss"]),
            beta=float(ph["beta"]),
            T_amb=float(ph["T_amb"]),
            dt=float(ph["dt"]),
            u_min=float(ac["u_min"]),
            u_max=float(ac["u_max"]),
            T_min=float(sf["T_min"]),
            T_max=float(sf["T_max"]),
            dist_std=float(di["std"]),
            dist_tau=float(di["tau"]),
            mCp_rel=float(rn["mCp_rel"]),
            k_loss_rel=float(rn["k_loss_rel"]),
        )
    except KeyError as exc:
        raise ValueError(f"plant spec {path} is missing the key {exc}") from None

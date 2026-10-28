"""Plant v2: nonlinear loss, OU disturbance, PlantSpec, calibration (Days 49+)."""

import dataclasses
import math

import numpy as np
import pytest

from experiments.plant_v2.calibrate import check, pick_u_max
from src.analytical import (
    euler_response,
    holdable_disturbance_range,
    local_time_constant,
    ode_reference,
    steady_state,
)
from src.disturbance import OUDisturbance
from src.plant import ActuatedThermalPlant, PlantConfig, heat_loss
from src.spec import PlantSpec, load_plant_spec


def _plant(beta=0.005, dt=1.0, T0=25.0):
    cfg = PlantConfig(mCp=500.0, k_loss=5.0, T_amb=25.0, dt=dt, beta=beta)
    return cfg, ActuatedThermalPlant(cfg, initial_temp=T0, seed=0)


# --- beta = 0 is the legacy plant, bit for bit ------------------------------
def test_beta_zero_reproduces_the_legacy_formula_exactly():
    cfg, plant = _plant(beta=0.0, T0=40.0)
    T = 40.0
    rng = np.random.default_rng(0)
    for _ in range(300):
        u, d = float(rng.uniform(0, 300)), float(rng.normal(0, 20))
        T = T + (u - 5.0 * (T - 25.0) - d) / 500.0 * 1.0  # the pre-v2 expression
        assert plant.step(u=u, d=d).temperatures["T"] == T  # equal, not approx


# --- the nonlinearity is real ------------------------------------------------
def test_heat_loss_is_odd_monotone_and_linear_at_beta_zero():
    cfg = PlantConfig(500.0, 5.0, 25.0, 1.0, beta=0.005)
    xs = np.linspace(-60, 90, 301)
    L = heat_loss(cfg, 25.0 + xs)
    assert np.all(np.diff(L) > 0)
    np.testing.assert_allclose(heat_loss(cfg, 25.0 - xs), -heat_loss(cfg, 25.0 + xs))
    lin = PlantConfig(500.0, 5.0, 25.0, 1.0, beta=0.0)
    np.testing.assert_allclose(heat_loss(lin, 25.0 + xs), 5.0 * xs)


def test_superposition_holds_at_beta_zero_and_fails_otherwise():
    def response(beta, u):
        _, p = _plant(beta=beta)
        for _ in range(200):
            p.step(u=u)
        return p.state.temperatures["T"] - 25.0

    for beta, should_add in ((0.0, True), (0.005, False)):
        r1, r2, r12 = (
            response(beta, 100.0),
            response(beta, 150.0),
            response(beta, 250.0),
        )
        assert math.isclose(r1 + r2, r12, rel_tol=1e-9) == should_add


def test_small_signal_matches_the_linear_plant():
    """Near ambient the nonlinear plant must agree with the linear one."""
    _, nl = _plant(beta=0.005)
    _, lin = _plant(beta=0.0)
    for _ in range(100):
        nl.step(u=2.0)
        lin.step(u=2.0)
    assert nl.state.temperatures["T"] == pytest.approx(
        lin.state.temperatures["T"], abs=2e-3
    )


# --- closed forms and references ----------------------------------------------
def test_closed_forms_refuse_the_nonlinear_plant():
    cfg = PlantConfig(500.0, 5.0, 25.0, 1.0, beta=0.005)
    with pytest.raises(ValueError, match="beta"):
        euler_response(cfg, [1], 25.0, 100.0)


@pytest.mark.parametrize("u,d", [(200.0, 10.0), (430.0, -20.0), (0.0, 60.0)])
def test_steady_state_root_satisfies_the_energy_balance(u, d):
    cfg = PlantConfig(500.0, 5.0, 25.0, 1.0, beta=0.005)
    T = steady_state(cfg, u, d)
    assert u - heat_loss(cfg, T) - d == pytest.approx(0.0, abs=1e-9)


def test_simulator_converges_to_true_ode_first_order():
    errs = []
    for dt in (2.0, 1.0, 0.5, 0.25):
        cfg, p = _plant(dt=dt)
        n = int(round(300.0 / dt))
        sim = [25.0] + [p.step(u=300.0, d=10.0).temperatures["T"] for _ in range(n)]
        ref = ode_reference(cfg, np.arange(n + 1) * dt, 25.0, 300.0, 10.0)
        errs.append(float(np.max(np.abs(np.array(sim) - ref))))
    for a, b in zip(errs[:-1], errs[1:], strict=True):
        assert a / b == pytest.approx(2.0, abs=0.1)


def test_ode_reference_matches_closed_form_at_beta_zero():
    from src.analytical import continuous_response

    cfg = PlantConfig(500.0, 5.0, 25.0, 1.0, beta=0.0)
    t = np.linspace(0, 400, 81)
    np.testing.assert_allclose(
        ode_reference(cfg, t, 25.0, 200.0, 10.0),
        continuous_response(cfg, t, 25.0, 200.0, 10.0),
        atol=1e-8,
    )


def test_local_time_constant_shrinks_when_hot():
    cfg = PlantConfig(500.0, 5.0, 25.0, 1.0, beta=0.005)
    assert local_time_constant(cfg, 25.0) == pytest.approx(100.0)
    assert local_time_constant(cfg, 80.0) < 70.0


def test_euler_guard_uses_the_steepest_slope():
    # stable for beta = 0 (dt*k/mCp = 1.8) but not once the loss slope grows
    PlantConfig(mCp=500.0, k_loss=900.0, T_amb=25.0, dt=1.0, beta=0.0)
    with pytest.raises(ValueError, match="unstable"):
        PlantConfig(mCp=500.0, k_loss=900.0, T_amb=25.0, dt=1.0, beta=0.005)
    with pytest.raises(ValueError, match="beta"):
        PlantConfig(500.0, 5.0, 25.0, 1.0, beta=-0.1)


# --- OU disturbance --------------------------------------------------------------
@pytest.mark.parametrize("dt", [1.0, 0.25])
def test_ou_has_the_stated_std_and_autocorrelation(dt):
    ou = OUDisturbance(std=30.0, tau=10.0, dt=dt)
    rng = np.random.default_rng(1)
    x = np.array([ou.next(rng) for _ in range(200_000)])
    assert x.std() == pytest.approx(30.0, rel=0.02)
    lag = int(round(10.0 / dt))  # one correlation time
    ac = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert ac == pytest.approx(math.exp(-1.0), abs=0.02)


def test_temperature_variance_no_longer_depends_on_dt():
    """The Day-48 finding (Var(T) ~ dt for white noise) is gone."""
    var = {}
    for dt in (1.0, 0.25):
        cfg = PlantConfig(500.0, 5.0, 25.0, dt, beta=0.0)
        plant = ActuatedThermalPlant(cfg, initial_temp=25.0, seed=3)
        ou = OUDisturbance(30.0, 10.0, dt)
        xs = []
        n = int(round(150_000 / dt))
        burn = int(round(2000 / dt))
        for i in range(n):
            plant.step(u=0.0, d=ou.next(plant.rng))
            if i >= burn:
                xs.append(plant.state.temperatures["T"])
        var[dt] = float(np.var(xs))
    assert var[0.25] / var[1.0] == pytest.approx(1.0, rel=0.15)


def test_ou_uses_the_rng_it_is_given_and_validates():
    a, b = OUDisturbance(5.0, 10.0, 1.0), OUDisturbance(5.0, 10.0, 1.0)
    seq_a = [a.next(np.random.default_rng(7)) for _ in range(1)]
    seq_b = [b.next(np.random.default_rng(7)) for _ in range(1)]
    assert seq_a == seq_b
    a.reset()
    for kw in ({"std": -1}, {"tau": 0.0}, {"dt": 0.0}, {"std": float("nan")}):
        args = {"std": 5.0, "tau": 10.0, "dt": 1.0, **kw}
        with pytest.raises(ValueError):
            OUDisturbance(**args)


# --- the spec ---------------------------------------------------------------------
def test_shipped_spec_is_valid_stable_and_has_the_calibrated_numbers():
    spec = load_plant_spec()
    assert (spec.u_max, spec.T_min, spec.T_max, spec.beta) == (430.0, 40.0, 80.0, 0.005)
    rep = spec.design_report()
    assert rep["lower_bound_reachable"] and rep["safe_action_exists_at_all_corners"]
    assert (
        2.0 <= rep["upper_margin_worst_corner_K"]
        and rep["upper_margin_nominal_K"] <= 15.0
    )
    assert (
        len(spec.fingerprint()) == 16
        and spec.fingerprint() == load_plant_spec().fingerprint()
    )
    assert len(spec.corners()) == 5


@pytest.mark.parametrize(
    "change,msg",
    [
        ({"T_min": 20.0}, "never bind|can never bind"),
        ({"T_min": 85.0}, "T_min < T_max"),
        ({"u_max": 380.0}, "unreachable"),
        ({"u_max": 100.0}, "unreachable"),
        ({"dist_std": 90.0}, "no safe action"),
        ({"u_min": 500.0}, "u_min < u_max"),
        ({"k_loss_rel": 1.0}, "fractions"),
    ],
)
def test_spec_rejects_each_broken_design_rule(change, msg):
    with pytest.raises(ValueError, match=msg):
        dataclasses.replace(load_plant_spec(), **change)


def test_spec_fingerprint_changes_with_any_number():
    spec = load_plant_spec()
    assert dataclasses.replace(spec, u_max=440.0).fingerprint() != spec.fingerprint()


def test_spec_with_a_missing_key_names_it(tmp_path):
    bad = tmp_path / "p.yaml"
    bad.write_text("physics: {mCp: 1}\n")
    with pytest.raises(ValueError, match="missing"):
        load_plant_spec(bad)


def test_the_lower_bound_really_binds_when_the_heater_is_off():
    spec = load_plant_spec()
    plant = ActuatedThermalPlant(spec.plant_config(), initial_temp=spec.T_max, seed=0)
    low = False
    for _ in range(600):
        if plant.step(u=0.0, d=0.0).temperatures["T"] < spec.T_min:
            low = True
    assert low


def test_the_upper_bound_really_binds_at_full_power():
    spec = load_plant_spec()
    plant = ActuatedThermalPlant(spec.plant_config(), initial_temp=spec.T_amb, seed=0)
    for _ in range(1500):
        plant.step(u=spec.u_max, d=0.0)
    assert plant.state.temperatures["T"] > spec.T_max


# --- the calibration script is the source of the numbers -------------------------
def test_calibration_picks_the_shipped_u_max_and_all_rules_pass():
    spec = load_plant_spec()
    assert pick_u_max() == spec.u_max
    rep = check(spec.u_max, spec.T_min)
    assert all(v for k, v in rep.items() if k.endswith("_ok") or k.startswith("R1"))


def test_holdable_range_is_consistent_with_the_dynamics():
    spec = load_plant_spec()
    cfg = spec.plant_config()
    d_lo, d_hi = holdable_disturbance_range(cfg, spec.u_max, spec.T_min, spec.T_max)
    # at T_min with the heater at u_max and load d_hi the temperature is stationary
    assert spec.u_max - heat_loss(cfg, spec.T_min) - d_hi == pytest.approx(
        0.0, abs=1e-9
    )
    assert -heat_loss(cfg, spec.T_max) - d_lo == pytest.approx(0.0, abs=1e-9)


def test_plantspec_type_is_frozen():
    spec = load_plant_spec()
    assert isinstance(spec, PlantSpec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.u_max = 1.0  # type: ignore[misc]

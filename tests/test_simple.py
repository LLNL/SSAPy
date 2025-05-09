import pytest
from unittest.mock import MagicMock, patch
from astropy.time import Time
import numpy as np

# Patch imports from simple.py module
@pytest.fixture(autouse=True)
def patch_dependencies():
    with patch("simple.AccelKepler", return_value="kepler"), \
         patch("simple.AccelThirdBody", side_effect=lambda body: f"thirdbody-{body}"), \
         patch("simple.AccelHarmonic", side_effect=lambda body, a, b: f"harmonic-{body}-{a}-{b}"), \
         patch("simple.AccelSolRad", return_value="solrad"), \
         patch("simple.AccelEarthRad", return_value="earthrad"), \
         patch("simple.AccelDrag", return_value="drag"), \
         patch("simple.get_body", side_effect=lambda name, **kwargs: f"{name}-body"), \
         patch("simple.RK78Propagator", side_effect=lambda accel, h: f"prop-{accel}-{h}"), \
         patch("simple.Orbit.fromKeplerianElements", return_value="orbit-from-kepler"), \
         patch("simple.Orbit", side_effect=lambda r, v, t0: f"orbit-from-rv-{r}-{v}-{t0}"), \
         patch("simple.rv", return_value=(["r-out"], ["v-out"])), \
         patch("simple.get_times", return_value=["t1", "t2"]), \
         patch("simple.points_on_circle", return_value=[np.array([1, 2, 3])] * 4):
        yield

from ssapy import simple

def test_keplerian_prop():
    result = simple.keplerian_prop(20)
    assert result == "prop-kepler-20"

def test_threebody_prop():
    result1 = simple.threebody_prop(10)
    result2 = simple.threebody_prop(15)  # Check cached
    assert result1 == result2
    assert "thirdbody-moon" in result1

def test_fourbody_prop():
    result1 = simple.fourbody_prop(10)
    result2 = simple.fourbody_prop(15)
    assert result1 == result2
    assert "thirdbody-moon" in result1
    assert "thirdbody-Sun" in result1

def test_best_prop():
    result1 = simple.best_prop(5)
    result2 = simple.best_prop(10)
    assert result1 == result2
    assert "harmonic-Earth-body-140-140" in result1
    assert "solrad" in result1
    assert "drag" in result1

def test_ssapy_kwargs():
    kw = simple.ssapy_kwargs()
    assert kw["mass"] == 250
    assert kw["area"] == 0.022

def test_ssapy_prop():
    result = simple.ssapy_prop(30)
    assert "prop" in result

def test_ssapy_orbit_from_elements():
    r, v, t = simple.ssapy_orbit(a=7000e3)
    assert r == ["r-out"]
    assert v == ["v-out"]
    assert t == ["t1", "t2"]

def test_ssapy_orbit_from_rv():
    r = [7000e3, 0, 0]
    v = [0, 7.5e3, 0]
    r_out, v_out = simple.ssapy_orbit(r=r, v=v)
    assert r_out == ["r-out"]
    assert v_out == ["v-out"]

def test_ssapy_orbit_missing_params():
    with pytest.raises(ValueError):
        simple.ssapy_orbit()

def test_get_similar_orbits():
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7.5e3, 0])
    result = simple.get_similar_orbits(r0, v0)
    assert result.shape[2] == 4

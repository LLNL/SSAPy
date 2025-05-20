import pytest
import numpy as np
from astropy.time import Time
from unittest.mock import patch, MagicMock

from ssapy.simple import (
    keplerian_prop,
    threebody_prop,
    fourbody_prop,
    best_prop,
    ssapy_kwargs,
    ssapy_prop,
    ssapy_orbit,
    get_similar_orbits,
)

@pytest.mark.timeout(30)
def test_keplerian_prop():
    prop = keplerian_prop(20)
    assert hasattr(prop, 'step')  # Should have RK78Propagator-like properties

@pytest.mark.timeout(30)
def test_threebody_prop():
    prop = threebody_prop(30)
    assert hasattr(prop, 'step')

@pytest.mark.timeout(30)
def test_fourbody_prop():
    prop = fourbody_prop(40)
    assert hasattr(prop, 'step')

@pytest.mark.timeout(30)
def test_best_prop():
    kwargs = dict(mass=200, area=0.02, CD=2.5, CR=1.1)
    prop = best_prop(50, kwargs)
    assert hasattr(prop, 'step')

@pytest.mark.timeout(30)
def test_ssapy_kwargs():
    kwargs = ssapy_kwargs(100, 0.01, 2.1, 1.2)
    assert kwargs == {'mass': 100, 'area': 0.01, 'CD': 2.1, 'CR': 1.2}

@pytest.mark.timeout(30)
def test_ssapy_prop():
    prop = ssapy_prop(60)
    assert hasattr(prop, 'step')

@pytest.mark.timeout(30)
@patch("ssapy.orbit.Orbit")
@patch("ssapy.utils.get_times")
@patch("ssapy.compute.rv")
def test_ssapy_orbit_with_keplerian(mock_rv, mock_get_times, mock_orbit):
    mock_get_times.return_value = [Time("2025-01-01") + i for i in range(3)]
    mock_orbit.fromKeplerianElements.return_value = MagicMock()
    mock_rv.return_value = (np.zeros((3, 3)), np.ones((3, 3)))

    r, v, t = ssapy_orbit(a=7000e3)
    assert r.shape == v.shape == (3, 3)
    assert len(t) == 3

@pytest.mark.timeout(30)
@patch("ssapy.utils.points_on_circle")
@patch("ssapy.simple.ssapy_orbit")
def test_get_similar_orbits(mock_ssapy_orbit, mock_points_on_circle):
    mock_points_on_circle.return_value = [np.array([7000e3, 0, 0])] * 2
    mock_ssapy_orbit.return_value = (
        np.random.rand(3, 5), np.random.rand(3, 5), [Time("2025-01-01") + i for i in range(5)]
    )

    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7.5e3, 0])
    trajectories = get_similar_orbits(r0, v0, num_orbits=2)
    assert trajectories.shape[2] == 2
    assert trajectories.shape[0] == 3

@pytest.mark.timeout(30)
def test_ssapy_orbit_errors():
    with pytest.raises(ValueError):
        ssapy_orbit()

    with pytest.raises(ValueError):
        ssapy_orbit(a=None, r=None, v=None)

@pytest.mark.timeout(30)
@patch("ssapy.compute.rv", side_effect=RuntimeError("Failed"))
@patch("ssapy.utils.get_times")
@patch("ssapy.orbit.Orbit")
def test_ssapy_orbit_runtime_error(mock_orbit, mock_get_times, mock_rv):
    mock_get_times.return_value = [Time("2025-01-01") + i for i in range(2)]
    mock_orbit.fromKeplerianElements.return_value = MagicMock()
    r, v, t  = ssapy_orbit(a=7000e3)
    assert np.isnan(r).all() and np.isnan(v).all() and np.isnan(t).all()

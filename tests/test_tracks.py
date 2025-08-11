import numpy as np
from astropy.time import Time
from astropy import units as u
import math
from functools import partial
import pytest

import ssapy
from ssapy.constants import EARTH_MU, RGEO
from ssapy.correlate_tracks import (
    CircVelocityPrior, ZeroRadialVelocityPrior, GaussPrior, VolumeDistancePrior,
    orbit_to_param, make_param_guess, make_optimizer, fit_arc_blind, fit_arc,
    fit_arc_with_gaussian_prior, data_for_satellite, wrap_angle_difference,
    radeczn, param_to_orbit, Track, TrackGauss,TrackBase, MHT, summarize_tracklet,
    summarize_tracklets, iterate_mht, fit_arc_blind_via_track, Hypothesis, time_ordered_satIDs
)
from ssapy.orbit import Orbit
from ssapy import propagator, rvsampler
from .ssapy_test_helpers import sample_GEO_orbit, sample_LEO_orbit, checkAngle, checkSphere

@pytest.fixture
def sample_data():
    dtype = [
        ('satID', 'int'),
        ('time', 'O'),  # Object type to hold astropy Time instances
        ('rStation_GCRF', 'float', (3,)),
        ('vStation_GCRF', 'float', (3,))
    ]
    data = np.zeros(10, dtype=dtype)
    data['satID'] = [2, 3, 1, 5, 4, 3, 2, 1, 5, 4]
    
    # Generate Time objects based on a reference GPS time
    times = np.linspace(0, 100, 10)
    data['time'] = [Time(t, format='gps') for t in times]
    
    data['rStation_GCRF'] = np.random.rand(10, 3)
    data['vStation_GCRF'] = np.random.rand(10, 3)
    return data

@pytest.fixture
def sample_arc():
    dtype = [('satID', 'int'), ('rStation_GCRF', 'float', (3,)), ('vStation_GCRF', 'float', (3,)),
             ('time', 'float'), ('ra', 'float'), ('dec', 'float'), ('pmra', 'float'), ('pmdec', 'float')]
    arc = np.zeros(10, dtype=dtype)
    arc['satID'] = np.arange(10)
    arc['rStation_GCRF'] = np.random.rand(10, 3)
    arc['vStation_GCRF'] = np.random.rand(10, 3)
    arc['time'] = np.linspace(0, 100, 10)
    arc['ra'] = np.random.rand(10)
    arc['dec'] = np.random.rand(10)
    arc['pmra'] = np.random.rand(10)
    arc['pmdec'] = np.random.rand(10)
    return arc

@pytest.fixture
def sample_guess():
    return np.array([1, 2, 3, 4, 5, 6, 123456789])

@pytest.fixture
def sample_gaussian_prior():
    mu = np.array([1, 2, 3, 4, 5, 6, 123456789])
    cinvcholfac = np.eye(6)
    return mu, cinvcholfac

@pytest.fixture
def sample_propagator():
    return propagator.KeplerianPropagator()

@pytest.fixture
def sample_truth():
    return {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

@pytest.fixture
def sample_hypotheses():
    return [Hypothesis([], nsat=1000)]

@pytest.fixture
def mht_instance(sample_data, sample_truth, sample_hypotheses, sample_propagator):
    return MHT(data=sample_data, nsat=1000, truth=sample_truth,
               hypotheses=sample_hypotheses, propagator=sample_propagator)

@pytest.mark.parametrize("mode, expected_cls", [
    ('rv', rvsampler.LMOptimizer),
    ('equinoctial', rvsampler.EquinoctialLMOptimizer),
])
 
def test_make_optimizer_modes(mode, expected_cls):
    param = list(range(9))
    optimizer = make_optimizer(mode=mode, param=param, lsq=False)
    assert optimizer == expected_cls if not isinstance(optimizer, partial) else optimizer.func == expected_cls

@pytest.mark.parametrize("mode", ['invalid', None])
def test_make_optimizer_invalid_mode(mode):
    with pytest.raises(ValueError):
        make_optimizer(mode=mode, param=[1]*9, lsq=False)


@pytest.mark.parametrize("mode", ['rv', 'equinoctial'])
def test_orbit_to_param_and_back(mode):
    original = sample_GEO_orbit(t=1000)
    params = orbit_to_param(original, mode=mode)
    recovered = param_to_orbit(params, mode=mode)
    np.testing.assert_allclose(recovered.r, original.r, atol=1e-6)
    np.testing.assert_allclose(recovered.v, original.v, atol=1e-6)


@pytest.mark.parametrize("input_angle, wrap_range, center, expected", [
    (3, 2 * math.pi, 0.5, (3 + math.pi) % (2 * math.pi) - math.pi),
    (3, 360, 0.25, (3 + 0.25 * 360) % 360 - 0.25 * 360),
    (1000, 360, 0.5, (1000 + 0.5 * 360) % 360 - 0.5 * 360),
])
 
def test_wrap_angle_difference_values(input_angle, wrap_range, center, expected):
    result = wrap_angle_difference(input_angle, wrap_range, center=center)
    assert pytest.approx(result, rel=1e-6) == expected

 
def test_data_for_satellite_behavior(sample_data):
    result = data_for_satellite(sample_data, [1, 3])
    assert set(result['satID']) <= {1, 3}

 
def test_circ_velocity_prior_properties():
    prior = CircVelocityPrior(sigma=0.2)
    assert isinstance(prior, CircVelocityPrior)
    assert math.isclose(prior.sigma, 0.2)

 
def test_zero_radial_velocity_prior_properties():
    prior = ZeroRadialVelocityPrior(sigma=0.3)
    assert isinstance(prior, ZeroRadialVelocityPrior)
    assert math.isclose(prior.sigma, 0.3)

 
def test_gauss_prior_properties():
    mu = np.zeros(6)
    cinv = np.eye(6)
    translator = lambda o: np.ones(6)
    prior = GaussPrior(mu, cinv, translator)
    assert np.array_equal(prior.mu, mu)
    assert np.array_equal(prior.cinvcholfac, cinv)

 
def test_volume_distance_prior_behavior():
    prior = VolumeDistancePrior(scale=RGEO)
    orbit = sample_LEO_orbit(t=0)
    logprob = prior(orbit, 7000e3)
    assert isinstance(logprob, float)
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

@pytest.mark.parametrize("mode, translatorcls", [
    ('rv', rvsampler.ParamOrbitRV),
    ('angle', rvsampler.ParamOrbitAngle),
    ('equinoctial', rvsampler.ParamOrbitEquinoctial),
])
 
def test_make_optimizer_lsq(mode, translatorcls):
    param = list(range(9))
    opt = make_optimizer(mode=mode, param=param, lsq=True)
    assert isinstance(opt, partial)
    assert opt.func == rvsampler.LeastSquaresOptimizer
    assert opt.keywords['translatorcls'] == translatorcls

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

 
def test_fit_arc_blind_returns_expected_values(sample_arc):
    chi2, params, result = fit_arc_blind(sample_arc, mode='rv')
    assert isinstance(chi2, float)
    assert isinstance(params, np.ndarray)
    assert hasattr(result, 'residual')

 
def test_fit_arc_with_gaussian_prior_success(sample_arc, sample_gaussian_prior):
    mu, cinv = sample_gaussian_prior
    chi2, params, result = fit_arc_with_gaussian_prior(sample_arc, mu, cinv, mode='rv')
    assert isinstance(chi2, float)
    assert isinstance(params, np.ndarray)
    assert hasattr(result, 'residual')

 
def test_data_for_satellite_behavior(sample_data):
    result = data_for_satellite(sample_data, [1, 3])
    assert set(result['satID']) <= {1, 3}

 
def test_mht_lifecycle(mht_instance):
    mht_instance.run()
    assert mht_instance.track2hyp
    mht_instance.add_tracklet(1)
    assert mht_instance.track2hyp
    pruned = mht_instance.prune_tracks(1)
    assert isinstance(pruned, np.ndarray)

 
def test_summarize_tracklets_structure():
    times = [Time(1000, format='gps'), Time(2000, format='gps')]
    data = np.array([
        {'satID': 1, 'time': t, 'ra': 10*u.deg, 'dec': 20*u.deg,
         'rStation_GCRF': [6371e3, 0, 0]*u.m, 'vStation_GCRF': [0, 0, 0]*u.m/u.s} for t in times
    ], dtype=object)
    summarized = summarize_tracklets(data)
    for key in ['dra', 'ddec', 'pmra', 'pmdec', 'dpmra', 'dpmdec', 't_baseline']:
        assert key in summarized.dtype.names

 
def test_radeczn_output_shapes():
    orbit = sample_GEO_orbit(t=1000)
    arc = {'time': np.array([1000, 2000]),
           'rStation_GCRF': np.array([[6371e3, 0, 0]] * 2),
           'vStation_GCRF': np.array([[0, 0, 0]] * 2)}
    rr, dd, zz, pmrr, pmdd, dzzdt, nwrap = radeczn(orbit, arc)
    for arr in [rr, dd, zz, pmrr, pmdd, dzzdt, nwrap]:
        assert arr.shape == (2,)

 
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

 
def test_make_param_guess_modes():
    arc = sample_arc
    rvguess = [7000, 0, 0, 0, 7.5, 0]
    for mode in ['rv', 'equinoctial', 'angle']:
        result = make_param_guess(rvguess, arc, mode, orbitattr=['mass'])
        assert isinstance(result, list) or isinstance(result, np.ndarray)

 
def test_fit_arc_result_structure(sample_arc, sample_guess):
    chi2, params, result = fit_arc(sample_arc, sample_guess, mode='rv')
    assert isinstance(chi2, float)
    assert isinstance(params, np.ndarray)
    assert hasattr(result, 'residual')

 
def test_track_usage():
    arc = sample_arc
    guess = [7000, 0, 0, 0, 7.5, 0, 123456789]
    track = Track([1, 2], arc, guess=guess, mode='rv', propagator=propagator.KeplerianPropagator())
    assert isinstance(track, Track)
    assert track.param is not None
    assert isinstance(track.param, np.ndarray)

 
def test_track_gauss_functionality():
    track = Track([1, 2], sample_arc, mode='rv', propagator=propagator.KeplerianPropagator())
    gauss_track = track.gaussian_approximation()
    assert isinstance(gauss_track, TrackGauss)

 
def test_iterate_mht_returns_valid():
    data = sample_data()
    mht = MHT(data, nsat=1000, hypotheses=[Hypothesis([], nsat=1000)], propagator=propagator.KeplerianPropagator())
    result = iterate_mht(data, mht, nminlength=2, trimends=1)
    assert isinstance(result, MHT)

 
def test_fit_arc_blind_via_track_execution():
    result = fit_arc_blind_via_track(sample_data(), propagator=propagator.KeplerianPropagator())
    assert isinstance(result, list)
    assert all(isinstance(r, Track) for r in result)

 
def test_summarize_tracklet_output():
    arc = np.array([
        {'time': Time(1000, format='gps'), 'ra': 10*u.deg, 'dec': 20*u.deg, 'sigma': 0.1*u.deg},
        {'time': Time(2000, format='gps'), 'ra': 11*u.deg, 'dec': 21*u.deg, 'sigma': 0.1*u.deg},
    ], dtype=object)
    pos, unc, pm, pm_unc = summarize_tracklet(arc)
    for val in list(pos) + list(unc) + list(pm) + list(pm_unc):
        assert hasattr(val, 'unit')

 
def test_make_param_guess_shapes(sample_arc):
    rvguess = [7000e3, 0, 0, 0, 7.5e3, 0]
    param = make_param_guess(rvguess, sample_arc, mode='rv')
    assert len(param) >= 7

 
def test_time_ordered_satIDs():
    data = np.array([(1, Time(1000, format='gps')), (2, Time(900, format='gps'))],
                    dtype=[('satID', int), ('time', Time)])
    result = time_ordered_satIDs(data)
    assert result == [2, 1]

 
def test_summarize_tracklets_output_fields(sample_data):
    summarized = summarize_tracklets(sample_data)
    required_fields = ['dra', 'ddec', 'pmra', 'pmdec', 'dpmra', 'dpmdec', 't_baseline']
    for field in required_fields:
        assert field in summarized.dtype.names

 
def test_trackbase_instantiation(sample_data):
    tb = TrackBase([1, 2], sample_data)
    assert tb.satIDs == [1, 2]
    assert isinstance(tb.times, np.ndarray)
    assert tb.volume > 0

 
def test_track_gaussian_approximation(sample_data):
    t = Track([1, 2, 3, 4], sample_data)
    gauss_track = t.gaussian_approximation()
    assert isinstance(gauss_track, (TrackGauss, Track))

 
def test_mht_run_and_hypothesis(sample_data):
    mht = MHT(sample_data, nsat=100)
    mht.run()
    assert len(mht.hypotheses) > 0
    for h in mht.hypotheses:
        assert hasattr(h, 'lnprob')

def test_iterate_mht(sample_data):
    mht = MHT(sample_data, nsat=100)
    mht.run()
    new_mht = iterate_mht(sample_data, mht)
    assert isinstance(new_mht, MHT)
    assert len(new_mht.hypotheses) > 0

def test_track_repr(sample_data):
    track = Track([1, 2, 3], sample_data)
    rep = repr(track)
    assert 'Track' in rep and 'chi2' in rep

def test_gauss_repr(sample_data):
    track = Track([1, 2, 3, 4], sample_data)
    gauss = track.gaussian_approximation()
    rep = repr(gauss)
    assert 'Track' in rep

def test_track_addto(sample_data):
    t1 = Track([1, 2, 3], sample_data)
    t2 = t1.addto(4)
    assert 4 in t2.satIDs and len(t2.satIDs) == 4
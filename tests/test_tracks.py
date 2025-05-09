import pytest
import numpy as np
from astropy.time import Time
from astropy import units as u
import math
import unittest

import ssapy
from ssapy.correlate_tracks import (CircVelocityPrior, ZeroRadialVelocityPrior, GaussPrior, VolumeDistancePrior, orbit_to_param, partial, 
                                    make_param_guess, make_optimizer,fit_arc_blind, fit_arc, fit_arc_with_gaussian_prior, data_for_satellite,
                                    wrap_angle_difference, radeczn, param_to_orbit, orbit_to_param, TrackBase, Track, TrackGauss, MHT,
                                    data_for_satellite, Hypothesis, time_ordered_satIDs, summarize_tracklet, summarize_tracklets, iterate_mht,
                                    fit_arc_blind_via_track)
from ssapy.constants import EARTH_MU, RGEO
from .ssapy_test_helpers import sample_GEO_orbit, sample_LEO_orbit, checkAngle, checkSphere 
from ssapy.utils import normed
from ssapy.orbit import Orbit
from ssapy import propagator
from ssapy import rvsampler

####################
## TESTING PRIORS ##
####################
def test_circ_velocity_prior_init():
    sigma = np.log(2) / 3.5
    prior = CircVelocityPrior(sigma=sigma)
    assert prior.sigma == sigma

def test_circ_velocity_prior_call():
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)  # Sample a LEO orbit
    distance = np.linalg.norm(orbit.r)  # Calculate distance from position vector
    prior = CircVelocityPrior(sigma=np.log(2) / 3.5)

    # Calculate expected values
    loggmorv2 = np.log(EARTH_MU / np.linalg.norm(orbit.r) / np.dot(orbit.v, orbit.v))
    chi0 = loggmorv2 / prior.sigma
    expected_logprior = -0.5 * chi0**2

    result = prior(orbit, distance)
    assert pytest.approx(result[0]) == expected_logprior

def test_circ_velocity_prior_call_chi():
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)  # Sample a LEO orbit
    distance = np.linalg.norm(orbit.r)  # Calculate distance from position vector
    prior = CircVelocityPrior(sigma=np.log(2) / 3.5)

    # Calculate expected values
    loggmorv2 = np.log(EARTH_MU / np.linalg.norm(orbit.r) / np.dot(orbit.v, orbit.v))
    chi0 = loggmorv2 / prior.sigma
    expected_chi = chi0

    # Call the method with chi=True and verify output
    result = prior(orbit, distance, chi=True)
    assert pytest.approx(result[0]) == expected_chi

# Test ZeroRadialVelocityPrior initialization
def test_zero_radial_velocity_prior_init():
    sigma = 0.25
    prior = ZeroRadialVelocityPrior(sigma=sigma)
    assert prior.sigma == sigma

# Test __call__ method with valid inputs
def test_zero_radial_velocity_prior_call():
    # Use utility function to sample an orbit
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)  # Sample a LEO orbit
    distance = np.linalg.norm(orbit.r)  # Calculate distance from position vector
    prior = ZeroRadialVelocityPrior(sigma=0.25)

    # Calculate expected values
    vdotr = np.dot(normed(orbit.r), normed(orbit.v))
    chi0 = vdotr / prior.sigma
    expected_logprior = -0.5 * chi0**2

    # Call the method and verify output
    result = prior(orbit, distance)
    assert pytest.approx(result[0]) == expected_logprior

# Test __call__ method with chi=True
def test_zero_radial_velocity_prior_call_chi():
    # Use utility function to sample an orbit
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)  # Sample a LEO orbit
    distance = np.linalg.norm(orbit.r)  # Calculate distance from position vector
    prior = ZeroRadialVelocityPrior(sigma=0.25)

    # Calculate expected values
    vdotr = np.dot(normed(orbit.r), normed(orbit.v))
    chi0 = vdotr / prior.sigma
    expected_chi = chi0

    # Call the method with chi=True and verify output
    result = prior(orbit, distance, chi=True)
    assert pytest.approx(result[0]) == expected_chi

# Test GaussPrior initialization with orbit_to_param
def test_gauss_prior_init_with_orbit_to_param():
    mu = np.zeros(6)  # Example mean (all zeros)
    cinvcholfac = np.eye(6)  # Example inverse Cholesky factor (identity matrix)
    translator = lambda orbit: orbit_to_param(orbit, mode='rv')  # Use orbit_to_param as translator

    prior = GaussPrior(mu=mu, cinvcholfac=cinvcholfac, translator=translator)

    assert np.array_equal(prior.mu, mu)
    assert np.array_equal(prior.cinvcholfac, cinvcholfac)
    assert prior.translator == translator

# Test __call__ method with orbit_to_param translator
def test_gauss_prior_call_with_orbit_to_param():
    # Use utility function to sample an orbit
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)  # Sample a LEO orbit
    distance = np.linalg.norm(orbit.r)  # Calculate distance from position vector

    mu = np.zeros(6)  # Example mean (all zeros)
    cinvcholfac = np.eye(6)  # Example inverse Cholesky factor (identity matrix)
    translator = translator = partial(orbit_to_param, mode='rv', fitonly=True)

    prior = GaussPrior(mu=mu, cinvcholfac=cinvcholfac, translator=translator)

    # Calculate expected values
    param = translator(orbit)
    chi0 = cinvcholfac.dot(param - mu)
    expected_logprior = -0.5 * np.sum(chi0**2)

    # Call the method and verify output
    result = prior(orbit, distance)
    assert pytest.approx(result) == expected_logprior

# Test __call__ method with chi=True and orbit_to_param translator
def test_gauss_prior_call_chi_with_orbit_to_param():
    # Use utility function to sample an orbit
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)  # Sample a LEO orbit
    distance = np.linalg.norm(orbit.r)  # Calculate distance from position vector

    mu = np.zeros(6)  # Example mean (all zeros)
    cinvcholfac = np.eye(6)  # Example inverse Cholesky factor (identity matrix)
    translator = lambda orbit: orbit_to_param(orbit, mode='rv')  # Use orbit_to_param as translator

    prior = GaussPrior(mu=mu, cinvcholfac=cinvcholfac, translator=translator)

    # Calculate expected values
    param = translator(orbit)
    chi0 = cinvcholfac.dot(param - mu)
    expected_chi = chi0

    # Call the method with chi=True and verify output
    result = prior(orbit, distance, chi=True)
    assert np.allclose(result, expected_chi)

def test_volume_distance_prior_logprior():
    """Test the log prior probability calculation."""
    # Create an instance of VolumeDistancePrior
    prior = VolumeDistancePrior(scale=RGEO)

    # Generate a sample LEO orbit
    t = 0  # Example time
    orbit = sample_LEO_orbit(t) 
    # Define a distance (e.g., a realistic distance in meters)
    distance = 7000e3  # 7000 km in meters

    # Calculate the log prior probability
    logprior = prior(orbit, distance, chi=False)

    # Expected results
    scaled_distance = distance / RGEO
    expected_logprior = 2 * np.log(scaled_distance) - scaled_distance - (2 * np.log(2) - 2 + 1e-9)

    # Assert that the calculated log prior matches the expected value
    assert np.isclose(logprior, expected_logprior), f"Expected {expected_logprior}, got {logprior}"

def test_volume_distance_prior_chi():
    """Test the chi calculation."""
    # Create an instance of VolumeDistancePrior
    prior = VolumeDistancePrior(scale=RGEO)

    # Generate a sample LEO orbit
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)  # Updated function name

    # Define a distance (e.g., a realistic distance in meters)
    distance = 7000e3  # 7000 km in meters

    # Calculate chi corresponding to the log prior probability
    chi_value = prior(orbit, distance, chi=True)

    # Expected results
    scaled_distance = distance / RGEO
    expected_logprior = 2 * np.log(scaled_distance) - scaled_distance - (2 * np.log(2) - 2 + 1e-9)
    expected_chi = np.sqrt(-2 * expected_logprior)

    # Assert that the calculated chi matches the expected value
    assert np.isclose(chi_value, expected_chi), f"Expected {expected_chi}, got {chi_value}"

def test_volume_distance_prior_zero_distance():
    """Test behavior when distance is zero."""
    # Create an instance of VolumeDistancePrior
    prior = VolumeDistancePrior(scale=RGEO)

    # Generate a sample LEO orbit
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)   # Updated function name

    # Define a distance of zero
    distance = 0.0

    # Calculate the log prior probability
    with pytest.raises(ValueError, match="divide by zero encountered in log"):
        prior(orbit, distance, chi=False)

def test_volume_distance_prior_negative_distance():
    """Test behavior when distance is negative."""
    # Create an instance of VolumeDistancePrior
    prior = VolumeDistancePrior(scale=RGEO)

    # Generate a sample LEO orbit
    t = 0  # Example time
    orbit = sample_LEO_orbit(t)  # Updated function name

    # Define a negative distance
    distance = -7000e3  # Negative distance

    # Calculate the log prior probability
    with pytest.raises(ValueError, match="invalid value encountered in log"):
        prior(orbit, distance, chi=False)


#######################
## TESTING FUNCTIONS ##
#######################

# Mock data for testing
def create_mock_arc(use_pm=False):
    """Create a mock observation arc for testing."""
    time = np.array([Time('2025-05-08T00:00:00'), Time('2025-05-08T01:00:00')])
    rStation_GCRF = np.array([[7000, 0, 0], [7000, 0, 0]]) * u.km
    vStation_GCRF = np.array([[0, 7.5, 0], [0, 7.5, 0]]) * u.km / u.s

    dtype = [('time', object), ('rStation_GCRF', object), ('vStation_GCRF', object)]
    if use_pm:
        pmra = np.array([0.1, 0.1])  # Example proper motion data
        dtype += [('pmra', float)]
        data = np.array(list(zip(time, rStation_GCRF, vStation_GCRF, pmra)), dtype=dtype)
    else:
        data = np.array(list(zip(time, rStation_GCRF, vStation_GCRF)), dtype=dtype)

    return data

# Test for mode='rv'
def test_make_param_guess_rv():
    rvguess = [7000, 0, 0, 0, 7.5, 0]  # Example state vector
    arc = create_mock_arc()  # Mock observation arc
    mode = 'rv'
    orbitattr = ['mass', 'area']

    result = make_param_guess(rvguess, arc, mode, orbitattr)

    # Expected results
    extraparam = [1, 0]  # Default values for 'mass' and 'area'
    epoch = (arc['time'][0].gps + (arc['time'][1].gps - arc['time'][0].gps) / 2)
    expected = rvguess + extraparam + [epoch]

    assert result == expected

# Test for mode='equinoctial'
def test_make_param_guess_equinoctial():
    rvguess = [7000, 0, 0, 0, 7.5, 0]  # Example state vector
    arc = create_mock_arc()  # Mock observation arc
    mode = 'equinoctial'
    orbitattr = ['cr', 'cd']

    result = make_param_guess(rvguess, arc, mode, orbitattr)

    # Expected results
    extraparam = [1, 1]  # Default values for 'cr' and 'cd'
    epoch = (arc['time'][0].gps + (arc['time'][1].gps - arc['time'][0].gps) / 2)

    # Convert rvguess to equinoctial elements
    orbInit = Orbit(r=rvguess[:3], v=rvguess[3:6], t=epoch)
    equinoctialElements = list(orbInit.equinoctialElements)

    expected = equinoctialElements + extraparam + [epoch]

    assert result == expected

# Test for mode='angle'
def test_make_param_guess_angle():
    rvguess = [7000, 0, 0, 0, 7.5, 0]  # Example state vector
    arc = create_mock_arc()  # Mock observation arc
    mode = 'angle'
    orbitattr = ['log10area']

    result = make_param_guess(rvguess, arc, mode, orbitattr)

    # Expected results
    extraparam = [-1]  # Default value for 'log10area'
    epoch = (arc['time'][0].gps + (arc['time'][1].gps - arc['time'][0].gps) / 2)

    # Compute initial observation position and velocity
    initObsPos = 0.5 * (arc[0]['rStation_GCRF'].to(u.m).value + arc[1]['rStation_GCRF'].to(u.m).value)
    initObsVel = 0.5 * (arc[0]['vStation_GCRF'].to(u.m/u.s).value + arc[1]['vStation_GCRF'].to(u.m/u.s).value)

    # Compute radial velocity observation rate
    radecrate = ssapy.compute.rvObsToRaDecRate(rvguess[:3], rvguess[3:], initObsPos, initObsVel)

    expected = list(radecrate) + extraparam + [epoch] + list(initObsPos) + list(initObsVel)

    assert np.allclose(result, expected)

# Test for ValueError on invalid mode
def test_make_param_guess_invalid_mode():
    rvguess = [7000, 0, 0, 0, 7.5, 0]  # Example state vector
    arc = create_mock_arc()  # Mock observation arc
    mode = 'invalid_mode'  # Invalid mode

    with pytest.raises(ValueError, match='unrecognized mode'):
        make_param_guess(rvguess, arc, mode)

def test_make_optimizer_valid_modes():
    """Test make_optimizer with valid modes."""
    param = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example parameter list

    # Test 'rv' mode
    optimizer = make_optimizer(mode='rv', param=param, lsq=False)
    assert optimizer == rvsampler.LMOptimizer, "Expected LMOptimizer for 'rv' mode"

    # Test 'angle' mode
    optimizer = make_optimizer(mode='angle', param=param, lsq=False)
    assert isinstance(optimizer, partial), "Expected a partial function for 'angle' mode"
    assert optimizer.func == rvsampler.LMOptimizerAngular, "Expected LMOptimizerAngular for 'angle' mode"
    assert optimizer.keywords['initObsPos'] == param[-6:-3], "Incorrect initObsPos for 'angle' mode"
    assert optimizer.keywords['initObsVel'] == param[-3:], "Incorrect initObsVel for 'angle' mode"

    # Test 'equinoctial' mode
    optimizer = make_optimizer(mode='equinoctial', param=param, lsq=False)
    assert optimizer == rvsampler.EquinoctialLMOptimizer, "Expected EquinoctialLMOptimizer for 'equinoctial' mode"

def test_make_optimizer_valid_modes_lsq():
    """Test make_optimizer with valid modes and lsq=True."""
    param = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example parameter list

    # Test 'rv' mode
    optimizer = make_optimizer(mode='rv', param=param, lsq=True)
    assert isinstance(optimizer, partial), "Expected a partial function for 'rv' mode with lsq=True"
    assert optimizer.func == rvsampler.LeastSquaresOptimizer, "Expected LeastSquaresOptimizer for 'rv' mode with lsq=True"
    assert optimizer.keywords['translatorcls'] == rvsampler.ParamOrbitRV, "Expected ParamOrbitRV for translatorcls"

    # Test 'angle' mode
    optimizer = make_optimizer(mode='angle', param=param, lsq=True)
    assert isinstance(optimizer, partial), "Expected a partial function for 'angle' mode with lsq=True"
    assert optimizer.func == rvsampler.LeastSquaresOptimizer, "Expected LeastSquaresOptimizer for 'angle' mode with lsq=True"
    assert optimizer.keywords['translatorcls'] == rvsampler.ParamOrbitAngle, "Expected ParamOrbitAngle for translatorcls"

    # Test 'equinoctial' mode
    optimizer = make_optimizer(mode='equinoctial', param=param, lsq=True)
    assert isinstance(optimizer, partial), "Expected a partial function for 'equinoctial' mode with lsq=True"
    assert optimizer.func == rvsampler.LeastSquaresOptimizer, "Expected LeastSquaresOptimizer for 'equinoctial' mode with lsq=True"
    assert optimizer.keywords['translatorcls'] == rvsampler.ParamOrbitEquinoctial, "Expected ParamOrbitEquinoctial for translatorcls"

def test_make_optimizer_invalid_mode():
    """Test make_optimizer with an invalid mode."""
    param = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example parameter list

    with pytest.raises(ValueError, match="unknown mode"):
        make_optimizer(mode='invalid_mode', param=param, lsq=False)

def test_make_optimizer_angle_mode_param_length():
    """Test make_optimizer with 'angle' mode and insufficient param length."""
    param = [1, 2, 3]  # Insufficient parameter list for 'angle' mode

    with pytest.raises(IndexError):
        make_optimizer(mode='angle', param=param, lsq=False)

def test_make_optimizer_partial_function():
    """Test that make_optimizer returns a partial function for 'angle' mode."""
    param = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example parameter list

    optimizer = make_optimizer(mode='angle', param=param, lsq=False)
    assert isinstance(optimizer, partial), "Expected a partial function for 'angle' mode"
    assert optimizer.func == rvsampler.LMOptimizerAngular, "Expected LMOptimizerAngular for 'angle' mode"
    assert optimizer.keywords['initObsPos'] == param[-6:-3], "Incorrect initObsPos for 'angle' mode"
    assert optimizer.keywords['initObsVel'] == param[-3:], "Incorrect initObsVel for 'angle' mode"


@pytest.fixture
def sample_arc():
    """Fixture to create a sample arc for testing."""
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
    """Fixture to create a sample guess for testing."""
    return np.array([1, 2, 3, 4, 5, 6, 123456789])  # Example guess with epoch in GPS time

@pytest.fixture
def sample_gaussian_prior():
    """Fixture to create sample Gaussian prior parameters."""
    mu = np.array([1, 2, 3, 4, 5, 6, 123456789])  # Example mean
    cinvcholfac = np.eye(6)  # Example inverse Cholesky factor
    return mu, cinvcholfac

def test_fit_arc_blind_via_track_basic(sample_arc=sample_arc(), sample_propagator=propagator.KeplerianPropagator()):
    """Test basic functionality of fit_arc_blind_via_track."""
    tracklist = fit_arc_blind_via_track(
        sample_arc, propagator=sample_propagator, verbose=False,
        reset_if_too_uncertain=False, mode='rv', priors=None,
        order='forward', approximate=False, orbitattr=None, factor=1.0
    )

    # Assertions
    assert len(tracklist) > 0, "Tracklist should not be empty."
    assert isinstance(tracklist[0], Track), "Tracklist should contain Track instances."
    assert tracklist[0].satIDs[0] == 0, "First track should start with the first satellite ID."

def test_fit_arc_blind_via_track_reset(sample_arc=sample_arc(), sample_propagator=propagator.KeplerianPropagator()):
    """Test reset behavior when uncertainty grows too large."""
    tracklist = fit_arc_blind_via_track(
        sample_arc, propagator=sample_propagator, verbose=False,
        reset_if_too_uncertain=True, mode='rv', priors=None,
        order='forward', approximate=False, orbitattr=None, factor=1.0
    )

    # Assertions
    assert len(tracklist) > 0, "Tracklist should not be empty."
    assert any(track.chi2 == 0 for track in tracklist), "Reset should occur if uncertainty grows too large."

def test_fit_arc_blind_via_track_verbose(sample_arc=sample_arc(), sample_propagator=propagator.KeplerianPropagator(), capsys):
    """Test verbose output of fit_arc_blind_via_track."""
    fit_arc_blind_via_track(
        sample_arc, propagator=sample_propagator, verbose=True,
        reset_if_too_uncertain=False, mode='rv', priors=None,
        order='forward', approximate=False, orbitattr=None, factor=1.0
    )

    assert "resetting" in captured.out or "dt:" in captured.out, "Verbose output should contain fitting details."

def test_fit_arc_blind_valid(sample_arc=sample_arc()):
    """Test fit_arc_blind with valid inputs."""
    arc = sample_arc
    chi2, params, result = fit_arc_blind(arc, mode='rv', verbose=False, factor=2)
    assert isinstance(chi2, float), "Expected chi2 to be a float"
    assert isinstance(params, np.ndarray), "Expected params to be a numpy array"
    assert hasattr(result, 'residual'), "Expected result to have residual attribute"

def test_fit_arc_blind_empty_arc():
    """Test fit_arc_blind with an empty arc."""
    arc = np.array([], dtype=[('satID', 'int')])
    with pytest.raises(ValueError, match="len\(arc\) must be > 0"):
        fit_arc_blind(arc, mode='rv')

def test_fit_arc_blind_invalid_factor(sample_arc=sample_arc()):
    """Test fit_arc_blind with an invalid factor."""
    arc = sample_arc
    with pytest.raises(AssertionError, match="Geometric factor must be greater than or equal to 1"):
        fit_arc_blind(arc, mode='rv', factor=2)

def test_fit_arc_valid(sample_arc(), sample_guess()):
    """Test fit_arc with valid inputs."""
    arc = sample_arc
    guess = sample_guess
    chi2, params, result = fit_arc(arc, guess, mode='rv', verbose=False)
    assert isinstance(chi2, float), "Expected chi2 to be a float"
    assert isinstance(params, np.ndarray), "Expected params to be a numpy array"
    assert hasattr(result, 'residual'), "Expected result to have residual attribute"

def test_fit_arc_invalid_epoch(sample_arc(), sample_guess()):
    """Test fit_arc with an invalid epoch in the guess."""
    arc = sample_arc
    guess = np.array([1, 2, 3, 4, 5, 6, 0])  # Invalid epoch
    with pytest.raises(ValueError, match="inconsistent epoch and guess"):
        fit_arc(arc, guess, mode='rv', verbose=False)

def test_fit_arc_with_gaussian_prior_valid(sample_arc(), sample_gaussian_prior()):
    """Test fit_arc_with_gaussian_prior with valid inputs."""
    arc = sample_arc
    mu, cinvcholfac = sample_gaussian_prior
    chi2, params, result = fit_arc_with_gaussian_prior(arc, mu, cinvcholfac, mode='rv', verbose=False)
    assert isinstance(chi2, float), "Expected chi2 to be a float"
    assert isinstance(params, np.ndarray), "Expected params to be a numpy array"
    assert hasattr(result, 'residual'), "Expected result to have residual attribute"

def test_fit_arc_with_gaussian_prior_invalid_mu_length(sample_arc(), sample_gaussian_prior()):
    """Test fit_arc_with_gaussian_prior with an invalid mu length."""
    arc = sample_arc
    mu, cinvcholfac = sample_gaussian_prior
    mu = np.array([1, 2, 3])  # Invalid length
    with pytest.raises(ValueError, match="mu must have length matching number of parameters"):
        fit_arc_with_gaussian_prior(arc, mu, cinvcholfac, mode='rv', verbose=False)

def test_fit_arc_with_gaussian_prior_invalid_cinvcholfac(sample_arc(), sample_gaussian_prior()):
    """Test fit_arc_with_gaussian_prior with an invalid cinvcholfac."""
    arc = sample_arc
    mu, cinvcholfac = sample_gaussian_prior
    cinvcholfac = np.array([[1, 2], [3, 4]])  # Invalid shape
    with pytest.raises(ValueError, match="cinvcholfac must be square matrix"):
        fit_arc_with_gaussian_prior(arc, mu, cinvcholfac, mode='rv', verbose=False)

@pytest.fixture
def sample_data():
    """Fixture to create a sample dataset with 'satID' column."""
    dtype = [('satID', 'int'), ('ra', 'float'), ('dec', 'float')]
    data = np.array([
        (1, 10.0, 20.0),
        (2, 15.0, 25.0),
        (3, 20.0, 30.0),
        (4, 25.0, 35.0),
        (5, 30.0, 40.0)
    ], dtype=dtype)
    return data

def test_data_for_satellite_valid(sample_data):
    """Test data_for_satellite with valid satIDList."""
    data = sample_data
    satIDList = [2, 4]
    result = data_for_satellite(data, satIDList)
    assert len(result) == 2, "Expected 2 matching rows"
    assert result['satID'][0] == 2, "Expected first row to have satID 2"
    assert result['satID'][1] == 4, "Expected second row to have satID 4"

def test_data_for_satellite_single_satID(sample_data):
    """Test data_for_satellite with a single satID."""
    data = sample_data
    satIDList = [3]
    result = data_for_satellite(data, satIDList)
    assert len(result) == 1, "Expected 1 matching row"
    assert result['satID'][0] == 3, "Expected row to have satID 3"

def test_data_for_satellite_empty_satIDList(sample_data):
    """Test data_for_satellite with an empty satIDList."""
    data = sample_data
    satIDList = []
    result = data_for_satellite(data, satIDList)
    assert len(result) == 0, "Expected no matching rows for empty satIDList"

def test_data_for_satellite_invalid_satID(sample_data):
    """Test data_for_satellite with a satID not in the data."""
    data = sample_data
    satIDList = [999]  # Nonexistent satID
    with pytest.raises(ValueError, match="satID 999 not found!"):
        data_for_satellite(data, satIDList)

def test_data_for_satellite_ignore_negative_satID(sample_data):
    """Test data_for_satellite with a negative satID (-1)."""
    data = sample_data
    satIDList = [-1, 3]
    result = data_for_satellite(data, satIDList)
    assert len(result) == 1, "Expected 1 matching row (ignoring -1)"
    assert result['satID'][0] == 3, "Expected row to have satID 3"

def test_data_for_satellite_multiple_matches(sample_data):
    """Test data_for_satellite with multiple matches."""
    data = sample_data
    satIDList = [1, 2, 3, 4, 5]
    result = data_for_satellite(data, satIDList)
    assert len(result) == len(data), "Expected all rows to match"
    assert np.array_equal(result['satID'], data['satID']), "Expected all satIDs to match"

def test_data_for_satellite_no_matches(sample_data):
    """Test data_for_satellite with no matching satIDs."""
    data = sample_data
    satIDList = [-1]  # Only -1, which should be ignored
    result = data_for_satellite(data, satIDList)
    assert len(result) == 0, "Expected no matching rows when satIDList contains only -1"

def test_wrap_angle_difference_single_value():
    # Test with a single value and default center
    assert wrap_angle_difference(3, 2 * math.pi) == pytest.approx(3 - 2 * math.pi, rel=1e-6)
    
    # Test with a single value and custom center
    assert wrap_angle_difference(3, 2 * math.pi, center=0.25) == pytest.approx((3 + 0.25 * 2 * math.pi) % (2 * math.pi) - 0.25 * 2 * math.pi, rel=1e-6)
    
    # Test with value at the boundary of the wrap range
    assert wrap_angle_difference(2 * math.pi, 2 * math.pi) == pytest.approx(0, rel=1e-6)
    assert wrap_angle_difference(-2 * math.pi, 2 * math.pi) == pytest.approx(0, rel=1e-6)

def test_wrap_angle_difference_array():
    # Test with an array of values
    input_array = np.array([3, -3, 2 * math.pi, -2 * math.pi])
    expected_output = np.array([(3 + math.pi) % (2 * math.pi) - math.pi,
                                (-3 + math.pi) % (2 * math.pi) - math.pi,
                                (2 * math.pi + math.pi) % (2 * math.pi) - math.pi,
                                (-2 * math.pi + math.pi) % (2 * math.pi) - math.pi])
    np.testing.assert_allclose(wrap_angle_difference(input_array, 2 * math.pi), expected_output, rtol=1e-6)

def test_wrap_angle_difference_edge_cases():
    # Test with zero angle difference
    assert wrap_angle_difference(0, 2 * math.pi) == pytest.approx(0, rel=1e-6)
    
    # Test with wrap range of zero (should return the input as-is)
    assert wrap_angle_difference(3, 0) == pytest.approx(3, rel=1e-6)
    
    # Test with negative wrap range (invalid case, should raise an error)
    with pytest.raises(ValueError):
        wrap_angle_difference(3, -2 * math.pi)

def test_wrap_angle_difference_custom_center():
    # Test with a custom center value
    assert wrap_angle_difference(3, 360, center=0.5) == pytest.approx((3 + 0.5 * 360) % 360 - 0.5 * 360, rel=1e-6)
    assert wrap_angle_difference(3, 360, center=0.25) == pytest.approx((3 + 0.25 * 360) % 360 - 0.25 * 360, rel=1e-6)
    
    # Test with center at 0 (no offset)
    assert wrap_angle_difference(3, 360, center=0) == pytest.approx(3 % 360, rel=1e-6)

def test_wrap_angle_difference_large_values():
    # Test with large values for angle difference
    assert wrap_angle_difference(1000, 360) == pytest.approx((1000 + 0.5 * 360) % 360 - 0.5 * 360, rel=1e-6)
    assert wrap_angle_difference(-1000, 360) == pytest.approx((-1000 + 0.5 * 360) % 360 - 0.5 * 360, rel=1e-6)

def test_wrap_angle_difference_negative_center():
    # Test with a negative center value
    assert wrap_angle_difference(3, 360, center=-0.5) == pytest.approx((3 - 0.5 * 360) % 360 - (-0.5 * 360), rel=1e-6)

def test_radeczn_GEO_orbit():
    # Create a sample GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    
    # Create a mock observation arc
    arc = {
        'time': np.array([1000, 2000]),
        'rStation_GCRF': np.array([[6371e3, 0, 0], [6371e3, 0, 0]]),  # Station at Earth's surface
        'vStation_GCRF': np.array([[0, 0, 0], [0, 0, 0]])  # Station velocity
    }
    
    rr, dd, zz, pmrr, pmdd, dzzdt, nwrap = radeczn(orbit, arc)
    
    # Verify outputs
    assert rr.shape == (2,)
    assert dd.shape == (2,)
    assert zz.shape == (2,)
    assert pmrr.shape == (2,)
    assert pmdd.shape == (2,)
    assert dzzdt.shape == (2,)
    assert nwrap.shape == (2,)
    
    # Check angular outputs
    checkAngle(rr[0], rr[1], atol=1e-6)
    checkAngle(dd[0], dd[1], atol=1e-6)

def test_radeczn_LEO_orbit():
    # Create a sample LEO orbit
    orbit = sample_LEO_orbit(t=1000)
    
    # Create a mock observation arc
    arc = {
        'time': np.array([1000, 2000]),
        'rStation_GCRF': np.array([[6371e3, 0, 0], [6371e3, 0, 0]]),  # Station at Earth's surface
        'vStation_GCRF': np.array([[0, 0, 0], [0, 0, 0]])  # Station velocity
    }
    
    rr, dd, zz, pmrr, pmdd, dzzdt, nwrap = radeczn(orbit, arc)
    
    # Verify outputs
    assert rr.shape == (2,)
    assert dd.shape == (2,)
    assert zz.shape == (2,)
    assert pmrr.shape == (2,)
    assert pmdd.shape == (2,)
    assert dzzdt.shape == (2,)
    assert nwrap.shape == (2,)
    
    # Check angular outputs
    checkAngle(rr[0], rr[1], atol=1e-6)
    checkAngle(dd[0], dd[1], atol=1e-6)

def test_radeczn_multiple_orbits():
    # Create multiple sample orbits
    orbit1 = sample_GEO_orbit(t=1000)
    orbit2 = sample_LEO_orbit(t=1000)
    orbit = [orbit1, orbit2]
    
    # Create a mock observation arc
    arc = {
        'time': np.array([1000, 2000]),
        'rStation_GCRF': np.array([[6371e3, 0, 0], [6371e3, 0, 0]]),  # Station at Earth's surface
        'vStation_GCRF': np.array([[0, 0, 0], [0, 0, 0]])  # Station velocity
    }
    
    rr, dd, zz, pmrr, pmdd, dzzdt, nwrap = radeczn(orbit, arc)
    
    # Verify outputs
    assert rr.shape == (2,)
    assert dd.shape == (2,)
    assert zz.shape == (2,)
    assert pmrr.shape == (2,)
    assert pmdd.shape == (2,)
    assert dzzdt.shape == (2,)
    assert nwrap.shape == (2,)
    
    # Check angular outputs
    checkSphere(rr, dd, rr, dd, atol=1e-6)

def test_radeczn_edge_case_empty_arc():
    # Create a sample GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    
    # Create an empty observation arc
    arc = {
        'time': np.array([]),
        'rStation_GCRF': np.array([]),
        'vStation_GCRF': np.array([])
    }
    
    rr, dd, zz, pmrr, pmdd, dzzdt, nwrap = radeczn(orbit, arc)
    
    # Verify outputs are empty
    assert rr.size == 0
    assert dd.size == 0
    assert zz.size == 0
    assert pmrr.size == 0
    assert pmdd.size == 0
    assert dzzdt.size == 0
    assert nwrap.size == 0

def test_radeczn_edge_case_single_observation():
    # Create a sample GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    
    # Create a single observation arc
    arc = {
        'time': np.array([1000]),
        'rStation_GCRF': np.array([[6371e3, 0, 0]]),  # Station at Earth's surface
        'vStation_GCRF': np.array([[0, 0, 0]])  # Station velocity
    }
    
    rr, dd, zz, pmrr, pmdd, dzzdt, nwrap = radeczn(orbit, arc)
    
    # Verify outputs
    assert rr.shape == (1,)
    assert dd.shape == (1,)
    assert zz.shape == (1,)
    assert pmrr.shape == (1,)
    assert pmdd.shape == (1,)
    assert dzzdt.shape == (1,)
    assert nwrap.shape == (1,)

def test_param_to_orbit_mode_rv():
    # Generate parameters for a GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    param = np.hstack([orbit.r, orbit.v, [1000]])  # Combine position, velocity, and time
    
    # Convert parameters to orbit
    converted_orbit = param_to_orbit(param, mode='rv')
    
    # Verify the converted orbit matches the original
    np.testing.assert_allclose(converted_orbit.r, orbit.r, atol=1e-6)
    np.testing.assert_allclose(converted_orbit.v, orbit.v, atol=1e-6)
    assert converted_orbit.t.gps == orbit.t.gps

def test_param_to_orbit_mode_equinoctial():
    # Generate parameters for a GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    aa, h, k, p, q, l = orbit.equinoctialElements
    param = np.hstack([aa, h, k, p, q, l, [1000]])  # Combine equinoctial elements and time
    
    # Convert parameters to orbit
    converted_orbit = param_to_orbit(param, mode='equinoctial')
    
    # Verify the converted orbit matches the original
    np.testing.assert_allclose(converted_orbit.equinoctialElements, orbit.equinoctialElements, atol=1e-6)
    assert converted_orbit.t.gps == orbit.t.gps

def test_param_to_orbit_mode_angle():
    # Generate parameters for a GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    rStation = np.array([6371e3, 0, 0])  # Station at Earth's surface
    vStation = np.array([0, 0, 0])  # Station velocity
    rr, dd, zz, pmrr, pmdd, dzzdt = ssapy.compute.rvObsToRaDecRate(
        orbit.r, orbit.v, rStation, vStation)
    param = np.hstack([rr, dd, zz, pmrr, pmdd, dzzdt, rStation, vStation, [1000]])  # Combine angle parameters
    
    # Convert parameters to orbit
    converted_orbit = param_to_orbit(param, mode='angle')
    
    # Verify the converted orbit matches the original
    np.testing.assert_allclose(converted_orbit.r, orbit.r, atol=1e-6)
    np.testing.assert_allclose(converted_orbit.v, orbit.v, atol=1e-6)
    assert converted_orbit.t.gps == orbit.t.gps

def test_orbit_to_param_mode_rv():
    # Generate a GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    
    # Convert orbit to parameters
    param = orbit_to_param(orbit, mode='rv')
    
    # Verify the converted parameters match the original orbit
    np.testing.assert_allclose(param[:3], orbit.r, atol=1e-6)  # Position
    np.testing.assert_allclose(param[3:6], orbit.v, atol=1e-6)  # Velocity
    assert param[-1] == orbit.t.gps

def test_orbit_to_param_mode_equinoctial():
    # Generate a GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    
    # Convert orbit to parameters
    param = orbit_to_param(orbit, mode='equinoctial')
    
    # Verify the converted parameters match the original orbit
    eqel = orbit.equinoctialElements
    np.testing.assert_allclose(param[:6], eqel, atol=1e-6)  # Equinoctial elements
    assert param[-1] == orbit.t.gps

def test_orbit_to_param_mode_angle():
    # Generate a GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    rStation = np.array([6371e3, 0, 0])  # Station at Earth's surface
    vStation = np.array([0, 0, 0])  # Station velocity
    
    # Convert orbit to parameters
    param = orbit_to_param(orbit, mode='angle', rStation=rStation, vStation=vStation)
    
    # Verify the converted parameters match the original orbit
    rr, dd, zz, pmrr, pmdd, dzzdt = ssapy.compute.rvObsToRaDecRate(
        orbit.r, orbit.v, rStation, vStation)
    np.testing.assert_allclose(param[:6], [rr, dd, zz, pmrr, pmdd, dzzdt], atol=1e-6)
    np.testing.assert_allclose(param[-6:-3], rStation, atol=1e-6)  # Station position
    np.testing.assert_allclose(param[-3:], vStation, atol=1e-6)  # Station velocity
    assert param[-7] == orbit.t.gps

def test_round_trip_conversion():
    # Generate a GEO orbit
    orbit = sample_GEO_orbit(t=1000)
    
    # Convert orbit to parameters and back to orbit
    param = orbit_to_param(orbit, mode='rv')
    converted_orbit = param_to_orbit(param, mode='rv')
    
    # Verify the round-trip conversion matches the original orbit
    np.testing.assert_allclose(converted_orbit.r, orbit.r, atol=1e-6)
    np.testing.assert_allclose(converted_orbit.v, orbit.v, atol=1e-6)
    assert converted_orbit.t.gps == orbit.t.gps

class TestTrackBase(unittest.TestCase):
    def setUp(self):
        self.satIDs = [1, 2, 3]
        self.data = sample_data()
        self.volume = 1.0
        self.mode = 'rv'
        self.propagator = propagator.KeplerianPropagator()
        self.track = TrackBase(self.satIDs, self.data, self.volume, self.mode, self.propagator)

    def test_lnprob(self):
        # Test the lnprob property
        self.track.covar = np.eye(6)  # Mock covariance matrix
        self.track.chi2 = 10  # Mock chi2 value
        result = self.track.lnprob
        self.assertTrue(np.isfinite(result), "lnprob should return a finite value.")

    def test_predict(self):
        # Test the predict method
        arc0 = {'time': np.array([2451545.0]), 'rStation_GCRF': np.zeros(3), 'vStation_GCRF': np.zeros(3)}
        mean, covar = self.track.predict(arc0)
        self.assertEqual(mean.shape[0], 4, "Mean should have 4 parameters.")
        self.assertEqual(covar.shape[0], 4, "Covariance matrix should have 4x4 dimensions.")

    def test_gate(self):
        # Test the gate method
        arc = sample_arc()
        chi2 = self.track.gate(arc)
        self.assertTrue(chi2 >= 0, "Chi2 should be non-negative.")

    def test_propagaterdz(self):
        # Test the propagaterdz method
        param = np.zeros(6)  # Mock parameters
        arc0 = {'time': np.array([2451545.0]), 'rStation_GCRF': np.zeros(3), 'vStation_GCRF': np.zeros(3)}
        result = self.track.propagaterdz(param, arc0)
        self.assertEqual(len(result), 4, "Result should contain 4 elements.")

    def test_update(self):
        # Test the update method
        time = 2451545.0
        result = self.track.update(time)
        self.assertIsNone(result, "Update should return None.")

    def test_repr(self):
        # Test the __repr__ method
        self.track.chi2 = 10
        self.track.lnprob = -5
        result = repr(self.track)
        self.assertIn("Track, chi2:", result, "__repr__ should include 'Track, chi2:'")
        self.assertIn("lnprob:", result, "__repr__ should include 'lnprob:'")

def test_track_initialization_with_guess():
    # Create sample data
    satIDs = [1, 2]
    data = {
        'time': np.array([1000, 2000]),
        'rStation_GCRF': np.array([[6371e3, 0, 0], [6371e3, 0, 0]]),
        'vStation_GCRF': np.array([[0, 0, 0], [0, 0, 0]])
    }
    guess = np.array([6371e3, 0, 0, 0, 0, 0])  # Initial guess for orbit parameters
    mode = 'rv'
    priors = None
    propagator_instance = propagator.KeplerianPropagator()

    # Initialize Track with a guess
    track = Track(satIDs, data, guess=guess, mode=mode, priors=priors, propagator=propagator_instance)

    # Verify attributes
    assert track.satIDs == satIDs
    assert track.data == data
    assert track.mode == mode
    assert track.propagator == propagator_instance
    assert track.param is not None
    assert track.chi2 is not None
    assert track.success is not None

def test_track_initialization_blind():
    # Create sample data
    satIDs = [1, 2]
    data = {
        'time': np.array([1000, 2000]),
        'rStation_GCRF': np.array([[6371e3, 0, 0], [6371e3, 0, 0]]),
        'vStation_GCRF': np.array([[0, 0, 0], [0, 0, 0]])
    }
    mode = 'rv'
    priors = None
    propagator_instance = propagator.KeplerianPropagator()

    # Initialize Track without a guess (blind fitting)
    track = Track(satIDs, data, mode=mode, priors=priors, propagator=propagator_instance)

    # Verify attributes
    assert track.satIDs == satIDs
    assert track.data == data
    assert track.mode == mode
    assert track.propagator == propagator_instance
    assert track.param is not None
    assert track.chi2 is not None
    assert track.success is not None

def test_track_gaussian_approximation():
    # Create sample data
    satIDs = [1, 2, 3, 4]
    data = {
        'time': np.array([1000, 2000, 3000, 4000]),
        'rStation_GCRF': np.array([[6371e3, 0, 0]] * 4),
        'vStation_GCRF': np.array([[0, 0, 0]] * 4)
    }
    mode = 'rv'
    priors = None
    propagator_instance = propagator.KeplerianPropagator()

    # Initialize Track
    track = Track(satIDs, data, mode=mode, priors=priors, propagator=propagator_instance)

    # Get Gaussian approximation
    gauss_track = track.gaussian_approximation()

    # Verify Gaussian approximation is a TrackGauss instance
    assert isinstance(gauss_track, TrackGauss)

def test_track_addto():
    # Create sample data
    satIDs = [1, 2]
    data = {
        'time': np.array([1000, 2000]),
        'rStation_GCRF': np.array([[6371e3, 0, 0], [6371e3, 0, 0]]),
        'vStation_GCRF': np.array([[0, 0, 0], [0, 0, 0]])
    }
    mode = 'rv'
    priors = None
    propagator_instance = propagator.KeplerianPropagator()

    # Initialize Track
    track = Track(satIDs, data, mode=mode, priors=priors, propagator=propagator_instance)

    # Add a new satellite ID
    new_track = track.addto(3)

    # Verify new track includes the additional satellite ID
    assert new_track.satIDs == [1, 2, 3]

class TestTrackGauss(unittest.TestCase):
    def setUp(self):
        self.satIDs = [1, 2, 3]
        self.data = sample_data()
        self.param = np.zeros(6)  # Mock parameters
        self.covar = np.eye(6)  # Mock covariance matrix
        self.chi2 = 10.0  # Mock chi2 value
        self.mode = 'rv'
        self.propagator = propagator.KeplerianPropagator()
        self.track = TrackGauss(self.satIDs, self.data, self.param, self.covar, self.chi2, self.mode, self.propagator)

    def test_gaussian_approximation(self):
        # Test the gaussian_approximation method
        result = self.track.gaussian_approximation(self.propagator)
        self.assertEqual(result, self.track, "Gaussian approximation should return self.")

    def test_update(self):
        # Test the update method
        t = 2  # Mock time
        t.to = 3 # Mock time difference
        self.track.update(t)
        self.assertTrue(np.array_equal(self.track.param, self.param), "Parameters should be updated.")

    def test_at(self):
        # Test the at method
        t = 0  # Mock time
        newtrack = self.track.at(t)
        self.assertIsInstance(newtrack, TrackGauss, "at() should return a TrackGauss instance.")
        self.assertNotEqual(newtrack.param, self.track.param, "Parameters should differ after update.")

    def test_addto(self):
        # Test the addto method
        satid = 4
        newtrack = self.track.addto(satid)
        self.assertIsInstance(newtrack, TrackGauss, "addto() should return a TrackGauss instance.")
        self.assertIn(satid, newtrack.satIDs, "New satellite ID should be added to satIDs.")

@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    dtype = [('satID', 'int'), ('time', 'float'), ('rStation_GCRF', 'float', (3,)), ('vStation_GCRF', 'float', (3,))]
    data = np.zeros(10, dtype=dtype)
    data['satID'] = [2, 3, 1, 5, 4, 3, 2, 1, 5, 4]  # Repeated satellite IDs
    data['time'] = np.linspace(0, 100, 10)  # Ordered times
    data['rStation_GCRF'] = np.random.rand(10, 3)
    data['vStation_GCRF'] = np.random.rand(10, 3)
    return data

@pytest.fixture
def sample_truth():
    """Fixture to create sample truth data."""
    return {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

@pytest.fixture
def sample_hypotheses():
    """Fixture to create sample hypotheses."""
    return [Hypothesis([], nsat=1000)]

@pytest.fixture
def sample_propagator():
    """Fixture to create a sample propagator."""
    class MockPropagator:
        def __init__(self):
            pass
    return MockPropagator()

@pytest.fixture
def mht_instance(sample_data, sample_truth, sample_hypotheses, sample_propagator):
    """Fixture to create an MHT instance."""
    return MHT(data=sample_data, nsat=1000, truth=sample_truth, hypotheses=sample_hypotheses, propagator=sample_propagator)

def test_mht_initialization(mht_instance, sample_data, sample_truth, sample_hypotheses):
    """Test initialization of MHT class."""
    assert mht_instance.data.shape == sample_data.shape, "Data should be copied correctly."
    assert mht_instance.nsat == 1000, "Number of satellites should be initialized correctly."
    assert mht_instance.truth == sample_truth, "Truth data should be initialized correctly."
    assert mht_instance.hypotheses == sample_hypotheses, "Hypotheses should be initialized correctly."

def test_mht_run(mht_instance):
    """Test the run method."""
    mht_instance.run(verbose=False, order='forward')
    assert len(mht_instance.track2hyp) > 0, "Track-to-hypothesis mapping should be populated after running."

def test_mht_add_tracklet(mht_instance):
    """Test the add_tracklet method."""
    satid = 1  # Use a sample satellite ID
    mht_instance.add_tracklet(satid)
    assert len(mht_instance.track2hyp) > 0, "Track-to-hypothesis mapping should be updated after adding a tracklet."

def test_mht_prune_tracks(mht_instance):
    """Test the prune_tracks method."""
    satid = 1  # Use a sample satellite ID
    keephyp = mht_instance.prune_tracks(satid)
    assert isinstance(keephyp, np.ndarray), "Prune tracks should return a boolean array."

def test_mht_prune_stale_hypotheses(mht_instance):
    """Test the prune_stale_hypotheses method."""
    newdeadtracks = []  # No dead tracks initially
    keephyp = mht_instance.prune_stale_hypotheses(newdeadtracks)
    assert isinstance(keephyp, np.ndarray), "Prune stale hypotheses should return a boolean array."

def test_mht_prune(mht_instance):
    """Test the prune method."""
    satid = 1  # Use a sample satellite ID
    mht_instance.prune(satid, nkeepmax=5, pkeep=1e-9, keeponlytrue=False, nconfirm=6)
    assert len(mht_instance.hypotheses) > 0, "Hypotheses should be pruned correctly."

def test_summarize_tracklets():
    # Create mock data
    data = np.array([
        {'satID': 1, 'time': Time(1000, format='gps'), 'ra': 10*u.deg, 'dec': 20*u.deg, 
         'rStation_GCRF': np.array([6371e3, 0, 0])*u.m, 'vStation_GCRF': np.array([0, 0, 0])*u.m/u.s},
        {'satID': 1, 'time': Time(2000, format='gps'), 'ra': 11*u.deg, 'dec': 21*u.deg, 
         'rStation_GCRF': np.array([6371e3, 0, 0])*u.m, 'vStation_GCRF': np.array([0, 0, 0])*u.m/u.s},
        {'satID': 2, 'time': Time(3000, format='gps'), 'ra': 12*u.deg, 'dec': 22*u.deg, 
         'rStation_GCRF': np.array([6371e3, 0, 0])*u.m, 'vStation_GCRF': np.array([0, 0, 0])*u.m/u.s},
    ], dtype=object)

    # Run function
    summarized_data = summarize_tracklets(data, posuncfloor=0.1*u.deg, pmuncfloor=0.01*u.deg/u.s)

    # Verify new fields are added
    assert 'dra' in summarized_data.dtype.names
    assert 'ddec' in summarized_data.dtype.names
    assert 'pmra' in summarized_data.dtype.names
    assert 'pmdec' in summarized_data.dtype.names
    assert 'dpmra' in summarized_data.dtype.names
    assert 'dpmdec' in summarized_data.dtype.names
    assert 't_baseline' in summarized_data.dtype.names

    # Verify uncertainty floors are applied
    assert np.all(summarized_data['dra'] >= 0.1*u.deg)
    assert np.all(summarized_data['dpmra'] >= 0.01*u.deg/u.s)

def test_summarize_tracklet():
    # Create mock short arc data
    arc = np.array([
        {'time': Time(1000, format='gps'), 'ra': 10*u.deg, 'dec': 20*u.deg, 'sigma': 0.1*u.deg},
        {'time': Time(2000, format='gps'), 'ra': 11*u.deg, 'dec': 21*u.deg, 'sigma': 0.1*u.deg},
        {'time': Time(3000, format='gps'), 'ra': 12*u.deg, 'dec': 22*u.deg, 'sigma': 0.1*u.deg},
    ], dtype=object)

    # Run function
    mean_pos, mean_unc, pm, pm_unc = summarize_tracklet(arc)

    # Verify outputs
    assert mean_pos[0].unit == u.rad
    assert mean_pos[1].unit == u.rad
    assert mean_unc[0].unit == u.rad
    assert mean_unc[1].unit == u.rad
    assert pm[0].unit == u.rad/u.s
    assert pm[1].unit == u.rad/u.s
    assert pm_unc[0].unit == u.rad/u.s
    assert pm_unc[1].unit == u.rad/u.s

    # Verify proper motion is calculated
    assert pm[0] != 0*u.rad/u.s
    assert pm[1] != 0*u.rad/u.s

def test_iterate_mht():
    # Create mock data
    data = np.array([
        {'satID': 1, 'time': Time(1000, format='gps'), 'ra': 10*u.deg, 'dec': 20*u.deg, 
         'rStation_GCRF': np.array([6371e3, 0, 0])*u.m, 'vStation_GCRF': np.array([0, 0, 0])*u.m/u.s},
        {'satID': 2, 'time': Time(2000, format='gps'), 'ra': 11*u.deg, 'dec': 21*u.deg, 
         'rStation_GCRF': np.array([6371e3, 0, 0])*u.m, 'vStation_GCRF': np.array([0, 0, 0])*u.m/u.s},
        {'satID': 3, 'time': Time(3000, format='gps'), 'ra': 12*u.deg, 'dec': 22*u.deg, 
         'rStation_GCRF': np.array([6371e3, 0, 0])*u.m, 'vStation_GCRF': np.array([0, 0, 0])*u.m/u.s},
    ], dtype=object)

    # Create mock MHT object
    oldmht = MHT(data, nsat=1000, hypotheses=[Hypothesis([], nsat=1000)], mode='rv')

    # Run function
    newmht = iterate_mht(data, oldmht, nminlength=2, trimends=1)

    # Verify new MHT object is created
    assert isinstance(newmht, MHT)

    # Verify new hypotheses are generated
    assert len(newmht.hypotheses) > 0

    # Verify tracks are refined
    for hyp in newmht.hypotheses:
        for track in hyp.tracks:
            assert len(track.satIDs) > 0
            assert len(track.satIDs) <= len(data['satID']) - 2  # Accounts for trimming


if __name__ == '__main__':
    import sys

    # Run unittest classes
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

    # Run pytest-style functions
    import pytest
    sys.exit(pytest.main([__file__]))
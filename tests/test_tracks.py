import pytest
import numpy as np
from unittest.mock import MagicMock
import ssapy
from ssapy.correlate_tracks import CircVelocityPrior, ZeroRadialVelocityPrior, GaussPrior, VolumeDistancePrior, orbit_to_param, partial, make_param_guess
from ssapy.constants import EARTH_MU, RGEO
from .ssapy_test_helpers import sample_LEO_orbit 
from ssapy.utils import normed

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
## testing functions ##
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

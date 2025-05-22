import numpy as np
from astropy.time import Time
import pytest
import sys
import importlib
import types

import ssapy
from ssapy.utils import get_angle
from ssapy.compute import (groundTrack, radecRate, radecRateObsToRV, lb_to_unit, nby3shape, calculate_orbital_elements, moon_shine, earth_shine, sun_shine, calc_M_v, M_v_lambertian,
                           calc_gamma, moon_normal_vector, lunar_lagrange_points, lunar_lagrange_points_circular, lagrange_points_lunar_frame, gcrf_to_itrf, angle_between_vectors, get_body, rotation_matrix_from_vectors)
from ssapy.constants import EARTH_MU,  MOON_RADIUS, EARTH_RADIUS
from .ssapy_test_helpers import sample_GEO_orbit
from ssapy.propagator import KeplerianPropagator
from ssapy.orbit import EarthObserver

# Mock implementations for dependencies
def mock_rv(orbit, time, propagator=None):
    """Mock implementation of rv function."""
    r = np.tile(orbit.r, (len(time), 1, 1))  # Repeat position for all times
    v = np.tile(orbit.v, (len(time), 1, 1))  # Repeat velocity for all times
    return r, v

def mock_ellipsoid_cartToSphere(x, y, z):
    """Mock implementation of Ellipsoid.cartToSphere."""
    lon = np.arctan2(y, x)  # Mock longitude
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))  # Mock latitude
    height = np.sqrt(x**2 + y**2 + z**2) - 6371e3  # Mock height (subtract Earth's radius)
    return lon, lat, height

# Replace actual functions with mocks for testing
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    monkeypatch.setattr("ssapy.rv", mock_rv)
    monkeypatch.setattr("ssapy.Ellipsoid.cartToSphere", mock_ellipsoid_cartToSphere)

def test_groundTrack_invalid_format():
    # Use sample_GEO_orbit and create time
    orbit = sample_GEO_orbit
    time = Time([0, 100, 200], format='gps')

    # Run groundTrack with invalid format
    with pytest.raises(ValueError, match="Format must be either 'cartesian' or 'geodetic'"):
        groundTrack(orbit, time, format='invalid')

# Mock implementation of `radec` function
def mock_radec(
    orbit, time, obsPos=None, obsVel=None, observer=None,
    propagator=None, obsAngleCorrection=None
):
    """Mock implementation of radec function."""
    n = len(time) if isinstance(time, (list, np.ndarray)) else 1
    ra = np.linspace(0, np.pi, n)  # Mock RA values
    dec = np.linspace(-np.pi / 2, np.pi / 2, n)  # Mock Dec values
    slant = np.linspace(1e7, 1e8, n)  # Mock slant range
    raRate = np.full(n, 1e-6)  # Mock RA rate
    decRate = np.full(n, 1e-6)  # Mock Dec rate
    slantRate = np.full(n, 1e3)  # Mock slant range rate
    return ra, dec, slant, raRate, decRate, slantRate

# Replace actual `radec` function with mock for testing
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    monkeypatch.setattr("ssapy.radec", mock_radec)


def mock_lb_to_unit(ra, dec):
    """Mock implementation of lb_to_unit function."""
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)

# Replace actual `lb_to_unit` function with mock for testing
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    monkeypatch.setattr("ssapy.compute.lb_to_unit", mock_lb_to_unit)


def test_radecRateObsToRV_without_obsVel():
    # Define inputs
    ra = np.array([0.1, 0.2])  # Right ascension in radians
    dec = np.array([0.3, 0.4])  # Declination in radians
    slantRange = np.array([1e7, 1e7])  # Slant range in meters
    obsPos = np.array([[6371e3, 0, 0], [6371e3, 0, 0]])  # Observer position in meters

    # Run radecRateObsToRV function
    r, v = radecRateObsToRV(
        ra, dec, slantRange, obsPos=obsPos
    )

    # Verify outputs
    assert r.shape == (2, 3)  # Object position shape
    assert v is None  # Velocity should be None
    assert np.allclose(r[0], obsPos[0] + mock_lb_to_unit(ra[0], dec[0]) * slantRange[0])

    
def test_nby3shape_1d_array():
    # Test with a 1-dimensional array
    arr = np.array([1, 2, 3])
    result = nby3shape(arr)
    expected = np.array([[1, 2, 3]])
    assert result.shape == (1, 3)
    assert np.array_equal(result, expected)

def test_nby3shape_2d_array_with_3_columns():
    # Test with a 2-dimensional array that already has 3 columns
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    result = nby3shape(arr)
    expected = arr
    assert result.shape == (2, 3)
    assert np.array_equal(result, expected)

def test_nby3shape_2d_array_with_transposable_shape():
    # Test with a 2-dimensional array that needs transposing
    arr = np.array([[1, 4], [2, 5], [3, 6]])
    result = nby3shape(arr)
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    assert result.shape == (2, 3)
    assert np.array_equal(result, expected)


def test_nby3shape_empty_array():
    # Test with an empty array
    arr = np.array([])
    with pytest.raises(ValueError):
        nby3shape(arr)


def test_calculate_orbital_elements_single_object():
    # Test with a single celestial object
    r = np.array([[1e7, 1e7, 1e7]])  # Position vector in meters
    v = np.array([[1e3, 2e3, 3e3]])  # Velocity vector in meters per second
    result = calculate_orbital_elements(r, v, mu_barycenter=EARTH_MU)
    
    # Validate the keys in the returned dictionary
    assert set(result.keys()) == {'a', 'e', 'i', 'tl', 'ap', 'raan', 'ta', 'L'}
    
    # Validate the shapes of the returned arrays
    assert len(result['a']) == 1
    assert len(result['e']) == 1
    assert len(result['i']) == 1
    assert len(result['tl']) == 1
    assert len(result['ap']) == 1
    assert len(result['raan']) == 1
    assert len(result['ta']) == 1
    assert len(result['L']) == 1

    # Validate the values (approximate checks)
    assert result['a'][0] > 0  # Semi-major axis should be positive
    assert 0 <= result['e'][0] < 1  # Eccentricity should be between 0 and 1
    assert 0 <= result['i'][0] <= np.pi  # Inclination should be between 0 and pi

def test_calculate_orbital_elements_invalid_inputs():
    # Test with invalid inputs (e.g., mismatched shapes)
    r = np.array([[1e7, 1e7]])  # Invalid position vector shape
    v = np.array([[1e3, 2e3, 3e3]])  # Valid velocity vector shape
    
    with pytest.raises(ValueError):
        calculate_orbital_elements(r, v, mu_barycenter=EARTH_MU)


def test_calculate_orbital_elements_high_eccentricity():
    # Test with high eccentricity orbit
    r = np.array([[1e7, 0, 0]])  # Position vector in meters
    v = np.array([[1e3, 1e3, 1e3]])  # Velocity vector in meters per second
    result = calculate_orbital_elements(r, v, mu_barycenter=EARTH_MU)
    
    # Validate the values
    assert result['e'][0] > 0.9  # Eccentricity should be high


def test_moon_shine_single_object():
    # Test with a single satellite and Moon position
    r_moon = np.array([[1e8, 1e8, 1e8]])  # Position of the Moon
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    r_earth = np.array([[0, 0, 0]])  # Position of the Earth
    r_sun = np.array([[1e11, 0, 0]])  # Position of the Sun

    result = moon_shine(
        r_moon=r_moon,
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.20,
        albedo_moon=0.12,
        albedo_back=0.50,
        albedo_front=0.05,
        area_panels=100
    )

    # Validate the keys in the returned dictionary
    assert set(result.keys()) == {'moon_bus', 'moon_panels'}

    # Validate the shapes of the returned arrays
    assert result['moon_bus'].shape == (1,)
    assert result['moon_panels'].shape == (1,)

    # Validate the values (approximate checks)
    assert result['moon_bus'][0] >= 0  # Flux should be non-negative
    assert result['moon_panels'][0] >= 0  # Flux should be non-negative

def test_moon_shine_zero_albedo():
    # Test with zero albedo for all surfaces
    r_moon = np.array([[1e8, 1e8, 1e8]])  # Position of the Moon
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    r_earth = np.array([[0, 0, 0]])  # Position of the Earth
    r_sun = np.array([[1e11, 0, 0]])  # Position of the Sun

    result = moon_shine(
        r_moon=r_moon,
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.0,
        albedo_moon=0.0,
        albedo_back=0.0,
        albedo_front=0.0,
        area_panels=100
    )

    # Validate the values (all fluxes should be zero)
    assert np.all(result['moon_bus'] == 0)
    assert np.all(result['moon_panels'] == 0)

def test_moon_shine_large_distance():
    # Test with large distances between objects
    r_moon = np.array([[1e12, 1e12, 1e12]])  # Position of the Moon
    r_sat = np.array([[1e11, 1e11, 1e11]])  # Position of the satellite
    r_earth = np.array([[0, 0, 0]])  # Position of the Earth
    r_sun = np.array([[1e13, 0, 0]])  # Position of the Sun

    result = moon_shine(
        r_moon=r_moon,
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.20,
        albedo_moon=0.12,
        albedo_back=0.50,
        albedo_front=0.05,
        area_panels=100
    )

    # Validate the values (fluxes should be very small due to large distances)
    assert np.all(result['moon_bus'] < 1e-10)
    assert np.all(result['moon_panels'] < 1e-10)

def test_moon_shine_invalid_inputs():
    # Test with invalid input shapes
    r_moon = np.array([[1e8, 1e8]])  # Invalid Moon position shape
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Valid satellite position
    r_earth = np.array([[0, 0, 0]])  # Valid Earth position
    r_sun = np.array([[1e11, 0, 0]])  # Valid Sun position

    with pytest.raises(ValueError):
        moon_shine(
            r_moon=r_moon,
            r_sat=r_sat,
            r_earth=r_earth,
            r_sun=r_sun,
            radius=0.4,
            albedo=0.20,
            albedo_moon=0.12,
            albedo_back=0.50,
            albedo_front=0.05,
            area_panels=100
        )

def test_earth_shine_single_object():
    # Test with a single satellite and Earth position
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    r_earth = np.array([[1.496e11, 0, 0]])  # Position of the Earth
    r_sun = np.array([[0, 0, 0]])  # Position of the Sun

    result = earth_shine(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.20,
        albedo_earth=0.30,
        albedo_back=0.50,
        area_panels=100
    )

    # Validate the keys in the returned dictionary
    assert set(result.keys()) == {'earth_bus', 'earth_panels'}

    # Validate the shapes of the returned arrays
    assert result['earth_bus'].shape == (1,)
    assert result['earth_panels'].shape == (1,)

    # Validate the values (approximate checks)
    assert result['earth_bus'][0] >= 0  # Flux should be non-negative
    assert result['earth_panels'][0] >= 0  # Flux should be non-negative

def test_earth_shine_multiple_objects():
    # Test with multiple satellites and Earth positions
    r_sat = np.array([[1e7, 1e7, 1e7], [2e7, 2e7, 2e7]])  # Position of the satellites
    r_earth = np.array([[1.496e11, 0, 0], [1.496e11, 0, 0]])  # Position of the Earth
    r_sun = np.array([[0, 0, 0], [0, 0, 0]])  # Position of the Sun

    result = earth_shine(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.20,
        albedo_earth=0.30,
        albedo_back=0.50,
        area_panels=100
    )

    # Validate the keys in the returned dictionary
    assert set(result.keys()) == {'earth_bus', 'earth_panels'}

    # Validate the shapes of the returned arrays
    assert result['earth_bus'].shape == (2,)
    assert result['earth_panels'].shape == (2,)

    # Validate the values (approximate checks)
    for i in range(2):
        assert result['earth_bus'][i] >= 0  # Flux should be non-negative
        assert result['earth_panels'][i] >= 0  # Flux should be non-negative

def test_earth_shine_zero_albedo():
    # Test with zero albedo for all surfaces
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    r_earth = np.array([[1.496e11, 0, 0]])  # Position of the Earth
    r_sun = np.array([[0, 0, 0]])  # Position of the Sun

    result = earth_shine(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.0,
        albedo_earth=0.0,
        albedo_back=0.0,
        area_panels=100
    )

    # Validate the values (all fluxes should be zero)
    assert np.all(result['earth_bus'] == 0)
    assert np.all(result['earth_panels'] == 0)

def test_earth_shine_large_distance():
    # Test with large distances between objects
    r_sat = np.array([[1e12, 1e12, 1e12]])  # Position of the satellite
    r_earth = np.array([[1.496e11, 0, 0]])  # Position of the Earth
    r_sun = np.array([[0, 0, 0]])  # Position of the Sun

    result = earth_shine(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.20,
        albedo_earth=0.30,
        albedo_back=0.50,
        area_panels=100
    )

    # Validate the values (fluxes should be very small due to large distances)
    assert np.all(result['earth_bus'] < 1e-10)
    assert np.all(result['earth_panels'] < 1e-10)

def test_earth_shine_invalid_inputs():
    # Test with invalid input shapes
    r_sat = np.array([[1e7, 1e7]])  # Invalid satellite position shape
    r_earth = np.array([[1.496e11, 0, 0]])  # Valid Earth position
    r_sun = np.array([[0, 0, 0]])  # Valid Sun position

    with pytest.raises(ValueError):
        earth_shine(
            r_sat=r_sat,
            r_earth=r_earth,
            r_sun=r_sun,
            radius=0.4,
            albedo=0.20,
            albedo_earth=0.30,
            albedo_back=0.50,
            area_panels=100
        )

def test_sun_shine_single_object():
    # Test with a single satellite position
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    r_earth = np.array([[0, 0, 0]])  # Position of the Earth
    r_sun = np.array([[1e11, 0, 0]])  # Position of the Sun

    result = sun_shine(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.20,
        albedo_front=0.05,
        area_panels=100
    )

    # Validate the keys in the returned dictionary
    assert set(result.keys()) == {'sun_bus', 'sun_panels'}

    # Validate the shapes of the returned arrays
    assert result['sun_bus'].shape == (1,)
    assert result['sun_panels'].shape == (1,)

    # Validate the values (approximate checks)
    assert result['sun_bus'][0] >= 0  # Flux should be non-negative
    assert result['sun_panels'][0] >= 0  # Flux should be non-negative

def test_sun_shine_zero_albedo():
    # Test with zero albedo for all surfaces
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    r_earth = np.array([[0, 0, 0]])  # Position of the Earth
    r_sun = np.array([[1e11, 0, 0]])  # Position of the Sun

    result = sun_shine(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.0,
        albedo_front=0.0,
        area_panels=100
    )

    # Validate the values (all fluxes should be zero)
    assert np.all(result['sun_bus'] == 0)
    assert np.all(result['sun_panels'] == 0)

def test_calc_M_v_single_object():
    # Test with a single satellite position
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    r_earth = np.array([[0, 0, 0]])  # Position of the Earth
    r_sun = np.array([[1e11, 0, 0]])  # Position of the Sun

    Mag_v, components = calc_M_v(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.20,
        sun_Mag=4.80,
        albedo_earth=0.30,
        albedo_moon=0.12,
        albedo_back=0.50,
        albedo_front=0.05,
        area_panels=100,
        return_components=True
    )

    # Validate the magnitude value
    assert Mag_v >= 0  # Magnitude should be non-negative

    # Validate the components dictionary
    assert set(components.keys()) == {'sun_bus', 'sun_panels', 'earth_bus', 'earth_panels', 'moon_bus', 'moon_panels'}

def test_calc_M_v_no_moon():
    # Test with no Moon contribution
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    r_earth = np.array([[0, 0, 0]])  # Position of the Earth
    r_sun = np.array([[1e11, 0, 0]])  # Position of the Sun

    Mag_v = calc_M_v(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        radius=0.4,
        albedo=0.20,
        sun_Mag=4.80,
        albedo_earth=0.30,
        albedo_moon=0.12,
        albedo_back=0.50,
        albedo_front=0.05,
        area_panels=100,
        r_moon=False
    )

    # Validate the magnitude value
    assert Mag_v >= 0  # Magnitude should be non-negative

def test_M_v_lambertian_single_object():
    # Test with a single satellite position and time
    r_sat = np.array([[1e7, 1e7, 1e7]])  # Position of the satellite
    times = np.array([0])  # Single time value

    Mag_v = M_v_lambertian(
        r_sat=r_sat,
        times=times,
        radius=1.0,
        albedo=0.20,
        sun_Mag=4.80,
        albedo_earth=0.30,
        albedo_moon=0.12,
        plot=False
    )

    # Validate the magnitude value
    assert Mag_v >= 0  # Magnitude should be non-negative

def test_lunar_lagrange_points():
    # Test with a single time point
    t = Time("2024-01-01")

    lagrange_points = lunar_lagrange_points(t)

    # Validate the keys in the returned dictionary
    assert set(lagrange_points.keys()) == {"L1", "L2", "L3", "L4", "L5"}

    # Validate the shapes of the returned arrays
    for key in lagrange_points:
        assert lagrange_points[key].shape == (3,)

def test_lunar_lagrange_points_circular():
    # Test with a single time point
    t = Time("2024-01-01")

    lagrange_points = lunar_lagrange_points_circular(t)

    # Validate the keys in the returned dictionary
    assert set(lagrange_points.keys()) == {"L1", "L2", "L3", "L4", "L5"}

    # Validate the shapes of the returned arrays
    for key in lagrange_points:
        assert lagrange_points[key].shape == (3,)

def test_lagrange_points_lunar_frame():
    # Test the function with default parameters
    lagrange_points = lagrange_points_lunar_frame()

    # Validate the keys in the returned dictionary
    assert set(lagrange_points.keys()) == {"L1", "L2", "L3", "L4", "L5"}

    # Validate the shapes of the returned arrays
    for key in lagrange_points:
        assert lagrange_points[key].shape == (3,)

    # Validate L4 and L5 positions (approximate checks)
    assert np.isclose(np.linalg.norm(lagrange_points["L4"]), np.linalg.norm(lagrange_points["L5"]))

MODULE_NAME = "ssapy.compute"

@pytest.fixture(autouse=True)
def cleanup_module_cache():
    """Ensure a clean import state before each test."""
    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]
    yield
    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]

def test_import_erfa_present():
    # Create a fake `erfa` module
    fake_erfa = types.ModuleType("erfa")
    sys.modules["erfa"] = fake_erfa

    # Remove fallback if it was accidentally cached
    sys.modules.pop("astropy._erfa", None)

    # Import and test
    import ssapy.compute
    importlib.reload(ssapy.compute)

    assert ssapy.compute.erfa is fake_erfa
import numpy as np
import pytest

from ssapy.ellipsoid import Ellipsoid

@pytest.fixture
def sample_ellipsoid():
    """Fixture for a standard ellipsoid with WGS84 flattening."""
    return Ellipsoid(Req=6378137.0, f=1 / 298.257223563)

@pytest.mark.timeout(30)
def test_sphere_to_cart_and_back_scalar(sample_ellipsoid):
    lon = np.deg2rad(30)
    lat = np.deg2rad(45)
    height = 1000.0

    x, y, z = sample_ellipsoid.sphereToCart(lon, lat, height)
    lon2, lat2, height2 = sample_ellipsoid.cartToSphere(x, y, z)

    assert np.isclose(lon, lon2, atol=1e-9)
    assert np.isclose(lat, lat2, atol=1e-9)
    assert np.isclose(height, height2, atol=1e-3)

@pytest.mark.timeout(30)
def test_sphere_to_cart_broadcasting(sample_ellipsoid):
    lon = np.deg2rad(30)
    lat = np.deg2rad([0, 45, 90])
    height = 500.0

    x, y, z = sample_ellipsoid.sphereToCart(lon, lat, height)
    assert x.shape == lat.shape
    assert y.shape == lat.shape
    assert z.shape == lat.shape

@pytest.mark.timeout(30)
def test_cart_to_sphere_broadcasting_safe(sample_ellipsoid):
    # Slightly offset points near the principal axes to avoid edge cases
    x = np.array([6378137.0, 1.0, 100.0])
    y = np.array([0.1, 6378137.0, 200.0])
    z = np.array([0.1, 0.2, 6356752.314245])

    lon, lat, height = sample_ellipsoid.cartToSphere(x, y, z)

    # Check output shapes
    assert lon.shape == x.shape
    assert lat.shape == x.shape
    assert height.shape == x.shape

    # Check values are finite
    assert np.all(np.isfinite(lon))
    assert np.all(np.isfinite(lat))
    assert np.all(np.isfinite(height))

    # Optional: Check known ranges
    assert np.all((lat >= -np.pi / 2) & (lat <= np.pi / 2))
    assert np.all((lon >= -np.pi) & (lon <= np.pi))


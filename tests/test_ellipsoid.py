import numpy as np
from ssapy.ellipsoid import Ellipsoid

@pytest.fixture
def sample_ellipsoid():
    """Fixture for a standard ellipsoid with WGS84 flattening."""
    return Ellipsoid(a=6378137.0, f=1 / 298.257223563)

def test_sphere_to_cart_and_back_scalar(sample_ellipsoid):
    lon = np.deg2rad(30)
    lat = np.deg2rad(45)
    height = 1000.0

    x, y, z = sample_ellipsoid.sphereToCart(lon, lat, height)
    lon2, lat2, height2 = sample_ellipsoid.cartToSphere(x, y, z)

    assert np.isclose(lon, lon2, atol=1e-9)
    assert np.isclose(lat, lat2, atol=1e-9)
    assert np.isclose(height, height2, atol=1e-3)


def test_sphere_to_cart_and_back_vectorized(sample_ellipsoid):
    lon = np.deg2rad(np.array([0, 90, 180]))
    lat = np.deg2rad(np.array([0, 45, -45]))
    height = np.array([0, 1000, 500])

    x, y, z = sample_ellipsoid.sphereToCart(lon, lat, height)
    lon2, lat2, height2 = sample_ellipsoid.cartToSphere(x, y, z)

    assert np.allclose(lon, lon2, atol=1e-9)
    assert np.allclose(lat, lat2, atol=1e-9)
    assert np.allclose(height, height2, atol=1e-2)


def test_sphere_to_cart_broadcasting(sample_ellipsoid):
    lon = np.deg2rad(30)
    lat = np.deg2rad([0, 45, 90])
    height = 500.0

    x, y, z = sample_ellipsoid.sphereToCart(lon, lat, height)
    assert x.shape == lat.shape
    assert y.shape == lat.shape
    assert z.shape == lat.shape


def test_cart_to_sphere_broadcasting(sample_ellipsoid):
    x = np.array([6378137, 0, 0])
    y = np.array([0, 6378137, 0])
    z = np.array([0, 0, 6356752.314245])

    lon, lat, height = sample_ellipsoid.cartToSphere(x, y, z)
    assert lon.shape == x.shape
    assert lat.shape == x.shape
    assert height.shape == x.shape

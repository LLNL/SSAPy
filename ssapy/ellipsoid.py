"""
Class to handle transformations between ECEF x,y,z coords and geodetic
longitude, latitude, and height.

Technically, only handles a one-axis ellipsoid, defined via a flattening
parameter f, but that's good enough for simple Earth models.
"""

import numpy as np

from ._ssapy import Ellipsoid
from .utils import continueClass


@continueClass
class Ellipsoid:
    def sphereToCart(self, lon, lat, height):
        lon, lat, height = np.broadcast_arrays(lon, lat, height)
        lon = np.ascontiguousarray(lon)
        lat = np.ascontiguousarray(lat)
        height = np.ascontiguousarray(height)

        x = np.empty_like(lon)
        y = np.empty_like(lon)
        z = np.empty_like(lon)

        self._sphereToCart(
            lon.ctypes.data, lat.ctypes.data, height.ctypes.data, lon.size,
            x.ctypes.data, y.ctypes.data, z.ctypes.data
        )

        return x, y, z

    def cartToSphere(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        z = np.ascontiguousarray(z)

        lon = np.empty_like(x)
        lat = np.empty_like(x)
        height = np.empty_like(x)

        self._cartToSphere(
            x.ctypes.data, y.ctypes.data, z.ctypes.data, x.size,
            lon.ctypes.data, lat.ctypes.data, height.ctypes.data
        )

        return lon, lat, height




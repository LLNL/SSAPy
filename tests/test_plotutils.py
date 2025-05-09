import pytest
import numpy as np
from astropy.time import Time
from matplotlib.figure import Figure
from matplotlib.testing.decorators import cleanup

from ssapy.plotUtils import (
    globe_plot, koe_plot, koe_hist_2d, scatter_2d, scatter_3d,
    check_numpy_array, check_type, scatter_dot_colors_scaled
)

@cleanup
def test_check_numpy_array_behavior():
    assert check_numpy_array(np.array([1, 2, 3])) == "numpy array"
    assert check_numpy_array([np.array([1]), np.array([2])]) == "list of numpy array"
    assert check_numpy_array([1, 2, 3]) == "not numpy"

@cleanup
def test_check_type_behavior():
    assert check_type(None) is None
    assert check_type([np.array([1]), np.array([2])]) == "List of arrays"
    assert check_type([1, 2, 3]) == "List of non-arrays"
    assert check_type(np.array([1, 2, 3])) == "Single array or list"

@cleanup
def test_scatter_dot_colors_scaled():
    colors = scatter_dot_colors_scaled(5)
    assert colors.shape == (5, 4)
    assert np.all((colors >= 0) & (colors <= 1))

@cleanup
def test_globe_plot_generates_figure():
    r = np.random.rand(10, 3) * 6.4e6  # Rough Earth radius scale
    t = Time("2025-01-01") + np.linspace(0, 1, 10) * 86400
    fig, ax = globe_plot(r, t)
    assert isinstance(fig, Figure)

@cleanup
def test_koe_plot_creates_figure():
    r = np.random.rand(10, 3) * 6.4e6
    v = np.random.rand(10, 3) * 1e3
    t = Time("2025-01-01") + np.linspace(0, 1, 10) * 86400
    fig, ax = koe_plot(r, v, t)
    assert isinstance(fig, Figure)

@cleanup
def test_koe_hist_2d_creates_figure():
    class DummyData:
        def __init__(self):
            self.a = np.random.uniform(1e7, 5e7, 100)
            self.e = np.random.uniform(0, 1, 100)
            self.i = np.radians(np.random.uniform(0, 90, 100))
            self.ta = np.radians(np.random.uniform(0, 360, 100))
    data = DummyData()
    fig = koe_hist_2d(data)
    assert isinstance(fig, Figure)

@cleanup
def test_scatter_2d_creates_plot():
    x = np.random.rand(100)
    y = np.random.rand(100)
    cs = np.random.rand(100)
    fig, ax = scatter_2d(x, y, cs)
    assert isinstance(fig, Figure)

@cleanup
def test_scatter_3d_creates_plot():
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    cs = np.random.rand(100)
    fig, ax = scatter_3d(x, y, z, cs)
    assert isinstance(fig, Figure)

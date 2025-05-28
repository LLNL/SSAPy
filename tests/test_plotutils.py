import numpy as np
import os
import matplotlib.pyplot as plt
import pytest
from astropy.time import Time
from PIL import Image as PILImage

from ssapy.plotUtils import (
    load_earth_file, draw_earth, load_moon_file, draw_moon,
    ground_track_plot, save_plot, groundTrackVideo,
    globe_plot, koe_plot, koe_hist_2d, orbit_plot,
    scatter_2d, scatter_3d, scatter_dot_colors_scaled,
    check_numpy_array, check_type, draw_dashed_circle,
    format_date_axis, save_plot_to_pdf, set_color_theme
)
from ssapy.utils import find_file

@pytest.fixture
def dummy_r():
    return np.random.rand(100, 3) * 1e7

@pytest.fixture
def dummy_t():
    return Time("2000-01-01") + np.linspace(0, 1, 100) * 86400

def test_load_earth_file_patch():
    result = load_earth_file()
    assert isinstance(result, PILImage.Image)
    assert result.size == (1080, 540)

def test_load_moon_file_patch():
    result = load_moon_file()
    assert isinstance(result, PILImage.Image)
    assert result.size == (1080, 540)


def test_check_numpy_array_behavior():
    assert check_numpy_array(np.array([1, 2, 3])) == "numpy array"
    assert check_numpy_array([np.array([1]), np.array([2])]) == "list of numpy array"
    assert check_numpy_array([1, 2, 3]) == "not numpy"
    assert check_numpy_array([]) == "not numpy"
    assert check_numpy_array("string") == "not numpy"
    assert check_numpy_array([np.array([1]), 2]) == "not numpy"
    assert check_numpy_array(["string", 123]) == "not numpy"
    assert check_numpy_array([[np.array([1])]]) == "not numpy"

def test_check_type_behavior():
    assert check_type(None) is None
    assert check_type([np.array([1]), np.array([2])]) == "List of arrays"
    assert check_type([1, 2, 3]) == "List of non-arrays"
    assert check_type(np.array([1, 2])) == "Single array or list"
    assert check_type(Time("2023-01-01")) == "Single array or list"
    assert check_type([np.array([1]), 2]) == "List of non-arrays"
    assert check_type([]) == "List of arrays"
    assert check_type([[np.array([1])]]) == "List of arrays"
    assert check_type("string") == "Not a list or array"
    assert check_type(123) == "Not a list or array"

def test_orbit_plot_basic(dummy_r, dummy_t):
    fig, axes = orbit_plot(dummy_r, t=dummy_t, frame='gcrf', show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert all(hasattr(ax, 'plot') for ax in axes)

def test_scatter_dot_colors_scaled_shape():
    assert scatter_dot_colors_scaled(0).shape == (0, 4)
    assert scatter_dot_colors_scaled(1).shape == (1, 4)
    assert scatter_dot_colors_scaled(10).shape == (10, 4)

def test_set_color_theme_variants():
    fig, ax = plt.subplots()
    fig, _ = set_color_theme(fig, ax, theme="black")
    assert ax.xaxis.label.get_color() == "white"

    fig, ax = plt.subplots()
    fig, _ = set_color_theme(fig, ax, theme="white")
    assert ax.xaxis.label.get_color() == "black"

    fig, ax = plt.subplots()
    fig, _ = set_color_theme(fig, ax, theme="dark")
    assert ax.xaxis.label.get_color() == "white"

def test_draw_dashed_circle():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_dashed_circle(ax, np.array([0, 0, 1]), radius=1.0, dashes=6)
    assert len(ax.lines) == 6

def test_format_date_axis_formats_labels():
    time_array = Time(['2024-07-01T00:00:00', '2024-07-01T06:00:00', '2024-07-01T12:00:00'])
    fig, ax = plt.subplots()
    ax.plot(time_array.decimalyear, np.random.rand(len(time_array)))
    format_date_axis(time_array, ax)
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert any(":" in label for label in xticklabels)

def test_save_plot_creates_file(tmp_path):
    fig, _ = plt.subplots()
    save_path = tmp_path / "fig_test.png"
    save_plot(fig, str(save_path))
    assert save_path.exists()

def test_save_plot_to_pdf_creates_file(tmp_path):
    fig, _ = plt.subplots()
    save_path = tmp_path / "plot.pdf"
    save_plot_to_pdf(fig, str(save_path))
    assert save_path.exists()

@pytest.fixture
def dummy_ground_data(tmp_path):
    r = np.random.rand(100, 3) * 1e7
    t = Time("2025-01-01") + np.linspace(0, 1, 100) * 86400
    img = PILImage.new("RGB", (5400, 2700), color="blue")
    path = tmp_path / "earth.png"
    img.save(path)
    return r, t, str(path)

def test_ground_track_plot_no_mock(dummy_ground_data, tmp_path):
    r, t, img_path = dummy_ground_data

    save_path = tmp_path / "ground_track.png"
    ground_track_plot(r, t, save_path=str(save_path))

    assert save_path.exists()

class MockStableData:
    def __init__(self):
        self.a = np.random.uniform(1 * 6371e3, 18 * 6371e3, 1000)
        self.e = np.random.uniform(0, 1, 1000)
        self.i = np.radians(np.random.uniform(0, 90, 1000))
        self.ta = np.radians(np.random.uniform(0, 360, 1000))

def test_koe_hist_2d_basic(monkeypatch):
    monkeypatch.setattr("ssapy.plotUtils.set_color_theme", lambda fig, ax, theme='black': (fig, ax))
    data = MockStableData()
    fig = koe_hist_2d(data)
    assert isinstance(fig, plt.Figure)

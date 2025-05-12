import pytest
import numpy as np
import os
import io
from astropy.time import Time
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.testing.decorators import cleanup
from PIL import Image as PILImage
import ipyvolume as ipv
from erfa import gst94

from ssapy.compute import groundTrack, calculate_orbital_elements, get_body 
from ssapy.utils import find_file
from ssapy.plotUtils import (
    globe_plot, koe_plot, koe_hist_2d, scatter_2d, scatter_3d,
    check_numpy_array, check_type, scatter_dot_colors_scaled,
    load_earth_file, draw_earth, load_moon_file, draw_moon,
    ground_track_plot, save_plot, groundTrackVideo, orbit_plot,
    globe_plot, koe_plot, set_color_theme, koe_hist_2d, orbit_divergence_plot,
    set_color_theme, draw_dashed_circle, format_date_axis, save_plot_to_pdf 
)

def test_load_earth_file():
    fake_file_path = "fake_path/earth.png"
    with patch("find_file", return_value=fake_file_path) as mock_find_file:
        mock_image = MagicMock(spec=PILImage.Image)
        mock_image.size = (5400, 2700)  # Original size of the image
        with patch("PILImage.open", return_value=mock_image) as mock_open:
            # Mock the resize method to return a resized image
            resized_image = MagicMock(spec=PILImage.Image)
            resized_image.size = (1080, 540)  # Resized dimensions
            mock_image.resize.return_value = resized_image

            # Call the function
            result = load_earth_file()

            # Assertions
            mock_find_file.assert_called_once_with("earth", ext=".png")
            mock_open.assert_called_once_with(fake_file_path)
            mock_image.resize.assert_called_once_with((1080, 540))
            assert result == resized_image
            assert result.size == (1080, 540)

def test_draw_earth():
    fake_texture = MagicMock()
    with patch("load_earth_file", return_value=fake_texture) as mock_load_earth_file:
        # Mock the ipv.plot_mesh function
        fake_plot_mesh = MagicMock()
        with patch("ipv.plot_mesh", return_value=fake_plot_mesh) as mock_plot_mesh:
            fake_gst = np.array([0.1, 0.2, 0.3])  # Fake GST values
            with patch("gst94", return_value=fake_gst) as mock_gst94:
                # Define test inputs
                test_time = np.array([100000, 200000, 300000])  # GPS seconds
                test_ngrid = 50
                test_R = 6371000  # Earth radius in meters
                test_rfactor = 1.5

                # Call the function
                result = draw_earth(test_time, ngrid=test_ngrid, R=test_R, rfactor=test_rfactor)

                # Assertions
                mock_load_earth_file.assert_called_once()
                mock_gst94.assert_called_once()
                mock_plot_mesh.assert_called_once()

                # Check that the plot_mesh function was called with the correct arguments
                args, kwargs = mock_plot_mesh.call_args
                x, y, z = args[:3]
                u, v = kwargs["u"], kwargs["v"]

                # Verify the dimensions of the mesh grid
                assert x.shape == (test_ngrid, test_ngrid)
                assert y.shape == (test_ngrid, test_ngrid)
                assert z.shape == (test_ngrid, test_ngrid)
                assert u.shape == (len(test_time), test_ngrid, test_ngrid)
                assert v.shape == u.shape

                # Verify the texture argument
                assert kwargs["texture"] == fake_texture

                # Verify the scaling of the mesh grid
                assert np.allclose(x * test_R * test_rfactor, x)
                assert np.allclose(y * test_R * test_rfactor, y)
                assert np.allclose(z * test_R * test_rfactor, z)

def test_load_moon_file():
    fake_file_path = "fake_path/moon.png"
    with patch("find_file", return_value=fake_file_path) as mock_find_file:
        # Mock the PILImage.open function to return a mock image
        mock_image = MagicMock(spec=PILImage.Image)
        mock_image.size = (5400, 2700)  # Original size of the image
        with patch("PILImage.open", return_value=mock_image) as mock_open:
            # Mock the resize method to return a resized image
            resized_image = MagicMock(spec=PILImage.Image)
            resized_image.size = (1080, 540)  # Resized dimensions
            mock_image.resize.return_value = resized_image

            # Call the function
            result = load_moon_file()

            # Assertions
            mock_find_file.assert_called_once_with("moon", ext=".png")
            mock_open.assert_called_once_with(fake_file_path)
            mock_image.resize.assert_called_once_with((1080, 540))
            assert result == resized_image
            assert result.size == (1080, 540)

def test_draw_moon():
    # Mock the load_moon_file function to return a fake texture
    fake_texture = MagicMock()
    with patch("load_moon_file", return_value=fake_texture) as mock_load_moon_file:
        # Mock the ipv.plot_mesh function
        fake_plot_mesh = MagicMock()
        with patch("ipv.plot_mesh", return_value=fake_plot_mesh) as mock_plot_mesh:
            # Mock the erfa.gst94 function
            fake_gst = np.array([0.1, 0.2, 0.3])  # Fake GST values
            with patch("gst94", return_value=fake_gst) as mock_gst94:
                # Define test inputs
                test_time = np.array([100000, 200000, 300000])  # GPS seconds
                test_ngrid = 50
                test_R = 1737400  # Moon radius in meters
                test_rfactor = 1.2

                # Call the function
                result = draw_moon(test_time, ngrid=test_ngrid, R=test_R, rfactor=test_rfactor)

                # Assertions
                mock_load_moon_file.assert_called_once()
                mock_gst94.assert_called_once()
                mock_plot_mesh.assert_called_once()

                # Check that the plot_mesh function was called with the correct arguments
                args, kwargs = mock_plot_mesh.call_args
                x, y, z = args[:3]
                u, v = kwargs["u"], kwargs["v"]

                # Verify the dimensions of the mesh grid
                assert x.shape == (test_ngrid, test_ngrid)
                assert y.shape == (test_ngrid, test_ngrid)
                assert z.shape == (test_ngrid, test_ngrid)
                assert u.shape == (len(test_time), test_ngrid, test_ngrid)
                assert v.shape == u.shape

                # Verify the texture argument
                assert kwargs["texture"] == fake_texture

                # Verify the scaling of the mesh grid
                assert np.allclose(x * test_R * test_rfactor, x)
                assert np.allclose(y * test_R * test_rfactor, y)
                assert np.allclose(z * test_R * test_rfactor, z)

@pytest.mark.mpl_cleanup
def test_ground_track_plot():
    fake_lon = np.array([0.1, 0.2, 0.3])  # Fake longitude in radians
    fake_lat = np.array([0.4, 0.5, 0.6])  # Fake latitude in radians
    fake_height = np.array([1000, 2000, 3000])  # Fake height in meters
    with patch("groundTrack", return_value=(fake_lon, fake_lat, fake_height)) as mock_groundTrack:
        # Mock the load_earth_file function to return a fake image
        fake_image = MagicMock()
        with patch("load_earth_file", return_value=fake_image) as mock_load_earth_file:
            # Mock the save_plot function
            with patch("save_plot") as mock_save_plot:
                # Define test inputs
                test_r = np.array([[7000, 8000, 9000]])  # Orbit positions (example values)
                test_t = np.array([100000, 200000, 300000])  # GPS seconds (example values)
                test_ground_stations = np.array([[45, -90], [30, 60]])  # Ground station lat/lon
                test_save_path = "test_path.png"

                # Call the function
                ground_track_plot(test_r, test_t, ground_stations=test_ground_stations, save_path=test_save_path)

                # Assertions
                mock_groundTrack.assert_called_once_with(test_r, test_t)
                mock_load_earth_file.assert_called_once()
                mock_save_plot.assert_called_once_with(plt.gcf(), test_save_path)

                # Verify the plot contains the expected elements
                ax = plt.gca()
                assert len(ax.lines) > 0  # Check that a line was plotted
                assert len(ax.collections) == len(test_ground_stations)  # Check ground station markers

                # Verify axis limits
                assert ax.get_xlim() == (-180, 180)
                assert ax.get_ylim() == (-90, 90)

def test_groundTrackVideo():
    # Mock ipv.figure to return a fake figure object
    fake_ipvfig = MagicMock()
    with patch("ipv.figure", return_value=fake_ipvfig) as mock_ipv_figure:
        # Mock ipv.scatter, ipv.plot, ipv.animation_control, and ipv.show
        fake_scatter = MagicMock()
        fake_plot = MagicMock()
        fake_animation_control = MagicMock()
        fake_show = MagicMock()
        with patch("ipv.scatter", return_value=fake_scatter) as mock_ipv_scatter, \
             patch("ipv.plot", return_value=fake_plot) as mock_ipv_plot, \
             patch("ipv.animation_control", return_value=fake_animation_control) as mock_ipv_animation_control, \
             patch("ipv.show", return_value=fake_show) as mock_ipv_show:
            # Mock draw_earth function
            fake_draw_earth = MagicMock()
            with patch("draw_earth", return_value=fake_draw_earth) as mock_draw_earth:
                # Define test inputs
                test_r = np.array([[7000, 8000, 9000], [7100, 8100, 9100], [7200, 8200, 9200]])  # Example positions
                test_time = np.array([100000, 200000, 300000])  # GPS seconds

                # Call the function
                groundTrackVideo(test_r, test_time)

                # Assertions
                mock_ipv_figure.assert_called_once_with(width=1000, height=500)
                mock_draw_earth.assert_called_once_with(test_time)
                mock_ipv_scatter.assert_called_once()
                mock_ipv_plot.assert_called_once()
                mock_ipv_animation_control.assert_called_once_with(
                    [fake_draw_earth, fake_scatter, fake_plot],
                    sequence_length=len(test_time),
                    interval=0
                )
                mock_ipv_show.assert_called_once()

                # Verify scatter plot arguments
                scatter_args, scatter_kwargs = mock_ipv_scatter.call_args
                assert np.array_equal(scatter_args[0], test_r[:, 0, None])  # x-coordinates
                assert np.array_equal(scatter_args[1], test_r[:, 1, None])  # y-coordinates
                assert np.array_equal(scatter_args[2], test_r[:, 2, None])  # z-coordinates
                assert scatter_kwargs["marker"] == "sphere"
                assert scatter_kwargs["color"] == "magenta"
                assert scatter_kwargs["size"] == 10

                # Verify plot arguments
                plot_args, plot_kwargs = mock_ipv_plot.call_args
                assert np.array_equal(plot_args[0], test_r[:, 0])  # x-coordinates
                assert np.array_equal(plot_args[1], test_r[:, 1])  # y-coordinates
                assert np.array_equal(plot_args[2], test_r[:, 2])  # z-coordinates
                assert plot_kwargs["color"] == "white"
                assert plot_kwargs["linewidth"] == 1

                # Verify camera settings
                assert fake_ipvfig.camera.position == (-2, 0, 0.2)
                assert fake_ipvfig.camera.up == (0, 0, 1)


def test_check_numpy_array_behavior():
    # Test for a single NumPy array
    assert check_numpy_array(np.array([1, 2, 3])) == "numpy array"

    # Test for a list of NumPy arrays
    assert check_numpy_array([np.array([1]), np.array([2])]) == "list of numpy array"

    # Test for a list that is not made of NumPy arrays
    assert check_numpy_array([1, 2, 3]) == "not numpy"

    # Test for an empty list
    assert check_numpy_array([]) == "not numpy"

    # Test for input that is neither a NumPy array nor a list
    assert check_numpy_array("not a numpy array") == "not numpy"

    # Test for a list containing mixed types (NumPy arrays and non-NumPy arrays)
    assert check_numpy_array([np.array([1]), 2]) == "not numpy"

    # Test for a list containing only non-NumPy arrays
    assert check_numpy_array(["string", 123, None]) == "not numpy"

    # Test for a nested list of NumPy arrays (edge case)
    assert check_numpy_array([[np.array([1]), np.array([2])]]) == "not numpy"

def test_check_type_behavior():
    # Test for None input
    assert check_type(None) is None

    # Test for a list of NumPy arrays
    assert check_type([np.array([1]), np.array([2])]) == "List of arrays"

    # Test for a list of non-arrays
    assert check_type([1, 2, 3]) == "List of non-arrays"

    # Test for a single NumPy array
    assert check_type(np.array([1, 2, 3])) == "Single array or list"

    # Test for a single Time object
    assert check_type(Time("2023-01-01")) == "Single array or list"

    # Test for a mixed list (contains both arrays and non-arrays)
    assert check_type([np.array([1]), 2]) == "List of non-arrays"

    # Test for an empty list
    assert check_type([]) == "List of arrays"  # Empty list should pass the `all()` check

    # Test for a nested list of arrays (edge case)
    assert check_type([[np.array([1]), np.array([2])]]) == "List of arrays"

    # Test for input that is neither a list nor an array
    assert check_type("not a list or array") == "Not a list or array"

    # Test for unsupported types (e.g., integer, dictionary)
    assert check_type(123) == "Not a list or array"
    assert check_type({"key": "value"}) == "Not a list or array"

@pytest.fixture
def dummy_r():
    return np.random.rand(100, 3) * 1e7  # 100 position vectors

@pytest.fixture
def dummy_t():
    from astropy.time import Time
    return Time('2000-01-01') + np.linspace(0, 1, 100) * 86400  # 1-day interval

def test_orbit_plot_basic(dummy_r, dummy_t):
    fig, axes = orbit_plot(dummy_r, t=dummy_t, frame='gcrf', show=False)
    assert isinstance(fig, Figure)
    assert isinstance(axes, list)
    assert len(axes) == 4
    for ax in axes:
        assert hasattr(ax, 'plot') 

@pytest.mark.parametrize("frame", ["gcrf", "itrf", "lunar", "lunar fixed"])
def test_orbit_plot_frames(dummy_r, dummy_t, frame):
    fig, axes = orbit_plot(dummy_r, t=dummy_t, frame=frame, show=False)
    assert isinstance(fig, Figure)

def test_invalid_frame(dummy_r, dummy_t):
    with pytest.raises(ValueError, match="Unknown plot type"):
        orbit_plot(dummy_r, t=dummy_t, frame="invalid_frame")

def test_incompatible_r_t_length():
    r = np.random.rand(100, 3)
    from astropy.time import Time
    t = Time('2000-01-01') + np.linspace(0, 1, 50) * 86400  # only 50 times
    with pytest.raises(ValueError):
        orbit_plot(r, t=t, frame='itrf')

def test_list_of_r_arrays_with_single_t():
    r = [np.random.rand(100, 3) for _ in range(3)]
    from astropy.time import Time
    t = Time('2000-01-01') + np.linspace(0, 1, 100) * 86400
    fig, axes = orbit_plot(r, t=t, frame='gcrf', show=False)
    assert isinstance(fig, Figure)

def test_plot_saving(tmp_path, dummy_r, dummy_t):
    save_path = tmp_path / "test_plot.png"
    orbit_plot(dummy_r, t=dummy_t, frame='gcrf', save_path=str(save_path), show=False)
    assert save_path.exists()

@pytest.fixture
def dummy_data():
    r = np.random.rand(50, 3) * 1e7
    t = np.linspace(0, 1, 50)
    return r, t

@patch("find_file")
@patch("PILImage.open")
def test_globe_plot_basic(mock_open, mock_find_file, dummy_data):
    # Mock image loading
    mock_img = MagicMock()
    mock_img.resize.return_value = mock_img
    mock_img.size = (5400, 2700)
    mock_img.__array__ = lambda self: np.ones((2700, 5400, 3)) * 255  # mock image data
    mock_open.return_value = mock_img
    mock_find_file.return_value = "dummy_path.png"

    r, t = dummy_data
    fig, ax = globe_plot(r, t, show=False)

    assert isinstance(fig, Figure)
    assert hasattr(ax, 'scatter')
    assert hasattr(ax, 'plot_surface')

@patch("find_file")
@patch("PILImage.open")
def test_globe_plot_with_limits(mock_open, mock_find_file, dummy_data):
    mock_img = MagicMock()
    mock_img.resize.return_value = mock_img
    mock_img.size = (5400, 2700)
    mock_img.__array__ = lambda self: np.ones((2700, 5400, 3)) * 255
    mock_open.return_value = mock_img
    mock_find_file.return_value = "dummy_path.png"

    r, t = dummy_data
    fig, ax = globe_plot(r, t, limits=2.0)

    assert ax.get_xlim() == (-2.0, 2.0)

@patch("find_file")
@patch("PILImage.open")
def test_globe_plot_save(tmp_path, mock_open, mock_find_file, dummy_data):
    mock_img = MagicMock()
    mock_img.resize.return_value = mock_img
    mock_img.size = (5400, 2700)
    mock_img.__array__ = lambda self: np.ones((2700, 5400, 3)) * 255
    mock_open.return_value = mock_img
    mock_find_file.return_value = "dummy_path.png"

    save_path = tmp_path / "globe_plot.png"
    r, t = dummy_data

    # Patch save_plot if needed, otherwise ensure the file is created by implementation
    with patch("your_module.save_plot") as mock_save:
        globe_plot(r, t, save_path=str(save_path))
        mock_save.assert_called_once()

@patch("find_file")
@patch("PILImage.open")
def test_globe_plot_extreme_angles(mock_open, mock_find_file, dummy_data):
    mock_img = MagicMock()
    mock_img.resize.return_value = mock_img
    mock_img.size = (5400, 2700)
    mock_img.__array__ = lambda self: np.ones((2700, 5400, 3)) * 255
    mock_open.return_value = mock_img
    mock_find_file.return_value = "dummy_path.png"

    r, t = dummy_data
    fig, ax = globe_plot(r, t, el=90, az=180)
    assert isinstance(fig, Figure)


@pytest.fixture
def dummy_orbit_data():
    r = np.random.rand(100, 3) * 1e7  # 100 position vectors
    v = np.random.rand(100, 3) * 1e3  # 100 velocity vectors
    t = Time("2025-01-01", scale="utc") + np.linspace(0, 365.25, 100)
    return r, v, t

@patch("calculate_orbital_elements")
@patch("set_color_theme")
def test_koe_plot_basic(mock_set_color_theme, mock_calc_elements, dummy_orbit_data):
    r, v, t = dummy_orbit_data

    # Mock orbital elements
    mock_calc_elements.return_value = {
        'a': np.ones(len(r)) * 42164e3,  # GEO altitude in meters
        'e': np.linspace(0, 0.1, len(r)),
        'i': np.linspace(0, np.pi/4, len(r)),
    }

    fig, ax = koe_plot(r, v, t=t, elements=['a', 'e', 'i'], body='Earth')

    assert isinstance(fig, Figure)
    assert hasattr(ax, 'plot')

@patch("calculate_orbital_elements")
@patch("set_color_theme")
def test_koe_plot_for_moon(mock_set_color_theme, mock_calc_elements, dummy_orbit_data):
    r, v, t = dummy_orbit_data

    mock_calc_elements.return_value = {
        'a': np.ones(len(r)) * 10000e3,
        'e': np.linspace(0, 0.1, len(r)),
        'i': np.linspace(0, np.pi / 6, len(r)),
    }

    fig, ax = koe_plot(r, v, t=t, body='Moon')
    assert isinstance(fig, Figure)

@patch("calculate_orbital_elements")
@patch("set_color_theme")
@patch("save_plot")
def test_koe_plot_saves_file(mock_save_plot, mock_set_color_theme, mock_calc_elements, dummy_orbit_data):
    r, v, t = dummy_orbit_data

    mock_calc_elements.return_value = {
        'a': np.ones(len(r)) * 42164e3,
        'e': np.linspace(0, 0.1, len(r)),
        'i': np.linspace(0, np.pi / 4, len(r)),
    }

    save_path = "test_koe_plot.png"
    koe_plot(r, v, t=t, save_path=save_path)
    mock_save_plot.assert_called_once()

@pytest.fixture
def mock_stable_data():
    class StableData:
        def __init__(self):
            self.a = np.random.uniform(1 * 6371e3, 18 * 6371e3, 1000)
            self.e = np.random.uniform(0, 1, 1000)
            self.i = np.radians(np.random.uniform(0, 90, 1000))
            self.ta = np.radians(np.random.uniform(0, 360, 1000))
    return StableData()

@patch("set_color_theme")
def test_koe_hist_2d_basic(mock_set_theme, mock_stable_data):
    fig = koe_hist_2d(mock_stable_data)
    assert isinstance(fig, Figure)

@patch("set_color_theme")
def test_koe_hist_2d_logscale(mock_set_theme, mock_stable_data):
    fig = koe_hist_2d(mock_stable_data, logscale=True)
    assert isinstance(fig, Figure)

@patch("set_color_theme")
@patch("save_plot")
def test_koe_hist_2d_saves_file(mock_save_plot, mock_set_theme, mock_stable_data):
    save_path = "test_histogram.pdf"
    fig = koe_hist_2d(mock_stable_data, save_path=save_path)
    mock_save_plot.assert_called_once_with(fig, save_path)

@patch("set_color_theme")
def test_koe_hist_2d_custom_title(mock_set_theme, mock_stable_data):
    custom_title = "My Custom Title"
    fig = koe_hist_2d(mock_stable_data, title=custom_title)
    assert fig._suptitle.get_text() == custom_title

def test_scatter_dot_colors_scaled_output():
    num_colors = 10
    colors = scatter_dot_colors_scaled(num_colors)

    # Check type
    assert isinstance(colors, np.ndarray)

    # Check shape
    assert colors.shape == (num_colors, 4)

    # Check value range for RGBA
    assert np.all((colors >= 0) & (colors <= 1))

def test_scatter_dot_colors_scaled_single_color():
    colors = scatter_dot_colors_scaled(1)
    assert colors.shape == (1, 4)

def test_scatter_dot_colors_scaled_zero():
    colors = scatter_dot_colors_scaled(0)
    assert colors.shape == (0, 4)

def test_scatter_2d_creates_plot():
    x = np.random.rand(100)
    y = np.random.rand(100)
    cs = np.random.rand(100)

    fig, ax = scatter_2d(x, y, cs, xlabel='X', ylabel='Y', title='Test', cbar_label='Color', dotsize=10)
    
    assert isinstance(fig, Figure)
    assert hasattr(ax, 'scatter')

def test_scatter_3d_returns_figure_and_ax():
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    cs = np.random.rand(100)
    fig, ax = scatter_3d(x, y, z, cs, title="Test 3D Scatter")
    assert isinstance(fig, Figure)
    assert hasattr(ax, "scatter")

def test_scatter_3d_without_cs():
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    fig, ax = scatter_3d(x, y, z)
    assert isinstance(fig, Figure)

def test_scatter_3d_with_combined_xyz_array():
    xyz = np.random.rand(100, 3)
    cs = np.random.rand(100)
    fig, ax = scatter_3d(xyz, cs=cs, title="From Combined Input")
    assert isinstance(fig, Figure)

@patch("save_plot")
def test_scatter_3d_save_plot(mock_save):
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    cs = np.random.rand(100)
    scatter_3d(x, y, z, cs, save_path="output.png")
    mock_save.assert_called_once()

@pytest.fixture
def sample_orbit_data():
    return np.random.rand(100, 3, 3) * 1e7  # 3 orbits, 100 steps

@pytest.fixture
def fake_moon_position():
    # shape should be (3, n)
    return np.random.rand(3, 100) * 1e8

@patch("save_plot")
@patch("get_body")
def test_orbit_divergence_plot_with_moon_calc(mock_get_body, mock_save_plot, sample_orbit_data):
    # Mock Moon body with position() method
    mock_moon = MagicMock()
    mock_moon.position.return_value = np.random.rand(3, 100) * 1e8
    mock_get_body.return_value = mock_moon

    t = Time("2025-01-01")
    orbit_divergence_plot(sample_orbit_data, t=t, title="Test Title")

    # Confirm moon position was used
    mock_moon.position.assert_called_once_with(t)

@patch("save_plot")
def test_orbit_divergence_plot_with_provided_r_moon(mock_save_plot, sample_orbit_data, fake_moon_position):
    orbit_divergence_plot(sample_orbit_data, r_moon=fake_moon_position, title="With Provided Moon")
    # Should not raise

def test_orbit_divergence_plot_with_bad_r_moon_shape(sample_orbit_data):
    bad_r_moon = np.random.rand(100, 3)  # Invalid shape (should be (3, n))

    with pytest.raises(IndexError, match="input moon data shape"):
        orbit_divergence_plot(sample_orbit_data, r_moon=bad_r_moon)

@patch("save_plot")
def test_orbit_divergence_plot_saves_plot(mock_save_plot, sample_orbit_data, fake_moon_position):
    orbit_divergence_plot(sample_orbit_data, r_moon=fake_moon_position, save_path="orbit_plot.png")
    mock_save_plot.assert_called_once()

def test_set_color_theme_black():
    fig, ax = plt.subplots()
    fig, _ = set_color_theme(fig, ax, theme='black')
    
    assert fig.get_facecolor()[:3] == (0, 0, 0)
    assert ax.get_facecolor()[:3] == (0, 0, 0)
    assert ax.xaxis.label.get_color() == 'white'
    assert ax.yaxis.label.get_color() == 'white'

def test_set_color_theme_white():
    fig, ax = plt.subplots()
    fig, _ = set_color_theme(fig, ax, theme='white')
    
    assert fig.get_facecolor()[:3] == (1, 1, 1)
    assert ax.get_facecolor()[:3] == (1, 1, 1)
    assert ax.xaxis.label.get_color() == 'black'
    assert ax.yaxis.label.get_color() == 'black'

def test_set_color_theme_dark_alias():
    fig, ax = plt.subplots()
    fig, _ = set_color_theme(fig, ax, theme='dark')
    
    assert fig.get_facecolor()[:3] == (0, 0, 0)
    assert ax.xaxis.label.get_color() == 'white'

def test_set_color_theme_with_3d_axes():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig, _ = set_color_theme(fig, ax, theme='black')

    # Check z-axis label and background color
    assert ax.zaxis.label.get_color() == 'white'
    assert ax.get_facecolor()[:3] == (0, 0, 0)

@patch("ssapy.rotation_matrix_from_vectors")
def test_draw_dashed_circle_executes(mock_rotation):
    # Identity rotation (no change) for simplicity
    mock_rotation.return_value = np.eye(3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    normal_vector = np.array([0, 0, 1])
    radius = 5
    dashes = 10
    draw_dashed_circle(ax, normal_vector, radius, dashes)

    # Expect the number of plotted lines to equal number of dashes
    assert len(ax.lines) == dashes

def test_draw_dashed_circle_uses_rotation():
    from ssapy.utils import rotation_matrix_from_vectors  # Import real util
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    normal_vector = np.array([1, 0, 0])  # Rotate z to x
    radius = 5
    dashes = 5
    draw_dashed_circle(ax, normal_vector, radius, dashes)

    # This test just checks that the function executes with a real rotation matrix
    assert len(ax.lines) == dashes


def test_format_date_axis_under_1_day():
    time_array = Time(['2024-07-01T00:00:00', '2024-07-01T04:00:00', '2024-07-01T08:00:00', 
                       '2024-07-01T12:00:00', '2024-07-01T16:00:00', '2024-07-01T20:00:00'])

    fig, ax = plt.subplots()
    ax.plot(time_array.decimalyear, np.random.rand(len(time_array)))

    format_date_axis(time_array, ax)

    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert len(xticklabels) == 6
    assert all(":" in label for label in xticklabels)  # Should include hour:minute

def test_format_date_axis_under_1_month():
    time_array = Time(['2024-07-01', '2024-07-05', '2024-07-10', '2024-07-15', '2024-07-20', '2024-07-25'])

    fig, ax = plt.subplots()
    ax.plot(time_array.decimalyear, np.random.rand(len(time_array)))

    format_date_axis(time_array, ax)

    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert len(xticklabels) == 6
    assert all("-" in label and len(label.split("-")) == 3 for label in xticklabels)  # dd-Mon-YYYY format

def test_format_date_axis_over_1_month():
    time_array = Time(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'])

    fig, ax = plt.subplots()
    ax.plot(time_array.decimalyear, np.random.rand(len(time_array)))

    format_date_axis(time_array, ax)

    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert len(xticklabels) <= 6
    assert all(label.count("-") == 1 and len(label.split("-")) == 2 for label in xticklabels)  # Mon-YYYY

@pytest.fixture
def sample_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig

@patch("PILImage.open")
@patch("PdfPages")
@patch("os.remove")
@patch("os.path.exists",return_value=False)
def test_save_plot_to_pdf_creates_new(
    mock_exists, mock_rename, mock_pdfpages, mock_pil_open, tmp_path, sample_figure
):
    pdf_path = tmp_path / "test_plot.pdf"
    dummy_image = MagicMock()
    mock_pil_open.return_value = dummy_image

    save_plot_to_pdf(sample_figure, str(pdf_path))

    mock_pdfpages.assert_called_once()
    mock_rename.assert_called_once()
    assert not os.path.exists(str(pdf_path) + "_temp.pdf")  # temp should be renamed

@patch("PILImage.open")
@patch("PdfPages")
@patch("os.remove")
@patch("os.path.exists", return_value=True)
@patch("PdfMerger")
def test_save_plot_to_pdf_appends_to_existing(
    mock_merger_class, mock_exists, mock_remove, mock_pdfpages, mock_pil_open, tmp_path, sample_figure
):
    pdf_path = tmp_path / "existing_plot.pdf"
    temp_pdf_path = str(pdf_path).replace(".pdf", "_temp.pdf")

    dummy_image = MagicMock()
    mock_pil_open.return_value = dummy_image

    mock_merger = MagicMock()
    mock_merger_class.return_value = mock_merger

    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%mock existing PDF")

    save_plot_to_pdf(sample_figure, str(pdf_path))

    mock_pdfpages.assert_called_once()
    mock_merger.append.assert_called()
    mock_merger.write.assert_called_once()
    mock_remove.assert_called_once_with(temp_pdf_path)

@pytest.fixture
def sample_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig

@patch("save_plot_to_pdf")
def test_save_plot_calls_pdf_save(mock_save_pdf, sample_figure):
    save_path = "plot.pdf"
    save_plot(sample_figure, save_path)
    mock_save_pdf.assert_called_once_with(sample_figure, save_path)

def test_save_plot_defaults_to_png(tmp_path, sample_figure):
    save_path = tmp_path / "my_figure"  # No extension
    save_plot(sample_figure, str(save_path))
    actual_file = tmp_path / "my_figure.png"
    assert actual_file.exists()

@patch("plt.Figure.savefig", side_effect=RuntimeError("Saving failed"))
def test_save_plot_exception_handling(mock_savefig, sample_figure, capsys):
    save_path = "bad_path.png"
    save_plot(sample_figure, save_path)

    captured = capsys.readouterr()
    assert "Error occurred while saving the figure: Saving failed" in captured.out

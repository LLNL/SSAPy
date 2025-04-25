"""
Utility functions for plotting.
"""


from .body import get_body
from .compute import groundTrack, lagrange_points_lunar_frame, calculate_orbital_elements
from .constants import RGEO, LD, EARTH_RADIUS, MOON_RADIUS, EARTH_MU, MOON_MU
from .utils import find_file, Time, find_smallest_bounding_cube, gcrf_to_itrf, gcrf_to_lunar_fixed, gcrf_to_lunar

import numpy as np
import os
import re
import io

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as mplcolors

from PyPDF2 import PdfMerger
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as PILImage
import ipyvolume as ipv

from typing import Union


def load_earth_file():
    """
    Loads and resizes an image of the Earth.

    This function locates a file named "earth.png" using the `find_file` function, 
    opens it as an image using the `PILImage.open` method, and resizes it to 
    1/5th of its original dimensions (5400x2700 scaled down to 1080x540). 
    The resized image is then returned.

    Returns:
        PIL.Image.Image: The resized Earth image.
    """
    earth = PILImage.open(find_file("earth", ext=".png"))
    earth = earth.resize((5400 // 5, 2700 // 5))
    return earth


def draw_earth(time, ngrid=100, R=EARTH_RADIUS, rfactor=1):
    """
    Parameters
    ----------
    time : array_like or astropy.time.Time (n,)
        If float (array), then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC
    ngrid: int
        Number of grid points in Earth model.
    R: float
        Earth radius in meters.  Default is WGS84 value.
    rfactor: float
        Factor by which to enlarge Earth (for visualization purposes)

    """
    earth = load_earth_file()

    from numbers import Real
    from erfa import gst94
    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    # Need earth rotation angle for times
    # Just use erfa.gst94.
    # This ignores precession/nutation, ut1-tt and polar motion, but should
    # be good enough for visualization.
    if isinstance(time, Time):
        time = time.gps
    if isinstance(time, Real):
        time = np.array([time])

    mjd_tt = 44244.0 + (time + 51.184) / 86400
    gst = gst94(2400000.5, mjd_tt)

    u = u - (gst / (2 * np.pi))[:, None, None]
    v = np.broadcast_to(v, u.shape)

    return ipv.plot_mesh(
        x * R * rfactor, y * R * rfactor, z * R * rfactor,
        u=u, v=v,
        wireframe=False,
        texture=earth
    )


def load_moon_file():
    """
    Loads and resizes an image of the Moon.

    This function locates a file named "moon.png" using the `find_file` function, 
    opens it as an image using the `PILImage.open` method, and resizes it to 
    1/5th of its original dimensions (5400x2700 scaled down to 1080x540). 
    The resized image is then returned.

    Returns:
        PIL.Image.Image: The resized Moon image.
    """
    moon = PILImage.open(find_file("moon", ext=".png"))
    moon = moon.resize((5400 // 5, 2700 // 5))
    return moon


def draw_moon(time, ngrid=100, R=MOON_RADIUS, rfactor=1):
    """
    Parameters
    ----------
    time : array_like or astropy.time.Time (n,)
        If float (array), then should correspond to GPS seconds;
        i.e., seconds since 1980-01-06 00:00:00 UTC
    ngrid: int
        Number of grid points in Earth model.
    R: float
        Earth radius in meters.  Default is WGS84 value.
    rfactor: float
        Factor by which to enlarge Earth (for visualization purposes)

    """
    moon = load_moon_file()

    from numbers import Real
    from erfa import gst94
    lat = np.linspace(-np.pi / 2, np.pi / 2, ngrid)
    lon = np.linspace(-np.pi, np.pi, ngrid)
    lat, lon = np.meshgrid(lat, lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    u = np.linspace(0, 1, ngrid)
    v, u = np.meshgrid(u, u)

    # Need earth rotation angle for t
    # Just use erfa.gst94.
    # This ignores precession/nutation, ut1-tt and polar motion, but should
    # be good enough for visualization.
    if isinstance(time, Time):
        time = time.gps
    if isinstance(time, Real):
        time = np.array([time])

    mjd_tt = 44244.0 + (time + 51.184) / 86400
    gst = gst94(2400000.5, mjd_tt)

    u = u - (gst / (2 * np.pi))[:, None, None]
    v = np.broadcast_to(v, u.shape)

    return ipv.plot_mesh(
        x * R * rfactor, y * R * rfactor, z * R * rfactor,
        u=u, v=v,
        wireframe=False,
        texture=moon
    )


def ground_track_plot(r, t, ground_stations=None, save_path=False):
    """
    Parameters
    ----------
    r : (3,) array_like - Orbit positions in meters.
    t: (n,) array_like - array of Astropy Time objects or time in gps seconds.

    optional - ground_stations: (n,2) array of of ground station (lat,lon) in degrees
    """
    lon, lat, height = groundTrack(r, t)

    fig = plt.figure(figsize=(15, 12))
    plt.imshow(load_earth_file(), extent=[-180, 180, -90, 90])
    plt.plot(np.rad2deg(lon), np.rad2deg(lat))
    if ground_stations is not None:
        for ground_station in ground_stations:
            plt.scatter(ground_station[1], ground_station[0], s=50, color='Red')
    plt.ylim(-90, 90)
    plt.xlim(-180, 180)
    plt.show()
    if save_path:
        save_plot(fig, save_path)


def groundTrackVideo(r, time):
    """
    Parameters
    ----------
    r : (3,) array_like
        Position of orbiting object in meters.
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC
    """
    ipvfig = ipv.figure(width=2000 / 2, height=1000 / 2)
    ipv.style.set_style_dark()
    ipv.style.box_off()
    ipv.style.axes_off()
    widgets = []
    widgets.append(draw_earth(time))
    widgets.append(
        ipv.scatter(
            r[:, 0, None],
            r[:, 1, None],
            r[:, 2, None],
            marker='sphere',
            color='magenta',
            size=10  # Increase the dot size (default is 1)
        )
    )
    # Line plot showing the path
    widgets.append(
        ipv.plot(
            r[:, 0],
            r[:, 1],
            r[:, 2],
            color='white',
            linewidth=1
        )
    )
    ipv.animation_control(widgets, sequence_length=len(time), interval=0)
    ipv.xyzlim(-10_000_000, 10_000_000)
    ipvfig.camera.position = (-2, 0, 0.2)
    ipvfig.camera.up = (0, 0, 1)
    ipv.show()


def check_numpy_array(variable: Union[np.ndarray, list]) -> str:
    """
    Checks if the input variable is a NumPy array, a list of NumPy arrays, or neither.

    Parameters
    ----------
    variable : Union[np.ndarray, list]
        The variable to check. It can either be a NumPy array or a list of NumPy arrays.

    Returns
    -------
    str
        Returns a string indicating the type of the variable:
        - "numpy array" if the variable is a single NumPy array,
        - "list of numpy array" if it is a list of NumPy arrays,
        - "not numpy" if it is neither.
    """
    if isinstance(variable, np.ndarray):
        return "numpy array"
    elif isinstance(variable, list):
        if len(variable) == 0:  # Handle empty list explicitly
            return "not numpy"
        elif all(isinstance(item, np.ndarray) for item in variable):
            return "list of numpy array"
    return "not numpy"


def check_type(t):
    """
    Determines the type of the input and provides a description based on its structure.

    This function checks the input `t` and returns a string describing its type or structure:
    - If `t` is `None`, it returns `None`.
    - If `t` is a `list`, it checks whether all elements in the list are either lists or NumPy arrays:
        - Returns "List of arrays" if all elements are lists or arrays.
        - Returns "List of non-arrays" if not all elements are lists or arrays.
    - If `t` is a `Time` object or a NumPy array, it returns "Single array or list".
    - For any other type, it returns "Not a list or array".

    Args:
        t (Any): The input to be checked.

    Returns:
        str or None: A description of the type or structure of `t`.
    """
    if t is None:
        return None
    elif isinstance(t, list):
        # Check if each element is a list or array
        if all(isinstance(item, (list, np.ndarray)) for item in t):
            return "List of arrays"
        else:
            return "List of non-arrays"
    elif isinstance(t, (Time, np.ndarray)):
        return "Single array or list"
    else:
        return "Not a list or array"
    

def orbit_plot(r, t=None, title='', figsize=(7, 7), save_path=False, frame="gcrf", show=False):
    """
    Plots the trajectory of one or more orbits in various views and coordinate frames.

    This function visualizes the position data of one or more orbits in 2D and 3D plots. 
    It supports different reference frames (e.g., GCRF, ITRF, Lunar) and allows customization 
    of plot appearance, including figure size, title, and saving the output.

    The function generates the following plots:
    - XY plane scatter plot with Earth/Moon markers and optional Lagrange points.
    - XZ plane scatter plot with Earth/Moon markers.
    - YZ plane scatter plot with Earth/Moon markers.
    - 3D scatter plot of the orbit(s) with Earth/Moon represented as spheres.

    Args:
        r (numpy.ndarray or list of numpy.ndarray): The position data of the orbit(s). 
            Can be a single NumPy array for one orbit or a list of arrays for multiple orbits.
        t (numpy.ndarray or list, optional): Time data corresponding to the position data. 
            Must match the shape of `r` or be a list of arrays for multiple orbits. 
            Defaults to None.
        title (str, optional): The title of the plot. Defaults to an empty string.
        figsize (tuple, optional): The size of the figure in inches (width, height). 
            Defaults to (7, 7).
        save_path (str or bool, optional): Path to save the plot. If False, the plot is not saved. 
            Defaults to False.
        frame (str, optional): The reference frame for the plot. Accepted values are 
            "gcrf", "itrf", "lunar", "lunar fixed", or "lunar axis". Defaults to "gcrf".
        show (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object.
            - axes (list): A list of subplot axes [ax1, ax2, ax3, ax4].

    Raises:
        ValueError: If the input `r` or `t` is not in a valid format or if the specified frame is not recognized.

    Notes:
        - The function supports transformations between coordinate frames and adjusts the plot accordingly.
        - Orbital bodies (Earth, Moon) are represented as spheres, scaled appropriately.
        - Lagrange points are plotted for Lunar frames, with markers and labels.
        - The bounds of the plots are dynamically adjusted based on the input data.
        - The function allows saving the plot to a specified path and optionally displaying it.
        - The axes are styled with a black background and white labels/ticks for better visibility.
    """
    input_type = check_numpy_array(r)
    t_type = check_type(t)
    
    if input_type == "numpy array":
        num_orbits = 1
        r = [r]

    if input_type == "list of numpy array":
        num_orbits = len(r)

    fig = plt.figure(dpi=100, figsize=figsize, facecolor='black')
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    bounds = {"lower": np.array([np.inf, np.inf, np.inf]), "upper": np.array([-np.inf, -np.inf, -np.inf])}

    # Check if all arrays in `r` are the same shape
    same_shape = all(np.shape(arr)[0] == np.shape(r[0]) for arr in r)
    for orbit_index in range(num_orbits):
        xyz = r[orbit_index]
        
        if t_type is None:
            if frame == "gcrf":
                r_moon = np.atleast_2d(get_body("moon").position(Time("2000-1-1")))
            else:
                raise ValueError("Need to provide t or list of t for each orbit in itrf, lunar or lunar fixed frames")
        else:
            if frame == "gcrf":
                if t_type == "Single array or list":
                    t_current = t
                elif t_type == "List of non-arrays" or t_type == "List of arrays":
                    t_current = max(t, key=len)
            else:
                if input_type == "numpy array":
                    # Single array case
                    t_current = t
                    if np.shape(t)[0] != np.shape(r)[1]:
                        raise ValueError("For a single numpy array 'r', 't' must be a 1D array of the same length as the first dimension of 'r'.")

                elif input_type == "list of numpy array":
                    if same_shape:
                        if t_type == "Single array or list":
                            t_current = t
                        elif t_type == "List of non-arrays" or t_type == "List of arrays":
                            t_current = max(t, key=len)
                        # Single `t` array is allowed
                        if len(t_current) != len(xyz):
                            raise ValueError("When 'r' is a list of arrays with the same shape, 't' must be a single 1D array matching the length of the first dimension of the arrays in 'r'.")
                    else:
                        # `t` must be a list of 1D arrays
                        if t_type == "Single array or list":
                            raise ValueError("When 'r' is a list of differing size numpy arrays, 't' must be a list of 1D arrays of equal length to the corresponding arrays in 'r'.")
                        elif t_type == "List of non-arrays" or t_type == "List of arrays":
                            if len(xyz) == len(t[orbit_index]):
                                t_current = t[orbit_index]
                            else:
                                print(f"length of t: {len(t_current)} and r: {len(xyz)}")
                                raise ValueError(f"'t' must be a 1D array matching the length of the first dimension of 'r[{orbit_index}]'.")
                                
            r_moon = get_body("moon").position(t_current).T
        r_earth = np.zeros(np.shape(r_moon))

        # Dictionary of frame transformations and titles
        def get_main_category(frame):
            variant_mapping = {
                "gcrf": "gcrf",
                "gcrs": "gcrf",
                "itrf": "itrf",
                "itrs": "itrf",
                "lunar": "lunar",
                "lunar_fixed": "lunar",
                "lunar fixed": "lunar",
                "lunar_centered": "lunar",
                "lunar centered": "lunar",
                "lunarearthfixed": "lunar axis",
                "lunarearth": "lunar axis",
                "lunar axis": "lunar axis",
                "lunar_axis": "lunar axis",
                "lunaraxis": "lunar axis",
            }
            return variant_mapping.get(frame.lower())

        frame_transformations = {
            "gcrf": ("GCRF", None),
            "itrf": ("ITRF", gcrf_to_itrf),
            "lunar": ("Lunar Frame", gcrf_to_lunar_fixed),
            "lunar axis": ("Moon on x-axis Frame", gcrf_to_lunar),
        }

        # Check if the frame is in the dictionary, and set central_dot accordingly
        frame = get_main_category(frame)
        if frame in frame_transformations:
            title2, transform_func = frame_transformations[frame]
            if transform_func:
                xyz = transform_func(xyz, t_current)
                r_moon = transform_func(r_moon, t_current)
                r_earth = transform_func(r_earth, t_current)
        else:
            raise ValueError("Unknown plot type provided. Accepted: gcrf, itrf, lunar, lunar fixed")

        xyz = xyz / RGEO
        r_moon = r_moon / RGEO
        r_earth = r_earth / RGEO

        lower_bound_temp, upper_bound_temp = find_smallest_bounding_cube(xyz, pad=1)
        bounds["lower"] = np.minimum(bounds["lower"], lower_bound_temp)
        bounds["upper"] = np.maximum(bounds["upper"], upper_bound_temp)

        if np.size(r_moon[:, 0]) > 1:
            grey_colors = cm.Greys(np.linspace(0, .8, len(r_moon[:, 0])))[::-1]
            blues = cm.Blues(np.linspace(.4, .9, len(r_moon[:, 0])))[::-1]
        else:
            grey_colors = "grey"
            blues = 'Blue'
        plot_settings = {
            "gcrf": {
                "primary_color": "blue",
                "primary_size": (EARTH_RADIUS / RGEO),
                "secondary_x": r_moon[:, 0],
                "secondary_y": r_moon[:, 1],
                "secondary_z": r_moon[:, 2],
                "secondary_color": grey_colors,
                "secondary_size": (MOON_RADIUS / RGEO)
            },
            "itrf": {
                "primary_color": "blue",
                "primary_size": (EARTH_RADIUS / RGEO),
                "secondary_x": r_moon[:, 0],
                "secondary_y": r_moon[:, 1],
                "secondary_z": r_moon[:, 2],
                "secondary_color": grey_colors,
                "secondary_size": (MOON_RADIUS / RGEO)
            },
            "lunar": {
                "primary_color": "grey",
                "primary_size": (MOON_RADIUS / RGEO),
                "secondary_x": r_earth[:, 0],
                "secondary_y": r_earth[:, 1],
                "secondary_z": r_earth[:, 2],
                "secondary_color": blues,
                "secondary_size": (EARTH_RADIUS / RGEO)
            },
            "lunar axis": {
                "primary_color": "blue",
                "primary_size": (EARTH_RADIUS / RGEO),
                "secondary_x": r_moon[:, 0],
                "secondary_y": r_moon[:, 1],
                "secondary_z": r_moon[:, 2],
                "secondary_color": grey_colors,
                "secondary_size": (MOON_RADIUS / RGEO)
            }
        }
        try:
            stn = plot_settings[frame]
        except KeyError:
            raise ValueError("Unknown plot type provided. Accepted: 'gcrf', 'itrf', 'lunar', 'lunar fixed'")

        if input_type == "numpy array":
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, len(xyz[:, 0])))
        else:
            scatter_dot_colors = cm.rainbow(np.linspace(0, 1, num_orbits))[orbit_index]
        
        ax1.scatter(xyz[:, 0], xyz[:, 1], color=scatter_dot_colors, s=1)
        ax1.add_patch(plt.Circle(xy=(0, 0), radius=1, color='white', linestyle='dashed', fill=False))  # Circle marking GEO
        ax1.add_patch(plt.Circle(xy=(0, 0), radius=stn['primary_size'], color=stn['primary_color'], linestyle='dashed', fill=False))  # Circle marking EARTH or MOON
        if r_moon[:, 0] is not False:
            ax1.scatter(stn['secondary_x'], stn['secondary_y'], color=stn['secondary_color'], s=stn['secondary_size'])
        ax1.set_aspect('equal')
        ax1.set_xlabel('x [GEO]', color='white')
        ax1.set_ylabel('y [GEO]', color='white')
        ax1.set_title(f'Frame: {title2}', color='white')
        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                if bounds["lower"][0] <= pos[0] <= bounds["upper"][0] and bounds["lower"][1] <= pos[1] <= bounds["upper"][1]:
                    ax1.scatter(pos[0], pos[1], color='white', label=point, s=10)
                    ax1.text(pos[0], pos[1], point, color='white')

        ax2.scatter(xyz[:, 0], xyz[:, 2], color=scatter_dot_colors, s=1)
        ax2.add_patch(plt.Circle(xy=(0, 0), radius=1, color='white', linestyle='dashed', fill=False))  # Circle marking GEO
        ax2.add_patch(plt.Circle(xy=(0, 0), radius=stn['primary_size'], color=stn['primary_color'], linestyle='dashed', fill=False))  # Circle marking EARTH or MOON
        if r_moon[:, 0] is not False:
            ax2.scatter(stn['secondary_x'], stn['secondary_z'], color=stn['secondary_color'], s=stn['secondary_size'])
        ax2.set_aspect('equal')
        ax2.set_xlabel('x [GEO]', color='white')
        ax2.set_ylabel('z [GEO]', color='white')
        ax2.yaxis.tick_right()  # Move y-axis ticks to the right
        ax2.yaxis.set_label_position("right")  # Move y-axis label to the right
        ax2.set_title(f'{title}', color='white')
        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                if bounds["lower"][0] <= pos[0] <= bounds["upper"][0] and bounds["lower"][2] <= pos[2] <= bounds["upper"][2]:
                    ax2.scatter(pos[0], pos[2], color='white', label=point, s=10)
                    ax2.text(pos[0], pos[2], point, color='white')

        ax3.scatter(xyz[:, 1], xyz[:, 2], color=scatter_dot_colors, s=1)
        ax3.add_patch(plt.Circle(xy=(0, 0), radius=1, color='white', linestyle='dashed', fill=False))
        ax3.add_patch(plt.Circle(xy=(0, 0), radius=stn['primary_size'], color=stn['primary_color'], linestyle='dashed', fill=False))  # Circle marking EARTH or MOON
        if r_moon[:, 0] is not False:
            ax1.scatter(stn['secondary_y'], stn['secondary_z'], color=stn['secondary_color'], s=stn['secondary_size'])
        ax3.set_aspect('equal')
        ax3.set_xlabel('y [GEO]', color='white')
        ax3.set_ylabel('z [GEO]', color='white')
        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                if bounds["lower"][1] <= pos[1] <= bounds["upper"][1] and bounds["lower"][2] <= pos[2] <= bounds["upper"][2]:
                    ax3.scatter(pos[1], pos[2], color='white', label=point, s=10)
                    ax3.text(pos[1], pos[2], point, color='white')
        
        # Create a 3d sphere of the Earth and Moon
        u = np.linspace(0, 2 * np.pi, 180)
        v = np.linspace(-np.pi/2, np.pi/2, 180)
        
        ax4.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=scatter_dot_colors, s=1)
        mesh_x = np.outer(np.cos(u), np.cos(v)).T * stn['primary_size'] + 0
        mesh_y = np.outer(np.sin(u), np.cos(v)).T * stn['primary_size'] + 0
        mesh_z = np.outer(np.ones(np.size(u)), np.sin(v)).T * stn['primary_size'] + 0
        ax4.plot_surface(mesh_x, mesh_y, mesh_z, color=stn['primary_color'], alpha=0.6, edgecolor='none')
        if r_moon[:, 0] is not False:
            ax4.scatter3D(stn['secondary_x'], stn['secondary_y'], stn['secondary_z'], color=stn['secondary_color'], s=stn['secondary_size'])

        ax4.set_xlabel('x [GEO]', color='white')
        ax4.set_ylabel('y [GEO]', color='white')
        ax4.set_zlabel('z [GEO]', color='white')
        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                if bounds["lower"][0] <= pos[0] <= bounds["upper"][0] and bounds["lower"][1] <= pos[1] <= bounds["upper"][1] and bounds["lower"][2] <= pos[2] <= bounds["upper"][2]:
                    ax4.scatter(pos[0], pos[1], pos[2], color='white', label=point, s=10)
                    ax4.text(pos[0], pos[1], pos[2], point, color='white')

    ax1.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax1.set_ylim(bounds["lower"][1], bounds["upper"][1])

    ax2.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax2.set_ylim(bounds["lower"][2], bounds["upper"][2])

    ax3.set_xlim(bounds["lower"][1], bounds["upper"][1])
    ax3.set_ylim(bounds["lower"][2], bounds["upper"][2])

    ax4.set_xlim(bounds["lower"][0], bounds["upper"][0])
    ax4.set_ylim(bounds["lower"][1], bounds["upper"][1])
    ax4.set_zlim(bounds["lower"][2], bounds["upper"][2])
    ax4.set_box_aspect([1, 1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('black')
        ax.tick_params(axis='both', colors='white')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    if save_path:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close()
    return fig, [ax1, ax2, ax3, ax4]


def globe_plot(r, t, limits=False, title='', figsize=(7, 8), save_path=False, el=30, az=0, scale=1):
    """
    Plot a 3D scatter plot of position vectors on a globe representation.

    Parameters:
    - r (array-like): Position vectors with shape (n, 3), where n is the number of points.
    - t (array-like): Time array corresponding to the position vectors. This parameter is not used in the current function implementation but is included for consistency.
    - limits (float, optional): The limit for the plot axes. If not provided, it is calculated based on the data. Default is False.
    - title (str, optional): Title of the plot. Default is an empty string.
    - figsize (tuple of int, optional): Figure size (width, height) in inches. Default is (7, 8).
    - save_path (str, optional): Path to save the generated plot. If not provided, the plot will not be saved. Default is False.
    - el (int, optional): Elevation angle (in degrees) for the view of the plot. Default is 30.
    - az (int, optional): Azimuth angle (in degrees) for the view of the plot. Default is 0.
    - scale (int, optional): Scale factor for resizing the Earth image. Default is 1.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    - ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis object used in the plot.

    The function creates a 3D scatter plot of the position vectors on a globe. The globe is represented using a textured Earth image, and the scatter points are colored using a rainbow colormap. The plot's background is set to black, and the plot is displayed with customizable elevation and azimuth angles.

    Example usage:
    ```
    import numpy as np
    from your_module import globe_plot

    # Example data
    r = np.array([[1, 2, 3], [4, 5, 6]])  # Replace with actual data
    t = np.arange(len(r))  # Replace with actual time data
    
    globe_plot(r, t, save_path='globe_plot.png')
    ```
    """
    x = r[:, 0] / RGEO
    y = r[:, 1] / RGEO
    z = r[:, 2] / RGEO
    if limits is False:
        limits = np.nanmax(np.abs([x, y, z])) * 1.2
    
    earth_png = PILImage.open(find_file("earth", ext=".png"))
    earth_png = earth_png.resize((5400 // scale, 2700 // scale))
    bm = np.array(earth_png.resize([int(d) for d in earth_png.size])) / 256.
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
    mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
    mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * EARTH_RADIUS / RGEO

    scatter_dot_colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.tick_params(axis='both', colors='white')
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5)
    ax.set_facecolor('black')  # Set plot background color to black
    ax.scatter(x, y, z, color=scatter_dot_colors, s=1)
    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)
    ax.view_init(elev=el, azim=az)
    ax.set_xlim([-limits, limits])
    ax.set_ylim([-limits, limits])
    ax.set_zlim([-limits, limits])
    ax.set_xlabel('x [GEO]', color='white')  # Set x-axis label color to white
    ax.set_ylabel('y [GEO]', color='white')  # Set y-axis label color to white
    ax.set_zlabel('z [GEO]', color='white')  # Set z-axis label color to white
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
    ax.tick_params(axis='z', colors='white')  # Set z-axis tick color to white
    ax.set_aspect('equal')
    fig, ax = set_color_theme(fig, ax, theme='black')
    if save_path:
        save_plot(fig, save_path)
    return fig, ax


def koe_plot(r, v, t=np.linspace(Time("2025-01-01", scale='utc'), Time("2026-01-01", scale='utc'), int(365.25*24)), elements=['a', 'e', 'i'], save_path=False, body='Earth'):
    """
    Plot Keplerian orbital elements over time for a given trajectory.

    Parameters:
    - r (array-like): Position vectors for the orbit.
    - v (array-like): Velocity vectors for the orbit.
    - t (array-like, optional): Time array for the plot, given as a sequence of `astropy.time.Time` objects or a `Time` object with `np.linspace`. Default is one year of hourly intervals starting from "2025-01-01".
    - elements (list of str, optional): List of orbital elements to plot. Options include 'a' (semi-major axis), 'e' (eccentricity), and 'i' (inclination). Default is ['a', 'e', 'i'].
    - save_path (str, optional): Path to save the generated plot. If not provided, the plot will not be saved. Default is False.
    - body (str, optional): The celestial body for which to calculate the orbital elements. Options are 'Earth' or 'Moon'. Default is 'Earth'.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    - ax1 (matplotlib.axes.Axes): The primary axis object used in the plot.

    The function calculates orbital elements for the given position and velocity vectors, and plots these elements over time. It creates a plot with two y-axes: one for the eccentricity and inclination, and the other for the semi-major axis. The x-axis represents time in decimal years. 

    Example usage:
    ```
    import numpy as np
    from astropy.time import Time
    from your_module import koe_plot

    # Example data
    r = np.array([[[1, 0, 0], [0, 1, 0]]])  # Replace with actual data
    v = np.array([[[0, 1, 0], [-1, 0, 0]]])  # Replace with actual data
    t = Time("2025-01-01", scale='utc') + np.linspace(0, int(1 * 365.25), int(365.25 * 24))
    
    koe_plot(r, v, t, save_path='orbital_elements_plot.png')
    ```
    """
    if 'earth' in body.lower():
        orbital_elements = calculate_orbital_elements(r, v, mu_barycenter=EARTH_MU)
    else:
        orbital_elements = calculate_orbital_elements(r, v, mu_barycenter=MOON_MU)
    fig, ax1 = plt.subplots(dpi=100)
    fig.patch.set_facecolor('white')
    ax1.plot([], [], label='semi-major axis [GEO]', c='C0', linestyle='-')
    ax2 = ax1.twinx()
    set_color_theme(fig, *[ax1, ax2], theme='white')

    ax1.plot(Time(t).decimalyear, [x for x in orbital_elements['e']], label='eccentricity', c='C1')
    ax1.plot(Time(t).decimalyear, [x for x in orbital_elements['i']], label='inclination [rad]', c='C2')
    ax1.set_xlabel('Year')
    ax1.set_ylim((0, np.pi / 2))
    ylabel = ax1.set_ylabel('', color='black')
    x = ylabel.get_position()[0] + 0.05
    y = ylabel.get_position()[1]
    fig.text(x - 0.001, y - 0.225, 'Eccentricity', color='C1', rotation=90)
    fig.text(x, y - 0.05, '/', color='k', rotation=90)
    fig.text(x, y - 0.025, 'Inclination [Radians]', color='C2', rotation=90)

    ax1.legend(loc='upper left')
    a = [x / RGEO for x in orbital_elements['a']]
    ax2.plot(Time(t).decimalyear, a, label='semi-major axis [GEO]', c='C0', linestyle='-')
    ax2.set_ylabel('semi-major axis [GEO]', color='C0')
    ax2.yaxis.label.set_color('C0')
    ax2.tick_params(axis='y', colors='C0')
    ax2.spines['right'].set_color('C0')
    
    if np.abs(np.max(a) - np.min(a)) < 2:
        ax2.set_ylim((np.min(a) - 0.5, np.max(a) + 0.5))
    format_date_axis(t, ax1)
    
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return fig, ax1


def koe_hist_2d(stable_data, title="Initial orbital elements of\n1 year stable cislunar orbits", limits=[1, 50], bins=200, logscale=False, cmap='coolwarm', save_path=False):
    """
    Create a 2D histogram plot for various Keplerian orbital elements of stable cislunar orbits.

    Parameters:
    - stable_data (object): An object with attributes `a`, `e`, `i`, and `ta`, which are arrays of semi-major axis, eccentricity, inclination, and true anomaly, respectively.
    - title (str, optional): Title of the figure. Default is "Initial orbital elements of\n1 year stable cislunar orbits".
    - limits (list, optional): Color scale limits for the histogram. Default is [1, 50].
    - bins (int, optional): Number of bins for the 2D histograms. Default is 200.
    - logscale (bool or str, optional): Whether to use logarithmic scaling for the color bar. Default is False. Can also be 'log' to apply logarithmic scaling.
    - cmap (str, optional): Colormap to use for the histograms. Default is 'coolwarm'.
    - save_path (str, optional): Path to save the generated plot. If not provided, the plot will not be saved. Default is False.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the 2D histograms.

    This function creates a 3x3 grid of 2D histograms showing the relationships between various orbital elements, including semi-major axis, eccentricity, inclination, and true anomaly. The color scale of the histograms can be adjusted with a logarithmic or linear normalization. The plot is customized with labels and a color bar.

    Example usage:
    ```
    import numpy as np
    from your_module import koe_hist_2d

    # Example data
    class StableData:
        def __init__(self):
            self.a = np.random.uniform(1, 20, 1000)
            self.e = np.random.uniform(0, 1, 1000)
            self.i = np.radians(np.random.uniform(0, 90, 1000))
            self.ta = np.radians(np.random.uniform(0, 360, 1000))

    stable_data = StableData()
    koe_hist_2d(stable_data, save_path='orbit_histograms.pdf')
    ```
    """
    if logscale or logscale == 'log':
        norm = mplcolors.LogNorm(limits[0], limits[1])
    else:
        norm = mplcolors.Normalize(limits[0], limits[1])
    fig, axes = plt.subplots(dpi=100, figsize=(10, 8), nrows=3, ncols=3)
    fig.patch.set_facecolor('white')
    st = fig.suptitle(title, fontsize=12)
    st.set_x(0.46)
    st.set_y(0.9)
    ax = axes.flat[0]
    ax.hist2d([x / RGEO for x in stable_data.a], [x for x in stable_data.e], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("eccentricity")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 1, 0.2))
    ax.set_xlim((1, 18))
    axes.flat[1].set_axis_off()
    axes.flat[2].set_axis_off()

    ax = axes.flat[3]
    ax.hist2d([x / RGEO for x in stable_data.a], [np.degrees(x) for x in stable_data.i], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("inclination [deg]")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.set_xlim((1, 18))
    ax = axes.flat[4]
    ax.hist2d([x for x in stable_data.e], [np.degrees(x) for x in stable_data.i], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax.set_yticks(np.arange(0, 91, 15))
    axes.flat[5].set_axis_off()

    ax = axes.flat[6]
    ax.hist2d([x / RGEO for x in stable_data.a], [np.degrees(x) for x in stable_data.ta], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("semi-major axis [GEO]")
    ax.set_ylabel("True Anomaly [deg]")
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.arange(0, 361, 60))
    ax.set_xlim((1, 18))
    ax = axes.flat[7]
    ax.hist2d([x for x in stable_data.e], [np.degrees(x) for x in stable_data.ta], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax.set_yticks(np.arange(0, 361, 60))
    ax = axes.flat[8]
    ax.hist2d([np.degrees(x) for x in stable_data.i], [np.degrees(x) for x in stable_data.ta], bins=bins, norm=norm, cmap=cmap)
    ax.set_xlabel("inclination [deg]")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_yticks(np.arange(0, 361, 60))

    im = fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, norm=norm, cmap=cmap)
    fig, ax = set_color_theme(fig, ax, theme='white')
    if save_path:
        save_plot(fig, save_path)
    return fig



def scatter_2d(x, y, cs, xlabel='x', ylabel='y', title='', cbar_label='', dotsize=1, colorsMap='jet', colorscale='linear', colormin=False, colormax=False, save_path=False):
    """
    Create a 2D scatter plot with optional color mapping.

    Parameters:
    - x (numpy.ndarray): Array of x-coordinates.
    - y (numpy.ndarray): Array of y-coordinates.
    - cs (numpy.ndarray): Array of values for color mapping.
    - xlabel (str, optional): Label for the x-axis. Default is 'x'.
    - ylabel (str, optional): Label for the y-axis. Default is 'y'.
    - title (str, optional): Title of the plot. Default is an empty string.
    - cbar_label (str, optional): Label for the color bar. Default is an empty string.
    - dotsize (int, optional): Size of the dots in the scatter plot. Default is 1.
    - colorsMap (str, optional): Colormap to use for the color mapping. Default is 'jet'.
    - colorscale (str, optional): Scale for the color mapping, either 'linear' or 'log'. Default is 'linear'.
    - colormin (float, optional): Minimum value for color scaling. If False, it is set to the minimum value of `cs`. Default is False.
    - colormax (float, optional): Maximum value for color scaling. If False, it is set to the maximum value of `cs`. Default is False.
    - save_path (str, optional): File path to save the plot. If not provided, the plot is not saved. Default is False.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes._subplots.AxesSubplot): The 2D axis object.

    This function creates a 2D scatter plot with optional color mapping based on the values provided in `cs`. 
    The color mapping can be adjusted using either a linear or logarithmic scale. The plot can be customized with axis labels, title, and colormap. 
    The plot can also be saved to a specified file path.

    Example usage:
    ```
    import numpy as np
    from your_module import scatter_2d

    # Example data
    x = np.random.rand(100)
    y = np.random.rand(100)
    cs = np.random.rand(100)

    scatter_2d(x, y, cs, xlabel='X-axis', ylabel='Y-axis', cbar_label='Color Scale', title='2D Scatter Plot')
    ```
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if colormax is False:
        colormax = np.max(cs)
    if colormin is False:
        colormin = np.min(cs)
    cm = plt.get_cmap(colorsMap)
    if colorscale == 'linear':
        cNorm = mplcolors.Normalize(vmin=colormin, vmax=colormax)
    elif colorscale == 'log':
        cNorm = mplcolors.LogNorm(vmin=colormin, vmax=colormax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm)
    ax.scatter(x, y, c=scalarMap.to_rgba(cs), s=dotsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap, shrink=.5, label=f'{cbar_label}', pad=0.04)
    plt.tight_layout()
    fig, ax = set_color_theme(fig, ax, theme='black')
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return


def scatter_3d(x, y=None, z=None, cs=None, xlabel='x', ylabel='y', zlabel='z', cbar_label='', dotsize=1, colorsMap='jet', title='', save_path=False):
    """
    Create a 3D scatter plot with optional color mapping.

    Parameters:
    - x (numpy.ndarray): Array of x-coordinates or a 2D array with shape (n, 3) representing the x, y, z coordinates.
    - y (numpy.ndarray, optional): Array of y-coordinates. Required if `x` is not a 2D array with shape (n, 3). Default is None.
    - z (numpy.ndarray, optional): Array of z-coordinates. Required if `x` is not a 2D array with shape (n, 3). Default is None.
    - cs (numpy.ndarray, optional): Array of values for color mapping. Default is None.
    - xlabel (str, optional): Label for the x-axis. Default is 'x'.
    - ylabel (str, optional): Label for the y-axis. Default is 'y'.
    - zlabel (str, optional): Label for the z-axis. Default is 'z'.
    - cbar_label (str, optional): Label for the color bar. Default is an empty string.
    - dotsize (int, optional): Size of the dots in the scatter plot. Default is 1.
    - colorsMap (str, optional): Colormap to use for the color mapping. Default is 'jet'.
    - title (str, optional): Title of the plot. Default is an empty string.
    - save_path (str, optional): File path to save the plot. If not provided, the plot is not saved. Default is False.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis object.

    This function creates a 3D scatter plot with optional color mapping based on the values provided in `cs`. 
    The plot can be customized with axis labels, title, and colormap. The plot can also be saved to a specified file path.

    Example usage:
    ```
    import numpy as np
    from your_module import scatter_3d

    # Example data
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    cs = np.random.rand(100)

    scatter_3d(x, y, z, cs, xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis', cbar_label='Color Scale', title='3D Scatter Plot')
    ```
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if x.ndim > 1:
        r = x
        x = r[:, 0]
        y = r[:, 1]
        z = r[:, 2]
    if cs is None:
        ax.scatter(x, y, z, s=dotsize)
    else:
        cm = plt.get_cmap(colorsMap)
        cNorm = mplcolors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm)
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=dotsize)
        scalarMap.set_array(cs)
        fig.colorbar(scalarMap, shrink=.5, label=f'{cbar_label}', pad=0.075)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.tight_layout()
    fig, ax = set_color_theme(fig, ax, theme='black')
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return fig, ax


def scatter_dot_colors_scaled(num_colors):
    """
    Generates a scaled array of colors using the rainbow colormap.

    This function creates a list of colors evenly spaced across the rainbow colormap. 
    The number of colors generated is determined by the `num_colors` parameter.

    Args:
        num_colors (int): The number of colors to generate.

    Returns:
        numpy.ndarray: An array of RGBA color values, where each entry corresponds to a color in the rainbow colormap.

    Example:
        >>> scatter_dot_colors_scaled(5)
        array([[1.        , 0.        , 0.        , 1.        ],
               [0.75      , 0.75      , 0.        , 1.        ],
               [0.        , 1.        , 0.        , 1.        ],
               [0.        , 0.75      , 0.75      , 1.        ],
               [0.        , 0.        , 1.        , 1.        ]])
    """
    return cm.rainbow(np.linspace(0, 1, num_colors))


def orbit_divergence_plot(rs, r_moon=[], t=False, limits=False, title='', save_path=False):
    """
    Plot multiple cislunar orbits in the GCRF frame with respect to the Earth and Moon.

    Parameters:
    - rs (numpy.ndarray): A 3D array of shape (n, 3, m) where n is the number of time steps, 
                          3 represents the x, y, z coordinates, and m is the number of orbits.
    - r_moon (numpy.ndarray, optional): A 2D array of shape (3, n) representing the Moon's position at each time step.
                                        If not provided, it is calculated based on the time `t`.
    - t (astropy.time.Time, optional): The time at which to calculate the Moon's position if `r_moon` is not provided. Default is False.
    - limits (float, optional): The plot limits in units of Earth's radius (GEO). If not provided, it is calculated as 1.2 times the maximum norm of `rs`. Default is False.
    - title (str, optional): The title of the plot. Default is an empty string.
    - save_path (str, optional): The file path to save the plot. If not provided, the plot is not saved. Default is False.

    Returns:
    None

    This function creates a 3-panel plot of multiple cislunar orbits in the GCRF frame. Each panel represents a different plane (xy, xz, yz) with Earth at the center.
    The orbits are plotted with color gradients to indicate progression. The Moon's position is also plotted if provided or calculated.

    Example usage:
    ```
    import numpy as np
    from astropy.time import Time
    from your_module import orbit_divergence_plot

    # Example data
    rs = np.random.randn(100, 3, 5)  # 5 orbits with 100 time steps each
    t = Time("2025-01-01")

    orbit_divergence_plot(rs, t=t, title='Cislunar Orbits')
    ```
    """
    if limits is False:
        limits = np.nanmax(np.linalg.norm(rs, axis=1) / RGEO) * 1.2
        print(f'limits: {limits}')
    if np.size(r_moon) < 1:
        moon = get_body("moon")
        r_moon = moon.position(t)
    else:
        # print('Lunar position(s) provided.')
        if r_moon.ndim != 2:
            raise IndexError(f"input moon data shape: {np.shape(r_moon)}, input should be 2 dimensions.")
            return None
        if np.shape(r_moon)[1] == 3:
            r_moon = r_moon.T
            # print(f"Tranposed input to {np.shape(r_moon)}")
    fig = plt.figure(dpi=100, figsize=(15, 4))
    for i in range(rs.shape[-1]):
        r = rs[:, :, i]
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        xm = r_moon[0] / RGEO
        ym = r_moon[1] / RGEO
        zm = r_moon[2] / RGEO
        scatter_dot_colors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Creating plot
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, color=scatter_dot_colors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(xm, ym, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('y [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], y[0], r'$\leftarrow$ start')
        plt.text(x[-1], y[-1], r'$\leftarrow$ end')

        plt.subplot(1, 3, 2)
        plt.scatter(x, z, color=scatter_dot_colors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(xm, zm, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], z[0], r'$\leftarrow$ start')
        plt.text(x[-1], z[-1], r'$\leftarrow$ end')
        plt.title(f'{title}')

        plt.subplot(1, 3, 3)
        plt.scatter(y, z, color=scatter_dot_colors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(ym, zm, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('y [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(y[0], z[0], r'$\leftarrow$ start')
        plt.text(y[-1], z[-1], r'$\leftarrow$ end')
    plt.tight_layout()
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return


def set_color_theme(fig, *axes, theme):
    """
    Set the color theme of the figure and axes to white or black and the text color to white or black.

    Parameters:
    - fig (matplotlib.figure.Figure): The figure to modify.
    - axes (list of matplotlib.axes._subplots.AxesSubplot): One or more axes to modify.
    - theme (str) either black/dark or white.
    
    Returns:
    - fig (matplotlib.figure.Figure): The modified figure.
    - axes (tuple of matplotlib.axes._subplots.AxesSubplot): The modified axes.

    This function changes the background color of the given figure and its axes to black or white. 
    It also sets the color of all text items (title, labels, tick labels) to white or black.

    Example usage:
    ```
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    set_color_theme(fig, ax, theme='black')
    plt.show()
    ```
    """
    if theme == 'black' or theme == 'dark':
        background = 'black'
        secondary = 'white'
    else:
        background = 'white'
        secondary = 'black'
    
    fig.patch.set_facecolor(background)

    for ax in axes:
        ax.set_facecolor(background)
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if hasattr(ax, 'zaxis'):
            ax_items.append(ax.zaxis.label)
        ax_items += ax.get_xticklabels() + ax.get_yticklabels()
        if hasattr(ax, 'get_zticklabels'):
            ax_items += ax.get_zticklabels()
        ax_items += ax.get_xticklines() + ax.get_yticklines()
        if hasattr(ax, 'get_zticklines'):
            ax_items += ax.get_zticklines()
        for item in ax_items:
            item.set_color(secondary)

    return fig, axes


def draw_dashed_circle(ax, normal_vector, radius, dashes, dash_length=0.1, label='Dashed Circle'):
    """
    Draw a dashed circle on a 3D axis with a given normal vector.

    Parameters:
    - ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis on which to draw the circle.
    - normal_vector (array-like): A 3-element array representing the normal vector to the plane of the circle.
    - radius (float): The radius of the circle.
    - dashes (int): The number of dashes to be used in drawing the circle.
    - dash_length (float, optional): The relative length of each dash, as a fraction of the circle's circumference. Default is 0.1.
    - label (str, optional): The label for the circle. Default is 'Dashed Circle'.

    Returns:
    None

    This function draws a dashed circle on a 3D axis. The circle is defined in the xy-plane, then rotated to align with the given normal vector. The circle is divided into dashes to create the dashed effect.

    Example usage:
    ```
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from your_module import draw_dashed_circle

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    normal_vector = [0, 0, 1]
    radius = 5
    dashes = 20

    draw_dashed_circle(ax, normal_vector, radius, dashes)

    plt.show()
    ```
    """
    from .utils import rotation_matrix_from_vectors
    # Define the circle in the xy-plane
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    z_circle = np.zeros_like(theta)
    
    # Stack the coordinates into a matrix
    circle_points = np.vstack((x_circle, y_circle, z_circle)).T
    
    # Create the rotation matrix to align z-axis with the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal_vector)
    
    # Rotate the circle points
    rotated_points = circle_points @ rotation_matrix.T
    
    # Create dashed effect
    dash_points = []
    dash_gap = int(len(theta) / dashes)
    for i in range(dashes):
        start_idx = i * dash_gap
        end_idx = start_idx + int(dash_length * len(theta))
        dash_points.append(rotated_points[start_idx:end_idx])
    
    # Plot the dashed circle in 3D
    for points in dash_points:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'k--', label=label)
        label = None  # Only one label


# #####################################################################
# Formatting x axis
# #####################################################################
def format_date_axis(time_array, ax):
    """
    Format the x-axis of a plot with time-based labels depending on the span of the time array.

    Parameters:
    - time_array (array-like): An array of time objects (e.g., astropy.time.Time) to be used for the x-axis labels.
    - ax (matplotlib.axes.Axes): The matplotlib axes object on which to set the x-axis labels.

    Returns:
    None

    This function formats the x-axis labels of a plot based on the span of the provided time array. The labels are 
    set to show either hours and day-month or month-year formats, depending on the time span.

    The function performs the following steps:
    1. If the time span is less than one month:
        - If the time span is less than a day, the labels show 'HH:MM dd-Mon'.
        - Otherwise, the labels show 'dd-Mon-YYYY'.
    2. If the time span is more than one month, the labels show 'Mon-YYYY'.

    The function selects six nearly evenly spaced points in the time array to set the x-axis labels.

    Example usage:
    ```
    import matplotlib.pyplot as plt
    from astropy.time import Time
    import numpy as np

    # Example time array
    time_array = Time(['2024-07-01T00:00:00', '2024-07-01T06:00:00', '2024-07-01T12:00:00', 
                       '2024-07-01T18:00:00', '2024-07-02T00:00:00'])

    fig, ax = plt.subplots()
    ax.plot(time_array.decimalyear, np.random.rand(len(time_array)))
    format_date_axis(time_array, ax)
    plt.show()
    ```
    """
    n = 6  # Number of nearly evenly spaced points to select
    time_span_in_months = (time_array[-1].datetime - time_array[0].datetime).days / 30
    if time_span_in_months < 1:
        # Get the time span in hours
        time_span_in_hours = (time_array[-1].datetime - time_array[0].datetime).total_seconds() / 3600

        if time_span_in_hours < 24:
            # If the time span is less than a day, format the x-axis with hh:mm dd-mon
            selected_times = np.linspace(time_array[0], time_array[-1], n)
            selected_hour_strings = [t.strftime('%H:%M') for t in selected_times]
            selected_day_month_strings = [t.strftime('%d-%b') for t in selected_times]
            selected_tick_labels = [f'{hour} {day_month}' for hour, day_month in zip(selected_hour_strings, selected_day_month_strings)]
            selected_decimal_years = [t.decimalyear for t in selected_times]
            # Set the x-axis tick positions and labels
            ax.set_xticks(selected_decimal_years)
            ax.set_xticklabels(selected_tick_labels)
            return
    if n >= time_span_in_months:
        # Get evenly spaced points in the time_array
        selected_indices = np.round(np.linspace(0, len(time_array) - 1, n)).astype(int)
        selected_times = time_array[selected_indices]
        selected_month_year_strings = [t.strftime('%d-%b-%Y') for t in selected_times]
    else:
        # Get the first of n nearly evenly spaced months in the time
        step = int(len(time_array) / (n - 1)) - 1
        selected_times = time_array[::step]
        selected_month_year_strings = [t.strftime('%b-%Y') for t in selected_times]
    selected_decimal_years = [t.decimalyear for t in selected_times]
    # Set the x-axis tick positions and labels
    ax.set_xticks(selected_decimal_years)
    ax.set_xticklabels(selected_month_year_strings)

    # Optional: Rotate the tick labels for better visibility
    plt.xticks(rotation=0)
    return


save_plot_to_pdf_call_count = 0


def save_plot_to_pdf(figure, pdf_path):
    """
    Save a Matplotlib figure to a PDF file, with support for merging with existing PDFs.

    Parameters:
    - figure (matplotlib.figure.Figure): The Matplotlib figure to be saved.
    - pdf_path (str): The path to the PDF file. If the file exists, the figure will be appended to it.

    Returns:
    None

    This function saves a Matplotlib figure as a PNG in-memory and then converts it to a PDF.
    If the specified PDF file already exists, the new figure is appended to it. Otherwise,
    a new PDF file is created. The function also keeps track of how many times it has been called
    using a global variable `save_plot_to_pdf_call_count`.

    The function performs the following steps:
    1. Expands the user directory if the path starts with `~`.
    2. Generates a temporary PDF path by appending "_temp.pdf" to the original path.
    3. Saves the figure as a PNG in-memory using a BytesIO buffer.
    4. Opens the in-memory PNG using PIL and creates a new figure to display the image.
    5. Saves the new figure with the image into a temporary PDF.
    6. If the specified PDF file exists, merges the temporary PDF with the existing one.
       Otherwise, renames the temporary PDF to the specified path.
    7. Closes the original and temporary figures and prints a message indicating the save location.

    Example usage:
    ```
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    save_plot_to_pdf(fig, '~/Desktop/my_plot.pdf')
    ```
    """
    global save_plot_to_pdf_call_count
    save_plot_to_pdf_call_count += 1
    if '~' == pdf_path[0]:
        pdf_path = os.path.expanduser(pdf_path)
    if '.' in pdf_path:
        temp_pdf_path = re.sub(r"\.[^.]+$", "_temp.pdf", pdf_path)
    else:
        temp_pdf_path = f"{pdf_path}_temp.pdf"
    # Save the figure as a PNG in-memory using BytesIO
    png_buffer = io.BytesIO()
    figure.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
    # Rewind the buffer to the beginning
    png_buffer.seek(0)
    # Open the in-memory PNG using PIL
    png_image = PILImage.open(png_buffer)
    with PdfPages(temp_pdf_path) as pdf:
        # Create a new figure and axis to display the image
        img_fig, img_ax = plt.subplots()
        img_ax.imshow(png_image)
        img_ax.axis('off')
        # Save the figure with the image into the PDF
        pdf.savefig(img_fig, dpi=300, bbox_inches='tight')
    if os.path.exists(pdf_path):
        merger = PdfMerger()
        with open(pdf_path, "rb") as main_pdf, open(temp_pdf_path, "rb") as temp_pdf:
            merger.append(main_pdf)
            merger.append(temp_pdf)
            with open(pdf_path, "wb") as merged_pdf:
                merger.write(merged_pdf)
        os.remove(temp_pdf_path)
    else:
        os.rename(temp_pdf_path, pdf_path)
    plt.close(figure)
    plt.close(img_fig)  # Close the figure and new figure created
    print(f"Saved figure {save_plot_to_pdf_call_count} to {pdf_path}")
    return


def save_plot(figure, save_path, dpi=200):
    """
    Save a Python figure as a PNG/JPG/PDF/ect. image. If no extension is given in the save_path, a .png is defaulted.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to be saved.
        save_path (str): The file path where the image will be saved.

    Returns:
        None
    """
    if save_path.lower().endswith('.pdf'):
        save_plot_to_pdf(figure, save_path)
        return
    try:
        base_name, extension = os.path.splitext(save_path)
        if extension.lower() == '':
            save_path = base_name + '.png'
        # Save the figure as a PNG image
        figure.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(figure)  # Close the figure to release resources
        # print(f"Figure saved at: {save_path}")
    except Exception as e:
        print(f"Error occurred while saving the figure: {e}")


def save_animated_gif(gif_name, frames, fps=30):
    """
    Create a GIF from a sequence of image frames.

    Parameters:
    - gif_name (str): The name of the output GIF file, including the .gif extension.
    - frames (list of str): A list of file paths to the image frames to be included in the GIF.
    - fps (int, optional): Frames per second for the GIF. Default is 30.

    Returns:
    None

    This function uses the imageio library to write a GIF file. It prints messages indicating
    the start and completion of the GIF writing process. Each frame is read from the provided
    file paths and appended to the GIF.

    Example usage:
    frames = ['frame1.png', 'frame2.png', 'frame3.png']
    write_gif('output.gif', frames, fps=24)
    """
    import imageio
    print(f'Writing gif: {gif_name}')
    with imageio.get_writer(gif_name, mode='I', duration=1 / fps) as writer:
        for i, filename in enumerate(frames):
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f'Wrote {gif_name}')
    return

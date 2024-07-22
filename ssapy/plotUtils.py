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


def load_earth_file():
    earth = PILImage.open(find_file("earth", ext=".png"))
    earth = earth.resize((5400 // 5, 2700 // 5))
    return earth


def drawEarth(time, ngrid=100, R=EARTH_RADIUS, rfactor=1):
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
    moon = PILImage.open(find_file("moon", ext=".png"))
    moon = moon.resize((5400 // 5, 2700 // 5))
    return moon


def drawMoon(time, ngrid=100, R=MOON_RADIUS, rfactor=1):
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


def groundTrackPlot(r, t, ground_stations=None, save_path=False):
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
    widgets.append(drawEarth(time))
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


def check_numpy_array(variable):
    if isinstance(variable, np.ndarray):
        return "numpy array"
    elif isinstance(variable, list) and all(isinstance(item, np.ndarray) for item in variable):
        return "list of numpy array"
    else:
        return "not numpy"


def orbit_plot(r, t=[], limits=False, title='', figsize=(7, 7), save_path=False, frame="gcrf", show=False):
    """
    Parameters
    ----------
    r : (n,3) or array of [(n,3), ..., (n,3)] array_like
        Position of orbiting object(s) in meters.
    t: optional - t when r was calculated.
    limits: optional - x and y limits of the plot
    title: optional - title of the plot
    """

    def _make_scatter(fig, ax1, ax2, ax3, ax4, r, t, limits, title='', orbit_index='', num_orbits=1, frame=False):
        if np.size(t) < 1:
            if frame in ["itrf", "lunar", "lunar_fixed"]:
                raise("Need to provide t for itrf, lunar or lunar fixed frames")
            r_moon = np.atleast_2d(get_body("moon").position(Time("2000-1-1")))
        else:
            r_moon = get_body("moon").position(t).T

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
                r = transform_func(r, t)
                r_moon = transform_func(r_moon, t)
        else:
            raise ValueError("Unknown plot type provided. Accepted: gcrf, itrf, lunar, lunar fixed")

        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        xm = r_moon[:, 0] / RGEO
        ym = r_moon[:, 1] / RGEO
        zm = r_moon[:, 2] / RGEO
            
        if np.size(xm) > 1:
            gradient_colors = cm.Greys(np.linspace(0, .8, len(xm)))[::-1]
            blues = cm.Blues(np.linspace(.4, .9, len(xm)))[::-1]
        else:
            gradient_colors = "grey"
            blues = 'Blue'
        plot_settings = {
            "gcrf": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "itrf": ("blue", 50, 1, xm, ym, zm, gradient_colors),
            "lunar": ("grey", 25, 1.3, xm, ym, zm, blues),
            "lunar axis": ("blue", 50, 1, -xm, -ym, -zm, gradient_colors)
        }
        try:
            stn = plot_settings[frame]
        except KeyError:
            raise ValueError("Unknown plot type provided. Accepted: 'gcrf', 'itrf', 'lunar', 'lunar fixed'")
        if limits is False:
            lower_bound, upper_bound = find_smallest_bounding_cube(r / RGEO)
            lower_bound = lower_bound * 1.2
            upper_bound = upper_bound * 1.2

        if orbit_index == '':
            angle = 0
            dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))
        else:
            angle = orbit_index * 10
            dotcolors = cm.rainbow(np.linspace(0, 1, num_orbits))[orbit_index]
        ax1.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        ax1.scatter(x, y, color=dotcolors, s=1)
        ax1.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax1.scatter(stn[3], stn[4], color=stn[6], s=5)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x [GEO]')
        ax1.set_ylabel('y [GEO]')
        ax1.set_xlim((lower_bound[0], upper_bound[0]))
        ax1.set_ylim((lower_bound[1], upper_bound[1]))
        ax1.set_title(f'Frame: {title2}', color='white')
        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                ax1.scatter(pos[0], pos[1], color=color, label=point)
                ax1.text(pos[0], pos[1], point, color=color)

        ax2.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        ax2.scatter(x, z, color=dotcolors, s=1)
        ax2.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax2.scatter(stn[3], stn[5], color=stn[6], s=5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x [GEO]')
        ax2.set_ylabel('z [GEO]')
        ax2.set_xlim((lower_bound[0], upper_bound[0]))
        ax2.set_ylim((lower_bound[2], upper_bound[2]))
        ax2.set_title(f'{title}', color='white')
        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                ax2.scatter(pos[0], pos[2], color=color, label=point)
                ax2.text(pos[0], pos[2], point, color=color)

        ax3.add_patch(plt.Circle((0, 0), stn[2], color='white', linestyle='dashed', fill=False))
        ax3.scatter(y, z, color=dotcolors, s=1)
        ax3.scatter(0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax3.scatter(stn[4], stn[5], color=stn[6], s=5)
        ax3.set_aspect('equal')
        ax3.set_xlabel('y [GEO]')
        ax3.set_ylabel('z [GEO]')
        ax3.set_xlim((lower_bound[1], upper_bound[1]))
        ax3.set_ylim((lower_bound[2], upper_bound[2]))
        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                print(pos)
                ax3.scatter(pos[1], pos[2], color=color, label=point)
                ax3.text(pos[1], pos[2], point, color=color)

        ax4.scatter3D(x, y, z, color=dotcolors, s=1)
        ax4.scatter3D(0, 0, 0, color=stn[0], s=stn[1])
        if xm is not False:
            ax4.scatter3D(stn[3], stn[4], stn[5], color=stn[6], s=5)
        ax4.set_xlim([lower_bound[0], upper_bound[0]])
        ax4.set_ylim([lower_bound[1], upper_bound[1]])
        ax4.set_zlim([lower_bound[2], upper_bound[2]])
        ax4.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax4.set_xlabel('x [GEO]')
        ax4.set_ylabel('y [GEO]')
        ax4.set_zlabel('z [GEO]')
        if 'lunar' in frame:
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            for (point, pos), color in zip(lagrange_points_lunar_frame().items(), colors):
                if 'axis' in frame:
                    pass
                else:
                    pos[0] = pos[0] - LD / RGEO
                ax4.scatter(pos[0], pos[1], pos[2], color=color, label=point)
                ax4.text(pos[0], pos[1], pos[2], point, color=color)
    
        return fig, ax1, ax2, ax3, ax4
    input_type = check_numpy_array(r)

    fig = plt.figure(dpi=100, figsize=figsize, facecolor='black')
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    if input_type == "numpy array":
        fig, ax1, ax2, ax3, ax4 = _make_scatter(fig, ax1, ax2, ax3, ax4, r=r, t=t, limits=limits, title=title, frame=frame)
    if input_type == "list of numpy array":
        num_orbits = np.shape(r)[0]
        for i, row in enumerate(r):
            fig, ax1, ax2, ax3, ax4 = _make_scatter(fig, ax1, ax2, ax3, ax4, r=row, t=t, limits=limits, title=title, orbit_index=i, num_orbits=num_orbits, frame=frame)

    # Set axis color to white
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        if i == 3:
            ax.tick_params(axis='z', colors='white')

    # Set text color to white
    for ax in [ax1, ax2, ax3, ax4]:
        for text in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]:
            text.set_color('white')
    
    #Save the plot
    fig.patch.set_facecolor('black')
    if show:
        plt.show()
    if save_path:
        save_plot(fig, save_path)
    return fig, [ax1, ax2, ax3, ax4]


def globe_plot(r, t, limits=False, title='', figsize=(7, 8), save_path=False, el=30, az=0, scale=1):
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

    dotcolors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.tick_params(axis='both', colors='white')
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5)
    ax.set_facecolor('black')  # Set plot background color to black
    ax.scatter(x, y, z, color=dotcolors, s=1)
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
    fig, ax = make_black(fig, ax)
    if save_path:
        save_plot(fig, save_path)
    return fig, ax


def koe_plot(r, v, t=Time("2025-01-01", scale='utc') + np.linspace(0, int(1 * 365.25), int(365.25 * 24)), elements=['a', 'e', 'i'], save_path=False, body='Earth'):
    if 'earth' in body.lower():
        orbital_elements = calculate_orbital_elements(r, v, mu_barycenter=EARTH_MU)
    else:
        orbital_elements = calculate_orbital_elements(r, v, mu_barycenter=MOON_MU)
    fig, ax1 = plt.subplots(dpi=100)
    fig.patch.set_facecolor('white')
    ax1.plot([], [], label='semi-major axis [GEO]', c='C0', linestyle='-')
    ax2 = ax1.twinx()
    make_white(fig, *[ax1, ax2])

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
    date_format(t, ax1)
    
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return fig, ax1


def koe_2dhist(stable_data, title="Initial orbital elements of\n1 year stable cislunar orbits", limits=[1, 50], bins=200, logscale=False, cmap='coolwarm', save_path=False):
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
    fig, ax = make_white(fig, ax)
    if save_path:
        save_plot(fig, save_path)
    return fig



def scatter2d(x, y, cs, xlabel='x', ylabel='y', title='', cbar_label='', dotsize=1, colorsMap='jet', colorscale='linear', colormin=False, colormax=False, save_path=False):
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
    fig, ax = make_black(fig, ax)
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return


def scatter3d(x, y=None, z=None, cs=None, xlabel='x', ylabel='y', zlabel='z', cbar_label='', dotsize=1, colorsMap='jet', title='', save_path=False):
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
    fig, ax = make_black(fig, ax)
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return fig, ax


def dotcolors_scaled(num_colors):
    return cm.rainbow(np.linspace(0, 1, num_colors))


# Make a plot of multiple cislunar orbit in GCRF frame.
def orbit_divergence_plot(rs, r_moon=[], t=False, limits=False, title='', save_path=False):
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
        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Creating plot
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(xm, ym, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('y [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], y[0], '$\leftarrow$ start')
        plt.text(x[-1], y[-1], '$\leftarrow$ end')

        plt.subplot(1, 3, 2)
        plt.scatter(x, z, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(xm, zm, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('x [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(x[0], z[0], '$\leftarrow$ start')
        plt.text(x[-1], z[-1], '$\leftarrow$ end')
        plt.title(f'{title}')

        plt.subplot(1, 3, 3)
        plt.scatter(y, z, color=dotcolors, s=1)
        plt.scatter(0, 0, color="blue", s=50)
        plt.scatter(ym, zm, color="grey", s=5)
        plt.axis('scaled')
        plt.xlabel('y [GEO]')
        plt.ylabel('z [GEO]')
        plt.xlim((-limits, limits))
        plt.ylim((-limits, limits))
        plt.text(y[0], z[0], '$\leftarrow$ start')
        plt.text(y[-1], z[-1], '$\leftarrow$ end')
    plt.tight_layout()
    plt.show(block=False)
    if save_path:
        save_plot(fig, save_path)
    return


def make_white(fig, *axes):
    fig.patch.set_facecolor('white')

    for ax in axes:
        ax.set_facecolor('white')
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
            item.set_color('black')

    return fig, axes


def make_black(fig, *axes):
    fig.patch.set_facecolor('black')

    for ax in axes:
        ax.set_facecolor('black')
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
            item.set_color('white')

    return fig, axes


def draw_dashed_circle(ax, normal_vector, radius, dashes, dash_length=0.1, label='Dashed Circle'):
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
def date_format(time_array, ax):
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


save_plot_to_pdf_call_count = 0


def save_plot_to_pdf(figure, pdf_path):
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
    Save a Python figure as a PNG image.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to be saved.
        save_path (str): The file path where the PNG image will be saved.

    Returns:
        None
    """
    if save_path.lower().endswith('.pdf'):
        save_plot_to_pdf(figure, save_path)
        return
    try:
        base_name, extension = os.path.splitext(save_path)
        if extension.lower() != '.png':
            save_path = base_name + '.png'
        # Save the figure as a PNG image
        figure.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(figure)  # Close the figure to release resources
        # print(f"Figure saved at: {save_path}")
    except Exception as e:
        print(f"Error occurred while saving the figure: {e}")


def write_gif(gif_name, frames, fps=30):
    import imageio
    print(f'Writing gif: {gif_name}')
    with imageio.get_writer(gif_name, mode='I', duration=1 / fps) as writer:
        for i, filename in enumerate(frames):
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f'Wrote {gif_name}')
    return

from .body import get_body
from .compute import groundTrack
from .constants import RGEO, EARTH_RADIUS
from .utils import find_file, Time, norm, gcrf_to_ecef, gcrf_to_lunar, gcrf_to_stationary_lunar

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


def groundTrackPlot(r, time, ground_stations=None):
    """
    Parameters
    ----------
    r : (3,) array_like - Orbit positions in meters.
    times: (n,) array_like - array of Astropy Time objects or time in gps seconds.

    optional - ground_stations: (n,2) array of of ground station (lat,lon) in degrees
    """
    lon, lat, height = groundTrack(r, time)

    plt.figure(figsize=(15, 12))
    plt.imshow(load_earth_file(), extent=[-180, 180, -90, 90])
    plt.plot(np.rad2deg(lon), np.rad2deg(lat))
    if ground_stations is not None:
        for ground_station in ground_stations:
            plt.scatter(ground_station[1], ground_station[0], s=50, color='Red')
    plt.ylim(-90, 90)
    plt.xlim(-180, 180)
    plt.show()


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


def _make_scatter(x, y, z, xm=False, ym=False, zm=False, limits=False, title='', figsize=(7, 7), orbit_index='', save_path=False, plot_type=False):
    # DETERMINE WHAT TYPE OF SCATTER THIS IS
    if limits is False:
        limits = np.nanmax(np.abs([x, y, z])) * 1.2

    dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))
    if np.size(xm) > 1:
        gradient_colors = cm.Greys(np.linspace(0.5, 1, len(xm)))
    else:
        gradient_colors = "grey"

    # Central dot color, central dot size, dashed line radius
    plot_settings = {
        "gcrf": ("blue", 50, 1),
        "ecef": ("blue", 50, 1),
        "lunar": ("blue", 50, 1),
        "stationary_lunar": ("grey", 25, 1.3)
    }

    # Check if the plot_type is in the dictionary, and set central_dot accordingly
    if plot_type in plot_settings:
        stn = plot_settings[plot_type]
    else:
        raise ValueError("Unknown plot type provided. Accepted: gcrf, ecef, lunar, stationary_lunar")

    # Creating plot
    plt.rcParams.update({'font.size': 9, 'figure.facecolor': 'w'})
    fig = plt.figure(dpi=100, figsize=figsize)

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter3D(x, y, z, color=dotcolors, s=1)
    ax1.scatter3D(0, 0, 0, color=stn[0], s=stn[1])
    if xm is not False:
        ax1.scatter3D(xm, ym, zm, color=gradient_colors, s=5)
    ax1.view_init(elev=30, azim=60)
    ax1.set_xlim([-limits, limits])
    ax1.set_ylim([-limits, limits])
    ax1.set_zlim([-limits, limits])
    ax1.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
    ax1.set_xlabel('x [GEO]')
    ax1.set_ylabel('y [GEO]')
    ax1.set_zlabel('z [GEO]')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.add_patch(plt.Circle((0, 0), stn[2], color='black', linestyle='dashed', fill=False))
    ax2.scatter(x, y, color=dotcolors, s=1)
    ax2.scatter(0, 0, color=stn[0], s=stn[1])
    if xm is not False:
        ax2.scatter(xm, ym, color=gradient_colors, s=5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x [GEO]')
    ax2.set_ylabel('y [GEO]')
    ax2.set_xlim((-limits, limits))
    ax2.set_ylim((-limits, limits))
    ax2.text(x[0], y[0], f'← start {orbit_index}')
    ax2.text(x[-1], y[-1], f'← end {orbit_index}')
    ax2.set_title(f'{title}')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.add_patch(plt.Circle((0, 0), stn[2], color='black', linestyle='dashed', fill=False))
    ax3.scatter(x, z, color=dotcolors, s=1)
    ax3.scatter(0, 0, color=stn[0], s=stn[1])
    if xm is not False:
        ax3.scatter(xm, zm, color=gradient_colors, s=5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('x [GEO]')
    ax3.set_ylabel('z [GEO]')
    ax3.set_xlim((-limits, limits))
    ax3.set_ylim((-limits, limits))
    ax3.text(x[0], z[0], f'← start {orbit_index}')
    ax3.text(x[-1], z[-1], f'← end {orbit_index}')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.add_patch(plt.Circle((0, 0), stn[2], color='black', linestyle='dashed', fill=False))
    ax4.scatter(y, z, color=dotcolors, s=1)
    ax4.scatter(0, 0, color=stn[0], s=stn[1])
    if xm is not False:
        ax4.scatter(ym, zm, color=gradient_colors, s=5)
    ax4.set_aspect('equal')
    ax4.set_xlabel('y [GEO]')
    ax4.set_ylabel('z [GEO]')
    ax4.set_xlim((-limits, limits))
    ax4.set_ylim((-limits, limits))
    ax4.text(y[0], z[0], f'← start {orbit_index}')
    ax4.text(y[-1], z[-1], f'← end {orbit_index}')

    plt.tight_layout(pad=2.0, h_pad=2.0)
    if save_path:
        if save_path.lower().endswith('.png'):
            save_plot_to_png(fig, save_path)
        else:
            save_plot_to_pdf(fig, save_path)
    return fig


def gcrf_plot(r, times=[], limits=False, title='', save_path=False, figsize=(7, 7)):
    """
    Parameters
    ----------
    r : (n,3) or array of [(n,3), ..., (n,3)] array_like
        Position of orbiting object(s) in meters.
    times: optional - times when r was calculated.
    limits: optional - x and y limits of the plot
    title: optional - title of the plot
    """

    if np.size(times) < 1:
        r_moon = np.atleast_2d(get_body("moon").position(Time("2000-1-1")))
    else:
        r_moon = get_body("moon").position(times).T
    xm = r_moon[:, 0] / RGEO
    ym = r_moon[:, 1] / RGEO
    zm = r_moon[:, 2] / RGEO
    input_type = check_numpy_array(r)
    if input_type == "numpy array":
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        fig = _make_scatter(x=x, y=y, z=z, xm=xm, ym=ym, zm=zm, limits=limits, title=title, figsize=figsize, save_path=save_path, plot_type="gcrf")
    if input_type == "list of numpy array":
        limits_plot = 0
        for i, row in enumerate(r):
            if limits is False and limits_plot < np.nanmax(norm(row) / RGEO) * 1.2:
                limits_plot = np.nanmax(norm(row) / RGEO) * 1.2
            else:
                limits_plot = limits
            x = row[:, 0] / RGEO
            y = row[:, 1] / RGEO
            z = row[:, 2] / RGEO
            fig = _make_scatter(x=x, y=y, z=z, xm=xm, ym=ym, zm=zm, limits=limits_plot, title=title, orbit_index=i, figsize=figsize, save_path=save_path, plot_type="gcrf")
    return fig


def ecef_plot(r, times, limits=False, title='', save_path=False, figsize=(7, 7)):
    """
    Parameters
    ----------
    r : (n,3) or array of [(n,3), ..., (n,3)] array_like
        Position of orbiting object(s) in meters.
    times: times when r was calculated.
    limits: optional - x and y limits of the plot
    title: optional - title of the plot
    """

    input_type = check_numpy_array(r)
    r = gcrf_to_ecef(r, times)
    if input_type == "numpy array":
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        fig = _make_scatter(x=x, y=y, z=z, limits=limits, title=title, save_path=save_path, plot_type="ecef")
    if input_type == "list of numpy array":
        limits_plot = 0
        for i, row in enumerate(r):
            if limits is False and limits_plot < np.nanmax(norm(row) / RGEO) * 1.2:
                limits_plot = np.nanmax(norm(row) / RGEO) * 1.2
            else:
                limits_plot = limits
            x = row[:, 0] / RGEO
            y = row[:, 1] / RGEO
            z = row[:, 2] / RGEO
            fig = _make_scatter(x=x, y=y, z=z, limits_plot=limits_plot, title=title, orbit_index=i, save_path=save_path, plot_type="ecef")
    return fig


def lunar_plot(r, times, limits=False, title='', save_path=False, figsize=(7, 7)):
    """
    Parameters
    ----------
    r : (n,3) or array of [(n,3), ..., (n,3)] array_like
        Position of orbiting object(s) in meters.
    times: array of Astropy time objects
    limits: optional - x and y limits of the plot
    title: optional - title of the plot
    """

    input_type = check_numpy_array(r)
    r = gcrf_to_lunar(r, times)
    r_moon = gcrf_to_lunar(get_body("moon").position(times).T, times)
    xm = r_moon[:, 0] / RGEO
    ym = r_moon[:, 1] / RGEO
    zm = r_moon[:, 2] / RGEO
    if input_type == "numpy array":
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        fig = _make_scatter(x, y, z, xm, ym, zm, limits=limits, title=title, save_path=save_path, plot_type="lunar")
    if input_type == "list of numpy array":
        limits_plot = 0
        for i, row in enumerate(r):
            if limits is False and limits_plot < np.nanmax(norm(row) / RGEO) * 1.2:
                limits_plot = np.nanmax(norm(row) / RGEO) * 1.2
            else:
                limits_plot = limits
            x = row[:, 0] / RGEO
            y = row[:, 1] / RGEO
            z = row[:, 2] / RGEO
            fig = _make_scatter(x, y, z, xm, ym, zm, limits=limits_plot, title=title, orbit_index=i, save_path=save_path, plot_type="lunar")
    return fig


def lunar_stationary_plot(r, times, limits=False, title='', save_path=False, figsize=(7, 7)):
    """
    Parameters
    ----------
    r : (n,3) or array of [(n,3), ..., (n,3)] array_like
        Position of orbiting object(s) in meters.
    times: array of Astropy time objects
    limits: optional - x and y limits of the plot
    title: optional - title of the plot
    """
    input_type = check_numpy_array(r)
    r = gcrf_to_stationary_lunar(r, times)
    if input_type == "numpy array":
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        fig = _make_scatter(x, y, z, limits=limits, title=title, save_path=save_path, plot_type="stationary_lunar")
    if input_type == "list of numpy array":
        limits_plot = 0
        for i, row in enumerate(r):
            if limits is False and limits_plot < np.nanmax(norm(row) / RGEO) * 1.2:
                limits_plot = np.nanmax(norm(row) / RGEO) * 1.2
            else:
                limits_plot = limits
            x = row[:, 0] / RGEO
            y = row[:, 1] / RGEO
            z = row[:, 2] / RGEO
            fig = _make_scatter(x, y, z, limts=limits_plot, title=title, orbit_index=i, save_path=save_path, plot_type="stationary_lunar")
    return fig


def tracking_plot(r, times, ground_stations=None, limits=False, title='', figsize=(7, 8), save_path=False, elev=30, azim=90, scale=5):
    """
    Create a 3D tracking plot of satellite positions over time on Earth's surface.

    Parameters
    ----------
    r : numpy.ndarray or list of numpy.ndarray
        Satellite positions in GCRF coordinates. If a single numpy array, it represents the satellite's position vector over time. If a list of numpy arrays, it represents multiple satellite position vectors.

    times : numpy.ndarray
        Timestamps corresponding to the satellite positions.

    ground_stations : list of tuples, optional
        List of ground stations represented as (latitude, longitude) pairs. Default is None.

    limits : float or bool, optional
        The plot limits for x, y, and z axes. If a float, it sets the limits for all axes. If False, the limits are automatically determined based on the data. Default is False.

    title : str, optional
        Title for the plot. Default is an empty string.

    figsize : tuple, optional
        Figure size in inches (width, height). Default is (7, 8).

    save_path : str or bool, optional
        Path to save the plot as an image or PDF. If False, the plot is not saved. Default is False.

    elev : float, optional
        Elevation angle for the 3D plot view. Default is 30 degrees.

    azim : float, optional
        Azimuthal angle for the 3D plot view. Default is 90 degrees.

    scale : int, optional
        Scaling factor for the Earth's image. Default is 5.

    frame : str, optional
        Coordinate frame for the satellite positions, "gcrf" or "ecef". Default is "gcrf".

    Returns
    -------
    matplotlib.figure.Figure
        The created tracking plot figure.

    Notes
    -----
    - The function supports plotting the positions of one or multiple satellites over time.
    - Ground station locations can be optionally displayed on the plot.
    - The limits parameter can be set to specify the plot's axis limits or automatically determined if set to False.
    - The frame parameter determines the coordinate frame for the satellite positions, "gcrf" (default) or "ecef".

    Example Usage
    -------------
    - Single satellite tracking plot:
      tracking_plot(r_satellite, times, ground_stations=[(40, -75)], title="Satellite Tracking")

    - Multiple satellite tracking plot:
      tracking_plot([r_satellite_1, r_satellite_2], times, title="Multiple Satellite Tracking")

    - Save the plot as a PNG image:
      tracking_plot(r_satellite, times, save_path="satellite_tracking.png")

    - Customize the plot view:
      tracking_plot(r_satellite, times, elev=45, azim=120)

    - Set custom axis limits:
      tracking_plot(r_satellite, times, limits=500)
    """
    def _make_plot(r, times, ground_stations, limits, title, figsize, save_path, elev, azim, scale, orbit_index=''):
        lon, lat, height = groundTrack(r, times)
        r = gcrf_to_ecef(r, times)
        x = r[:, 0] / RGEO
        y = r[:, 1] / RGEO
        z = r[:, 2] / RGEO
        if limits is False:
            limits = np.nanmax(np.abs([x, y, z])) * 1.2

        dotcolors = cm.rainbow(np.linspace(0, 1, len(x)))

        # Creating plot
        plt.rcParams.update({'font.size': 9, 'figure.facecolor': 'w'})
        fig = plt.figure(dpi=100, figsize=figsize)
        earth_png = PILImage.open(find_file("earth", ext=".png"))
        earth_png = earth_png.resize((5400 // scale, 2700 // scale))
        ax_gt = fig.add_subplot(2, 2, (3, 4))
        ax_gt.imshow(earth_png, extent=[-180, 180, -90, 90])
        ax_gt.plot(np.rad2deg(lon), np.rad2deg(lat))
        if ground_stations is not None:
            for ground_station in ground_stations:
                ax_gt.scatter(ground_station[1], ground_station[0], s=50, color='Red')
        ax_gt.set_ylim(-90, 90)
        ax_gt.set_xlim(-180, 180)
        ax_gt.set_xlabel('longitude [degrees]')
        ax_gt.set_ylabel('latitude [degrees]')

        bm = np.array(earth_png.resize([int(d) for d in earth_png.size])) / 256.
        lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
        lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
        mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * 0.15126911409197252
        mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * 0.15126911409197252
        mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * 0.15126911409197252

        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_3d.scatter3D(x, y, z, color=dotcolors, s=1)
        ax_3d.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)
        ax_3d.view_init(elev=elev, azim=azim)
        ax_3d.set_xlim([-limits, limits])
        ax_3d.set_ylim([-limits, limits])
        ax_3d.set_zlim([-limits, limits])
        ax_3d.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax_3d.set_xlabel('x [GEO]')
        ax_3d.set_ylabel('y [GEO]')
        ax_3d.set_zlabel('z [GEO]')

        ax_3d_r = fig.add_subplot(2, 2, 2, projection='3d')
        ax_3d_r.scatter3D(x, y, z, color=dotcolors, s=1)
        ax_3d_r.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)
        ax_3d_r.view_init(elev=elev, azim=180 + azim)
        ax_3d_r.set_xlim([-limits, limits])
        ax_3d_r.set_ylim([-limits, limits])
        ax_3d_r.set_zlim([-limits, limits])
        ax_3d_r.set_aspect('equal')  # aspect ratio is 1:1:1 in data space
        ax_3d_r.set_xlabel('x [GEO]')
        ax_3d_r.set_ylabel('y [GEO]')
        ax_3d_r.set_zlabel('z [GEO]')

        plt.tight_layout()
        if save_path:
            if save_path.lower().endswith('.png'):
                save_plot_to_png(fig, save_path)
            else:
                save_plot_to_pdf(fig, save_path)
        return fig

    input_type = check_numpy_array(r)
    if input_type == "numpy array":
        fig = _make_plot(
            r, times, ground_stations=ground_stations,
            limits=limits, title=title, figsize=figsize,
            save_path=save_path, elev=elev, azim=azim,
            scale=scale)
    if input_type == "list of numpy array":
        limits_plot = 0
        for i, row in enumerate(r):
            if limits is False and limits_plot < np.nanmax(norm(row) / RGEO) * 1.2:
                limits_plot = np.nanmax(norm(row) / RGEO) * 1.2
            else:
                limits_plot = limits
            fig = _make_plot(
                row, times, ground_stations=ground_stations,
                limits=limits_plot, title=title, figsize=figsize,
                elev=elev, azim=azim, save_path=save_path,
                scale=scale, orbit_index=i
            )
    return fig


def plotRVTrace(chain, lnprob, lnprior, start=0, fig=None):
    """Plot rv MCMC traces.

    Parameters
    ----------
    chain : array_like (nStep, nChain, 6)
        MCMC sample chain.
    lnprob : array_like (nStep, nChain)
        Log posterior probability of sample chain.
    lnprior : array_like (nStep, nChain)
        Log posterior probability of sample chain.
    start : int, optional
        Optional starting index for plot
    fig : matplotlib Figure, optional
        Where to make plots.  Will be created if not explicitly supplied.
    """
    import matplotlib.pyplot as plt
    if fig is None:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8.5, 11))
        axes = axes.ravel()
    else:
        try:
            axes = np.array(fig.axes).reshape((4, 2)).ravel()
        except Exception:
            raise ValueError("Provided figure has {} axes, but plot requires "
                             "dimensions K={}".format(np.array(fig.axes).shape, (4, 2)))

    labels = ["x (km)", "y (km)", "z (km)",
              "vx (km/s)", "vy (km/s)", "vz (km/s)"]
    xs = np.arange(start, chain.shape[0])
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(xs, chain[start:, :, i] / 1000, alpha=0.1, color='k')
        ax.set_ylabel(label)
    axes[-2].plot(xs, lnprob[start:], alpha=0.1, color='k')
    axes[-2].set_ylabel("lnprob")
    axes[-1].plot(xs, lnprior[start:], alpha=0.1, color='k')
    axes[-1].set_ylabel("lnprior")
    return fig


def plotRVCorner(chain, lnprob, lnprior, start=0, **kwargs):
    """Make rv samples corner plot.

    Parameters
    ----------
    chain : array_like (nStep, nChain, 6)
        MCMC sample chain.
    lnprob : array_like (nStep, nChain)
        Log posterior probability of sample chain.
    lnprior : array_like (nStep, nChain)
        Log prior probability of sample chain.
    start : int, optional
        Optional starting index for selecting samples
    kwargs : optional kwargs to pass on to corner.corner

    Returns
    -------
    fig : matplotlib.Figure
        Figure instance from corner.corner.
    """
    import corner

    chain = chain[start:] / 1000
    lnprob = np.atleast_3d(lnprob[start:])
    lnprior = np.atleast_3d(lnprior[start:])
    data = np.concatenate([chain, lnprob, lnprior], axis=2)
    data = data.reshape((-1, 8))
    labels = ["x (km)", "y (km)", "z (km)",
              "vx (km/s)", "vy (km/s)", "vz (km/s)",
              "lnprob", "lnprior"]

    return corner.corner(data, labels=labels, **kwargs)


######################################################################
# Formatting x axis
######################################################################
def format_xaxis_decimal_year(time_array, ax):
    n = 5  # Number of nearly evenly spaced points to select
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
        step = int(len(time_array) / (n - 1))
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
    ######################################################################
    # Save figures appended to a pdf.
    ######################################################################
    import io
    import os
    import re
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


def save_plot_to_png(figure, save_path, dpi=200):
    """
    Save a Python figure as a PNG image.

    Parameters:
        figure (matplotlib.figure.Figure): The figure object to be saved.
        save_path (str): The file path where the PNG image will be saved.

    Returns:
        None
    """
    try:
        # Save the figure as a PNG image
        figure.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(figure)  # Close the figure to release resources
        print(f"Figure saved at: {save_path}")
    except Exception as e:
        print(f"Error occurred while saving the figure: {e}")

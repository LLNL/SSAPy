# flake8: noqa: E501

"""
Functions to simplify certain aspects of SSAPy. 
For when you want a quick answer and do not need want to use high fidelity defaults.
"""

from .utils import get_times, points_on_circle
from . import Orbit, rv
from .accel import AccelKepler, AccelSolRad, AccelEarthRad, AccelDrag
from .body import get_body
from .gravity import AccelHarmonic, AccelThirdBody
from .propagator import RK78Propagator

from astropy.time import Time
import numpy as np
from typing import Union, List, Tuple


def keplerian_prop(integration_timestep: float = 10) -> RK78Propagator:
    """
    Create and return an RK78Propagator for a Keplerian orbit.

    This propagator uses only the Keplerian acceleration for a two-body problem.

    Parameters:
    ----------
    integration_timestep : float, optional
        The time step for the RK78Propagator integration (default is 10 seconds).

    Returns:
    -------
    RK78Propagator
        An instance of RK78Propagator configured with Keplerian acceleration.
    """
    return RK78Propagator(AccelKepler(), h=integration_timestep)


accel_3_cache = None
def threebody_prop(integration_timestep: float = 10) -> RK78Propagator:
    """
    Create and return an RK78Propagator with a set of accelerations for a three-body problem.

    The three bodies considered are Earth (Keplerian effect), Moon, and the Earth itself.

    Parameters:
    ----------
    integration_timestep : float, optional
        The time step for the RK78Propagator integration (default is 10 seconds).

    Returns:
    -------
    RK78Propagator
        An instance of RK78Propagator configured with the three-body accelerations.
    """
    global accel_3_cache
    if accel_3_cache is None:
        accel_3_cache = AccelKepler() + AccelThirdBody(get_body("moon"))
    return RK78Propagator(accel_3_cache, h=integration_timestep)


accel_4_cache = None
def fourbody_prop(integration_timestep: float = 10) -> RK78Propagator:
    """
    Create and return an RK78Propagator with a set of accelerations for a four-body problem.

    The four bodies considered are Earth (Keplerian effect), Moon, Sun, and the Earth itself.

    Parameters:
    ----------
    integration_timestep : float, optional
        The time step for the RK78Propagator integration (default is 10 seconds).

    Returns:
    -------
    RK78Propagator
        An instance of RK78Propagator configured with the four-body accelerations.
    """
    global accel_4_cache
    if accel_4_cache is None:
        accel_4_cache = AccelKepler() + AccelThirdBody(get_body("moon")) + AccelThirdBody(get_body("Sun"))
    return RK78Propagator(accel_4_cache, h=integration_timestep)


accel_best_cache = None
def best_prop(integration_timestep=10, kwargs=dict(mass=250, area=.022, CD=2.3, CR=1.3)):
    """
    Create and return an RK78Propagator with a comprehensive set of accelerations.

    Parameters:
    ----------
    integration_timestep : float, optional
        The time step for the RK78Propagator integration (default is 10 seconds).
    kwargs : dict, optional
        Dictionary of parameters for non-conservative forces (mass, area, CD, CR).
        If not provided, defaults are used.

    Returns:
    -------
    RK78Propagator
        An instance of RK78Propagator configured with cached accelerations.
    """
    global accel_best_cache
    if accel_best_cache is None:
        aEarth = AccelKepler() + AccelHarmonic(get_body("Earth", model="EGM2008"), 140, 140)
        aMoon = AccelThirdBody(get_body("moon")) + AccelHarmonic(get_body("moon"), 20, 20)
        aSun = AccelThirdBody(get_body("Sun"))
        aMercury = AccelThirdBody(get_body("Mercury"))
        aVenus = AccelThirdBody(get_body("Venus"))
        aMars = AccelThirdBody(get_body("Mars"))
        aJupiter = AccelThirdBody(get_body("Jupiter"))
        aSaturn = AccelThirdBody(get_body("Saturn"))
        aUranus = AccelThirdBody(get_body("Uranus"))
        aNeptune = AccelThirdBody(get_body("Neptune"))
        nonConservative = AccelSolRad(**kwargs) + AccelEarthRad(**kwargs) + AccelDrag(**kwargs)
        planets = aMercury + aVenus + aMars + aJupiter + aSaturn + aUranus + aNeptune
        accel_best_cache = aEarth + aMoon + aSun + planets + nonConservative
    return RK78Propagator(accel_best_cache, h=integration_timestep)


def ssapy_kwargs(mass=250, area=0.022, CD=2.3, CR=1.3):
    """
    Generate a dictionary of default parameters for a space object used in simulations.

    Parameters:
    ----------
    mass : float, optional
        Mass of the object in kilograms (default is 250 kg).
    area : float, optional
        Cross-sectional area of the object in square meters (default is 0.022 m^2).
    CD : float, optional
        Drag coefficient of the object (default is 2.3).
    CR : float, optional
        Radiation pressure coefficient of the object (default is 1.3).

    Returns:
    -------
    dict
        A dictionary containing the parameters for the space object.
    """
    # Asteroid parameters
    kwargs = dict(
        mass=mass,  # [kg]
        area=area,  # [m^2]
        CD=CD,  # Drag coefficient
        CR=CR,  # Radiation pressure coefficient
    )
    return kwargs


def ssapy_prop(integration_timestep=60, propkw=ssapy_kwargs()):
    """
    Setup and return an RK78 propagator with specified accelerations and radiation pressure effects.

    Parameters:
    ----------
    integration_timestep : int
        Time step for the numerical integration (in seconds).
    propkw : dict, optional
        Keyword arguments for radiation pressure accelerations. If None, default arguments are used.

    Returns:
    -------
    RK78Propagator
        An RK78 propagator configured with the specified accelerations and time step.
    """
    # Accelerations - pass a body object or string of body name.
    moon = get_body("moon")
    sun = get_body("Sun")
    Mercury = get_body("Mercury")
    Venus = get_body("Venus")
    Earth = get_body("Earth", model="EGM2008")
    Mars = get_body("Mars")
    Jupiter = get_body("Jupiter")
    Saturn = get_body("Saturn")
    Uranus = get_body("Uranus")
    Neptune = get_body("Neptune")
    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
    aSun = AccelThirdBody(sun)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)
    aSolRad = AccelSolRad(**propkw)
    aEarthRad = AccelEarthRad(**propkw)
    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad
    # Build propagator
    prop = RK78Propagator(accel, h=integration_timestep)
    return prop


# Uses the current best propagator and acceleration models in ssapy
def ssapy_orbit(orbit=None, a=None, e=0, i=0, pa=0, raan=0, ta=0, r=None, v=None, duration=(30, 'day'), freq=(1, 'hr'), t0=Time("2025-01-01", scale='utc'), t=None, prop=ssapy_prop()):
    """
    Compute the orbit of a spacecraft given either Keplerian elements or position and velocity vectors.

    Parameters:
    - orbit (Orbit, optional): An Orbit object if you already have an orbit defined.
    - a (float, optional): Semi-major axis of the orbit in meters.
    - e (float, optional): Eccentricity of the orbit (default is 0, i.e., circular orbit).
    - i (float, optional): Inclination of the orbit in degrees.
    - pa (float, optional): Argument of perigee in degrees.
    - raan (float, optional): Right ascension of the ascending node in degrees.
    - ta (float, optional): True anomaly in degrees.
    - r (array-like, optional): Position vector in meters.
    - v (array-like, optional): Velocity vector in meters per second.
    - duration (tuple, optional): Duration of the simulation as a tuple (value, unit), where unit is 'day', 'hour', etc. Default is 30 days.
    - freq (tuple, optional): Frequency of the output as a tuple (value, unit), where unit is 'day', 'hour', etc. Default is 1 hour.
    - t0 (str, optional): Start date of the simulation in 'YYYY-MM-DD' format. Default is "2025-01-01".
    - t (array-like, optional): Specific times at which to compute the orbit. If None, times will be generated based on duration and frequency.
    - prop (function, optional): A function to compute the perturbation effects. Default is `ssapy_prop()`.

    Returns:
    - r (array-like): Position vectors of the spacecraft at the specified times.
    - v (array-like): Velocity vectors of the spacecraft at the specified times.
    - t (array-like): Times at which the orbit was computed. Returned only if `t` was None.

    Raises:
    - ValueError: If neither Keplerian elements nor position and velocity vectors are provided.
    - RuntimeError or ValueError: If an error occurs during computation.
    """
    t0 = Time(t0, scale='utc')
    if t is None:
        time_is_None = True
        t = get_times(duration=duration, freq=freq, t0=t0)
    else:
        t0 = t[0]
        time_is_None = False

    if orbit is not None:
        pass
    elif a is not None:
        kElements = [a, e, i, pa, raan, ta]
        orbit = Orbit.fromKeplerianElements(*kElements, t0)
    elif r is not None and v is not None:
        orbit = Orbit(r, v, t0)
    else:
        raise ValueError("Either Keplerian elements (a, e, i, pa, raan, ta) or position and velocity vectors (r, v) must be provided.")

    try:
        r, v = rv(orbit, t, prop)
        if time_is_None:
            return r, v, t
        else:
            return r, v
    except (RuntimeError, ValueError) as err:
        print(err)
        return np.nan, np.nan, np.nan


def get_similar_orbits(r0, v0, rad=1e5, num_orbits=4, duration=(90, 'days'), freq=(1, 'hour'), start_date="2025-1-1", mass=250):
    """
    Generate similar orbits by varying the initial position.

    Parameters:
    ----------
    r0 : array_like
        Initial position vector of shape (3,).
    v0 : array_like
        Initial velocity vector of shape (3,).
    rad : float
        Radius of the circle around the initial position to generate similar orbits.
    num_orbits : int
        Number of similar orbits to generate.
    duration : tuple
        Duration of the orbit simulation.
    freq : tuple
        Frequency of output data.
    start_date : str
        Start date for the simulation.
    mass : float
        Mass of the satellite.

    Returns:
    -------
    trajectories : ndarray
        Stacked array of shape (3, n_times, num_orbits) containing the trajectories.
    """
    r0 = np.reshape(r0, (1, 3))
    v0 = np.reshape(v0, (1, 3))
    print(r0, v0)
    for idx, point in enumerate(points_on_circle(r0, v0, rad=rad, num_points=num_orbits)):
        # Calculate entire satellite trajectory
        r, v = ssapy_orbit(r=point, v=v0, duration=duration, freq=freq, start_date=start_date, integration_timestep=10, mass=mass, area=mass / 19000 + 0.01, CD=2.3, CR=1.3)
        if idx == 0:
            trajectories = np.concatenate((r0, v0), axis=1)[:len(r)]
        rv = np.concatenate((r, v), axis=1)
        trajectories = np.dstack((trajectories, rv))
    return trajectories

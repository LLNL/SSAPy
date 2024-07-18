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


def keplerian_prop(integration_timestep=10):
    return RK78Propagator(AccelKepler(), h=integration_timestep)


accel_3_cache = None
def threebody_prop(integration_timestep=10):
    global accel_3_cache
    if accel_3_cache is None:
        accel_3_cache = AccelKepler() + AccelThirdBody(get_body("moon"))
    return RK78Propagator(accel_3_cache, h=integration_timestep)


accel_4_cache = None
def fourbody_prop(integration_timestep=10):
    global accel_4_cache
    if accel_4_cache is None:
        accel_4_cache = AccelKepler() + AccelThirdBody(get_body("moon")) + AccelThirdBody(get_body("Sun"))
    return RK78Propagator(accel_4_cache, h=integration_timestep)


accel_best_cache = None
def best_prop(integration_timestep=10, kwargs=dict(mass=250, area=.022, CD=2.3, CR=1.3)):
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
    # Asteroid parameters
    kwargs = dict(
        mass=mass,  # [kg]
        area=area,  # [m^2]
        CD=CD,  # Drag coefficient
        CR=CR,  # Radiation pressure coefficient
    )
    return kwargs


def ssapy_prop(integration_timestep=60, propkw=ssapy_kwargs()):
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
def ssapy_orbit(orbit=None, a=None, e=0, i=0, pa=0, raan=0, ta=0, r=None, v=None, duration=(30, 'day'), freq=(1, 'hr'), start_date="2025-01-01", t=None, integration_timestep=10, mass=250, area=0.022, CD=2.3, CR=1.3, prop=ssapy_prop()):
    # Everything is in SI units, except time.
    # density #kg/m^3 --> density
    t0 = Time(start_date, scale='utc')
    if t is None:
        time_is_None = True
        t = get_times(duration=duration, freq=freq, t=t)
    else:
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


# Generate orbits near stable orbit.
def get_similar_orbits(r0, v0, rad=1e5, num_orbits=4, duration=(90, 'days'), freq=(1, 'hour'), start_date="2025-1-1", mass=250):
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

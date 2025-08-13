import os

def _get_datadir():
    """Get data directory, downloading data if needed (lazy loading)."""
    from .data_utils import get_data_dir
    return get_data_dir()

# Make datadir a property that downloads data when first accessed
class _DataDir:
    def __init__(self):
        self._path = None
    
    def __str__(self):
        if self._path is None:
            self._path = _get_datadir()
        return self._path
    
    def __fspath__(self):
        return str(self)

datadir = _DataDir()

from . import _ssapy
from .orbit import Orbit, EarthObserver, OrbitalObserver
from .propagator import (
    KeplerianPropagator, SeriesPropagator, RK4Propagator, SGP4Propagator,
    SciPyPropagator, RK78Propagator, RK8Propagator
)
from .compute import rv, dircos, radec, altaz, quickAltAz, radecRate, groundTrack
from .accel import Accel, AccelKepler, AccelSum, AccelEarthRad, AccelSolRad, AccelDrag, AccelConstNTW
from .linker import ModelSelectorParams, BinarySelectorParams, Linker
from .orbit_solver import TwoPosOrbitSolver, GaussTwoPosOrbitSolver
from .orbit_solver import DanchickTwoPosOrbitSolver, SheferTwoPosOrbitSolver
from .orbit_solver import ThreeAngleOrbitSolver
from .particles import Particles
from .rvsampler import GEOProjectionInitializer, DistanceProjectionInitializer
from .rvsampler import circular_guess
from .rvsampler import GaussianRVInitializer, DirectInitializer
from .rvsampler import RVProbability, EmceeSampler, MVNormalProposal
from .rvsampler import RVSigmaProposal, MHSampler, LMOptimizer
from .ellipsoid import Ellipsoid
from .body import (
    EarthOrientation, MoonOrientation, MoonPosition, Body, get_body
)
from .gravity import HarmonicCoefficients, AccelThirdBody, AccelHarmonic

from . import constants
from . import plotUtils
from . import io
from . import utils
from . import simple

from astropy.time import Time, TimeDelta
import astropy.units as u
from datetime import timedelta
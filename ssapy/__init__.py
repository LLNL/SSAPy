import os

class _DataDir:
    def __str__(self):
        from .data_loader import get_data_dir
        data_path = get_data_dir()
        return str(data_path) if data_path else ""
    
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
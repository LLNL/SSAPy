"""
Tools for orbital dynamics, satellite tracking, and celestial mechanics.
"""

import numpy as np
from astropy.time import Time
import astropy.units as u

from .body import get_body
from .constants import RGEO, LD, EARTH_RADIUS, EARTH_MU, MOON_RADIUS, MOON_MU
from .propagator import KeplerianPropagator
from .utils import (
    norm, normed, unitAngle3, get_angle, LRU_Cache, lb_to_unit, sunPos, _gpsToTT,
    iers_interp, rotation_matrix_from_vectors, angle_between_vectors, gcrf_to_itrf
)
from .orbit import Orbit
from .ellipsoid import Ellipsoid

try:
    import erfa
except ImportError:
    # Let this raise
    import astropy._erfa as erfa


def _doSqueeze(squeezeOrbit, squeezeTime, *args):
    out = []
    for i, arg in enumerate(args):
        if squeezeTime:
            if squeezeOrbit:
                out.append(arg[0, 0])  # scalar
            else:
                out.append(np.squeeze(arg, axis=1))  # (nOrbit,)
        else:
            if squeezeOrbit:
                out.append(np.squeeze(arg, axis=0))  # (nTime,)
            else:
                out.append(arg)  # (nOrbit, nTime)
    if len(out) == 1:
        return out[0]
    return tuple(out)


def _countOrbit(orbit):
    # orbit is one of:
    # 1) scalar Orbit
    # 2) vector Orbit
    # 3) list of scalar Orbit
    # convert to (2), set nOrbit, squeezeOrbit, and orbit.

    # check 1) and 2)
    if isinstance(orbit, Orbit):
        # check shape of r to decide scalar vs vector
        if orbit.r.ndim == 1:  # scalar Orbit
            nOrbit = 1
            squeezeOrbit = True
            if 'kozaiMeanKeplerianElements' in orbit.__dict__:
                kMKE = orbit.__dict__['kozaiMeanKeplerianElements']
            else:
                kMKE = None
            orbit = Orbit(
                np.atleast_2d(orbit.r),
                np.atleast_2d(orbit.v),
                np.atleast_1d(orbit.t),
                mu=orbit.mu,
                propkw={k: np.atleast_1d(v) for k, v in orbit.propkw.items()},
            )
            # Copying just kozaiMeanKeplerianElements for now, though maybe
            # ought to retain other (all?) lazy_properties attributes too?
            if kMKE is not None:
                orbit.kozaiMeanKeplerianElements = kMKE
        else:  # vector Orbit
            nOrbit = orbit.r.shape[0]
            squeezeOrbit = False
    else:
        import warnings
        warnings.warn(
            "list of Orbit syntax is deprecated.  "
            "Please use a vector Orbit instead.",
            DeprecationWarning
        )
        # list of scalar orbit
        nOrbit = len(orbit)
        squeezeOrbit = False
        # assumes all orbits have the same keywords and mu!
        propkeys = [] if nOrbit == 0 else orbit[0].propkw.keys()
        mu = None if nOrbit == 0 else orbit[0].mu
        orbit = Orbit(
            np.array([o.r for o in orbit]),
            np.array([o.v for o in orbit]),
            np.array([o.t for o in orbit]),
            mu=mu,
            propkw={k: np.array([o.propkw[k] for o in orbit])
                    for k in propkeys}
        )
    return nOrbit, squeezeOrbit, orbit


def _countTime(time):
    if isinstance(time, Time):
        time = time.gps
    squeezeTime = False
    try:
        nTime = len(time)
    except TypeError:
        time = np.array([time])
        nTime = 1
        squeezeTime = True
    return nTime, squeezeTime, time


def _countR(r):
    # orbit is one of:
    # 1) scalar r
    # 2) vector r
    # 3) list of scalar Orbit
    # convert to (2), set nOrbit, squeezeOrbit, and orbit.
    squeezeR = False
    if np.shape(r)[-1] == 3:
        pass
    else:
        raise ValueError(f"Incorrect r dimensions. Expected shape (n, 3), but got {np.shape(r)}.")
    # check 1) and 2)
    if r.ndim < 3:  # scalar r
        nR = 1
        r = np.reshape(np.atleast_3d(r), (nR, np.shape(r)[0], np.shape(r)[1]))
        squeezeR = True
    else:
        nR = np.shape(r)[0]
    return nR, squeezeR, r


def _processObserver(
    observer, obsPos, obsVel, nTime, squeezeTime, time, doObsVel=False
):
    # We always need obsPos output, but only sometimes obsVel.  So only require
    # obsVel if doObsVel is True.  (but always return it, and always compute it
    # if what we got as input was in an observer).
    squeezeObsPos = False
    if observer is None:
        if obsPos is None:
            raise ValueError(
                "Exactly one of obsPos and observer must be specified"
            )
        if doObsVel and obsVel is None:
            raise ValueError(
                "Exactly one of obsVel and observer required for doObsVel"
            )

    if observer is not None:
        if obsPos is not None:
            raise ValueError(
                "Exactly one of obsPos and observer must be specified"
            )
        if doObsVel and obsVel is not None:
            raise ValueError(
                "Exactly one of obsVel and observer required for doObsVel"
            )
        try:
            nObservers = len(observer)
        except TypeError:
            observer = [observer]
            nObservers = 1
            squeezeObsPos = squeezeTime
        # Now generate obsPos
        if nObservers == nTime:
            obsPos = np.empty((nObservers, 3), dtype=float)
            obsVel = np.empty((nObservers, 3), dtype=float)
            for i, (obs, t) in enumerate(zip(observer, time)):
                obsPos[i], obsVel[i] = obs.getRV(t)
        else:
            if nObservers == 1:
                obsPos, obsVel = observer[0].getRV(time)
            elif nTime == 1:
                obsPos = np.empty((nObservers, 3), dtype=float)
                obsVel = np.empty((nObservers, 3), dtype=float)
                for i, obs in enumerate(observer):
                    obsPos[i], obsVel[i] = obs.getRV(time[0])
            else:
                raise ValueError("observer and time must be broadcastable")

    if doObsVel:
        obsPos, obsVel = np.broadcast_arrays(obsPos, obsVel)
    obsPos = np.atleast_1d(obsPos)
    if obsPos.ndim == 2:
        nObsPos = obsPos.shape[0]
    else:
        obsPos = np.atleast_2d(obsPos)
        nObsPos = 1
        squeezeObsPos = True
    # times and obsPos must align
    if nObsPos != nTime:
        if nObsPos == 1:
            obsPos = np.broadcast_to(obsPos, (nTime, 3))
            if doObsVel:
                obsVel = np.broadcast_to(obsVel, (nTime, 3))
            nObsPos = nTime
        elif nTime == 1:
            time = np.broadcast_to(time, (nObsPos,))
            nTime = nObsPos
        else:
            raise ValueError("obsPos and time must be broadcastable")
    squeezeTime &= squeezeObsPos
    return obsPos, obsVel, time, squeezeTime


class HashableArrayContainer:
    """
    A container for NumPy arrays that makes them hashable and immutable.

    This class wraps a NumPy array and ensures that it is immutable by setting
    its `writeable` flag to `False`. It also provides implementations for
    `__hash__` and `__eq__`, enabling the wrapped array to be used as a key in
    hash-based data structures like dictionaries and sets.

    Attributes:
        arr (numpy.ndarray): The wrapped NumPy array, which is immutable.

    Methods:
        __hash__():
            Returns a hash value for the array based on its byte representation.
        
        __eq__(rhs):
            Compares the wrapped array with another `HashableArrayContainer` instance
            for equality based on element-wise comparison.
    
    Examples:
        >>> import numpy as np
        >>> arr1 = np.array([1, 2, 3])
        >>> arr2 = np.array([1, 2, 3])
        >>> container1 = HashableArrayContainer(arr1)
        >>> container2 = HashableArrayContainer(arr2)
        >>> container1 == container2
        True
        >>> hash(container1) == hash(container2)
        True
        >>> my_dict = {container1: "value"}
        >>> my_dict[container2]
        'value'
    """

    def __init__(self, arr):
        self.arr = arr
        self.arr.flags.writeable = False

    def __hash__(self):
        return hash(self.arr.data.tobytes())

    def __eq__(self, rhs):
        return np.all(self.arr == rhs.arr)


def rv(orbit, time, propagator=KeplerianPropagator()):
    """Calculate positions and velocities on the outer product of all supplied
    orbits and times.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Desired orbit(s)
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    propagator : Propagator, optional
        The propagator instance to use.

    Notes
    -----
    If orbit or time is scalar valued (as opposed to a list of Orbit, e.g.) then
    the corresponding dimension of the output will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    Returns
    -------
    r : array_like (n, m, 3)
        Position in meters.
    v : array_like (n, m, 3)
        Velocity in meters per second.
    """
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
    nTime, squeezeTime, time = _countTime(time)
    print(nOrbit, squeezeOrbit, orbit, nTime, squeezeTime, time)
    outR, outV = _rv(orbit, HashableArrayContainer(time), propagator)

    return _doSqueeze(squeezeOrbit, squeezeTime, outR, outV)


def __rv(orbit, time, propagator):
    outR, outV = propagator._getRVMany(orbit, time.arr)

    return outR, outV


_rv = LRU_Cache(__rv, maxsize=16)


def groundTrack(orbit, time, propagator=KeplerianPropagator(), format='geodetic'):
    """Calculate satellite ground track on the outer product of all supplied times and
    state vectors or orbits.

    Parameters
    ----------
    r : array_like (n,3) Position (m)

    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    or

    orbit : r : array_like (n,3) Position (m) or
        Orbit or list of Orbit (n,)
        Desired orbit(s)
    propagator : Propagator, optional
        The propagator instance to use.
    format : 'geodetic' or 'cartesian'
        If 'geodetic', then returns longitude, latitude, height.
        If 'cartesian', then returns xyz in ITRF frame.

    Notes
    -----
    If orbit or time is scalar valued (as opposed to a list of Orbit, e.g.) then
    the corresponding dimension of the output will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    Returns
    -------
    lon, lat, height : array_like (n, m, 3)
        Radians and meters.

    or

    x, y, z : array_like(n, m, 3)
        Meters.
    """
    if format not in ['cartesian', 'geodetic']:
        raise ValueError("Format must be either 'cartesian' or 'geodetic'")

    nTime, squeezeTime, time = _countTime(time)
    if isinstance(orbit, Orbit):
        nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
        r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    else:
        nOrbit, squeezeOrbit, r = _countR(orbit)  # (n, m, 3)

    # Reverse the math in EarthObserver.getRV
    mjd_tt = _gpsToTT(time)
    d_ut1_tt_mjd, pmx, pmy = iers_interp(time)
    pn = erfa.pnm80(2400000.5, mjd_tt)
    gst = erfa.gst94(2400000.5, mjd_tt + d_ut1_tt_mjd)
    cg, sg = np.cos(gst), np.sin(gst)
    gstMat = np.zeros([len(time), 3, 3], dtype=float)
    gstMat[:, 0, 0] = cg
    gstMat[:, 0, 1] = sg
    gstMat[:, 1, 0] = -sg
    gstMat[:, 1, 1] = cg
    gstMat[:, 2, 2] = 1.0

    polar = np.zeros([len(time), 3, 3], dtype=float)
    polar[:] = np.eye(3)

    polar[:, 0, 2] = pmx
    polar[:, 1, 2] = -pmy
    polar[:, 2, 0] = -pmx
    polar[:, 2, 1] = pmy

    U = gstMat @ pn
    itrs = np.einsum(
        "tab,ntb->nta",
        (polar @ U),
        r
    )
    x = itrs[..., 0]
    y = itrs[..., 1]
    z = itrs[..., 2]
    if format == 'cartesian':
        return _doSqueeze(squeezeOrbit, squeezeTime, x, y, z)
    elif format == 'geodetic':
        ellipsoid = Ellipsoid()
        lon, lat, height = ellipsoid.cartToSphere(x, y, z)
        return _doSqueeze(squeezeOrbit, squeezeTime, lon, lat, height)


def _obsAngleCorrection(
    r, v, obsPos, obsVel, orbit, time, propagator, correctionType, max_iter=10
):
    if correctionType == "linear":
        # First order correction is to back up satellite by v * dt
        # where dt is the approximate time delay from the 0th order distance.
        # Important: *don't* do anything to the observer position, since we
        # really did do the observation at the originally given time and place.
        dt = norm(r - obsPos) / 299792458
        r -= np.einsum("nmi,nm->nmi", v, dt)
        # no change to v since linear assumption _is_ that v is constant for the
        # light time duration.
    elif correctionType == "exact":
        # Solve |r(t-dt) - r_obs(t)| = c dt for dt.
        dist = norm(r - obsPos)
        dt, dt_previous = dist / 299792458, np.inf
        iter = 0
        while np.any(np.abs(dt - dt_previous)) > 1e-12:  # picosecond accurate
            if iter > max_iter:
                raise RuntimeError(
                    "Exact light time correction did not converge in "
                    "{} iterations".format(iter)
                )
            # Unfortunately, can't just do
            # r, v = rv(orbit, time-dt, propagator=propagator)
            # because ssapy.rv isn't parallelized for n orbits and n, m times.
            # so do one orbit at a time.
            r = np.empty((len(orbit), len(time), 3))
            v = np.empty((len(orbit), len(time), 3))
            for i, o in enumerate(orbit):
                r[i], v[i] = rv(o, time - dt[i], propagator=propagator)
            dist = norm(r - obsPos)
            dt, dt_previous = dist / 299792458, dt
            iter += 1
    else:
        raise ValueError(
            "Invalid value for correctionType: {}"
            .format(correctionType)
        )
    # Apply velocity correction (i.e., diurnal aberration for an Earth-based
    # observer, and whatever you call the analogous correction for an orbital
    # observer).
    dr = norm(r - obsPos)
    r += obsVel * dr[..., None] / 299792458
    return r, v


def dircos(
    orbit, time, obsPos=None, obsVel=None, observer=None,
    propagator=KeplerianPropagator(), obsAngleCorrection=None
):
    """ Calculate observed direction-cosines of orbiting objects as viewed at
    specified times and positions.

    The direction cosines are the cosines of the angles between the vector
    pointing towards the orbiting object and the x, y, z axes.  An equivalent
    description is that they are the components of the unit vector pointing
    towards the orbiting object.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbit(s) for which to calculate direction cosines.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    obsPos : array_like (m, 3), optional
        Position of observer at given time.
    obsVel : array_like (m, 3), optional
        Velocity of observer at given time.  Only required if correcting for
        diurnal aberration.
    observer : Observer or list of Observers (m,), optional
        Observer(s) used to calculate obsPos and obsVel.
    propagator : Propagator, optional
        The propagator instance to use.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    Exactly 1 of `obsPos` and `observer` must be supplied.  `observer` and
    `obsPos` follow similar broadcasting rules as detailed below explicitly only
    for `obsPos`.

    The length of `time` and `obsPos` must match or be broadcastable to match.
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if both
    `time` and `obsPos` are scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    dircos : array_like (n, m, 3)
        Direction cosines on the outer product of orbit(s) and time/obsPos.
    """
    nTime, squeezeTime, time = _countTime(time)
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
    obsPos, obsVel, time, squeezeTime = _processObserver(
        observer, obsPos, obsVel, nTime, squeezeTime, time,
        doObsVel=obsAngleCorrection
    )

    # At this point, should have len(orbit) == n
    # and len(obsPos) == len(time) == nTime
    # So get positions of orbits...
    r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    if obsAngleCorrection:
        r, v = _obsAngleCorrection(
            r, v, obsPos, obsVel, orbit, time, propagator, obsAngleCorrection
        )
    dc = normed(r - obsPos)

    return _doSqueeze(squeezeOrbit, squeezeTime, dc)


def radec(
    orbit, time, obsPos=None, obsVel=None, observer=None,
    propagator=KeplerianPropagator(), obsAngleCorrection=None, rate=False
):
    """ Calculate observed right ascension, declination, and slant range of
    orbiting objects as viewed at specified times and positions.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbit(s) for which to calculate ra and dec.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    obsPos : array_like (m, 3), optional
        Positions of observers at given times.
    obsVel : array_like (m, 3), optional
        Velocity of observers at given times.
    observer : Observer or list of Observers (m,), optional
        Observer(s) used to calculate obsPos.
    propagator : Propagator, optional
        The propagator instance to use.
    rate : bool
        If True, return additionally the time derivatives of the quantity, times
        cos(dec) in the case of right ascension.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    Exactly 1 of `obsPos` and `observer` must be supplied.  `observer` and
    `obsPos` follow similar broadcasting rules as detailed below explicitly only
    for `obsPos`.

    The length of `time` and `obsPos` must match or be broadcastable to match.
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if both
    `time` and `obsPos` are scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    ra, dec : array_like (n, m)
        Right ascension and declination in radians.
    range : array_like (n, m)
        (Slant) range in meters.
    If rate, also:
    raRate : array_like (n, m)
        Time derivatives of right ascension times cos(dec), rad / sec.
    decRate : array_like (n, m)
        Time derivative of declination, rad / sec.
    rangeRate : array_like (n, m)
        Time derivative of range, meter / s.
    """
    nTime, squeezeTime, time = _countTime(time)
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
    obsPos, obsVel, time, squeezeTime = _processObserver(
        observer, obsPos, obsVel, nTime, squeezeTime, time,
        doObsVel=obsAngleCorrection
    )

    if rate and obsVel is None:
        raise ValueError('obsVel must not be None if rate is True.')

    # At this point, should have len(orbit) == n
    # and len(obsPos) == len(time) == nTime
    # So get positions of orbits...
    r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    if obsAngleCorrection:
        r, v = _obsAngleCorrection(
            r, v, obsPos, obsVel, orbit, time, propagator, obsAngleCorrection
        )

    ra, dec, slantRange, raRate, decRate, rangeRate = rvObsToRaDecRate(
        r, v, obsPos, obsVel
    )
    res = (ra, dec, slantRange)
    if rate:
        res = res + (raRate, decRate, rangeRate)

    return _doSqueeze(*((squeezeOrbit, squeezeTime) + res))


def altaz(
    orbit, time, observer, propagator=KeplerianPropagator(),
    obsAngleCorrection=None
):
    """ Calculate observed altitude and azimuth of orbiting objects as viewed at
    specified times and locations.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbits for which to calculate direction cosines.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    observer : EarthObserver
        Where on Earth to compute alt and az.
    propagator : Propagator, optional
        The propagator instance to use.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if `time` is
    scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    alt, az : array_like (n, m)
        Altitude and azimuth in radians.
    """
    from astropy.coordinates import SkyCoord, AltAz, GCRS

    nTime, squeezeTime, time = _countTime(time)
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)
    obsPos, obsVel = observer.getRV(time)

    r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    if obsAngleCorrection:
        r, v = _obsAngleCorrection(
            r, v, obsPos, obsVel, orbit, time, propagator, obsAngleCorrection
        )

    # Using astropy backend for AltAz computation
    sc = SkyCoord(
        x=r[..., 0], y=r[..., 1], z=r[..., 2],
        unit='m',
        representation_type='cartesian',
        frame=GCRS(obstime=Time(time, format='gps'))
    )
    aa = sc.transform_to(AltAz(location=observer._location))

    return _doSqueeze(squeezeOrbit, squeezeTime, aa.alt.radian, aa.az.radian)


def quickAltAz(
    orbit, time, observer, propagator=KeplerianPropagator(),
    obsAngleCorrection=None
):
    """ Quickly estimate observed altitude and azimuth of orbiting objects as
    viewed at specified times and locations.

    This algorithm approximates "up" as pointing directly away from the center
    of the Earth, instead of normal to the reference ellipsoid.  Use `altAz` if
    you want values wrt the reference ellipsoid.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbits for which to calculate direction cosines.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    observer : EarthObserver
        Where on Earth to compute alt and az.
    propagator : Propagator, optional
        The propagator instance to use.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if `time` is
    scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    alt, az : array_like (n, m)
        Altitude and azimuth in radians.
    """
    nTime, squeezeTime, time = _countTime(time)
    nOrbit, squeezeOrbit, orbit = _countOrbit(orbit)

    ro, vo = observer.getRV(time)  # (m, 3)
    r, v = rv(orbit, time, propagator=propagator)  # (n, m, 3)
    if obsAngleCorrection:
        r, v = _obsAngleCorrection(
            r, v, ro, vo, orbit, time, propagator, obsAngleCorrection
        )

    up = normed(ro)
    east = normed(vo)
    north = np.cross(up, east)

    dr = r - ro
    northing = np.einsum("nmi,mi->nm", dr, north)
    easting = np.einsum("nmi,mi->nm", dr, east)

    alt = np.pi / 2 - unitAngle3(
        normed(np.broadcast_to(ro, r.shape)), normed(dr)
    )
    az = np.arctan2(easting, northing)
    az %= 2 * np.pi

    return _doSqueeze(squeezeOrbit, squeezeTime, alt, az)


def radecRate(
    orbit, time, obsPos=None, obsVel=None, observer=None,
    propagator=KeplerianPropagator(),
    obsAngleCorrection=None
):
    """Calculate ra/dec rate and slant range rate of orbit at specified times
    and observer positions and velocities.

    DEPRECATED.  Use radec(..., rate=True) in new code.

    Parameters
    ----------
    orbit : Orbit or list of Orbit (n,)
        Orbit(s) for which to calculate slant range rate.
    time : array_like or astropy.time.Time (m,)
        If float (array), then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    obsPos : array_like (m, 3), optional
        Positions of observers at given times.
    obsVel : array_like (m, 3), optional
        Velocity of observers at given times.
    observer : Observer or list of Observers (m,), optional
        Observer(s) used to calculate obsPos and obsVel.
    propagator : Propagator, optional
        The propagator instance to use.
    obsAngleCorrection : {None, "linear", "exact"}, optional
        Correct actual angle to observed angle, meaning account for light-time
        delay, and aberration due to the observer's velocity.  None means don't
        do any correction.  "linear" means do an aberration correction and first
        order light-time correction.  "exact" means do an aberration correction
        and iteratively solve for the exact light-time correction.  (The
        "linear" correction is almost always sufficiently accurate).

    Notes
    -----
    Exactly 1 of `obsPos` and `observer` must be supplied.  `observer` and
    `obsPos` follow similar broadcasting rules as detailed below explicitly only
    for `obsPos`.  If `obsPos` is specified, `obsVel` must also be specified and
    congruent to `obsPos`.

    The length of `time` and `obsPos` must match or be broadcastable to match.
    If `orbit` is scalar-valued (an Orbit instead of a list of Orbit), then that
    dimension will be squeezed out in the return value.  Likewise, if both
    `time` and `obsPos` are scalar, that dimension will be squeezed out.

    For Keplerian orbit propagation it is more efficient to use a "vector Orbit"
    instead of a list of single scalar Orbits.

    When doing light time corrections, the time argument is the arrival time of
    the photons at the observer, as opposed to the emission time at the
    satellite.

    Returns
    -------
    ra : array_like (n, m)
        right ascension in radians
    raRate : array_like (n, m)
        Rate of change of right ascension*cos(dec) in radians per second.
    dec : array_link (n, m)
        declination in radians
    decRate : array_like (n, m)
        Rate of change of declination in radians per second.
    slantRange : array_like (n, m)
        Range in meters
    slantRangeRate : array_like (n, m)
        Slant range rate in meters per second.
    """
    import warnings
    warnings.warn("This function is deprecated; use "
                  "ssapy.compute.radec(..., rate=True)")
    ra, dec, slant, raRate, decRate, slantRate = radec(
        orbit, time, obsPos=obsPos, obsVel=obsVel, observer=observer,
        propagator=propagator, obsAngleCorrection=obsAngleCorrection
    )

    return raRate, decRate, slantRate


def rvObsToRaDecRate(r, v, obsPos=None, obsVel=None):
    """Convert object and observer position and velocity to angles.

    This only does the geometric part; it ignores light travel time,
    which may be applied to the object location before input.  Assumes
    that r, v, obsPos, and obsVel all have common shape.

    Parameters
    ----------
    r : array_like (..., 3)
        object position in meters
    v : array_like (..., 3)
        object velocity in meters per second
    obsPos : array_like (..., 3)
        observer position in meters
    obsVel : array_like (..., 3), optional
        observer velocity in meters per second

    Returns
    -------
    ra : array_like (...)
        right ascension in radians
    dec : array_link (...)
        declination in radians
    slantRange : array_like (...)
        Range in meters
    raRate : array_like (...)
        Rate of change of right ascension*cos(dec) in radians per second.
    decRate : array_like (...)
        Rate of change of declination in radians per second.
    slantRangeRate : array_like (...)
        Slant range rate in meters per second.
    """
    if obsPos is None:
        obsPos = np.zeros_like(r)

    dr = r - obsPos
    slantRange = norm(dr)
    ra = np.arctan2(dr[..., 1], dr[..., 0])
    dec = np.arcsin(dr[..., 2] / slantRange)

    if obsVel is None:
        obsVel = np.zeros_like(v)
    dv = v - obsVel

    # Now need to rotate to ra/dec coords
    sd, cd = np.sin(dec), np.cos(dec)
    sa, ca = np.sin(ra), np.cos(ra)

    rHat = normed(dr)  # (n, m, 3)

    raHat = np.zeros_like(rHat, dtype=float)
    raHat[..., 0] = -sa
    raHat[..., 1] = ca

    decHat = np.zeros_like(raHat)
    decHat[..., 0] = -sd * ca
    decHat[..., 1] = -sd * sa
    decHat[..., 2] = cd

    raRate = np.einsum("...i,...i->...", dv, raHat) / slantRange
    decRate = np.einsum("...i,...i->...", dv, decHat) / slantRange
    rangeRate = np.einsum("...i,...i->...", dv, rHat)
    return ra, dec, slantRange, raRate, decRate, rangeRate


def radecRateObsToRV(ra, dec, slantRange, raRate=None, decRate=None, slantRangeRate=None, obsPos=None, obsVel=None):
    """Convert object angles and observer position to 3D observer position

    This only does the geometric part; it ignores light travel time.  This
    is the inverse of rvObsToRaDecRate.

    If obsVel is None, then the returned velocity will also be None.

    Parameters
    ----------
    ra : array_like (...)
        right ascension in radians
    dec : array_like (...)
        declination in radians
    slantRange : array_like (...)
        Range in meters
    raRate : array_like (...)
        Rate of change of right ascension*cos(dec) in radians per second.
    decRate : array_like (...)
        Rate of change of declination in radians per second.
    slantRangeRate : array_like (...)
        Slant range rate in meters per second.
    obsPos : array_like (..., 3)
        Observer position in meters
    obsVel : array_like (..., 3)
        Observer velocity in meters

    Returns
    -------
    r : array_like (..., 3)
        object position in meters
    v : array_like (..., 3)
        object velocity in meters per second
        observer velocity in meters per second

    v is None if obsVel is None.
    """

    if obsPos is None:
        return ValueError('obsPos must be set!')

    rHat = lb_to_unit(ra, dec)

    sd, cd = np.sin(dec), np.cos(dec)
    sa, ca = np.sin(ra), np.cos(ra)

    raHat = np.zeros_like(rHat, dtype=float)
    raHat[..., 0] = -sa
    raHat[..., 1] = ca

    decHat = np.zeros_like(raHat)
    decHat[..., 0] = -sd * ca
    decHat[..., 1] = -sd * sa
    decHat[..., 2] = cd
    if len(rHat.shape) > 1:
        slantRange = np.array(slantRange)[..., None]
        raRate = np.array(raRate)[..., None]
        decRate = np.array(decRate)[..., None]
        slantRangeRate = np.array(slantRangeRate)[..., None]

    r = obsPos + rHat * slantRange
    if obsVel is None:
        v = None
    else:
        v = (obsVel + rHat * slantRangeRate + slantRange * (raHat * raRate + decHat * decRate))
    return r, v


def earthShadowCoords(r, time):
    """Determine components of position `r` parallel and perpendicular to sun
    unit vector.

    The sun unit vector points from the center of the sun through the center of
    the Earth.  Decomposing a satellite position into these coordinates yields a
    simple model of whether or not the satellite is in Earth's shadow:

        r_par, r_perp = earthShadowCoords(r, time)
        inShadow = r_par > 0 and r_perp < EARTH_RADIUS

    Parameters
    ----------
    r : ndarray (3,)
        Position (GCRF) in meters
    time : float or astropy.time.Time
        If float, then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC

    Returns
    -------
    r_par : float
        Position of satellite projected onto the sun unit vector in meters.
    r_perp : float
        Distance of satellite from Earth-Sun line in meters.
    """
    if isinstance(time, Time):
        time = time.gps
    r_sun = sunPos(time, fast=False)
    if r.ndim == 1:
        n_sun = - r_sun / norm(r_sun)
        r_par = np.dot(r, n_sun)
        r_perp = norm(r - n_sun * r_par)
    else:
        n_sun = - r_sun / norm(r_sun)[:, None]
        r_par = np.sum(r * n_sun, axis=1)
        r_perp = norm(r - n_sun * r_par[:, None])
    return r_par, r_perp


def find_passes(
    orbit, observers,
    tStart, tSpan, dt,
    propagator=KeplerianPropagator(),
    horizon=np.deg2rad(20)
):
    """Find satellite overhead passes for a collection of observers.

    Uses a brute force test of a grid of time points from tStart to tStart+tSpan
    separated by dt.

    Returns passes even if they occur during the daytime or if the satellite is
    not illuminated by the sun.  The only criterion for a successful "pass" is
    for the topocentric altitude of the satellite to be above the input
    `horizon`.  More details about a pass can subsequently be obtained by
    running the `refine_passes` function.

    Note this function is only suitable for `EarthObserver`s and not
    `OrbitalObserver`s.

    Parameters
    ----------
    orbit : Orbit
        Satellite orbit in question.
    observers : List of EarthObserver.
        Earth observers for which to check satellite visibility.
    tStart : float or astropy.time.Time
        Beginning of search window.
        If float, then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    tSpan : float or Quantity
        Time span in which to search for passes.
        If float, then seconds.
    dt : float or Quantity
        Time increment to use during search.  Satellite visibility will be
        computed every dt increment.  Smaller values will decrease the
        probability that a short duration pass will be missed, but will make the
        search take longer to complete.
        If float, then seconds.
    propagator : Propagator, optional
        The propagator instance to use.
    horizon : float or Quantity, optional
        Minimum altitude for which to consider a satellite "visible".
        If float, then should be in radians.

    Returns
    -------
    passDict : dict
        keys are `EarthObserver`s.
        values are lists (possibly empty) of `astropy.Time` corresponding to
        visible passes of the satellite.  Only one time is returned per pass,
        so multiple times in the return list indicate multiple distinct passes.
    """
    import astropy.units as u
    if isinstance(tStart, Time):
        tStart = tStart.gps
    if isinstance(tSpan, u.Quantity):
        tSpan = tSpan.to(u.s).value
    if isinstance(dt, u.Quantity):
        dt = dt.to(u.s).value
    if isinstance(horizon, u.Quantity):
        horizon = horizon.to(u.rad).value
    times = np.arange(tStart, tStart + tSpan, dt)
    out = {}
    for observer in observers:
        out[observer] = []
        alt, _ = quickAltAz(orbit, times, observer, propagator=propagator)
        w = np.nonzero(alt > horizon)[0]
        for wi in w:
            if (wi - 1) in w:  # already caught this pass
                continue
            out[observer].append(Time(times[wi], format='gps'))
    return out


def refine_pass(
    orbit,
    observer,
    time,
    propagator=KeplerianPropagator(),
    horizon=np.deg2rad(20),
    maxSpan=86400.0
):
    """Refine a satellite overhead pass.

    Parameters
    ----------
    orbit : Orbit
        Orbit in question.
    observer : EarthObserver.
        Observer for which to refine pass.
    time : float or astropy.time.Time
        A time when satellite is visible to observer.  (Found using
        find_passes, for instance)
        If float, then should correspond to GPS seconds; i.e., seconds
        since 1980-01-06 00:00:00 UTC
    propagator : Propagator, optional
        The propagator instance to use.
    horizon : float or Quantity, optional
        Minimum altitude for which to consider a satellite "visible".
        If float, then should be in radians.
    maxSpan : float or Quantity, optional
        Maximum amount of time before or after `time` to search for
        rise/set times, or time of max altitude.
        If float, then seconds.

    Returns
    -------
    dict
        Key/values are:
        tStart : Time
            Time at which Orbit rises above horizon
        tEnd : Time
            Time at which Orbit rises above horizon
        tMaxAlt : Time
            Time Orbit passes through maximum altitude
        maxAlt : Quantity
            Maximum altitude
        duration : Quantity
            Duration of pass.
        illumAtStart : bool
            Is satellite illuminated at `tStart`?
        illumAtEnd : bool
            Is satellite illuminated at `tEnd`?
        tTerminator : Time or None
            If illumAtStart != illumAtEnd, then time satellite passes through
            cylindrical terminator shadow.  Otherwise, None.
        sunAltStart : Quantity
            Altitude of sun at `tStart`
        sunAltEnd : Quantity
            Altitude of sun at `tEnd`
    """
    from scipy.optimize import bisect, minimize_scalar
    import astropy.units as u
    import warnings

    if isinstance(time, Time):
        time = time.gps
    if isinstance(horizon, u.Quantity):
        horizon = horizon.to(u.rad).value
    if isinstance(maxSpan, u.Quantity):
        maxSpan = maxSpan.to(u.s).value

    def dalt(t):
        return (
            horizon - quickAltAz(orbit, t, observer, propagator=propagator)[0]
        )

    # Height "outside" of cylindrical Earth shadow model.
    def dshadow(t):
        r, _ = rv(orbit, t, propagator=propagator)
        r_par, r_perp = earthShadowCoords(r, t)
        height = r_perp - EARTH_RADIUS
        if r_par < 0:
            # a bit of a hack, if the sat is on the sun side, then we'll
            # return abs(dshadow) so that the value is positive, which indicates
            # illumination.
            return abs(height)
        else:
            return height

    # Bracket and then bisect to find rise/set times
    def bracket(f, x, dx, xmax):
        x0, x1 = x, x + dx
        f0, f1 = f(x0), f(x1)
        while f0 * f1 > 0 and abs(x1 - x0) < xmax:
            x1 += dx
            f1 = f(x1)
        return x1

    tLow = bracket(dalt, time, -300, maxSpan)
    if dalt(tLow) * dalt(time) > 0:  # didn't bracket, use tLow and issue warning
        warnings.warn("Failed to bracket.  tStart is not rise time!")
        tStart = tLow
    else:
        tStart = bisect(dalt, tLow, time)
    tStart = Time(tStart, format='gps')
    tStart.format = 'iso'

    tHigh = bracket(dalt, time, 300, maxSpan)
    if dalt(tHigh) * dalt(time) > 0:
        warnings.warn("Failed to bracket.  tEnd is not set time!")
        tEnd = tHigh
    else:
        tEnd = bisect(dalt, time, tHigh)
    tEnd = Time(tEnd, format='gps')
    tEnd.format = 'iso'

    # Find maximum altitude
    dalt1 = dalt(tStart.gps)
    dalt2 = dalt(time)
    dalt3 = dalt(tEnd.gps)
    if dalt2 < dalt1 and dalt2 < dalt3:
        result = minimize_scalar(dalt, (tStart.gps, time, tEnd.gps))
        tMaxAlt = Time(result.x, format='gps')
        tMaxAlt.format = 'iso'
        maxAlt = -result.fun + horizon
    else:
        ama = np.argmin([dalt1, dalt2, dalt3])
        tMaxAlt = Time([tStart.gps, time, tEnd.gps][ama], format='gps')
        tMaxAlt.format = 'iso'
        maxAlt = -[dalt1, dalt2, dalt3][ama] + horizon
    duration = (tEnd - tStart).to(u.min)

    # Find illumination state
    illumAtStart = dshadow(tStart.gps) > 0
    sunAltStart = np.rad2deg(observer.sunAlt(tStart.gps)) * u.deg
    illumAtEnd = dshadow(tEnd.gps) > 0
    sunAltEnd = np.rad2deg(observer.sunAlt(tEnd.gps)) * u.deg

    # If we transition between shadow/illuminated, then find the time of
    # transition (time of terminator pass)
    if illumAtStart != illumAtEnd:
        tTerminator = bisect(dshadow, tStart.gps, tEnd.gps)
        tTerminator = Time(tTerminator, format='gps')
        tTerminator.format = 'iso'
    else:
        tTerminator = None

    return {
        "orbit": orbit,
        "observer": observer,
        "propagator": propagator,
        "horizon": horizon,
        "tStart": tStart,
        "tEnd": tEnd,
        "altStart": np.rad2deg(-dalt(tStart.gps) + horizon) * u.deg,
        "altEnd": np.rad2deg(-dalt(tEnd.gps) + horizon) * u.deg,
        "tMaxAlt": tMaxAlt,
        "maxAlt": np.rad2deg(maxAlt) * u.deg,
        "duration": duration,
        "illumAtStart": illumAtStart,
        "illumAtEnd": illumAtEnd,
        "tTerminator": tTerminator,
        "sunAltStart": sunAltStart,
        "sunAltEnd": sunAltEnd
    }


def nby3shape(arr_):
    """
    Reshapes or transforms an input NumPy array into a shape compatible with (n, 3).

    This function takes a NumPy array and ensures it has a shape of `(n, 3)`, 
    where `n` is the number of rows. The behavior depends on the dimensionality of the input array:

    - If the input array is 1-dimensional, it reshapes it into a `(1, 3)` array.
    - If the input array is 2-dimensional:
        - If the second dimension already has size 3, the array is returned as-is.
        - Otherwise, the array is transposed.

    Parameters:
        arr_ (numpy.ndarray): The input NumPy array to reshape or transform.

    Returns:
        numpy.ndarray: The reshaped or transformed array with a shape of `(n, 3)`.

    Raises:
        ValueError: If the input array cannot be reshaped or transformed into the desired shape.

    Examples:
        >>> import numpy as np
        >>> arr1 = np.array([1, 2, 3])
        >>> nby3shape(arr1)
        array([[1, 2, 3]])

        >>> arr2 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> nby3shape(arr2)
        array([[1, 2, 3],
               [4, 5, 6]])

        >>> arr3 = np.array([[1, 4], [2, 5], [3, 6]])
        >>> nby3shape(arr3)
        array([[1, 2, 3],
               [4, 5, 6]])
    """

    if arr_.ndim == 1:
        return np.reshape(arr_, (1, 3))
    if arr_.ndim == 2:
        if np.shape(arr_)[1] == 3:
            return arr_
        else:
            return arr_.T


def calculate_orbital_elements(r_, v_, mu_barycenter=EARTH_MU):
    """
    Calculate the Keplerian orbital elements from position and velocity vectors.

    This function computes the Keplerian orbital elements (semi-major axis, eccentricity, inclination, true longitude, argument of periapsis, longitude of ascending node, true anomaly, and specific angular momentum) for one or more celestial objects given their position and velocity vectors.

    Parameters:
    ----------
    r_ : (n, 3) numpy.ndarray
        Array of position vectors (in meters) of the celestial objects. Each row represents a position vector.
    v_ : (n, 3) numpy.ndarray
        Array of velocity vectors (in meters per second) of the celestial objects. Each row represents a velocity vector.
    mu_barycenter : float, optional
        Gravitational parameter (standard gravitational constant times mass) of the central body (default is `EARTH_MU`). This parameter defines the gravitational influence of the central body on the orbiting object.

    Returns:
    -------
    dict
        A dictionary containing the orbital elements for each celestial object:
        - 'a': Semi-major axis (in meters).
        - 'e': Eccentricity (dimensionless).
        - 'i': Inclination (in radians).
        - 'tl': True longitude (in radians).
        - 'ap': Argument of periapsis (in radians).
        - 'raan': Longitude of ascending node (in radians).
        - 'ta': True anomaly (in radians).
        - 'L': Specific angular momentum (in meters squared per second).

    Notes:
    ------
    - Position and velocity vectors should be provided in the same units.
    - The function assumes that the input vectors are provided in an array where each row corresponds to a different celestial object.
    - Orbital elements are computed using standard orbital mechanics formulas.
    - The inclination is measured from the reference plane, and the argument of periapsis and true anomaly are measured in the orbital plane.

    Example:
    --------
    >>> r = np.array([[1e7, 1e7, 1e7], [1e8, 1e8, 1e8]])
    >>> v = np.array([[1e3, 2e3, 3e3], [4e3, 5e3, 6e3]])
    >>> calculate_orbital_elements(r, v, mu_barycenter=3.986e14)
    {'a': [1.5707e7, 2.234e8], 'e': [0.123, 0.456], 'i': [0.785, 0.654], 'tl': [2.345, 3.456], 'ap': [0.123, 0.456], 'raan': [1.234, 2.345], 'ta': [0.567, 1.678], 'L': [1.234e10, 2.345e11]}
    """
    # mu_barycenter - all bodies interior to Earth
    # 1.0013415732186798 #All bodies of solar system
    mu_ = mu_barycenter
    rarr = nby3shape(r_)
    varr = nby3shape(v_)
    aarr = []
    earr = []
    incarr = []
    true_longitudearr = []
    argument_of_periapsisarr = []
    longitude_of_ascending_nodearr = []
    true_anomalyarr = []
    hmagarr = []
    for r, v in zip(rarr, varr):
        r = np.array(r)  # print(f'r: {r}')
        v = np.array(v)  # print(f'v: {v}')

        rmag = np.sqrt(r.dot(r))
        vmag = np.sqrt(v.dot(v))

        h = np.cross(r, v)
        hmag = np.sqrt(h.dot(h))
        n = np.cross(np.array([0, 0, 1]), h)

        a = 1 / ((2 / rmag) - (vmag ** 2) / mu_)

        evector = np.cross(v, h) / (mu_) - r / rmag
        e = np.sqrt(evector.dot(evector))

        inc = np.arccos(h[2] / hmag)

        if np.dot(r, v) > 0:
            true_anomaly = np.arccos(np.dot(evector, r) / (e * rmag))
        else:
            true_anomaly = 2 * np.pi - np.arccos(np.dot(evector, r) / (e * rmag))
        if evector[2] >= 0:
            argument_of_periapsis = np.arccos(np.dot(n, evector) / (e * np.sqrt(n.dot(n))))
        else:
            argument_of_periapsis = 2 * np.pi - np.arccos(np.dot(n, evector) / (e * np.sqrt(n.dot(n))))
        if n[1] >= 0:
            longitude_of_ascending_node = np.arccos(n[0] / np.sqrt(n.dot(n)))
        else:
            longitude_of_ascending_node = 2 * np.pi - np.arccos(n[0] / np.sqrt(n.dot(n)))

        true_longitude = true_anomaly + argument_of_periapsis + longitude_of_ascending_node
        aarr.append(a)
        earr.append(e)
        incarr.append(inc)
        true_longitudearr.append(true_longitude)
        argument_of_periapsisarr.append(argument_of_periapsis)
        longitude_of_ascending_nodearr.append(longitude_of_ascending_node)
        true_anomalyarr.append(true_anomaly)
        hmagarr.append(hmag)
    return {
        'a': aarr,
        'e': earr,
        'i': incarr,
        'tl': true_longitudearr,
        'ap': argument_of_periapsisarr,
        'raan': longitude_of_ascending_nodearr,
        'ta': true_anomalyarr,
        'L': hmagarr
    }


######################################################################################
# Lambertian brightness functions
######################################################################################


def moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels):  # In SI units, takes single values or arrays returns a fractional flux
    """
    Calculate the fractional solar flux reflected from the Moon to the satellite.

    This function computes the fraction of solar flux reflected from the Moon to the satellite, including contributions from both the front and back surfaces of the satellite's solar panels.

    Parameters:
    ----------
    r_moon : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the Moon.
    r_sat : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the satellite.
    r_earth : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the Earth.
    r_sun : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the Sun.
    radius : float
        Radius of the satellite in meters.
    albedo : float
        Albedo of the satellite's surface.
    albedo_moon : float
        Albedo of the Moon.
    albedo_back : float
        Albedo of the back surface of the satellite's solar panels.
    albedo_front : float
        Albedo of the front surface of the satellite's solar panels.
    area_panels : float
        Area of the satellite's solar panels in square meters.

    Returns:
    -------
    dict
        Dictionary containing the flux contributions from the Moon to the satellite:
        - 'moon_bus': Fraction of solar flux reflected off the satellite's bus from the Moon.
        - 'moon_panels': Fraction of solar flux reflected off the satellite's panels from the Moon.

    Notes:
    ------
    - The function assumes that the solar panels are always facing the Sun and calculates flux based on the phase angles.
    - Fraction of solar flux contributions from both the front and back surfaces of the solar panels are computed.

    Example:
    --------
    >>> r_moon = np.array([[1e8, 1e8, 1e8]])
    >>> r_sat = np.array([[1e7, 1e7, 1e7]])
    >>> r_earth = np.array([[0, 0, 0]])
    >>> r_sun = np.array([[1e11, 0, 0]])
    >>> moon_shine(r_moon, r_sat, r_earth, r_sun, radius=0.4, albedo=0.20, albedo_moon=0.12, albedo_back=0.50, albedo_front=0.05, area_panels=100)
    {'moon_bus': array([...]), 'moon_panels': array([...])}
    """
    # https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    moon_phase_angle = get_angle(r_sun, r_moon, r_sat)  # Phase of the moon as viewed from the sat.
    sun_angle = get_angle(r_sun, r_sat, r_moon)  # angle from Sun to object to Earth
    moon_to_earth_angle = get_angle(r_moon, r_sat, r_earth)
    r_moon_sat = np.linalg.norm(r_sat - r_moon, axis=-1)
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)  # Earth is the observer.
    flux_moon_to_sat = 2 / 3 * albedo_moon * MOON_RADIUS**2 / (np.pi * (r_moon_sat)**2) * (np.sin(moon_phase_angle) + (np.pi - moon_phase_angle) * np.cos(moon_phase_angle))  # Fraction of sunlight reflected from the Moon to satellite
    # Fraction of light from back of solar panel
    flux_back = np.zeros_like(sun_angle)
    flux_back[sun_angle > np.pi / 2] = np.abs(albedo_back * area_panels / (np.pi * r_earth_sat[sun_angle > np.pi / 2]**2) * np.cos(np.pi - moon_to_earth_angle[sun_angle > np.pi / 2]) * flux_moon_to_sat[sun_angle > np.pi / 2])  # Fraction of Moon light reflected off back of solar panels - which are assumed to be always facing the Sun. Angle: Sun - Observer - Sat
    flux_front = np.zeros_like(sun_angle)
    flux_front[sun_angle < np.pi / 2] = np.abs(albedo_front * area_panels / (np.pi * r_earth_sat[sun_angle < np.pi / 2]**2) * np.cos(moon_to_earth_angle[sun_angle < np.pi / 2]) * flux_moon_to_sat[sun_angle < np.pi / 2])  # Fraction of Sun light scattered off front of the solar panels - which are assumed to be always facing the Sun. Angle: Sun - Sat - Observer
    flux_panels = flux_back + flux_front
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_moon_to_sat
    return {'moon_bus': flux_bus, 'moon_panels': flux_panels}


def earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels):  # In SI units, takes single values or arrays returns a flux
    """
    Calculate the fractional flux of sunlight reflected from the Earth to the satellite.

    This function computes the flux of sunlight reflected from the Earth to the satellite, including contributions from the back surface of the satellite's solar panels.

    Parameters:
    ----------
    r_sat : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the satellite.
    r_earth : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the Earth.
    r_sun : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the Sun.
    radius : float
        Radius of the satellite in meters.
    albedo : float
        Albedo of the satellite's surface.
    albedo_earth : float
        Albedo of the Earth.
    albedo_back : float
        Albedo of the back surface of the satellite's solar panels.
    area_panels : float
        Area of the satellite's solar panels in square meters.

    Returns:
    -------
    dict
        Dictionary containing the flux contributions from the Earth to the satellite:
        - 'earth_bus': Fraction of solar flux reflected off the satellite's bus from the Earth.
        - 'earth_panels': Fraction of solar flux reflected off the satellite's panels from the Earth.

    Notes:
    ------
    - The function assumes that the solar panels are always facing the Sun and calculates flux based on the phase angle.
    - Fraction of solar flux contributions from the back surface of the solar panels are computed.

    Example:
    --------
    >>> r_sat = np.array([[1e7, 1e7, 1e7]])
    >>> r_earth = np.array([[1.496e11, 0, 0]])
    >>> r_sun = np.array([[0, 0, 0]])
    >>> earth_shine(r_sat, r_earth, r_sun, radius=0.4, albedo=0.20, albedo_earth=0.30, albedo_back=0.50, area_panels=100)
    {'earth_bus': array([...]), 'earth_panels': array([...])}
    """
    # https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    phase_angle = get_angle(r_sun, r_sat, r_earth)  # angle from Sun to object to Earth
    earth_angle = np.pi - phase_angle  # Sun to Earth to object.
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)  # Earth is the observer.
    flux_earth_to_sat = 2 / 3 * albedo_earth * EARTH_RADIUS**2 / (np.pi * (r_earth_sat)**2) * (np.sin(earth_angle) + (np.pi - earth_angle) * np.cos(earth_angle))  # Fraction of sunlight reflected from the Earth to satellite
    # Fraction of light from back of solar panel
    flux_back = np.zeros_like(phase_angle)
    flux_back[phase_angle > np.pi / 2] = albedo_back * area_panels / (np.pi * r_earth_sat[phase_angle > np.pi / 2]**2) * np.cos(np.pi - phase_angle[phase_angle > np.pi / 2]) * flux_earth_to_sat[phase_angle > np.pi / 2]  # Fraction of Earth light reflected off back of solar panels - which are assumed to be always facing the Sun. Angle: Sun - Observer - Sat
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_earth_to_sat
    return {'earth_bus': flux_bus, 'earth_panels': flux_back}


def sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels):  # In SI units, takes single values or arrays returns a fractional flux
    """
    Calculate the fractional flux of sunlight reflected from the Sun to the satellite.

    This function computes the fractional flux of sunlight reflected from the Sun to the satellite, including contributions from the front surface of the satellite's solar panels.

    Parameters:
    ----------
    r_sat : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the satellite.
    r_earth : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the Earth.
    r_sun : (n, 3) numpy.ndarray
        Array of coordinates representing the position of the Sun.
    radius : float
        Radius of the satellite in meters.
    albedo : float
        Albedo of the satellite's surface.
    albedo_front : float
        Albedo of the front surface of the satellite's solar panels.
    area_panels : float
        Area of the satellite's solar panels in square meters.

    Returns:
    -------
    dict
        Dictionary containing the flux contributions from the Sun to the satellite:
        - 'sun_bus': Fraction of solar flux reflected off the satellite's bus from the Sun.
        - 'sun_panels': Fraction of solar flux reflected off the satellite's panels from the Sun.

    Notes:
    ------
    - The function assumes that the solar panels are always facing the Sun and calculates flux based on the phase angle.
    - Fraction of solar flux contributions from the front surface of the solar panels are computed.

    Example:
    --------
    >>> r_sat = np.array([[1e7, 1e7, 1e7]])
    >>> r_earth = np.array([[0, 0, 0]])
    >>> r_sun = np.array([[1e11, 0, 0]])
    >>> sun_shine(r_sat, r_earth, r_sun, radius=0.4, albedo=0.20, albedo_front=0.05, area_panels=100)
    {'sun_bus': array([...]), 'sun_panels': array([...])}
    """
    # https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    phase_angle = get_angle(r_sun, r_sat, r_earth)  # angle from Sun to object to Earth
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)  # Earth is the observer.
    flux_front = np.zeros_like(phase_angle)
    flux_front[phase_angle < np.pi / 2] = albedo_front * area_panels / (np.pi * r_earth_sat[phase_angle < np.pi / 2]**2) * np.cos(phase_angle[phase_angle < np.pi / 2])  # Fraction of Sun light scattered off front of the solar panels - which are assumed to be always facing the Sun. Angle: Sun - Sat - Observer
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * (r_earth_sat)**2) * (np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle))  # Fraction of light reflected off satellite from Sun
    return {'sun_bus': flux_bus, 'sun_panels': flux_front}


def calc_M_v(r_sat, r_earth, r_sun, r_moon=False, radius=0.4, albedo=0.20, sun_Mag=4.80, albedo_earth=0.30, albedo_moon=0.12, albedo_back=0.50, albedo_front=0.05, area_panels=100, return_components=False):
    """
    Calculate the apparent magnitude (M_v) of a satellite due to reflections from the Sun, Earth, and optionally the Moon.

    This function computes the apparent magnitude of a satellite based on its reflected light from the Sun, Earth, and optionally the Moon. It uses separate functions to calculate the flux contributions from each of these sources and combines them to determine the overall apparent magnitude.

    Parameters:
    ----------
    r_sat : (n, 3) numpy.ndarray
        Position of the satellite in meters.
    r_earth : (n, 3) numpy.ndarray
        Position of the Earth in meters.
    r_sun : (n, 3) numpy.ndarray
        Position of the Sun in meters.
    r_moon : (n, 3) numpy.ndarray or False, optional
        Position of the Moon in meters. If False, the Moon's contribution is ignored (default is False).
    radius : float, optional
        Radius of the satellite in meters (default is 0.4 m).
    albedo : float, optional
        Albedo of the satellite's surface, representing its reflectivity (default is 0.20).
    sun_Mag : float, optional
        Solar magnitude (apparent magnitude of the Sun) used in magnitude calculations (default is 4.80).
    albedo_earth : float, optional
        Albedo of the Earth, representing its reflectivity (default is 0.30).
    albedo_moon : float, optional
        Albedo of the Moon, representing its reflectivity (default is 0.12).
    albedo_back : float, optional
        Albedo of the back surface of the satellite (default is 0.50).
    albedo_front : float, optional
        Albedo of the front surface of the satellite (default is 0.05).
    area_panels : float, optional
        Area of the satellite's panels in square meters (default is 100 m^2).
    return_components : bool, optional
        If True, returns the magnitude as well as the flux components from the Sun, Earth, and Moon (default is False).

    Returns:
    -------
    float
        The apparent magnitude (M_v) of the satellite as observed from Earth.

    dict, optional
        If `return_components` is True, a dictionary containing the flux components from the Sun, Earth, and Moon.

    Notes:
    ------
    - The function uses separate calculations for flux contributions from the Sun, Earth, and Moon:
        - `sun_shine` calculates the flux from the Sun.
        - `earth_shine` calculates the flux from the Earth.
        - `moon_shine` calculates the flux from the Moon (if applicable).
    - The apparent magnitude is calculated based on the distances between the satellite, Sun, Earth, and optionally the Moon, as well as their respective albedos and other parameters.

    Example usage:
    --------------
    >>> r_sat = np.array([[1e7, 2e7, 3e7]])
    >>> r_earth = np.array([1.496e11, 0, 0])
    >>> r_sun = np.array([0, 0, 0])
    >>> Mag_v = calc_M_v(r_sat, r_earth, r_sun, return_components=True)
    >>> Mag_v
    (15.63, {'sun_bus': 0.1, 'sun_panels': 0.2, 'earth_bus': 0.05, 'earth_panels': 0.1, 'moon_bus': 0.03, 'moon_panels': 0.07})
    """
    r_sun_sat = np.linalg.norm(r_sat - r_sun, axis=-1)
    frac_flux_sun = {'sun_bus': 0, 'sun_panels': 0}
    frac_flux_earth = {'earth_bus': 0, 'earth_panels': 0}
    frac_flux_moon = {'moon_bus': 0, 'moon_panels': 0}
    frac_flux_sun = sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels)
    frac_flux_earth = earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels)
    if r_moon is not False:
        frac_flux_moon = moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels)
    merged_dict = {**frac_flux_sun, **frac_flux_earth, **frac_flux_moon}
    total_frac_flux = sum(merged_dict.values())
    Mag_v = (2.5 * np.log10((r_sun_sat / (10 * u.Unit('parsec').to(u.Unit('m'))))**2) + sun_Mag) - 2.5 * np.log10(total_frac_flux)
    if return_components:
        return Mag_v, merged_dict
    else:
        return Mag_v


def M_v_lambertian(r_sat, times, radius=1.0, albedo=0.20, sun_Mag=4.80, albedo_earth=0.30, albedo_moon=0.12, plot=False):
    """
    Calculate the apparent magnitude (M_v) of a satellite due to reflections from the Sun, Earth, and Moon.

    This function computes the apparent magnitude of a satellite based on its reflected light from the Sun, Earth, and Moon, using the Lambertian reflection model. It optionally generates plots to visualize the results.

    Parameters:
    ----------
    r_sat : (n, 3) numpy.ndarray
        Position of the satellite in meters.
    times : Time or array_like
        The times corresponding to the satellite and celestial body positions. Used to obtain the positions of the Sun and Moon.
    radius : float, optional
        Radius of the satellite in meters (default is 1.0 m).
    albedo : float, optional
        Albedo of the satellite's surface, representing its reflectivity (default is 0.20).
    sun_Mag : float, optional
        Solar magnitude (apparent magnitude of the Sun) used in magnitude calculations (default is 4.80).
    albedo_earth : float, optional
        Albedo of the Earth, representing its reflectivity (default is 0.30).
    albedo_moon : float, optional
        Albedo of the Moon, representing its reflectivity (default is 0.12).
    plot : bool, optional
        If True, generates plots to visualize solar phase angle and magnitudes (default is False).

    Returns:
    -------
    numpy.ndarray
        The apparent magnitude (M_v) of the satellite as observed from Earth, considering reflections from the Sun, Earth, and Moon.

    Notes:
    ------
    - The function uses a Lambertian reflection model to compute the fraction of sunlight reflected by the satellite, Earth, and Moon.
    - The apparent magnitude is calculated based on the distances between the satellite, Sun, Earth, and Moon, as well as their respective albedos.
    - The function generates four subplots if `plot` is set to True:
      1. Solar phase angle of the satellite.
      2. Solar magnitude (M_v) of the satellite due to the Sun.
      3. Magnitude (M_v) of the satellite due to reflections from the Earth.
      4. Magnitude (M_v) of the satellite due to reflections from the Moon.

    Example usage:
    --------------
    >>> r_sat = np.array([[1e7, 2e7, 3e7]])
    >>> times = Time("2024-01-01")
    >>> M_v = M_v_lambertian(r_sat, times, plot=True)
    >>> M_v
    array([15.63])
    """
    pc_to_m = 3.085677581491367e+16
    r_sun = get_body('Sun').position(times).T
    r_moon = get_body('Moon').position(times).T
    r_earth = np.zeros_like(r_sun)

    r_sun_sat = np.linalg.norm(r_sat - r_sun, axis=-1)
    r_earth_sat = np.linalg.norm(r_sat, axis=-1)
    r_moon_sat = np.linalg.norm(r_sat - r_moon, axis=-1)

    sun_angle = get_angle(r_sun, r_sat, r_earth)
    earth_angle = np.pi - sun_angle
    moon_phase_angle = get_angle(r_sun, r_moon, r_sat)  # Phase of the moon as viewed from the sat.
    moon_to_earth_angle = get_angle(r_moon, r_sat, r_earth)

    flux_moon_to_sat = 2 / 3 * albedo_moon * MOON_RADIUS**2 / (np.pi * (r_moon_sat)**2) * (np.sin(moon_phase_angle) + (np.pi - moon_phase_angle) * np.cos(moon_phase_angle))  # Fraction of sunlight reflected from the Moon to satellite
    flux_earth_to_sat = 2 / 3 * albedo_earth * EARTH_RADIUS**2 / (np.pi * (r_earth_sat)**2) * (np.sin(earth_angle) + (np.pi - earth_angle) * np.cos(earth_angle))  # Fraction of sunlight reflected from the Earth to satellite

    frac_flux_sun = 2 / 3 * albedo * radius**2 / (np.pi * (r_earth_sat)**2) * (np.sin(sun_angle) + (np.pi - sun_angle) * np.cos(sun_angle))  # Fraction of light reflected off satellite from Sun
    frac_flux_earth = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_earth_to_sat
    frac_flux_moon = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_moon_to_sat
    Mag_v = (2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(frac_flux_sun + frac_flux_earth + frac_flux_moon)
    if plot:
        import matplotlib.pyplot as plt
        sun_scale = 149597870700.0 * (RGEO / np.max(r_earth_sat) ) * 0.75
        color_map ='inferno_r'
        fig = plt.figure(figsize=(18, 4))
        ax = fig.add_subplot(1, 4, 1)
        ax.scatter(r_earth[:, 0], r_earth[:, 1], c='Blue', s=10)
        scatter = ax.scatter(r_sat[:, 0] / RGEO, r_sat[:, 1] / RGEO, c=sun_angle, cmap=color_map)
        colorbar = plt.colorbar(scatter)
        ax.scatter(r_sun[:, 0] / sun_scale, r_sun[:, 1] / sun_scale, c=plt.cm.Oranges(np.linspace(0.25, 0.75, len(r_sat[:, 0]))), s=10)
        ax.set_title('Solar Phase')
        ax.set_xlabel('X [GEO]')
        ax.set_ylabel('Y [GEO]')
        ax.axis('equal')

        ax = fig.add_subplot(1, 4, 2)
        ax.scatter(r_earth[0], r_earth[1], c='Blue', s=10)
        scatter = ax.scatter(r_sat[:, 0] / RGEO, r_sat[:, 1] / RGEO, c=(2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(frac_flux_sun), cmap=color_map)
        colorbar = plt.colorbar(scatter)
        ax.scatter(r_sun[:, 0] / sun_scale, r_sun[:, 1] / sun_scale, c=plt.cm.Oranges(np.linspace(0.25, 0.75, len(r_sat[:, 0]))), s=10)
        ax.set_title('Solar M_v')
        ax.axis('equal')

        ax = fig.add_subplot(1, 4, 3)
        ax.scatter(r_earth[:, 0], r_earth[:, 1], c='Blue', s=10)
        scatter = ax.scatter(r_sat[:, 0] / RGEO, r_sat[:, 1] / RGEO, c=(2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(frac_flux_earth), cmap=color_map)
        colorbar = plt.colorbar(scatter)
        ax.scatter(r_sun[:, 0] / sun_scale, r_sun[:, 1] / sun_scale, c=plt.cm.Oranges(np.linspace(0.25, 0.75, len(r_sat[:, 0]))), s=10)

        ax.set_title('Earth M_v')
        ax.axis('equal')

        ax = fig.add_subplot(1, 4, 4)
        ax.scatter(r_earth[:, 0], r_earth[:, 1], c='Blue', s=10)
        scatter = ax.scatter(r_sat[:, 0] / RGEO, r_sat[:, 1] / RGEO, c=(2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(frac_flux_moon), cmap=color_map)
        ax.scatter(r_moon[:, 0] / RGEO, r_moon[:, 1] / RGEO, c=plt.cm.Greys(np.linspace(0.5, 1, len(r_sat[:, 0]))), s=5)

        colorbar = plt.colorbar(scatter)
        ax.set_title('Lunar M_v')
        ax.axis('equal')
        plt.show()

    return Mag_v


def calc_gamma(r, t):
    """
    Calculate the gamma angle between position and velocity vectors in the ITRF frame.

    Parameters:
    r (numpy.ndarray): The position vectors in the GCRF frame, shaped (n, 3), where n is the number of vectors.
    t (numpy.ndarray or astropy.time.Time): The times corresponding to the position vectors. Can be an array of GPS seconds or an Astropy Time object.

    Returns:
    numpy.ndarray: An array of gamma angles in degrees between the position and velocity vectors for each time point.

    Notes:
    - This function first converts the given position vectors from the GCRF frame to the ITRF frame.
    - It then calculates the angle between the position and velocity vectors at each time point in the ITRF frame.
    - If the input time array is an Astropy Time object, it converts it to GPS time before processing.
    """
    r_itrf, v_itrf = gcrf_to_itrf(r, t, v=True)
    if isinstance(t[0], Time):
        t = t.gps
    gamma = np.degrees(np.apply_along_axis(lambda x: angle_between_vectors(x[:3], x[3:]), 1, np.concatenate((r_itrf, v_itrf), axis=1))) - 90
    return gamma



def moon_normal_vector(t):
    """
    Calculate the normal vector to the Moon's orbital plane at a given time.

    This function computes the normal vector to the Moon's orbital plane by taking the cross product of the Moon's position vectors at two different times: the given time `t` and one week later. The resulting vector is normalized to provide the direction of the orbital plane's normal.

    Parameters:
    ----------
    t : Time
        The time at which to calculate the normal vector to the Moon's orbital plane.

    Returns:
    -------
    numpy.ndarray
        A 3-element array representing the normal vector to the Moon's orbital plane at the given time. The vector is normalized to have a unit length.

    Notes:
    ------
    - The function assumes a circular orbit for the Moon and uses a time step of one week (604800 seconds) to calculate the normal vector.
    - The normal vector is perpendicular to the plane defined by the Moon's position vectors at the given time and one week later.

    Example usage:
    --------------
    >>> t = Time("2024-01-01")
    >>> normal_vector = moon_normal_vector(t)
    >>> normal_vector
    array([-0.093,  0.014,  0.995])
    """
    r = get_body("moon").position(t).T
    r_random = get_body("moon").position(t.gps + 604800).T
    return np.cross(r, r_random) / np.linalg.norm(np.cross(r, r_random))


def lunar_lagrange_points(t):
    """
    Calculate the positions of the lunar Lagrange points in the GCRF frame at a given time.

    This function computes the positions of the five Lagrange points (L1, L2, L3, L4, and L5) in the Earth-Moon system at a specific time `t`. It considers the positions of the Earth and Moon and uses the orbital period of the Moon to find the Lagrange points.

    Parameters:
    ----------
    t : Time
        The time at which to calculate the Lagrange points. The position of the Moon at this time is used to compute the Lagrange points.

    Returns:
    -------
    dict
        A dictionary containing the coordinates of the five Lagrange points:
        - "L1": Position of the first Lagrange point between the Earth and the Moon.
        - "L2": Position of the second Lagrange point beyond the Moon.
        - "L3": Position of the third Lagrange point directly opposite the Moon, relative to the Earth.
        - "L4": Position of the fourth Lagrange point, calculated as the Moon's position offset by one-sixth of the lunar period.
        - "L5": Position of the fifth Lagrange point, calculated as the Moon's position offset by negative one-sixth of the lunar period.

    Notes:
    ------
    - The function assumes a circular orbit for the Moon.
    - The lunar period used is approximately 2.36 million seconds.
    - The gravitational parameters of the Earth and Moon are denoted as `EARTH_MU` and `MOON_MU`, respectively.

    Example usage:
    --------------
    >>> t = Time("2024-01-01")
    >>> lagrange_points = lunar_lagrange_points(t)
    >>> lagrange_points["L1"]
    array([1.02e6, 0.0, 0.0])
    >>> lagrange_points["L4"]
    array([1.5e6, 1.5e6, 0.0])
    """

    r = get_body("moon").position(t).T
    d = np.linalg.norm(r)  # Distance between Earth and Moon
    unit_vector_moon = r / np.linalg.norm(r, axis=-1)
    # plane_vector = np.cross(r, r_random)
    lunar_period_seconds = 2.3605915968e6

    # Coefficients of the quadratic equation
    a = EARTH_MU - MOON_MU
    b = 2 * MOON_MU * d
    c = -MOON_MU * d**2

    # Solve the quadratic equation
    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
        L1_from_moon = (-b - np.sqrt(discriminant)) / (2*a) * unit_vector_moon
        L2_from_moon = (-b + np.sqrt(discriminant)) / (2*a) * unit_vector_moon
    else:
        print("Discriminate is less than 0! THAT'S WEIRD FIX IT.")
        L1_from_moon = None
        L2_from_moon = None

    return {
        "L1": L1_from_moon + r,
        "L2": L2_from_moon + r,
        "L3": -r,
        "L4": get_body("moon").position(t.gps + lunar_period_seconds / 6).T,
        "L5": get_body("moon").position(t.gps - lunar_period_seconds / 6).T
    }


def lunar_lagrange_points_circular(t):
    """
    Calculate the positions of the lunar Lagrange points in the GCRF frame for a given time.

    This function calculates the positions of the five Lagrange points (L1, L2, L3, L4, and L5) in the Earth-Moon system at a specific time `t`. It accounts for the rotation of the Moon's orbit around the Earth, providing the positions in a circular approximation of the Earth-Moon system.

    Parameters:
    ----------
    t : Time
        The time at which to calculate the Lagrange points. The position of the Moon at this time is used to compute the Lagrange points.

    Returns:
    -------
    dict
        A dictionary containing the coordinates of the five Lagrange points:
        - "L1": Position of the first Lagrange point between the Earth and the Moon.
        - "L2": Position of the second Lagrange point beyond the Moon.
        - "L3": Position of the third Lagrange point directly opposite the Moon, relative to the Earth.
        - "L4": Position of the fourth Lagrange point, forming an equilateral triangle with the Earth and Moon.
        - "L5": Position of the fifth Lagrange point, forming an equilateral triangle with the Earth and Moon, but on the opposite side.

    Notes:
    ------
    - The function assumes a circular orbit for the Moon and uses a rotation matrix to align the z-axis with the Moon's normal vector.
    - The positions of L4 and L5 are calculated using a rotation matrix to align with the Moon's orientation.
    - The gravitational parameters of the Earth and Moon are denoted as `EARTH_MU` and `MOON_MU`, respectively.

    Example usage:
    --------------
    >>> t = Time("2024-01-01")
    >>> lagrange_points = lunar_lagrange_points_circular(t)
    >>> lagrange_points["L1"]
    array([1.02e6, 0.0, 0.0])
    >>> lagrange_points["L4"]
    array([1.5e6, 1.5e6, 0.0])
    """
    r = get_body("moon").position(t).T
    d = np.linalg.norm(r)  # Distance between Earth and Moon
    unit_vector_moon = r / np.linalg.norm(r, axis=-1)

    # Coefficients of the quadratic equation
    a = EARTH_MU - MOON_MU
    b = 2 * MOON_MU * d
    c = -MOON_MU * d**2

    # Solve the quadratic equation
    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
        L1_from_moon = (-b - np.sqrt(discriminant)) / (2*a) * unit_vector_moon
        L2_from_moon = (-b + np.sqrt(discriminant)) / (2*a) * unit_vector_moon
    else:
        print("Discriminate is less than 0! THAT'S WEIRD FIX IT.")
        L1_from_moon = None
        L2_from_moon = None

    # L45
    # Create the rotation matrix to align z-axis with the normal vector
    normal_vector = moon_normal_vector(t)
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal_vector)
    theta = np.radians(60) + np.arctan2(unit_vector_moon[1], unit_vector_moon[0])
    L4 = np.vstack((d * np.cos(theta), d * np.sin(theta), np.zeros_like(theta))).T
    L4 = np.squeeze(L4 @ rotation_matrix.T)
    theta = -np.radians(60) + np.arctan2(unit_vector_moon[1], unit_vector_moon[0])
    L5 = np.vstack((d * np.cos(theta), d * np.sin(theta), np.zeros_like(theta))).T
    L5 = np.squeeze(L5 @ rotation_matrix.T)

    return {
        "L1": L1_from_moon + r,
        "L2": L2_from_moon + r,
        "L3": -r,
        "L4": L4,
        "L5": L5
    }


def lagrange_points_lunar_frame():
    """
    Calculate the positions of the lunar Lagrange points in the lunar frame, This frame is defined by the coordinate transformation in utils.py gcrf_to_lunar().

    Returns:
    -------
    dict
        A dictionary containing the coordinates of the five Lagrange points:
        - "L1": Position of the first Lagrange point between the Earth and the Moon.
        - "L2": Position of the second Lagrange point beyond the Moon.
        - "L3": Position of the third Lagrange point directly opposite the Moon, relative to the Earth.
        - "L4": Position of the fourth Lagrange point, forming an equilateral triangle with the Earth and Moon.
        - "L5": Position of the fifth Lagrange point, forming an equilateral triangle with the Earth and Moon, but on the opposite side.

    Notes:
    ------
    - The function assumes that the Earth and Moon are the two primary bodies, with the Earth-Moon distance denoted as `LD`.
    - The gravitational parameters of the Earth and Moon are denoted as `EARTH_MU` and `MOON_MU`, respectively.
    - The positions of L4 and L5 are calculated using the fact that these points form an equilateral triangle with the Earth and Moon.

    Example usage:
    --------------
    >>> lagrange_points = lagrange_points_lunar_frame()
    >>> lagrange_points["L1"]
    array([1.01e6, 0.0, 0.0])
    >>> lagrange_points["L4"]
    array([1.5e6, 1.5e6, 0.0])
    """
    r = np.array([LD / RGEO, 0, 0])
    d = np.linalg.norm(r)  # Distance between Earth and Moon
    unit_vector_moon = r / np.linalg.norm(r, axis=-1)

    # Coefficients of the quadratic equation
    a = EARTH_MU - MOON_MU
    b = 2 * MOON_MU * d
    c = -MOON_MU * d**2

    # Solve the quadratic equation
    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
        L1_from_moon = (-b - np.sqrt(discriminant)) / (2*a) * unit_vector_moon
        L2_from_moon = (-b + np.sqrt(discriminant)) / (2*a) * unit_vector_moon
    else:
        print("Discriminate is less than 0! THAT'S WEIRD FIX IT.")
        L1_from_moon = None
        L2_from_moon = None

    # L45
    theta = np.radians(60)
    L4 = np.squeeze(np.vstack((d * np.cos(theta), d * np.sin(theta), np.zeros_like(theta))).T)
    L5 = np.squeeze(np.vstack((d * np.cos(-theta), d * np.sin(-theta), np.zeros_like(-theta))).T)
    return {
        "L1": L1_from_moon + r,
        "L2": L2_from_moon + r,
        "L3": -r,
        "L4": L4,
        "L5": L5
    }

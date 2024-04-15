import sys
import os
import numpy as np
from astropy.time import Time
import astropy.units as u

from . import datadir
from .constants import MOON_RADIUS, WGS84_EARTH_OMEGA

try:
    import erfa
except ImportError:
    import astropy._erfa as erfa


def find_file(filename, ext=None):
    """ Find a file in the current directory or the ssapy datadir.  If ext is
    not None, also try appending ext to the filename.
    """
    candidates = [
        filename,
        os.path.join(datadir, filename),
        filename + ext,
        os.path.join(datadir, filename + ext),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(filename)


def _wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def num_wraps(ang):
    """
    Return number of times angle `ang` has wrapped around.

    Returns
    -------
    int
        Number of wraps
    """
    import astropy.units as u
    if isinstance(ang, u.Quantity):
        ang = float(ang / u.rad)
    return int(np.floor(ang / (2 * np.pi)))


class lazy_property:
    """
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value


def newton_raphson(guess, f, fprime=None, eps=3e-16, maxiter=100, **kwargs):
    """
    Find a root of a univariate function with known gradient using
    Newton-Raphson iterations.

    Parameters
    ----------
    guess : float
        Initial guess.
    f : callable function of 1 argument.
        Function for which to find a zero.  If fprime is None, then the return
        value of f should be a 2-tuple with the value of the function and the
        first derivative.  If fprime is given separately, then f should just
        return the value of the function.
    fprime : callable function of 1 argument, optional
        Derivative of f.  Default None.
    eps : float, optional
        Absolute tolerance for finding a zero.  Default 3e-16.
    maxiter : int, optional
        Maximum number of iterations to try.  Default 100.

    Returns
    -------
    solution: float
    """
    delta = np.inf
    niter = 0
    x = guess
    while np.abs(delta) > eps and niter < maxiter:
        if fprime is None:
            val, grad = f(x, **kwargs)
        else:
            val = f(x, **kwargs)
            grad = fprime(x, **kwargs)
        delta = val / grad
        x -= delta
        niter += 1
    return x


def find_extrema_brackets(fs):
    """
    Find triplets bracketing extrema from an ordered set of function
    evaluations.
    """
    # Want to find places where the sign of the finite difference changes.
    diffs = np.trim_zeros(np.diff(fs), 'b')
    w = np.where(diffs == 0)[0]
    while np.any(w):
        diffs[w] = diffs[w + 1]
        w = np.where(diffs == 0)[0]
    w = np.where(diffs[:-1] * diffs[1:] < 0)[0]
    out = []
    for wi in w:
        j = 2
        while fs[wi + j] == fs[wi + 1]:
            j += 1
        out.append((wi, wi + 1, wi + j))
    return out


def find_all_zeros(f, low, high, n=100):
    """
    Attempt to find all zeros of a function between given bounds.  Uses n
    initial samples to characterize the zero-landscape of the interval.
    """
    from scipy.optimize import minimize_scalar, brentq
    # Step one: Evaluate the function n times.
    xs = np.linspace(low, high, n)
    fs = np.array([f(x) for x in xs])
    # Step two: Identify extrema existence.
    extrema = find_extrema_brackets(fs)
    # Step three: Fill in points for extrema that may lead to a zero-crossing.
    newx, newf = [], []
    for extremum in extrema:
        is_minimum = fs[extremum[1]] < fs[extremum[0]]
        if (fs[extremum[1]] - fs[extremum[0]]) * fs[extremum[0]] > 0:
            continue
        # make a record of function evaluations.

        def f2(x):
            result = f(x)
            newx.append(x)
            newf.append(result)
            if not is_minimum:
                result *= -1
            return result
        minimize_scalar(f2, tuple(xs[i] for i in extremum))

    xs = np.concatenate([xs, newx])
    fs = np.concatenate([fs, newf])
    a = np.argsort(xs)
    xs = xs[a]
    fs = fs[a]
    # Step four: Find zero crossings
    zeros = list(xs[fs == 0.0])
    for w in np.where(fs[1:] * fs[:-1] < 0)[0]:
        zeros.append(brentq(f, xs[w], xs[w + 1]))
    return zeros


# Not quite as fast as specializing to shape = (3,), but more portable, and
# faster than np.linalg.norm().
# These compute the norm over the last axis and maintain the other leading axes.
def normSq(arr):
    return np.einsum("...i,...i", arr, arr)


# Faster to not dispatch back to normSq
def norm(arr):
    return np.sqrt(np.einsum("...i,...i", arr, arr))


# Faster to not dispatch back to norm
def normed(arr):
    return arr / np.sqrt(np.einsum("...i,...i", arr, arr))[..., None]


def unitAngle3(r1, r2):
    """Robustly compute angle between unit vectors r1 and r2.
    Vectorized for multiple triplets.
    """
    r1, r2 = np.broadcast_arrays(r1, r2)
    dsq = normSq(r1 - r2)
    out = 2 * np.arcsin(0.5 * np.sqrt(dsq))
    # Following will almost never be the case.  Exception is when r1, r2 are
    # nearly antipodal.
    w = dsq < 3.99
    if not np.all(w):
        try:
            out[~w] = np.pi - np.arcsin(norm(np.cross(r1[~w], r2[~w])))
        except (TypeError, IndexError):
            out = np.pi - np.arcsin(norm(np.cross(r1, r2)))
    return out


def cluster_emcee_walkers(
    chain, lnprob, lnprior, thresh_multiplier=1, verbose=False
):
    """
    Down-select emcee walkers to those with the largest mean posteriors

    Follows the algorithm of Hou, Goodman, Hogg et al. (2012)
    """
    import emcee
    from distutils.version import LooseVersion
    if LooseVersion(emcee.__version__) < LooseVersion("3.0"):
        raise ValueError("emcee version at least 3.0rc2 required")
    if verbose:
        out = "Clustering emcee walkers with threshold multiplier {:3.2f}"
        out = out.format(thresh_multiplier)
        print(out)
    chain = np.array(chain)
    lnprob = np.array(lnprob)
    # ## lnprob.shape == (Nsteps, Nwalkers) => lk.shape == (Nwalkers,)
    lk = -np.mean(np.array(lnprob), axis=0)
    nwalkers = len(lk)
    ndx = np.argsort(lk)
    lks = lk[ndx]
    d = np.diff(lks)
    thresh = np.cumsum(d) / np.arange(1, nwalkers)
    selection = d > (thresh_multiplier * thresh)
    if np.any(selection):
        nkeep = np.argmax(selection)
    else:
        nkeep = nwalkers
    if verbose:
        print("chain, lnprob:", chain.shape, lnprob.shape)
    chain = chain[:, ndx[0:nkeep], :]
    lnprob = lnprob[:, ndx[0:nkeep]]
    lnprior = lnprior[:, ndx[0:nkeep]]
    if verbose:
        print("New chain, lnprob:", chain.shape, lnprob.shape)
    return chain, lnprob, lnprior


def subsample_high_lnprob(chain, lnprob, lnprior, nSample, thresh=-10):
    """Select MCMC samples with probabilities above some relative threshold

    Parameters
    ----------
    chain : array_like, (nWalker, nStep, 6)
        Input MCMC chain from EmceeSampler.sample()
    lnprob : array_like, (nWalker, nStep)
        Input MCMC lnprob from EmceeSampler.sample()
    lnprior : array_like, (nWalker, nStep)
        Input MCMC lnprior from EmceeSampler.sample()
    nSample : int
        Number of samples to return.  If fewer than nSample samples
        exceed threshold, then return the nSample most probable
        samples.
    thresh : float, optional
        Threshold w.r.t. max(lnprob) below which to exclude samples.

    Returns
    -------
    samples : array_like, (nSample, 6)
        Output samples
    """
    chain = chain.reshape(-1, 6)
    lnprob = lnprob.reshape(-1)
    lnprior = lnprior.reshape(-1)
    asort = np.argsort(-lnprob)
    lnprobmax = lnprob[asort[0]]
    goodLnProbThresh = lnprobmax + thresh
    nGood = np.searchsorted(-lnprob, -goodLnProbThresh, sorter=asort)
    nGood = max(nGood, nSample)
    permutation = np.random.permutation(nGood)
    s = asort[permutation[:nSample]]
    return chain[s], lnprob[s], lnprior[s]


def resample(particles, ln_weights, obs_times=None, pod=False):
    """
    Resample particles to achieve more uniform weights.
    This produces the same number of particles as are input (i.n. dimension
    matches "particles")
    From "Bayesian Multiple Target Tracking" (2nd ed.) p. 100

    :param particles: 2D array of particles. Each row is an nD particles
    :type particles: numpy.ndarray

    :param ln_weights: 1D array of particle log-weights. Length should match
        number of particles
    :type ln_weights: numpy.ndarray

    :param obs_times: Time at which observation occured. This is only used if
        pod=True, defaults to None
    :type obs_times: [type], optional

    :param pod: Specify whether this is the special case of regularizing orbit
        parameters, defaults to False
    :type pod: bool, optional

    :return: Resampled particles and weights (same number as input)
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    import bisect
    num_particles = particles.shape[0]
    # weights /= np.sum(weights)
    weights = get_normed_weights(ln_weights)

    # 1.
    cumul_weights = np.cumsum(weights)

    # 2.
    # This is cumulative weights for a new set of particles with equal weights
    cumul_new_weights = np.arange(num_particles, dtype=float) / num_particles
    # The offset of the first value sets the sampling used to match the
    # cumulative weight ranges of the old particles to the new particles
    new_weight_offset = np.random.uniform(high=1. / num_particles)
    cumul_new_weights += new_weight_offset

    # 3.
    # For j=1:J, do the following
    # For m such that C_{j-1} <= u_m <= C_j, set xi^m = x^j
    # Select the new particle values based on the weights of the original
    # particles. Particles with large weight are repeated, while particles with
    # small weights may be eliminated.
    new_particle_inds = [bisect.bisect_left(cumul_weights, x)
                         for x in cumul_new_weights]
    resampled_states = particles[new_particle_inds, ]

    # 4. Perturb states (i.e., regularization)
    if pod:
        # Special case: regularization for preliminary orbit determination
        # For now, we're ignoring this and doing it the same way
        reg_delta, reg_weights = regularize_default(particles, weights)
        # 5. Redefine weights to uniform distribution
        new_particles = resampled_states
        new_particles[:, 0:6] = resampled_states[:, 0:6] + reg_delta
    else:
        reg_delta, reg_weights = regularize_default(particles, weights)
        # 5. Redefine weights to uniform distribution
        new_particles = resampled_states + reg_delta

    return new_particles, reg_weights


def get_normed_weights(ln_weights):
    ln_wts_norm = np.logaddexp.reduce(ln_weights)
    weights = np.exp(ln_weights - ln_wts_norm)
    return weights


def regularize_default(particles, weights, num_particles_out=None, dimension=6):
    """
    Perform particle regularization. This generates a perturbation from the
    particles' original values to prevent singularity issues.
    From "Bayesian Multiple Target Tracking" (2nd ed.) p. 101-102

    :param particles: Particles to reqularize. Each row is an nD particle
    :type particles: numpy.ndarray

    :param weights: 1D array of particle weight. Should be same length as number
         of particles.
    :type weights: numpy.ndarray

    :param num_particles_out: Number of particles to return, defaults to None.
        If not specified, will return the same number of particles as in the
        input particles.
    :type num_particles_out: int | None, optional

    :param dimension: Dimension of the parameter space for resampling. Assumes
        the first `dimension` columns of `particles` are the parameters to use.
        Any remaining columns are carried through without modification (e.g.,
        time columns). Default: 6
    :type dimension: int, optional

    :return: Deltas from original particles and their weights
    :rtype: (numpy.ndarray, numpy.ndarry)
    """
    num_particles_in, dim_in = particles.shape
    if num_particles_out is None:
        num_particles_out = num_particles_in
    kernel_cov = get_kernel_cov(particles[:, 0:dimension], weights)
    # print("kernel_cov:", kernel_cov)
    window_size = (4. / (num_particles_in * (dimension + 1))) ** (1. / (dimension + 4.))
    # Use a smaller window size for multi-modal distributions
    adj_windows_size = window_size / 2.
    # Generate deltas of new particles from a normal distribution with zero mean
    # and covariance according to the input particles
    delta = np.random.multivariate_normal(mean=np.zeros(dimension),
                                          cov=kernel_cov * (adj_windows_size**2),
                                          size=num_particles_out)
    weights_out = np.ones(num_particles_out) / float(num_particles_out)
    return delta, weights_out


def get_kernel_cov(kernel_mat, weights):
    """
    Get the covariance matrix of kernel_mat. This a wrapper for numpy's cov

    :param kernel_mat: Data matrix
    :type kernel_mat: numpy.ndarray

    :param weights: 1D array of weights of the data
    :type weights: numpy.ndarray

    :return: numpy.ndarray
    :rtype: Covariance matrix
    """
    return np.cov(kernel_mat.transpose(), aweights=weights, bias=True)


class LRU_Cache:
    """Simplified Least Recently Used Cache.

    Mostly stolen from http://code.activestate.com/recipes/577970-simplified-lru-cache/,
    but added a method for dynamic resizing.  The least recently used cached item is
    overwritten on a cache miss.

    Parameters:
        user_function:  A python function to cache.
        maxsize:        Maximum number of inputs to cache.  [Default: 1024]

    Example::

        >>> def slow_function(*args) # A slow-to-evaluate python function
        >>>    ...
        >>>
        >>> v1 = slow_function(*k1)  # Calling function is slow
        >>> v1 = slow_function(*k1)  # Calling again with same args is still slow
        >>> cache = LRU_Cache(slow_function)
        >>> v1 = cache(*k1)  # Returns slow_function(*k1), slowly the first time
        >>> v1 = cache(*k1)  # Returns slow_function(*k1) again, but fast this time.
    """
    def __init__(self, user_function, maxsize=1024):
        if maxsize < 0:
            raise ValueError("Invalid maxsize", maxsize)

        # Link layout:     [PREV, NEXT, KEY, RESULT]
        self.root = root = [None, None, None, None]
        self.user_function = user_function
        self.cache = cache = {}

        last = root
        for i in range(maxsize):
            key = object()
            cache[key] = last[1] = last = [last, root, key, None]
        root[0] = last

    def __call__(self, *key):
        if len(self.cache) == 0:  # useful to allow zero when profiling...
            return self.user_function(*key)
        cache = self.cache
        root = self.root
        link = cache.get(key)
        if link is not None:
            # Cache hit: move link to last position
            # print("hit")
            link_prev, link_next, _, result = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            last = root[0]
            last[1] = root[0] = link
            link[0] = last
            link[1] = root
            return result
        # Cache miss: evaluate and insert new key/value at root, then increment root
        #             so that just-evaluated value is in last position.
        # print("miss")
        result = self.user_function(*key)
        root = self.root  # re-establish root in case user_function modified it due to recursion
        root[2] = key
        root[3] = result
        oldroot = root
        root = self.root = root[1]
        root[2], oldkey = None, root[2]
        root[3], _ = None, root[3]
        del cache[oldkey]
        cache[key] = oldroot
        return result

    def resize(self, maxsize):
        """Resize the cache.

        Increasing the size of the cache is non-destructive, i.e., previously cached inputs remain
        in the cache.  Decreasing the size of the cache will necessarily remove items from the
        cache if the cache is already filled.  Items are removed in least recently used order.

        Parameters:
            maxsize:    The new maximum number of inputs to cache.
        """
        oldsize = len(self.cache)
        if maxsize == oldsize:
            return
        else:
            root = self.root
            cache = self.cache
            if maxsize < 0:
                raise ValueError("Invalid maxsize", maxsize)
            if maxsize < oldsize:
                for i in range(oldsize - maxsize):
                    # Delete root.next
                    current_next_link = root[1]
                    new_next_link = root[1] = root[1][1]
                    new_next_link[0] = root
                    del cache[current_next_link[2]]
            else:  # maxsize > oldsize:
                for i in range(maxsize - oldsize):
                    # Insert between root and root.next
                    key = object()
                    cache[key] = link = [root, root[1], key, None]
                    root[1][0] = link
                    root[1] = link


def catalog_to_apparent(
    ra, dec, t, observer=None, pmra=0, pmdec=0, parallax=0,
    skipAberration=False
):
    """
    Convert ra/dec of stars from catalog positions (J2000/ICRS) to apparent
    positions, correcting for proper motion, parallax, annual and diurnal
    aberration.

    Parameters
    ----------
    ra : array_like
        J2000 right ascension in radians.
    dec : array_like
        J2000 declination in radians.
    t : float or astropy.time.Time
        If float, then should correspond to GPS seconds; i.e., seconds since
        1980-01-06 00:00:00 UTC
    observer : Observer, optional
        Observer to use for diurnal aberration correction.  If None, then no
        diurnal aberration correction is performed.
    pmra, pmdec : array_like, optional
        proper motion in right ascension / declination, in milliarcsec per year
    parallax : array_like, optional
        annual parallax in arcseconds
    skipAberration : bool, optional
        Don't apply aberration correction.  Mostly useful during testing...

    Returns
    -------
    ara, adec : array_like
        Apparent RA and dec in radians
    """
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time
    import astropy.units as u

    if isinstance(t, Time):
        tTime = t
        t = t.gps
    else:
        tTime = Time(t, format='gps')

    # Code below modeled after SOFA iauPmpx, iauAb routines, but a bit simpler
    # since SSA requirements are only ~arcsec or so.
    # aulty = (u.au / (299792458 * u.m / u.s)).to(u.year).value
    obsPos, obsVel = get_body_barycentric_posvel('earth', tTime)
    pob = obsPos.xyz.to(u.AU).value
    vob = obsVel.xyz.to(u.m / u.s).value
    sr, cr = np.sin(ra), np.cos(ra)
    sd, cd = np.sin(dec), np.cos(dec)
    x = cr * cd
    y = sr * cd
    z = sd
    p = np.array([x, y, z]).T
    dt = (t - Time("J2000").gps) / (86400 * 365.25)
    pdrad = np.deg2rad(pmdec / (1000 * 3600))
    prrad = np.deg2rad(pmra / (1000 * 3600)) / cd
    pxrad = np.deg2rad(parallax / 3600)

    # Proper motion and parallax
    pdz = z * pdrad
    pm = np.array([
        -prrad * y - pdz * cr,
        prrad * x - pdz * sr,
        pdrad * cd
    ]).T
    p += dt * pm - pxrad[..., None] * pob
    p = normed(p)

    # Aberration
    if not skipAberration:
        if observer is not None:
            _, dvob = observer.getRV(tTime)
            vob += dvob
        p += vob / 299792458.
        p = normed(p)

    ra = np.arctan2(p[:, 1], p[:, 0])
    dec = np.arcsin(p[:, 2])
    return ra, dec


def rv_to_ntw(r, v, rcoord):
    """Convert coordinates to NTW coordinates, using r, v to define NTW system.

    T gives the projection of (rcoord - r) along V  (tangent to track)
    W gives the projection of (rcoord - r) along (V cross r) (normal to plane)
    N gives the projection of (rcoord - r) along (V cross (V cross r))
        (in plane, perpendicular to T)

    Parameters
    ----------
    r : array_like (n, 3)
        central positions defining coordinate system
    v : array_like (n, 3)
        velocity defining coordinate system
    rcoord : array_like (n, 3)
        positions to transform to NTW coordinates

    Returns
    -------
    ntw : array_like (n, 3)
        n, t, w positions
    """
    def dot(x, y):
        return np.einsum('...i, ...i', x, y)
    dr = rcoord - r
    t = dot(normed(v), dr)
    w = dot(normed(np.cross(r, v)), dr)
    n = dot(normed(np.cross(v, np.cross(r, v))), dr)
    return np.array([n, t, w]).T.copy()


def ntw_to_r(r, v, ntw, relative=False):
    """Convert NTW coordinates to cartesian coordinates, using r, v to define
    NTW system.

    T gives the projection of (rcoord - r) along V  (tangent to track)
    W gives the projection of (rcoord - r) along (V cross r) (normal to plane)
    N gives the projection of (rcoord - r) along (V cross (V cross r))
        (in plane, perpendicular to T)

    Parameters
    ----------
    r : array_like (n, 3)
        central positions defining coordinate system
    v : array_like (n, 3)
        velocity defining coordinate system
    ntw : array_like (n, 3)
        ntw coordinates to transform to cartesian coordinates
    relative : bool
        if True, just rotate the NTW coordinates to Cartesian; do not offset
        the origin so that NTW = 0 -> Cartesian r.

    Returns
    -------
    r : array_like (n, 3)
        cartesian x, y, z coordinates
    """
    tvec = normed(v)
    wvec = normed(np.cross(r, v))
    nvec = normed(np.cross(tvec, wvec))
    mat = np.array([nvec, tvec, wvec])
    ret = np.dot(ntw, mat)
    if not relative:
        ret += r
    return ret


def lb_to_tp(lb, b):
    """Convert lb-like coordinates to theta-phi like coordinates.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    This just sets theta = pi - b and renames lb -> phi.  Everything is in
    radians.
    """
    return np.pi / 2 - b, lb % (2 * np.pi)


def tp_to_lb(t, p):
    """Convert theta-phi-like coordinates to lb-like coordinates.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    This just sets b = pi - theta and renames phi -> lb.  Everything is in
    radians.
    """
    return p % (2 * np.pi), np.pi / 2 - t


def tp_to_unit(t, p):
    """Convert theta-phi-like coordinates to unit vectors.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    Everything is in radians.
    """
    z = np.cos(t)
    x = np.cos(p) * np.sin(t)
    y = np.sin(p) * np.sin(t)
    return np.concatenate([q[..., np.newaxis] for q in (x, y, z)], axis=-1)


def lb_to_unit(r, d):
    """Convert lb-like coordinates to unit vectors.

    Everything is in radians.
    """
    return tp_to_unit(*lb_to_tp(r, d))


def unit_to_tp(unit):
    """Convert unit vectors to theta-phi-like coordinates.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    Everything is in radians.
    """
    norm = np.sqrt(np.sum(unit**2., axis=-1))
    unit = unit / norm[..., None]
    t = np.arccos(unit[..., 2])
    p = np.arctan2(unit[..., 1], unit[..., 0])
    return t, p % (2 * np.pi)


def xyz_to_tp(x, y, z):
    """Convert x, y, z vectors to theta-phi-like coordinates.

    Here 'theta-phi' coordinates refers to the system where theta is the
    angle between zenith and the point in question, and phi is the
    corresponding azimuthal angle.

    Everything is in radians.
    """
    norm = np.sqrt(x**2 + y**2 + z**2)
    t = np.arccos(z / norm)
    p = np.arctan2(y / norm, x / norm)
    return t, p % (2 * np.pi)


def unit_to_lb(unit):
    """Convert unit vectors to lb-like coordinates.

    Everything is in radians.
    """
    return tp_to_lb(*unit_to_tp(unit))


def xyz_to_lb(x, y, z):
    """Convert x, y, z vectors to lb-like coordinates.

    Everything is in radians.
    """
    return tp_to_lb(*xyz_to_tp(x, y, z))


def lb_to_tan(lb, b, mul=None, mub=None, lcen=None, bcen=None):
    """Convert lb-like coordinates & proper motions to orthographic tangent plane.

    Everything is in radians.  If mul is None (default), transformed
    proper motions will not be returned.  The tangent plane is always chosen
    so that +Y is towards b = 90, and +X is towards +lb.

    Parameters
    ----------
    lb : array_like (n)
        right ascension of point
    b : array_like (n)
        declination of point
    mul : array_like (n)
        proper motion in ra of point (arbitrary units)
        rate of change in lb is mul / np.cos(b); i.e., length of proper motion
        vector on sphere is np.hypot(mul, mub)
    mub : array_like (n)
        proper motion in dec of point (arbitrary units)
    lcen : array_like (n)
        right ascension to use for center of tangent plane
        if None, use spherical mean of (lb, b)
    bcen : array_like (n)
        declination to use for center of tangent plane

    Returns
    -------
    if mul is None, (x, y) otherwise (x, y, vx, vy)
    x : array_like (n)
        x coordinate of tangent plane projection of lb, b
    y : array_like (n)
        y coordinate of tangent plane projection of lb, b
    vx : array_like (n)
        x coordinate of tangent plane projection of (mul, mub)
    vy : array_like (n)
        y coordinate of tangent plane projection of (mul, mub)
    """
    up = np.array([0, 0, 1])
    unit = lb_to_unit(lb, b)
    if lcen is None:
        lcen, bcen = unit_to_lb(np.mean(unit, axis=0).reshape(1, -1))
    unitcen = lb_to_unit(lcen, bcen)
    if len(unitcen.shape) == 1:
        unitcen = unitcen[None, :]
    rahat = np.cross(up, unitcen)
    m = norm(rahat) < 1e-10
    rahat[m, :] = np.array([1, 0, 0])[None, :]
    rahat /= np.sqrt(np.sum(rahat**2, axis=1, keepdims=True))
    dechat = np.cross(unitcen, rahat)
    dechat /= np.sqrt(np.sum(dechat**2, axis=1, keepdims=True))
    xx = np.sum(rahat * unit, axis=1)
    yy = np.sum(dechat * unit, axis=1)
    res = (xx, yy)
    if mul is not None:
        rahat2 = np.cross(up, unit)
        rahat2 /= np.sqrt(np.sum(rahat2**2, axis=1, keepdims=True))
        dechat2 = np.cross(unit, rahat2)
        dechat2 /= np.sqrt(np.sum(dechat2**2, axis=1, keepdims=True))
        vv = mul[:, None] * rahat2 + mub[:, None] * dechat2
        vx = np.sum(rahat * vv, axis=1)
        vy = np.sum(dechat * vv, axis=1)
        rr = np.hypot(xx, yy)
        m = np.abs(rr) > 1e-9
        vr = (vx[m] * xx[m] + vy[m] * yy[m]) / rr[m]
        va = (vy[m] * xx[m] - vx[m] * yy[m]) / rr[m]
        vr /= np.sum(unitcen[m, :] * unit[m, :], axis=1)
        vx[m] = vr * xx[m] / rr[m] - va * yy[m] / rr[m]
        vy[m] = vr * yy[m] / rr[m] + va * xx[m] / rr[m]
        res = res + (vx, vy)
    return res


def tan_to_lb(xx, yy, lcen, bcen):
    """Convert orthographic tangent plane coordinates to lb coordinates.

    Parameters
    ----------
    xx : array_like (n)
        tangent plane x coordinate (radians)
    yy : array_like (n)
        targent plane y coordinate (radians
    lcen : float, array_like (n)
        right ascension of center of tangent plane
    bcen : float, array_linke (n)
        declination of center of tangent plane

    Returns
    -------
    lb : array_like (n)
        right ascension corresponding to xx, yy
    b : array_like (n)
        declination corresponding to xx, yy
    """
    unitcen = lb_to_unit(lcen, bcen)
    if len(unitcen.shape) == 1:
        unitcen = unitcen[None, :]
    up = np.array([0, 0, 1])
    rahat = np.cross(up, unitcen)
    m = norm(rahat) < 1e-10
    rahat[m, :] = np.array([1, 0, 0])[None, :]
    rahat /= np.sqrt(np.sum(rahat**2, axis=1, keepdims=True))
    dechat = np.cross(unitcen, rahat)
    dechat /= np.sqrt(np.sum(dechat**2, axis=1, keepdims=True))
    xcoord = xx
    ycoord = yy
    zcoord = np.sqrt(1 - xcoord**2 - ycoord**2)
    unit = (xcoord.reshape(-1, 1) * rahat + ycoord.reshape(-1, 1) * dechat + zcoord.reshape(-1, 1) * unitcen)
    return unit_to_lb(unit)


def sample_points(x, C, npts, sqrt=False):
    """Sample points around x according to covariance matrix.

    Parameters
    ----------
    x : array_like (n)
        point to sample around
    C : array_like (n, n)
        covariance matrix corresponding to x
    npts : int
        number of points to sample
    sqrt : bool
        use sqrt(C) rather than an SVD.  The SVD is often more stable.

    Returns
    -------
    Gaussian samples around x corresponding to covariance matrix C
    """
    n = C.shape[0]
    xa = np.random.randn(npts, n)
    if not sqrt:
        sqrtdiag = np.sqrt(np.diag(C))
        scalecovar = sqrtdiag[None, :] * sqrtdiag[:, None]
        uu, ss, vvh = np.linalg.svd(C / scalecovar)
        if np.any(ss < 0):
            raise ValueError('negative eigenvalues in C!')
        sqrtC = uu.dot(np.diag(ss**0.5)).dot(vvh)
    else:
        sqrtC = C
        sqrtdiag = 1
    xa = sqrtC.dot(xa.T).T
    if not sqrt:
        xa *= sqrtdiag[None, :]
    xa += x[None, :]
    return xa


def sigma_points(f, x, C, scale=1, fixed_dimensions=None):
    """Compute f(sigma points) for sigma points of C around x.

    There are many possible definitions of sigma points.  This one
    takes the singular value decomposition of C and uses
    the eigenvectors times sqrt(dimension)*scale*(+/-1) as the sigma points.
    It then evaluates the given function f at x plus those sigma points.

    Parameters
    ----------
    f : function
        the function to evaluate at the sigma points
    x : array_like (n)
        the central value to evaluate the function around
    C : array_like (n, n)
        the covariance matrix corresponding to x
    scale : float
        return scale-sigma points rather than n-sigma points.  e.g.,
        for 5 sigma, set scale = 5.
    fixed_dimensions : array_like, (n), bool
        boolean array specifying dimensions of x that are fixed
        and not specified in C
    """
    if fixed_dimensions is None:
        fixed = np.zeros(len(x), dtype='bool')
    else:
        fixed = np.array(fixed_dimensions).astype('bool') is not False
    free = ~fixed
    # sqrtC = linalg.sqrtm(C)
    sqrtdiag = np.sqrt(np.diag(C))
    scalecovar = sqrtdiag.reshape(1, -1) * sqrtdiag.reshape(-1, 1)
    uu, ss, vvh = np.linalg.svd(C / scalecovar)
    sqrtC = uu.dot(np.diag(ss**0.5)).dot(vvh)
    n = C.shape[0]
    xsigma = [x[free]] + [x[free] + np.sqrt(n) * sgn * scale * sqrtCvec * sqrtdiag
                          for sqrtCvec in sqrtC for sgn in [-1, 1]]
    xsigma = np.array(xsigma)
    xsigmaall = np.zeros((xsigma.shape[0], xsigma.shape[1] + np.sum(fixed)),
                         dtype=xsigma.dtype)
    xsigmaall[:, free] = xsigma
    xsigmaall[:, fixed] = x[None, fixed]
    if f is not None:
        fsigma = f(xsigmaall)
    else:
        fsigma = xsigmaall
    return fsigma


def unscented_transform_mean_covar(f, x, C, scale=1):
    """Compute mean and covar using unscented transform given a transformation
    f, a point x, and a covariance C.

    This uses the sigma point convention from sigma_points.  It assumes that
    f(sigma_points)[i] is f evaluated at the ith sigma point.  If f does
    not obey this convention, this function will produce undefined results.

    Parameters
    ----------
    f : function
        the function to evaluate at the sigma points.
    x : array_like (n)
        the central value to evaluate the function around
    C : array_like (n, n)
        the covariance matrix corresponding to x
    scale : float
        return scale-sigma points rather than n-sigma points.  e.g.,
        for 5 sigma, set scale = 5.
    """
    fsigma = sigma_points(f, x, C, scale=scale)
    mean = fsigma[0]
    dmean = fsigma[1:, ...] - mean[:, ...]
    covar = np.cov(dmean, ddof=0)
    return mean, covar


def _gpsToTT(t):
    # Check if t is astropy Time object if so convert to GPS seconds if not assume GPS seconds.  Convert to TT seconds by adding 51.184.
    # Divide by 86400 to get TT days.
    # Then add the TT time of the GPS epoch, expressed as an MJD, which
    # is 44244.0
    if isinstance(t, Time):
        t = t.gps
    return 44244.0 + (t + 51.184) / 86400


def sunPos(t, fast=True):
    """Compute GCRF position of the sun.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC
    fast : bool
        Use fast approximation?

    Returns
    -------
    r : array_like (n)
        position in meters
    """
    if isinstance(t, Time):
        t = t.gps
    if fast:
        # MG section 3.3.2
        T = (_gpsToTT(t) - 51544.5) / 36525.0
        M = 6.239998880168239 + 628.3019326367721 * T
        lam = (4.938234585592756 + M + 0.03341335890206922 * np.sin(M) + 0.00034906585039886593 * np.sin(2 * M))
        rs = (149.619 - 2.499 * np.cos(M) - 0.021 * np.cos(2 * M)) * 1e9
        obliquity = 0.40909280420293637
        co, so = np.cos(obliquity), np.sin(obliquity)
        cl, sl = np.cos(lam), np.sin(lam)
        r = rs * np.array([cl, sl * co, sl * so])
    else:
        pvh, _ = erfa.epv00(2400000.5, _gpsToTT(t))
        r = pvh['p'] * -149597870700  # AU -> m
    return r


def moonPos(t):
    """Compute GCRF position of the moon.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    r : array_like (n)
        position in meters
    """
    if isinstance(t, Time):
        t = t.gps
    # MG section 3.3.2
    T = (_gpsToTT(t) - 51544.5) / 36525.0
    # fundamental arguments (3.47)
    L0 = 3.810335976843669 + 8399.684719711557 * T
    lb = 2.3555473221057053 + 8328.69142518676 * T
    lp = 6.23999591310851 + 628.3019403162209 * T
    D = 5.198467889454092 + 7771.377143901714 * T
    F = 1.6279179861529427 + 8433.46617912181 * T
    # moon longitude (3.48)
    dL = (22640 * np.sin(lb) + 769 * np.sin(2 * lb) - 4586 * np.sin(lb - 2 * D) + 2370 * np.sin(2 * D) - 668 * np.sin(lp) - 412 * np.sin(2 * F) - 212 * np.sin(2 * lb - 2 * D) - 206 * np.sin(lb + lp - 2 * D) + 192 * np.sin(lb + 2 * D) - 165 * np.sin(lp - 2 * D) + 148 * np.sin(lb - lp) - 125 * np.sin(D) - 110 * np.sin(lb + lp) - 55 * np.sin(2 * F - 2 * D))

    L = L0 + np.deg2rad(dL / 3600)
    # moon latitude (3.49)
    beta = np.deg2rad((
        18520 * np.sin(F + L - L0 + np.deg2rad((412 * np.sin(2 * F) + 541 * np.sin(lp)) / 3600)) - 526 * np.sin(F - 2 * D) + 44 * np.sin(lb + F - 2 * D) - 31 * np.sin(-lb + F - 2 * D) - 25 * np.sin(-2 * lb + F) - 23 * np.sin(lp + F - 2 * D) + 21 * np.sin(-lb + F) + 11 * np.sin(-lp + F - 2 * D)
    ) / 3600)
    # moon distance (3.50)
    r = (
        385000 - 20905 * np.cos(lb) - 3699 * np.cos(2 * D - lb) - 2956 * np.cos(2 * D) - 570 * np.cos(2 * lb) + 246 * np.cos(2 * lb - 2 * D) - 205 * np.cos(lp - 2 * D) - 171 * np.cos(lb + 2 * D) - 152 * np.cos(lb + lp - 2 * D)
    ) * 1e3
    r_ecliptic = r * np.array([
        np.cos(L) * np.cos(beta),
        np.sin(L) * np.cos(beta),
        np.sin(beta)
    ])
    obliquity = 0.40909280420293637
    co, so = np.cos(obliquity), np.sin(obliquity)
    rot = np.array([[1, 0, 0], [0, co, so], [0, -so, co]])
    return rot.T @ r_ecliptic


def iers_interp(t):
    """Interpolate IERS values

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    dut1: array_like (n)
        Time difference UT1 - TT in days.
    pmx, pmy : array_like (n)
        Polar motion values in radians.
    """
    from astropy.utils import iers
    from scipy.interpolate import interp1d
    from astropy import units as u
    if isinstance(t, Time):
        t = t.gps
    if not hasattr(iers_interp, '_interp'):
        table = iers.earth_orientation_table.get()
        ts = Time(table['MJD'], format='mjd', scale='utc')
        tgps = ts.gps
        dut1tt = ts.ut1.mjd - ts.tt.mjd
        pmx = table['PM_x'].to(u.rad).value
        pmy = table['PM_y'].to(u.rad).value
        iers_interp._interp = interp1d(
            tgps,
            np.array([dut1tt, pmx, pmy]),
            bounds_error=False,
            fill_value=(
                np.array([dut1tt[0], pmx[0], pmy[0]]),
                np.array([dut1tt[-1], pmx[-1], pmy[-1]])
            )
        )
    return iers_interp._interp(t)


def gcrf_to_teme(t):
    """ Return the rotation matrix that converts GCRS cartesian coordinates
    to TEME cartesian coordinates.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    rot : array (3,3)
        Rotation matrix to apply to GCRS to yield TEME.
    """
    if isinstance(t, Time):
        t = t.gps
    d_ut1_tt_mjd, _, _ = iers_interp(t)
    mjd_tt = _gpsToTT(t)
    ut1 = mjd_tt + d_ut1_tt_mjd
    gst = erfa.gmst82(2400000.5, ut1)
    era = erfa.era00(2400000.5, ut1)
    c2i = erfa.c2i00b(2400000.5, mjd_tt)
    return erfa.rxr(erfa.rv2m([0, 0, era - gst]), c2i)


def teme_to_gcrf(t):
    """ Return the rotation matrix that converts TEME cartesian coordinates
    to GCRS cartesian coordinates.

    Parameters
    ----------
    t : float or astropy.time.Time
        If float or array of float, then should correspond to GPS seconds; i.e.,
        seconds since 1980-01-06 00:00:00 UTC

    Returns
    -------
    rot : array (3,3)
        Rotation matrix to apply to TEME to yield GCRS.
    """
    return erfa.tr(gcrf_to_teme(t))


def gcrf_to_lunar(r, times):
    from .body import MoonPosition

    class MoonRotator:
        def __init__(self):
            self.mpm = MoonPosition()

        def __call__(self, r, t):
            rmoon = self.mpm(t)
            vmoon = (self.mpm(t + 5.0) - self.mpm(t - 5.0)) / 10.
            xhat = normed(rmoon.T).T
            vpar = np.einsum("ab,ab->b", xhat, vmoon) * xhat
            vperp = vmoon - vpar
            yhat = normed(vperp.T).T
            zhat = np.cross(xhat, yhat, axisa=0, axisb=0).T
            R = np.empty((3, 3, len(t)))
            R[0] = xhat
            R[1] = yhat
            R[2] = zhat
            return np.einsum("abc,cb->ca", R, r)
    rotator = MoonRotator()
    if isinstance(times, Time):
        times = times.gps
    return rotator(r, times)


def gcrf_to_stationary_lunar(r, times):
    from .body import get_body
    return gcrf_to_lunar(r, times) - gcrf_to_lunar(get_body('moon').position(times).T, times)


def gcrf_to_ecef(r_gcrf, t):
    if isinstance(t, Time):
        t = t.gps
    rotation_angles = WGS84_EARTH_OMEGA * (t - t[0])
    cos_thetas = np.cos(rotation_angles)
    sin_thetas = np.sin(rotation_angles)

    # Create an array of 3x3 rotation matrices
    Rz = np.array([[cos_thetas, -sin_thetas, np.zeros_like(cos_thetas)],
                  [sin_thetas, cos_thetas, np.zeros_like(cos_thetas)],
                  [np.zeros_like(cos_thetas), np.zeros_like(cos_thetas), np.ones_like(cos_thetas)]]).T

    # Apply the rotation matrices to all rows of r_gcrf simultaneously
    r_ecef = np.einsum('ijk,ik->ij', Rz, r_gcrf)
    return r_ecef


# Stolen from https://github.com/lsst/utils/blob/main/python/lsst/utils/wrappers.py
INTRINSIC_SPECIAL_ATTRIBUTES = frozenset(
    (
        "__qualname__",
        "__module__",
        "__metaclass__",
        "__dict__",
        "__weakref__",
        "__class__",
        "__subclasshook__",
        "__name__",
        "__doc__",
    )
)


def isAttributeSafeToTransfer(name, value):
    # Stolen from https://github.com/lsst/utils/blob/main/python/lsst/utils/wrappers.py
    """Return True if an attribute is safe to monkeypatch-transfer to another
    class.
    This rejects special methods that are defined automatically for all
    classes, leaving only those explicitly defined in a class decorated by
    `continueClass` or registered with an instance of `TemplateMeta`.
    """
    if name.startswith("__") and (
        value is getattr(object, name, None) or name in INTRINSIC_SPECIAL_ATTRIBUTES
    ):
        return False
    return True


def continueClass(cls):
    # Stolen from https://github.com/lsst/utils/blob/main/python/lsst/utils/wrappers.py
    """Re-open the decorated class, adding any new definitions into the
    original.
    For example:
    .. code-block:: python
        class Foo:
            pass
        @continueClass
        class Foo:
            def run(self):
                return None
    is equivalent to:
    .. code-block:: python
        class Foo:
            def run(self):
                return None
    .. warning::
        Python's built-in `super` function does not behave properly in classes
        decorated with `continueClass`.  Base class methods must be invoked
        directly using their explicit types instead.
    """
    orig = getattr(sys.modules[cls.__module__], cls.__name__)
    for name in dir(cls):
        # Common descriptors like classmethod and staticmethod can only be
        # accessed without invoking their magic if we use __dict__; if we use
        # getattr on those we'll get e.g. a bound method instance on the dummy
        # class rather than a classmethod instance we can put on the target
        # class.
        attr = cls.__dict__.get(name, None) or getattr(cls, name)
        if isAttributeSafeToTransfer(name, attr):
            setattr(orig, name, attr)
    return orig


def get_times(duration=(30, 'day'), freq=(1, 'hr'), t=Time("2025-01-01", scale='utc')):
    """
    Calculate a list of times spaced equally apart over a specified duration.

    Parameters
    ----------
    duration: int
        The length of time to calculate times for.
    freq: int, unit: str
        frequency of time outputs in units provided
    t: ssapy.utils.Time, optional
        The starting time. Default is "2025-01-01".

    Returns
    -------
    times: array-like
        A list of times spaced equally apart over the specified duration.
    """
    if isinstance(t, str):
        t = Time(t, scale='utc')
    unit_dict = {'second': 1, 'sec': 1, 's': 1, 'minute': 60, 'min': 60, 'hour': 3600, 'hr': 3600, 'h': 3600, 'day': 86400, 'd': 86400, 'week': 604800, 'month': 2630016, 'mo': 2630016, 'year': 31557600, 'yr': 31557600}
    dur_val, dur_unit = duration
    freq_val, freq_unit = freq
    if dur_unit[-1] == 's' and len(dur_unit) > 1:
        dur_unit = dur_unit[:-1]
    if freq_unit[-1] == 's' and len(freq_unit) > 1:
        freq_unit = freq_unit[:-1]
    if dur_unit.lower() not in unit_dict:
        raise ValueError(f'Error, {dur_unit} is not a valid time unit. Valid options are: {", ".join(unit_dict.keys())}.')
    if freq_unit.lower() not in unit_dict:
        raise ValueError(f'Error, {freq_unit} is not a valid time unit. Valid options are: {", ".join(unit_dict.keys())}.')
    dur_seconds = dur_val * unit_dict[dur_unit.lower()]
    freq_seconds = freq_val * unit_dict[freq_unit.lower()]
    timesteps = int(dur_seconds / freq_seconds) + 1

    times = t + np.linspace(0, dur_seconds, timesteps) / unit_dict['day'] * u.day
    return times


def interpolate_points_between(r, m):
    """
    Interpolates points between the given points.

    Args:
        r: An (n, 3) numpy array of the original points.
        m: The number of points to interpolate between each pair of points.

    Returns:
        An (n * m) numpy array of the interpolated points.
    """
    n = len(r)
    interpolated_points = np.empty((0, 3))
    for i in range(n):
        # Generate 1D linearly spaced arrays along each axis using np.linspace()
        x = np.linspace(r[i, 0], r[i, 0], m)
        y = np.linspace(r[i, 1], r[i, 1], m)
        z = np.linspace(r[i, 2], r[i, 2], m)
        interpolated_points = np.vstack((interpolated_points, np.array([x, y, z]).T))
    return np.vstack((interpolated_points, r[-1]))


def check_lunar_collision(r, times, m=1000):
    from .body import get_body
    """
    Checks if the trajectory of a particle intersects with the Moon.

    Parameters
    ----------
    r : np.array
        The particle's trajectory in Cartesian coordinates.
    times : an array of astropy.Time where r points are calculated.
    m : int, optional
        The number of points to interpolate between. Defaults to 1000.

    Returns
    -------
    np.array
        Indexes where collision occurs.
    """
    # For a time step of 1 hour, m=1000 will be sensitive of collisions up to 482 km/s
    new_r = interpolate_points_between(r, m)
    # Time span of integration
    times_new = Time(np.linspace(times.decimalyear[0], times.decimalyear[-1], int(len(times) * m + 1)), format='decimalyear', scale='utc')
    moon_r = get_body('moon').position(times_new).T
    collision_index = np.where(np.linalg.norm(new_r - moon_r, axis=-1) < MOON_RADIUS)
    if np.size(collision_index) > 0:
        collision_times = times_new[collision_index]
        nearest_indices = find_nearest_indices(times.decimalyear, collision_times.decimalyear)
        return np.array(list(set(nearest_indices)))
    else:
        return []


def find_nearest_indices(A, B):
    # Calculate the absolute differences between B and A using broadcasting
    abs_diff = np.abs(B[:, np.newaxis] - A)
    # Find the index of the minimum absolute difference for each element of B
    nearest_indices = np.argmin(abs_diff, axis=1)
    return nearest_indices


def integrate_orbit_best_model(r=None, v=None, t=None, koe=None, duration=None, freq=None, start_date=None, mass=250, area=0.22):
    """
    Integrate satellite orbit trajectory using either (r, v) or Keplerian orbital elements.

    Parameters:
    -----------
    r : array_like, optional
        Initial position vector [x, y, z] in meters.
    v : array_like, optional
        Initial velocity vector [vx, vy, vz] in meters per second.
    t : array_like, optional
        Array of time values for integration in seconds since the start date.
    koe : dict, optional
        Dictionary containing Keplerian orbital elements:
            - 'a': Semi-major axis in meters
            - 'e': Eccentricity
            - 'i': Inclination in radians
            - 'trueAnomaly': True anomaly in radians
            - 'pa': Argument of perigee in radians
            - 'raan': Right ascension of ascending node in radians
    duration : tuple, optional
        Duration of integration (value, unit), e.g., (30, 'day').
    freq : tuple, optional
        Frequency of integration steps (value, unit), e.g., (1, 'hr').
    start_date : str, optional
        Start date in the format "YYYY-MM-DD" for time calculations.
    mass: float, optional
        Mass of the orbiting object in kg.
    area: float, optional
        Cross sectional area of the orbiting object in m^2.
    Returns:
    --------
    r : ndarray
        Array of position vectors [x, y, z] over the integrated trajectory.
    v : ndarray
        Array of velocity vectors [vx, vy, vz] corresponding to the position vectors.

    Notes:
    ------
    The function integrates the satellite orbit trajectory using either initial position and
    velocity vectors or Keplerian orbital elements.

    Either (r, v) or 'koe' must be provided, but not both.
    If 'koe' are provided, Keplerian-to-Cartesian conversion must be implemented.

    If 't' is not provided, 'duration', 'freq', and 'start_date' must be provided for time calculation.
    """

    from .accel import AccelKepler, AccelEarthRad, AccelSolRad, AccelDrag
    from .body import get_body
    from .compute import rv
    from .gravity import AccelThirdBody, AccelHarmonic
    from .orbit import Orbit
    from .propagator import RK78Propagator

    # Time span of integration #Only takes integer number of days. Unless providing your own time object.
    if t is None and start_date is None:
        raise ValueError("Either an array of times 't' or a 'start_date', 'duration' and 'freq' must be provided.")
    if t is None:
        t = get_times(duration=duration, freq=freq, t=start_date)

    if (r is None or v is None) and koe is None:
        raise ValueError("Either (r, v) or 'koe' must be provided.")

    if (r is not None and v is not None) and koe is not None:
        raise ValueError("Both (r, v) and 'koe' cannot be provided simultaneously.")

    # Properties of sat
    kwargs = dict(
        mass=mass,  # [kg] --> was 1e4
        area=area,  # [m^2]
        CD=2.3,  # Drag coefficient
        CR=1.3,  # Radiation pressure coefficient
    )

    if koe is not None:
        # Extract Keplerian orbital elements from the dictionary
        a = koe.get('a', None)
        e = koe.get('e', None)
        i = koe.get('i', None)
        trueAnomaly = koe.get('trueAnomaly', None)
        pa = koe.get('pa', None)
        raan = koe.get('raan', None)
        if None in (a, e, i, trueAnomaly, pa, raan):
            raise ValueError("Incomplete koe dictionary.")
        kElements = [a, e, i, pa, raan, trueAnomaly]
        orbit = Orbit.fromKeplerianElements(*kElements, t[0], propkw=kwargs)
    elif (r is not None or v is not None):
        orbit = Orbit(r, v, t[0], propkw=kwargs)

    # Most accurate trajectory
    earth = get_body("earth")
    moon = get_body("moon")
    sun = get_body("sun")

    # Accelerations - pass a body object or string of bo
    # dy name.
    aEarth = AccelKepler() + AccelHarmonic(earth)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon)
    aSun = AccelThirdBody(sun)
    aSolRad = AccelSolRad()
    aEarthRad = AccelEarthRad()
    aDrag = AccelDrag()
    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad + aDrag
    prop = RK78Propagator(accel, h=10.0)

    # Calculate entire satellite trajectory
    try:
        r, v = rv(orbit, t, prop)
        return r, v
    except (RuntimeError, ValueError) as err:
        print(err)
        return np.nan, np.nan

import pytest
import numpy as np

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude, Latitude
from astropy.table import QTable

import ssapy
import ssapy.utils as utils
from .ssapy_test_helpers import timer
from ssapy.constants import RGEO, VGEO


def random_vector(min_mag, max_mag):
    """Generate a random 3D vector with magnitude between min_mag and max_mag."""
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)
    magnitude = np.random.uniform(min_mag, max_mag)
    return vec * magnitude


@pytest.mark.slow
@timer
def test_particles():
    np.random.seed(562)

    time0 = Time(2458316., format='jd')
    lon, lat, elevation = 100.0, 33.0, 1300.0
    observer = ssapy.EarthObserver(lon, lat, elevation)

    # True orbit
    r = random_vector(RGEO * 0.9, RGEO * 1.1)
    v = random_vector(VGEO * 0.9, VGEO * 1.1)
    orbit = ssapy.Orbit(r, v, time0)

    particles = []
    epoch_times = []
    lnprobs = []
    time = time0
    for _ in range(2):  # two epochs
        epoch_times.append(time)
        times = time + np.linspace(0, 10, 5) * u.h
        ra, dec = ssapy.radec(orbit, times, observer=observer)
        r_station, v_station = observer.getRV(times)

        arc = QTable()
        arc['ra'] = Longitude(ra * u.rad)
        arc['dec'] = Latitude(dec * u.rad)
        arc['rStation_GCRF'] = r_station * u.m
        arc['vStation_GCRF'] = v_station * u.m / u.s
        arc['time'] = Time(times)
        arc['sigma'] = np.ones(5) * u.arcsec

        initializer = ssapy.GaussianRVInitializer(r, v, rSigma=0.1 * RGEO, vSigma=0.1 * VGEO)
        priors = [ssapy.rvsampler.RPrior(RGEO, RGEO * 0.2), ssapy.rvsampler.APrior(RGEO, RGEO * 0.2)]
        rvprob = ssapy.RVProbability(arc, time, priors=priors)
        sampler = ssapy.EmceeSampler(rvprob, initializer, nWalker=50)
        chain, lnprob, lnprior = sampler.sample(nBurn=200, nStep=50)
        chain, lnprob, lnprior = utils.cluster_emcee_walkers(chain, lnprob, lnprior)

        particles.append(ssapy.Particles(chain, rvprob, lnpriors=lnprior))
        lnprobs.append(lnprob.ravel())
        time += 12 * u.h

    epoch_times = Time(epoch_times)

    for i, p in enumerate(particles):
        assert np.isclose(p.epoch, epoch_times[i].gps)
        lnL = p.lnlike(p.orbits)
        assert lnL.shape == (p.orbits.shape[0],)
        mean = p.mean()
        assert mean.shape == (6,)
        np.testing.assert_allclose(lnL + p.lnpriors, lnprobs[i], rtol=1e-4, atol=1e-4)

    # Fuse test
    p0, p1 = particles
    r0, v0 = p0.mean()[0:3], p0.mean()[3:6]
    p0.fuse(p1)
    fused_mean = p0.mean()
    assert fused_mean.shape == (6,)
    np.testing.assert_allclose(fused_mean[0:3], r0, rtol=0.1)
    np.testing.assert_allclose(fused_mean[3:6], v0, rtol=0.1)

    # Reset test
    p0.reset_to_pseudo_prior()
    r_reset = p0.mean()[0:3]
    assert not np.allclose(r_reset, r0)

    print("All ssapy.particles methods tested successfully.")


if __name__ == '__main__':
    test_particles()

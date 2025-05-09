import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude, Latitude
from astropy.table import QTable

import ssapy
from ssapy.constants import RGEO, VGEO
from ssapy.particles import Particles
from ssapy.rvsampler import RPrior, APrior, GaussianRVInitializer
from ssapy.utils import cluster_emcee_walkers, get_normed_weights

# Helper for test
def random_vector(min_mag, max_mag):
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)
    magnitude = np.random.uniform(min_mag, max_mag)
    return vec * magnitude

@pytest.fixture
def prepared_particles():
    np.random.seed(42)
    time = Time(2458316., format='jd')
    observer = ssapy.EarthObserver(100.0, 33.0, 1300.0)

    r = random_vector(RGEO * 0.9, RGEO * 1.1)
    v = random_vector(VGEO * 0.9, VGEO * 1.1)
    orbit = ssapy.Orbit(r, v, time)
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

    priors = [RPrior(RGEO, RGEO * 0.2), APrior(RGEO, RGEO * 0.2)]
    initializer = GaussianRVInitializer(r, v, rSigma=0.1 * RGEO, vSigma=0.1 * VGEO)
    rvprob = ssapy.RVProbability(arc, time, priors=priors)
    sampler = ssapy.EmceeSampler(rvprob, initializer, nWalker=30)
    chain, lnprob, lnprior = sampler.sample(nBurn=100, nStep=10)
    chain, lnprob, lnprior = cluster_emcee_walkers(chain, lnprob, lnprior)

    return Particles(chain, rvprob, lnpriors=lnprior), rvprob

def test_repr(prepared_particles):
    particles, _ = prepared_particles
    rep = repr(particles)
    assert "Particles(r=" in rep

def test_lnlike_shape(prepared_particles):
    particles, _ = prepared_particles
    lnL = particles.lnlike(particles.orbits)
    assert lnL.shape == (particles.num_particles,)

def test_draw_orbit(prepared_particles):
    particles, _ = prepared_particles
    orbit = particles.draw_orbit()
    assert len(orbit) == 1
    assert hasattr(orbit[0], 'r') and hasattr(orbit[0], 'v')

def test_move(prepared_particles):
    particles, _ = prepared_particles
    moved = particles.move(particles.rvprobability.epoch + 1 * u.h)
    assert moved.shape == (particles.num_particles, 6)

def test_fuse_and_reweight(prepared_particles):
    p0, _ = prepared_particles
    p1, _ = prepared_particles
    old_particles = p0.particles.copy()
    p0.fuse(p1)
    assert p0.particles.shape[1] == 6
    assert not np.allclose(p0.particles, old_particles)

def test_resample(prepared_particles):
    particles, _ = prepared_particles
    initial_count = particles.particles.shape[0]
    particles.resample(num_particles=initial_count)
    assert particles.particles.shape[0] == initial_count

def test_reset_to_pseudo_prior(prepared_particles):
    particles, _ = prepared_particles
    original = particles.mean().copy()
    particles.particles += np.random.normal(0, 1e3, particles.particles.shape)
    particles.reset_to_pseudo_prior()
    reset = particles.mean()
    assert not np.allclose(original, reset)

def test_invalid_resample_raises(prepared_particles):
    particles, _ = prepared_particles
    with pytest.raises(ValueError):
        particles.resample(num_particles=particles.num_particles + 10)

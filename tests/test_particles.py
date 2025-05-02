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


@pytest.mark.slow
@timer
def test_particles():
    np.random.seed(562)

    time0 = Time(2458316., format='jd')

    # Setup observer location
    lon = 100.0
    lat = 33.0
    elevation = 1300.0
    earthObserver = ssapy.EarthObserver(lon, lat, elevation)

    #
    # Create orbit for the target
    #
    # a = RGEO * 0.98
    # e = 0.01
    # i = 0.001
    # pa = 0.0
    # raan = 1.2
    # trueAnomaly = 2.03
    # orbit = Orbit.fromKeplerianElements(a, e, i, raan, pa, trueAnomaly, time0)

    r = np.random.uniform(RGEO*0.9, RGEO*1.1)
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi)
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    r = np.array([x, y, z])
    # Pick a velocity near VGEO
    v = np.random.uniform(VGEO*0.9, VGEO*1.1)
    # Pick a random direction
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi)
    vx = v * np.cos(theta) * np.sin(phi)
    vy = v * np.sin(theta) * np.sin(phi)
    vz = v * np.cos(phi)
    v = np.array([vx, vy, vz])
    orbit = ssapy.Orbit(r, v, time0)

    num_epochs = 2
    particles = []
    epoch_times = []
    lnprobs = []
    ch_rmean = []
    ch_vmean = []
    time = time0
    for _ in range(num_epochs):
        epoch_times.append(time)

        # Build observation times
        earthTimes = time + np.linspace(0, 10, 5)*u.h
        earthRaDec = ssapy.radec(orbit, earthTimes, observer=earthObserver)
        earthRStation, earthVStation = earthObserver.getRV(earthTimes)

        arc = QTable()
        arc['ra'] = Longitude(earthRaDec[0]*u.rad)
        arc['dec'] = Latitude(earthRaDec[1]*u.rad)
        arc['rStation_GCRF'] = earthRStation*u.m
        arc['vStation_GCRF'] = earthVStation*u.m/u.s
        arc['time'] = Time(earthTimes)
        arc['sigma'] = np.ones(5, dtype=float)*u.arcsec

        # initialize somewhere near the truth, but perturbed a bit and with large uncertainties
        initializer = ssapy.GaussianRVInitializer(r, v, rSigma=0.1*RGEO, vSigma=0.1*VGEO)
        # Use R and A priors
        priors = [ssapy.rvsampler.RPrior(RGEO, RGEO*0.2), ssapy.rvsampler.APrior(RGEO, RGEO*0.2)]

        # Run MCMC sampler to generate particles for the observed epoch
        rvprob = ssapy.RVProbability(arc, time, priors=priors)
        sampler = ssapy.EmceeSampler(rvprob, initializer, nWalker=50)
        chain, lnprob, lnprior = sampler.sample(nBurn=500, nStep=100)
        chain, lnprob, lnprior = ssapy.utils.cluster_emcee_walkers(chain, lnprob, lnprior, thresh_multiplier=4)

        ch_rmean.append(np.mean(chain[...,0:3], axis=(0,1)))
        ch_vmean.append(np.mean(chain[...,3:6], axis=(0,1)))

        # Create Particles object from sampler output
        particles.append(ssapy.Particles(chain, rvprob, lnpriors=lnprior))
        lnprobs.append(lnprob.ravel())

        time += 12.0 * u.h
    epoch_times = Time(epoch_times)
    # Check that epochs in particles class match the input epoch times
    for i in range(num_epochs):
        np.testing.assert_almost_equal(particles[i].epoch, epoch_times[i].gps)

    # Check that particles likelihood matches that stored from Sampler.
    # There is some offset here because of the prior in Sampler that is not used in the
    # Particles method.
    for i in range(num_epochs):
        lnP = particles[i].lnlike(particles[i].orbits) + particles[i].lnpriors
        np.testing.assert_allclose(lnP, lnprobs[i])

    # Check that mean particle values in each epoch match reference values
    rref, vref = ssapy.rv(orbit, epoch_times)

    for i in range(num_epochs):
        pmean = particles[i].mean()
        rmean = pmean[0:3]
        vmean = pmean[3:6]

        print("--------------------")
        print("r difference wrt GEO")
        # print((ch_rmean[i]-rref[i])/RGEO)
        print((rmean - rref[i]) / RGEO)
        print("v difference wrt GEO")
        # print((ch_vmean[i]-vref[i])/VGEO)
        print((vmean - vref[i]) / VGEO)
        np.testing.assert_allclose(rmean, rref[i], rtol=0, atol=RGEO*0.001)
        np.testing.assert_allclose(vmean, vref[i], rtol=0, atol=VGEO*0.001)

    # Check that Particles.fuse method gives correct mean of particle values
    particles[0].fuse(particles[1])
    pmean = particles[0].mean()
    rmean = pmean[0:3]
    vmean = pmean[3:6]
    print("--------------------")
    print("   AFTER PARTICLE FUSION   ")
    print("r difference wrt GEO")
    print((rmean - rref[i]) / RGEO)
    print("v difference wrt GEO")
    print((vmean - vref[i]) / VGEO)
    np.testing.assert_allclose(rmean, rref[0], rtol=0, atol=RGEO*0.001)
    np.testing.assert_allclose(vmean, vref[0], rtol=0, atol=VGEO*0.001)


if __name__ == '__main__':
    test_particles()

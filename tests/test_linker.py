import pytest
import numpy as np

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude, Latitude
from astropy.table import QTable

import ssapy
from .ssapy_test_helpers import timer
from ssapy.constants import RGEO, VGEO

def _mock_chain_from_rv(r, v, nSteps=10):
    """ Make a mock MCMC chain output from an orbit model
    """
    params = np.concatenate((r, v))
    chain = np.zeros((nSteps, 6))
    for istep in range(nSteps):
        chain[istep, :] = params
    return chain


def _create_iods(num_epochs=10, num_obs_per_track=3, nBurn=500, nStep=100):
    from ssapy.rvsampler import RPrior, APrior
    np.random.seed(27)

    sigma_arcsec = 1.0
    sigma_rad = sigma_arcsec * np.pi / (180.*3600.)

    time0 = Time(2458316., format='jd')

    # Setup observer location
    lon = 100.0
    lat = 33.0
    elevation = 1300.0
    earthObserver = ssapy.EarthObserver(lon, lat, elevation)

    # Create orbit for target
    orbit = ssapy.Orbit.fromKeplerianElements(RGEO*0.98, 0.01, 0.001, 0.0, 1.2, 1.03, time0)

    # Build observation times
    earthTimes = time0 + np.linspace(0, 4, num_obs_per_track)*u.h
    earthRaDec = ssapy.radec(orbit, earthTimes, observer=earthObserver)
    earthRStation, earthVStation = earthObserver.getRV(earthTimes)

    # Create multiple tracks for a single object
    iods = []
    time = time0
    for iepoch in range(num_epochs):
        print("IOD epoch {:d}".format(iepoch+1))

        arc = QTable()
        err_ra = np.random.randn(len(earthRaDec[0])) * sigma_rad
        err_dec = np.random.randn(len(earthRaDec[1])) * sigma_rad
        arc['ra'] = Longitude((earthRaDec[0] + err_ra)*u.rad)
        arc['dec'] = Latitude((earthRaDec[1] + err_dec)*u.rad)
        arc['rStation_GCRF'] = earthRStation*u.m
        arc['vStation_GCRF'] = earthVStation*u.m/u.s
        arc['time'] = Time(earthTimes)
        arc['sigma'] = sigma_arcsec * np.ones(num_obs_per_track, dtype=float)*u.arcsec

        # initialize somewhere near the truth, but perturbed a bit and with large uncertainties
        r, v = ssapy.rv(orbit, time)
        initializer = ssapy.GaussianRVInitializer(r, v, rSigma=0.1*RGEO, vSigma=0.1*VGEO)
        # Use R and A priors
        priors = [RPrior(RGEO, RGEO*0.2), APrior(RGEO, RGEO*0.2)]

        # Run MCMC sampler to generate particles for the observed epoch
        rvprob = ssapy.RVProbability(arc, time, priors=priors)

        sampler = ssapy.EmceeSampler(rvprob, initializer, nWalker=50)
        chain, lnprob, lnprior = sampler.sample(nBurn=nBurn, nStep=nStep)
        chain, lnprob, lnprior = ssapy.utils.cluster_emcee_walkers(chain, lnprob, lnprior, thresh_multiplier=4)
        chmean = np.mean(chain, axis=(0, 1))
        print("r err:", (chmean[0:3] - r) / r)
        print("v err:", (chmean[3:6] - v) / v)
        print("lnprobs:", np.min(lnprob), np.mean(lnprob), np.max(lnprob))
        #
        # chain = _mock_chain_from_rv(r[0], v[0])

        # write_mcmc_chain(chain, lnprob, "chain_{:d}.pkl".format(iepoch))

        # Create Particles object from sampler output
        iods.append(ssapy.Particles(chain, rvprob, lnpriors=lnprior))

        time += 6.0 * u.h
    return iods


@pytest.mark.slow
@timer
def test_orbit_selector_sampling():
    iods = _create_iods(num_epochs=4, num_obs_per_track=6, nBurn=500, nStep=500)
    lkr = ssapy.Linker(iods)
    print(lkr)

    # for j in range(len(iods)):
    #     for i, iod in enumerate(iods):
    #         print(j,i,iod.lnlike(iods[j].draw_orbit()))

    for itrack in range(lkr.num_tracks):
        lkr.orbit_selectors[itrack] = lkr.sample_orbit_selectors_from_data_conditional(itrack)
    print(lkr)


@pytest.mark.slow
@timer
def test_linker():
    np.random.seed(27)

    time0 = Time(2458316., format='jd')

    num_epochs = 4
    num_obs_per_track = 6

    # iods = _create_iods(num_epochs=num_epochs, num_obs_per_track=6, nBurn=500, nStep=500)
    iods = _create_iods(num_epochs=num_epochs, num_obs_per_track=6, nBurn=500, nStep=100)
    lkr = ssapy.Linker(iods)

    orbit_selector_means = ssapy.ModelSelectorParams(num_epochs, num_epochs, 0.0)
    p_orbit_means = ssapy.ModelSelectorParams(num_epochs, num_epochs, 0.0)

    # Do the linking
    nSteps = 50
    for istep in range(nSteps):
        lkr.update_params_using_carlin_chib()

        # Add to running means
        orbit_selector_means.params += lkr.orbit_selectors.params
        p_orbit_means.params += lkr.p_orbit.params

    orbit_selector_means.params /= float(nSteps)
    p_orbit_means.params /= float(nSteps)

    #
    # Test linking accuracy
    #
    # 1 - Check average values of orbit selector parameters
    print("Mean orbit_selectors:\n", orbit_selector_means)
    # 2 - Check average values of P_orbit parameters
    print("Mean P_orbit:\n", p_orbit_means)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--prof", action='store_true')
    parser.add_argument("--prof_out", type=str, default=None)
    parser.add_argument("--prof_png", type=str, default=None)
    args = parser.parse_args()

    if args.prof:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    test_orbit_selector_sampling()
    test_linker()

    if args.prof:
        import pstats
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumtime')
        ps.print_stats(30)
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
        if args.prof_out:
            pr.dump_stats(args.prof_out)
            if args.prof_png:
                import subprocess
                cmd = "gprof2dot -f pstats {} -n1 -e1 | dot -Tpng -o {}".format(args.prof_out, args.prof_png)
                subprocess.run(cmd, shell=True)

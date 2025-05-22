import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude, Latitude
from astropy.table import QTable

import ssapy
from ssapy.constants import RGEO, VGEO

def _create_iods_small(num_epochs=2, num_obs_per_track=3, nBurn=10, nStep=10):
    from ssapy.rvsampler import RPrior, APrior
    np.random.seed(42)

    sigma_arcsec = 1.0
    sigma_rad = sigma_arcsec * np.pi / (180. * 3600.)
    time0 = Time(2458316., format='jd')

    lon, lat, elevation = 100.0, 33.0, 1300.0
    observer = ssapy.EarthObserver(lon, lat, elevation)
    orbit = ssapy.Orbit.fromKeplerianElements(RGEO * 0.98, 0.01, 0.001, 0.0, 1.2, 1.03, time0)
    times = time0 + np.linspace(0, 4, num_obs_per_track) * u.h
    coord = ssapy.radec(orbit, times, observer=observer)
    rstation, vstation = observer.getRV(times)

    iods = []
    for _ in range(num_epochs):
        arc = QTable()
        arc['ra'] = Longitude((coord[0] + np.random.randn(len(coord[0])) * sigma_rad) * u.rad)
        arc['dec'] = Latitude((coord[1] + np.random.randn(len(coord[1])) * sigma_rad) * u.rad)
        arc['rStation_GCRF'] = rstation * u.m
        arc['vStation_GCRF'] = vstation * u.m / u.s
        arc['time'] = Time(times)
        arc['sigma'] = sigma_arcsec * np.ones(num_obs_per_track) * u.arcsec

        r, v = ssapy.rv(orbit, time0)
        initializer = ssapy.GaussianRVInitializer(r, v, rSigma=0.1 * RGEO, vSigma=0.1 * VGEO)
        priors = [RPrior(RGEO, RGEO * 0.2), APrior(RGEO, RGEO * 0.2)]
        rvprob = ssapy.RVProbability(arc, time0, priors=priors)

        sampler = ssapy.EmceeSampler(rvprob, initializer, nWalker=10)
        chain, lnprob, lnprior = sampler.sample(nBurn=nBurn, nStep=nStep)
        chain, lnprob, lnprior = ssapy.utils.cluster_emcee_walkers(chain, lnprob, lnprior, thresh_multiplier=4)

        iods.append(ssapy.Particles(chain, rvprob, lnpriors=lnprior))

    return iods


def test_model_selector_params_normalize():
    p = ssapy.ModelSelectorParams(3, 3, init_val=1.0)
    p.normalize()
    for i in range(3):
        assert np.isclose(np.sum(p[i]), 1.0)

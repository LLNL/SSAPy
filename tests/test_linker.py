import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude, Latitude
from astropy.table import QTable

import ssapy
from ssapy.constants import RGEO, VGEO


# Helper: simplified particle generation
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
    ra, dec = ssapy.radec(orbit, times, observer=observer)
    rstation, vstation = observer.getRV(times)

    iods = []
    for _ in range(num_epochs):
        arc = QTable()
        arc['ra'] = Longitude((ra[0] + np.random.randn(len(ra[0])) * sigma_rad) * u.rad)
        arc['dec'] = Latitude((dec[1] + np.random.randn(len(dec[1])) * sigma_rad) * u.rad)
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


def test_binary_selector_linking():
    p = ssapy.BinarySelectorParams(4, 4)
    p.params[0, 0] = 1
    p.params[1, 1] = 1
    p.params[2, 1] = 1
    linked = p.get_linked_track_indices(1)
    assert set(linked) == {1, 2}
    unlinked = p.get_unlinked_track_indices()
    assert 3 in unlinked


def test_sample_p_orbit():
    iods = _create_iods_small()
    linker = ssapy.Linker(iods)
    p = linker.sample_Porbit_conditional_dist(1)
    assert np.isclose(np.sum(p), 1.0)


def test_update_orbit_parameters_unlinked():
    iods = _create_iods_small()
    linker = ssapy.Linker(iods)
    linker.orbit_selectors.params[:, :] = 0  # Unlink all
    linker.update_orbit_parameters()
    for iod in linker.iods:
        assert hasattr(iod, 'reset_to_pseudo_prior')


def test_sample_orbit_selectors():
    iods = _create_iods_small()
    linker = ssapy.Linker(iods)
    result = linker.sample_orbit_selectors_from_data_conditional(1, verbose=False)
    assert isinstance(result, np.ndarray)
    assert np.sum(result) == 1


def test_update_params_using_carlin_chib():
    iods = _create_iods_small()
    linker = ssapy.Linker(iods)
    linker.update_params_using_carlin_chib(verbose=False)
    assert isinstance(linker.orbit_selectors.params, np.ndarray)


def test_linker_repr():
    iods = _create_iods_small()
    linker = ssapy.Linker(iods)
    rep = repr(linker)
    assert "Linker" in rep
    assert "orbit_selectors" in rep
    assert "p_orbit" in rep


def test_linker_save_step(tmp_path):
    iods = _create_iods_small()
    linker = ssapy.Linker(iods)
    outfile_head = tmp_path / "linker_test"
    linker.save_step(str(outfile_head))
    assert (tmp_path / "linker_test_orbit_selectors.txt").exists()
    assert (tmp_path / "linker_test_p_orbit.txt").exists()
    for i in range(len(iods)):
        assert (tmp_path / f"linker_test_particles_{i}.txt").exists()


def test_linker_sample_loop():
    iods = _create_iods_small()
    linker = ssapy.Linker(iods)
    linker.sample(nStep=2)  # Just 2 steps for speed
    assert isinstance(linker.orbit_selectors.params, np.ndarray)

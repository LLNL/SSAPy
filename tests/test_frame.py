import unittest
import json
import time
import numpy as np
from astropy.time import Time

import ssapy
from .ssapy_test_helpers import timer

try:
    import orekit
    orekit.initVM()
    from orekit.pyhelpers import setup_orekit_curdir
    setup_orekit_curdir()
    from org.orekit.time import TimeScalesFactory
    ts = TimeScalesFactory.getUTC()
except:
    no_orekit = True
else:
    no_orekit = False

# List to store benchmark results
benchmark_results = []

def log_result(test_name, execution_time, status):
    """Log benchmark result to a global list."""
    benchmark_results.append({
        "test_name": test_name,
        "execution_time": execution_time,
        "status": status
    })

@timer
def test_vallado():
    test_name = "test_vallado"
    start_time = time.time()
    try:
        t = Time("2004-04-06 07:51:28.386009")
        # These are in km and km/s
        rteme = np.array([5094.18016210, 6127.64465950, 6380.34453270])
        vteme = np.array([-4.7461314870, 0.7858180410, 5.5319312880])
        rgcrf = np.array([5102.50895290, 6123.01139910, 6378.13693380])
        vgcrf = np.array([-4.7432201610, 0.7905364950, 5.5337557240])

        # No longer need these?
        ddpsi = -0.052195/206265 # arcsec->radians
        ddeps = -0.003875/206265 # arcsec->radians

        rot = ssapy.utils.teme_to_gcrf(t)

        # 10 cm precision
        np.testing.assert_allclose(np.dot(rot, rteme), rgcrf, rtol=0, atol=1e-4)
        # 0.1 mm/s precision
        np.testing.assert_allclose(np.dot(rot, vteme), vgcrf, rtol=0, atol=1e-7)

        # Test reverse direction too
        np.testing.assert_allclose(np.dot(rot.T, rgcrf), rteme, rtol=0, atol=1e-4)
        np.testing.assert_allclose(np.dot(rot.T, vgcrf), vteme, rtol=0, atol=1e-7)

        rotT = ssapy.utils.gcrf_to_teme(t)
        np.testing.assert_allclose(np.dot(rotT, rgcrf), rteme, rtol=0, atol=1e-4)
        np.testing.assert_allclose(np.dot(rotT, vgcrf), vteme, rtol=0, atol=1e-7)

        status = "passed"
    except Exception as e:
        status = f"failed: {str(e)}"
    finally:
        execution_time = time.time() - start_time
        log_result(test_name, execution_time, status)

@unittest.skipIf(no_orekit, 'Unable to import orekit')
@timer
def test_teme_orekit():
    test_name = "test_teme_orekit"
    start_time = time.time()
    try:
        from org.orekit.frames import FramesFactory
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.time import AbsoluteDate
        from org.orekit.utils import PVCoordinates

        # Some cheap utilities to convert between java PVCoordinates and numpy vectors
        def v3toarr(v3):
            p3 = v3.getPosition()
            return np.array([p3.x, p3.y, p3.z])

        def arrtov3(arr):
            return PVCoordinates(Vector3D(float(arr[0]), float(arr[1]), float(arr[2])))


        TEME = FramesFactory.getTEME()
        GCRF = FramesFactory.getGCRF()

        for _ in range(100):
            rteme = np.random.uniform(low=-1000, high=1000, size=(10, 3))
            t = Time(np.random.uniform(low=0, high=1e8), format='gps')
            rot = ssapy.utils.teme_to_gcrf(t)
            rgcrf = np.dot(rot, rteme.T).T

            oreDate = AbsoluteDate(t.isot, TimeScalesFactory.getUTC())
            transform = TEME.getTransformTo(GCRF, oreDate)

            for rteme_, rgcrf_ in zip(rteme, rgcrf):
                rtest = v3toarr(transform.transformPVCoordinates(arrtov3(rteme_)))
                np.testing.assert_allclose(rtest, rgcrf_, rtol=0, atol=1e-3)  # 1m precision

        status = "passed"
    except Exception as e:
        status = f"failed: {str(e)}"
    finally:
        execution_time = time.time() - start_time
        log_result(test_name, execution_time, status)

@timer
def test_MG_5_1():
    test_name = "test_MG_5_1"
    start_time = time.time()
    try:
        """Exercise 5.1 from Montenbruck and Gill"""
        import astropy.units as u
        t0 = Time("1999-03-04T00:00:00", scale='utc')
        mjd_tt_j2000 = 51544.5
        mjd_tt = ssapy.utils._gpsToTT(t0.gps)
        dut1_mjd = t0.ut1.mjd - t0.tt.mjd

        # Using erfa instead of rolling our own...
        try:
            import erfa
        except ImportError:
            import astropy._erfa as erfa

        prec = erfa.pmat76(2400000.5, mjd_tt)
        nut = erfa.nutm80(2400000.5, mjd_tt)
        gst = erfa.gst94(2400000.5, mjd_tt + dut1_mjd)
        sg, cg = np.sin(gst), np.cos(gst)
        GHA = np.array([
            [cg, sg, 0],
            [-sg, cg, 0],
            [0, 0, 1]
        ])

        # Precession transformation matrix
        MG_prec = np.array([
            [+0.99999998, +0.00018581, +0.00008074],
            [-0.00018581, +0.99999998, -0.00000001],
            [-0.00008074, -0.00000001, +1.00000000]
        ])

        # Nutation transformation matrix
        MG_nut = np.array([
            [+1.00000000, +0.00004484, +0.00001944],
            [-0.00004484, +1.00000000, +0.00003207],
            [-0.00001944, -0.00003207, +1.00000000]
        ])

        # Earth rotation transformation matrix
        MG_GHA = np.array([
            [-0.94730417, +0.32033547, +0.00000000],
            [-0.32033547, -0.94730417, +0.00000000],
            [+0.00000000, +0.00000000, +1.00000000]
        ])

        np.testing.assert_allclose(prec, MG_prec, rtol=0, atol=1e-8)
        np.testing.assert_allclose(nut, MG_nut, rtol=0, atol=1e-8)
        np.testing.assert_allclose(GHA, MG_GHA, rtol=0, atol=1e-8)

        # Get dut1, polar motion from astropy, compare to M&G
        dut1_mjd_2, pmx, pmy = ssapy.utils.iers_interp(t0.gps)
        np.testing.assert_allclose(dut1_mjd, dut1_mjd_2, rtol=0, atol=1e-11)
        np.testing.assert_allclose(pmx, 0.00000033, rtol=0, atol=1e-8)
        np.testing.assert_allclose(pmy, 0.00000117, rtol=0, atol=1e-8)

        # We don't have a route to compute polar motion in ssapy.  So just assert
        # polar motion transformation matrix from MG for now.
        pol = np.array([
            [+1.0, +0.0, +pmx],
            [+0.0, +1.0, -pmy],
            [-pmx, +pmy, +1.0]
        ])

        # Transformation matrix from International Celestial Reference System (ICRS) to
        # International Terrestrial Reference System (ITRS)
        U = pol@GHA@nut@prec
        MG_U = np.array([
            [-0.94737803, +0.32011696, -0.00008431],
            [-0.32011696, -0.94737803, -0.00006363],
            [-0.00010024, -0.00003330, +0.99999999]
        ])
        np.testing.assert_allclose(U, MG_U, rtol=0, atol=1e-8)

        # Now see if we can do a getRV on an EarthObserver
        # Use transformation matrix without polar motion since that's what we'll
        # have available in production generally.
        U = GHA@nut@prec
        lon = np.deg2rad(70+44/60+11.7/3600)
        lat = np.deg2rad(-30-14/60-26.6/3600)
        elevation = 2715.0
        observer = ssapy.EarthObserver(lon, lat, elevation)
        r_itrs = observer._location.itrs.cartesian.xyz.to(u.m).value
        r_gcrs = U.T@r_itrs

        dUdt = (
            np.array([[0,1,0],[-1,0,0],[0,0,0]], dtype=float)
            * 1.002737909350795*2*np.pi/86400
            @ U
        )
        v_gcrs = dUdt.T @ r_itrs

        r, v = observer.getRV(t0)

        np.testing.assert_allclose(r, r_gcrs, rtol=0, atol=2)
        np.testing.assert_allclose(v, v_gcrs, rtol=0, atol=1e-4)

        status = "passed"
    except Exception as e:
        status = f"failed: {str(e)}"
    finally:
        execution_time = time.time() - start_time
        log_result(test_name, execution_time, status)

if __name__ == '__main__':
    test_vallado()
    try:
        test_teme_orekit()
    except unittest.case.SkipTest:
        print("Skipping test_teme_orekit()")
    test_MG_5_1()

    # Save benchmark results to a JSON file
    with open("benchmark_results_frame.json", "w") as f:
        json.dump(benchmark_results, f)
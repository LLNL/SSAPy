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
        # Example from Table 3-6 of Vallado textbook
        t = Time("2004-04-06 07:51:28.386009")
        rteme = np.array([5094.18016210, 6127.64465950, 6380.34453270])
        vteme = np.array([-4.7461314870, 0.7858180410, 5.5319312880])
        rgcrf = np.array([5102.50895290, 6123.01139910, 6378.13693380])
        vgcrf = np.array([-4.7432201610, 0.7905364950, 5.5337557240])

        rot = ssapy.utils.teme_to_gcrf(t)

        np.testing.assert_allclose(np.dot(rot, rteme), rgcrf, rtol=0, atol=1e-4)
        np.testing.assert_allclose(np.dot(rot, vteme), vgcrf, rtol=0, atol=1e-7)
        np.testing.assert_allclose(np.dot(rot.T, rgcrf), rteme, rtol=0, atol=1e-4)
        np.testing.assert_allclose(np.dot(rot.T, vgcrf), vteme, rtol=0, atol=1e-7)

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
                rtest = transform.transformPVCoordinates(PVCoordinates(Vector3D(*rteme_))).getPosition()
                np.testing.assert_allclose(rtest, rgcrf_, rtol=0, atol=1e-3)

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
        # Exercise 5.1 from Montenbruck and Gill
        import astropy.units as u
        t0 = Time("1999-03-04T00:00:00", scale='utc')
        mjd_tt = ssapy.utils._gpsToTT(t0.gps)
        dut1_mjd = t0.ut1.mjd - t0.tt.mjd
        prec = ssapy.utils.teme_to_gcrf(t0)

        np.testing.assert_allclose(prec, np.eye(3), rtol=0, atol=1e-8)

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
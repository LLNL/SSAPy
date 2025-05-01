import numpy as np
import astropy.units as u
from astropy.time import Time

import ssapy
from ssapy.utils import normed
from .ssapy_test_helpers import timer, checkAngle


@timer
def testGauss():
    # Testing out Gauss algorithm.  Works for small fractions of an orbital period.
    np.random.seed(5)
    for _ in range(1000):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        # Pick two points close together in time, infer the orbit!
        t1 = np.random.uniform(0, orbit.period)
        t2 = np.random.uniform(t1, t1 + 0.001*orbit.period)
        r1, _ = ssapy.rv(orbit, t1)
        r2, _ = ssapy.rv(orbit, t2)
        solver = ssapy.GaussTwoPosOrbitSolver(r1, r2, t1, t2)
        orbit2 = solver.solve().at(0)

        np.testing.assert_allclose(orbit.a, orbit2.a, atol=1, rtol=0)
        np.testing.assert_allclose(orbit.e, orbit2.e, atol=1e-9, rtol=0)
        checkAngle(orbit.i, orbit2.i, atol=1e-9, rtol=0)
        checkAngle(orbit.pa, orbit2.pa, atol=1e-9, rtol=0)
        checkAngle(orbit.raan, orbit2.raan, atol=1e-9, rtol=0)
        checkAngle(orbit.trueAnomaly, orbit2.trueAnomaly, atol=2e-7, rtol=0)


@timer
def testDanchick():
    np.random.seed(57)
    for _ in range(1000):
        dnu = np.inf
        # Only valid for dnu < pi
        while dnu > np.pi:
            a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
            e = np.random.uniform(0.02, 0.98)
            pa = np.random.uniform(0, 2*np.pi)
            raan = np.random.uniform(0, 2*np.pi)
            i = np.random.uniform(0, np.pi)
            trueAnomaly = np.random.uniform(0, 2*np.pi)
            orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
            t1 = np.random.uniform(0, orbit.period)
            t2 = np.random.uniform(t1, t1 + 0.001*orbit.period)
            nu1 = orbit.at(t1).trueAnomaly
            nu2 = orbit.at(t2).trueAnomaly
            dnu = nu2 - nu1
        r1, v1 = ssapy.rv(orbit, t1)
        r2, v2 = ssapy.rv(orbit, t2)

        orbit2 = ssapy.DanchickTwoPosOrbitSolver(
            r1, r2, t1, t2).solve()
        np.testing.assert_allclose(r1, ssapy.rv(orbit2, t1)[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(r2, ssapy.rv(orbit2, t2)[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(v1, ssapy.rv(orbit2, t1)[1], atol=1e-2, rtol=0)
        np.testing.assert_allclose(v2, ssapy.rv(orbit2, t2)[1], atol=1e-2, rtol=0)


@timer
def testShefer():
    np.random.seed(577)
    nTest = 1000
    nRobust = 0
    for _ in range(nTest):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        # TODO: Allow e > 1
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        t1 = np.random.uniform(0, orbit.period)
        t2 = np.random.uniform(t1, t1 + 0.001*orbit.period)

        r1, v1 = ssapy.rv(orbit, t1)
        r2, v2 = ssapy.rv(orbit, t2)
        orbit2 = ssapy.SheferTwoPosOrbitSolver(
            r1, r2, t1, t2, robust=True, nExam=200).solve()

        np.testing.assert_allclose(r1, ssapy.rv(orbit2, t1)[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(r2, ssapy.rv(orbit2, t2)[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(v1, ssapy.rv(orbit2, t1)[1], atol=1e-1, rtol=0)
        np.testing.assert_allclose(v2, ssapy.rv(orbit2, t2)[1], atol=1e-1, rtol=0)

    # Now let's try adding a few orbit wraps
    for _ in range(nTest):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        lam = np.random.randint(1, 5)
        t1 = np.random.uniform(0, 1)*orbit.period
        dt = np.random.uniform(lam, lam+1)*orbit.period
        t2 = t1 + dt
        r1, v1 = ssapy.rv(orbit, t1)
        r2, v2 = ssapy.rv(orbit, t2)

        orbit2 = ssapy.SheferTwoPosOrbitSolver(
            r1, r2, t1, t2, lam=lam).solve()
        r1_test, _ = ssapy.rv(orbit2, t1)
        r2_test, _ = ssapy.rv(orbit2, t2)
        # Sometimes the initial guess doesn't work well, so check we can solve robustly in those
        # cases.
        try:
            np.testing.assert_allclose(r1, r1_test, atol=1, rtol=0)
            np.testing.assert_allclose(r2, r2_test, atol=1, rtol=0)
        except AssertionError:
            nRobust += 1
            orbit2 = ssapy.SheferTwoPosOrbitSolver(
                r1, r2, t1, t2, lam=lam, robust=True, nExam=500).solve()
            r1_test, _ = ssapy.rv(orbit2, t1)
            r2_test, _ = ssapy.rv(orbit2, t2)
            np.testing.assert_allclose(r1, r1_test, atol=1, rtol=0)
            np.testing.assert_allclose(r2, r2_test, atol=1, rtol=0)

    print("Reverted to robust methods {} of {} times.".format(nRobust, nTest))


@timer
def testKappaSignPlane():
    np.random.seed(5772)
    for _ in range(1000):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        t1 = np.random.uniform(0, orbit.period)
        t2 = np.random.uniform(0, orbit.period)
        r1, _ = ssapy.rv(orbit, t1)
        r2, _ = ssapy.rv(orbit, t2)
        solver1 = ssapy.SheferTwoPosOrbitSolver(r1, r2, t1, t2)
        solver2 = ssapy.SheferTwoPosOrbitSolver(
            r1, r2, t1, t2, kappaSign=-1)
        # Orbital plane is fixed regardless of kappaSign, but which node is labeled
        # ascending/descending are flipped, as is the inclination angle.
        np.testing.assert_allclose(
            solver1.raan, (solver2.raan+np.pi) % (2*np.pi), rtol=0, atol=1e-9)
        np.testing.assert_allclose(solver1.i, np.pi-solver2.i, rtol=0, atol=1e-9)


@timer
def testThreeAngles():
    np.random.seed(57721)
    failedSolve = 0
    failedTest = 0
    ntest = 1000
    for _ in range(ntest):
        a = np.random.uniform(7e6, 5e7)  # Roughly LEO to GEO
        e = np.random.uniform(0.02, 0.98)
        pa = np.random.uniform(0, 2*np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        i = np.random.uniform(0, np.pi)
        trueAnomaly = np.random.uniform(0, 2*np.pi)
        orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, trueAnomaly, 0)
        # Observation points.  Roughly uniformly distributed within cube
        # surrounding earth
        R1 = np.random.uniform(-7e6, 7e6, size=3)
        R2 = np.random.uniform(-7e6, 7e6, size=3)
        R3 = np.random.uniform(-7e6, 7e6, size=3)
        # Algorithm only seems stable for relatively small time separations
        t1 = np.random.uniform(0, orbit.period)
        dt21 = np.random.uniform(0, orbit.period/10)
        t2 = t1 + dt21
        dt32 = np.random.uniform(0, orbit.period/10)
        t3 = t2 + dt32
        r1, _ = ssapy.rv(orbit, t1)
        r2, _ = ssapy.rv(orbit, t2)
        r3, _ = ssapy.rv(orbit, t3)
        e1 = normed(r1 - R1)
        e2 = normed(r2 - R2)
        e3 = normed(r3 - R3)
        solver = ssapy.ThreeAngleOrbitSolver(
            e1, e2, e3, R1, R2, R3, t1, t2, t3)

        # Basic dot/cross product orthogonality check
        np.testing.assert_allclose(
            np.dot(solver.d1, solver.e2), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d1, solver.e3), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d2, solver.e1), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d2, solver.e3), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d3, solver.e1), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.dot(solver.d3, solver.e2), 0, rtol=0, atol=1e-10)

        assert solver.t21 > 0
        assert solver.t32 > 0
        assert solver.t31 > 0

        try:
            orbit2 = solver.solve()
        except ValueError:
            failedSolve += 1
            continue
        try:
            r1_test, _ = ssapy.rv(orbit2, t1)
            r2_test, _ = ssapy.rv(orbit2, t2)
            r3_test, _ = ssapy.rv(orbit2, t3)
        except RuntimeError:
            # Count these as failures too
            failedTest += 1
            continue
        e1_test = normed(r1_test-R1)
        e2_test = normed(r2_test-R2)
        e3_test = normed(r3_test-R3)
        try:
            np.testing.assert_allclose(
                e1, e1_test, rtol=0, atol=5e-6)  # ~arcsec-ish
            np.testing.assert_allclose(e2, e2_test, rtol=0, atol=5e-6)
            np.testing.assert_allclose(e3, e3_test, rtol=0, atol=5e-6)
        except AssertionError:
            failedTest += 1
            # print("----- Failed test in three angle orbit solver -----")
            # print("Times deltas:", dt21, dt32)
            # print(orbit)
            # print(orbit2)
            continue

    print("ThreeAngleOrbitSolver failed to solve {} times out of {}".format(
        failedSolve, ntest))
    print("ThreeAngleOrbitSolver failed test {} times out of {}".format(
        failedTest, ntest))


@timer
def test_MG_2_6():
    """Exercise 2.6 from Montenbruck and Gill

    Tests orbit determination from two position vectors
    """
    # Elements provided in MG as a solution
    a_ref = 28196776.0 # meters
    e_ref = 0.7679436
    i_ref = np.deg2rad(20.315)
    Omega_ref = np.deg2rad(359.145)
    omega_ref = np.deg2rad(179.425)
    M0_ref = np.deg2rad(29.236)

    # Specify satellite positions at two times
    r1 = np.array([11959978.0, -16289478.0, -5963827.0])
    r2 = np.array([39863390.0, -13730547.0, -4862350.0])
    t1 = Time(2455198.0, format='jd')
    t2 = t1 + 2.5*u.hour

    orbit = ssapy.SheferTwoPosOrbitSolver(r1, r2, t1, t2).solve()

    # Test that determined elements are close to reference values
    np.testing.assert_allclose(orbit.a, a_ref, atol=1e-1, rtol=0)
    np.testing.assert_allclose(orbit.e, e_ref, atol=1e-6, rtol=0)
    np.testing.assert_allclose(orbit.i, i_ref, atol=1e-5, rtol=0)
    np.testing.assert_allclose(orbit.pa, omega_ref, atol=1e-5, rtol=0)
    np.testing.assert_allclose(orbit.raan, Omega_ref, atol=1e-5, rtol=0)
    np.testing.assert_allclose(orbit.meanAnomaly, M0_ref, atol=1e-5, rtol=0)


if __name__ == '__main__':
    testGauss()
    testDanchick()
    testShefer()
    testKappaSignPlane()
    testThreeAngles()
    test_MG_2_6()

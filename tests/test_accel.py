import numpy as np
from astropy.time import Time
import astropy.units as u

import ssapy
from ssapy.utils import norm, iers_interp
from ssapy_test_helpers import timer

iers_interp(0.0)  # Prime the IERS interpolant cache
earth = ssapy.get_body("earth")
moon = ssapy.get_body("moon")
sun = ssapy.get_body("sun")

# When testing against M&G results, use M&G functions for the moon and sun positions.
sun_MG = ssapy.Body(
    sun.mu,
    sun.radius,
    position=ssapy.utils.sunPos
)

moon_MG = ssapy.Body(
    moon.mu,
    moon.radius,
    position=ssapy.utils.moonPos
)

earth_MG = ssapy.Body(
    3986004.418e8,
    6378137.0,
    orientation=earth.orientation,
    harmonics=earth.harmonics
)


@timer
def test_MG_3_1():
    """Exercise 3.1 from Montenbruck and Gill
    Tests implementation of harmonic acceleration
    """
    
    r = np.array([6525.919e3, 1710.416e3, 2508.886e3])

    # M&G results
    mg = {}
    mg[2] =  (-6.97922756436043556500,-1.82928105379987315790,-2.69001658551662448100)
    mg[4] =  (-6.97931189286893438606,-1.82931487068582887545,-2.68999140120129842657)
    mg[6] =  (-6.97922066700256138461,-1.82927878079804373535,-2.68997263886697091095)
    mg[8] =  (-6.97927699746898344557,-1.82928186346070154045,-2.68998582282075027194)
    mg[10] = (-6.97924447942545889134,-1.82928331385817810606,-2.68997524437058332936)
    mg[12] = (-6.97924725687663993767,-1.82928130662047960797,-2.68998625958353532184)
    mg[14] = (-6.97924919386132991406,-1.82928546814451764568,-2.68999164568811455212)
    mg[16] = (-6.97926211023274856160,-1.82928438361093181896,-2.68999719587274421784)
    mg[18] = (-6.97926208121088365033,-1.82928491800309545035,-2.68999523790429417858)
    mg[20] = (-6.97926186199732700999,-1.82928315091240034640,-2.68999053339306737342)

    for n in range(2, 22, 2):
        ah = ssapy.AccelHarmonic(earth, n, n) + ssapy.AccelKepler(earth.mu)
        a = ah(r, v=None, t=None, _E=np.eye(3))
        np.testing.assert_allclose(
            a, mg[n],
            rtol=1e-5, atol=1e-5
        )

    # Check with coord frame rotation
    for n in range(2, 22, 2):
        ah = ssapy.AccelHarmonic(earth, n, n) + ssapy.AccelKepler(earth.mu)
        a = ah(r, v=None, t=Time("J2000").gps)


@timer
def test_MG_3_2():
    """Exercise 3.2 from Montenbruck and Gill
    Tests implementation of position of the Moon
    """
    t0 = Time("2006-03-14", scale='tt')
    tt = t0 + np.linspace(0, 4, 5)*u.d
    mg_moon = [[-387105.185,  106264.577,  61207.474],
               [-403080.629,   33917.735,  21704.832],
               [-401102.631,  -39906.188, -18757.478],
               [-381055.373, -111853.486, -58337.911],
               [-343564.315, -178551.672, -95178.733]]
    for i, t in enumerate(tt):
        with np.printoptions(precision=10):
            # meter comparison, which matches available precision of M&G text
            np.testing.assert_allclose(
                ssapy.utils.moonPos(t)*1e-3,
                mg_moon[i],
                rtol=1e-4, atol=0
            )

    # Bonus: test sun position
    mg_sun = [[147659000.747, -16236318.162,  -7039305.429],
              [147984936.336, -13880545.131,  -6017952.822],
              [148266334.848, -11520595.242,  -4994789.325],
              [148503148.336,  -9157181.657,  -3970124.132],
              [148695342.555,  -6791017.418,  -2944266.385]]
    for i, t in enumerate(tt):
        np.testing.assert_allclose(
            ssapy.utils.sunPos(t.gps)*1e-3,
            mg_sun[i],
            rtol=0, atol=2e1
        )


@timer
def test_accel_point_mass():
    t0 = Time("2006-03-14", scale='tt')
    tt = t0 + np.linspace(0, 4, 5)*u.d

    aSun = ssapy.AccelThirdBody(sun_MG)
    aMoon = ssapy.AccelThirdBody(moon_MG)

    r = np.array([6525.919e3, 1710.416e3, 2508.886e3])

    # Results from running the M&G c++ code
    mg_a_sun = [[4.787594435959e-07, -1.506248263435e-07,  -1.366241532366e-07],
                [4.866522489960e-07, -1.392951306147e-07,  -1.316538330541e-07],
                [4.940957008119e-07, -1.277321516968e-07, -1.265820866453e-07],
                [5.010821227119e-07, -1.159509575260e-07, -1.214154657237e-07],
                [5.076044504898e-07, -1.039668095728e-07, -1.161606059082e-07]]

    mg_a_moon = [[6.467853044291e-07, -4.239660210384e-07, -3.514197439071e-07],
                 [8.711708484499e-07, -2.325007452337e-07, -2.484300880695e-07],
                 [9.961090151248e-07,  2.378287793727e-08, -1.098384738332e-07],
                 [9.984573326195e-07,  3.078472767560e-07,  4.421770492659e-08],
                 [8.712100529065e-07,  5.761948337171e-07,  1.898894833647e-07]]

    for i, t in enumerate(tt):
        np.testing.assert_allclose(
            aSun(r, None, t.gps),
            mg_a_sun[i],
            rtol=1e-6, atol=0
        )
        np.testing.assert_allclose(
            aMoon(r, None, t.gps),
            mg_a_moon[i],
            rtol=3e-4, atol=0
        )


@timer
def test_angles():
    # Using erfa instead of rolling our own...
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    t0 = Time("2006-03-14", scale='tt')
    tt = t0 + np.linspace(0, 4, 5)*u.d
    mg_eps = [
        4.090787388561810678e-01,
        4.090787326421966852e-01,
        4.090787264282123581e-01,
        4.090787202142280310e-01,
        4.090787140002436484e-01
    ]
    mg_nut = [
        (-4.479169934526547680e-06, 4.713856308779495319e-05),
        (-5.159286627511575472e-06, 4.725132305213035371e-05),
        (-5.883590187152149949e-06, 4.727163096558973510e-05),
        (-6.565069375976380403e-06, 4.719608789658485384e-05),
        (-7.118421904936305293e-06, 4.703624800004708149e-05)
    ]
    for eps, (dpsi, deps), t in zip(mg_eps, mg_nut, tt):
        np.testing.assert_allclose(
            eps,
            erfa.obl80(t.tt.jd1, t.tt.jd2),
            rtol=0, atol=1e-10
        )
        dpsi0, deps0 = erfa.nut80(t.tt.jd1, t.tt.jd2)
        np.testing.assert_allclose(dpsi, dpsi0, rtol=0, atol=1e-17)
        np.testing.assert_allclose(deps, deps0, rtol=0, atol=1e-17)


@timer
def test_MG_3_4_accel():
    """Exercise 3.4 from Montenbruck and Gill
    Tests implementation of harmonic, lunar, solar radiation, and other accelerations
    """
    aSun = ssapy.AccelThirdBody(sun_MG)
    aMoon = ssapy.AccelThirdBody(moon_MG)
    aH2020 = ssapy.AccelHarmonic(earth_MG, n_max=20, m_max=20) + ssapy.AccelKepler(mu=earth_MG.mu)

    t = Time("1999-03-01", scale='utc')

    # LEO sat
    kElements = [7178e3, 0.001, np.deg2rad(98.57), 0.0, 0.0, 0.0]
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t)

    np.testing.assert_allclose(orbit.r, (7170822.0, 0, 0))
    np.testing.assert_allclose(orbit.v, (0, -1111.575723, 7376.070927))

    np.testing.assert_allclose(
        aH2020(orbit.r, orbit.v, t.gps),
        (-7.7617896946453779, -3.442000486531e-05, -7.65885416643e-06),
        rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        aSun(orbit.r, orbit.v, t.gps),
        (4.818380258e-07, -2.588815598e-07, -1.122388926e-07),
        rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        aMoon(orbit.r, orbit.v, t.gps),
        (6.025729967e-07, -7.750060382e-07, -3.421012515e-07),
        rtol=2e-5, atol=0
    )

    area = 5.0  # [m^2]
    mass = 1000.0  # [kg]
    CR = 1.3  # radiation pressure coefficient
    aSR = ssapy.AccelSolRad()
    np.testing.assert_allclose(
        aSR(orbit.r, orbit.v, t.gps, area=area, mass=mass, CR=CR),
        (-2.837492087e-08, 9.488692618e-09, 4.113851724e-09),
        rtol=0, atol=1e-14
    )

    CD = 2.3  # drag coefficient
    aD = ssapy.AccelDrag()
    np.testing.assert_allclose(
        aD(orbit.r, orbit.v, t.gps, area=area, mass=mass, CD=CD),
        (0, 5.898500809e-09, -2.66186444e-08),
        rtol=2e-7, atol=1e-18
    )


    # GEO sat
    kElements = [42166.0e3, 0.0004, np.deg2rad(0.02), 0.0, 0.0, 0.0]
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t)

    np.testing.assert_allclose(orbit.r, (42149133.60000000149, 0, 0))
    np.testing.assert_allclose(orbit.v, (0, 3075.8232599877492248, 1.0736649055318405743))

    np.testing.assert_allclose(
        aH2020(orbit.r, orbit.v, t.gps),
        (-0.22437613764116937087, -5.3801359310012078524e-08, -1.0983725775895437048e-09),
        rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        aSun(orbit.r, orbit.v, t.gps),
        (2.83298772019e-06, -1.52232389565e-06, -6.60008183945e-07),
        rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        aMoon(orbit.r, orbit.v, t.gps),
        (3.39179549202e-06, -4.02173355464e-06, -1.77526369393e-06),
        rtol=3e-5, atol=0
    )

    area = 10.0  # [m^2]
    mass = 1000.0  # [kg]
    CR = 1.3  # radiation pressure coefficient
    aSR = ssapy.AccelSolRad()
    np.testing.assert_allclose(
        aSR(orbit.r, orbit.v, t.gps, area=area, mass=mass, CR=CR),
        (-5.677334498e-08, 1.899001522e-08, 8.23317921e-09),
        rtol=0, atol=1e-14
    )

    CD = 2.3  # drag coefficient
    aD = ssapy.AccelDrag()
    np.testing.assert_allclose(
        aD(orbit.r, orbit.v, t.gps, area=area, mass=mass, CD=CD),
        (0, 0, 0),
        rtol=0, atol=1e-16
    )


@timer
def test_MG_3_4():
    t = Time("1999-03-01", scale='utc')
    # LEO sat
    kElements = [7178e3, 0.001, np.deg2rad(98.57), 0.0, 0.0, 0.0]
    kwargs = dict(
        mass = 1000.0,  # [kg]
        area = 5.0,  # [m^2]
        CD = 2.3,
        CR = 1.3,
    )
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)

    aSun = ssapy.AccelThirdBody(sun_MG)
    aMoon = ssapy.AccelThirdBody(moon_MG)
    aH2020 = ssapy.AccelHarmonic(earth_MG, n_max=20, m_max=20) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH1010 = ssapy.AccelHarmonic(earth_MG, n_max=10, m_max=10) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH44 = ssapy.AccelHarmonic(earth_MG, n_max=4, m_max=4) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH22 = ssapy.AccelHarmonic(earth_MG, n_max=2, m_max=2) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH20 = ssapy.AccelHarmonic(earth_MG, n_max=2, m_max=0) + ssapy.AccelKepler(mu=earth_MG.mu)
    aSolRad = ssapy.AccelSolRad()
    aDrag = ssapy.AccelDrag()

    a0 = ssapy.AccelSum([aH2020, aSun, aMoon, aSolRad, aDrag])
    a1 = aH2020 + aSun + aMoon + aSolRad + aDrag
    a_to_J20 = ssapy.AccelSum([aH20, aSun, aMoon, aSolRad, aDrag])
    a_to_J22 = ssapy.AccelSum([aH22, aSun, aMoon, aSolRad, aDrag])
    a_to_J44 = ssapy.AccelSum([aH44, aSun, aMoon, aSolRad, aDrag])
    a_to_J1010 = ssapy.AccelSum([aH1010, aSun, aMoon, aSolRad, aDrag])
    a_noSun = ssapy.AccelSum([aH2020, aMoon, aSolRad, aDrag])
    a_noMoon = ssapy.AccelSum([aH2020, aSun, aSolRad, aDrag])
    a_noSolRad = ssapy.AccelSum([aH2020, aSun, aMoon, aDrag])
    a_noDrag = ssapy.AccelSum([aH2020, aSun, aMoon, aSolRad])

    h = 50.0
    times = t + np.linspace(0, orbit.period, 1000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK4Propagator(a0, h=h)
    )
    r1, v1 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK4Propagator(a1, h=h)
    )
    # Test AccelSum([a1, a2, a3, ...]) vs. a1 + a2 + a3 + ...
    np.testing.assert_equal(r0, r1)
    np.testing.assert_equal(v0, v1)

    print("Remote sensing, 1 rev")
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [600, 224, 148, 23, 3, 6, 1, 1]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.RK4Propagator(a, h=h)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=10)

    print()
    print()
    print("Remote sensing, 1 day")
    # Repeat for 1 day
    times = t + np.linspace(0, 86400, 1000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK4Propagator(a0, h=h)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [5028, 3038, 1925, 459, 34, 66, 14, 105]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.RK4Propagator(a, h=h)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=30)

    # GEO sat
    kElements = [42166e3, 0.0004, np.deg2rad(0.02), 0.0, 0.0, 0.0]
    kwargs = dict(
        mass = 1000.0,  # [kg]
        area = 10.0,  # [m^2]
        CD = 2.3,
        CR = 1.3,
    )
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)

    print()
    print()
    print("GEOstationary, 1 day")
    # Repeat for 1 day
    times = t + np.linspace(0, 86400, 1000)*u.s
    h = 1000.0
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK4Propagator(a0, h=h)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [671, 2, 0, 0, 3143, 5080, 415, 0]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.RK4Propagator(a, h=h)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=2)

    print()
    print()
    print("GEOstationary, 2 days")
    # Repeat for 1 day
    times = t + np.linspace(0, 2*86400, 2000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK4Propagator(a0, h=h)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [2534, 10, 0, 0, 4834, 5438, 830, 0]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.RK4Propagator(a, h=h)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=2)
    print()
    print()


@timer
def test_MG_3_4_scipy():
    t = Time("1999-03-01", scale='utc')
    # LEO sat
    kElements = [7178e3, 0.001, np.deg2rad(98.57), 0.0, 0.0, 0.0]
    kwargs = dict(
        mass = 1000.0,  # [kg]
        area = 5.0,  # [m^2]
        CD = 2.3,
        CR = 1.3,
    )
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)
    ode_kwargs = dict(
        rtol=1e-7
    )

    aSun = ssapy.AccelThirdBody(sun_MG)
    aMoon = ssapy.AccelThirdBody(moon_MG)
    aH2020 = ssapy.AccelHarmonic(earth_MG, n_max=20, m_max=20) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH1010 = ssapy.AccelHarmonic(earth_MG, n_max=10, m_max=10) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH44 = ssapy.AccelHarmonic(earth_MG, n_max=4, m_max=4) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH22 = ssapy.AccelHarmonic(earth_MG, n_max=2, m_max=2) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH20 = ssapy.AccelHarmonic(earth_MG, n_max=2, m_max=0) + ssapy.AccelKepler(mu=earth_MG.mu)
    aSolRad = ssapy.AccelSolRad()
    aDrag = ssapy.AccelDrag()

    a0 = ssapy.AccelSum([aH2020, aSun, aMoon, aSolRad, aDrag])
    a_to_J20 = ssapy.AccelSum([aH20, aSun, aMoon, aSolRad, aDrag])
    a_to_J22 = ssapy.AccelSum([aH22, aSun, aMoon, aSolRad, aDrag])
    a_to_J44 = ssapy.AccelSum([aH44, aSun, aMoon, aSolRad, aDrag])
    a_to_J1010 = ssapy.AccelSum([aH1010, aSun, aMoon, aSolRad, aDrag])
    a_noSun = ssapy.AccelSum([aH2020, aMoon, aSolRad, aDrag])
    a_noMoon = ssapy.AccelSum([aH2020, aSun, aSolRad, aDrag])
    a_noSolRad = ssapy.AccelSum([aH2020, aSun, aMoon, aDrag])
    a_noDrag = ssapy.AccelSum([aH2020, aSun, aMoon, aSolRad])

    times = t + np.linspace(0, orbit.period, 1000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.SciPyPropagator(a0, ode_kwargs=ode_kwargs)
    )

    print("Remote sensing, 1 rev")
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [600, 224, 148, 23, 3, 6, 1, 1]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.SciPyPropagator(a, ode_kwargs=ode_kwargs)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=10)

    print()
    print()
    print("Remote sensing, 1 day")
    # Repeat for 1 day
    times = t + np.linspace(0, 86400, 1000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.SciPyPropagator(a0, ode_kwargs=ode_kwargs)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [5028, 3038, 1925, 459, 34, 66, 14, 105]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.SciPyPropagator(a, ode_kwargs=ode_kwargs)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=30)

    # GEO sat
    kElements = [42166e3, 0.0004, np.deg2rad(0.02), 0.0, 0.0, 0.0]
    kwargs = dict(
        mass = 1000.0,  # [kg]
        area = 10.0,  # [m^2]
        CD = 2.3,
        CR = 1.3,
    )
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)

    print()
    print()
    print("GEOstationary, 1 day")
    # Repeat for 1 day
    times = t + np.linspace(0, 86400, 1000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.SciPyPropagator(a0, ode_kwargs=ode_kwargs)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [671, 2, 0, 0, 3143, 5080, 415, 0]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.SciPyPropagator(a, ode_kwargs=ode_kwargs)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=2)

    print()
    print()
    print("GEOstationary, 2 days")
    # Repeat for 1 day
    times = t + np.linspace(0, 2*86400, 2000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.SciPyPropagator(a0, ode_kwargs=ode_kwargs)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [2534, 10, 0, 0, 4834, 5438, 830, 0]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.SciPyPropagator(a, ode_kwargs=ode_kwargs)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=2)
    print()
    print()


@timer
def test_MG_3_4_rk78():
    t = Time("1999-03-01", scale='utc')
    # LEO sat
    kElements = [7178e3, 0.001, np.deg2rad(98.57), 0.0, 0.0, 0.0]
    kwargs = dict(
        mass = 1000.0,  # [kg]
        area = 5.0,  # [m^2]
        CD = 2.3,
        CR = 1.3,
    )
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)
    ode_kwargs = dict(
        rtol=1e-7
    )

    aSun = ssapy.AccelThirdBody(sun_MG)
    aMoon = ssapy.AccelThirdBody(moon_MG)
    aH2020 = ssapy.AccelHarmonic(earth_MG, n_max=20, m_max=20) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH1010 = ssapy.AccelHarmonic(earth_MG, n_max=10, m_max=10) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH44 = ssapy.AccelHarmonic(earth_MG, n_max=4, m_max=4) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH22 = ssapy.AccelHarmonic(earth_MG, n_max=2, m_max=2) + ssapy.AccelKepler(mu=earth_MG.mu)
    aH20 = ssapy.AccelHarmonic(earth_MG, n_max=2, m_max=0) + ssapy.AccelKepler(mu=earth_MG.mu)
    aSolRad = ssapy.AccelSolRad()
    aDrag = ssapy.AccelDrag()

    a0 = ssapy.AccelSum([aH2020, aSun, aMoon, aSolRad, aDrag])
    a_to_J20 = ssapy.AccelSum([aH20, aSun, aMoon, aSolRad, aDrag])
    a_to_J22 = ssapy.AccelSum([aH22, aSun, aMoon, aSolRad, aDrag])
    a_to_J44 = ssapy.AccelSum([aH44, aSun, aMoon, aSolRad, aDrag])
    a_to_J1010 = ssapy.AccelSum([aH1010, aSun, aMoon, aSolRad, aDrag])
    a_noSun = ssapy.AccelSum([aH2020, aMoon, aSolRad, aDrag])
    a_noMoon = ssapy.AccelSum([aH2020, aSun, aSolRad, aDrag])
    a_noSolRad = ssapy.AccelSum([aH2020, aSun, aMoon, aDrag])
    a_noDrag = ssapy.AccelSum([aH2020, aSun, aMoon, aSolRad])

    times = t + np.linspace(0, orbit.period, 1000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK78Propagator(a0, h=10)
    )

    print("Remote sensing, 1 rev")
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [600, 224, 148, 23, 3, 6, 1, 1]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.RK78Propagator(a, h=10, tol=(1e-3,)*3+(1e-6,)*3)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=10)

    print()
    print()
    print("Remote sensing, 1 day")
    # Repeat for 1 day
    times = t + np.linspace(0, 86400, 1000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK78Propagator(a0, h=10)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [5028, 3038, 1925, 459, 34, 66, 14, 105]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.RK78Propagator(a, h=10, tol=(1e-3,)*3+(1e-6,)*3)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=30)

    # GEO sat
    kElements = [42166e3, 0.0004, np.deg2rad(0.02), 0.0, 0.0, 0.0]
    kwargs = dict(
        mass = 1000.0,  # [kg]
        area = 10.0,  # [m^2]
        CD = 2.3,
        CR = 1.3,
    )
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)

    print()
    print()
    print("GEOstationary, 1 day")
    # Repeat for 1 day
    times = t + np.linspace(0, 86400, 1000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK78Propagator(a0, h=10)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [671, 2, 0, 0, 3143, 5080, 415, 0]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.RK78Propagator(a, h=10, tol=(1e-2,)*3+(1e-5,)*3)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=2)

    print()
    print()
    print("GEOstationary, 2 days")
    # Repeat for 1 day
    times = t + np.linspace(0, 2*86400, 2000)*u.s
    r0, v0 = ssapy.rv(
        orbit,
        times,
        propagator=ssapy.RK78Propagator(a0, h=10)
    )
    for a, label, target in zip(
        [a_to_J20, a_to_J22, a_to_J44, a_to_J1010, a_noSun, a_noMoon, a_noSolRad, a_noDrag],
        ["J20", "J22", "J44", "J1010", "Sun", "Moon", "Radiation Pressure", "Atmospheric Drag"],
        [2534, 10, 0, 0, 4834, 5438, 830, 0]
    ):
        r, v = ssapy.rv(
            orbit,
            times,
            propagator=ssapy.RK78Propagator(a, h=10, tol=(1e-2,)*3+(1e-5,)*3)
        )
        drho = norm(r-r0)
        print(f"{label:20s} {int(np.max(drho)):5d}")
        np.testing.assert_allclose(np.max(drho), target, rtol=0, atol=2)
    print()
    print()


@timer
def test_RK4_vs_analytic():
    # Test that analytic Keplerian propagation matches RK4 propagator when
    # acceleration is purely Keplerian
    np.random.seed(577215664)
    t0 = Time("1982-03-14", scale='utc')
    times = t0 + np.linspace(0, 24, 1000)*u.h

    for _ in range(10):
        while True:
            # Pick a distance near GEO
            r = np.random.uniform(4e7, 5e7)
            # Pick a random direction (not uniform on sphere)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            r = np.array([x, y, z])
            # Pick a velocity near VGEO
            v = np.random.uniform(2.7e3, 3.3e3)
            # and a randomish direction
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            vx = v * np.cos(theta) * np.sin(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(phi)
            v = np.array([vx, vy, vz])
            orbit = ssapy.Orbit(r, v, t0)
            if norm(orbit.periapsis) > 1e7:
                break

        r1, v1 = ssapy.rv(orbit, times)
        r2, v2 = ssapy.rv(
            orbit, times,
            propagator=ssapy.RK4Propagator(ssapy.AccelKepler(), h=70.0)
        )
        np.testing.assert_allclose(r1, r2, rtol=0, atol=10)

    times = t0 + np.linspace(0, 5, 1000)*u.h
    for _ in range(10):
        while True:
            # Repeat near LEO
            r = np.random.uniform(7e6, 1e7)
            # Pick a random direction (not uniform on sphere)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            r = np.array([x, y, z])
            # Pick a velocity near VLEO
            v = np.random.uniform(7.7e3, 7.9e3)
            # and a randomish direction
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            vx = v * np.cos(theta) * np.sin(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(phi)
            v = np.array([vx, vy, vz])
            orbit = ssapy.Orbit(r, v, t0.gps)
            if norm(orbit.periapsis) > 6475e3:
                break

        r1, v1 = ssapy.rv(orbit, times)
        r2, v2 = ssapy.rv(
            orbit, times,
            propagator=ssapy.RK4Propagator(ssapy.AccelKepler(), h=30.0)
        )
        np.testing.assert_allclose(r1, r2, rtol=0, atol=10)


@timer
def test_scipy_propagator():
    # Test that analytic Keplerian propagation matches SciPy ODE propagator when
    # acceleration is purely Keplerian
    np.random.seed(577215664)
    t0 = Time("1982-03-14", scale='utc')
    times = t0 + np.linspace(0, 24, 1000)*u.h

    for _ in range(10):
        while True:
            # Pick a distance near GEO
            r = np.random.uniform(4e7, 5e7)
            # Pick a random direction (not uniform on sphere)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            r = np.array([x, y, z])
            # Pick a velocity near VGEO
            v = np.random.uniform(2.7e3, 3.3e3)
            # and a randomish direction
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            vx = v * np.cos(theta) * np.sin(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(phi)
            v = np.array([vx, vy, vz])
            orbit = ssapy.Orbit(r, v, t0)
            if norm(orbit.periapsis) > 1e7:
                break

        r1, v1 = ssapy.rv(orbit, times)
        r2, v2 = ssapy.rv(
            orbit, times,
            propagator=ssapy.SciPyPropagator(ssapy.AccelKepler())
        )
        np.testing.assert_allclose(r1, r2, rtol=0, atol=100)

    times = t0 + np.linspace(0, 5, 1000)*u.h
    for _ in range(10):
        while True:
            # Repeat near LEO
            r = np.random.uniform(7e6, 1e7)
            # Pick a random direction (not uniform on sphere)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            r = np.array([x, y, z])
            # Pick a velocity near VLEO
            v = np.random.uniform(7.7e3, 7.9e3)
            # and a randomish direction
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            vx = v * np.cos(theta) * np.sin(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(phi)
            v = np.array([vx, vy, vz])
            orbit = ssapy.Orbit(r, v, t0.gps)
            if norm(orbit.periapsis) > 6475e3:
                break

        r1, v1 = ssapy.rv(orbit, times)
        r2, v2 = ssapy.rv(
            orbit, times,
            propagator=ssapy.SciPyPropagator(ssapy.AccelKepler())
        )
        np.testing.assert_allclose(r1, r2, rtol=0, atol=100)


@timer
def test_RK8():
    # Test that analytic Keplerian propagation matches RK8 propagator when
    # acceleration is purely Keplerian
    np.random.seed(5772156)
    t0 = Time("1982-03-14", scale='utc')
    # times = t0 + np.linspace(0, 24, 1000)*u.h
    times = t0 + np.arange(1000)*70*u.s

    for _ in range(10):
        while True:
            # Pick a distance near GEO
            r = np.random.uniform(4e7, 5e7)
            # Pick a random direction (not uniform on sphere)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            r = np.array([x, y, z])
            # Pick a velocity near VGEO
            v = np.random.uniform(2.7e3, 3.3e3)
            # and a randomish direction
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            vx = v * np.cos(theta) * np.sin(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(phi)
            v = np.array([vx, vy, vz])
            orbit = ssapy.Orbit(r, v, t0)
            if norm(orbit.periapsis) > 1e7:
                break

        rk4 = ssapy.RK4Propagator(ssapy.AccelKepler(), h=70.0)
        rk8 = ssapy.RK8Propagator(ssapy.AccelKepler(), h=70.0)

        r0, v0 = ssapy.rv(orbit, times)
        r4, v4 = ssapy.rv(orbit, times, propagator=rk4)
        r8, v8 = ssapy.rv(orbit, times, propagator=rk8)

        np.testing.assert_allclose(r0, r4, rtol=0, atol=2)
        np.testing.assert_allclose(r0, r8, rtol=0, atol=1e-6)

    times = t0 + np.arange(0, 500)*30*u.s
    for _ in range(10):
        while True:
            # Repeat near LEO
            r = np.random.uniform(7e6, 1e7)
            # Pick a random direction (not uniform on sphere)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            r = np.array([x, y, z])
            # Pick a velocity near VLEO
            v = np.random.uniform(7.7e3, 7.9e3)
            # and a randomish direction
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            vx = v * np.cos(theta) * np.sin(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(phi)
            v = np.array([vx, vy, vz])
            orbit = ssapy.Orbit(r, v, t0.gps)
            if norm(orbit.periapsis) > 6475e3:
                break

        rk4 = ssapy.RK4Propagator(ssapy.AccelKepler(), h=20.0)
        rk8 = ssapy.RK8Propagator(ssapy.AccelKepler(), h=20.0)

        r0, v0 = ssapy.rv(orbit, times)
        r4, v4 = ssapy.rv(orbit, times, propagator=rk4)
        r8, v8 = ssapy.rv(orbit, times, propagator=rk8)

        np.testing.assert_allclose(r0, r4, rtol=0, atol=1)
        np.testing.assert_allclose(r0, r8, rtol=0, atol=1e-5)


@timer
def test_RK78():
    # Test that analytic Keplerian propagation matches RK78 propagator when
    # acceleration is purely Keplerian
    np.random.seed(5772156)
    t0 = Time("1982-03-14", scale='utc')
    # times = t0 + np.linspace(0, 24, 1000)*u.h
    times = t0 + np.arange(1000)*70*u.s

    for _ in range(10):
        while True:
            # Pick a distance near GEO
            r = np.random.uniform(4e7, 5e7)
            # Pick a random direction (not uniform on sphere)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            r = np.array([x, y, z])
            # Pick a velocity near VGEO
            v = np.random.uniform(2.7e3, 3.3e3)
            # and a randomish direction
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            vx = v * np.cos(theta) * np.sin(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(phi)
            v = np.array([vx, vy, vz])
            orbit = ssapy.Orbit(r, v, t0)
            if norm(orbit.periapsis) > 1e7:
                break

        r1, v1 = ssapy.rv(orbit, times)
        r2, v2 = ssapy.rv(
            orbit, times,
            propagator=ssapy.RK78Propagator(
                ssapy.AccelKepler(),
                h=60.0,
                tol=(1e-6,)*3+(1e-9,)*3
            )
        )
        np.testing.assert_allclose(r1, r2, rtol=0, atol=1e-2)

    times = t0 + np.arange(0, 500)*30*u.s
    for _ in range(10):
        while True:
            # Repeat near LEO
            r = np.random.uniform(7e6, 1e7)
            # Pick a random direction (not uniform on sphere)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            r = np.array([x, y, z])
            # Pick a velocity near VLEO
            v = np.random.uniform(7.7e3, 7.9e3)
            # and a randomish direction
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            vx = v * np.cos(theta) * np.sin(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(phi)
            v = np.array([vx, vy, vz])
            orbit = ssapy.Orbit(r, v, t0.gps)
            if norm(orbit.periapsis) > 6475e3:
                break

        r1, v1 = ssapy.rv(orbit, times)
        r2, v2 = ssapy.rv(
            orbit, times,
            propagator=ssapy.RK78Propagator(
                ssapy.AccelKepler(),
                h=60.0,
                tol=(1e-6,)*3+(1e-9,)*3
            )
        )
        np.testing.assert_allclose(r1, r2, rtol=0, atol=1e-2)


@timer
def test_reverse():
    t = Time("1999-03-01", scale='utc')
    # LEO sat
    kElements = [7178e3, 0.001, np.deg2rad(98.57), 0.0, 0.0, 0.0]
    kwargs = dict(
        mass = 1000.0,  # [kg]
        area = 5.0,  # [m^2]
        CD = 2.3,
        CR = 1.3,
    )
    orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)

    aH44 = ssapy.AccelHarmonic(earth_MG, n_max=4, m_max=4) + ssapy.AccelKepler(mu=earth_MG.mu)
    aSun = ssapy.AccelThirdBody(sun_MG)
    aMoon = ssapy.AccelThirdBody(moon_MG)
    aSolRad = ssapy.AccelSolRad()
    aDrag = ssapy.AccelDrag()

    accel = ssapy.AccelSum([aH44, aSun, aMoon, aSolRad, aDrag])

    times = t + np.linspace(0, orbit.period, 1000)*u.s

    for prop in [
        ssapy.RK4Propagator(accel, h=25.0),
        ssapy.RK8Propagator(accel, h=100.0),
        ssapy.SciPyPropagator(
            accel,
            ode_kwargs=dict(
                method='DOP853',
                rtol=1e-9,
                atol=(1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4)
            )
        )
    ]:
        r0, v0 = ssapy.rv(
            orbit,
            times,
            propagator=prop
        )

        orbitFinal = ssapy.Orbit(r0[-1], v0[-1], times[-1], propkw=kwargs)
        r1, v1 = ssapy.rv(
            orbitFinal,
            times,
            propagator=prop
        )

        np.testing.assert_allclose(r0, r1, rtol=0, atol=1)
        np.testing.assert_allclose(v0, v1, rtol=0, atol=1e-3)

        # Should be able to start from the middle too.
        orbitMiddle = ssapy.Orbit(r0[500], v0[500], times[500], propkw=kwargs)
        r2, v2 = ssapy.rv(
            orbitMiddle,
            times,
            propagator=prop
        )

        np.testing.assert_allclose(r0, r2, rtol=0, atol=1)
        np.testing.assert_allclose(v0, v2, rtol=0, atol=1e-3)


@timer
def test_Hohmann_transfer():
    """
    Test Hohmann transfer between Low Earth Orbit (LEO) and Geostationary 
    Orbit (GEO).  Check that the semi-major axis (a), eccentricity (e), and
    inclination (i) are close to expectations.
    """
    earthrad = ssapy.constants.WGS84_EARTH_RADIUS
    rleo = earthrad + 300*1000
    rgeo = ssapy.constants.RGEO
    rellip = (rleo + rgeo)/2
    mu = ssapy.constants.WGS84_EARTH_MU
    v1sq = mu*(2/rleo-1/rleo)
    v2sq = mu*(2/rgeo-1/rgeo)
    vesq1 = mu*(2/rleo-1/rellip)
    vesq2 = mu*(2/rgeo-1/rellip)
    dv1 = np.sqrt(vesq1)-np.sqrt(v1sq)
    dv2 = np.sqrt(v2sq)-np.sqrt(vesq2)
    T1 = 2*np.pi*np.sqrt(rleo**3/mu)
    Te = 2*np.pi*np.sqrt(rellip**3/mu)
    t0 = Time('2020-01-01T00:00:00').gps
    orbleo = ssapy.Orbit.fromKeplerianElements(rleo, 0, 0, 0, 0, 0, t0)
    # Two burns, separated by half a period for the elliptical orbit
    burn1 = ssapy.AccelConstNTW(
        [0, dv1, 0], time_breakpoints=[t0+T1-0.5, t0+T1+0.5])
    burn2 = ssapy.AccelConstNTW(
        [0, dv2, 0], time_breakpoints=[t0+T1+Te/2-0.5, t0+T1+Te/2+0.5])
    accel = ssapy.AccelSum([ssapy.AccelKepler(mu), burn1, burn2])
    prop = ssapy.propagator.default_numerical(accel=accel)
    rr, vv = ssapy.compute.rv(orbleo, t0+np.arange(8640)*300, prop)
    orbfinal = ssapy.Orbit(r=rr[-1], v=vv[-1], t=t0+(8640-1)*300)
    np.testing.assert_allclose(orbfinal.a, ssapy.constants.RGEO, atol=10)
    np.testing.assert_allclose(orbfinal.e, 0, atol=1e-5)
    np.testing.assert_allclose(orbfinal.i, 0, atol=1e-5)

def test_inclination_change():
    """
    Test inclination change with a small delta-v.  Check that the semi-major
    axis (a), eccentricity (e), and inclination (i) are close to expectations.
    """
    rgeo = ssapy.constants.RGEO
    mu = ssapy.constants.WGS84_EARTH_MU
    t0 = Time('2020-01-01T00:00:00').gps
    T1 = 2*np.pi*np.sqrt(rgeo**3/mu)
    di = 0.01  # change inclination by 0.01 rad
    n = 2 * np.pi / T1  # mean motion
    dvi = di * n * rgeo  # approximate delta-v needed for plane change
    burni = ssapy.AccelConstNTW(
        [0, 0, dvi],
        time_breakpoints=[t0+T1-0.5, t0+T1+0.5]
    )
    acceli = ssapy.AccelSum([ssapy.AccelKepler(mu), burni])
    propi = ssapy.propagator.default_numerical(accel=acceli)
    orbgeo = ssapy.Orbit.fromKeplerianElements(rgeo, 0, 0, 0, 0, 0, t0)
    rri, vvi = ssapy.compute.rv(orbgeo, t0+np.arange(8640)*200, propi)
    orbfinal = ssapy.Orbit(r=rri[-1], v=vvi[-1], t=t0+(8640-1)*200)
    np.testing.assert_allclose(orbfinal.a, ssapy.constants.RGEO, atol=1)
    np.testing.assert_allclose(orbfinal.e, 0, atol=1e-6)
    np.testing.assert_allclose(orbfinal.i, di, atol=1e-6)

def test_bielliptic_transfer():
    """
    Test bielliptic transfer between two orbits.  Check that the semi-major
    axis (a), eccentricity (e), and inclination (i) are close to expectations.
    """
    mu = ssapy.constants.WGS84_EARTH_MU
    t0 = Time('2020-01-01T00:00:00').gps
    a1 = 7000000
    a4 = 105000000
    rb = 210000000
    a2 = (a1+rb)/2
    a3 = (a4+rb)/2
    v1sq = mu*(2/a1-1/a1)
    v2sq1 = mu*(2/a1-1/a2)
    v2sq2 = mu*(2/rb-1/a2)
    v3sq1 = mu*(2/rb-1/a3)
    v3sq2 = mu*(2/a4-1/a3)
    v4sq = mu*(2/a4-1/a4)
    dv1 = np.sqrt(v2sq1)-np.sqrt(v1sq)
    dv2 = np.sqrt(v3sq1)-np.sqrt(v2sq2)
    dv3 = np.sqrt(v4sq)-np.sqrt(v3sq2)
    T1 = 2*np.pi*np.sqrt(a1**3/mu)
    T2 = 2*np.pi*np.sqrt(a2**3/mu)
    T3 = 2*np.pi*np.sqrt(a3**3/mu)
    T4 = 2*np.pi*np.sqrt(a4**3/mu)
    orbi = ssapy.Orbit.fromKeplerianElements(a1, 0, 0, 0, 0, 0, t0)
    # Three burns, separated by half periods for the elliptical orbit
    dt = 1  # seconds
    burn1 = ssapy.AccelConstNTW(
        [0, dv1/dt, 0],
        time_breakpoints=[t0+T1-dt/2, t0+T1+dt/2])
    burn2 = ssapy.AccelConstNTW(
        [0, dv2/dt, 0],
        time_breakpoints=[t0+T1+T2/2-dt/2, t0+T1+T2/2+dt/2])
    burn3 = ssapy.AccelConstNTW(
        [0, dv3/dt, 0],
        time_breakpoints=[t0+T1+T2/2+T3/2-dt/2, t0+T1+T2/2+T3/2+dt/2])
    accel = ssapy.AccelSum([
            ssapy.AccelKepler(mu), burn1, burn2, burn3])
    prop = ssapy.propagator.default_numerical(accel=accel)
    tt = t0+(T1+(T2+T3)/2+T4)*np.linspace(0, 1, 10000)
    rr, vv = ssapy.rv(
        orbi, tt, prop)
    orbf = ssapy.Orbit(r=rr[-1], v=vv[-1], t=tt[-1])
    np.testing.assert_allclose(orbf.a, a4, atol=100)
    np.testing.assert_allclose(orbf.e, 0, atol=1e-4)
    np.testing.assert_allclose(orbf.i, 0, atol=1e-6)


if __name__ == '__main__':
    test_MG_3_1()
    test_MG_3_2()
    test_accel_point_mass()
    test_angles()
    test_MG_3_4_accel()
    test_MG_3_4()
    test_MG_3_4_scipy()
    test_MG_3_4_rk78()
    test_RK4_vs_analytic()
    test_scipy_propagator()
    test_RK8()
    test_RK78()
    test_reverse()
    test_Hohmann_transfer()
    test_inclination_change()
    test_bielliptic_transfer()

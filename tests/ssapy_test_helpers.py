import numpy as np


def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2


def checkAngle(a, b, rtol=0, atol=1e-14):
    diff = (a-b)%(2*np.pi)
    absdiff = np.min([np.abs(diff), np.abs(2*np.pi-diff)], axis=0)
    np.testing.assert_allclose(absdiff, 0, rtol=0, atol=atol)


def checkSphere(lon1, lat1, lon2, lat2, atol=1e-14, verbose=False):
    from ssapy.utils import unitAngle3
    x1 = np.cos(lon1)*np.cos(lat1)
    y1 = np.sin(lon1)*np.cos(lat1)
    z1 = np.sin(lat1)
    x2 = np.cos(lon2)*np.cos(lat2)
    y2 = np.sin(lon2)*np.cos(lat2)
    z2 = np.sin(lat2)
    da = unitAngle3(np.array([x1, y1, z1]).T, np.array([x2, y2, z2]).T)
    if verbose:
        print(f"max angle difference {np.rad2deg(np.max(da))*3600} arcsec")
    np.testing.assert_allclose(da, 0, rtol=0, atol=atol)

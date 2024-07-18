# flake8: noqa: E501

from .constants import EARTH_RADIUS, WGS84_EARTH_OMEGA
from .accel import AccelKepler
from .body import get_body, MoonPosition
from .compute import groundTrack, rv
from .constants import RGEO
from .orbit import Orbit
from .propagator import RK78Propagator
from .utils import hms_to_dd, dd_to_hms, dd_to_dms

import numpy as np
from astropy.time import Time

# VECTOR FUNCTIONS FOR COORDINATE MATH
def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def getAngle(a, b, c):  # a,b,c where b is the vertex
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    ba = np.subtract(a, b)
    bc = np.subtract(c, b)
    cosine_angle = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.arccos(cosine_angle)


def angle_between_vectors(vector1, vector2):
    return np.arccos(np.clip(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)), -1.0, 1.0))


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def normed(arr):
    return arr / np.sqrt(np.einsum("...i,...i", arr, arr))[..., None]


def einsum_norm(a, indices='ij,ji->i'):
    return np.sqrt(np.einsum(indices, a, a))


def normSq(arr):
    return np.einsum("...i,...i", arr, arr)


def norm(arr):
    return np.sqrt(np.einsum("...i,...i", arr, arr))


def rotate_vector(v_unit, theta, phi):
    v_unit = v_unit / np.linalg.norm(v_unit, axis=-1)
    if np.all(np.abs(v_unit) != np.max(np.abs(v_unit))):
        perp_vector = np.cross(v_unit, np.array([1, 0, 0]))
    else:
        perp_vector = np.cross(v_unit, np.array([0, 1, 0]))
    perp_vector /= np.linalg.norm(perp_vector)

    theta = np.radians(theta)
    phi = np.radians(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    R1 = np.array([
        [cos_theta + (1 - cos_theta) * perp_vector[0]**2, 
         (1 - cos_theta) * perp_vector[0] * perp_vector[1] - sin_theta * perp_vector[2], 
         (1 - cos_theta) * perp_vector[0] * perp_vector[2] + sin_theta * perp_vector[1]],
        [(1 - cos_theta) * perp_vector[1] * perp_vector[0] + sin_theta * perp_vector[2], 
         cos_theta + (1 - cos_theta) * perp_vector[1]**2, 
         (1 - cos_theta) * perp_vector[1] * perp_vector[2] - sin_theta * perp_vector[0]],
        [(1 - cos_theta) * perp_vector[2] * perp_vector[0] - sin_theta * perp_vector[1], 
         (1 - cos_theta) * perp_vector[2] * perp_vector[1] + sin_theta * perp_vector[0], 
         cos_theta + (1 - cos_theta) * perp_vector[2]**2]
    ])

    # Apply the rotation matrix to v_unit to get the rotated unit vector
    v1 = np.dot(R1, v_unit)

    # Rotation matrix for rotation about v_unit
    R2 = np.array([[cos_phi + (1 - cos_phi) * v_unit[0]**2,
                    (1 - cos_phi) * v_unit[0] * v_unit[1] - sin_phi * v_unit[2],
                    (1 - cos_phi) * v_unit[0] * v_unit[2] + sin_phi * v_unit[1]],
                   [(1 - cos_phi) * v_unit[1] * v_unit[0] + sin_phi * v_unit[2],
                    cos_phi + (1 - cos_phi) * v_unit[1]**2,
                    (1 - cos_phi) * v_unit[1] * v_unit[2] - sin_phi * v_unit[0]],
                   [(1 - cos_phi) * v_unit[2] * v_unit[0] - sin_phi * v_unit[1],
                    (1 - cos_phi) * v_unit[2] * v_unit[1] + sin_phi * v_unit[0],
                    cos_phi + (1 - cos_phi) * v_unit[2]**2]])

    v2 = np.dot(R2, v1)
    return v2 / np.linalg.norm(v2, axis=-1)


def rotate_points_3d(points, axis=np.array([0, 0, 1]), theta=-np.pi / 2):
    """
    Rotate a set of 3D points about a 3D axis by an angle theta in radians.

    Args:
        points (np.ndarray): The set of 3D points to rotate, as an Nx3 array.
        axis (np.ndarray): The 3D axis to rotate about, as a length-3 array. Default is the z-axis.
        theta (float): The angle to rotate by, in radians. Default is pi/2.

    Returns:
        np.ndarray: The rotated set of 3D points, as an Nx3 array.
    """
    # Normalize the axis to be a unit vector
    axis = axis / np.linalg.norm(axis)

    # Compute the quaternion representing the rotation
    qw = np.cos(theta / 2)
    qx, qy, qz = axis * np.sin(theta / 2)

    # Construct the rotation matrix from the quaternion
    R = np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])

    # Apply the rotation matrix to the set of points
    rotated_points = np.dot(R, points.T).T

    return rotated_points


def perpendicular_vectors(v):
    """Returns two vectors that are perpendicular to v and each other."""
    # Check if v is the zero vector
    if np.allclose(v, np.zeros_like(v)):
        raise ValueError("Input vector cannot be the zero vector.")

    # Choose an arbitrary non-zero vector w that is not parallel to v
    w = np.array([1., 0., 0.])
    if np.allclose(v, w) or np.allclose(v, -w):
        w = np.array([0., 1., 0.])
    u = np.cross(v, w)
    if np.allclose(u, np.zeros_like(u)):
        w = np.array([0., 0., 1.])
        u = np.cross(v, w)
    w = np.cross(v, u)

    return u, w


def points_on_circle(r, v, rad, num_points=4):
    # Convert inputs to numpy arrays
    r = np.array(r)
    v = np.array(v)

    # Find the perpendicular vectors to the given vector v
    if np.all(v[:2] == 0):
        if np.all(v[2] == 0):
            raise ValueError("The given vector v must not be the zero vector.")
        else:
            u = np.array([1, 0, 0])
    else:
        u = np.array([-v[1], v[0], 0])
    u = u / np.linalg.norm(u)
    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-15:
        # v is parallel to z-axis
        w = np.array([0, 1, 0])
    else:
        w = w / w_norm
    # Generate a sequence of angles for equally spaced points
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Compute the x, y, z coordinates of each point on the circle
    x = rad * np.cos(angles) * u[0] + rad * np.sin(angles) * w[0]
    y = rad * np.cos(angles) * u[1] + rad * np.sin(angles) * w[1]
    z = rad * np.cos(angles) * u[2] + rad * np.sin(angles) * w[2]

    # Apply rotation about z-axis by 90 degrees
    rot_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rotated_points = np.dot(rot_matrix, np.column_stack((x, y, z)).T).T

    # Translate the rotated points to the center point r
    points_rotated = rotated_points + r.reshape(1, 3)

    return points_rotated


def dms_to_rad(coords):
    from astropy.coordinates import Angle
    if isinstance(coords, (list, tuple)):
        return [Angle(coord).radian for coord in coords]
    else:
        return Angle(coords).radian
    return


def dms_to_deg(coords):
    from astropy.coordinates import Angle
    if isinstance(coords, (list, tuple)):
        return [Angle(coord).deg for coord in coords]
    else:
        return Angle(coords).deg
    return


def rad0to2pi(angles):
    return (2 * np.pi + angles) * (angles < 0) + angles * (angles > 0)


def deg0to360(array_):
    try:
        return [i % 360 for i in array_]
    except TypeError:
        return array_ % 360


def deg0to360array(array_):
    return [i % 360 for i in array_]


def deg90to90(val_in):
    if hasattr(val_in, "__len__"):
        val_out = []
        for i, v in enumerate(val_in):
            while v < -90:
                v += 90
            while v > 90:
                v -= 90
            val_out.append(v)
    else:
        while val_in < -90:
            val_in += 90
        while val_in > 90:
            val_in -= 90
        val_out = val_in
    return val_out


def deg90to90array(array_):
    return [i % 90 for i in array_]


def cart2sph_deg(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy) * (180 / np.pi)
    az = (np.arctan2(y, x)) * (180 / np.pi)
    return az, el, r


def cart_to_cyl(x, y, z):
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta, z


def inert2rot(x, y, xe, ye, xs=0, ys=0):  # Places Earth at (-1,0)
    earth_theta = np.arctan2(ye - ys, xe - xs)
    theta = np.arctan2(y - ys, x - xs)
    distance = np.sqrt(np.power((x - xs), 2) + np.power((y - ys), 2))
    xrot = distance * np.cos(np.pi + (theta - earth_theta))
    yrot = distance * np.sin(np.pi + (theta - earth_theta))
    return xrot, yrot


def sim_lonlatrad(x, y, z, xe, ye, ze, xs, ys, zs):
    # convert all to geo coordinates
    x = x - xe
    y = y - ye
    z = z - ze
    xs = xs - xe
    ys = ys - ye
    zs = zs - ze
    # convert x y z to lon lat radius
    longitude, latitude, radius = cart2sph_deg(x, y, z)
    slongitude, slatitude, sradius = cart2sph_deg(xs, ys, zs)
    # correct so that Sun is at (0,0)
    longitude = deg0to360(slongitude - longitude)
    latitude = latitude - slatitude
    return longitude, latitude, radius


def sun_ra_dec(time_):
    out = get_body(Time(time_, format='mjd'))
    return out.ra.to('rad').value, out.dec.to('rad').value


def ra_dec(r=None, v=None, x=None, y=None, z=None, vx=None, vy=None, vz=None, r_earth=np.array([0, 0, 0]), v_earth=np.array([0, 0, 0]), input_unit='si'):
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth

    d_earth_mag = einsum_norm(r, 'ij,ij->i')
    ra = rad0to2pi(np.arctan2(r[:, 1], r[:, 0]))  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    return ra, dec


def lonlat_distance(lat1, lat2, lon1, lon2):
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers. Use 3956 for miles
    # calculate the result
    return (c * EARTH_RADIUS)


def altitude2zenithangle(altitude, deg=True):
    if deg:
        out = 90 - altitude
    else:
        out = np.pi / 2 - altitude
    return out


def zenithangle2altitude(zenith_angle, deg=True):
    if deg:
        out = 90 - zenith_angle
    else:
        out = np.pi / 2 - zenith_angle
    return out


def rightasension2hourangle(right_ascension, local_time):
    if type(right_ascension) is not str:
        right_ascension = dd_to_hms(right_ascension)
    if type(local_time) is not str:
        local_time = dd_to_dms(local_time)
    _ra = float(right_ascension.split(':')[0])
    _lt = float(local_time.split(':')[0])
    if _ra > _lt:
        __ltm, __lts = local_time.split(':')[1:]
        local_time = f'{24 + _lt}:{__ltm}:{__lts}'

    return dd_to_dms(hms_to_dd(local_time) - hms_to_dd(right_ascension))


def equatorial_to_horizontal(observer_latitude, declination, right_ascension=None, hour_angle=None, local_time=None, hms=False):
    if right_ascension is not None:
        hour_angle = rightasension2hourangle(right_ascension, local_time)
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    elif hour_angle is not None:
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    elif right_ascension is not None and hour_angle is not None:
        print('Both right_ascension and hour_angle parameters are provided.\nUsing hour_angle for calculations.')
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    else:
        print('Either right_ascension or hour_angle must be provided.')

    observer_latitude, hour_angle, declination = np.radians([observer_latitude, hour_angle, declination])

    zenith_angle = np.arccos(np.sin(observer_latitude) * np.sin(declination) + np.cos(observer_latitude) * np.cos(declination) * np.cos(hour_angle))

    altitude = zenithangle2altitude(zenith_angle, deg=False)

    _num = np.sin(declination) - np.sin(observer_latitude) * np.cos(zenith_angle)
    _den = np.cos(observer_latitude) * np.sin(zenith_angle)
    azimuth = np.arccos(_num / _den)

    if observer_latitude < 0:
        azimuth = np.pi - azimuth
    altitude, azimuth = np.degrees([altitude, azimuth])

    return azimuth, altitude


def horizontal_to_equatorial(observer_latitude, azimuth, altitude):
    altitude, azimuth, latitude = np.radians([altitude, azimuth, observer_latitude])
    zenith_angle = zenithangle2altitude(altitude)

    zenith_angle = [-zenith_angle if latitude < 0 else zenith_angle][0]

    declination = np.sin(latitude) * np.cos(zenith_angle)
    declination = declination + (np.cos(latitude) * np.sin(zenith_angle) * np.cos(azimuth))
    declination = np.arcsin(declination)

    _num = np.cos(zenith_angle) - np.sin(latitude) * np.sin(declination)
    _den = np.cos(latitude) * np.cos(declination)
    hour_angle = np.arccos(_num / _den)

    if (latitude > 0 > declination) or (latitude < 0 < declination):
        hour_angle = 2 * np.pi - hour_angle

    declination, hour_angle = np.degrees([declination, hour_angle])

    return hour_angle, declination


_ecliptic = 0.409092601  # np.radians(23.43927944)
cos_ec = 0.9174821430960974
sin_ec = 0.3977769690414367


def equatorial_xyz_to_ecliptic_xyz(xq, yq, zq):
    xc = xq
    yc = cos_ec * yq + sin_ec * zq
    zc = -sin_ec * yq + cos_ec * zq
    return xc, yc, zc


def ecliptic_xyz_to_equatorial_xyz(xc, yc, zc):
    xq = xc
    yq = cos_ec * yc - sin_ec * zc
    zq = sin_ec * yc + cos_ec * zc
    return xq, yq, zq


def xyz_to_ecliptic(xc, yc, zc, xe=0, ye=0, ze=0, degrees=False):
    x_ast_to_earth = xc - xe
    y_ast_to_earth = yc - ye
    z_ast_to_earth = zc - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ec_longitude = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    ec_latitude = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ec_longitude), np.degrees(ec_latitude)
    else:
        return ec_longitude, ec_latitude


def xyz_to_equatorial(xq, yq, zq, xe=0, ye=0, ze=0, degrees=False):
    # RA / DEC calculation - assumes XY plane to be celestial equator, and -x axis to be vernal equinox
    x_ast_to_earth = xq - xe
    y_ast_to_earth = yq - ye
    z_ast_to_earth = zq - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def ecliptic_xyz_to_equatorial(xc, yc, zc, xe=0, ye=0, ze=0, degrees=False):
    # Convert ecliptic cartesian into equitorial cartesian
    x_ast_to_earth, y_ast_to_earth, z_ast_to_earth = ecliptic_xyz_to_equatorial_xyz(xc - xe, yc - ye, zc - ze)
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def equatorial_to_ecliptic(right_ascension, declination, degrees=False):
    ra, dec = np.radians(right_ascension), np.radians(declination)
    ec_latitude = np.arcsin(cos_ec * np.sin(dec) - sin_ec * np.cos(dec) * np.sin(ra))
    ec_longitude = np.arctan((cos_ec * np.cos(dec) * np.sin(ra) + sin_ec * np.sin(dec)) / (np.cos(dec) * np.cos(ra)))
    if degrees:
        return deg0to360(np.degrees(ec_longitude)), np.degrees(ec_latitude)
    else:
        return rad0to2pi(ec_longitude), ec_latitude


def ecliptic_to_equatorial(lon, lat, degrees=False):
    lon, lat = np.radians(lon), np.radians(lat)
    ra = np.arctan((cos_ec * np.cos(lat) * np.sin(lon) - sin_ec * np.sin(lat)) / (np.cos(lat) * np.cos(lon)))
    dec = np.arcsin(cos_ec * np.sin(lat) + sin_ec * np.cos(lat) * np.sin(lon))
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def proper_motion_ra_dec(r=None, v=None, x=None, y=None, z=None, vx=None, vy=None, vz=None, r_earth=np.array([0, 0, 0]), v_earth=np.array([0, 0, 0]), input_unit='si'):
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth

    # Distances to Earth and Sun
    d_earth_mag = einsum_norm(r, 'ij,ij->i')

    # RA / DEC calculation
    ra = rad0to2pi(np.arctan2(r[:, 1], r[:, 0]))  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    ra_unit_vector = np.array([-np.sin(ra), np.cos(ra), np.zeros(np.shape(ra))]).T
    dec_unit_vector = -np.array([np.cos(np.pi / 2 - dec) * np.cos(ra), np.cos(np.pi / 2 - dec) * np.sin(ra), -np.sin(np.pi / 2 - dec)]).T
    pmra = (np.einsum('ij,ij->i', v, ra_unit_vector)) / d_earth_mag * 206265  # arcseconds / second
    pmdec = (np.einsum('ij,ij->i', v, dec_unit_vector)) / d_earth_mag * 206265  # arcseconds / second

    if input_unit == 'si':
        return pmra, pmdec
    elif input_unit == 'rebound':
        pmra = pmra / (31557600 * 2 * np.pi)
        pmdec = pmdec / (31557600 * 2 * np.pi)  # arcseconds * (au/sim_time)/au, convert to arcseconds / second
        return pmra, pmdec
    else:
        print('Error - units provided not available, provide either SI or rebound units.')
        return



def gcrf_to_lunar(r, t, v=None):
    class MoonRotator:
        def __init__(self):
            self.mpm = MoonPosition()

        def __call__(self, r, t):
            rmoon = self.mpm(t)
            vmoon = (self.mpm(t + 5.0) - self.mpm(t - 5.0)) / 10.
            xhat = normed(rmoon.T).T
            vpar = np.einsum("ab,ab->b", xhat, vmoon) * xhat
            vperp = vmoon - vpar
            yhat = normed(vperp.T).T
            zhat = np.cross(xhat, yhat, axisa=0, axisb=0).T
            R = np.empty((3, 3, len(t)))
            R[0] = xhat
            R[1] = yhat
            R[2] = zhat
            return np.einsum("abc,cb->ca", R, r)
    rotator = MoonRotator()
    if v is None:
        return rotator(r, t)
    else:
        r_lunar = rotator(r, t)
        v_lunar = v_from_r(r_lunar, t)
        return r_lunar, v_lunar


def gcrf_to_lunar_fixed(r, t, v=None):
    r_lunar = gcrf_to_lunar(r, t) - gcrf_to_lunar(get_body('moon').position(t).T, t)
    if v is None:
        return r_lunar
    else:
        v = v_from_r(r_lunar, t)
        return r_lunar, v


def gcrf_to_radec(gcrf_coords):
    x, y, z = gcrf_coords
    # Calculate right ascension in radians
    ra = np.arctan2(y, x)
    # Convert right ascension to degrees
    ra_deg = np.degrees(ra)
    # Normalize right ascension to the range [0, 360)
    ra_deg = ra_deg % 360
    # Calculate declination in radians
    dec_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    # Convert declination to degrees
    dec_deg = np.degrees(dec_rad)
    return (ra_deg, dec_deg)


def gcrf_to_ecef_bad(r_gcrf, t):
    if isinstance(t, Time):
        t = t.gps
    r_gcrf = np.atleast_2d(r_gcrf)
    rotation_angles = WGS84_EARTH_OMEGA * (t - Time("1980-3-20T11:06:00", format='isot').gps)
    cos_thetas = np.cos(rotation_angles)
    sin_thetas = np.sin(rotation_angles)

    # Create an array of 3x3 rotation matrices
    Rz = np.array([[cos_thetas, -sin_thetas, np.zeros_like(cos_thetas)],
                  [sin_thetas, cos_thetas, np.zeros_like(cos_thetas)],
                  [np.zeros_like(cos_thetas), np.zeros_like(cos_thetas), np.ones_like(cos_thetas)]]).T

    # Apply the rotation matrices to all rows of r_gcrf simultaneously
    r_ecef = np.einsum('ijk,ik->ij', Rz, r_gcrf)
    return r_ecef


def gcrf_to_lat_lon(r, t):
    lon, lat, height = groundTrack(r, t)
    return lon, lat, height


def gcrf_to_itrf(r_gcrf, t, v=None):
    x, y, z = groundTrack(r_gcrf, t, format='cartesian')
    _ = np.array([x, y, z]).T
    if v is None:
        return _
    else:
        return _, v_from_r(_, t)


def gcrf_to_sim_geo(r_gcrf, t, h=10):
    if np.min(np.diff(t.gps)) < h:
        h = np.min(np.diff(t.gps))
    r_gcrf = np.atleast_2d(r_gcrf)
    r_geo, v_geo = rv(Orbit.fromKeplerianElements(*[RGEO, 0, 0, 0, 0, 0], t=t[0]), t, propagator=RK78Propagator(AccelKepler(), h=h))
    angle_geo_to_x = np.arctan2(r_geo[:, 1], r_geo[:, 0])
    c = np.cos(angle_geo_to_x)
    s = np.sin(angle_geo_to_x)
    rotation = np.array([[c, -s, np.zeros_like(c)], [s, c, np.zeros_like(c)], [np.zeros_like(c), np.zeros_like(c), np.ones_like(c)]]).T
    return np.einsum('ijk,ik->ij', rotation, r_gcrf)


# Function still in development, not 100% accurate.
def gcrf_to_itrf_astropy(state_vectors, t):
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, SkyCoord, get_body_barycentric, solar_system_ephemeris, ICRS

    sc = SkyCoord(x=state_vectors[:, 0] * u.m, y=state_vectors[:, 1] * u.m, z=state_vectors[:, 2] * u.m, representation_type='cartesian', frame=GCRS(obstime=t))
    sc_itrs = sc.transform_to(ITRS(obstime=t))
    with solar_system_ephemeris.set('de430'):  # other options: builtin, de432s
        earth = get_body_barycentric('earth', t)
    earth_center_itrs = SkyCoord(earth.x, earth.y, earth.z, representation_type='cartesian', frame=ICRS()).transform_to(ITRS(obstime=t))
    itrs_coords = SkyCoord(
        sc_itrs.x.value - earth_center_itrs.x.to_value(u.m),
        sc_itrs.y.value - earth_center_itrs.y.to_value(u.m),
        sc_itrs.z.value - earth_center_itrs.z.to_value(u.m),
        representation_type='cartesian',
        frame=ITRS(obstime=t)
    )
    # Extract Cartesian coordinates and convert to meters
    itrs_coords_meters = np.array([itrs_coords.x,
                                  itrs_coords.y,
                                  itrs_coords.z]).T
    return itrs_coords_meters


def v_from_r(r, t):
    if isinstance(t[0], Time):
        t = t.gps        
    delta_r = np.diff(r, axis=0)
    delta_t = np.diff(t)
    v = delta_r / delta_t[:, np.newaxis]
    v = np.vstack((v, v[-1]))
    return v

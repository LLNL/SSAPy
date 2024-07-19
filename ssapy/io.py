"""
Collection of functions to read from and write to various file formats.
"""

import datetime
import numpy as np
from astropy.time import Time
import astropy.units as u
from .constants import EARTH_MU


telescope_catalog = {
    "511": {
        "name": "SST",
        "x": -1496.451 * u.km,
        "y": -5096.210 * u.km,
        "z": 3523.795 * u.km
    },
    "241": {
        "name": "Diego Garcia",
        "x": 1907.068 * u.km,
        "y": 6030.792 * u.km,
        "z": -817.291 * u.km
    }
}


def read_tle_catalog(fname, n_lines=2):
    """
    Read in a TLE catalog file

    Parameters
    ----------
    fname : string
        The filename
    n_lines : int
        number of lines per tle, usually 2 but sometimes 3

    Returns
    -------
    catalog : list
        lists containing TLEs
    """
    with open(fname) as f:
        dat = f.readlines()

    tles = [
        [dat[i + _].rstrip() for _ in range(n_lines)]
        for i in range(0, len(dat), n_lines)
    ]
    return tles


def read_tle(sat_name, tle_filename):
    """
    Get the TLE data from the file for the satellite with the given name

    Parameters
    ---------
    sat_name : str
        NORAD name of the satellite
    tle_filename : str
        Path and name of file where TLE is

    Returns
    -------
    line1, line2 : str
        Both lines of the satellite TLE
    """
    with open(tle_filename) as tle_f:
        lines = [_.rstrip() for _ in tle_f.readlines()]
    try:
        sat_line_ind = lines.index(sat_name)
    except ValueError:
        raise KeyError(
            "No satellite '{}' in file '{}'".format(sat_name, tle_filename))
    try:
        line1 = lines[sat_line_ind + 1]
        line2 = lines[sat_line_ind + 2]
        return line1, line2
    except IndexError:
        raise IOError("Incorrectly formatted TLE file")


def _rvt_from_tle_tuple(tle_tuple):
    """
    Get r, v, t (in the TEME frame!) from TLE tuple

    Parameters
    ----------
    tle_tuple : 2-tuple of str
        Line1 and Line2 of TLE as strings

    Returns
    -------
    r : array_like (3,)
        Position in meters in TEME frame
    v : array_like 3(,)
        Velocity in meters in TEME frame
    t : float
        Time in GPS seconds; i.e., seconds since 1980-01-06 00:00:00 UTC.

    Notes
    -----
    This function returns positions and velocities in the TEME frame!  This is
    not the same frame as used by ssapy.Orbit constructor.  Please use
    ssapy.Orbit.fromTLETuple() to construct an Orbit object from a TLE.
    """
    from sgp4.api import Satrec
    line1, line2 = tle_tuple
    sat = Satrec.twoline2rv(line1, line2)
    e, r, v = sat.sgp4_tsince(0)
    epoch_time = Time(sat.jdsatepoch, format='jd')
    epoch_time += sat.jdsatepochF * u.d
    # Convert from km to m
    return np.array(r) * 1e3, np.array(v) * 1e3, epoch_time.gps


def parse_tle(tle):
    """
    Parse a TLE returning Kozai mean orbital elements.

    Parameters
    ----------
    tle : 2-tuple of str
        Line1 and Line2 of TLE as strings

    Returns
    -------
    a : float
        Kozai mean semi-major axis in meters
    e : float
        Kozai mean eccentricity
    i : float
        Kozai mean inclination in radians
    pa : float
        Kozai mean periapsis argument in radians
    raan : float
        Kozai mean right ascension of the ascending node in radians
    trueAnomaly : float
        Kozai mean true anomaly in radians
    t : float
        GPS seconds; i.e., seconds since 1980-01-06 00:00:00 UTC

    Notes
    -----
    Dynamic TLE terms, including the drag coefficient and ballistic coefficient,
    are ignored in this function.
    """
    from sgp4.ext import days2mdhms, invjday, jday
    from .orbit import (
        _ellipticalEccentricToTrueAnomaly,
        _ellipticalMeanToEccentricAnomaly
    )
    # just grabbing the bits we care about for now.
    line1, line2 = tle
    assert line1[0] == '1'
    assert line2[0] == '2'
    year = int(line1[18:20])
    epochDay = float(line1[20:32])
    i = float(line2[8:16])
    raan = float(line2[17:25])
    e = float(line2[26:33]) / 1e7
    pa = float(line2[34:42])
    meanAnomaly = float(line2[43:51])
    meanMotion = float(line2[52:63])

    # Adjust units
    if year >= 57:
        year += 1900
    else:
        year += 2000
    mon, day, hr, minute, sec = days2mdhms(year, epochDay)
    jdsatepoch = jday(year, mon, day, hr, minute, sec)
    sec_whole, sec_frac = divmod(sec, 1.0)
    try:
        epoch = datetime.datetime(
            year, mon, day, hr, minute, int(sec_whole), int(sec_frac * 1e6 // 1.0)
        )
    except ValueError:
        year, mon, day, hr, minute, sec = invjday(jdsatepoch)
        epoch = datetime.datetime(
            year, mon, day, hr, minute, int(sec_whole), int(sec_frac * 1e6 // 1.0)
        )

    # Assuming decimal year UTC?
    i = np.deg2rad(i)
    raan = np.deg2rad(raan)
    pa = np.deg2rad(pa)
    meanAnomaly = np.deg2rad(meanAnomaly)
    epoch = Time(epoch)

    period = 86400. / meanMotion
    a = (period**2 * EARTH_MU / (2 * np.pi)**2)**(1. / 3)
    trueAnomaly = _ellipticalEccentricToTrueAnomaly(
        _ellipticalMeanToEccentricAnomaly(
            meanAnomaly, e
        ),
        e
    )
    return a, e, i, pa, raan, trueAnomaly, epoch.gps


def make_tle(a, e, i, pa, raan, trueAnomaly, t):
    """
    Create a TLE from Kozai mean orbital elements

    Parameters
    ----------
    a : float
        Kozai mean semi-major axis in meters
    e : float
        Kozai mean eccentricity
    i : float
        Kozai mean inclination in radians
    pa : float
        Kozai mean periapsis argument in radians
    raan : float
        Kozai mean right ascension of the ascending node in radians
    trueAnomaly : float
        Kozai mean true anomaly in radians
    t : float or astropy.time.Time
        If float, then should correspond to GPS seconds; i.e., seconds since
        1980-01-06 00:00:00 UTC

    Notes
    -----
    Dynamic TLE terms, including the drag coefficient and ballistic coefficient,
    are ignored in this function.
    """
    from .orbit import (
        _ellipticalEccentricToMeanAnomaly,
        _ellipticalTrueToEccentricAnomaly
    )

    line1 = "1 99999U 99999ZZZ "
    line2 = "2 99999 "

    if not isinstance(t, Time):
        t = Time(t, format='gps')
    year, day, hour, min, sec = t.utc.yday.split(':')

    day = float(day) + (float(hour) + (float(min) + float(sec) / 60) / 60) / 24

    line1 += "{:02d}".format(int(year) % 100)
    line1 += "{:012.8f}".format(day)
    line1 += " +.00000000 +00000-0  99999-0 0 0000"

    def checksum(s):
        check = 0
        for c in s:
            if c == "-":
                check += 1
            else:
                try:
                    d = int(c)
                except Exception:
                    continue
                check += d
        return str(check % 10)
    line1 += checksum(line1)

    line2 += "{:8.4f} ".format(np.rad2deg(i % np.pi))
    line2 += "{:8.4f} ".format(np.rad2deg(raan % (2 * np.pi)))
    line2 += "{:07d} ".format(int(e * 1e7))
    line2 += "{:8.4f} ".format(np.rad2deg(pa % (2 * np.pi)))
    meanAnomaly = _ellipticalEccentricToMeanAnomaly(
        _ellipticalTrueToEccentricAnomaly(
            trueAnomaly % (2 * np.pi), e
        ),
        e
    )

    line2 += "{:8.4f} ".format(np.rad2deg(meanAnomaly % (2 * np.pi)))
    meanMotion = np.sqrt(EARTH_MU / np.abs(a**3))  # rad/s
    meanMotion *= 86400 / (2 * np.pi)
    line2 += "{:11.8f} ".format(meanMotion)
    line2 += "   0"
    line2 += checksum(line2)

    return line1, line2


def get_tel_pos_itrf_to_gcrs(time, tel_label="511"):
    """
    Convert telescope locations in ITRF (i.e., fixed to the earth) to GCRS
    (i.e., geocentric celestial frame)

    :param time: Time at which to evaluate the position
    :type time: astropy.time.Time
    """
    tp = telescope_catalog[tel_label]
    # Astropy:
    rITRF = ac.CartesianRepresentation(tp["x"], tp["y"], tp["z"])
    itrs = ac.ITRS(rITRF, obstime=time)
    gcrs = itrs.transform_to(ac.GCRS(obstime=time))
    return gcrs.cartesian.xyz

# Additional reference:
# https://fas.org/spp/military/program/track/space_pulvermacher.pdf
#
# Right ascension:
# The angle between the vernal equinox
# and the projection of the radius vector on the equatorial plane, regarded as
# positive when measured eastward from the vernal equinox. [AFSPCI 60-102]
# Measurement units must be degrees.
#
# Declination:
# The angle between the celestial equator and a radius vector, regarded as
# positive when measured north from the celestial equator.
# [AFSPCI 60-102] Measurement units must be degrees.
#
# For type 9:
# SensorLocation described as an E,F,G (Earth-Fixed Geocentric) position vector
# of mobile sensor measured in meters
# Values are epoched to the observation's epoch.
#
# Reference: [AFSPCI 60-102] Air Force Space Command (AFSPC) Astrodynamic
# Standards, AFSPC Instruction 60-102, 11 March 1996.

# =============================================================================
# JEM implementation
# =============================================================================


b3dtype = np.dtype([
    ('secClass', 'U1'),
    ('satID', np.int32),
    ('sensID', np.int32),
    ('year', np.int16),
    ('day', np.int16),
    ('hour', np.int8),
    ('minute', np.int8),
    ('second', np.float64),
    ('polarAngle', np.float64),
    ('azimuthAngle', np.float64),
    ('range', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('slantRangeRate', np.float64),
    ('type', np.int8),
    ('equinoxType', np.int8)
])


def parseB3Line(line):
    """
    Read one line of a B3 file and parse into distinct catalog components
    """
    try:
        type_ = int(line[74])
    except ValueError:
        type_ = -999

    secClass = str(line[0])
    satID = int(line[1:6])
    sensID = int(line[6:9])
    year = int(line[9:11])
    year = 1900 + year if year > 50 else 2000 + year
    day = int(line[11:14])
    hour = int(line[14:16])
    minute = int(line[16:18])
    millisec = int(line[18:23])
    sec = millisec / 1000.0
    # Note, I'm assuming that "-51280" is -5.128 degrees, and that
    # the overpunching only applies when dec <= -10 degrees.
    polarAngleStr = line[23:29]
    for ch, i in zip("JKLMNOPQR", range(1, 10)):
        polarAngleStr = polarAngleStr.replace(ch, "-{}".format(i))
    polarAngle = float(polarAngleStr) / 1e4
    if type_ in [5, 9]:
        hh = float(line[30:32])
        mm = float(line[32:34])
        sss = float(line[34:37]) / 10
        azimuthAngle = 15 * (hh + (mm + sss / 60) / 60)
    else:
        azimuthAngle = float(line[30:37]) / 1e4
    try:
        range_ = float(line[38:45]) / 1e5
    except ValueError:
        range_ = np.nan
    else:
        try:
            rangeExp = int(line[45])
        except ValueError:
            rangeExp = 0
        range_ = range_ * 10**rangeExp
    if type_ in [8, 9]:
        slantRangeRate = np.nan
        try:
            x = float(line[46:55])
            y = float(line[55:64])
            z = float(line[65:73])
        except ValueError:
            x = y = z = np.nan
    else:
        x = y = z = np.nan
        try:
            slantRangeRate = float(line[47:54])/1e5
        except ValueError:
            slantRangeRate = np.nan
    if type_ in [5, 9]:
        equinoxType = int(line[75])
    else:
        equinoxType = -999
    return np.array([(
        secClass,
        satID,
        sensID,
        year, day, hour, minute, sec,
        polarAngle, azimuthAngle,
        range_,
        x, y, z,
        slantRangeRate,
        type_,
        equinoxType
    )], dtype=b3dtype)


def parseB3(filename):
    """
    Load data from a B3 observation file

    :param filename: Name of the B3 obs file to load
    :type filename: string

    :return: A catalog of observations
    :rtype: astropy.table.Table

    Note that angles and positions are output in TEME frame.
    """
    from astropy.table import Table
    from astropy.time import Time
    from datetime import datetime, timedelta
    data = []
    for line in open(filename, 'r'):
        data.append(parseB3Line(line))
    data = Table(np.hstack(data))
    dts = [datetime.strptime("{} {} {} {}".format(
        d['year'], d['day'], d['hour'], d['minute']),
                                      "%Y %j %H %M")
           + timedelta(0, seconds=d['second'])
           for d in data]
    data['date'] = Time(dts)
    data['mjd'] = data['date'].mjd
    data['gps'] = data['date'].gps

    return data



# =============================================================================
# MDS implementation
# =============================================================================
b3obs_types = [
    "Range rate only",  # 0
    "Azimuth & elevation",  # 1
    "Range, azimuth, & elevation",  # 2
    "Range, azimuth, elevation, & range rate",  # 3
    # 4
    "Range, azimuth, elevation, & range rate (extra measurements for azimuth rate, elevation rate, etc are ignored)",
    "Right Ascension & Declination",  # 5
    "Range only",  # 6
    "UNDEFINED",  # 7
    "Space-based azimuth, elevation, sometimes range and EFG position of the sensor",  # 8
    "Space-based right ascension, declination, sometimes range and EFG position of the sensor",  # 9
]


overpunched = ['J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']


def parse_overpunched(line):
    line_vals = [x for x in line]
    if line_vals[0] in overpunched:
        val = -(overpunched.index(line_vals[0]) + 1)
        line_vals[0] = str(val)
        line = ''.join(line_vals)
    return line


def b3obs2pos(b3line):
    """
    Return an SGP4 Satellite imported from B3OBS data.

    Intended to mimic the sgp4 twoline2rv function

    Data format reference:
    http://help.agi.com/odtk/index.html?page=source%2Fod%2FODObjectsMeasurementFormatsandMeasurementTypes.htm
    """
    from astropy.coordinates import Angle

    if (len(b3line) >= 76):
        security_classification = str(b3line[0])
        satellite_id = int(b3line[1:6])  # corresponds to SSC number
        sensor_id = b3line[6:9]
        two_digit_year = int(b3line[9:11])
        day_of_year = float(b3line[11:14])
        # hms = float(b3line[14:20] + '.' + b3line[20:23])
        hours = float(b3line[14:16])
        minutes = float(b3line[16:18])
        seconds = float(b3line[18:20] + '.' + b3line[20:23])
        #
        obs_type = int(b3line[74])
        # print("obs_type:", obs_type)
        #
        el_or_dec = float(parse_overpunched(b3line[23:25]) +
                          '.' + b3line[25:29])  # degrees
        # Column 30 is blank
        if obs_type == 5 or obs_type == 9: # Have RA
            az_or_ra = Angle(b3line[30:32] + 'h' + b3line[32:34] + 'm' +
                             b3line[34:36] + '.' + b3line[36] + 's')
        else: # Have azimuth
            az_or_ra = float(b3line[30:33] +
                             '.' + b3line[33:37])  # degrees
        # Column 38 is blank
        range_in_km = np.nan
        if not b3line[38:45].isspace() and not b3line[45].isspace():
            range_in_km = float(b3line[38:40] + '.' + b3line[40:45])
            range_exp = float(b3line[45])
            range_in_km *= 10.**range_exp

        obs_type_data = np.nan
        if obs_type < 8:
            if not b3line[46:73].isspace():
                slant_range = float(b3line[47:49] + '.' + b3line[49:54])
        else:  # obs_type == 8 or 9
            x_sat = b3line[46:55]
            y_sat = b3line[55:64]
            z_sat = b3line[64:73]
        # Column 74 is blank
        equinox = int(b3line[75])
    else:
        raise ValueError("B3OBS format error")

    epochdays = day_of_year

    if two_digit_year <= 50:
        year = two_digit_year + 2000
    else:
        year = two_digit_year + 1900

    epoch = datetime.datetime(year, 1, 1) + \
        datetime.timedelta(days=day_of_year-1, hours=hours,
                           minutes=minutes, seconds=seconds)

    ra_deg = np.nan
    dec_deg = np.nan
    if obs_type == 5:
        # Convert from hms to deg
        ra_deg = az_or_ra.deg
        dec_deg = el_or_dec  # already in degrees

    try:
        tel = telescope_catalog[sensor_id]
        x = tel["x"].to(u.km).value
        y = tel["y"].to(u.km).value
        z = tel["z"].to(u.km).value
        tel_pos = np.array([x, y, z]) * u.km
    except KeyError:
        tel_pos = None

    return {"satnum": satellite_id,
            "sensnum": int(sensor_id),
            "epoch": epoch,
            "ra": ra_deg,
            "dec": dec_deg,
            "range": range_in_km,
            "tel_pos": tel_pos
            }


def load_b3obs_file(file_name):
    """
    Convenience function to load all entries in a B3OBS file
    """
    f = open(file_name, 'r')
    dat = f.readlines()
    pos = [b3obs2pos(dat[iline][0:76]) for iline in range(len(dat))]
    f.close()

    catalog = {
        "ra": np.array([d["ra"] for d in pos]),
        "dec": np.array([d["dec"] for d in pos]),
        "time": np.array([d["epoch"] for d in pos]),
        "sensnum": [d["sensnum"] for d in pos],
        "satnum": [d["satnum"] for d in pos],
        "tel_pos": np.array([d["tel_pos"] for d in pos])
    }
    return catalog

######################################################################
# MPI
######################################################################


def mpi_scatter(scatter_array):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD  # Defines the default communicator
    num_procs = comm.Get_size()  # Stores the number of processes in size.
    rank = comm.Get_rank()  # Stores the rank (pid) of the current process
    # stat = MPI.Status()
    print(f'Number of procs: {num_procs}, rank: {rank}')
    remainder = np.size(scatter_array) % num_procs
    base_load = np.size(scatter_array) // num_procs
    if rank == 0:
        print('All processors will process at least {0} simulations.'.format(
            base_load))
        print('{0} processors will process an additional simulations'.format(
            remainder))
    load_list = np.concatenate((np.ones(remainder) * (base_load + 1),
                                np.ones(num_procs - remainder) * base_load))
    if rank == 0:
        print('load_list={0}'.format(load_list))
    if rank < remainder:
        scatter_array_local = np.zeros(base_load + 1, dtype=np.int64)
    else:
        scatter_array_local = np.zeros(base_load, dtype=np.int64)
    disp = np.zeros(num_procs)
    for i in range(np.size(load_list)):
        if i == 0:
            disp[i] = 0
        else:
            disp[i] = disp[i - 1] + load_list[i - 1]
    comm.Scatterv([scatter_array, load_list, disp, MPI.DOUBLE], scatter_array_local)
    print(f"Process {rank} received the scattered arrays: {scatter_array_local}")
    return scatter_array_local, rank


def mpi_scatter_exclude_rank_0(scatter_array):
    # Function is for rank 0 to be used as a saving processor - all other processors will complete tasks.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    rank = comm.Get_rank()
    print(f'Number of procs: {num_procs}, rank: {rank}')

    num_workers = num_procs - 1
    remainder = np.size(scatter_array) % num_workers
    base_load = np.size(scatter_array) // num_workers

    if rank == 0:
        print(f'All processors will process at least {base_load} simulations.')
        print(f'{remainder} processors will process an additional simulation.')

    load_list = np.concatenate((np.zeros(1), np.ones(remainder) * (base_load + 1),
                                np.ones(num_workers - remainder) * base_load))

    if rank == 0:
        print(f'load_list={load_list}')

    scatter_array_local = np.zeros(int(load_list[rank]), dtype=np.int64)

    disp = np.zeros(num_procs)
    for i in range(1, num_procs):
        disp[i] = disp[i - 1] + load_list[i - 1]

    if rank == 0:
        dummy_recvbuf = np.zeros(1, dtype=np.int64)
        comm.Scatterv([scatter_array, load_list, disp, MPI.INT64_T], dummy_recvbuf)
    else:
        comm.Scatterv([scatter_array, load_list, disp, MPI.INT64_T], scatter_array_local)
        print(f"Process {rank} received the {len(scatter_array_local)} element scattered array: {scatter_array_local}")

    return scatter_array_local, rank

# =============================================================================
# File Handling Functions
# =============================================================================
import os


def exists(pathname):
    if os.path.isdir(pathname):
        exists = True
    elif os.path.isfile(pathname):
        exists = True
    else:
        exists = False
    return exists


def mkdir(pathname):
    if not exists(pathname):
        os.makedirs(pathname)
        print("Directory '%s' created" % pathname)        
    return


def rmdir(source_):
    if not exists(source_):
        print(f'{source_}, does not exist, no delete.')
    else:
        import shutil
        print(f'Deleted {source_}') 
        shutil.rmtree(source_)
    return


def rmfile(pathname):
    if exists(pathname):
        os.remove(pathname)
        print("File: '%s' deleted." % pathname)        
    return


def listdir(dir_path='*', files_only=False, exclude=None):
    from glob import glob
    expanded_paths = glob(dir_path)

    if files_only:
        files = [f for f in expanded_paths if os.path.isfile(f)]
        print(f'{len(files)} files in {dir_path}')
    else:
        files = expanded_paths
        print(f'{len(files)} files in {dir_path}')

    if exclude:
        new_files = [file for file in files if exclude not in os.path.basename(file)]
        files = new_files
    return sorted(files)


def get_memory_usage():
    import os
    import psutil

    print(f"Memory used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.2f} GB")
######################################################################
# Load and Save Functions
######################################################################
######################################################################
# Pickles
######################################################################


def psave(filename_, data_):
    from six.moves import cPickle as pickle  # for performance
    with open(filename_, 'wb') as f:
        pickle.dump(data_, f)
    f.close()
    return


def pload(filename_):
    from six.moves import cPickle as pickle  # for performance
    try:
        # print('Openning: ' + current_filename)
        with open(filename_, 'rb') as f:
            data = pickle.load(f)
        f.close()
    except (EOFError, FileNotFoundError, OSError, pickle.UnpicklingError) as err:
        print(f'{err} - current_filename')
        return []
    return data


def merge_dicts(file_names, save_path):
    number_of_files = len(file_names); master_dict = {}
    for count, file in enumerate(file_names):
        print(f'Merging dict: {count+1} of {number_of_files}, name: {file}, num of master keys: {len(master_dict.keys())}, num of new keys: {len(master_dict.keys())}')
        master_dict.update(pload(file))
    print('Beginning final save.')
    psave(save_path, master_dict)
    return
######################################################################
# Sliceable Numpys save and load
######################################################################


def npsave(filename_, data_):
    try:
        with open(filename_, 'wb') as f:
            np.save(filename_, data_, allow_pickle=True)
        f.close()
    except (EOFError, FileNotFoundError, OSError) as err:
        print(f'{err} - saving')
        return


def npload(filename_):
    try:
        with open(filename_, 'rb') as f:
            data = np.load(filename_, allow_pickle=True)
        f.close()
    except (EOFError, FileNotFoundError, OSError) as err:
        print(f'{err} - loading')
        return []
    return data


import h5py
######################################################################
# HDF5 py files h5py
######################################################################


def h5append(filename, pathname, append_data):
    """
    Append data to key in HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the key in the HDF5 file.
        append_data (any): The data to be appended.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "r+") as f:
            if pathname in f:
                path_data_old = np.array(f.get(pathname))
                new_data = np.append(path_data_old, np.array(append_data))
                f[pathname] = new_data
            else:
                f.create_dataset(pathname, data=np.array(append_data), maxshape=None)
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except (ValueError, KeyError) as err:
        print(f"Error: {err}")


def h5overwrite(filename, pathname, new_data):
    """
    Overwrite key in HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the key in the HDF5 file.
        new_data (any): The data to be overwritten.

    Returns:
        None
    """
    try:
        with h5py.File(filename, "a") as f:
            f.create_dataset(pathname, data=new_data, maxshape=None)
        f.close()
    except (FileNotFoundError, ValueError, KeyError):
        try:
            with h5py.File(filename, 'r+') as f:
                del f[pathname]
            f.close()
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'Error: {err}')
        try:
            with h5py.File(filename, "a") as f:
                f.create_dataset(pathname, data=new_data, maxshape=None)
            f.close()
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f'File: {filename}{pathname}, Error: {err}')


def h5save(filename, pathname, data):
    """
    Save data to HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the data in the HDF5 file.
        data (any): The data to be saved.

    Returns:
        None
    """
    with h5py.File(filename, "a") as f:
        try:
            f.create_dataset(pathname, data=data, maxshape=None)
        except ValueError as err:
            print(f"Did not save, key: {pathname} exists in file: {filename}. {err}")


def h5load(filename, pathname):
    """
    Load data from HDF5 file.

    Args:
        filename_ (str): The filename of the HDF5 file.
        pathname_ (str): The path to the data in the HDF5 file.

    Returns:
        The data loaded from the HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        data = np.array(f.get(pathname))
    f.close()
    return data


def h5loadall(filename_):
    """
    Load all data from HDF5 file.

    Args:
        filename_ (str): The filename of the HDF5 file.

    Returns:
        A dictionary of data loaded from the HDF5 file.
    """
    with h5py.File(filename_, "r") as f:
        # List all groups
        keys = list(f.keys())
        return_data = {key: np.array(f.get(key)) for key in keys}

    return return_data, keys


def h5keys(filename):
    """
    List all groups in HDF5 file.

    Args:
        filename_ (str): The filename of the HDF5 file.

    Returns:
        A list of group keys in the HDF5 file.
    """
    with h5py.File(filename, "r") as f:
        # List all groups
        group_keys = list(f.keys())
    return group_keys


def h5_key_exists(filename, key):
    """
    Checks if a key exists in an HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        key (str): The key to check.

    Returns:
        True if the key exists, False otherwise.
    """

    try:
        with h5py.File(filename, 'r') as f:
            return str(key) in f
    except IOError:
        return False


import pandas as pd
######################################################################
# CSV
######################################################################


def makedf(df):
    if isinstance(df, (list, dict)):
        return pd.DataFrame.from_dict(df)
    else:
        return df


def header_csv(file_name, sep=None):
    """
    Get the header of a CSV file.

    Args:
        file_name (str): The filename of the CSV file.
        sep (str) optional: The delimiter used in the CSV file.

    Returns:
        A list of the header fields.
    """
    import csv
    if sep is None:
        sep = guess_csv_delimiter(file_name)  # Guess the delimiter
    with open(file_name, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=sep)
        fieldnames = reader.fieldnames
    return fieldnames


def read_csv(file_name, sep=None, dtypes=None, col=False, to_np=False, drop_nan=False, skiprows=[]):
    """
    Read a CSV file with options.

    Parameters
    ----------
    file_name : str
        The path to the CSV file.
    sep : str, optional
        The delimiter used in the CSV file. If None, delimiter will be guessed.
    dtypes : dict, optional
        Dictionary specifying data types for columns.
    col : bool or list of str, optional
        Specify columns to read. If False, read all columns.
    to_np : bool, optional
        Convert the loaded data to a NumPy array.
    drop_nan : bool, optional
        Drop rows with missing values (NaNs) from the loaded DataFrame.
    skiprows : list of int, optional
        Rows to skip while reading the CSV file.

    Returns
    -------
    DataFrame or NumPy array
        The loaded data in either a DataFrame or a NumPy array format.
    """
    if sep is None:
        sep = guess_csv_delimiter(file_name)  # Guess the delimiter

    if col is False:
        try:
            df = pd.read_csv(file_name, sep=sep, on_bad_lines='skip', skiprows=skiprows, dtype=dtypes)
        except TypeError:
            df = pd.read_csv(file_name, sep=sep, skiprows=skiprows, dtype=object)
    else:
        try:
            if not isinstance(col, list):
                col = [col]
            df = pd.read_csv(file_name, sep=sep, usecols=col, on_bad_lines='skip', skiprows=skiprows, dtype=dtypes)
        except TypeError:
            df = pd.read_csv(file_name, sep=sep, usecols=col, skiprows=skiprows, dtype=object)

    if drop_nan:
        df = df.dropna()

    if to_np:
        return np.squeeze(df.to_numpy())
    else:
        return df


def guess_csv_delimiter(file_name):
    import csv
    """
    Guess the delimiter used in a CSV file.

    Args:
        file_name (str): The path to the CSV file.

    Returns:
        str: Guessed delimiter (one of ',', '\t', ';')
    """
    with open(file_name, 'r', newline='') as file:
        sample = file.read(4096)  # Read a sample of the file's contents
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter


def save_csv(file_name, df, sep='\t', dtypes=None):
    """
    Save a Pandas DataFrame to a CSV file.

    Args:
        file_name (str): The path to the CSV file.
        df (DataFrame): The Pandas DataFrame to save.
        sep (str): The delimiter used in the CSV file.
        dtypes (dict): A dictionary specifying data types for columns.

    Returns:
        None
    """
    df = makedf(df)

    if dtypes:
        df = df.astype(dtypes)

    df.to_csv(file_name, index=False, sep=sep)
    print(f'Saved {file_name} successfully.')
    return


def append_csv(file_names, save_path='combined_data.csv', sep=None, dtypes=None, progress=None):
    """
    Appends multiple CSV files into a single CSV file.

    Args:
        file_names (list): A list of CSV file names.
        save_path (str): The path to the output CSV file. If not specified, the output will be saved to the current working directory.
        sep (str): The delimiter used in the CSV files.
        dtypes (dict): A dictionary specifying data types for columns.

    Returns:
        None
    """

    error_files = []
    dataframes = []
    for i, file in enumerate(file_names):
        try:
            if sep is None:
                sep = guess_csv_delimiter(file)  # Guess the delimiter
            df = pd.read_csv(file, sep=sep)
            dataframes.append(df)
            if progress is not None:
                get_memory_usage()
                print(f"Appended {i+1} of {len(file_names)}.")
        except FileNotFoundError:
            error_files.append(file)

    combined_df = pd.concat(dataframes, ignore_index=True)

    if dtypes:
        combined_df = combined_df.astype(dtypes)

    if save_path:
        combined_df.to_csv(save_path, sep=sep, index=False)
    else:
        combined_df.to_csv('combined_data.csv', sep=sep, index=False)

    print(f'The final dataframe has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.')
    if error_files:
        print(f'The following files could not be found: {error_files}')


def append_csv_on_disk(csv_files, output_file):
    import csv
    # Assumes each file has the same delimiters
    delimiter = guess_csv_delimiter(csv_files[0])
    # Open the output file for writing
    with open(output_file, 'w', newline='') as outfile:
        # Initialize the CSV writer
        writer = csv.writer(outfile, delimiter=delimiter)

        # Write the header row from the first CSV file
        with open(csv_files[0], 'r', newline='') as first_file:
            reader = csv.reader(first_file, delimiter=delimiter)
            header = next(reader)
            writer.writerow(header)

            # Write the data rows from the first CSV file
            for row in reader:
                writer.writerow(row)

        # Write the data rows from the remaining CSV files
        for file in csv_files[1:]:
            with open(file, 'r', newline='') as infile:
                reader = csv.reader(infile, delimiter=delimiter)
                next(reader)  # Skip the header row
                for row in reader:
                    writer.writerow(row)
    print(f'Completed appending of: {output_file}.')


def save_csv_line(file_name, df, sep='\t', dtypes=None):
    """
    Save a Pandas DataFrame to a CSV file, appending the DataFrame to the file if it exists.

    Args:
        file_name (str): The path to the CSV file.
        df (DataFrame): The Pandas DataFrame to save.
        sep (str): The delimiter used in the CSV file.

    Returns:
        None
    """
    df = makedf(df)
    if dtypes:
        df = df.astype(dtypes)
    if exists(file_name):
        df.to_csv(file_name, mode='a', index=False, header=False, sep=sep)
    else:
        save_csv(file_name, df, sep=sep)
    return


# Create a lock to synchronize access to the file
import threading
file_lock = threading.Lock()


def exists_in_csv(csv_file, column_name, number, sep='\t'):
    import csv
    with file_lock:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=sep)
                for row in reader:
                    if row[column_name] == str(number):
                        return True
        except IOError:
            return False
    return False


def pd_flatten(data, factor=1):
    tmp = []
    for x in data:
        try:
            tmp.extend(x[1:-1].split(','))
        except TypeError:
            tmp.append(x)
    return [float(x) / factor for x in tmp]

#
# TURN AN ARRAY SAVED AS A STRING BACK INTO AN ARRAY


def str_to_array(s):
    s = s.replace('[', '').replace(']', '')  # Remove square brackets
    return np.array([float(x) for x in s.split(',')])


def pdstr_to_arrays(df):
    return df.apply(str_to_array).to_numpy()

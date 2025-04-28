"""
Collection of functions to read from and write to various file formats.
"""

import datetime
import numpy as np
import os
import csv
import pandas as pd
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
    """
    Parse and adjust a string containing overpunched numeric values.

    This function processes a given string to handle overpunched numeric values, 
    which are a legacy encoding method for representing negative numbers in specific 
    positions. If the first character of the string matches an overpunched value, 
    it is replaced with its corresponding negative numeric value.

    Overpunched values are defined as:
        ['J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']

    The index of the overpunched character determines the numeric value, starting 
    from -1 for 'J', -2 for 'K', and so on.

    Args:
        line (str): The input string to be parsed and adjusted.

    Returns:
        str: The modified string where overpunched values have been replaced 
             with their numeric equivalents.
    """
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

# =============================================================================
# File Handling Functions
# =============================================================================
def file_exists_extension_agnostic(filename):
    """
    Check if a file with the given name and any extension exists.

    Parameters:
    ----------
    filename : str
        The name of the file to check, without extension.

    Returns:
    -------
    bool
        True if a file with the given name and any extension exists, False otherwise.
    """
    from glob import glob
    name, _ = os.path.splitext(filename)
    return bool(glob(f"{name}.*"))


def exists(pathname):
    """
    Check if a file or directory exists.

    Parameters:
    ----------
    pathname : str
        The path to the file or directory.

    Returns:
    -------
    bool
        True if the path exists as either a file or a directory, False otherwise.
    """
    if os.path.isdir(pathname) or os.path.isfile(pathname):
        return True
    else:
        return False


def mkdir(pathname):
    """
    Creates a directory if it does not exist.

    Parameters:
    ----------
    pathname : str
        The path to the directory to be created.
    """
    if not exists(pathname):
        os.makedirs(pathname)
        print("Directory '%s' created" % pathname)        
    return


def rmdir(source_):
    """
    Deletes a directory and its contents if it exists.

    Parameters:
    ----------
    source_ : str
        The path to the directory to be deleted.
    """
    if not exists(source_):
        print(f'{source_}, does not exist, no delete.')
    else:
        import shutil
        print(f'Deleted {source_}') 
        shutil.rmtree(source_)
    return


def rmfile(pathname):
    """
    Deletes a file if it exists.

    Parameters:
    ----------
    pathname : str
        The path to the file to be deleted.
    """
    if exists(pathname):
        os.remove(pathname)
        print("File: '%s' deleted." % pathname)        
    return


def _sortbynum(files, index=0):
    """
    Sorts a list of file paths based on numeric values in the filenames.

    This function assumes that each filename contains at least one numeric value
    and sorts the files based on the first numeric value found in the filename.

    Parameters:
    ----------
    files : list
        List of file paths to be sorted. Each file path can be a full path or just a filename.
    index: int
        Index of the number in the string do you want to sort on.

    Returns:
    -------
    list
        List of file paths sorted by numeric values in their filenames.

    Notes:
    -----
    - This function extracts the first numeric value it encounters in each filename.
    - If no numeric value is found in a filename, the function may raise an error.
    - The numeric value can appear anywhere in the filename.
    - The function does not handle cases where filenames have different directory prefixes.
    
    Raises:
    ------
    ValueError:
        If a filename does not contain any numeric value.

    Examples:
    --------
    >>> _sortbynum(['file2.txt', 'file10.txt', 'file1.txt'])
    ['file1.txt', 'file2.txt', 'file10.txt']

    >>> _sortbynum(['/path/to/file2.txt', '/path/to/file10.txt', '/path/to/file1.txt'])
    ['/path/to/file1.txt', '/path/to/file2.txt', '/path/to/file10.txt']
    """
    import re
    if len(files[0].split('/')) > 1:
        files_shortened = []
        file_prefix = '/'.join(files[0].split('/')[:-1])
        for file in files:
            files_shortened.append(file.split('/')[-1])
        files_sorted = sorted(files_shortened, key=lambda x: float(re.findall(r"(\d+)", x)[index]))
        sorted_files = []
        for file in files_sorted:
            sorted_files.append(f'{file_prefix}/{file}')
    else:
        sorted_files = sorted(files, key=lambda x: float(re.findall(r"(\d+)", x)[index]))
    return sorted_files


def listdir(dir_path='*', files_only=False, exclude=None, sorted=False, index=0):
    """
    Lists files and directories in a specified path with optional filtering and sorting.

    Parameters:
    ----------
    dir_path : str, default='*'
        The directory path or pattern to match files and directories.
    files_only : bool, default=False
        If True, only returns files, excluding directories.
    exclude : str or None, optional
        If specified, excludes files and directories whose base name contains this string.
    sorted : bool, default=False
        If True, sorts the resulting list by numeric values in filenames.
    index : int, default=0
        sorted required to be true. Index of the digit used for sorting.

    Returns:
    -------
    list
        A list of file or directory paths based on the specified filters and sorting.
    """
    from glob import glob
    if '*' not in dir_path:
        dir_path = os.path.join(dir_path, '*')
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
    if sorted:
        return _sortbynum(files, index=index)
    else:
        return files


def get_memory_usage():
    """
    Print the memory usage of the current process.

    This function retrieves the memory usage of the current Python process 
    using the `psutil` library and prints it in gigabytes (GB). The memory 
    usage is calculated based on the resident set size (RSS), which represents 
    the portion of memory occupied by the process in RAM.

    Args:
        None

    Returns:
        None: The function does not return a value. It prints the memory usage directly.
    """
    import os
    import psutil

    print(f"Memory used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.2f} GB")

######################################################################
# Load and Save Functions
######################################################################
######################################################################
# Pickles
######################################################################


def save_pickle(filename_, data_):
    """
    Save data to a pickle file.

    This function serializes the given data and saves it to the specified file path 
    using the pickle module. The file is opened in write-binary mode to ensure 
    proper saving of the data.

    Args:
        filename_ (str): The path to the file where the data will be saved.
        data_ (object): The data to be serialized and saved to the pickle file.

    Returns:
        None: The function does not return a value. The data is saved to the specified file.
    """
    from six.moves import cPickle as pickle  # for performance
    with open(filename_, 'wb') as f:
        pickle.dump(data_, f)
    f.close()
    return


def load_pickle(filename_):
    """
    Load data from a pickle file.

    Args:
        filename_ (str): The path to the pickle file to be loaded.

    Returns:
        object: The data loaded from the pickle file. If an error occurs, 
                an empty list is returned.
    """
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
    """
    Merge multiple dictionaries stored in pickle files into a single dictionary and save the result.

    Args:
        file_names (list of str): A list of file paths to pickle files containing dictionaries to merge.
        save_path (str): The file path where the merged dictionary will be saved as a pickle file.

    Returns:
        None: The function does not return a value. The merged dictionary is saved to `save_path`.
    """
    number_of_files = len(file_names); master_dict = {}
    for count, file in enumerate(file_names):
        print(f'Merging dict: {count+1} of {number_of_files}, name: {file}, num of master keys: {len(master_dict.keys())}, num of new keys: {len(master_dict.keys())}')
        master_dict.update(load_pickle(file))
    print('Beginning final save.')
    save_pickle(save_path, master_dict)
    return


######################################################################
# Sliceable Numpys save and load
######################################################################


def save_np(filename_, data_):
    """
    Save a NumPy array to a binary file.

    This function saves a NumPy array to a file in .npy format. If the file cannot be created or written to, it handles common exceptions and prints an error message.

    Parameters:
    ----------
    filename_ : str
        The path to the file where the NumPy array will be saved.
    data_ : numpy.ndarray
        The NumPy array to be saved.

    Returns:
    -------
    None
        The function does not return any value. It handles exceptions internally and prints error messages if any issues occur.

    Examples:
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> save_np('array.npy', arr)
    """
    try:
        with open(filename_, 'wb') as f:
            np.save(filename_, data_, allow_pickle=True)
        f.close()
    except (EOFError, FileNotFoundError, OSError) as err:
        print(f'{err} - saving')
        return


def load_np(filename_):
    """
    Load a NumPy array from a binary file.

    This function loads a NumPy array from a file in .npy format. If the file cannot be read, it handles common exceptions and prints an error message. If loading fails, it returns an empty list.

    Parameters:
    ----------
    filename_ : str
        The path to the file from which the NumPy array will be loaded.

    Returns:
    -------
    numpy.ndarray or list
        The loaded NumPy array. If an error occurs during loading, returns an empty list.

    Examples:
    --------
    >>> arr = load_np('array.npy')
    >>> print(arr)
    [1 2 3 4 5]
    """
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


def append_h5(filename, pathname, append_data):
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
        with h5py.File(filename, "a") as f:
            if pathname in f:
                path_data_old = np.array(f.get(pathname))
                append_data = np.append(path_data_old, np.array(append_data))
                del f[pathname]
            f.create_dataset(pathname, data=np.array(append_data), maxshape=None)
    except FileNotFoundError:
        print(f"File not found: {filename}\nCreating new dataset: {filename}")
        save_h5(filename, pathname, append_data)

    except (ValueError, KeyError) as err:
        print(f"Error: {err}")


def overwrite_h5(filename, pathname, new_data):
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
    except (BlockingIOError, OSError) as err:
        print(f"\n{err}\nPath: {pathname}\nFile: {filename}\n")
        return None
            

def save_h5(filename, pathname, data):
    """
    Save data to HDF5 file with recursive attempt in case of write errors.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the data in the HDF5 file.
        data (any): The data to be saved.
        max_retries (int): Maximum number of recursive retries in case of write errors.
        retry_delay (tuple): A tuple representing the range of delay (in seconds) between retries.

    Returns:
        None
    """
    try:
        try:
            with h5py.File(filename, "a") as f:
                f.create_dataset(pathname, data=data, maxshape=None)
                f.flush()
            return
        except ValueError as err:
            print(f"Did not save, key: {pathname} exists in file: {filename}. {err}")
            return # If the key already exists, no need to retry
    except (BlockingIOError, OSError) as err:
        print(f"\n{err}\nPath: {pathname}\nFile: {filename}\n")
        return None


def read_h5(filename, pathname):
    """
    Load data from HDF5 file.

    Args:
        filename (str): The filename of the HDF5 file.
        pathname (str): The path to the data in the HDF5 file.

    Returns:
        The data loaded from the HDF5 file.
    """
    try:
        with h5py.File(filename, 'r') as f:
            data = f.get(pathname)
            if data is None:
                return None
            else:
                return np.array(data)
    except (ValueError, KeyError, TypeError):
        return None
    except FileNotFoundError:
        print(f'File not found. {filename}')
        raise
    except (BlockingIOError, OSError) as err:
        print(f"\n{err}\nPath: {pathname}\nFile: {filename}\n")
        raise


def read_h5_all(file_path):
    """
    Read all datasets from an HDF5 file into a dictionary.

    This function recursively traverses an HDF5 file and extracts all datasets into a dictionary. The keys of the dictionary are the paths to the datasets, and the values are the dataset contents. 

    Parameters:
    ----------
    file_path : str
        The path to the HDF5 file from which datasets will be read.

    Returns:
    -------
    dict
        A dictionary where keys are the paths to datasets within the HDF5 file, and values are the contents of these datasets.

    Examples:
    --------
    >>> data = read_h5_all('example.h5')
    >>> print(data.keys())
    dict_keys(['/group1/dataset1', '/group2/dataset2'])
    >>> print(data['/group1/dataset1'])
    [1, 2, 3, 4, 5]
    """
    data_dict = {}

    with h5py.File(file_path, 'r') as file:
        # Recursive function to traverse the HDF5 file and populate the dictionary
        def traverse(group, path=''):
            for key, item in group.items():
                new_path = f"{path}/{key}" if path else key

                if isinstance(item, h5py.Group):
                    traverse(item, path=new_path)
                else:
                    data_dict[new_path] = item[()]

        traverse(file)
    return data_dict


def combine_h5(filename, files, verbose=False, overwrite=False):
    """
    Combine multiple HDF5 files into a single HDF5 file.

    This function reads datasets from a list of HDF5 files and writes them to a specified output HDF5 file. If `overwrite` is `True`, it will remove any existing file at the specified `filename` before combining the files. The `verbose` parameter, if set to `True`, will display progress bars during the process.

    Parameters:
    ----------
    filename : str
        The path to the output HDF5 file where the combined datasets will be stored.
    
    files : list of str
        A list of paths to the HDF5 files to be combined.

    verbose : bool, optional
        If `True`, progress bars will be displayed for the file and key processing. Default is `False`.

    overwrite : bool, optional
        If `True`, any existing file at `filename` will be removed before writing the new combined file. Default is `False`.

    Returns:
    -------
    None
        The function performs file operations and does not return any value.

    Examples:
    --------
    >>> combine_h5('combined.h5', ['file1.h5', 'file2.h5'], verbose=True, overwrite=True)
    """
    if verbose:
        from tqdm import tqdm
        iterable = enumerate(tqdm(files))
    else:
        iterable = enumerate(files)
    if overwrite:
        rmfile(filename)
    for idx, file in iterable:
        if verbose:
            iterable2 = tqdm(h5_keys(file))
        else:
            iterable2 = files
        for key in iterable2:
            try:
                if h5_key_exists(filename, key):
                    continue
                save_h5(filename, key, read_h5(file, key))
            except TypeError as err:
                print(read_h5(file, key))
                print(f'{err}, key: {key}, file: {file}')
    print('Completed HDF5 merge.')


def h5_keys(file_path):
    """
    List all groups in HDF5 file.

    Args:
        file_path (str): The file_path of the HDF5 file.

    Returns:
        A list of group keys in the HDF5 file.
    """
    keys_list = []
    with h5py.File(file_path, 'r') as file:
        # Recursive function to traverse the HDF5 file and collect keys
        def traverse(group, path=''):
            for key, item in group.items():
                new_path = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Group):
                    traverse(item, path=new_path)
                else:
                    keys_list.append(new_path)
        traverse(file)
    return keys_list


def h5_root_keys(file_path):
    """
    Retrieve the keys in the root group of an HDF5 file.

    This function opens an HDF5 file and returns a list of keys (dataset or group names) located in the root group of the file.

    Parameters:
    ----------
    file_path : str
        The path to the HDF5 file from which the root group keys are to be retrieved.

    Returns:
    -------
    list of str
        A list of keys in the root group of the HDF5 file. These keys represent the names of datasets or groups present at the root level of the file.
    """
    with h5py.File(file_path, 'r') as file:
        keys_in_root = list(file.keys())
        # print("Keys in the root group:", keys_in_root)
        return keys_in_root


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


######################################################################
# CSV
######################################################################


def makedf(df):
    """
    Convert an input into a pandas DataFrame.

    This function takes an input which can be a list or a dictionary and converts it into a pandas DataFrame. If the input is already a DataFrame, it returns it unchanged.

    Parameters:
    ----------
    df : list, dict, or pd.DataFrame
        The input data to be converted into a DataFrame. This can be a list or dictionary to be transformed into a DataFrame, or an existing DataFrame which will be returned as is.

    Returns:
    -------
    pd.DataFrame
        A DataFrame created from the input data if the input is a list or dictionary. If the input is already a DataFrame, the original DataFrame is returned unchanged.
    """
    if isinstance(df, (list, dict)):
        return pd.DataFrame.from_dict(df)
    else:
        return df


def read_csv_header(file_name, sep=None):
    """
    Get the header of a CSV file.

    Args:
        file_name (str): The filename of the CSV file.
        sep (str) optional: The delimiter used in the CSV file.

    Returns:
        A list of the header fields.
    """
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


def append_dict_to_csv(file_name, data_dict, delimiter='\t'):
    """
    Append data from a dictionary to a CSV file.

    This function appends rows of data to a CSV file, where each key-value pair in the dictionary represents a column. If the CSV file does not already exist, it creates the file and writes the header row using the dictionary keys.

    Parameters:
    ----------
    file_name : str
        Path to the CSV file where data will be appended.
    data_dict : dict
        Dictionary where keys are column headers and values are lists of data to be written to the CSV file. All lists should be of the same length.
    delimiter : str, optional
        The delimiter used in the CSV file (default is tab `\t`).

    Notes:
    ------
    - The function assumes that all lists in the dictionary `data_dict` have the same length.
    - If the CSV file already exists, only the data rows are appended. If it doesn't exist, a new file is created with the header row based on the dictionary keys.
    - The `delimiter` parameter allows specifying the delimiter used in the CSV file. Common values are `,` for commas and `\t` for tabs.

    Example:
    --------
    >>> data_dict = {
    >>>     'Name': ['Alice', 'Bob', 'Charlie'],
    >>>     'Age': [25, 30, 35],
    >>>     'City': ['New York', 'Los Angeles', 'Chicago']
    >>> }
    >>> append_dict_to_csv('people.csv', data_dict, delimiter=',')
    This will append data to 'people.csv', creating it if it does not exist, with columns 'Name', 'Age', 'City'.

    Dependencies:
    --------------
    - `os.path.exists`: Used to check if the file already exists.
    - `csv`: Standard library module used for reading and writing CSV files.
    """
    # Extract keys and values from the dictionary
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Determine the length of the arrays
    array_length = len(values[0])

    # Determine if file exists
    file_exists = os.path.exists(file_name)

    # Open the CSV file in append mode
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(keys)

        # Write each element from arrays as a new row
        for i in range(array_length):
            row = [values[j][i] for j in range(len(keys))]
            writer.writerow(row)


def guess_csv_delimiter(file_name):
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


def append_csv(file_names, save_path='combined_data.csv', sep=None, dtypes=False, progress=None):
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
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            error_files.append(file)
            print(f"Error processing file {file}: {e}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    if dtypes:
        combined_df = combined_df.astype(dtypes)

    if save_path:
        combined_df.to_csv(save_path, sep=sep, index=False)
    else:
        combined_df.to_csv('combined_data.csv', sep=sep, index=False)

    print(f'The final dataframe has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.')
    if error_files:
        print(f'The following files ERRORED and were not included: {error_files}')
    return


def append_csv_on_disk(csv_files, output_file):
    """
    Append multiple CSV files into a single CSV file.

    This function merges multiple CSV files into one output CSV file. The output file will contain the header row from the first CSV file and data rows from all input CSV files. 

    Parameters:
    ----------
    csv_files : list of str
        List of file paths to the CSV files to be merged. All CSV files should have the same delimiter and structure.
    output_file : str
        Path to the output CSV file where the merged data will be written.

    Notes:
    ------
    - The function assumes all input CSV files have the same delimiter. It determines the delimiter from the first CSV file using the `guess_csv_delimiter` function.
    - Only the header row from the first CSV file is included in the output file. Headers from subsequent files are ignored.
    - This function overwrites the output file if it already exists.

    Example:
    --------
    >>> csv_files = ['file1.csv', 'file2.csv', 'file3.csv']
    >>> output_file = 'merged_output.csv'
    >>> append_csv_on_disk(csv_files, output_file)
    Completed appending of: merged_output.csv.

    Dependencies:
    --------------
    - `guess_csv_delimiter` function: A utility function used to guess the delimiter of the CSV files.
    - `csv` module: Standard library module used for reading and writing CSV files.
    """
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


def save_csv_header(filename, header, delimiter='\t'):
    """
    Saves a header row to a CSV file with a specified delimiter.

    Parameters:
    filename (str): The name of the file where the header will be saved.
    header (list): A list of strings representing the column names.
    delimiter (str, optional): The delimiter to use between columns in the CSV file. 
                               Default is tab ('\t').

    Example:
    save_csv_header('output.csv', ['Name', 'Age', 'City'], delimiter=',')
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow(header)


def save_csv_array_to_line(filename, array, delimiter='\t'):
    """
    Appends a single row of data to a CSV file with a specified delimiter.

    Parameters:
    filename (str): The name of the file to which the row will be appended.
    array (list): A list of values representing a single row of data to be appended to the CSV file.
    delimiter (str, optional): The delimiter to use between columns in the CSV file. 
                               Default is tab ('\t').

    Example:
    save_csv_array_to_line('output.csv', ['Alice', 30, 'New York'], delimiter=',')
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow(array)


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


_column_data = None
def exists_in_csv(csv_file, column, number, sep='\t'):
    """
    Checks if a number exists in a specific column of a CSV file.

    This function reads a specified column from a CSV file and checks if a given number is present in that column.

    Parameters:
    ----------
    csv_file : str
        Path to the CSV file.
    column : str or int
        The column to search in.
    number : int or float
        The number to check for existence in the column.
    sep : str, default='\t'
        Delimiter used in the CSV file.

    Returns:
    -------
    bool
        True if the number exists in the column, False otherwise.
    """
    try:
        global _column_data
        if _column_data is None:
            _column_data = read_csv(csv_file, sep=sep, col=column, to_np=True)
        return np.isin(number, _column_data)
    except IOError:
        return False


def exists_in_csv_old(csv_file, column, number, sep='\t'):
    """
    Check if a specific value exists in a given column of a CSV file.

    This function reads a CSV file and checks whether the specified `number` 
    exists in the specified `column`. If the file cannot be opened or read, 
    the function returns `False`.

    Args:
        csv_file (str): The path to the CSV file.
        column (str): The name of the column to search in.
        number (int or str): The value to search for in the specified column.
        sep (str, optional): The delimiter used in the CSV file. Defaults to '\t'.

    Returns:
        bool: 
            - `True` if the value exists in the specified column.
            - `False` if the value does not exist or if the file cannot be opened.
    """
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                if row[column] == str(number):
                    return True
    except IOError:
        return False


def pd_flatten(data, factor=1):
    """
    Flatten and process a list of data values.

    This function takes a list of data values, attempts to split each value 
    based on commas (excluding the first and last characters), and flattens 
    the resulting list. If splitting fails (e.g., due to a `TypeError`), the 
    original value is added to the result. Finally, all values are converted 
    to floats and divided by the specified `factor`.

    Args:
        data (list): A list of data values to be processed. Each value can be 
                     a string or a type that supports slicing and splitting.
        factor (float, optional): A divisor applied to each processed value. 
                                  Defaults to 1.

    Returns:
        list: A list of processed float values, flattened and divided by `factor`.
    """
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
    """
    Convert a string representation of an array back into a NumPy array.

    This function takes a string formatted as an array (e.g., "[1.0, 2.0, 3.0]"),
    removes the square brackets, splits the elements by commas, and converts 
    them into a NumPy array of floats.

    Args:
        s (str): A string representation of an array, with elements separated 
                 by commas and enclosed in square brackets.

    Returns:
        numpy.ndarray: A NumPy array containing the float values extracted 
                       from the input string.
    """
    s = s.replace('[', '').replace(']', '')  # Remove square brackets
    return np.array([float(x) for x in s.split(',')])


def pdstr_to_arrays(df):
    """
    Convert a pandas Series or DataFrame with string representations of arrays 
    into a NumPy array of actual arrays.

    This function applies the `str_to_array` function to each element of the 
    input pandas object (Series or DataFrame), converting string representations 
    of arrays into NumPy arrays. The result is returned as a NumPy array.

    Args:
        df (pandas.Series or pandas.DataFrame): A pandas object containing 
                                                string representations of arrays.

    Returns:
        numpy.ndarray: A NumPy array where each element is a NumPy array 
                       derived from the corresponding string in the input.
    """
    return df.apply(str_to_array).to_numpy()


def get_all_files_recursive(path_name=os.getcwd()):
    """
    Recursively retrieve all file paths from a directory and its subdirectories.

    This function walks through the directory tree starting from the specified 
    path and collects the full paths of all files found. If no path is provided, 
    it defaults to the current working directory.

    Args:
        path_name (str, optional): The root directory to start the search. 
                                   Defaults to the current working directory.

    Returns:
        list: A list of full file paths for all files found in the directory 
              tree starting at `path_name`.
    """    
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path_name):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles

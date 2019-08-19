#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coordutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 07/13
# License: MIT - see LICENSE for the full text.

'''
Contains various useful tools for coordinate conversion, etc.

'''

#############
## LOGGING ##
#############

import logging
from fitsbits import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


#############
## IMPORTS ##
#############

from math import trunc, fabs, pi as pi_value


#######################
## ANGLE CONVERSIONS ##
#######################

def angle_wrap(angle, radians=False):
    '''Wraps the input angle to 360.0 degrees.

    Parameters
    ----------

    angle : float
        The angle to wrap around 360.0 deg.

    radians : bool
        If True, will assume that the input is in radians. The output will then
        also be in radians.

    Returns
    -------

    float
        Wrapped angle. If radians is True: input is assumed to be in radians,
        output is also in radians.

    '''

    if radians:
        wrapped = angle % (2.0*pi_value)
        if wrapped < 0.0:
            wrapped = 2.0*pi_value + wrapped

    else:

        wrapped = angle % 360.0
        if wrapped < 0.0:
            wrapped = 360.0 + wrapped

    return wrapped


def decimal_to_dms(decimal_value):
    '''Converts from decimal degrees (for declination coords) to DD:MM:SS.

    Parameters
    ----------

    decimal_value : float
        A decimal value to convert to degrees, minutes, seconds sexagesimal
        format.

    Returns
    -------

    tuple
        A four element tuple is returned: (sign, HH, MM, SS.ssss...)

    '''

    if decimal_value < 0:
        negative = True
        dec_val = fabs(decimal_value)
    else:
        negative = False
        dec_val = decimal_value

    degrees = trunc(dec_val)
    minutes_deg = dec_val - degrees

    minutes_mm = minutes_deg * 60.0
    minutes_out = trunc(minutes_mm)
    seconds = (minutes_mm - minutes_out)*60.0

    if negative:
        degrees = degrees
        return '-', degrees, minutes_out, seconds
    else:
        return '+', degrees, minutes_out, seconds


def decimal_to_hms(decimal_value):
    '''Converts from decimal degrees (for RA coords) to HH:MM:SS.

    Parameters
    ----------

    decimal_value : float
        A decimal value to convert to hours, minutes, seconds. Negative values
        will be wrapped around 360.0.

    Returns
    -------

    tuple
        A three element tuple is returned: (HH, MM, SS.ssss...)

    '''

    # wrap to 360.0
    if decimal_value < 0:
        dec_wrapped = 360.0 + decimal_value
    else:
        dec_wrapped = decimal_value

    # convert to decimal hours first
    dec_hours = dec_wrapped/15.0

    if dec_hours < 0:
        negative = True
        dec_val = fabs(dec_hours)
    else:
        negative = False
        dec_val = dec_hours

    hours = trunc(dec_val)
    minutes_hrs = dec_val - hours

    minutes_mm = minutes_hrs * 60.0
    minutes_out = trunc(minutes_mm)
    seconds = (minutes_mm - minutes_out)*60.0

    if negative:
        hours = -hours
        return hours, minutes_out, seconds
    else:
        return hours, minutes_out, seconds


def hms_str_to_tuple(hms_string):
    '''Converts a string of the form HH:MM:SS or HH MM SS to a tuple of the form
    (HH, MM, SS).

    Parameters
    ----------

    hms_string : str
        A RA coordinate string of the form 'HH:MM:SS.sss' or 'HH MM SS.sss'.

    Returns
    -------

    tuple
        A three element tuple is returned (HH, MM, SS.ssss...)

    '''

    if ':' in hms_string:
        separator = ':'
    else:
        separator = ' '

    hh, mm, ss = hms_string.split(separator)

    return int(hh), int(mm), float(ss)


def dms_str_to_tuple(dms_string):
    '''Converts a string of the form [+-]DD:MM:SS or [+-]DD MM SS to a tuple of
    the form (sign, DD, MM, SS).

    Parameters
    ----------

    dms_string : str
        A declination coordinate string of the form '[+-]DD:MM:SS.sss' or
        '[+-]DD MM SS.sss'. The sign in front of DD is optional. If it's not
        there, this function will assume that the coordinate string is a
        positive value.

    Returns
    -------

    tuple
        A four element tuple of the form: (sign, DD, MM, SS.ssss...).

    '''
    if ':' in dms_string:
        separator = ':'
    else:
        separator = ' '

    sign_dd, mm, ss = dms_string.split(separator)
    if sign_dd.startswith('+') or sign_dd.startswith('-'):
        sign, dd = sign_dd[0], sign_dd[1:]
    else:
        sign, dd = '+', sign_dd

    return sign, int(dd), int(mm), float(ss)


def hms_str_to_decimal(hms_string):
    '''Converts a HH:MM:SS string to decimal degrees.

    Parameters
    ----------

    hms_string : str
        A right ascension coordinate string of the form: 'HH:MM:SS.sss'
        or 'HH MM SS.sss'.

    Returns
    -------

    float
        The RA value in decimal degrees (wrapped around 360.0 deg if necessary.)

    '''
    return hms_to_decimal(*hms_str_to_tuple(hms_string))


def dms_str_to_decimal(dms_string):
    '''Converts a DD:MM:SS string to decimal degrees.

    Parameters
    ----------

    dms_string : str
        A declination coordinate string of the form: '[+-]DD:MM:SS.sss'
        or '[+-]DD MM SS.sss'.

    Returns
    -------

    float
        The declination value in decimal degrees.

    '''
    return dms_to_decimal(*dms_str_to_tuple(dms_string))


def hms_to_decimal(hours, minutes, seconds, returndeg=True):
    '''Converts from HH, MM, SS to a decimal value.

    Parameters
    ----------

    hours : int
        The HH part of a RA coordinate.

    minutes : int
        The MM part of a RA coordinate.

    seconds : float
        The SS.sss part of a RA coordinate.

    returndeg : bool
        If this is True, then will return decimal degrees as the output.
        If this is False, then will return decimal HOURS as the output.
        Decimal hours are sometimes used in FITS headers.

    Returns
    -------

    float
        The right ascension value in either decimal degrees or decimal hours
        depending on `returndeg`.

    '''

    if hours > 24:

        return None

    else:

        dec_hours = fabs(hours) + fabs(minutes)/60.0 + fabs(seconds)/3600.0

        if returndeg:

            dec_deg = dec_hours*15.0

            if dec_deg < 0:
                dec_deg = dec_deg + 360.0
            dec_deg = dec_deg % 360.0
            return dec_deg
        else:
            return dec_hours


def dms_to_decimal(sign, degrees, minutes, seconds):
    '''Converts from DD:MM:SS to a decimal value.

    Parameters
    ----------

    sign : {'+', '-', ''}
        The sign part of a Dec coordinate.

    degrees : int
        The DD part of a Dec coordinate.

    minutes : int
        The MM part of a Dec coordinate.

    seconds : float
        The SS.sss part of a Dec coordinate.

    Returns
    -------

    float
        The declination value in decimal degrees.

    '''

    dec_deg = fabs(degrees) + fabs(minutes)/60.0 + fabs(seconds)/3600.0

    if sign == '-':
        return -dec_deg
    else:
        return dec_deg

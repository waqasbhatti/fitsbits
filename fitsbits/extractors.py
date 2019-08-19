#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This contains functions to help with extracting bits of info from FITS files
useful for processing tasks.

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

import os.path
import re

from .operations import read_header


#######################
## UTILITY FUNCTIONS ##
#######################

def clean_fname(
        filename,
        substr=(r'\.gz',
                r'\.fz',
                r'\.fits',
                r'\-warped',
                r'\-convolved',
                r'\-subtracted',
                r'\-combined'),
        extras=(),
        basename=False,
):
    '''
    This removes all provided substrings from the given filename.

    Useful for getting to a filename base for a FITS file.

    Parameters
    ----------

    filename : str
        The filename to remove substrings from.

    substr : sequence of str
        All the patterns to remove from the filename.

    '''

    pattern_str = '|'.join(list(substr) + list(extras))
    cleaned = re.sub(r'%s' % pattern_str, '', filename)
    return os.path.basename(cleaned) if basename else cleaned


########################
## FITS FRAME HELPERS ##
########################

def extract_frame_key(fits, *args):
    '''This is a generic function to extract the frame key for an image.

    Generic frame keys are the basename of the FITS file with all extensions
    removed.

    Parameters
    ----------

    fits : str
        The absolute path to the FITS file.

    *args : extra args
        Any extra args passed in from another function.

    Returns
    -------

    str
        This returns the frame key.

    '''

    framekey = clean_fname(os.path.basename(fits))
    return framekey


def extract_frame_time(fits, *args):
    '''
    This is a generic function to extract the time from a FITS image.

    Generic frame times are retrieved from the 'JD' key of the FITS header.

    Parameters
    ----------

    fits : str
        The absolute path to the FITS file. Will automatically handle .fz FITS.

    *args : extra args
        Any extra args passed in from another function.

    Returns
    -------

    str
        This returns the frame time.

    '''

    hdr = read_header(fits)
    return hdr['jd']


def extract_frame_targetfield(fits, *args):
    '''
    This is a generic function to extract the target object from a FITS image.

    Generic target field names are extracted from the 'OBJECT' FITS header key.

    Parameters
    ----------

    fits : str
        The absolute path to the FITS file. Will automatically handle .fz FITS.

    *args : extra args
        Any extra args passed in from another function.

    Returns
    -------

    str
        This returns the frame target field.

    '''

    hdr = read_header(fits)
    return str(hdr['object'])

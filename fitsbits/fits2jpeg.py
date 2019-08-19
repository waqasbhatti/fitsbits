#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This is a script to export a FITS image to a full JPEG, stamp PNG, or JPEGs
of specified regions.

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
import sys
from argparse import ArgumentParser

from .export import (
    fits_to_full_jpeg,
    fits_radecbox_to_jpeg,
    fits_xybox_to_jpeg,
    fits_to_stamps
)


##########
## MAIN ##
##########

def main():
    '''
    This is the main function.

    '''

    aparser = ArgumentParser(
        description=('Convert a FITS image into a JPEG, make stamp PNGs, '
                     'or generate JPEGs of specified regions in a FITS image.')
    )

    aparser.add_argument(
        'fitsfile',
        action='store',
        type=str,
        help=("Path to the FITS file to operate on.")
    )

    aparser.add_argument(
        'operation',
        action='store',
        choices=['jpeg','stamp','radecbox','xybox'],
        type=str,
        help=("Operation to perform.")
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This is a script to extract FITS headers and specific header keys.

usage: fitsbits-header [-h] [-f] [-x EXT] [-k KEYS] [-e REGEX] [-l] fitsfile

Print the headers of a FITS file, or pull out specific header keys.

positional arguments:
  fitsfile              Path to the FITS file to operate on.

optional arguments:
  -h, --help            show this help message and exit
  -f, --full            Show the entire FITS header, including any blank lines
                        (these are usually headers 'bloated' to a certain size
                        to allow quick inserts without resizing).
  -x EXT, --ext EXT     Specific FITS extension to operate on. By default, the
                        first extension that looks like a FITS image will be
                        used. This will automatically work correctly on
                        .fits.fz files.
  -k KEYS, --keys KEYS  Show the value(s) of the specified header key(s). Pass
                        in a comma-separated value list to operate on multiple
                        header keys.
  -e REGEX, --regex REGEX
                        Looks up all header cards that contain the matching
                        regex.
  -l, --listhdus        List the HDUs in the FITS file.

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
import re

from astropy.io import fits as pyfits

from .operations import read_fits, get_header_keyword_list

import warnings
warnings.simplefilter("ignore")


##########
## MAIN ##
##########

def main():
    '''
    This is the main function.

    '''
    aparser = ArgumentParser(
        description=('Print the headers of a FITS file, '
                     'or pull out specific header keys.')
    )

    aparser.add_argument(
        'fitsfile',
        action='store',
        type=str,
        help=("Path to the FITS file to operate on.")
    )
    aparser.add_argument(
        '-f','--full',
        action='store_true',
        default=False,
        help=("Show the entire FITS header, including any blank lines "
              "(these are usually headers 'bloated' to a certain size "
              "to allow quick inserts without resizing).")
    )
    aparser.add_argument(
        '-x','--ext',
        action='store',
        default=None,
        type=int,
        help=("Specific FITS extension to operate on. "
              "By default, the first extension that looks like a "
              "FITS image will be used. This will automatically work "
              "correctly on .fits.fz files.")
    )
    aparser.add_argument(
        '-k','--keys',
        action='store',
        default=None,
        type=str,
        help=("Show the value(s) of the specified header key(s). "
              "Pass in a comma-separated value list to operate on "
              "multiple header keys.")
    )
    aparser.add_argument(
        '-e','--regex',
        action='store',
        default=None,
        type=str,
        help=("Looks up all header cards that contain the matching regex.")
    )
    aparser.add_argument(
        '-l','--listhdus',
        action='store_true',
        default=False,
        help=("List the HDUs in the FITS file.")
    )

    args = aparser.parse_args()

    if not args.fitsfile or not os.path.exists(args.fitsfile):

        print('Could not find the FITS file specified.')
        sys.exit(1)

    # if we're listing HDUs only
    if args.listhdus:

        hdul = pyfits.open(args.fitsfile)
        print(hdul.info())
        sys.exit(0)

    # if we have specific keys we want the values for
    if args.keys is not None:

        keys_to_get = args.keys.split(',')

        header_keys = get_header_keyword_list(
            args.fitsfile,
            keys_to_get,
            ext=args.ext
        )

        for key in keys_to_get:
            print('%10s -> %50s' % (key, header_keys[key]))

    # if we have a regex to look up in the keys
    elif args.regex is not None:

        img, hdr = read_fits(args.fitsfile, ext=args.ext)
        hdrstr = hdr.tostring(padding=False,
                              sep='\n',
                              endcard=False)
        hdrstr = hdrstr.split('\n')
        hdrstr = [x.strip() for x in hdrstr]
        match_regex = re.compile(r'%s' % args.regex)

        nmatches = 0
        for h in hdrstr:

            matching = match_regex.search(h)

            if matching is not None:
                print(h)
                nmatches = nmatches + 1

        if nmatches == 0:
            print("No matches to the regex string: '%s' found." % args.regex)

    # otherwise, we're in print-full-header mode
    else:

        img, hdr = read_fits(args.fitsfile, ext=args.ext)
        hdrstr = hdr.tostring(padding=False,
                              sep='\n',
                              endcard=False)

        hdrstr = hdrstr.split('\n')
        hdrstr = [x.strip() for x in hdrstr]

        if not args.full:
            hdrstr = [x for x in hdrstr if len(x) > 0]

        for card in hdrstr:
            print(card)


if __name__ == '__main__':

    # handle SIGPIPE sent by less, head, et al.
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # call main()
    main()

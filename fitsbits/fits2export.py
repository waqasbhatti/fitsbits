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

from . import operations
from ._extractors import clean_fname


##################################
## FUNCTIONS TO HANDLE COMMANDS ##
##################################

def handle_export_to_raster(args):
    '''
    This handles the export_to_raster command.

    '''

    if not args.fitsfile:
        LOGERROR("A FITS filename is required.")
        sys.exit(1)

    in_file = os.path.abspath(args.fitsfile)

    # decide the scaler
    if args.scale == 'zscale':
        scaler = operations.zscale_image
    elif args.scale == 'linear':
        scaler = operations.clipped_linscale_img
    elif args.scale == 'log':
        scaler = operations.clipped_logscale_img
    else:
        scaler = operations.clipped_linscale_img

    # see if there's a resize flag
    if args.resize is not None:
        resize = True
        resizefrac = float(args.resize)
    else:
        resize = False
        resizefrac = None

    if not args.output:
        outputf = '%s.jpg' % clean_fname(args.fitsfile)
    else:
        outputf = args.output

    fits_to_full_jpeg(
        in_file,
        out_fname=outputf,
        ext=args.ext,
        trim=True,
        trimkeys=[x.strip() for x in args.trim.split(',')],
        scale_func=scaler,
        resize=resize,
        resizefrac=resizefrac,
    )
    LOGINFO("Wrote %s" % os.path.abspath(outputf))


def handle_export_to_stamps(args):
    '''
    This handles the export_to_stamps command.

    '''

    if not args.fitsfile:
        LOGERROR("A FITS filename is required.")
        sys.exit(1)

    # decide the scaler
    if args.scale == 'zscale':
        scaler = operations.zscale_image
    elif args.scale == 'linear':
        scaler = operations.clipped_linscale_img
    elif args.scale == 'log':
        scaler = operations.clipped_logscale_img
    else:
        scaler = operations.clipped_linscale_img

    in_file = os.path.abspath(args.fitsfile)

    if not args.output:
        outputf = '%s-stamps-3x3.png' % clean_fname(args.fitsfile)
    else:
        outputf = args.output

    fits_to_stamps(
        in_file,
        outputf,
        fits_extension=args.ext,
        trimkeys=[x.strip() for x in args.trim.split(',')],
        scale_func=scaler,
        stampsize=args.stampsize,
    )
    LOGINFO("Wrote %s" % os.path.abspath(outputf))


def handle_export_to_radecbox(args):
    '''
    This handles the export_to_radecbox command.

    '''

    if not args.fitsfile:
        LOGERROR("A FITS filename is required.")
        sys.exit(1)

    # decide the scaler
    if args.scale == 'zscale':
        scaler = operations.zscale_image
    elif args.scale == 'linear':
        scaler = operations.clipped_linscale_img
    elif args.scale == 'log':
        scaler = operations.clipped_logscale_img
    else:
        scaler = operations.clipped_linscale_img

    in_file = os.path.abspath(args.fitsfile)

    # get the box spec
    box = [float(x.strip()) for x in args.box.split(',')]

    # generate the box info string
    if args.boxtype == 'center':
        box_infostr = 'RC%.3fDC%.3f-RW%.3fDW%.3f' % (
            box[0], box[1], box[2], box[3]
        )
    elif args.boxtype == 'bounds':
        box_infostr = 'RL%.3fRH%.3f-DL%.3fDH%.3f' % (
            box[0], box[1], box[2], box[3]
        )

    if not args.output:
        outputf = '%s-%s.png' % (clean_fname(args.fitsfile), box_infostr)
    else:
        outputf = args.output

    fits_radecbox_to_jpeg(
        in_file,
        box,
        boxtype=args.boxtype,
        wcsfrom=args.wcs,
        out_fname=outputf,
        ext=args.ext,
        trimkeys=[x.strip() for x in args.trim.split(',')],
        scale_func=scaler,
    )
    LOGINFO("Wrote %s" % os.path.abspath(outputf))


def handle_export_to_xybox(args):
    '''
    This handles the export_to_xybox command.

    '''

    if not args.fitsfile:
        LOGERROR("A FITS filename is required.")
        sys.exit(1)

    # decide the scaler
    if args.scale == 'zscale':
        scaler = operations.zscale_image
    elif args.scale == 'linear':
        scaler = operations.clipped_linscale_img
    elif args.scale == 'log':
        scaler = operations.clipped_logscale_img
    else:
        scaler = operations.clipped_linscale_img

    in_file = os.path.abspath(args.fitsfile)

    # get the box spec
    box = [float(x.strip()) for x in args.box.split(',')]

    # generate the box info string
    if args.boxtype == 'center':
        box_infostr = 'XC%iYC%i-XW%iYW%i' % (
            box[0], box[1], box[2], box[3]
        )
    elif args.boxtype == 'bounds':
        box_infostr = 'XL%iXH%i-YL%iYH%i' % (
            box[0], box[1], box[2], box[3]
        )

    if not args.output:
        outputf = '%s-%s.png' % (clean_fname(args.fitsfile), box_infostr)
    else:
        outputf = args.output

    fits_xybox_to_jpeg(
        in_file,
        box,
        boxtype=args.boxtype,
        out_fname=outputf,
        ext=args.ext,
        trimkeys=[x.strip() for x in args.trim.split(',')],
        scale_func=scaler,
    )
    LOGINFO("Wrote %s" % os.path.abspath(outputf))


##########
## MAIN ##
##########

def main():
    '''
    This is the main function.

    '''

    aparser = ArgumentParser(
        description=('Export a FITS image to a JPEG or PNG, '
                     'make stamp JPEGs or PNGs, '
                     'generate JPEGs or PNGs of specified '
                     'regions in a FITS image.')
    )

    #
    # add the subparsers
    #

    subparsers = aparser.add_subparsers(
        description=(
            "Use the --help flag with any command to see its arguments."
        ),
        title='Command',
    )
    subparsers.required = True
    subparsers.dest = 'command'

    export_to_raster = subparsers.add_parser(
        'image',
        help="Export the full image to a JPEG or PNG.",
    )
    export_to_stamps = subparsers.add_parser(
        'stamps',
        help="Generate stamps and then export to a JPEG or PNG.",
    )
    export_to_radecbox = subparsers.add_parser(
        'radecbox',
        help="Export a RA/Dec region of the image to a JPEG or PNG.",
    )
    export_to_xybox = subparsers.add_parser(
        'xybox',
        help="Export a x/y region of the image to a JPEG or PNG.",
    )

    #
    # the export_to_raster subparser
    #
    export_to_raster.add_argument(
        'fitsfile',
        action='store',
        type=str,
        help=("Path to the FITS file to operate on.")
    )
    export_to_raster.add_argument(
        '-x','--ext',
        action='store',
        default=None,
        type=int,
        help=("Specific FITS extension to operate on. "
              "By default, the first extension that looks like a "
              "FITS image will be used. This will automatically work "
              "correctly on .fits.fz files.")
    )
    export_to_raster.add_argument(
        '-o','--output',
        action='store',
        default=None,
        type=str,
        help=("The output file to write to. "
              "If the file ends in .jpg, a JPEG will be written. "
              "If it ends with .png, a PNG will be written. "
              "If not provided, the default action is to export to a JPEG.")
    )
    export_to_raster.add_argument(
        '-t','--trim',
        action='store',
        default='TRIMSEC,DATASEC,TRIMSEC0',
        type=str,
        help=("If this is set, the input FITS will be trimmed using "
              "any of the provided trim keys from the FITS header. "
              "Separate these with a comma. "
              "The default is 'TRIMSEC,DATASEC,TRIMSEC0'")
    )
    export_to_raster.add_argument(
        '-s','--scale',
        action='store',
        default='zscale',
        choices=['linear','zscale','log'],
        type=str,
        help=("The image scaling function to use. The default is 'zscale'.")
    )
    export_to_raster.add_argument(
        '-r','--resize',
        action='store',
        default=None,
        type=float,
        help=("If set, is the fraction to resize the "
              "image width and height by.")
    )
    export_to_raster.set_defaults(func=handle_export_to_raster)

    #
    # the export_to_stamps subparser
    #
    export_to_stamps.add_argument(
        'fitsfile',
        action='store',
        type=str,
        help=("Path to the FITS file to operate on.")
    )
    export_to_stamps.add_argument(
        '-x','--ext',
        action='store',
        default=None,
        type=int,
        help=("Specific FITS extension to operate on. "
              "By default, the first extension that looks like a "
              "FITS image will be used. This will automatically work "
              "correctly on .fits.fz files.")
    )
    export_to_stamps.add_argument(
        '-o','--output',
        action='store',
        default=None,
        type=str,
        help=("The output file to write to. "
              "If the file ends in .jpg, a JPEG will be written. "
              "If it ends with .png, a PNG will be written. "
              "If not provided, the default action is to export to a PNG.")
    )
    export_to_stamps.add_argument(
        '-t','--trim',
        action='store',
        default='TRIMSEC,DATASEC,TRIMSEC0',
        type=str,
        help=("If this is set, the input FITS will be trimmed using "
              "any of the provided trim keys from the FITS header. "
              "Separate these with a comma. "
              "The default is 'TRIMSEC,DATASEC,TRIMSEC0'")
    )
    export_to_stamps.add_argument(
        '-s','--scale',
        action='store',
        default='zscale',
        choices=['linear','zscale','log'],
        type=str,
        help=("The image scaling function to use. The default is 'zscale'.")
    )
    export_to_stamps.add_argument(
        '-e','--stampsize',
        action='store',
        default=128,
        type=int,
        help=("The size (width and height) of each stamp in pixels.")
    )
    export_to_stamps.set_defaults(func=handle_export_to_stamps)

    #
    # the export_to_radecbox subparser
    #
    export_to_radecbox.add_argument(
        'fitsfile',
        action='store',
        type=str,
        help=("Path to the FITS file to operate on.")
    )
    export_to_radecbox.add_argument(
        '-x','--ext',
        action='store',
        default=None,
        type=int,
        help=("Specific FITS extension to operate on. "
              "By default, the first extension that looks like a "
              "FITS image will be used. This will automatically work "
              "correctly on .fits.fz files.")
    )
    export_to_radecbox.add_argument(
        '-o','--output',
        action='store',
        default=None,
        type=str,
        help=("The output file to write to. "
              "If the file ends in .jpg, a JPEG will be written. "
              "If it ends with .png, a PNG will be written. "
              "If not provided, the default action is to export to a PNG.")
    )
    export_to_radecbox.add_argument(
        '-t','--trim',
        action='store',
        default='TRIMSEC,DATASEC,TRIMSEC0',
        type=str,
        help=("If this is set, the input FITS will be trimmed using "
              "any of the provided trim keys from the FITS header. "
              "Separate these with a comma. "
              "The default is 'TRIMSEC,DATASEC,TRIMSEC0'")
    )
    export_to_radecbox.add_argument(
        '-s','--scale',
        action='store',
        default='zscale',
        choices=['linear','zscale','log'],
        type=str,
        help=("The image scaling function to use. The default is 'zscale'.")
    )
    export_to_radecbox.add_argument(
        '-b','--boxtype',
        action='store',
        choices=['center','bounds'],
        default='center',
        type=str,
        help=("The type of box to use.")
    )
    export_to_radecbox.add_argument(
        '-a','--box',
        action='store',
        default=None,
        type=str,
        help=("If boxtype is 'bounds', "
              "specify coordinates as: 'RAmin,RAmax,DECmin,DECmax'. "
              "If boxtype is 'center', "
              "specify coordinates as: 'RAcenter,DECcenter,RAwidth,DECwidth'.")
    )
    export_to_radecbox.add_argument(
        '-w','--wcs',
        action='store',
        default=None,
        type=str,
        help=("If the image WCS is to be extracted from another file, "
              "specify that file's name here. By default, the WCS stored "
              "in the FITS image's header will be used to convert RA/Dec "
              "to pixel coordinates.")
    )
    export_to_radecbox.set_defaults(func=handle_export_to_radecbox)

    #
    # the export_to_xybox subparser
    #
    export_to_xybox.add_argument(
        'fitsfile',
        action='store',
        type=str,
        help=("Path to the FITS file to operate on.")
    )
    export_to_xybox.add_argument(
        '-x','--ext',
        action='store',
        default=None,
        type=int,
        help=("Specific FITS extension to operate on. "
              "By default, the first extension that looks like a "
              "FITS image will be used. This will automatically work "
              "correctly on .fits.fz files.")
    )
    export_to_xybox.add_argument(
        '-o','--output',
        action='store',
        default=None,
        type=str,
        help=("The output file to write to. "
              "If the file ends in .jpg, a JPEG will be written. "
              "If it ends with .png, a PNG will be written. "
              "If not provided, the default action is to export to a PNG.")
    )
    export_to_xybox.add_argument(
        '-t','--trim',
        action='store',
        default='TRIMSEC,DATASEC,TRIMSEC0',
        type=str,
        help=("If this is set, the input FITS will be trimmed using "
              "any of the provided trim keys from the FITS header. "
              "Separate these with a comma. "
              "The default is 'TRIMSEC,DATASEC,TRIMSEC0'")
    )
    export_to_xybox.add_argument(
        '-s','--scale',
        action='store',
        default='zscale',
        choices=['linear','zscale','log'],
        type=str,
        help=("The image scaling function to use. The default is 'zscale'.")
    )
    export_to_xybox.add_argument(
        '-b','--boxtype',
        action='store',
        choices=['center','bounds'],
        default='center',
        type=str,
        help=("The type of box to use.")
    )
    export_to_xybox.add_argument(
        '-a','--box',
        action='store',
        default=None,
        type=str,
        help=("If boxtype is 'bounds', "
              "specify coordinates as: 'Xmin,Xmax,Ymin,Ymax'. "
              "If boxtype is 'center', "
              "specify coordinates as: 'Xcenter,Ycenter,Xwidth,Ywidth'.")
    )
    export_to_xybox.set_defaults(func=handle_export_to_xybox)

    #
    # parse all the args
    #
    args = aparser.parse_args()

    #
    # dispatch the function
    #
    args.func(args)


if __name__ == '__main__':
    main()

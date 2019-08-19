#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This is a script to convert a directory of FITS files into a movie using
`ffmpeg`.

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
import glob
from argparse import ArgumentParser
import subprocess
import multiprocessing

# Ref: https://bugs.python.org/issue33725
# TLDR; Apple is trash at UNIX
if sys.platform == 'darwin':
    mp = multiprocessing.get_context('forkserver')
else:
    mp = multiprocessing

NCPUS = mp.cpu_count()

from . import operations
from .export import (
    make_frame_movie,
    make_frame_movie_from_list,
    parallel_frame_jpegs
)


#######################
## HANDLER FUNCTIONS ##
#######################

def handle_export_from_folder(args):
    '''
    This handles the export_from_folder command.

    '''

    # check if the folder exists
    if not os.path.exists(args.infolder):
        LOGERROR("The input folder: %s does not exist." % args.infolder)
        sys.exit(1)

    # check if there are any FITS files in the folder
    flist = glob.glob(os.path.join(args.infolder, args.fitsglob))

    if len(flist) == 0:
        LOGERROR("The input folder: %s does not "
                 "contain any FITS files matching: %s"
                 % (args.infolder, args.fitsglob))
        sys.exit(1)
    elif len(flist) > 1:
        LOGINFO("%s files to work on." % len(flist))
    else:
        LOGERROR("Can't make a movie with one file.")
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

    # see if there's a resize flag
    if args.resize is not None:
        resize = True
        resizefrac = float(args.resize)
    else:
        resize = False
        resizefrac = None

    LOGINFO("Converting FITS files to images if required...")

    # convert all of these files to JPEGs
    parallel_frame_jpegs(
        args.infolder,
        fitsglob=args.fitsglob,
        outf_extension=args.frametype,
        outf_postfix=args.framepostfix,
        ext=args.ext,
        trim=True,
        trimkeys=[x.strip() for x in args.trim.split(',')],
        scale_func=scaler,
        resize=resize,
        resizefrac=resizefrac,
        nworkers=args.workers,
    )

    # now, turn the JPEGs into movies
    make_frame_movie(
        args.infolder,
        args.outfile,
        framerate=args.framerate,
        crf=args.crf,
        preset=args.preset,
        rescale_to_width=None,
        fileglob='*.%s' % args.frametype
    )


def handle_export_from_list(args):
    '''
    This handles the export_from_list command.

    '''

    # check if the folder exists
    if not os.path.exists(args.infile):
        LOGERROR("The input file: %s does not exist." % args.infolder)
        sys.exit(1)

    # check if there are any FITS files in the file
    with open(args.infile,'r') as infd:
        flist = infd.readlines()
        flist = [os.path.abspath(x.strip('\n')) for x in flist]

    if len(flist) == 0:
        LOGERROR("The input file: %s does not "
                 "contain any FITS files." % args.infile)
        sys.exit(1)
    elif len(flist) > 1:
        LOGINFO("%s files to work on." % len(flist))
    else:
        LOGERROR("Can't make a movie with one file.")
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

    # see if there's a resize flag
    if args.resize is not None:
        resize = True
        resizefrac = float(args.resize)
    else:
        resize = False
        resizefrac = None

    LOGINFO("Converting FITS files to images if required...")

    # convert all of these files to JPEGs
    jpegs = parallel_frame_jpegs(
        flist,
        outf_extension=args.frametype,
        outf_postfix=args.framepostfix,
        ext=args.ext,
        trim=True,
        trimkeys=[x.strip() for x in args.trim.split(',')],
        scale_func=scaler,
        resize=resize,
        resizefrac=resizefrac,
        nworkers=args.workers,
    )

    # now, turn the JPEGs into movies
    make_frame_movie_from_list(
        [jpegs[x] for x in jpegs],
        args.outfile,
        framerate=args.framerate,
        crf=args.crf,
        preset=args.preset,
        rescale_to_width=None,
    )


##########
## MAIN ##
##########

def main():
    '''
    This is the main function.

    '''

    # check for ffmpeg
    try:
        subprocess.run('ffmpeg -version',
                       shell=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=True)

    except subprocess.CalledProcessError:

        LOGERROR("Could not find the `ffmpeg` command in your path.")

        if sys.platform == 'linux':
            LOGINFO("Install FFmpeg using your Linux distribution's "
                    "package manager "
                    "or use the static build from "
                    "https://www.johnvansickle.com/ffmpeg/ and "
                    "put it in a directory in your path "
                    "(~/bin or ~/.local/bin).")
        elif sys.platform == 'darwin':
            LOGINFO("Install FFmpeg using Homebrew (https://brew.sh/) with the "
                    "following command: brew install ffmpeg")
        elif sys.platform == 'win32':
            LOGINFO("Get FFmpeg for Windows: "
                    "https://ffmpeg.zeranoe.com/builds/")
        else:
            LOGWARNING("Please install FFmpeg for your platform: '%s'." %
                       sys.platform)

        sys.exit(1)

    #
    # parse the args now
    #
    aparser = ArgumentParser(
        description=(
            'Export a directory or list of FITS images to a MP4 movie. '
            'Requires ffmpeg.'
        )
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

    export_from_folder = subparsers.add_parser(
        'folder',
        help=("Export all FITS files from the specified folder to a movie. "
              "The files will be sorted alphabetically by their file names."),
    )
    export_from_list = subparsers.add_parser(
        'list',
        help=("Export all FITS files listed in the "
              "specified text file to a movie. "
              "There must be one input FITS filename per line."),
    )

    #
    # the export_from_folder subparser
    #
    export_from_folder.add_argument(
        'infolder',
        action='store',
        type=str,
        help=("Path to the folder containing FITS files to operate on.")
    )
    export_from_folder.add_argument(
        'outfile',
        action='store',
        type=str,
        help=("The filename of the movie file to generate.")
    )
    export_from_folder.add_argument(
        '-g','--fitsglob',
        action='store',
        default='*.fits*',
        type=str,
        help=("The file glob to use to find FITS files. "
              "The default is '*.fits*'.")
    )
    export_from_folder.add_argument(
        '-z','--framepostfix',
        action='store',
        default=None,
        type=str,
        help=("The filename postfix to use for the frame images generated "
              "from the input FITS files. If specified, the generated images "
              "will look like: "
              "{infolder}/{fits_fname}-{framepostfix}.{frametype}.")
    )
    export_from_folder.add_argument(
        '-e','--frametype',
        action='store',
        default='jpg',
        choices=['png','jpg'],
        type=str,
        help=("The type of image frame to generate from the "
              "FITS files and use for the movie. "
              "The default is 'jpg'.")
    )
    export_from_folder.add_argument(
        '-x','--ext',
        action='store',
        default=None,
        type=int,
        help=("Specific FITS extension to operate on. "
              "By default, the first extension that looks like a "
              "FITS image will be used. This will automatically work "
              "correctly on .fits.fz files.")
    )
    export_from_folder.add_argument(
        '-t','--trim',
        action='store',
        default='TRIMSEC,DATASEC,TRIMSEC0',
        type=str,
        help=("If this is set, the input FITS files will be trimmed using "
              "any of the provided trim keys from the FITS header. "
              "Separate these with a comma. "
              "The default is 'TRIMSEC,DATASEC,TRIMSEC0'")
    )
    export_from_folder.add_argument(
        '-s','--scale',
        action='store',
        default='zscale',
        choices=['linear','zscale','log'],
        type=str,
        help=("The image scaling function to use. The default is 'zscale'.")
    )
    export_from_folder.add_argument(
        '-r','--resize',
        action='store',
        default=None,
        type=float,
        help=("If set, is the fraction to resize the "
              "image width and height by.")
    )
    export_from_folder.add_argument(
        '-f','--framerate',
        action='store',
        default=10,
        type=int,
        help=("The frame rate (fps) to use for the movie. "
              "The default is 10.")
    )
    export_from_folder.add_argument(
        '-c','--crf',
        action='store',
        default=17,
        type=int,
        help=("The H.264 compression CRF to use for the movie (0--51). "
              "The default is 10. "
              "0 indicates lossless encoding from images to video. "
              "17 indicates visually lossless compression. "
              "51 indicates maximum lossy compression.")
    )
    export_from_folder.add_argument(
        '-p','--preset',
        action='store',
        choices=['veryslow','slower','slow','fast','veryfast'],
        default='slower',
        type=str,
        help=("The H.264 encoding speed to use for the movie. "
              "The default is 'slower'. 'veryslow' indicates slowest "
              "encoding speed but smallest MP4 files. "
              "'veryfast' indicates fastest "
              "encoding speed but largest MP4 files. ")
    )
    export_from_folder.add_argument(
        '-w','--workers',
        action='store',
        default=None,
        type=int,
        help=("The maximum number of parallel workers to use "
              "to convert FITS to images. The default is to use all CPUs.")
    )
    export_from_folder.set_defaults(func=handle_export_from_folder)

    #
    # the export_from_list subparser
    #
    export_from_list.add_argument(
        'infile',
        action='store',
        type=str,
        help=("Path to the file containing a list of FITS files to operate on.")
    )
    export_from_list.add_argument(
        'outfile',
        action='store',
        type=str,
        help=("The filename of the movie file to generate.")
    )
    export_from_list.add_argument(
        '-e','--frametype',
        action='store',
        default='jpg',
        choices=['png','jpg'],
        type=str,
        help=("The type of image frame to generate from the "
              "FITS files and use for the movie. "
              "The default is 'jpg'.")
    )
    export_from_list.add_argument(
        '-z','--framepostfix',
        action='store',
        default=None,
        type=str,
        help=("The filename postfix to use for the frame images generated "
              "from the input FITS files. If specified, the generated images "
              "will look like: "
              "{infolder}/{fits_fname}-{framepostfix}.{frametype}.")
    )
    export_from_list.add_argument(
        '-x','--ext',
        action='store',
        default=None,
        type=int,
        help=("Specific FITS extension to operate on. "
              "By default, the first extension that looks like a "
              "FITS image will be used. This will automatically work "
              "correctly on .fits.fz files.")
    )
    export_from_list.add_argument(
        '-t','--trim',
        action='store',
        default='TRIMSEC,DATASEC,TRIMSEC0',
        type=str,
        help=("If this is set, the input FITS files will be trimmed using "
              "any of the provided trim keys from the FITS header. "
              "Separate these with a comma. "
              "The default is 'TRIMSEC,DATASEC,TRIMSEC0'")
    )
    export_from_list.add_argument(
        '-s','--scale',
        action='store',
        default='zscale',
        choices=['linear','zscale','log'],
        type=str,
        help=("The image scaling function to use. The default is 'zscale'.")
    )
    export_from_list.add_argument(
        '-r','--resize',
        action='store',
        default=None,
        type=float,
        help=("If set, is the fraction to resize the "
              "image width and height by.")
    )
    export_from_list.add_argument(
        '-f','--framerate',
        action='store',
        default=10,
        type=int,
        help=("The frame rate (fps) to use for the movie. "
              "The default is 10.")
    )
    export_from_list.add_argument(
        '-c','--crf',
        action='store',
        default=17,
        type=int,
        help=("The H.264 compression CRF to use for the movie (0--51). "
              "The default is 10. "
              "0 indicates lossless encoding from images to video. "
              "17 indicates visually lossless compression. "
              "51 indicates maximum lossy compression.")
    )
    export_from_list.add_argument(
        '-p','--preset',
        action='store',
        choices=['veryslow','slower','slow','fast','veryfast'],
        default='slower',
        type=str,
        help=("The H.264 encoding speed to use for the movie. "
              "The default is 'slower'. 'veryslow' indicates slowest "
              "encoding speed but smallest MP4 files. "
              "'veryfast' indicates fastest "
              "encoding speed but largest MP4 files. ")
    )
    export_from_list.add_argument(
        '-w','--workers',
        action='store',
        default=None,
        type=int,
        help=("The maximum number of parallel workers to use "
              "to convert FITS to images. The default is to use all CPUs.")
    )
    export_from_list.set_defaults(func=handle_export_from_list)

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

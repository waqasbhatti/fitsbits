#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This module contains functions to extract various bits of information from
FITS files related to their quality, e.g. header keys related to the
environment, image background levels, etc.

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
import pickle
import sys

import multiprocessing
# Ref: https://bugs.python.org/issue33725
# TLDR; Apple is trash at UNIX
if sys.platform == 'darwin':
    mp = multiprocessing.get_context('forkserver')
else:
    mp = multiprocessing

import numpy as np
import numpy.ma as npma

from ._coordutils import hms_str_to_decimal, dms_str_to_decimal

from .operations import (
    read_fits,
    compressed_fits_ext,
    trim_image
)
from ._extractors import clean_fname

NCPUS = mp.cpu_count()


#######################
## FRAME WARP CHECKS ##
#######################

def check_frame_badwarp(
        frame,
        ext=None,
        margins=50,
        polyorder=(4,4),
        max_ratio=(50.0, 50.0),
        makeplot=False
):
    '''This checks if a warped ("shifted" to some astrometric reference
    coordinates) frame is bad, perhaps caused by a bad shift/convolution.

    Calculates the median of the rows and columns of the image taking into
    account the margin on either side (as specified by the margins kwarg). Then
    fits a straight line to the trend. Then fits a polynomial of the specified
    `polyorder`. The test for bad frame warps is then::

        (chi-sq of straight line fit / chi-sq of polynomial fit) > max_ratio

    Parameters
    ----------

    frame : str
        The input frame to process.

    ext : int or None
        The FITS extension containing the image to process. If None, determined
        automatically.

    margins : int
        The margins around the edges of the frame to leave out of the fits.

    polyorder : tuple of two ints
        The order of the polynomial in x and y directions respectively to fit in
        addition to a straight line fit.

    max_ratio : tuple of two floats
        The maximum ratio between the linear and polynomial fits in the x and y
        directions that is allowed to consider this frame as OK and not badly
        warped.

    makeplot : bool
        If True, plots the medians along the rows and columns and the respective
        fits.

    Returns
    -------

    dict
        Returns an info dict containing the fit and x-y median info.

    '''

    compressed_check = compressed_fits_ext(frame)
    if compressed_check and ext is None:
        ext = compressed_check[0]
    elif not compressed_check and ext is None:
        ext = 0

    image, header = read_fits(frame, ext=ext)

    clippedimage = image[margins:-margins, margins:-margins]

    # FIXME: this should probably be:
    # yimagecoordnum, ximagecoordnum = clippedimage.shape
    # and then use them separately to make aranges in each dimension
    imagecoordnum = np.arange(len(clippedimage))

    # get the medians in the x and y directions
    medx = np.nanmedian(clippedimage,axis=1)
    medy = np.nanmedian(clippedimage,axis=0)

    # fit a 1-degree polynomial
    lin_xfitcoeffs = np.polyfit(imagecoordnum,medx,1)
    lin_yfitcoeffs = np.polyfit(imagecoordnum,medy,1)

    xpolyorder, ypolyorder = polyorder

    # fit a polyorder-degree polynomial
    poly_xfitcoeffs = np.polyfit(imagecoordnum,medx,xpolyorder)
    poly_yfitcoeffs = np.polyfit(imagecoordnum,medy,ypolyorder)

    # linfit polynomial
    lin_xfitpoly = np.poly1d(lin_xfitcoeffs)
    lin_yfitpoly = np.poly1d(lin_yfitcoeffs)

    # polyorderfit polynomial
    poly_xfitpoly = np.poly1d(poly_xfitcoeffs)
    poly_yfitpoly = np.poly1d(poly_yfitcoeffs)

    lin_xfit = lin_xfitpoly(imagecoordnum)
    lin_yfit = lin_yfitpoly(imagecoordnum)

    poly_xfit = poly_xfitpoly(imagecoordnum)
    poly_yfit = poly_yfitpoly(imagecoordnum)

    lin_xfit_redchisq = (
        np.sum((medx - lin_xfit)*(medx - lin_xfit))/(len(imagecoordnum) - 2)
    )
    lin_yfit_redchisq = (
        np.sum((medy - lin_yfit)*(medy - lin_yfit))/(len(imagecoordnum) - 2)
    )

    poly_xfit_redchisq = (
        np.sum((medx - poly_xfit)*(medx - poly_xfit))/(len(imagecoordnum) -
                                                       xpolyorder)
    )
    poly_yfit_redchisq = (
        np.sum((medy - poly_yfit)*(medy - poly_yfit))/(len(imagecoordnum) -
                                                       ypolyorder)
    )

    check_ratio_x = lin_xfit_redchisq/poly_xfit_redchisq
    check_ratio_y = lin_yfit_redchisq/poly_yfit_redchisq

    if check_ratio_x > max_ratio[0] and check_ratio_y > max_ratio[1]:
        warp_ok = False
    else:
        warp_ok = True

    warpinfo = {
        'medx':medx,
        'medy':medy,
        'lin_xfitpoly':lin_xfitpoly,
        'lin_yfitpoly':lin_yfitpoly,
        'lin_xfit':lin_xfit,
        'lin_yfit':lin_yfit,
        'lin_xfit_redchisq':lin_xfit_redchisq,
        'lin_yfit_redchisq':lin_yfit_redchisq,
        'poly_xfitpoly':poly_xfitpoly,
        'poly_yfitpoly':poly_yfitpoly,
        'poly_xfit':poly_xfit,
        'poly_yfit':poly_yfit,
        'poly_xfit_redchisq':poly_xfit_redchisq,
        'poly_yfit_redchisq':poly_yfit_redchisq,
        'check_ratio_x': check_ratio_x,
        'check_ratio_y': check_ratio_y,
        'warp_ok':warp_ok,
        'kwargs':{
            'frame':frame,
            'ext':ext,
            'margins':margins,
            'polyorder':polyorder,
            'max_ratio':max_ratio,
        }
    }

    if makeplot:

        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(1, figsize=(6.4*2,4.8),clear=True)
            ax1, ax2 = fig.subplots(ncols=2, nrows=1)

            ax1.plot(imagecoordnum, medx, 'k-')
            ax1.plot(imagecoordnum, lin_xfit, 'r-', alpha=0.5, label='linear')
            ax1.plot(imagecoordnum, poly_xfit, 'b-', alpha=0.5,
                     label='polyorder %s' % xpolyorder)
            ax1.set_xlabel('image coord number')
            ax1.set_ylabel('x medians, linear fit, and poly fit')

            ax2.plot(imagecoordnum, medy, 'k-')
            ax2.plot(imagecoordnum, lin_yfit, 'r-', alpha=0.5, label='linear')
            ax2.plot(imagecoordnum, poly_yfit, 'b-', alpha=0.5,
                     label='polyorder %s' % xpolyorder)
            ax2.set_xlabel('image coord number')
            ax2.set_ylabel('y medians, linear fit, and poly fit')

            savename = '%s-warp-check.png' % clean_fname(frame)

            fig.savefig(savename, bbox_inches='tight', dpi=100)
            plt.close('all')

            LOGINFO('Wrote frame-warp diagnostic plot for %s to %s' %
                    (frame, savename))

            # add the diagnostic plot to the warpinfo dict
            warpinfo['plot'] = os.path.abspath(savename)

        except Exception:
            LOGEXCEPTION("Could not make a warp-check plot. "
                         "Is matplotlib installed?")

    return warpinfo


######################
## FRAME BACKGROUND ##
######################

def extract_img_background(frame,
                           ext=None,
                           custom_limits=None,
                           median_diffbelow=200.0,
                           image_min=None):
    '''
    This extracts the background of the image array provided:

    - masks the array to only values between the median and the min of flux

    img_array = image to find the background for

    custom_limits = use this to provide custom median and min limits for the
                    background extraction

    median_diffbelow = subtract this value from the median to get the upper
                       bound for background extraction

    image_min = use this value as the lower bound for background extraction

    '''

    if isinstance(frame, str):

        compressed_check = compressed_fits_ext(frame)
        if compressed_check and ext is None:
            ext = compressed_check[0]
        elif not compressed_check and ext is None:
            ext = 0

        image, header = read_fits(frame, ext=ext)

    elif isinstance(frame, tuple):
        image, header = frame

    if not custom_limits:

        backmax = np.nanmedian(image) - median_diffbelow
        backmin = image_min if image_min is not None else np.nanmin(image)

    else:

        backmin, backmax = custom_limits

    masked = npma.masked_outside(image, backmin, backmax)
    backmasked = npma.median(masked)

    return backmasked


######################################
## FRAME SOURCE AND PHOTOMETRY INFO ##
######################################

def tile_frame(image, tilesize=128):
    '''
    This returns a list of image tiles and the number of tiles in x, y.

    '''

    # tile the image
    ntiles_x = int(np.floor(image.shape[1]/tilesize))
    ntiles_y = int(np.floor(image.shape[0]/tilesize))

    if ntiles_x != ntiles_y:
        ntiles_y = ntiles_x

    tiles = [
        image[
            i*tilesize:i*tilesize +
            tilesize,
            j*tilesize:j*tilesize +
            tilesize
        ] for i in range(ntiles_y) for j in range(ntiles_x)
    ]

    return tiles, ntiles_x, ntiles_y


def frame_source_info(
        fits,
        stats_only=False,
        trim_frame=True,
        trim_headerkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
        calibrated_imagetype_headerkey='calibratedobject',
        fits_ext=None,
        fits_racenter_key='RA',
        fits_racenter_unit='hr',
        fits_declcenter_key='DEC',
        fits_declcenter_unit='deg',
        fits_background_mediandiffbelow=0.0,
        fits_background_tilesize=128,
        # these are used by HAT Survey instruments
        fits_environ_keys=(
            'Z',
            'MOONDIST',
            'MOONELEV',
            'MOONPH',
            'HA',
            'WIND',
            'HUMIDITY',
            'SKYTDIFF',
            'AMBTEMP',
            'DEWPT',
            'CCDTEMP',
            'CCDSTEMP',
            'SUNDIST',
            'SUNELEV',
            'MNTSTATE',
            'CHOMEHA',
            'CHOMEDEC',
            'IHUALM',
            'IHUALT',
            'IHUFOC',
        ),
        # these are also used by HAT Survey instruments
        fits_extra_keys=(
            'CMSERIAL',
            'EXPTIME',
            'MEXPTIME',
            'VERSION',
            'STNAME',
            'STVER',
            'STCOM',
            'OBSERVAT',
            'SITEID',
            'SITEVER',
            'SITELAT',
            'SITELONG',
            'SITEALT',
            'MTVER',
            'CMID',
            'CMVER',
            'TELID',
            'TELVER',
            'FOV',
            'BIASVER',
            'DARKVER',
            'FLATVER',
            'PROJID',
            'PROJTITL',
            'FILID',
            'FILTERS',
            'FOCUS',
            'IMAGETYP',
            'OBJECT',
            'JD',
            'MIDDATE',
            'MIDTIME',
            'PTVER',
            'HATP_SVN',
            # these are added by HATPI's PIPE-TrEx pipeline at various stages
            'PIPETREX',
            'MSTRBIAS',
            'MSTRDARK',
            'MSTRFLAT',
            'FRNGCORR',
            'RAWFRAME',
            'CMBCVTAG',
            'KERNSPEC',
            'CONVTRGT',
            'SUBTYPE',
            'REFHDTAG',
            'TGTHDTAG',
            'REFERIMG',
            'TARGTIMG',
            'COMBLIST',
            'NCOMBINE',
            'COMBNTYP',
            'MEDNTEMP',
            # these are added by FITSbits
            'FITSBITS',
            'PROCDT',
        ),
        extra_info_dict=None,
        writepickle=False,
        pickleprefix=None,
):
    '''This collects the following information about a frame:

    - all of the values in the fits_environ_keys from the FITS header
    - the overall median background value of the frame
    - the median background value of the frame in tiles over the frame
    - the MAD of the tiled background across the frame

    Parameters
    ----------

    fits : str or tuple of two elements
        If provided as a str, is the name of the FITS file to process. If
        provided as a tuple, this must contain the image to work on as an
        2D np.array in the first element and the image header as an
        astropy.io.fits.Header object in the second element.

    stats_only : bool
        If True, will return early with the image stats only. If False, will
        process headers and accompanying photometry info as well.

    trim_frame : bool
        If True, will trim the frame according to the header keys provided in
        the ``trim_headerkeys`` kwarg.

    trim_headerkeys : sequence of str
        The header keys to try one after the other when trying to get the
        TRIMSEC/DATASEC from the FITS header.

    calibrated_imagetype_headerkey : str
        This gives the IMAGETYP FITS header key value associated with the
        calibrated image. NOTE: PIPE-TrEx will use IMAGETYPE =
        'calibratedobject' for any calibrated object image that it produces.

    fits_ext : int
        Sets the extension of the FITS file to process. If None, will choose the
        first image-like extension automatically.

    fits_racenter_key : str
        The header key to use to extract the target field center right ascension
        from the FITS.

    fits_racenter_unit : {'deg', 'hms', 'hr'}
        The unit of the right ascension value in the FITS header.

    fits_declcenter_key : str
        The header key to use to extract the target field center declination
        from the FITS.

    fits_declcenter_unit : {'deg', 'dms'}
        The unit of the declination value in the FITS header.

    fits_environ_keys : list or tuple
        Sets the keys to extract from the FITS header for use as environmental
        indicators for the FITS frame. These can later be selected on when
        searching for good frames, etc.

    fits_extra_keys : list or tuple
        Sets extra keys to extract from the FITS header.

    fits_background_mediandiffbelow : float, default: 0.0
        Sets the value to be subtracted from the median value to form the final
        background value. This effectively lowers the background value to a
        value below the median. Useful if the median of an image is not
        considered as a valid measure of its background and a lower value must
        be adopted.

    fits_background_tilesize : int, default: 256
        Sets the x and y tilesize to use when generating estimates of the median
        background level of the frame and the associated MAD.

    extra_info_dict : dict or None
        An arbitrary dict of information to be added to the output dict and
        pickle.

    writepickle : bool, default: False
        Sets if a pickle should be created alongside the FITS file containing
        all the info extracted from the FITS, fistar, and fiphot.

    pickleprefix : str or None
        This is a prefix to attach to the pickle file name. Useful to indicate
        that the pickle was produced in a specific processing stage, for
        example.

    Returns
    -------

    dict
        Returns a dict with all of the values as key:val pairs and paths to the
        FITS. This can be used to insert stuff into a database easily.

    '''

    fits_info = {}

    # read in the FITS image
    if isinstance(fits, str):
        img, hdr = read_fits(fits, ext=fits_ext)
    elif isinstance(fits, tuple):
        img, hdr = fits

    # trim the image if told to do so
    if trim_frame is True and trim_headerkeys is not None:

        img = trim_image(img, hdr, trimkeys=trim_headerkeys)
        fits_info['trimmed'] = True

    else:
        fits_info['trimmed'] = False

    fits_info['shape'] = img.shape

    # check if the image is a raw image
    if (hdr['IMAGETYP'] == calibrated_imagetype_headerkey):
        fits_info['image_is_calibrated'] = True
    else:
        fits_info['image_is_calibrated'] = False

    #
    # 1. calculate image stats
    #

    # calculate the overall background
    fits_info['overall_background_median'] = extract_img_background(
        fits,
        ext=fits_ext,
        median_diffbelow=fits_background_mediandiffbelow
    )

    # get the overall stats
    fits_info['overall_mean'] = np.nanmean(img)
    fits_info['overall_stdev'] = np.nanstd(img)
    fits_info['overall_median'] = np.nanmedian(img)
    fits_info['overall_mad'] = np.nanmedian(
        np.abs(img - fits_info['overall_median'])
    )
    fits_info['overall_stdev_from_mad'] = 1.4826*fits_info['overall_mad']

    # tile the frame
    tiles, ntiles_x, ntiles_y = tile_frame(
        img,
        tilesize=fits_background_tilesize
    )
    fits_info['ntiles_x'] = ntiles_x
    fits_info['ntiles_y'] = ntiles_y

    # get the stats for each tile
    tile_means = [np.nanmean(x) for x in tiles]
    tile_medians = [np.nanmedian(x) for x in tiles]
    tile_mads = [np.nanmedian(np.abs(x - np.nanmedian(x))) for x in tiles]
    tile_stdevs = [np.nanstd(x) for x in tiles]
    tile_maxes = [np.nanmax(x) for x in tiles]
    tile_percentiles = [np.nanpercentile(x,(5,95)) for x in tiles]
    tile_ptps = [np.ptp(x) for x in tiles]

    tile_medians_mean = np.nanmean(tile_medians)
    tile_medians_stdev = np.nanstd(tile_medians)
    fits_info['tile_medians_mean'] = tile_medians_mean
    fits_info['tile_medians_stdev'] = tile_medians_stdev

    tile_medians_median = np.nanmedian(tile_medians)
    tile_medians_mad = np.nanmedian(np.abs(tile_medians - tile_medians_median))
    fits_info['tile_medians_median'] = tile_medians_median
    fits_info['tile_medians_mad'] = tile_medians_mad

    # these are per-tile values of each item
    fits_info['tile_means'] = np.reshape(np.array(tile_means),
                                         (ntiles_y, ntiles_x))
    fits_info['tile_medians'] = np.reshape(np.array(tile_medians),
                                           (ntiles_y, ntiles_x))

    fits_info['tile_mads'] = np.reshape(np.array(tile_mads),
                                        (ntiles_y, ntiles_x))
    fits_info['tile_maxes'] = np.reshape(np.array(tile_maxes),
                                         (ntiles_y, ntiles_x))
    fits_info['tile_stdevs'] = np.reshape(np.array(tile_stdevs),
                                          (ntiles_y, ntiles_x))
    fits_info['tile_percentiles'] = np.reshape(np.array(tile_percentiles),
                                               (ntiles_y, ntiles_x, 2))
    fits_info['tile_ptps'] = np.reshape(np.array(tile_ptps),
                                        (ntiles_y, ntiles_x))

    if stats_only:
        return fits_info

    #
    # 2. get the header keys out of the image
    #

    # get the environ keys out of the header
    for key in fits_environ_keys:
        if key in hdr:
            val = hdr[key]
            fits_info[key.lower()] = val
        else:
            fits_info[key.lower()] = None

    # get the extra keys out of the header
    if isinstance(fits_extra_keys, (list, tuple)):
        for key in fits_extra_keys:
            if key in hdr:
                val = hdr[key]
                fits_info[key.lower()] = val
            else:
                fits_info[key.lower()] = None

    # get the image center keys out of the header
    center_ra = hdr[fits_racenter_key]
    center_decl = hdr[fits_declcenter_key]

    if fits_racenter_unit == 'hr':
        center_ra = center_ra/24.0 * 360.0
    elif fits_racenter_unit == 'hms':
        center_ra = hms_str_to_decimal(center_ra)

    if fits_declcenter_unit == 'dms':
        center_decl = dms_str_to_decimal(center_decl)

    fits_info['center_ra'] = center_ra
    fits_info['center_decl'] = center_decl

    # write the kwargs to the output dict
    fits_info['kwargs'] = {
        'fits':(os.path.abspath(fits)
                if isinstance(fits, str) else 'from ndarray'),
        'fits_ext':fits_ext,
        'fits_racenter_key':fits_racenter_key,
        'fits_racenter_unit':fits_racenter_unit,
        'fits_declcenter_key':fits_declcenter_key,
        'fits_declenter_unit':fits_declcenter_unit,
        'fits_environ_keys':fits_environ_keys,
        'fits_extra_keys':fits_extra_keys,
        'fits_background_mediandiffbelow':fits_background_mediandiffbelow,
        'fits_background_tilesize':fits_background_tilesize,
    }

    if extra_info_dict is not None and isinstance(extra_info_dict, dict):
        fits_info.update(extra_info_dict)

    if writepickle:

        outpicklef = os.path.join(
            os.path.abspath(os.path.dirname(fits)),
            '%s-imginfo%s.pkl' % (
                clean_fname(fits, basename=True),
                '-%s' % pickleprefix if pickleprefix else ''
            )
        )
        with open(outpicklef, 'wb') as outfd:
            pickle.dump(fits_info, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    return fits_info


def _parallel_stats_worker(task):
    '''
    This is a parallel worker for getting frame stats.

    '''

    fits, kwargs = task

    try:

        stats = frame_source_info(fits, **kwargs)
        LOGINFO('Stats for %s OK' % fits)
        return stats

    except Exception:

        LOGEXCEPTION("Could not get stats for %s" % fits)
        return None


def parallel_frame_stats(fits_list,
                         nworkers=NCPUS,
                         maxworkertasks=1000,
                         **worker_kwargs):
    '''
    This runs frame stats in parallel for list of FITS.

    '''

    tasks = [(fits, worker_kwargs) for fits in fits_list]

    # initialize the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(_parallel_stats_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in zip(fits_list, results)}

    return resultdict

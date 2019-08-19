#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This contains code to detect streaks caused by satellites and meteors.

'''

#############
## LOGGING ##
#############

import logging
from pipetrex import log_sub, log_fmt, log_date_fmt

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

import gzip
import os.path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import transform, morphology, filters

from ..drivers.oprun import with_history

from .operations import read_fits, clipped_linscale_img, mask_image
from .export import nparray_to_full_jpeg

from pipetrex.photometry import srcextract


#######################
## UTILITY FUNCTIONS ##
#######################

def detect_streaks(
        fits,
        fistar_exe='fistar',
        fistar_timeout_sec=10.0,
        ccdextent='0:2047,0:2047',
        fluxthreshold=1000,
        edge_margin_px=50,
        fwhm_multiplier=1.5,
        saturation_adu=60000,
        saturation_frac=0.85,
        threshold_func=filters.threshold_triangle,
        threshold_func_kwargs=None,
        min_line_length=250,
        min_line_gap=75,
        min_lines_required=9,
        output_scalefunc=clipped_linscale_img,
        output_scalefunc_params=None,
        verbose=True,
        use_historydb_url=None,
        use_run_options=None,
):
    '''This detects streaks in an image.

    The method is:

    - Mask all pixels that have values >= saturation_frac*saturation_adu with
      the overall median value of the image.

    - Run a fistar source extraction to find sources on the image and note their
      FWHM and x,y positions.

    - Mask all of the sources detected using a box size of fwhm_multiplier*fwhm
      of each source and set the mask value to the overall median of the image.

    - Run a dilation operation that expands bright areas, hopefully enlarging
      the streak lines enough to be able to find them.

    - Threshold the image using a filter function to binarize the bright and
      dark areas of the image, hopefully making the streaks stand out easily.

    - Use the edge definitions produced by the previous step to run a
      probabilistic Hough transform to find lines on the image. If this finds
      more than ~10 lines, the image probably has streaks on it.

    You may want to look at the `thresholding functions in scikit-image
    <http://scikit-image.org/docs/dev/auto_examples/applications/plot_thresholding.html#sphx-glr-auto-examples-applications-plot-thresholding-py>`_
    to choose a good one for your images. The `threshold_triangle
    <http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_triangle>`_
    function appears to work well for HATPI images.

    Parameters
    ----------

    fits : str
        The FITS image file to process.

    fistar_exe : str
        The path to the `fistar` executable.

    fistar_timeout_sec : float
        The amount of time in seconds to wait for `fistar` to complete its work
        before timing out with a failure.

    ccdextent : str
        A string indicating the extent of the input image to process and extract
        sources from.

    fluxthreshold : float
        The minimum flux required for a source to extract it.

    edge_margin_px : int
        Removes image pixels within this amount of pixels from the image. This
        can help remove false detections of streaks on images that have been
        warped so that their edges are no longer the same as the actual CCD
        extent.

    fwhm_multiplier : float
        The multiplier applied to the FWHM of each detected source in order to
        expand its footprint and mask it out.

    saturation_adu : int
        The ADU level at which the image is saturated.

    saturation_frac : float
        The fraction of the ADU level at which pixel values will be masked.

    threshold_func : Python function
        The filter function to apply to segment the image into bright and dark
        binary values (to make streaks stand out). This is one of the functions
        in the scikit-image `filters` module or something that has the same
        signature as them.

    threshold_func_kwargs : dict or None
        Any kwargs to pass in to the `threshold_func` can be specified here as a
        dict.

    min_line_length : int
        The minimum length in pixels a line must have for it to be considered
        valid by the Hough transform function.

    min_line_gap : int
        The maximum length of a gap in pixels that can exist to consider
        detected line segments part of a single line.

    min_lines_required : int
        The minimum number of detected lines required to mark this frame as
        'streaky'.

    output_scalefunc : Python function
        The scaling function to apply to convert FITS pixel values to output
        pixel values suitable for visualization.

    output_scalefunc_kwargs : dict or None
        Any optional kwargs to pass into the `output_scalefunc` as a dict.

    verbose : bool
        If True, indicates progress.

    Returns
    -------

    dict
        This returns a dict with the number of lines found, the start and end
        coordinates of each line, and the paths to the JPEGS produced while
        processing the image. The dict key 'streaky' indicates if this function
        thought that the input had streaks in it.

    '''

    # run fistar on the FITS with gain = 1
    if use_historydb_url is not None:
        fistarf = srcextract.fistar_extract_sources_with_history(
            use_historydb_url,
            fits,
            fistar_exe=fistar_exe,
            fistar_timeout_sec=fistar_timeout_sec,
            ccdextent=ccdextent,
            fluxthreshold=fluxthreshold,
            ccd_gain=1.0,
            run_options=use_run_options
        )
    else:
        fistarf = srcextract.fistar_extract_sources(
            fits,
            fistar_exe=fistar_exe,
            fistar_timeout_sec=fistar_timeout_sec,
            ccdextent=ccdextent,
            fluxthreshold=fluxthreshold,
            ccd_gain=1.0,
        )

    # get the FWHM and ellipticities for all detected sources
    with gzip.open(fistarf,'rb') as infd:
        fistar = np.genfromtxt(infd,
                               usecols=(1,2,3,4,10,11),
                               names=('x','y','bgv','flux','fwhm','ellip'),
                               dtype='f8,f8,f8,f8,f8,f8',
                               comments='#')

    # load the image
    img, hdr = read_fits(fits)
    img_median = np.nanmedian(img)

    fits_basedir = os.path.dirname(fits)
    fits_basename = os.path.basename(fits).replace(
        '.fits',''
    ).replace(
        '.gz',''
    ).replace(
        'fz',''
    ).replace(
        '-warped',''
    ).replace(
        '-convolved',''
    )

    if verbose:
        LOGINFO('Masking saturated pixels...')

    # remove things within frac of saturation and take care of their leaks as
    # well
    masked, mask = mask_image(
        img,
        mask_saturated_min_adu=saturation_frac*saturation_adu,
        mask_direct_with=img_median,
        # use aggressive masking to kill saturated stars
        mask_saturated_growmaxpx=(20,20),
        inplace=True
    )

    if verbose:
        LOGINFO('%s sources to remove...' % fistar['x'].size)

    # set the flux values at the source locations to background level
    for x, y, fwhm in zip(fistar['x'],
                          fistar['y'],
                          fistar['fwhm']):

        stamp_size = fwhm*fwhm_multiplier
        xmin, xmax = int(x - stamp_size), int(x + stamp_size)
        ymin, ymax = int(y - stamp_size), int(y + stamp_size)
        if xmin < 0:
            xmin = 0
        if xmax > img.shape[1]:
            xmax = img.shape[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > img.shape[0]:
            ymax = img.shape[0] - 1

        img[ymin:ymax, xmin:xmax] = img_median

    srcsat_jpg = os.path.join(fits_basedir,
                              'streaks-srcsat-%s.jpg' % fits_basename)
    nparray_to_full_jpeg(
        img,
        srcsat_jpg,
        scale_func=output_scalefunc,
        scale_func_params=output_scalefunc_params
    )

    scaled_img = img[edge_margin_px:-edge_margin_px,
                     edge_margin_px:-edge_margin_px]

    if verbose:
        LOGINFO('Dilating edges...')

    dilated_jpg = os.path.join(fits_basedir,
                               'streaks-dilated-%s.jpg' % fits_basename)
    dilated = morphology.dilation(scaled_img)
    nparray_to_full_jpeg(
        dilated,
        dilated_jpg,
        scale_func=output_scalefunc,
        scale_func_params=output_scalefunc_params
    )

    if verbose:
        LOGINFO('Thresholding edges...')

    # do the thresholding to get a binary image compose of foreground and
    # background pixels
    if not threshold_func_kwargs:
        threshold_func_kwargs = {}
    thresholded = dilated > threshold_func(
        dilated,
        **threshold_func_kwargs)
    thresholded_jpg = os.path.join(fits_basedir,
                                   'streaks-thresholded-%s.jpg' % fits_basename)
    nparray_to_full_jpeg(
        thresholded,
        thresholded_jpg,
        scale_func=output_scalefunc,
        scale_func_params=output_scalefunc_params
    )

    if verbose:
        LOGINFO('Running Hough transform on detected edges...')

    # run the probabilistic hough transform on the thresholded image
    lines = transform.probabilistic_hough_line(
        thresholded,
        line_length=min_line_length,
        line_gap=min_line_gap,
    )

    scaled_image = output_scalefunc(thresholded)
    plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100))
    plt.imshow(scaled_image, cmap='gray', origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)

    LOGINFO('Streaks detected in image: %s' % len(lines))

    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]), 'r-', linewidth=10.0)

    lines_jpg = os.path.join(fits_basedir,
                             'streaks-detected-%s.jpg' % fits_basename)

    plt.savefig(
        lines_jpg,
        dpi=100,
        bbox_inches='tight',
        pad_inches=0.0
    )
    plt.close('all')

    return {
        'fits':fits,
        'streaky':True if len(lines) >= min_lines_required else False,
        'nlines':len(lines),
        'lines':lines,
        'srcsat_jpg': os.path.abspath(srcsat_jpg),
        'dilated_jpg': os.path.abspath(dilated_jpg),
        'thresholded_jpg':os.path.abspath(thresholded_jpg),
        'lines_jpg':os.path.abspath(lines_jpg),
    }


def detect_streaks_with_history(
        use_historydb_url,
        fits,
        fistar_exe='fistar',
        fistar_timeout_sec=10.0,
        ccdextent='0:2047,0:2047',
        fluxthreshold=1000,
        edge_margin_px=50,
        fwhm_multiplier=1.5,
        saturation_adu=60000,
        saturation_frac=0.85,
        threshold_func=filters.threshold_triangle,
        threshold_func_kwargs=None,
        min_line_length=250,
        min_line_gap=75,
        min_lines_required=9,
        output_scalefunc=clipped_linscale_img,
        output_scalefunc_params=None,
        verbose=True,
        overwrite=False,
        run_options=None,
):
    '''This detects streaks in an image and is history-enabled.

    '''

    return with_history(
        use_historydb_url,
        'streak detection',
        fits,
        overwrite,
        ['streak_info'],
        detect_streaks,
        fits,
        fistar_exe=fistar_exe,
        fistar_timeout_sec=fistar_timeout_sec,
        ccdextent=ccdextent,
        fluxthreshold=fluxthreshold,
        edge_margin_px=edge_margin_px,
        fwhm_multiplier=fwhm_multiplier,
        saturation_adu=saturation_adu,
        saturation_frac=saturation_frac,
        threshold_func=threshold_func,
        threshold_func_kwargs=threshold_func_kwargs,
        min_line_length=min_line_length,
        min_line_gap=min_line_gap,
        min_lines_required=min_lines_required,
        output_scalefunc=output_scalefunc,
        output_scalefunc_params=output_scalefunc_params,
        verbose=verbose,
        use_historydb_url=use_historydb_url,
        use_run_options=run_options,
        run_options=run_options,
    )

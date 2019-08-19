#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This module contains functions to export FITS images to other formats.

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

import os
import os.path
import subprocess

import multiprocessing
import sys

# Ref: https://bugs.python.org/issue33725
# TLDR; Apple is trash at UNIX
if sys.platform == 'darwin':
    mp = multiprocessing.get_context('forkserver')
else:
    mp = multiprocessing

import glob

import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from ._extractors import clean_fname
from . import operations

from astropy import wcs

############
## CONFIG ##
############

NCPUS = mp.cpu_count()

# get the ImageFont
fontpath = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')

# load the font
if os.path.exists(fontpath):
    fontsmall = ImageFont.truetype(fontpath, 12)
    fontnormal = ImageFont.truetype(fontpath, 20)
    fontlarge = ImageFont.truetype(fontpath, 28)
else:
    print('could not find bundled '
          'DejaVu Sans font, using ugly defaults...')
    fontsmall = ImageFont.load_default()
    fontnormal = ImageFont.load_default()
    fontlarge = ImageFont.load_default()


##################
## IMAGE STAMPS ##
##################

def img_to_stamps(img,
                  stampsize=256):
    '''Generate stamps for an image.

    The stamps are generated for the center, the corners, and the middle of the
    sides of the frame. This is useful to monitor image quality and star shapes
    as a function of position on the frame.

    Parameters
    ----------

    img : np.array of 2 dimensions
        The input image to process.

    stampsize : int
        The size in pixels of each stamp. Stamps are square so this describes
        both width and height.

    Returns
    -------

    dict
        Returns a dict of the form::

            {
                'topleft': 2D np.array cutout,
                'topcenter': 2D np.array cutout,
                'topright': 2D np.array cutout,
                'midleft':2D np.array cutout,
                'midcenter': 2D np.array cutout,
                'midright': 2D np.array cutout,
                'bottomleft': 2D np.array cutout,
                'bottomcenter': 2D np.array cutout,
                'bottomright': 2D np.array cutout
            }

    '''

    imgsizex, imgsizey = img.shape
    xstampsize, ystampsize = stampsize, stampsize

    # get the total number of possible stamps
    n_possible_xstamps = imgsizex/float(xstampsize)
    n_possible_ystamps = imgsizey/float(ystampsize)

    # if we can actually make stamps, then go ahead
    if (n_possible_xstamps >= 3) and (n_possible_ystamps >= 3):

        topleft = img[:xstampsize,:ystampsize]
        topcenter = img[
            int(imgsizex/2-xstampsize/2):int(imgsizex/2+xstampsize/2),
            :ystampsize
        ]
        topright = img[imgsizex-xstampsize:,:ystampsize]
        midleft = img[
            :xstampsize,
            int(imgsizey/2-ystampsize/2):int(imgsizey/2+ystampsize/2)
        ]
        midcenter = img[
            int(imgsizex/2-xstampsize/2):int(imgsizex/2+xstampsize/2),
            int(imgsizey/2-ystampsize/2):int(imgsizey/2+ystampsize/2)
        ]
        midright = img[
            imgsizex-xstampsize:,
            int(imgsizey/2-ystampsize/2):int(imgsizey/2+ystampsize/2)
        ]
        bottomleft = img[:xstampsize,imgsizey-ystampsize:]
        bottomcenter = img[
            int(imgsizex/2-xstampsize/2):int(imgsizex/2+xstampsize/2),
            imgsizey-ystampsize:
        ]
        bottomright = img[-xstampsize:,-ystampsize:]

        return {
            'topleft':topleft,
            'topcenter':topcenter,
            'topright':topright,
            'midleft':midleft,
            'midcenter':midcenter,
            'midright':midright,
            'bottomleft':bottomleft,
            'bottomcenter':bottomcenter,
            'bottomright':bottomright
        }

    else:
        LOGERROR('stampsize is too large for this image')
        return None


def fits_to_stamps(fits_image,
                   outfile,
                   fits_extension=None,
                   trimkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
                   scale_func=operations.clipped_linscale_img,
                   scale_func_params=None,
                   stampsize=128,
                   separatorwidth=1,
                   annotate=True,
                   fits_jdsrc=None,
                   fits_jdkey='JD',
                   frame_time=None):
    '''This turns an FITS image into a scaled version, stamps it, and returns an
    PNG/JPG file.

    '''

    compressed_ext = operations.compressed_fits_ext(fits_image)

    if fits_extension is None and compressed_ext:
        img, hdr = operations.read_fits(fits_image,
                                        ext=compressed_ext[0])
    elif (fits_extension is not None):
        img, hdr = operations.read_fits(fits_image, ext=fits_extension)
    else:
        img, hdr = operations.read_fits(fits_image)

    trimmed_img = operations.trim_image(img, hdr, trimkeys=trimkeys)

    if isinstance(scale_func_params, dict):
        scaled_img = scale_func(trimmed_img, **scale_func_params)
    else:
        scaled_img = scale_func(trimmed_img)

    image_stamps = img_to_stamps(scaled_img, stampsize=stampsize)

    toprow_xsize, toprow_ysize = image_stamps['topright'].shape
    toprow_separr = np.array([[255.0]*separatorwidth]*toprow_ysize)

    # build stacks

    topleft = image_stamps['topleft']
    midleft = image_stamps['midleft']
    bottomleft = image_stamps['bottomleft']

    topcenter = image_stamps['topcenter']
    midcenter = image_stamps['midcenter']
    bottomcenter = image_stamps['bottomcenter']

    topright = image_stamps['topright']
    midright = image_stamps['midright']
    bottomright = image_stamps['bottomright']

    toprow_stamp = np.hstack((topleft,
                              toprow_separr,
                              midleft,
                              toprow_separr,
                              bottomleft))

    midrow_xsize, midrow_ysize = midright.shape
    midrow_separr = np.array([[255.0]*separatorwidth]*midrow_ysize)

    # similarly, these should be midleft, midcenter, midright
    midrow_stamp = np.hstack((topcenter,
                              midrow_separr,
                              midcenter,
                              midrow_separr,
                              bottomcenter))

    bottomrow_xsize, bottomrow_ysize = bottomright.shape
    bottomrow_ysize = bottomright.shape[1]
    bottomrow_separr = np.array([[255.0]*separatorwidth]*bottomrow_ysize)

    # similarly, these should be bottomleft, bottomcenter, bottomright
    bottomrow_stamp = np.hstack((topright,
                                 bottomrow_separr,
                                 midright,
                                 bottomrow_separr,
                                 bottomright))

    full_stamp = np.vstack(
        (toprow_stamp,
         np.array([255.0]*(toprow_xsize*3 + separatorwidth*2)),
         midrow_stamp,
         np.array([255.0]*(midrow_xsize*3 + separatorwidth*2)),
         bottomrow_stamp)
    )

    full_stamp = np.flipud(full_stamp)
    pillow_image = Image.fromarray(full_stamp)
    pillow_image = pillow_image.convert('L')

    # annotate the image if told to do so
    if annotate:

        draw = ImageDraw.Draw(pillow_image)

        # if we're supposed to use another file for the JD source, do so
        # this is useful for subtracted images
        if fits_jdsrc is not None and os.path.exists(fits_jdsrc):
            framejd = operations.get_header_keyword(fits_jdsrc, fits_jdkey)

        elif frame_time is not None:
            framejd = frame_time

        else:
            framejd = hdr[fits_jdkey] if fits_jdkey in hdr else None

        if framejd is not None:

            timeannotation = '%.5f' % framejd
            draw.text((5, pillow_image.size[1] - 15),
                      timeannotation,
                      font=fontsmall,
                      fill=255)

        # draw the image basename
        basename_annotation = os.path.splitext(
            clean_fname(fits_image, basename=True)
        )[0]

        draw.text((5, 5),
                  basename_annotation,
                  font=fontsmall,
                  fill=255)

        del draw

    pillow_image.save(outfile)

    return outfile


def parallel_fits_stamp_worker(task):
    '''
    This is a parallel worker for the FITS to zscaled stamps process.

    '''

    fits, options = task

    try:

        outpngpath = '%s-stamps-3x3.png' % clean_fname(fits)
        donepng = fits_to_stamps(
            fits,
            outpngpath,
            **options
        )

        LOGINFO('%s -> %s OK' % (fits, donepng))
        return donepng

    except Exception:

        LOGEXCEPTION('could not convert %s to stamp PNG' % fits)
        return None


def parallel_fitslist_to_stamps(fitslist,
                                ext=None,
                                trimkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
                                stampsize=128,
                                separatorwidth=1,
                                annotate=True,
                                nworkers=NCPUS,
                                maxworkertasks=1000):
    '''
    This drives parallel execution of FITS to stamps.

    '''
    tasks = [(x, {'fits_extension':ext,
                  'trimkeys':trimkeys,
                  'stampsize':stampsize,
                  'annotate':annotate,
                  'separatorwidth':separatorwidth}) for x in
             fitslist]

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
    results = pool.map(parallel_fits_stamp_worker, tasks)
    pool.close()
    pool.join()

    return results


def parallel_fitsdir_to_stamps(fitsdir,
                               fitsglob,
                               ext=None,
                               trimkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
                               stampsize=128,
                               separatorwidth=1,
                               annotate=True,
                               nworkers=NCPUS,
                               maxworkertasks=1000):
    '''
    This makes stamps for a directory of FITS files.

    '''

    fitslist = glob.glob(os.path.join(fitsdir, fitsglob))

    return parallel_fitslist_to_stamps(
        fitslist,
        ext=ext,
        trimkeys=trimkeys,
        stampsize=stampsize,
        separatorwidth=separatorwidth,
        annotate=annotate,
        nworkers=nworkers,
        maxworkertasks=maxworkertasks
    )


######################
## FULL FRAME JPEGS ##
######################

def nparray_to_full_jpeg(
        array,
        outfile,
        scale_func=operations.clipped_linscale_img,
        scale_func_params=None,
        flip=True,
        resize=False,
        resizefrac=None,
):
    '''
    This writes a numpy array to a JPEG.

    '''

    # scale the image if requested
    if scale_func is not None and isinstance(scale_func_params, dict):
        scaled_img = scale_func(array,**scale_func_params)
    elif scale_func is not None:
        scaled_img = scale_func(array)
    else:
        scaled_img = array

    # flip the image if requested
    if flip is True:
        scaled_img = np.flipud(scaled_img)

    # convert to PIL.Image
    pillow_image = Image.fromarray(scaled_img)
    pillow_image = pillow_image.convert('L')

    # resize the image if requested
    if resize and resizefrac is not None and resizefrac > 0:
        pillow_image = pillow_image.resize(
            (int(scaled_img.shape[1]*resizefrac),
             int(scaled_img.shape[0]*resizefrac)),
            Image.BICUBIC
        )

    # save the file and return
    pillow_image.save(outfile, optimize=True, quality=85)
    return os.path.abspath(outfile)


def fits_to_full_jpeg(
        fits_image,
        out_fname=None,
        ext=None,
        trim=True,
        trimkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
        scale_func=operations.clipped_linscale_img,
        scale_func_params=None,
        flip=True,
        resize=False,
        resizefrac=None,
        annotate=True,
        fits_jdsrc=None,
        fits_jdkey='JD',
        frame_time=None,
        fits_imagetype_key='IMAGETYP',
        fits_exptime_key='EXPTIME',
        fits_filters_key='FILTERS',
        fits_project_key='PROJID',
        fits_object_key='OBJECT',
):
    '''This converts a FITS image to a full frame JPEG.

    The default scaling function is operations.clipped_linscale_img, mostly
    because it's faster than operations.zscale_img.

    '''

    # handle .fz and non-zero extension FITS reading
    compressed_ext = operations.compressed_fits_ext(fits_image)
    if ext is None and compressed_ext:
        img, hdr = operations.read_fits(fits_image,
                                        ext=compressed_ext[0])
    elif (ext is not None):
        img, hdr = operations.read_fits(fits_image, ext=ext)
    else:
        img, hdr = operations.read_fits(fits_image)

    # trim the image if requested
    if trim:
        trimmed_img = operations.trim_image(img, hdr, trimkeys=trimkeys)
    else:
        trimmed_img = img

    # scale the image if requested
    if scale_func is not None and isinstance(scale_func_params, dict):
        scaled_img = scale_func(trimmed_img,**scale_func_params)
    elif scale_func is not None:
        scaled_img = scale_func(trimmed_img)
    else:
        scaled_img = trimmed_img

    # flip the image if requested
    if flip is True:
        scaled_img = np.flipud(scaled_img)

    # convert to PIL.Image
    pillow_image = Image.fromarray(scaled_img)
    pillow_image = pillow_image.convert('L')

    # resize the image if requested
    if resize and resizefrac is not None and resizefrac > 0:
        pillow_image = pillow_image.resize(
            (int(scaled_img.shape[1]*resizefrac),
             int(scaled_img.shape[0]*resizefrac)),
            Image.BICUBIC
        )

    # annotate the image if told to do so
    if annotate:

        draw = ImageDraw.Draw(pillow_image)
        annotation = "%s: %s - %s - %s - proj%s - %s" % (
            clean_fname(fits_image, basename=True),
            (hdr[fits_imagetype_key].lower()
             if fits_imagetype_key in hdr else 'imgtype_unknown'),
            (hdr[fits_exptime_key]
             if fits_exptime_key in hdr else 'exptime_unknown'),
            (hdr[fits_filters_key].replace('+','') if
             fits_filters_key in hdr else 'filt_unknown'),
            (hdr[fits_project_key]
             if fits_project_key in hdr else '_unknown'),
            hdr[fits_object_key] if fits_object_key in hdr else 'obj_unknown'
        )
        draw.text((10,10),
                  annotation,
                  font=fontnormal,
                  fill=255)

        # now add the time as well

        # if we're supposed to use another file for the JD source, do so
        # this is useful for subtracted images
        if fits_jdsrc is not None and os.path.exists(fits_jdsrc):
            framejd = operations.get_header_keyword(fits_jdsrc, fits_jdkey)

        elif frame_time is not None:
            framejd = frame_time

        else:
            framejd = hdr[fits_jdkey] if fits_jdkey in hdr else None

        if framejd is not None:

            timeannotation = '%.5f' % framejd
            draw.text((10, pillow_image.size[1] - 40),
                      timeannotation,
                      font=fontlarge,
                      fill=255)

        del draw

    # finally, generate the output file name if None is given
    if not out_fname:

        out_fname = '%s-%s-%s-%s-proj%s-%s.jpg' % (
            fits_image.replace('.fits','').replace('.fz',''),
            (hdr[fits_imagetype_key].lower()
             if fits_imagetype_key in hdr else 'imgtype_unknown'),
            (hdr[fits_exptime_key]
             if fits_exptime_key in hdr else 'exptime_unknown'),
            (hdr[fits_filters_key].replace('+','') if
             fits_filters_key in hdr else 'filt_unknown'),
            hdr[fits_project_key] if fits_project_key in hdr else '_unknown',
            hdr[fits_object_key] if fits_object_key in hdr else 'obj_unknown'
        )

    # save the file and return
    pillow_image.save(out_fname, optimize=True, quality=85)
    return os.path.abspath(out_fname)


def parallel_jpeg_worker(task):
    '''
    This wraps imageutils.fits_to_full_jpeg.

    task[0] = FITS path
    task[1] = {'ext', 'resize', 'flip', 'outsizex', 'outsizey'}

    '''

    try:
        return fits_to_full_jpeg(task[0], **task[1])
    except Exception:
        LOGEXCEPTION('failed to make JPEG for %s' % (task[0],))
        return None


def parallel_frame_jpegs(
        infits,
        fitsglob='*.fits',
        outf_dir=None,
        outf_extension='jpg',
        outf_postfix=None,
        ext=None,
        trim=True,
        trimkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
        scale_func=operations.clipped_linscale_img,
        scale_func_params=None,
        flip=True,
        resize=False,
        resizefrac=None,
        annotate=True,
        fits_jdsrc=None,
        fits_jdkey='JD',
        frame_time=None,
        fits_imagetype_key='IMAGETYP',
        fits_exptime_key='EXPTIME',
        fits_filters_key='FILTERS',
        fits_project_key='PROJID',
        fits_object_key='OBJECT',
        nworkers=NCPUS,
        maxworkertasks=1000
):
    '''
    This makes JPEGs out of the frames in fitsdir.

    '''

    # initialize the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    if isinstance(infits,str):
        fitslist = sorted(glob.glob(os.path.join(os.path.abspath(infits),
                                                 fitsglob)))

    elif isinstance(infits, list):
        fitslist = infits

    if outf_postfix is None:
        outf_postfix = ''
    else:
        outf_postfix = '-%s' % outf_postfix

    out_flist = ['%s%s.%s' % (clean_fname(x),
                              outf_postfix,
                              outf_extension) for x in fitslist]

    if outf_dir is not None:

        if not os.path.exists(outf_dir):
            os.makedirs(outf_dir)

        out_flist = [os.path.join(outf_dir, x) for x in out_flist]

    # only make JPEGs that don't yet exist
    work_on_flist = []
    work_on_outlist = []
    for f, o in zip(fitslist, out_flist):
        if not os.path.exists(o):
            work_on_flist.append(f)
            work_on_outlist.append(o)

    tasks = [
        (x,{'ext':ext,
            'out_fname':y,
            'trim':trim,
            'scale_func':scale_func,
            'scale_func_params':scale_func_params,
            'trimkeys':trimkeys,
            'resize':resize,
            'flip':flip,
            'resize':resize,
            'resizefrac':resizefrac,
            'annotate':annotate,
            'fits_imagetype_key':fits_imagetype_key,
            'fits_exptime_key':fits_exptime_key,
            'fits_filters_key':fits_filters_key,
            'fits_project_key':fits_project_key,
            'fits_object_key':fits_object_key,
            'fits_jdsrc':fits_jdsrc,
            'fits_jdkey':fits_jdkey,
            'frame_time':frame_time})
        for x,y in zip(work_on_flist, work_on_outlist)
    ]

    # fire up the pool of workers
    pool.map(parallel_jpeg_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for x,y in zip(fitslist, out_flist) if os.path.exists(y)}
    return resultdict


#################################
## FITS TO SUB-IMAGE BOX JPEGS ##
#################################

def fits_radecbox_to_jpeg(
        fits_image,
        box,
        boxtype='center',
        wcsfrom=None,
        out_fname=None,
        ext=None,
        trim=True,
        trimkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
        scale_func=operations.zscale_image,
        scale_func_params=None,
        flip=True,
        annotate=True,
        fits_imagetype_key='IMAGETYP',
        fits_exptime_key='EXPTIME',
        fits_filters_key='FILTERS',
        fits_project_key='PROJID',
        fits_object_key='OBJECT',
        fits_jdsrc=None,
        fits_jdkey='JD',
        frame_time=None,
):
    '''
    This converts an radec box inside a FITS image to a JPEG.

    boxtype is either 'center' or 'bounds'.

    - if boxtype == 'bounds', then box is: [rmin, rmax, dmin, dmax]
    - if boxtype == 'center', then box is: [rcenter, dcenter, rwidth, dwidth]

    wcsfrom indicates where the WCS for the image comes from:

    - None, take the WCS from the image itself
    - a path to a file, take the WCS from the specified file.

    '''

    # handle .fz and non-zero extension FITS reading
    compressed_ext = operations.compressed_fits_ext(fits_image)

    if ext is None and compressed_ext:
        img, hdr = operations.read_fits(fits_image,
                                        ext=compressed_ext[0])

    elif (ext is not None):
        img, hdr = operations.read_fits(fits_image, ext=ext)
    else:
        img, hdr = operations.read_fits(fits_image)

    # get the WCS header
    try:
        if wcsfrom and os.path.exists(wcsfrom):
            w = wcs.WCS(wcsfrom)
        else:
            w = wcs.WCS(hdr)

    except Exception:

        LOGEXCEPTION("no WCS found for FITS: %s, can't continue" % fits_image)
        return None

    # convert the radec box into a box in pixel space
    if boxtype == 'bounds':

        rd = np.array([[box[0], box[2]],
                       [box[1], box[3]]])
        # we use 0 here for the origin because we'll be cutting using np.arrays
        LOGINFO('Requested coords = %s' % repr(rd))
        pix = w.all_world2pix(rd,0)

    elif boxtype == 'center':

        rd = np.array(
            [
                [box[0] - (box[2])/2.0,
                 box[1] - (box[3])/2.0],
                [box[0] + (box[2])/2.0,
                 box[1] + (box[3])/2.0],

            ]
        )

        LOGINFO('Requested coords = %s' % repr(rd))
        pix = w.all_world2pix(rd,0)

    # do the cutout using a box generated by the radec -> pix bits above
    x1, x2, y1, y2 = pix[0,0], pix[1,0], pix[0,1], pix[1,1]

    # figure out xmin, xmax, ymin, ymax
    if x1 > x2:
        xmin = x2
        xmax = x1
    else:
        xmin = x1
        xmax = x2

    if y1 > y2:
        ymin = y2
        ymax = y1
    else:
        ymin = y1
        ymax = y2

    # round the pix coords to integers
    xmin, xmax = int(np.round(xmin)), int(np.round(xmax))
    ymin, ymax = int(np.round(ymin)), int(np.round(ymax))

    LOGINFO('Pixel box xmin = %s, xmax = %s' % (xmin, xmax))
    LOGINFO('Pixel box ymin = %s, ymax = %s' % (ymin, ymax))

    #
    # now, read in the image
    #

    # trim the image if requested
    if trim:
        trimmed_img = operations.trim_image(img, hdr, trimkeys=trimkeys)
    else:
        trimmed_img = img

    # scale the image if requested
    if scale_func is not None and isinstance(scale_func_params, dict):
        scaled_img = scale_func(trimmed_img,**scale_func_params)
    elif scale_func is not None:
        scaled_img = scale_func(trimmed_img)
    else:
        scaled_img = trimmed_img

    # make sure we take care of edges
    if xmin < 0:
        xmin = 0
    if xmax >= scaled_img.shape[1]:
        xmax = scaled_img.shape[1] - 1
    if ymin < 0:
        ymin = 0
    if ymax >= scaled_img.shape[0]:
        ymax = scaled_img.shape[0] - 1

    #
    # apply the box
    #

    # numpy is y,x so make sure to reverse the order
    boxed_img = scaled_img[ymin:ymax, xmin:xmax]

    # flip the image if requested
    if flip is True:
        boxed_img = np.flipud(boxed_img)

    # convert to PIL.Image
    pillow_image = Image.fromarray(boxed_img)
    pillow_image = pillow_image.convert('L')

    # annotate the image if told to do so
    if annotate:

        draw = ImageDraw.Draw(pillow_image)

        # if we're supposed to use another file for the JD source, do so
        # this is useful for subtracted images
        if fits_jdsrc is not None and os.path.exists(fits_jdsrc):
            framejd = float(operations.get_header_keyword(fits_jdsrc,
                                                          fits_jdkey))

        elif frame_time is not None:
            framejd = frame_time

        else:
            framejd = float(hdr[fits_jdkey]) if fits_jdkey in hdr else None

        if framejd is not None:

            timeannotation = 'JD %.4f' % framejd
            draw.text((4, 2),
                      timeannotation,
                      font=fontsmall,
                      fill=255)

        del draw

    # finally, generate the output file name if None is given
    if not out_fname:

        if boxtype == 'center':
            box_infostr = 'RC%.3fDC%.3f-RW%.3fDW%.3f' % (
                box[0], box[1], box[2], box[3]
            )
        elif boxtype == 'bounds':
            box_infostr = 'RL%.3fRH%.3f-DL%.3fDH%.3f' % (
                box[0], box[1], box[2], box[3]
            )

        out_fname = '%s-%s-%s-%s-proj%s-%s-%s.jpg' % (
            fits_image.replace('.fits','').replace('.fz',''),
            (hdr[fits_imagetype_key].lower()
             if fits_imagetype_key in hdr else 'imgtype_unknown'),
            (hdr[fits_exptime_key]
             if fits_exptime_key in hdr else 'exptime_unknown'),
            (hdr[fits_filters_key].replace('+','') if
             fits_filters_key in hdr else 'filt_unknown'),
            hdr[fits_project_key] if fits_project_key in hdr else '_unknown',
            hdr[fits_object_key] if fits_object_key in hdr else 'obj_unknown',
            box_infostr,
        )

    # save the file and return
    pillow_image.save(out_fname)
    return os.path.abspath(out_fname)


def fits_xybox_to_jpeg(
        fits_image,
        box,
        boxtype='center',
        out_fname=None,
        ext=None,
        trim=True,
        trimkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
        scale_func=operations.zscale_image,
        scale_func_params=None,
        flip=True,
        annotate=True,
        fits_imagetype_key='IMAGETYP',
        fits_exptime_key='EXPTIME',
        fits_filters_key='FILTERS',
        fits_project_key='PROJID',
        fits_object_key='OBJECT',
        fits_jdsrc=None,
        fits_jdkey='JD',
        frame_time=None,
):
    '''
    This converts an x-y coords box inside a FITS image to a JPEG.

    boxtype is either 'center' or 'bounds'.

    - if boxtype == 'bounds', then box is: [xmin, xmax, ymin, max]
    - if boxtype == 'center', then box is: [xcenter, ycenter, xwidth, ywidth]

    '''

    # handle .fz and non-zero extension FITS reading
    compressed_ext = operations.compressed_fits_ext(fits_image)

    if ext is None and compressed_ext:
        img, hdr = operations.read_fits(fits_image,
                                        ext=compressed_ext[0])

    elif (ext is not None):
        img, hdr = operations.read_fits(fits_image, ext=ext)
    else:
        img, hdr = operations.read_fits(fits_image)

    #
    # trim the image if requested
    #
    if trim:
        trimmed_img = operations.trim_image(img, hdr, trimkeys=trimkeys)
    else:
        trimmed_img = img

    # scale the image if requested
    if scale_func is not None and isinstance(scale_func_params, dict):
        scaled_img = scale_func(trimmed_img,**scale_func_params)
    elif scale_func is not None:
        scaled_img = scale_func(trimmed_img)
    else:
        scaled_img = trimmed_img

    if boxtype == 'bounds':

        x1, x2 = box[0], box[1]
        y1, y2 = box[2], box[3]

        # figure out xmin, xmax, ymin, ymax
        if x1 > x2:
            xmin = x2
            xmax = x1
        else:
            xmin = x1
            xmax = x2

        if y1 > y2:
            ymin = y2
            ymax = y1
        else:
            ymin = y1
            ymax = y2

        # round the pix coords to integers
        xmin, xmax = int(np.round(xmin)), int(np.round(xmax))
        ymin, ymax = int(np.round(ymin)), int(np.round(ymax))

        # make sure we take care of edges
        if xmin < 0:
            xmin = 0
        if xmax >= img.shape[1]:
            xmax = img.shape[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax >= img.shape[0]:
            ymax = img.shape[0] - 1

        boxed_img = scaled_img[ymin:ymax, xmin:xmax]

    elif boxtype == 'center':

        # numpy is y,x
        x1, x2 = (box[0] - box[2]/2.0,
                  box[0] + box[2]/2.0)
        y1, y2 = (box[1] - box[3]/2.0,
                  box[1] + box[3]/2.0)

        # figure out xmin, xmax, ymin, ymax
        if x1 > x2:
            xmin = x2
            xmax = x1
        else:
            xmin = x1
            xmax = x2

        if y1 > y2:
            ymin = y2
            ymax = y1
        else:
            ymin = y1
            ymax = y2

        # round the pix coords to integers
        xmin, xmax = int(np.round(xmin)), int(np.round(xmax))
        ymin, ymax = int(np.round(ymin)), int(np.round(ymax))

        # make sure we take care of edges
        if xmin < 0:
            xmin = 0
        if xmax >= img.shape[1]:
            xmax = img.shape[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax >= img.shape[0]:
            ymax = img.shape[0] - 1

        boxed_img = scaled_img[ymin:ymax, xmin:xmax]

    # flip the image if requested
    if flip is True:
        boxed_img = np.flipud(boxed_img)

    # convert to PIL.Image
    pillow_image = Image.fromarray(boxed_img)
    pillow_image = pillow_image.convert('L')

    # annotate the image if told to do so
    if annotate:

        draw = ImageDraw.Draw(pillow_image)

        # if we're supposed to use another file for the JD source, do so
        # this is useful for subtracted images
        if fits_jdsrc is not None and os.path.exists(fits_jdsrc):
            framejd = float(operations.get_header_keyword(fits_jdsrc,
                                                          fits_jdkey))

        elif frame_time is not None:
            framejd = frame_time

        else:
            framejd = float(hdr[fits_jdkey]) if fits_jdkey in hdr else None

        if framejd is not None:

            timeannotation = 'JD %.4f' % framejd
            draw.text((4, 2),
                      timeannotation,
                      font=fontsmall,
                      fill=255)

        del draw

    # finally, generate the output file name if None is given
    if not out_fname:

        if boxtype == 'center':
            box_infostr = 'XC%.3fYC%.3f-XW%.3fYW%.3f' % (
                box[0], box[1], box[2], box[3]
            )
        elif boxtype == 'bounds':
            box_infostr = 'XL%.3fXH%.3f-YL%.3fYH%.3f' % (
                box[0], box[1], box[2], box[3]
            )

        out_fname = '%s-%s-%s-%s-proj%s-%s-%s.jpg' % (
            fits_image.replace('.fits','').replace('.fz',''),
            (hdr[fits_imagetype_key].lower()
             if fits_imagetype_key in hdr else 'imgtype_unknown'),
            (hdr[fits_exptime_key]
             if fits_exptime_key in hdr else 'exptime_unknown'),
            (hdr[fits_filters_key].replace('+','') if
             fits_filters_key in hdr else 'filt_unknown'),
            hdr[fits_project_key] if fits_project_key in hdr else '_unknown',
            hdr[fits_object_key] if fits_object_key in hdr else 'obj_unknown',
            box_infostr,
        )

    # save the file and return
    pillow_image.save(out_fname)
    return os.path.abspath(out_fname)


##################
## IMAGE MOVIES ##
##################

def make_frame_movie(imgdir,
                     outfile,
                     framerate=15,
                     crf=17,
                     ffmpeg_exe='ffmpeg',
                     rescale_to_width=1024,
                     preset='slower',
                     fileglob='*.jpg'):
    '''This makes frame movies for all frame jpegs/pngs in imgdir.

    Use fileglob to indicate the files to look for.

    Use ffmpeg_exe to set the path to the ffmpeg executable.

    crf sets the visual quality of the frames:

    - crf = 0  -> lossless encoding
    - crf = 17 -> visually lossless
    - crf = 51 -> maximum compression

    Use preset to set the compression quality of the encoding:

    'veryslow' -> slowest encoding but smallest MP4 movie files
    'slow'
    'fast'
    'veryfast' -> fastest encoding but largest MP4 movie files

    The framerate also affects the file size.

    To encode faster and make smaller movies, resize the input jpegs to smaller
    sizes, e.g. by using the resize=True and resizefrac=<1.0 kwargs to
    parallel_frame_jpegs.

    '''

    filepath = os.path.join(os.path.abspath(imgdir),fileglob)

    # rescale the movie frames if told to do so
    if rescale_to_width is not None and rescale_to_width > 0:
        rescale_opt = '-filter:v scale="%s:trunc(ow/a/2)*2" ' % rescale_to_width
    else:
        rescale_opt = ''

    # FFMPEG commandline
    FFMPEGCMD = (
        "{ffmpeg} -y -framerate {framerate} "
        "-pattern_type glob -i '{fileglob}' "
        "-an -c:v libx264 "
        "-crf {crf} -preset {preset} "
        "-pix_fmt yuv420p "
        "{rescale_opt}"
        "-movflags faststart {outfile}"
    )

    cmdtorun = FFMPEGCMD.format(ffmpeg=ffmpeg_exe,
                                fileglob=filepath,
                                framerate=framerate,
                                rescale_opt=rescale_opt,
                                outfile=outfile,
                                crf=crf,
                                preset=preset)

    proc = subprocess.run(cmdtorun, shell=True)
    LOGINFO('FFmpeg done. Return code was %s.' % proc.returncode)

    # check if we succeeded in making the output file
    if os.path.exists(outfile) and proc.returncode == 0:
        LOGINFO('Frame movie successfully created: %s' % outfile)
        return os.path.abspath(outfile)

    else:

        LOGERROR('Could not make frame movie for files in '
                 'directory: %s, using fileglob: %s' % (imgdir, fileglob))
        return None


def make_frame_movie_from_list(filelist,
                               outfile,
                               framerate=15,
                               crf=17,
                               ffmpeg_exe='ffmpeg',
                               rescale_to_width=1024,
                               preset='slower',
                               fileglob='*.jpg'):
    '''This makes frame movies for all frame jpegs/pngs in the given list.

    Use ffmpeg_exe to set the path to the ffmpeg executable.

    crf sets the visual quality of the frames:

    - crf = 0  -> lossless encoding
    - crf = 17 -> visually lossless
    - crf = 51 -> maximum compression

    Use preset to set the compression quality of the encoding:

    'veryslow' -> slowest encoding but smallest MP4 movie files
    'slow'
    'fast'
    'veryfast' -> fastest encoding but largest MP4 movie files

    The framerate also affects the file size.

    To encode faster and make smaller movies, resize the input jpegs to smaller
    sizes, e.g. by using the resize=True and resizefrac=<1.0 kwargs to
    parallel_frame_jpegs.

    '''

    # rescale the movie frames if told to do so
    if rescale_to_width is not None and rescale_to_width > 0:
        rescale_opt = '-filter:v scale="%s:trunc(ow/a/2)*2" ' % rescale_to_width
    else:
        rescale_opt = ''

    # generate the cat command
    catcommand = "cat %s" % ' '.join(filelist)

    # FFMPEG commandline
    FFMPEGCMD = (
        "{catcommand} | {ffmpeg} "
        "-y "
        "-f image2pipe "
        "-i - "
        "-framerate {framerate} "
        "-an -c:v libx264 "
        "-crf {crf} -preset {preset} "
        "-pix_fmt yuv420p "
        "{rescale_opt}"
        "-movflags faststart {outfile}"
    )

    cmdtorun = FFMPEGCMD.format(ffmpeg=ffmpeg_exe,
                                catcommand=catcommand,
                                framerate=framerate,
                                rescale_opt=rescale_opt,
                                outfile=outfile,
                                crf=crf,
                                preset=preset)

    proc = subprocess.run(cmdtorun, shell=True)
    LOGINFO('FFmpeg done. Return code was %s.' % proc.returncode)

    # check if we succeeded in making the output file
    if os.path.exists(outfile) and proc.returncode == 0:
        LOGINFO('Frame movie successfully created: %s' % outfile)
        return os.path.abspath(outfile)

    else:

        LOGERROR('Could not make frame movie for files in the input list.')
        return None

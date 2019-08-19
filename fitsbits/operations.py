#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This contains functions to perform basic operations on FITS images (reading,
writing, updating headers, etc.).

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

from datetime import datetime

import numpy as np
import numpy.ma as npma
import csv
import re

from scipy.interpolate import UnivariateSpline

from astropy.io import fits as pyfits
from astropy.io.fits import Card
from astropy.visualization import ZScaleInterval

from fitsbits import __gitrev__


########################
## READING FITS FILES ##
########################

def compressed_fits_ext(fits_file):
    '''
    Check if a fits file is a compressed FITS file. Return the extension numbers
    of the compressed image as a list if these exist, otherwise, return None.

    '''

    hdulist = pyfits.open(fits_file)

    compressed_img_exts = []

    for i, ext in enumerate(hdulist):
        if isinstance(ext, pyfits.hdu.compressed.CompImageHDU):
            compressed_img_exts.append(i)

    hdulist.close()

    if len(compressed_img_exts) < 1:
        return None
    else:
        return compressed_img_exts


def read_fits(fits_file,
              ext=None,
              mask_extension_name='IMGMASK',
              get_mask=False,
              use_mask=False):
    '''Shortcut function to get the header and data from a fits file and a given
    extension.

    If ext is None, will try to figure out the extension automatically.

    if get_mask = True, then will try to get and return the mask image from the
    FITS extension provided in mask_extension_name as the third element in the
    returned tuple.

    if use_mask = True, will apply the mask in the mask image from the FITS
    extension specified in mask_extension_name to the actual image array and
    return a masked np.ma.array.

    '''

    compressed_ext = compressed_fits_ext(fits_file)

    if ext is None and compressed_ext:
        cext = compressed_ext[0]
    elif (ext is not None):
        cext = ext
    else:
        cext = 0

    hdulist = pyfits.open(fits_file)
    img_header = hdulist[cext].header
    img_data = hdulist[cext].data

    if get_mask:

        try:
            img_mask = hdulist[mask_extension_name].data
            img_mask = img_mask == 1
        except Exception:
            img_mask = None

        hdulist.close()
        return img_data, img_header, img_mask

    elif use_mask:

        try:
            img_mask = hdulist[mask_extension_name].data
            img_mask = img_mask == 1

            masked_img = npma.array(img_data, mask=img_mask)

        except Exception:
            masked_img = img_data

        hdulist.close()
        return masked_img, img_header, img_mask

    else:
        hdulist.close()
        return img_data, img_header


def parse_fits_header(img_header,
                      parse_comments=False):
    '''
    This parses a FITS header and returns a dict.

    '''

    # remove SHOUTY keys
    dict_header = {x.lower():img_header[x] for x in img_header}

    # remove padding
    if '' in dict_header:
        del dict_header['']

    # parse the comment into a nested dict
    if parse_comments and 'comment' in dict_header:

        dict_header['extra'] = {}

        comment_list = [
            x.lstrip("= '").rstrip("'") for x in dict_header['comment']
        ]

        # now parse the items into a dict
        reader = csv.reader(comment_list, delimiter='=')

        for row in reader:
            key = row[0]
            val = row[1]

            try:
                float_val = float(val)
            except Exception:
                float_val = None

            try:
                int_float_val = int(float(val))
            except Exception:
                int_float_val = None

            if float_val is not None and float_val == int_float_val:
                val = int_float_val
            elif float_val is not None and float_val != int_float_val:
                val = float_val
            elif ('true' == val.lower() or
                  'yes' == val.lower()):
                val = True
            elif ('false' == val.lower() or
                  'no' == val.lower()):
                val = False

            if isinstance(val, str) and len(val) == 0:
                val = None

            dict_header['extra'][key] = val

        del dict_header['comment']

    return dict_header


def read_header(fits_file,
                ext=None,
                parse_comments=False):
    '''
    This reads the header of the FITS file and converts it into a dict.

    Removes any padding and converts the COMMENTS items into a nested dict.

    '''
    compressed_ext = compressed_fits_ext(fits_file)

    if ext is None and compressed_ext:
        cext = compressed_ext[0]
    elif (ext is not None):
        cext = ext
    else:
        cext = 0

    hdulist = pyfits.open(fits_file)
    img_header = hdulist[cext].header
    hdulist.close()

    return parse_fits_header(img_header,
                             parse_comments=parse_comments)


########################
## WRITING FITS FILES ##
########################

def new_hdulist_with_updated_header(newdata,
                                    oldheader,
                                    extrainfo=None):
    '''This makes a new HDUList using new frame data and updated header.

    The primary use case for this to make calibrated FITS files from raw FITS
    files.

    '''

    # new primary HDU, initialized with old header
    primhdu = pyfits.PrimaryHDU(data=newdata, header=oldheader)

    # make the header NAXIS, etc. consistent with new data shape and dtype
    primhdu.update_header()

    # new HDU list
    newhdulist = pyfits.HDUList([primhdu])

    # add the date when this was processed
    newhdulist[0].header['procdt'] = (
        datetime.utcnow().isoformat(),
        'last processed at UTC datetime'
    )

    # add the gitrev if available
    newhdulist[0].header['fitsbits'] = (
        __gitrev__,
        'fitsbits git revision'
    )

    # add any extra information provided
    if extrainfo is not None:
        newhdulist[0].header.update(extrainfo)

    return newhdulist


def add_mask_extension(hdulist,
                       imgarray,
                       usemaskarray=None,
                       mask_headerkey='imgmask'):
    '''This adds a mask extension to the FITS HDUList.

    By convention, usemaskarray is a np.int16 np.array of the same shape as
    imgarray, where:

    - 0 means unmasked
    - 1 means masked

    This translates directly to the use of this array in the mask argument of a
    np.ma.masked_array.

    If usemaskarray is None, this will be created automatically, with nothing
    masked.

    '''

    # initially, nothing is masked
    if usemaskarray:
        maskarray = usemaskarray
    else:
        maskarray = np.full_like(imgarray, 0, dtype=np.int16)
    maskhdu = pyfits.ImageHDU(data=maskarray, name=mask_headerkey)
    hdulist.append(maskhdu)
    return hdulist


##################
## FITS HEADERS ##
##################

def get_header_keyword(fits_file,
                       keyword,
                       ext=None):
    '''
    Get the value of a header keyword in a fits file optionally using an
    extension.

    '''

    # auto-check if the image is compressed, and get the correct extension
    if ext is None:

        compressed_ext = compressed_fits_ext(fits_file)
        if compressed_ext is not None:
            ext = compressed_ext[0]
        else:
            ext = 0

    hdulist = pyfits.open(fits_file)

    if keyword in hdulist[ext].header:
        val = hdulist[ext].header[keyword]
    else:
        val = None

    hdulist.close()
    return val


def get_header_keyword_list(fits_file,
                            keyword_list,
                            ext=None):
    '''
    This gets a list of FITS header keywords.

    '''

    # auto-check if the image is compressed, and get the correct extension
    if ext is None:

        compressed_ext = compressed_fits_ext(fits_file)
        if compressed_ext is not None:
            ext = compressed_ext[0]
        else:
            ext = 0

    hdulist = pyfits.open(fits_file)

    out_dict = {}

    for keyword in keyword_list:

        if keyword in hdulist[ext].header:
            out_dict[keyword] = hdulist[ext].header[keyword]
        else:
            out_dict[keyword] = None

    hdulist.close()
    return out_dict


def get_header_regex(fits_file,
                     header_regex,
                     ext=None):
    '''This gets all header keys and their vals matching the specified regex
    string.

    '''

    # auto-check if the image is compressed, and get the correct extension
    if ext is None:

        compressed_ext = compressed_fits_ext(fits_file)
        if compressed_ext is not None:
            ext = compressed_ext[0]
        else:
            ext = 0

    hdulist = pyfits.open(fits_file)
    header = hdulist[ext].header
    hdulist.close()

    hdrstr = header.tostring(padding=False,
                             sep='\n',
                             endcard=False)
    hdrstr = hdrstr.split('\n')
    hdrstr = [x.strip() for x in hdrstr]

    # we can handle either a compiled re.Pattern object (useful for parallel
    # drivers) or a string that we'll compile ourselves
    if isinstance(header_regex, re.Pattern):
        match_regex = header_regex
    elif isinstance(header_regex, str):
        match_regex = re.compile(r'%s' % header_regex)

    matches = {}

    # go through each header string
    for h in hdrstr:

        matching = match_regex.search(h)

        if matching is not None:
            kv = Card.fromstring(h)
            matches[kv.keyword] = kv.value

    return matches


###########################
## FITS IMAGE OPERATIONS ##
###########################

def trim_image(fits_img,
               fits_hdr,
               inplace=False,
               trimkeys=('TRIMSEC','DATASEC','TRIMSEC0'),
               custombox=None):
    '''
    Returns a trimmed image using the TRIMSEC header of the image header.

    custombox is a string of the form [Xlo:Xhi,Ylo:Yhi] and will trim the image
    to a custom size.

    '''
    if not inplace:
        fitsimg = fits_img[::]
    else:
        fitsimg = fits_img

    if custombox:

        trimsec = custombox

    else:

        trimsec = None

        for h in trimkeys:
            if h in fits_hdr:
                trimsec = fits_hdr[h]
                break
        else:
            if custombox is None:
                LOGERROR('no DATASEC or TRIMSEC in image header')
                return

    if trimsec and trimsec != '[0:0,0:0]':

        datasec = trimsec.strip('[]').split(',')

        try:
            datasec_y = [int(x) for x in datasec[0].split(':')]
            datasec_x = [int(x) for x in datasec[1].split(':')]

            trimmed_img = fitsimg[datasec_x[0]-1:datasec_x[1],
                                  datasec_y[0]-1:datasec_y[1]]
        except ValueError:
            LOGWARNING('datasec/trimsec not correctly set in FITS header, '
                       ' not trimming')
            trimmed_img = fitsimg

    else:
        LOGWARNING('datasec/trimsec not correctly set in FITS header, '
                   ' not trimming')
        trimmed_img = fitsimg

    return trimmed_img


def bias_overscan_correction(fits_img,
                             fits_header,
                             inplace=False,
                             biassec_keyword='BIASSEC',
                             custom_biassec=None):
    '''This does a bias overscan correction for the image.

    fits_img is a hdulist[x].data ndarray for FITS extension x

    fits_header is a hdulist[x].header structure for FITS extension x.

    biassec_keyword is the FITS header keyword to use to get the bias overscan
    section of the FITS image. If this keyword isn't found, the function will
    fall back to custom_biassec.

    custom_biassec is the bias section to use if biassec_keyword is not present
    in the FITS header or as an override.

    This returns the fits_img with the bias overscan correction subtracted.

    '''

    if not inplace:
        fitsimg = fits_img[::]
    else:
        fitsimg = fits_img

    if biassec_keyword in fits_header:
        biassec = fits_header[biassec_keyword]
    elif custom_biassec:
        biassec = custom_biassec
    else:
        LOGERROR("no BIASSEC defined or provided, can't continue")
        return None

    # parse the bias section
    biassec = biassec.strip('[]')
    xslice, yslice = biassec.split(',')
    xslicelo, xslicehi = (int(x) for x in xslice.split(':'))
    yslicelo, yslicehi = (int(y) for y in yslice.split(':'))

    # to handle some casting issues in newer numpy
    if fitsimg.dtype is not np.dtype('float32'):
        fitsimg = fitsimg.astype(np.float32)

    # calculate the medians along the short axis of the biassec.  we do this
    # because some cameras have biassecs as extra cols and some have them as
    # extra rows. we assume that the short axis is the more significant one
    # NOTE: numpy convention is y,x and we use FITS pixnum - 1 since they start
    # from 1 and numpy starts at 0
    overscan = fitsimg[yslicelo-1:yslicehi, xslicelo-1:xslicehi]
    overscan_shape = overscan.shape
    # numpy convention: rows (y) = 0, cols (x) = 1

    # if ncols > nrows, then short axis = rows -> axis = 0
    if overscan_shape[1] > overscan_shape[0]:
        medax = 0
    # if ncols < nrows, then short axis = cols -> axis = 1
    else:
        medax = 1

    medians = np.median(overscan, axis=medax)
    medvar = np.arange(medians.size)

    # fit a spline and do the correction
    try:

        overscan_spl = UnivariateSpline(medvar, medians)

        # get the data section of the image
        if 'TRIMSEC' in fits_header:
            trimsec = fits_header['TRIMSEC']
        elif 'DATASEC' in fits_header:
            trimsec = fits_header['DATASEC']
        else:
            raise ValueError("no TRIMSEC/DATASEC in FITS header, "
                             "can't continue")

        if trimsec != '[0:0,0:0]':

            datasec = trimsec.strip('[]').split(',')
            datasec_x = [int(x) for x in datasec[1].split(':')]
            datasec_y = [int(y) for y in datasec[0].split(':')]

        else:

            raise ValueError("invalid TRIMSEC/DATASEC in FITS header, "
                             "can't continue")

        # subtract overscan fit from the datasec of the image
        fitsimg[
            datasec_y[0]-1:datasec_y[1],
            datasec_x[0]-1:datasec_x[1]
        ] = (fitsimg[datasec_y[0]-1:datasec_y[1],
                     datasec_x[0]-1:datasec_x[1]] - overscan_spl(medvar))

        return fitsimg

    except Exception:

        LOGEXCEPTION("could not fit spline to BIASSEC, can't continue")
        raise


def mask_image(fits_img,
               inplace=False,
               mask_saturated=True,
               mask_saturated_min_adu=60000,
               mask_saturated_growmaxpx=(20,20),  # updown, leftright
               mask_frame_slices=None,
               mask_direct_with=None,
               existing_mask=None):
    '''This masks saturated pixels in a FITS image (by default) and optionally
    applies user-defined masks as well.

    If mask_saturated = True, will mask saturated pixels, using the following
    options:

    mask_saturated_min_adu sets the min ADU level to consider as saturated.

    mask_saturated_growmaxpx sets the number of pixels outside each saturated
    pixel to extend the mask to. This is a list of two values: one for 'updown'
    and one for 'leftright': 'updown' will extend the masks in the column
    direction for each masked pixel. 'leftright' will extend the masks in the
    row direction for each masked pixel. In this way, one can handle the pixel
    bleed direction of the camera.

    mask_frame_slices is a list of 1-indexed FITS slice notation strings of the
    form: 'xlo:xhi,ylo:yhi'. This is used to add custom masks to the frame for
    bad columns, etc.

    If mask_direct_with is not None, the mask will be applied directly to the
    fits image data in fits_img instead of just being written to the mask
    array. In this case, the value of mask_direct_with will be used as the
    substitute value for the existing pixel values. This can be used if you
    don't want the default return value of this function, which is a
    numpy.ma.masked_array.

    If existing_mask is not None, this mask will be added onto in this function.

    Returns:

    a tuple of fits_img, mask if mask_direct_with is not None
    a np.ma.masked_array if mask_direct_with is None

    '''

    if existing_mask is None:
        mask = np.full_like(fits_img, 0, dtype=np.int16)
    else:
        mask = existing_mask

    if not inplace:
        workimg = fits_img[::]
    else:
        workimg = fits_img

    # first, we'll mask the saturated regions of the image
    if mask_saturated:

        # we'll iterate through the image using these
        nrows = workimg.shape[0]
        ncols = workimg.shape[1]

        # we have to go through each pixel because masks may be non-contiguous
        # and we have to handle growing the mask individually for each pixel
        # that's masked
        for colind in range(ncols):
            for rowind in range(nrows):

                thispixval = workimg[rowind, colind]

                # handle updown extended masks
                if thispixval > mask_saturated_min_adu:

                    # generate the updown mask
                    updown_mask_ind = np.arange(rowind -
                                                mask_saturated_growmaxpx[0],
                                                rowind +
                                                mask_saturated_growmaxpx[0])

                    # make sure the mask doesn't go outside the image
                    # boundary
                    updown_mask_ind = updown_mask_ind[
                        (updown_mask_ind > 0) &
                        (updown_mask_ind < nrows)
                    ]

                    # generate the leftright mask
                    leftright_mask_ind = np.arange(colind -
                                                   mask_saturated_growmaxpx[1],
                                                   colind +
                                                   mask_saturated_growmaxpx[1])

                    # make sure the mask doesn't go outside the image
                    # boundary
                    leftright_mask_ind = leftright_mask_ind[
                        (leftright_mask_ind > 0) &
                        (leftright_mask_ind < ncols)
                    ]

                    #
                    # apply the masks
                    #

                    # first, extend the dimensions
                    updown_mask_ind = np.atleast_2d(updown_mask_ind)
                    leftright_mask_ind = np.atleast_2d(leftright_mask_ind)

                    # apply to mask array first
                    mask[updown_mask_ind, leftright_mask_ind.T] = 1

                    # mask[updown_mask_ind, colind] = 1
                    # mask[rowind, leftright_mask_ind] = 1

                    # if we're masking directly, also apply to the actual
                    # FITS image
                    if mask_direct_with is not None:

                        workimg[updown_mask_ind, leftright_mask_ind.T] = (
                            mask_direct_with
                        )

    #
    # end of saturated star masking
    #

    #
    # now we'll mask custom slices if provided
    #
    if mask_frame_slices is not None:

        for sli in mask_frame_slices:

            xslice, yslice = sli.split(',')
            xslicelo, xslicehi = (int(x) for x in xslice.split(':'))
            yslicelo, yslicehi = (int(y) for y in yslice.split(':'))

            # apply the mask
            mask[yslicelo-1:yslicehi, xslicelo-1:xslicehi] = 1

            if mask_direct_with is not None:
                workimg[yslicelo-1:yslicehi, xslicelo-1:xslicehi] = (
                    mask_direct_with
                )

    #
    # done with all masking
    #

    if mask_direct_with is not None:
        return fits_img, mask
    else:
        return npma.array(fits_img, mask=mask)


####################
## SCALING IMAGES ##
####################

def direct_linscale_img(img_array,
                        low,
                        high,
                        cap=255.0):
    '''
    This clips the image between the directly specified values low and high.

    '''

    img_med = np.nanmedian(img_array)
    clipped_linear_img = np.clip(img_array,
                                 img_med - low,
                                 img_med + high)
    return cap*clipped_linear_img/(img_med + high)


def clipped_linscale_img(img_array,
                         losig=2.0,
                         hisig=2.5,
                         cap=255.0):
    '''
    This clips the image between the values:

    [median(img_array) - losig*img_stdev, median(img_array) + hisig*img_stdev]

    and returns a linearly scaled image using the cap given.

    '''

    img_med = np.nanmedian(img_array)
    img_stdev = np.nanstd(img_array)

    clipped_linear_img = np.clip(img_array,
                                 img_med - losig*img_stdev,
                                 img_med + hisig*img_stdev)
    return cap*clipped_linear_img/(img_med + hisig*img_stdev)


def clipped_logscale_img(img_array,
                         cap=255.0,
                         lomult=1.0,
                         himult=2.5,
                         coeff=1000.0):
    '''
    This clips the image between the values:

    [median(img_array) - lomult*stdev(img_array),
     median(img_array) + himult*stdev(img_array)]

    and returns a log-scaled image using the cap given.

    logscale_img = np.log(coeff*(img/max(img))+1)/np.log(coeff)
    '''

    img_med, img_stdev = np.median(img_array), np.std(img_array)
    clipped_linear_img = np.clip(img_array,
                                 img_med-lomult*img_stdev,
                                 img_med+himult*img_stdev)

    clipped_linear_img = clipped_linear_img/(img_med+himult*img_stdev)

    # janky
    clipped_linear_img[clipped_linear_img < 0] = np.nan

    div = np.nanmax(clipped_linear_img)

    logscaled_img = (
        np.log(coeff*clipped_linear_img/div+1) /
        np.log(coeff)
    )

    return cap*logscaled_img


def zscale_image(imgarr):
    '''
    This zscales an image.

    '''

    zscaler = ZScaleInterval()
    scaled_vals = zscaler.get_limits(imgarr)
    return direct_linscale_img(imgarr, scaled_vals[0], scaled_vals[1])

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This module contains functions to perform operations on collections of FITS
files.

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
import glob
import multiprocessing
import sys
import re
import pickle

# Ref: https://bugs.python.org/issue33725
# TLDR; Apple is trash at UNIX
if sys.platform == 'darwin':
    mp = multiprocessing.get_context('forkserver')
else:
    mp = multiprocessing

import numpy as np
from sklearn.cluster import DBSCAN

from ._coordutils import hms_str_to_decimal, dms_str_to_decimal
from ._modtools import object_from_string
from ._extractors import extract_frame_targetfield
from .compression import safe_compress, safe_uncompress

from . import operations


############
## CONFIG ##
############

NCPUS = mp.cpu_count()


#######################
## UTILITY FUNCTIONS ##
#######################

def fits_header_worker(task):
    '''This wraps operations.get_header_keyword_list for the
    parallel_get_fits_headers function below.

    A task is a list:

    [fits_file, keyword_list, extension]

    '''

    try:
        return (
            task[0],
            operations.get_header_keyword_list(task[0],
                                               task[1],
                                               ext=task[2])
        )
    except Exception:

        LOGEXCEPTION('could not understand FITS file: %s' % task[0])
        return task[0], {}


def fits_regex_worker(task):
    '''
    This wraps operations.get_header_regex.

    A task is a list:

    [fits_file, regex_string, extension]

    '''

    try:
        return (
            task[0],
            operations.get_header_regex(task[0],
                                        task[1],
                                        ext=task[2])
        )
    except Exception:

        LOGEXCEPTION('could not understand FITS file: %s' % task[0])
        return task[0], {}


def parallel_regex_fits_headers_list(fitslist,
                                     regex,
                                     nworkers=4,
                                     maxworkertasks=1000,
                                     fitsext=None):
    '''This does a regex search using the specified expression and returns
    matching header key-value pairs for all files in the given list.

    '''

    LOGINFO("%s FITS images, getting keys matching regex '%s'..." %
            (len(fitslist), regex))

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    # pre-compile the pattern for speed
    if isinstance(regex, re.Pattern):
        task_regex = regex
    elif isinstance(regex, str):
        task_regex = re.compile(r'%s' % regex)

    tasks = [
        (os.path.abspath(x), task_regex, fitsext)
        for x in fitslist
    ]

    # fire up the pool of workers
    results = pool.map(fits_regex_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    LOGINFO('Done.')

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict


def parallel_regex_fits_headers(fitsdir,
                                regex,
                                fitsglob='*.fits*',
                                nworkers=4,
                                maxworkertasks=1000,
                                fitsext=None):
    '''This does a regex search using the specified expression and returns
    matching header key-value pairs for all files in the given list.

    '''
    # get a list of all fits files in the directory
    fitslist = glob.glob(os.path.join(fitsdir, fitsglob))

    LOGINFO("%s FITS images, getting keys matching regex '%s'..." %
            (len(fitslist), regex))

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    # pre-compile the pattern for speed
    if isinstance(regex, re.Pattern):
        task_regex = regex
    elif isinstance(regex, str):
        task_regex = re.compile(r'%s' % regex)

    tasks = [
        (os.path.abspath(x), task_regex, fitsext)
        for x in fitslist
    ]

    # fire up the pool of workers
    results = pool.map(fits_regex_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    LOGINFO('Done.')

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict


def parallel_get_fits_headers_list(fitslist,
                                   keywordlist,
                                   nworkers=4,
                                   maxworkertasks=1000,
                                   fitsext=None):

    '''
    This gets the specified keywords in keywordlist from the FITS headers for
    all files in given list.

    '''

    LOGINFO('%s FITS images, getting keywords %s...' %
            (len(fitslist), keywordlist))

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    tasks = [
        [os.path.abspath(x), keywordlist, fitsext]
        for x in fitslist
    ]

    # fire up the pool of workers
    results = pool.map(fits_header_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    LOGINFO('Done.')

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict


def parallel_get_fits_headers(fitsdir,
                              keywordlist,
                              fitsglob='*.fits*',
                              nworkers=4,
                              maxworkertasks=1000,
                              fitsext=None):
    '''
    This gets the specified keywords in keywordlist from the FITS headers for
    all files in the directory fitsdir, optionally filtering by the glob
    fitsglob.

    '''

    # get a list of all fits files in the directory
    fitslist = glob.glob(os.path.join(fitsdir, fitsglob))

    LOGINFO('Found %s FITS images in %s, getting keywords %s...' %
            (len(fitslist), fitsdir, keywordlist))

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    tasks = [
        [os.path.abspath(x), keywordlist, fitsext]
        for x in fitslist
    ]

    # fire up the pool of workers
    results = pool.map(fits_header_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    LOGINFO('Done.')

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict


################################
## FITS COMPRESSION FUNCTIONS ##
################################

def parallel_fpack_worker(task):
    '''
    This wraps fpack_frame for use in parallel_fpack_fitsdir.

    '''

    return task, safe_compress(task,
                               compression='fpack')


def parallel_gzip_worker(task):
    '''
    This wraps gzip_frame for use in parallel_gzip_fitsdir.

    '''

    return task, safe_compress(task,
                               compression='gzip')


def parallel_funpack_worker(task):
    '''
    This wraps funpack_frame for use with parallel_funpack_fitsdir.

    '''
    return task, safe_uncompress(task)


def parallel_gunzip_worker(task):
    '''
    This wraps gunzip_frame for use with parallel_gunzip_fitsdir.

    '''
    return task, safe_uncompress(task)


def parallel_fpack_fitslist(fitslist,
                            nworkers=NCPUS,
                            maxworkertasks=1000):
    '''
    This fpacks a directory of FITS files.

    '''

    tasks = fitslist

    # initialize the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(parallel_fpack_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in results}

    return resultdict


def parallel_funpack_fitslist(fitslist,
                              nworkers=NCPUS,
                              maxworkertasks=1000):
    '''
    This funpacks a directory of FITS files.

    '''

    tasks = fitslist

    # fire up the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    results = pool.map(parallel_funpack_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in results}

    return resultdict


def parallel_fpack_fitsdir(fitsdir,
                           fitsglob='*.fits',
                           outdir=None,
                           nworkers=NCPUS,
                           maxworkertasks=1000):
    '''
    This fpacks a directory of FITS files.

    '''

    # initialize the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    fitslist = sorted(glob.glob(os.path.join(os.path.abspath(fitsdir),
                                             fitsglob)))
    tasks = fitslist

    # fire up the pool of workers
    results = pool.map(parallel_fpack_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in results}

    return resultdict


def parallel_funpack_fitsdir(fitsdir,
                             fitsglob='*.fits.fz',
                             outdir=None,
                             nworkers=NCPUS,
                             maxworkertasks=1000):
    '''
    This funpacks a directory of FITS files.

    '''

    # initialize the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    fitslist = sorted(glob.glob(os.path.join(os.path.abspath(fitsdir),
                                             fitsglob)))
    tasks = fitslist

    # fire up the pool of workers
    results = pool.map(parallel_funpack_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in results}

    return resultdict


def parallel_gzip_fitslist(fitslist,
                           nworkers=NCPUS,
                           maxworkertasks=1000):
    '''
    This gzips a directory of FITS files.

    '''

    tasks = fitslist

    # initialize the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(parallel_gzip_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in results}

    return resultdict


def parallel_gunzip_fitslist(fitslist,
                             nworkers=NCPUS,
                             maxworkertasks=1000):
    '''
    This gunzips a directory of FITS files.

    '''

    tasks = fitslist

    # fire up the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    results = pool.map(parallel_gunzip_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in results}

    return resultdict


def parallel_gzip_fitsdir(fitsdir,
                          fitsglob='*.fits',
                          outdir=None,
                          nworkers=NCPUS,
                          maxworkertasks=1000):
    '''
    This gzips a directory of FITS files.

    '''

    # initialize the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    fitslist = sorted(glob.glob(os.path.join(os.path.abspath(fitsdir),
                                             fitsglob)))
    tasks = fitslist

    # fire up the pool of workers
    results = pool.map(parallel_gzip_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in results}

    return resultdict


def parallel_gunzip_fitsdir(fitsdir,
                            fitsglob='*.fits.fz',
                            outdir=None,
                            nworkers=NCPUS,
                            maxworkertasks=1000):
    '''
    This gunzips a directory of FITS files.

    '''

    # initialize the pool of workers
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    fitslist = sorted(glob.glob(os.path.join(os.path.abspath(fitsdir),
                                             fitsglob)))
    tasks = fitslist

    # fire up the pool of workers
    results = pool.map(parallel_gunzip_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    resultdict = {x:y for (x,y) in results}

    return resultdict


################################################
## LOOKING UP IMAGE TYPES AND OBSERVED FIELDS ##
################################################

def sort_fitslist_by_file_pattern(fitslist,
                                  patternlist,
                                  remove_pattern_regex=None):
    '''This sorts FITS files in the given list by given filename patterns.

    Useful for sorting by CCD if the CCD IDs are encoded into the filename as
    for HATPI::

        1-123456_1.fits -> where _1 is the CCD ID.

    Parameters
    ----------

    fitslist : list of str
        A list of FITS files to process.

    patternlist : list of str
        A list of patterns to break up the files in the input FITS list.

    remove_pattern_regex : raw str or None
        If `remove_pattern_regex` is not None, it must be a regex string
        specifying characters to remove from the patterns to form the keys of
        the dict returned, e.g. to turn the pattern '_1' in the example above to
        '1', use::

            remove_pattern_regex=r'_'

        This will turn the output dict from::

            {'_1':[list of files matching *_1.fits],
             '_2':[list of files matching *_2.fits],
            ...}

        into::

            {'1':[list of files matching *_1.fits],
             '2':[list of files matching *_2.fits],
            ...}

        which is much cleaner.

    Returns
    -------

    dict
        The dict returned is of the form::

            {x:y for x,y in zip(patternlist, matching_fits)}

    '''

    if remove_pattern_regex is not None:

        pattern_sub = re.compile(remove_pattern_regex)
        matching_dict = {re.sub(pattern_sub, '', x):[] for x in patternlist}

        for pattern in patternlist:
            for fits in fitslist:
                if pattern in fits:
                    matching_dict[re.sub(pattern_sub, '', pattern)].append(fits)

        return matching_dict

    else:

        matching_dict = {x:[] for x in patternlist}

        for pattern in patternlist:
            for fits in fitslist:
                if pattern in fits:
                    matching_dict[pattern].append(fits)

        return matching_dict


def get_image_types(fitsdir,
                    fitsglob='*.fits*',
                    type_headerkey='IMAGETYP',
                    sort_by_file_pattern=None,
                    remove_pattern_regex=None,
                    fitsext=None,
                    nworkers=4,
                    maxworkertasks=1000):
    '''This goes through all the FITS in fitsdir, and sorts by image type.

    Image type means 'flat', 'zero', 'object', 'dark', etc.

    Use `sort_by_file_pattern` and `remove_pattern_regex` to further sort by
    some file pattern after sorting by image type, e.g. use this to sort by CCD
    ID after sorting by image type.

    Parameters
    ----------

    fitsdir : str
        The directory to operate on.

    fitsglob : str
        The UNIX file glob used to find FITS files.

    type_headerkey : str
        The FITS header keyword that indicates the image type.

    sort_by_file_pattern : list of str
        A list of patterns to break up the files in the input FITS list.

    remove_pattern_regex : raw str or None
        If `remove_pattern_regex` is not None, it must be a regex string
        specifying characters to remove from the patterns to form the keys of
        the dict returned, e.g. to turn the pattern '_1' in the example above to
        '1', use::

            remove_pattern_regex=r'_'

        This will turn the output dict from::

            {'_1':[list of files matching *_1.fits],
             '_2':[list of files matching *_2.fits],
            ...}

        into::

            {'1':[list of files matching *_1.fits],
             '2':[list of files matching *_2.fits],
            ...}

        which is much cleaner.

    fitsext : int or None
        The FITS extension to operate on for each image. If None, will
        automatically figure out the FITS extension.

    nworkers : int
        The number of parallel workers to launch to get FITS info.

    maxworkertasks : int
        The maximum number of tasks that a worker will operate on before it's
        replaced with a fresh one to guard against memory leaks.

    Returns
    -------

    dict
        This returns a dict with the images sorted by image type (and further by
        the provided file patterns in `sort_by_file_pattern` if that's not
        None). Also saves the results in the `fitsdir` as a pickle
        called ``fitsbits-imagetypes.pkl``.

    '''

    # get all the image types for all of the FITS files in this directory
    fitsdir_imagetypes = parallel_get_fits_headers(
        fitsdir,
        [type_headerkey],
        fitsglob=fitsglob,
        nworkers=nworkers,
        maxworkertasks=maxworkertasks,
        fitsext=fitsext
    )

    if not fitsdir_imagetypes:
        LOGERROR('No images matching %s found in %s, '
                 'skipping this directory' % (fitsglob, fitsdir))
        return None

    # get the distinct image types
    all_imagetypes = list(
        {fitsdir_imagetypes[x][type_headerkey] for x in fitsdir_imagetypes
         if (type_headerkey in fitsdir_imagetypes[x])}
    )

    outpicklepath = os.path.join(os.path.abspath(fitsdir),
                                 'fitsbits-imagetypes.pkl')
    outpicklef = open(outpicklepath,'wb')

    outpickledict = {'unknown_bad':[]}

    for imagetype in all_imagetypes:

        outpickledict[imagetype] = []

        for image in fitsdir_imagetypes:

            if (type_headerkey in fitsdir_imagetypes[image] and
                fitsdir_imagetypes[image][type_headerkey] == imagetype):
                outpickledict[imagetype].append(image)

            elif (type_headerkey not in fitsdir_imagetypes[image]):
                outpickledict['unknown_bad'].append(image)

    if sort_by_file_pattern:

        file_patterns = {x:[] for x in outpickledict}

        for x in outpickledict:
            outpickledict[x] = sort_fitslist_by_file_pattern(
                outpickledict[x],
                sort_by_file_pattern,
                remove_pattern_regex=remove_pattern_regex
            )
            file_patterns[x].extend(list(outpickledict[x].keys()))

    else:

        file_patterns = None

    outpickledict['metainfo'] = {
        'sort_by_file_pattern':sort_by_file_pattern,
        'remove_pattern_regex':remove_pattern_regex,
        'file_patterns':file_patterns,
    }

    pickle.dump(outpickledict, outpicklef, pickle.HIGHEST_PROTOCOL)
    outpicklef.close()

    LOGINFO('Done. Image type DB written to %s' %
            os.path.abspath(outpicklepath))
    return outpickledict


def get_observed_objects(fitsdir,
                         fitsglob='*.fits*',
                         type_headerkey='IMAGETYP',
                         object_headerkey='OBJECT',
                         use_imagetypes=('object','calibratedobject'),
                         sort_by_file_pattern=None,
                         remove_pattern_regex=None,
                         fitsext=None,
                         nworkers=4,
                         maxworkertasks=1000):
    '''This sorts the FITS files in a directory by observed objects.

    Parameters
    ----------

    fitsdir : str
        The directory to operate on.

    fitsglob : str
        The UNIX file glob used to find FITS files.

    type_headerkey : str
        The FITS header keyword that indicates the image type.

    object_headerkey : str
        The FITS header keyword that indicates the observed object.

    use_imagetype : str
        The type of FITS image (specified by FITS header key `type_headerkey`)
        that will be considered.

    sort_by_file_pattern : list of str
        A list of patterns to break up the files in the input FITS list.

    remove_pattern_regex : raw str or None
        If `remove_pattern_regex` is not None, it must be a regex string
        specifying characters to remove from the patterns to form the keys of
        the dict returned, e.g. to turn the pattern '_1' in the example above to
        '1', use::

            remove_pattern_regex=r'_'

        This will turn the output dict from::

            {'_1':[list of files matching *_1.fits],
             '_2':[list of files matching *_2.fits],
            ...}

        into::

            {'1':[list of files matching *_1.fits],
             '2':[list of files matching *_2.fits],
            ...}

        which is much cleaner.

    fitsext : int or None
        The FITS extension to operate on for each image. If None, will
        automatically figure out the FITS extension.

    nworkers : int
        The number of parallel workers to launch to get FITS info.

    maxworkertasks : int
        The maximum number of tasks that a worker will operate on before it's
        replaced with a fresh one to guard against memory leaks.

    Returns
    -------

    dict
        This returns a dict with the images sorted by observed object (and
        further by the provided file patterns in `sort_by_file_pattern` if
        that's not None). Also saves the results in the `fitsdir` as a pickle
        called ``fitsbits-observedobjects.pkl``.

    '''

    # get all the image types for all of the FITS files in this directory
    fitsdir_imageobjects = parallel_get_fits_headers(
        fitsdir,
        [type_headerkey, object_headerkey],
        fitsglob=fitsglob,
        nworkers=nworkers,
        maxworkertasks=maxworkertasks,
        fitsext=fitsext
    )

    if not fitsdir_imageobjects:
        LOGERROR('No images matching %s found in %s, '
                 'skipping this directory' % (fitsglob, fitsdir))
        return None

    # get the distinct fields, filtering out fields in the input ignore
    # kwarg, and making sure to only get frames of imagetype
    all_imageobjects = []

    for x in fitsdir_imageobjects:
        if fitsdir_imageobjects[x][type_headerkey] in use_imagetypes:
            all_imageobjects.append(fitsdir_imageobjects[x][object_headerkey])

    all_imageobjects = set(all_imageobjects)

    outpicklepath = os.path.join(os.path.abspath(fitsdir),
                                 'fitsbits-observedobjects.pkl')
    outpicklef = open(outpicklepath,'wb')

    outpickledict = {}

    for imageobject in all_imageobjects:

        outpickledict[imageobject] = []

        for image in fitsdir_imageobjects:

            if fitsdir_imageobjects[image][object_headerkey] == imageobject:
                outpickledict[imageobject].append(image)

    if sort_by_file_pattern:

        file_patterns = {x:[] for x in outpickledict}

        for x in outpickledict:
            outpickledict[x] = sort_fitslist_by_file_pattern(
                outpickledict[x],
                sort_by_file_pattern,
                remove_pattern_regex=remove_pattern_regex
            )
            file_patterns[x].extend(list(outpickledict[x].keys()))

    else:

        file_patterns = None

    outpickledict['metainfo'] = {
        'sort_by_file_pattern':sort_by_file_pattern,
        'remove_pattern_regex':remove_pattern_regex,
        'file_patterns':file_patterns,
    }

    pickle.dump(outpickledict, outpicklef, pickle.HIGHEST_PROTOCOL)
    outpicklef.close()

    LOGINFO('Done. Image object DB written to %s' %
            os.path.abspath(outpicklepath))

    return outpickledict


def group_frames_by_pointing(
        fits_list,
        outfile=None,
        max_radius_deg=7.5,
        min_fits_per_group=10,
        wcs_list=None,
        center_ra_key='CRVAL1',
        center_ra_unit='deg',
        center_decl_key='CRVAL2',
        center_decl_unit='deg',
        extract_targetfields=True,
        targetfield_extractor=extract_frame_targetfield,
        nworkers=NCPUS,
        return_dict=True,
):
    '''
    This groups the FITS images in the input list by their center coordinates.

    Parameters
    ----------

    fits_list : list of str
        The list of FITS images to process.

    outfile : str or None
        The output pickle to write the group info to. If None, this is
        'fitsbits-pointing-groups.pkl' in the current working directory.

    max_radius_deg : float
        The maximum radius to consider when generating groups.

    min_fits_per_group : int
        The minimum number of FITS required to consider a cluster as valid.

    wcs_list : list of str or None
        If this is provided, must contain a WCS file produced by astrometry.net
        for each frame in `fits_list`. This will be preferentially used to get
        the frame pointing values.

    center_ra_key : str
        The FITS header key containing the center RA value.

    center_ra_unit : {'deg', 'hr', 'hms'}
        The unit of the center RA value. This is used to convert it to decimal
        degrees if necessary.

    center_decl_key : str
        The FITS header key containing the center declination value.

    center_decl_unit : {'deg', 'dms'}
        The unit of the center declination value. This is used to convert it to
        decimal degrees if necessary.

    extract_targetfields : bool
        If True, will use the function in the `targetfield_extractor` kwarg to
        extract the name of the observed target object from the FITS header. If
        False, will not do so; this allows one to use this function to group
        other types of files given pointing information from accompanying .wcs
        files. For example, set this kwarg to False and grouping of .fiphot or
        .iphot files by pointing information in .wcs files is then possible.

    targetfield_extractor : Python function
        A Python function that extracts the target object from a given FITS
        file. This should have the signature below::

            def targetfield_extractor(fits, *args)

        and return a string value for the FITS file's target field.

    Returns
    -------

    dict or str
        If ``return_dict`` is True: a dict is returned containing all groups
        found as keys and the number of objects, the median center values, and
        the FITS files (and WCS files) associated with the groups as keys. If
        return_dict is False, a str filename of the output pickle generated is
        returned.

    '''

    if wcs_list is not None and isinstance(wcs_list, list):
        extract_centers_list = wcs_list
    else:
        extract_centers_list = fits_list

    center_vals = parallel_get_fits_headers_list(
        extract_centers_list,
        [center_ra_key, center_decl_key],
        nworkers=nworkers
    )

    center_ras = [
        center_vals[os.path.abspath(x)][center_ra_key]
        for x in extract_centers_list
    ]
    center_decls = [
        center_vals[os.path.abspath(x)][center_decl_key]
        for x in extract_centers_list
    ]

    if center_ra_unit == 'hr':
        center_ras = [x/24.0*360.0 for x in center_ras]
    elif center_ra_unit == 'hms':
        center_ras = [hms_str_to_decimal(x) for x in center_ras]

    if center_decl_unit == 'dms':
        center_decls = [dms_str_to_decimal(x) for x in center_decls]

    fits_list = np.array(fits_list)
    if wcs_list is not None:
        wcs_list = np.array(wcs_list)
    center_ras = np.array(center_ras)
    center_decls = np.array(center_decls)

    cosdecl = np.cos(np.radians(center_decls))
    sindecl = np.sin(np.radians(center_decls))
    cosra = np.cos(np.radians(center_ras))
    sinra = np.sin(np.radians(center_ras))
    xyz = np.column_stack((cosra*cosdecl,sinra*cosdecl, sindecl))
    xyzdist = 2.0 * np.sin(np.radians(max_radius_deg)/2.0)

    fits_clusters = DBSCAN(eps=xyzdist, min_samples=min_fits_per_group).fit(xyz)
    group_labels = fits_clusters.labels_
    unique_labels = np.unique(group_labels)

    groups = {}

    for label in unique_labels:

        group_ind = group_labels == label
        group_fits = fits_list[group_ind]
        if wcs_list is not None:
            group_wcs = wcs_list[group_ind]
        else:
            group_wcs = None

        group_ra = center_ras[group_ind]
        group_decl = center_decls[group_ind]
        center_ra = np.median(group_ra)
        center_decl = np.median(group_decl)

        if isinstance(targetfield_extractor, str):
            tfextfunc = object_from_string(targetfield_extractor)
            if not tfextfunc and extract_targetfields:
                LOGERROR("Could not import target "
                         "field extractor function: %s" % targetfield_extractor)
                return None
        else:
            tfextfunc = targetfield_extractor

        # get the target field for this group
        if extract_targetfields and tfextfunc is not None:
            target_field = tfextfunc(group_fits[0])
        else:
            target_field = None

        groups[label] = {
            'targetfield':target_field,
            'nfits':group_fits.size,
            'center_ra':center_ra,
            'center_decl':center_decl,
            'fits':group_fits,
            'wcs':group_wcs,
            'ra':group_ra,
            'decl':group_decl,
        }

        LOGINFO(
            "Group %s centered at (%.3f, %.3f), "
            "observing target field: %s, "
            "with %s images." % (
                label,
                center_ra,
                center_decl,
                target_field,
                group_fits.size
            )
        )

    groups['dbscan'] = fits_clusters
    groups['labels'] = group_labels
    groups['unique_labels'] = unique_labels
    groups['kwargs'] = {
        'fits_list':fits_list,
        'max_radius_deg':max_radius_deg,
        'min_fits_per_group':min_fits_per_group,
        'wcs_list':wcs_list,
        'center_ra_key':center_ra_key,
        'center_ra_unit':center_ra_unit,
        'center_decl_key':center_decl_key,
        'center_decl_unit':center_decl_unit,
    }

    if not outfile:
        outfile = 'fitsbits-pointing-groups.pkl'

    with open(outfile, 'wb') as outfd:
        pickle.dump(groups, outfd, pickle.HIGHEST_PROTOCOL)

    return outfile


def filter_frames_by_headerkeys(fits_list,
                                filter_by,
                                nworkers=NCPUS,
                                maxworkertasks=1000,
                                fitsext=None):
    '''This filters frames by header keys.

    Parameters
    ----------

    fits_list : list of str
        A list of FITS files to process.

    filter_by : tuple
        This is a tuple of two elements:

        - To filter by enforcing specific header keys and values::

            filter_by = ('keyval', ('key1','key2',...), (val1, val2,,...))

          The keys will be up-cased automatically so ALL CAPS keys aren't
          required.

        - To filter by requiring that a regex matched key list have a specific
          value::

            filter_by = ('regex', 'regex pattern', val)

    nworkers : int or None
        The number of parallel workers to launch.

    maxworkertasks : int
        The maximum number of tasks that a worker will operate on before it's
        replaced with a fresh one to guard against memory leaks.

    fitsext : int or None
        The FITS extension to operate on for each image. If None, will
        automatically figure out the FITS extension.

    Returns
    -------

    list
        Returns a list of the matching FITS files.

    '''

    if filter_by[0] == 'keyval':

        filter_keys, filter_vals = filter_by[1:]

        fits_kv = parallel_get_fits_headers_list(
            fits_list,
            [x.upper() for x in filter_keys],
            nworkers=nworkers,
            maxworkertasks=maxworkertasks,
            fitsext=fitsext,
        )

        filtered_fits = []

        for f in fits_kv:

            if all(fits_kv[f][x.upper()] == y
                   for x, y in zip(filter_keys, filter_vals)):
                filtered_fits.append(f)

        return filtered_fits

    elif filter_by[0] == 'regex':

        filter_regex, filter_val = filter_by[1:]

        fits_kv = parallel_regex_fits_headers_list(
            fits_list,
            filter_regex,
            nworkers=nworkers,
            maxworkertasks=maxworkertasks,
            fitsext=fitsext,
        )

        filtered_fits = []

        for f in fits_kv:

            if all(fits_kv[f][x.upper()] == filter_val
                   for x in fits_kv[f].keys()):
                filtered_fits.append(f)

        return filtered_fits

    else:

        raise ValueError("filter_by[0] must be in {'keyval', 'regex'}")

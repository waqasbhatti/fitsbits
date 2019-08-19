# -*- coding: utf-8 -*-

'''This contains functions to safely compress and uncompress files.

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
import tempfile
import shutil

import filelock

from ._processes import run_shell


########################
## PATCHING SOME BITS ##
########################

# turn off the verbose logging in the filelock module
class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def critical(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


filelock.logger = DummyLogger


#############################################
## COMPRESSION AND DECOMPRESSION FUNCTIONS ##
#############################################

def safe_compress(infile,
                  compression='gzip',
                  remove_original=True,
                  lock_timeout_sec=30.0):
    '''
    This does a safe (paranoid) compression operation.

    1. the file to be worked on is locked
    2. the file to be worked on is copied to a temp input file
    3. the temp input file is operated upon to produce a temp output file
    4. the temp out file is moved to the expected output file
    5. the file to be worked on is removed if requested
    6. the file to be worked on is unlocked

    '''

    lock_file = os.path.join(os.path.dirname(infile),
                             '%s.lock' % os.path.basename(infile))
    if compression == 'gzip':
        outfile = os.path.join(os.path.dirname(infile),
                               '%s.gz' % os.path.basename(infile))
    elif compression == 'fpack':
        outfile = os.path.join(os.path.dirname(infile),
                               '%s.fz' % os.path.basename(infile))
    else:
        LOGERROR(
            "Unknown compression type requested: %s for file: %s" %
            (compression, infile)
        )
        return None

    # lock first
    flock = filelock.FileLock(lock_file)
    success = False

    try:

        # input file name for operation
        in_tempfd, in_tempfile = tempfile.mkstemp(
            dir=os.path.dirname(infile),
        )
        # output file name for operation
        out_tempfd, out_tempfile = tempfile.mkstemp(
            dir=os.path.dirname(infile),
        )

        with flock.acquire(timeout=lock_timeout_sec):

            shutil.copy(infile, in_tempfile)

            # perform the operation on the temp file
            if compression == 'gzip':
                cmd = 'gzip -c %s > %s' % (in_tempfile, out_tempfile)
            elif compression == 'fpack':
                cmd = 'fpack -C -Y -S %s > %s' % (in_tempfile, out_tempfile)

            success = run_shell(cmd,
                                raise_exceptions=False,
                                timeout=lock_timeout_sec)

            if not success:
                LOGERROR("Could not %s input file: %s" % (compression, infile))
            else:
                # if we succeded, move the temp_outfile to outfile
                os.replace(out_tempfile, outfile)

    except filelock.Timeout:

        LOGERROR(
            "Could not %s input file: %s because "
            "it is locked by another process."
            % (compression, infile)
        )
        success = False

    finally:

        # remove the input temp file
        try:
            os.unlink(in_tempfile)
        except Exception:
            pass

        # actually close the in and out tempfile FDs
        # if this isn't done, we get the dreaded "Too many open files" error
        try:
            os.close(in_tempfd)
        except Exception:
            pass
        try:
            os.close(out_tempfd)
        except Exception:
            pass

        # if we're going to remove the input file, do so
        if success and remove_original:
            os.unlink(infile)

        # we don't remove the lockfile to avoid race conditions where another
        # worker might try to open our file relying on the lockfile's existence
        # but it disappeared because we removed it here.

    #
    # the lock either expires or is successfully released
    #
    if success:
        return outfile
    else:
        return None


def safe_uncompress(infile,
                    lock_timeout_sec=30.0,
                    remove_original=True):
    '''
    This does a safe (paranoid) uncompression operation.

    1. the file to be worked on is locked
    2. the file to be worked on is copied to a temp input file
    3. the temp input file is operated upon to produce a temp output file
    4. the temp out file is moved to the expected output file
    5. the file to be worked on is removed if requested
    6. the file to be worked on is unlocked

    The type of compression used is taken from the extension of the file.

    - '.gz' -> does gunzip
    - '.fz' -> does funpack

    '''

    lock_file = os.path.join(os.path.dirname(infile),
                             '%s.lock' % os.path.basename(infile))

    if infile.endswith('.gz'):
        outfile = os.path.join(os.path.dirname(infile),
                               os.path.basename(infile).replace('.gz',''))
    elif infile.endswith('.fz'):
        outfile = os.path.join(os.path.dirname(infile),
                               os.path.basename(infile).replace('.fz',''))
    else:
        LOGERROR(
            "Unknown compressed file type for: %s." % infile
        )
        return None

    # lock first
    flock = filelock.FileLock(lock_file)
    success = False

    try:

        # input file name for operation
        in_tempfd, in_tempfile = tempfile.mkstemp(
            dir=os.path.dirname(infile),
        )
        # output file name for operation
        out_tempfd, out_tempfile = tempfile.mkstemp(
            dir=os.path.dirname(infile),
        )

        with flock.acquire(timeout=lock_timeout_sec):

            shutil.copy(infile, in_tempfile)

            # perform the operation on the temp file
            if infile.endswith('.gz'):
                cmd = 'gunzip -c %s > %s' % (in_tempfile, out_tempfile)
            elif infile.endswith('.fz'):
                cmd = 'funpack -C -S %s > %s' % (in_tempfile, out_tempfile)

            success = run_shell(cmd,
                                raise_exceptions=False,
                                timeout=lock_timeout_sec)

            if not success:
                LOGERROR("Could not uncompress input file: %s" % infile)
            else:
                # if we succeded, move the temp_outfile to outfile
                os.replace(out_tempfile, outfile)

    except filelock.Timeout:

        LOGERROR(
            "Could not uncompress input file: %s because "
            "it is locked by another process."
            % infile
        )
        success = False

    finally:

        # actually close the in and out tempfile FDs
        # if this isn't done, we get the dreaded "Too many open files" error
        try:
            os.close(in_tempfd)
        except Exception:
            pass
        # remove the input temp file
        try:
            os.unlink(in_tempfile)
        except Exception:
            pass

        try:
            os.close(out_tempfd)
        except Exception:
            pass
        try:
            os.unlink(out_tempfile)
        except Exception:
            pass

        # if we're going to remove the input file, do so
        if success and remove_original:
            os.unlink(infile)

        # we don't remove the lockfile to avoid race conditions where another
        # worker might try to open our file relying on the lockfile's existence
        # but it disappeared because we removed it here.

    #
    # the lock either expires or is successfully released
    #
    if success:
        return outfile
    else:
        return None


##############################
## SAFE COMPRESSION WRAPPER ##
##############################

def with_safe_compression(
        infile_args,
        infile_kwargs,
        outfile_compression,
        lock_timeout,
        wrapped_function,
        *wrapped_function_args,
        **wrapped_function_kwargs
):
    '''This wraps any other function with safe decompression and compression of
    the input and output files.

    Parameters
    ----------

    infile_args : list of bool
        This indicates which of the args of the function are to be decompressed
        automatically and then recompress after the wrapped function finishes.

    infile_kwargs : list of str
        This indicates which of the kwargs of the function are to be
        decompressed automatically and then recompress after the wrapped
        function finishes.

    outfile_compression : tuple of str
        This indicates how to compress any output files. This should be the same
        length as the function's returned values and contain strings from the
        set: {'gzip', 'fpack', None} to indicate how each output will be
        compressed.

    lock_timeout : int
        The timeout for the file locks in seconds. This should be as long as the
        longest timeout associated with the wrapped function.

    wrapped_function : Python function
        The function that will be run.

    wrapped_function_args : args
        The args for the function to run.

    wrapped_function_kwargs : kwargs
        The kwargs for the function to run.

    Returns
    -------

    returned_val : object
        Returns whatever the wrapped function returns.

    Notes
    -----

    The wrapped function should not try to automatically guess the output file's
    name because it'll be given a temporary file to work on, resulting in
    incorrect naming for any output files produced. This means all outfile=None
    type kwargs should actually be populated with the final output file's name.

    '''

    file_locks = {}
    args_to_use = list(wrapped_function_args)
    if wrapped_function_kwargs is None:
        kwargs_to_use = {}
    else:
        kwargs_to_use = wrapped_function_kwargs
    wrapped_function_returned = None
    reform_output = False

    try:

        # make locks on all the input files in the args
        for ind, f, handle in zip(range(len(args_to_use)),
                                  args_to_use,
                                  infile_args):

            # break out if we're going to ignore the rest of args
            if handle is ...:
                break

            if not handle:
                continue

            # special case of when the input arg is a list
            # e.g. for combine_frames
            if isinstance(f, list):

                for hnind, lf, hn in zip(range(len(f)), f, handle):

                    # break out if we're going to ignore the rest of items in
                    # this list
                    if hn is ...:
                        break

                    if not hn:
                        continue

                    lock_file = os.path.join(
                        os.path.dirname(lf),
                        '%s.lock' % os.path.basename(lf)
                    )
                    flock = filelock.FileLock(lock_file)
                    flock.acquire(timeout=lock_timeout)

                    in_tempfd, in_tempfile = tempfile.mkstemp(
                        dir=os.path.dirname(lf)
                    )
                    os.close(in_tempfd)

                    file_locks[f] = {'lockfile':lock_file,
                                     'lock':flock,
                                     'tempfd':in_tempfd,
                                     'tempfile':in_tempfile}

                    # copy this infile to the temporary file
                    shutil.copy(f, in_tempfile)

                    # generate the name of the uncompressed tempfile
                    uncompressed_tempfd, uncompressed_tempfile = (
                        tempfile.mkstemp(
                            dir=os.path.dirname(in_tempfile),
                        )
                    )
                    os.close(uncompressed_tempfd)
                    file_locks[uncompressed_tempfile] = {
                        'lockfile':None,
                        'lock':None,
                        'tempfd':None,
                        'tempfile':uncompressed_tempfile
                    }

                    if f.endswith('.gz'):
                        cmd = 'gunzip -c %s > %s' % (in_tempfile,
                                                     uncompressed_tempfile)
                        decomp = True
                    elif f.endswith('.fz'):
                        cmd = 'funpack -C -S %s > %s' % (in_tempfile,
                                                         uncompressed_tempfile)
                        decomp = True
                    else:
                        decomp = False

                    if decomp:
                        success = run_shell(cmd,
                                            raise_exceptions=False,
                                            timeout=lock_timeout)
                        if not success:
                            raise IOError(
                                "Could not uncompress input file: %s" % lf
                            )

                        # add the uncompressed tempfile to the input file args
                        # in place of the actual input file for the wrapped
                        # function
                        args_to_use[ind][hnind] = uncompressed_tempfile

            # normal case of a single file argument
            else:

                lock_file = os.path.join(
                    os.path.dirname(f),
                    '%s.lock' % os.path.basename(f)
                )
                flock = filelock.FileLock(lock_file)
                flock.acquire(timeout=lock_timeout)

                in_tempfd, in_tempfile = tempfile.mkstemp(
                    dir=os.path.dirname(f)
                )
                os.close(in_tempfd)

                file_locks[f] = {'lockfile':lock_file,
                                 'lock':flock,
                                 'tempfd':in_tempfd,
                                 'tempfile':in_tempfile}

                # copy this infile to the temporary file
                shutil.copy(f, in_tempfile)

                # generate the name of the uncompressed tempfile
                uncompressed_tempfd, uncompressed_tempfile = tempfile.mkstemp(
                    dir=os.path.dirname(in_tempfile),
                )
                os.close(uncompressed_tempfd)
                file_locks[uncompressed_tempfile] = {
                    'lockfile':None,
                    'lock':None,
                    'tempfd':None,
                    'tempfile':uncompressed_tempfile
                }

                if f.endswith('.gz'):
                    cmd = 'gunzip -c %s > %s' % (in_tempfile,
                                                 uncompressed_tempfile)
                    decomp = True
                elif f.endswith('.fz'):
                    cmd = 'funpack -C -S %s > %s' % (in_tempfile,
                                                     uncompressed_tempfile)
                    decomp = True
                else:
                    decomp = False

                if decomp:
                    success = run_shell(cmd,
                                        raise_exceptions=False,
                                        timeout=lock_timeout)
                    if not success:
                        raise IOError("Could not uncompress input file: %s" % f)

                    # add the uncompressed tempfile to the input file args in
                    # place of the actual input file for the wrapped function
                    args_to_use[ind] = uncompressed_tempfile

        # make locks on all the input files in the kwargs
        for kwarg in infile_kwargs:

            if kwarg not in wrapped_function_kwargs:
                continue

            f = wrapped_function_kwargs[kwarg]

            lock_file = os.path.join(
                os.path.dirname(f),
                '%s.lock' % os.path.basename(f)
            )
            flock = filelock.FileLock(lock_file)
            flock.acquire(timeout=lock_timeout)

            in_tempfd, in_tempfile = tempfile.mkstemp(
                dir=os.path.dirname(f)
            )

            file_locks[f] = {'lockfile':lock_file,
                             'lock':flock,
                             'tempfd':in_tempfd,
                             'tempfile':in_tempfile}

            # copy this infile to the temporary file
            shutil.copy(f, in_tempfile)

            # generate the name of the uncompressed tempfile
            uncompressed_tempfd, uncompressed_tempfile = tempfile.mkstemp(
                dir=os.path.dirname(in_tempfile),
            )
            os.close(uncompressed_tempfd)
            file_locks[uncompressed_tempfile] = {
                'lockfile':None,
                'lock':None,
                'tempfd':None,
                'tempfile':uncompressed_tempfile
            }

            if f.endswith('.gz'):
                cmd = 'gunzip -c %s > %s' % (in_tempfile,
                                             uncompressed_tempfile)
                decomp = True
            elif f.endswith('.fz'):
                cmd = 'funpack -C -S %s > %s' % (in_tempfile,
                                                 uncompressed_tempfile)
                decomp = True
            else:
                decomp = False

            if decomp:
                success = run_shell(cmd,
                                    raise_exceptions=False,
                                    timeout=lock_timeout)
                if not success:
                    raise IOError("Could not uncompress input file: %s" % f)

                # add the uncompressed tempfile to the input file kwargs in
                # place of the actual input file for the wrapped function
                kwargs_to_use[kwarg] = in_tempfile

        #
        # now all args and kwargs have been replaced with their uncompressed
        # equivalents. run the actual function.
        #
        args_to_use = tuple(args_to_use)

        wrapped_function_returned = wrapped_function(
            *args_to_use,
            **kwargs_to_use
        )

        #
        # handle the output files now
        #
        if isinstance(wrapped_function_returned, tuple):
            wrapped_function_returned = list(wrapped_function_returned)
            reform_output = 'tuple'
        else:
            wrapped_function_returned = [wrapped_function_returned]
            reform_output = 'single'

        for ind, outfile, compression in zip(
                range(len(wrapped_function_returned)),
                wrapped_function_returned,
                outfile_compression
        ):

            if compression is None or outfile is None:
                continue

            lock_file = os.path.join(
                os.path.dirname(outfile),
                '%s.lock' % os.path.basename(outfile)
            )
            flock = filelock.FileLock(lock_file)
            flock.acquire(timeout=lock_timeout)

            out_tempfd, out_tempfile = tempfile.mkstemp(
                dir=os.path.dirname(outfile)
            )
            os.close(out_tempfd)

            file_locks[outfile] = {'lockfile':lock_file,
                                   'lock':flock,
                                   'tempfd':out_tempfd,
                                   'tempfile':out_tempfile}

            # copy this infile to the temporary file
            shutil.copy(outfile, out_tempfile)

            # generate the name of the compressed tempfile
            compressed_tempfd, compressed_tempfile = tempfile.mkstemp(
                dir=os.path.dirname(out_tempfile),
            )
            os.close(compressed_tempfd)
            file_locks[compressed_tempfile] = {
                'lockfile':None,
                'lock':None,
                'tempfd':None,
                'tempfile':compressed_tempfile
            }

            # perform the operation on the temp file
            if compression == 'gzip':
                cmd = 'gzip -c %s > %s' % (out_tempfile, compressed_tempfile)
                ext = 'gz'
            elif compression == 'fpack':
                cmd = 'fpack -C -Y -S %s > %s' % (out_tempfile,
                                                  compressed_tempfile)
                ext = 'fz'

            success = run_shell(cmd,
                                raise_exceptions=False,
                                timeout=lock_timeout)

            if not success:
                raise IOError("Could not compress output file: %s" % outfile)

            # if compression succeeds, remove the temp output
            else:

                compressed_outfile = '%s.%s' % (outfile, ext)

                os.replace(compressed_tempfile, compressed_outfile)
                os.unlink(out_tempfile)
                os.unlink(outfile)

                # replace the output with the compressed output
                wrapped_function_returned[ind] = compressed_outfile

    finally:

        # release all the file locks
        for f in file_locks:

            #
            # Bonus fcntl.flock insanity: https://apenwarr.ca/log/20101213
            # We release locks FIRST to avoid the likely issue of fcntl locks
            # disappearing mysteriously as described there. Still not sure if
            # the subprocess call causes this issue (since close_fds is True by
            # default).
            #

            try:
                # release the lock ASAP since we're done with the input file
                file_locks[f]['lock'].release()
            except IOError as e:
                if e.errno == 9:
                    LOGWARNING(
                        "Lock file %s already closed." %
                        file_locks[f]['lock'].lock_file
                    )
                else:
                    LOGEXCEPTION("Could not unlock file: %s" % f)
            except Exception:
                pass

            # actually close the in tempfile FDs
            # if this isn't done, we get the dreaded "Too many open files" error
            try:
                if file_locks[f]['tempfd'] is not None:
                    os.close(file_locks[f]['tempfd'])
            except Exception:
                pass

            # remove the input temp file
            try:
                if file_locks[f]['tempfile'] is not None:
                    os.unlink(file_locks[f]['tempfile'])
            except Exception:
                pass

            # we don't remove the lockfile to avoid race conditions where
            # another worker might try to open our file relying on the
            # lockfile's existence but it disappeared because we removed it
            # here.

    #
    # if we made it to here, return the wrapped function's output
    #
    if reform_output == 'tuple':
        return tuple(wrapped_function_returned)
    else:
        return wrapped_function_returned[0]

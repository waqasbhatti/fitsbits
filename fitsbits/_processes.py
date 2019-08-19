# -*- coding: utf-8 -*-

'''This contains a shell subprocess driver function.

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

import subprocess
import shutil
import os
import os.path
from time import monotonic
import tempfile

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random_exponential
)
import filelock


#############
## RUNNERS ##
#############

def run_shell(command,
              debug=False,
              timeout=60.0,
              redirect_stdout=subprocess.PIPE,
              redirect_stderr=subprocess.PIPE,
              close_fds=True,
              raise_exceptions=True,
              return_outerr=False,
              time_process=False):
    '''
    This runs a subprocess with shell=True with a configurable timeout.

    '''

    if debug:
        LOGINFO('Command line used: %s' % command)

    start_time = monotonic()
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=redirect_stdout,
        stderr=redirect_stderr,
        close_fds=close_fds,
    )

    outs, errs = None, None

    try:

        outs, errs = proc.communicate(
            timeout=timeout
        )
        time_taken = monotonic() - start_time

        # if the process failed
        if proc.returncode != 0:

            time_taken = monotonic() - start_time

            LOGERROR('Process failed, returncode: %s. '
                     'Command line used: %s' % (proc.returncode, command))
            if errs is not None:
                LOGERROR('Returned stderr: %s' % errs.decode())
            if outs is not None:
                LOGERROR('Returned stdout: %s' % outs.decode())

            if time_process:
                returnval = (False, time_taken)
            else:
                returnval = False

            if raise_exceptions:
                raise subprocess.CalledProcessError(
                    proc.returncode,
                    command,
                    output=outs,
                    stderr=errs,
                )

        # otherwise, it succeeded
        else:

            if time_process:
                returnval = (True, time_taken)
            else:
                returnval = True

    except subprocess.TimeoutExpired:

        proc.kill()
        try:
            kouts, kerrs = proc.communicate(
                timeout=0.1
            )
        except Exception:
            kouts, kerrs = None, None

        LOGERROR('Process timed out after %s seconds. '
                 'Command line used: %s' % (timeout, command))

        if time_process is True:
            returnval = (False, timeout)
        else:
            returnval = False

        if kouts is not None:
            outs = kouts
        if kerrs is not None:
            errs = kerrs

        if raise_exceptions:
            raise

    except Exception:

        proc.kill()
        LOGEXCEPTION("Process failed and raised an exception.")
        LOGERROR('Command line used: %s' % command)

        time_taken = monotonic() - start_time

        if time_process is True:
            returnval = (False, timeout)
        else:
            returnval = False

        if raise_exceptions:
            raise

    if return_outerr:
        return returnval, outs, errs
    else:
        return returnval


def run_shell_retry(command,
                    debug=False,
                    timeout=60.0,
                    redirect_stdout=subprocess.PIPE,
                    redirect_stderr=subprocess.PIPE,
                    return_outerr=False,
                    raise_exceptions=False,
                    close_fds=True,
                    retries=1,
                    retry_wait=1.0):
    '''This runs a subprocess with shell=True, a configured timeout for the
    initial run, and a configurable number of retries with a constant wait time.

    '''

    @retry(stop=stop_after_attempt(retries),
           wait=wait_fixed(retry_wait),
           reraise=raise_exceptions)
    def runner(command, debug, timeout, close_fds, return_outerr,
               redirect_stdout, redirect_stderr):
        return run_shell(command,
                         redirect_stdout=redirect_stdout,
                         redirect_stderr=redirect_stderr,
                         debug=debug,
                         timeout=timeout,
                         close_fds=close_fds,
                         return_outerr=return_outerr)

    return runner(command, debug, timeout, close_fds, return_outerr,
                  redirect_stdout, redirect_stderr)


def run_shell_expbackoff(command,
                         debug=False,
                         timeout=60.0,
                         close_fds=True,
                         redirect_stdout=subprocess.PIPE,
                         redirect_stderr=subprocess.PIPE,
                         return_outerr=False,
                         raise_exceptions=False,
                         retries=4,
                         retry_multiplier=1.0,
                         retry_maxtime=30.0):
    '''This runs a subprocess with shell=True, a configured timeout
    for the initial run, and a configurable number of retries
    with an exponential backoff.

    '''

    @retry(stop=stop_after_attempt(retries),
           wait=wait_random_exponential(multiplier=retry_multiplier,
                                        max=retry_maxtime),
           reraise=raise_exceptions)
    def runner(command, debug, timeout, close_fds, return_outerr,
               redirect_stdout, redirect_stderr):
        return run_shell(command,
                         debug=debug,
                         redirect_stdout=redirect_stdout,
                         redirect_stderr=redirect_stderr,
                         timeout=timeout,
                         close_fds=close_fds,
                         return_outerr=return_outerr)

    return runner(command, debug, timeout, close_fds, return_outerr,
                  redirect_stdout, redirect_stderr)


###################################
## SAFE SUBPROCESS RUN FUNCTIONS ##
###################################

def run_safe(infile,
             outfile,
             command,
             debug=False,
             timeout=60.0,
             close_fds=True,
             redirect_stdout=subprocess.PIPE,
             redirect_stderr=subprocess.PIPE,
             raise_exceptions=True,
             return_outerr=False,
             time_process=False):
    '''
    This runs a subprocess with shell=True with a configurable timeout.

    This version tries to run things safely with file locks.

    infile is either a list of files or a single file.

    '''

    if isinstance(infile,str):
        infile_list = [infile]
    else:
        infile_list = infile

    success = False
    proc_results = False

    # output file name for operation
    out_tempfd, out_tempfile = tempfile.mkstemp(
        dir=os.path.dirname(outfile),
    )
    os.close(out_tempfd)

    locks = {}
    safe_command = command[::]

    try:

        # make locks on all the input files
        for f in infile_list:

            lock_file = os.path.join(os.path.dirname(f),
                                     '%s.lock' % os.path.basename(f))
            flock = filelock.FileLock(lock_file)
            flock.acquire(timeout=timeout)

            in_tempfd, in_tempfile = tempfile.mkstemp(
                dir=os.path.dirname(f)
            )

            # copy the this infile to the temporary file
            shutil.copy(f, in_tempfile)

            locks[f] = {'lockfile':lock_file,
                        'lock':flock,
                        'tempfd':in_tempfd,
                        'tempfile':in_tempfile}

            # replace all instances of infile with their temp
            # equivalents
            safe_command = safe_command.replace(
                f,
                in_tempfile
            )

        # replace the instance of the outfile with the temp outfile
        safe_command = safe_command.replace(
            outfile,
            out_tempfile
        )

        proc_results = run_shell(
            safe_command,
            debug=debug,
            timeout=timeout,
            close_fds=close_fds,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            raise_exceptions=raise_exceptions,
            return_outerr=return_outerr,
            time_process=time_process
        )

        if isinstance(proc_results, tuple) and not time_process:
            success = proc_results[0]
        elif isinstance(proc_results, tuple) and time_process:
            success = proc_results[0][0]
        else:
            success = proc_results

        if success:
            os.replace(out_tempfile, outfile)

    except filelock.Timeout:

        LOGERROR(
            "Could not run shell comand on input files: %r because "
            "they are locked by another process." % infile_list
        )
        success = False

    except Exception:

        success = False
        if raise_exceptions:
            raise

    finally:

        # release all the input file locks
        for f in locks:

            #
            # Bonus fcntl.flock insanity: https://apenwarr.ca/log/20101213
            # We release locks FIRST to avoid the likely issue of fcntl locks
            # disappearing mysteriously as described there. Still not sure if
            # the subprocess call causes this issue (since close_fds is True by
            # default).
            #

            try:
                # release the lock ASAP since we're done with the input file
                locks[f]['lock'].release()
            except IOError as e:
                if e.errno == 9:
                    LOGWARNING(
                        "Lock file %s already closed." %
                        locks[f]['lock'].lock_file
                    )
                else:
                    LOGEXCEPTION("Could not unlock file: %s" % f)
            except Exception:
                pass

            # remove the input temp file
            try:
                os.unlink(locks[f]['tempfile'])
            except Exception:
                pass

            # actually close the in_tempfile FDs
            # if this isn't done, we get the dreaded "Too many open files" error
            try:
                os.close(locks[f]['tempfd'])
            except Exception:
                pass

            # we don't remove the lockfile to avoid race conditions where
            # another worker might try to open our file relying on the
            # lockfile's existence but it disappeared because we removed it
            # here.

        # remove the output file FD
        try:
            os.close(out_tempfd)
        except Exception:
            pass

        # remove the output temp file
        try:
            os.unlink(out_tempfile)
        except Exception:
            pass

    #
    # the lock either expires or is successfully released
    #
    return proc_results


def run_safe_retry(infile,
                   outfile,
                   command,
                   debug=False,
                   timeout=60.0,
                   close_fds=True,
                   redirect_stdout=subprocess.PIPE,
                   redirect_stderr=subprocess.PIPE,
                   return_outerr=False,
                   raise_exceptions=False,
                   retries=1,
                   retry_wait=1.0):
    '''This runs a subprocess with shell=True, a configured timeout for the
    initial run, and a configurable number of retries with a constant wait time.

    '''

    @retry(stop=stop_after_attempt(retries),
           wait=wait_fixed(retry_wait),
           reraise=raise_exceptions)
    def runner(infile, outfile,
               command, debug, timeout, close_fds, return_outerr,
               redirect_stdout, redirect_stderr):
        return run_safe(infile,
                        outfile,
                        command,
                        redirect_stdout=redirect_stdout,
                        redirect_stderr=redirect_stderr,
                        debug=debug,
                        timeout=timeout,
                        close_fds=close_fds,
                        return_outerr=return_outerr)

    return runner(infile, outfile,
                  command, debug, timeout, close_fds, return_outerr,
                  redirect_stdout, redirect_stderr)


def run_safe_expbackoff(infile,
                        outfile,
                        command,
                        debug=False,
                        timeout=60.0,
                        close_fds=True,
                        redirect_stdout=subprocess.PIPE,
                        redirect_stderr=subprocess.PIPE,
                        return_outerr=False,
                        raise_exceptions=False,
                        retries=4,
                        retry_multiplier=1.0,
                        retry_maxtime=30.0):
    '''This runs a subprocess with shell=True, a configured timeout
    for the initial run, and a configurable number of retries
    with an exponential backoff.

    '''

    @retry(stop=stop_after_attempt(retries),
           wait=wait_random_exponential(multiplier=retry_multiplier,
                                        max=retry_maxtime),
           reraise=raise_exceptions)
    def runner(infile, outfile,
               command, debug, timeout, close_fds, return_outerr,
               redirect_stdout, redirect_stderr):
        return run_safe(infile, outfile,
                        command,
                        debug=debug,
                        redirect_stdout=redirect_stdout,
                        redirect_stderr=redirect_stderr,
                        timeout=timeout,
                        close_fds=close_fds,
                        return_outerr=return_outerr)

    return runner(infile, outfile,
                  command, debug, timeout, close_fds, return_outerr,
                  redirect_stdout, redirect_stderr)

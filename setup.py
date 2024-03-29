# -*- coding: utf-8 -*-

'''setup.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This sets up the package.

Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and
modified by me.

'''
import versioneer
__version__ = versioneer.get_version()

import sys
from setuptools import setup, find_packages

# pytesting stuff and imports copied wholesale from:
# https://docs.pytest.org/en/latest/goodpractices.html#test-discovery
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest

        if not self.pytest_args:
            targs = []
        else:
            targs = shlex.split(self.pytest_args)

        errno = pytest.main(targs)
        sys.exit(errno)


# set up the cmdclass
cmdclass = versioneer.get_cmdclass()
cmdclass['test'] = PyTest


# get the readme
def readme():
    with open('README.md') as f:
        return f.read()


# let's be lazy and put requirements in one place
# what could possibly go wrong?
with open('requirements.txt') as infd:
    INSTALL_REQUIRES = [x.strip('\n') for x in infd.readlines()]


EXTRAS_REQUIRE = {}

###############
## RUN SETUP ##
###############

setup(
    name='fitsbits',
    version=__version__,
    cmdclass=cmdclass,
    description=("Utilities for FITS files: safe (de)compression, "
                 "exporting to images/movies, parallelized ops on "
                 "collections, and QA"),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords='astronomy,FITS',
    url='https://github.com/waqasbhatti/fitsbits',
    author='Waqas Bhatti',
    author_email='waqas.afzal.bhatti@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=['pytest',],
    entry_points={
        'console_scripts':[
            'fitsbits-header=fitsbits.fitshdr:main',
            'fitsbits-export=fitsbits.fits2export:main',
            'fitsbits-movie=fitsbits.fits2mp4:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6',
)

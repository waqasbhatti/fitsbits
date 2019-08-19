FITSbits is a Python package that contains various utilities for handling FITS
files.

To install **[fitsbits](https://pypi.org/project/fitsbits/)** from the
Python Package Index (PyPI):

```bash
$ pip install fitsbits
```

See the [installation instructions](#installation) below for details.

This package requires Python >= 3.5.

# Contents

- [**compression.py**](https://github.com/waqasbhatti/fitsbits/blob/master/fitsbits/compression.py):
  contains functions to perform safe and atomic compression and decompression of
  FITS files in gzip and fpack formats. Requires the GNU gzip and CFITSIO
  fpack/funpack utilities.

- [**export.py**](https://github.com/waqasbhatti/fitsbits/blob/master/fitsbits/export.py):
  contains functions to export FITS images as JPEGs, generating stamps for FITS
  images and exporting these as PNGs, and turning collections of FITS JPEGs into
  movies.

- [**files.py**](https://github.com/waqasbhatti/fitsbits/blob/master/fitsbits/files.py):
  contains functions to work on collections of FITS files.

- [**fits2jpeg.py**](https://github.com/waqasbhatti/fitsbits/blob/master/fitsbits/fits2jpeg.py):
  command-line utility available as `fitsbits-fits2jpeg` when this package is
  installed; converts a FITS image to a JPEG.

- [**fits2mp4.py**](https://github.com/waqasbhatti/fitsbits/blob/master/fitsbits/fits2mp4.py):
  command-line utility available as `fitsbits-fits2mp4` when this package is
  installed; converts a series of FITS images to a MP4 movie.

- [**fitshdr.py**](https://github.com/waqasbhatti/fitsbits/blob/master/fitsbits/fitshdr.py):
  command-line utility available as `fitsbits-header` when this package is
  installed; extracts and dumps a FITS file's header to stdout.

- [**operations.py**](https://github.com/waqasbhatti/fitsbits/blob/master/fitsbits/operations.py):
  contains functions to perform various header and data extraction operations on
  FITS files.

- [**quality.py**](https://github.com/waqasbhatti/fitsbits/blob/master/fitsbits/quality.py):
  contains functions to assess the quality of FITS images.


# Changelog

Please see https://github.com/waqasbhatti/fitsbits/blob/master/CHANGELOG.md for
a list of changes applicable to tagged release versions.


# Installation

## Requirements

This package requires the following other packages:

- numpy
- scipy
- astropy
- matplotlib
- Pillow
- filelock
- tenacity

## Installing with pip

You can install fitsbits with:

```bash

(venv)$ pip install fitsbits
```

### Other installation methods

Install the latest version (may be unstable at times):

```bash
$ git clone https://github.com/waqasbhatti/fitsbits
$ cd fitsbits
$ python setup.py install
$ # or use pip install . to install requirements automatically
$ # or use pip install -e . to install in develop mode along with requirements
```

# License

`fitsbits` is provided under the MIT License. See the LICENSE file for the full
text.

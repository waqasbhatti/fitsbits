from ._version import get_versions
__version__ = get_versions()['version']
__gitrev__ = get_versions()['full-revisionid'][:7]
del get_versions

# the basic logging styles common to all modules
log_sub = '{'
log_fmt = '[{levelname} {asctime}] {message}'
log_date_fmt = '%Y-%m-%d %H:%M:%S%z'

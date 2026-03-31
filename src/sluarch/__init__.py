from importlib.metadata import version,PackageNotFoundError

try:
	__version__=version('sluarch')
except PackageNotFoundError:
	__version__='unknown'

from .sluarch import spindles_abs,spindles_rel,slowosc,kcomp

__all__=['spindles_abs','spindles_rel','spindles_rms','spindles_cov','slowosc','kcomp','__version__']

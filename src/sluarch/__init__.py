from importlib.metadata import PackageNotFoundError, version

try:
	__version__=version('sluarch')
except PackageNotFoundError:
	__version__='unknown'

from .sluarch import detect_kcomp, detect_slowosc, detect_spindles

__all__=['detect_kcomp','detect_slowosc','detect_spindles']

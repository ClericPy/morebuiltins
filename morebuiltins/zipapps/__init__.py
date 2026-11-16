from .activate_zipapps import activate
from .main import ZipApp, __version__, create_app, pip_install_target

__all__ = ["create_app", "activate", "__version__", "ZipApp", "pip_install_target"]
__doc__ = "Package your python code into one zip file, even a virtual environment."

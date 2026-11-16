import importlib.util
import zipfile
import zipimport
from pathlib import Path
from sys import modules


def activate(path=None):
    path = Path(path) if path else Path(__file__).parent
    path_str = path.absolute().as_posix()
    if zipfile.is_zipfile(path_str):
        # zipimport.zipimporter.load_module is deprecated since python3.10,
        # load with the importlib spec protocol instead
        loader = zipimport.zipimporter(path_str)
        spec = loader.find_spec("ensure_zipapps")
        if spec is None or spec.loader is None:
            raise ImportError(f"No module named 'ensure_zipapps' in {path_str}")
        module = importlib.util.module_from_spec(spec)
        # importlib requires the module to be registered before execution so
        # relative imports and dataclasses/pickling can find it by name
        modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            # only remove our own registration, not a module another call set
            if modules.get(spec.name) is module:
                modules.pop(spec.name, None)
        del module
        return True

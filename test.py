#! D:\downloads\python-3.12.1-embed-amd64\python.exe
import importlib
import sys
import time

sys.path.insert(0, "./")
import morebuiltins
from doc import make_docs


def test_all():
    import doctest

    for name in morebuiltins.__all__:
        module = importlib.import_module(name)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "[TEST]", module.__name__, flush=True)
        if hasattr(module, "test"):
            module.test()
        else:
            result = doctest.testmod(module)
            if result.failed:
                raise RuntimeError
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "[PASS]", module.__name__, flush=True)
    print("all test ok", flush=True)


def main():
    test_all()
    make_docs()


if __name__ == "__main__":
    main()

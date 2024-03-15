import time

import morebuiltins.ipc
import morebuiltins.time
import morebuiltins.urllib
import morebuiltins.utils

modules = [morebuiltins.time, morebuiltins.utils, morebuiltins.urllib, morebuiltins.ipc]


def show_docs():
    for module in modules:
        print("=" * 20)
        print(module.__name__)
        print("=" * 20)
        for name in module.__all__:
            member = vars(module)[name]
            doc = member.__doc__
            if doc:
                print(name + ":\n\t" + doc.split("\n")[0])


def test_all():
    import doctest

    for mod in modules:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "[TEST]", mod.__name__, flush=True)
        if hasattr(mod, "test"):
            mod.test()
        else:
            result = doctest.testmod(mod)
            if result.failed:
                raise RuntimeError
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "[PASS]", mod.__name__, flush=True)
    print("all test ok", flush=True)


def main():
    # test_all()
    show_docs()


if __name__ == "__main__":
    main()

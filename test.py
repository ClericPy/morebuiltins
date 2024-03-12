import time

import morebuiltins.ipc
import morebuiltins.time
import morebuiltins.urllib
import morebuiltins.utils


def test_all():
    import doctest

    for mod in [
        morebuiltins.time,
        morebuiltins.utils,
        morebuiltins.urllib,
    ]:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "test", mod.__name__, flush=True)
        result = doctest.testmod(mod)
        if result.failed:
            raise RuntimeError
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "pass", mod.__name__, flush=True)
    mod = morebuiltins.ipc
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "test", mod.__name__, flush=True)
    mod.test_ipc()
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "pass", mod.__name__, flush=True)
    print("all is ok")


if __name__ == "__main__":
    test_all()

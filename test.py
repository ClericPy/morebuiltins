import time

import morebuiltins._ipc
import morebuiltins._time
import morebuiltins._urllib
import morebuiltins._utils


def test_all():
    import doctest

    for mod in [
        morebuiltins._time,
        morebuiltins._utils,
        morebuiltins._urllib,
    ]:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "test", mod.__name__, flush=True)
        result = doctest.testmod(mod)
        if result.failed:
            raise RuntimeError
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "pass", mod.__name__, flush=True)
    mod = morebuiltins._ipc
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "test", mod.__name__, flush=True)
    mod.test_ipc()
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "pass", mod.__name__, flush=True)
    print("all is ok")


if __name__ == "__main__":
    test_all()

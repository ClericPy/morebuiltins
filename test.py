import morebuiltins._time
import morebuiltins._urllib
import morebuiltins._utils
import morebuiltins._ipc


def test_all():
    import doctest

    print("all is testing")
    for mod in [
        morebuiltins._time,
        morebuiltins._utils,
        morebuiltins._urllib,
    ]:
        result = doctest.testmod(mod)
        if result.failed:
            raise RuntimeError
    morebuiltins._ipc.test_ipc()
    print("all is ok")


if __name__ == "__main__":
    test_all()

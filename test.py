import morebuiltins._time
import morebuiltins._urllib
import morebuiltins._util


def test_all():
    import doctest

    print("all is testing")
    doctest.testmod(morebuiltins._time)
    doctest.testmod(morebuiltins._util)
    doctest.testmod(morebuiltins._urllib)
    print("all is ok")


if __name__ == "__main__":
    test_all()

import doctest

import morebuiltins.urllib
import morebuiltins.util


def test_all():
    doctest.testmod(morebuiltins.util)
    doctest.testmod(morebuiltins.urllib)


if __name__ == "__main__":
    test_all()

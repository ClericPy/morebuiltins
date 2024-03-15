from typing import Union, Optional
from time import gmtime, mktime, strftime, strptime, time, timezone

__all__ = ["ttime", "ptime"]


def ttime(
    timestamp: Optional[Union[float, int]] = None,
    tzone: int = int(-timezone / 3600),
    fmt="%Y-%m-%d %H:%M:%S",
) -> str:
    """From timestamp to timestring. Translate timestamp into human-readable: %Y-%m-%d %H:%M:%S.

    Examples:
        >>> ttime(1486572818.421858323, tzone=8)
        '2017-02-09 00:53:38'

    Args:
        timestamp (float, optional): the timestamp float. Defaults to time.time().
        tzone (int, optional): time compensation. Defaults to int(-time.timezone / 3600).
        fmt (str, optional): strftime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        str: time string formatted.
    """
    fix_tz = tzone * 3600
    timestamp = time() if timestamp is None else timestamp
    return strftime(fmt, gmtime(timestamp + fix_tz))


def ptime(
    timestring: Optional[str] = None,
    tzone: int = int(-timezone / 3600),
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> int:
    """From timestring to timestamp. Translate %Y-%m-%d %H:%M:%S into timestamp.
    Examples:
        >>> ptime("2018-03-15 01:27:56", tzone=8)
        1521048476

    Args:
        timestring (str, optional): string like 2018-03-15 01:27:56. Defaults to ttime().
        tzone (int, optional): time compensation. Defaults to int(-timezone / 3600).
        fmt (_type_, optional): strptime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        str: time string formatted.
    """
    fix_tz = -(tzone * 3600 + timezone)
    #: str(timestring) for datetime.datetime object
    if timestring:
        return int(mktime(strptime(str(timestring), fmt)) + fix_tz)
    else:
        return int(time())


if __name__ == "__main__":
    import doctest

    doctest.testmod()

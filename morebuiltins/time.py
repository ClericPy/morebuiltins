import json
import re
import typing
from datetime import datetime
from time import gmtime, mktime, strftime, strptime, time, timezone

__all__ = [
    "ttime",
    "ptime",
    "time_reaches",
]


def ttime(
    timestamp: typing.Union[float, int, None] = None,
    tzone: int = int(-timezone / 3600),
    fmt="%Y-%m-%d %H:%M:%S",
) -> str:
    """Translate timestamp into human-readable: %Y-%m-%d %H:%M:%S.

    Examples:
        >>> ttime(1486572818.421858323)
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
    timestr: str = None,
    tzone: int = int(-timezone / 3600),
    fmt: str = "%Y-%m-%d %H:%M:%S",
) -> int:
    """Translate %Y-%m-%d %H:%M:%S into timestamp.
    Examples:
        >>> ptime("2018-03-15 01:27:56")
        1521048476

    Args:
        timestr (str, optional): string like 2018-03-15 01:27:56. Defaults to ttime().
        tzone (int, optional): time compensation. Defaults to int(-timezone / 3600).
        fmt (_type_, optional): strptime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        str: time string formatted.
    """
    fix_tz = -(tzone * 3600 + timezone)
    #: str(timestr) for datetime.datetime object
    if timestr:
        return int(mktime(strptime(str(timestr), fmt)) + fix_tz)
    else:
        return int(time())


def _time_reaches(time_string, now=None):
    now = now or datetime.now()
    if "==" in time_string:
        # check time_string with strftime: %Y==2020
        fmt, target = time_string.split("==")
        current = now.strftime(fmt)
        # check current time format equals to target
        return current == target
    elif "!=" in time_string:
        # check time_string with strftime: %Y!=2020
        fmt, target = time_string.split("!=")
        current = now.strftime(fmt)
        # check current time format equals to target
        return current != target
    else:
        # other hours format: [1, 3, 11, 23]
        current_hour = now.hour
        if time_string[0] == "[" and time_string[-1] == "]":
            time_string_list = sorted(json.loads(time_string))
        else:
            nums = [int(num) for num in re.findall(r"\d+", time_string)]
            time_string_list = sorted(range(*nums))
        # check if current_hour is work hour
        return current_hour in time_string_list


def time_reaches(time_string, now=None):
    """Check the datetime whether it fit time_string. Support logic symbol:
    equal     => '=='
    not equal => '!='
    or        => '|'
    and       => ';' or '&'

    :: Test Code

        from morebuiltins.time import time_reaches, datetime

        now = datetime.strptime('2020-03-14 11:47:32', '%Y-%m-%d %H:%M:%S')

        oks = [
            '0, 24',
            '[1, 2, 3, 11]',
            '[1, 2, 3, 11];%Y==2020',
            '%d==14',
            '16, 24|[11]',
            '16, 24|%M==47',
            '%M==46|%M==47',
            '%H!=11|%d!=12',
            '16, 24|%M!=41',
        ]

        for time_string in oks:
            ok = time_reaches(time_string, now)
            print(ok, time_string)
            assert ok

        no_oks = [
            '0, 5',
            '[1, 2, 3, 5]',
            '[1, 2, 3, 11];%Y==2021',
            '%d==11',
            '16, 24|[12]',
            '%M==17|16, 24',
            '%M==46|[1, 2, 3]',
            '%H!=11&%d!=12',
            '%M!=46;%M!=47',
        ]

        for time_string in no_oks:
            ok = time_reaches(time_string, now)
            print(ok, time_string)
            assert not ok


    """
    if "|" in time_string:
        if "&" in time_string or ";" in time_string:
            raise ValueError('| can not use with "&" or ";"')
        return any(
            (
                _time_reaches(partial_work_hour, now)
                for partial_work_hour in time_string.split("|")
            )
        )
    else:
        if ("&" in time_string or ";" in time_string) and "|" in time_string:
            raise ValueError('| can not use with "&" or ";"')
        return all(
            (
                _time_reaches(partial_work_hour, now)
                for partial_work_hour in re.split("&|;", time_string)
            )
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()

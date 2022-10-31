import asyncio
import typing
from functools import wraps
from itertools import chain
from time import gmtime, mktime, strftime, strptime, time, timezone

__all__ = [
    # "print_mem",
    # "get_mem",
    # "curlparse",
    # "Null",
    # "null",
    # "itertools_chain",
    "slice_into_pieces",
    "slice_by_size",
    "ttime",
    "ptime",
    # "split_seconds",
    # "timeago",
    # "timepass",
    # "md5",
    # "Counts",
    "unique",
    # "unparse_qs",
    # "unparse_qsl",
    # "Regex",
    # "kill_after",
    # "UA",
    # "try_import",
    # "ensure_request",
    # "Timer",
    # "ClipboardWatcher",
    # "Saver",
    "guess_interval",
    # "split_n",
    # "find_one",
    # "register_re_findone",
    # "Cooldown",
    # "curlrequests",
    # "sort_url_query",
    "retry",
    # "get_readable_size",
    # "encode_as_base64",
    # "decode_as_base64",
    # "check_in_time",
    # "get_host",
    # "find_jsons",
    # "update_url",
    # "stagger_sort",
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
    timestr = str(timestr) if timestr else ttime()
    return int(mktime(strptime(timestr, fmt)) + fix_tz)


def slice_into_pieces(
    items: typing.Sequence, n: int
) -> typing.Generator[tuple, None, None]:
    """Slice a sequence into `n` pieces, return a generation of n pieces.
    Examples:
        >>> for chunk in slice_into_pieces(range(10), 3):
        ...     print(chunk)
        (0, 1, 2, 3)
        (4, 5, 6, 7)
        (8, 9)

    Args:
        seq (_type_): input a sequence.
        n (_type_): split the given sequence into `n` pieces.

    Returns:
        typing.Generator[tuple, None, None]: a generator with tuples.

    Yields:
        Iterator[typing.Generator[tuple, None, None]]: a tuple with n of items.
    """
    length = len(items)
    if length % n == 0:
        size = length // n
    else:
        size = length // n + 1
    for it in slice_by_size(items, size):
        yield it


def slice_by_size(
    items: typing.Sequence, size: int
) -> typing.Generator[tuple, None, None]:
    """Slice a sequence into chunks, return as a generation of tuple chunks with `size`.
    Examples:
        >>> for chunk in slice_by_size(range(10), 3):
        ...     print(chunk)
        (0, 1, 2)
        (3, 4, 5)
        (6, 7, 8)
        (9,)

    Args:
        items (typing.Sequence): _description_
        size (int): _description_

    Returns:
        typing.Generator[tuple, None, None]: a generator with tuples.

    Yields:
        Iterator[typing.Generator[tuple, None, None]]: a tuple with n of items.
    """
    filling = object()
    for it in zip(*(chain(items, [filling] * size),) * size):
        if filling in it:
            it = tuple(i for i in it if i is not filling)
        if it:
            yield it


def unique(
    items: typing.Sequence, key: typing.Callable = None
) -> typing.Generator[typing.Any, None, None]:
    """Unique the seq and keep the order(fast).

    Examples:
        >>> a = ['01', '1', '2']
        >>> list(unique(a, int))
        [1, 2]
        >>> list(unique(a))
        ['01', '1', '2']

    Args:
        items (typing.Sequence): raw sequence.
        key (typing.Callable): the function to normalize each item.

    Returns:
        typing.Generator[tuple, None, None]: a generator with unique items.

    Yields:
        Iterator[typing.Generator[tuple, None, None]]: the unique item.
    """
    seen: set = set()
    _add = seen.add
    if key:
        for item in items:
            _item = key(item)
            if _item not in seen:
                yield _item
                _add(_item)
    else:
        for item in items:
            if item not in seen:
                yield item
                _add(item)


def retry(
    tries=2,
    exceptions: typing.Tuple[typing.Type[BaseException]] = (Exception,),
    return_exception=False,
):
    """A decorator which will retry the function `tries` times while raising given exceptions.

    Examples:
        >>> func = lambda items: 1/items.pop(0)
        >>> items = [0, 1]
        >>> new_func = retry(tries=2, exceptions=(ZeroDivisionError,))(func)
        >>> new_func(items)
        1.0

    Args:
        tries (int, optional): try n times, if n==1 means no retry. Defaults to 1.
        exceptions (typing.Tuple[typing.Type[BaseException]], optional): only retry the given errors. Defaults to (Exception,).
        return_exception (bool, optional): raise the last exception or return it. Defaults to False.
    """

    def wrapper(function):
        @wraps(function)
        def retry_sync(*args, **kwargs):
            for _ in range(tries):
                try:
                    return function(*args, **kwargs)
                except exceptions as err:
                    error = err
            if return_exception:
                return error
            raise error

        @wraps(function)
        async def retry_async(*args, **kwargs):
            for _ in range(tries):
                try:
                    return await function(*args, **kwargs)
                except exceptions as err:
                    error = err
            if return_exception:
                return error
            raise error

        if asyncio.iscoroutinefunction(function):
            return retry_async
        else:
            return retry_sync

    return wrapper


def guess_interval(nums, accuracy=0):
    """Given a seq of number, return the median, only calculate interval >= accuracy.

    Example::
        # sorted_seq: [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
        # diffs: [8, 7, 10, 6, 13, 8]
        # median: 8
        >>> seq = [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
        >>> guess_interval(seq, 5)
        8

    """
    if not nums:
        return 0
    nums = sorted([int(i) for i in nums])
    if len(nums) == 1:
        return nums[0]
    diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
    diffs = [item for item in diffs if item >= accuracy]
    sorted_diff = sorted(diffs)
    return sorted_diff[len(diffs) // 2]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

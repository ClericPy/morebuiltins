import asyncio
import hashlib
import typing
from functools import wraps
from itertools import chain

__all__ = [
    # "curlparse",
    "slice_into_pieces",
    "slice_by_size",
    # "split_seconds",
    # "timeago",
    # "timepass",
    "get_hash",
    "unique",
    # "try_import",
    # "ensure_request",
    # "ClipboardWatcher",
    # "Saver",
    "guess_interval",
    # "find_one",
    # "register_re_findone",
    # "Cooldown",
    # "curlrequests",
    "url_query_update",
    "retry",
    # "get_host",
    # "find_jsons",
    # "update_url",
    # "stagger_sort",
]




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


def get_hash(
    string,
    n: typing.Union[tuple, list, int, None] = None,
    default: typing.Callable = lambda obj: str(obj).encode("utf-8"),
    func=hashlib.md5,
) -> str:
    """Get the md5_string from given string

    >>> get_hash(123456, 10)
    'a59abbe56e'
    >>> get_hash('test')
    '098f6bcd4621d373cade4e832627b4f6'
    >>> get_hash(['list_demo'], (5, 10))
    '7152a'
    >>> get_hash(['list_demo'], func=hashlib.sha256)
    'a6072e063d36a09052a9e5eb389a425a3dc158d3a5955808159a118aa192c718'
    >>> get_hash(['list_demo'], 16, func=hashlib.sha256)
    '389a425a3dc158d3'
    """
    if isinstance(string, bytes):
        _bytes = string
    else:
        _bytes = default(string)
    _temp = func(_bytes).hexdigest()
    if n is None:
        return _temp
    elif isinstance(n, (int, float)):
        start, end = (len(_temp) - n) // 2, (n - len(_temp)) // 2
        return _temp[start:end]
    elif isinstance(n, (tuple, list)):
        start, end = n[0], n[1]
        return _temp[start:end]




if __name__ == "__main__":
    import doctest

    doctest.testmod()

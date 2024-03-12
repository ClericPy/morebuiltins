import asyncio
import base64
import gzip
import hashlib
import json
import re
from enum import IntEnum
from functools import wraps
from itertools import groupby, islice
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

__all__ = [
    "slice_into_pieces",
    "slice_by_size",
    "get_hash",
    "unique",
    "guess_interval",
    "retry",
    "find_jsons",
    "code_inline",
    "stagger_sort",
    "read_size",
    "read_time",
]


def slice_into_pieces(
    items: Sequence, n: int
) -> Generator[Union[tuple, Sequence], None, None]:
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
        Generator[tuple, None, None]: a generator with tuples.

    Yields:
        Iterator[Generator[tuple, None, None]]: a tuple with n of items.
    """
    length = len(items)
    if length % n == 0:
        size = length // n
    else:
        size = length // n + 1
    for it in slice_by_size(items, size):
        yield it


def slice_by_size(
    items: Sequence, size: int, callback=tuple
) -> Generator[Union[tuple, Sequence], None, None]:
    """Slice a sequence into chunks, return as a generation of tuple chunks with `size`.
    Examples:
        >>> for chunk in slice_by_size(range(10), 3):
        ...     print(chunk)
        (0, 1, 2)
        (3, 4, 5)
        (6, 7, 8)
        (9,)

    Args:
        items (Sequence): _description_
        size (int): _description_

    Returns:
        Generator[tuple, None, None]: a generator with tuples.

    Yields:
        Iterator[Generator[tuple, None, None]]: a tuple with n of items.
    """
    iter_seq = iter(items)
    while True:
        chunk = callback(islice(iter_seq, size))
        if chunk:
            yield chunk
        else:
            break


def unique(
    items: Sequence, key: Optional[Callable] = None
) -> Generator[Any, None, None]:
    """Unique the seq and keep the order(fast).

    Examples:
        >>> a = ['01', '1', '2']
        >>> list(unique(a, int))
        [1, 2]
        >>> list(unique(a))
        ['01', '1', '2']

    Args:
        items (Sequence): raw sequence.
        key (Callable): the function to normalize each item.

    Returns:
        Generator[tuple, None, None]: a generator with unique items.

    Yields:
        Iterator[Generator[tuple, None, None]]: the unique item.
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
    exceptions: Tuple[Type[BaseException]] = (Exception,),
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
        exceptions (Tuple[Type[BaseException]], optional): only retry the given errors. Defaults to (Exception,).
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
    n: Union[tuple, list, int, None] = None,
    default: Callable = lambda obj: str(obj).encode("utf-8"),
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


def find_jsons(string, return_as="json", json_loader=json.loads):
    """Generator for finding the valid JSON string, only support dict and list.
    return_as could be 'json' / 'object' / 'index'.
    ::

        >>> list(find_jsons('string["123"]123{"a": 1}[{"a": 1, "b": [1,2,3]}]'))
        ['["123"]', '{"a": 1}', '[{"a": 1, "b": [1,2,3]}]']
        >>> list(find_jsons('string[]{}{"a": 1}'))
        ['[]', '{}', '{"a": 1}']
        >>> list(find_jsons('string[]|{}string{"a": 1}', return_as='index'))
        [(6, 8), (9, 11), (17, 25)]
        >>> list(find_jsons('xxxx[{"a": 1, "b": [1,2,3]}]xxxx', return_as='object'))
        [[{'a': 1, 'b': [1, 2, 3]}]]
    """

    def find_matched(string, left, right):
        _stack = []
        for index, char in enumerate(string):
            if char == left:
                _stack.append(index)
            elif char == right:
                try:
                    _stack.pop()
                except IndexError:
                    break
            else:
                continue
            if not _stack:
                return index

    search = re.search
    brackets_map = {"{": "}", "[": "]"}
    current_start = 0
    while string and isinstance(string, str):
        _match = search(r"[\[\{]", string)
        if not _match:
            break
        left = _match.group()
        right = brackets_map[left]
        _start = _match.span()[0]
        sub_string = string[_start:]
        _end = find_matched(sub_string, left, right)
        if _end is None:
            # not found matched, check next left
            string = sub_string
            continue
        string = sub_string[_end + 1 :]
        try:
            _partial = sub_string[: _end + 1]
            _loaded_result = json_loader(_partial)
            yield {
                "json": _partial,
                "object": _loaded_result,
                "index": (current_start + _start, current_start + _start + _end + 1),
            }.get(return_as, string)
        except (ValueError, TypeError):
            pass
        current_start += _start + _end + 1


def code_inline(
    source_code: str,
    encoder: Literal["b16", "b32", "b64", "b85"] = "b85",
) -> str:
    """Make the python source code inline.

    Args:
        source_code (str): python original code.
        encoder (Literal['b16', 'b32', 'b64', 'b85'], optional): base64.encoder. Defaults to "b85".

    Returns:
        new source code inline.
    ::

        >>> code1 = ''
        >>> code2 = code_inline("variable=12345")
        >>> # import base64,gzip;exec(gzip.decompress(base64.b85decode("ABzY8mBl+`0{<&ZEXqtw%1N~~G%_|Z1ptx!(o_xr000".encode("u8"))))
        >>> exec(code2)
        >>> variable
        12345

    """
    _encoder = getattr(base64, f"{encoder}encode")
    _source = source_code.encode(encoding="u8")
    _source = gzip.compress(_source)
    _source = _encoder(_source)
    code = _source.decode("u8")
    result = (
        "import base64,gzip;exec(gzip.decompress("
        f'base64.{encoder}decode("{code}".encode("u8"))))'
    )
    return result


class BytesUnit(IntEnum):
    B = 0
    KB = 1 * 1024**1
    MB = 1 * 1024**2
    GB = 1 * 1024**3
    TB = 1 * 1024**4
    PB = 1 * 1024**5
    EB = 1 * 1024**6
    ZB = 1 * 1024**7
    YB = 1 * 1024**8


class TimeUnit(IntEnum):
    secs = 0
    mins = 60
    hours = 60 * 60
    days = 60 * 60 * 24
    mons = 60 * 60 * 24 * 30
    years = 60 * 60 * 24 * 365


def read_num(b, enum_class=IntEnum, rounded: Optional[int] = None):
    unit = None
    for _unit in enum_class:
        if unit is None or _unit.value <= b:
            unit = _unit
        elif _unit.value > b:
            break
    assert unit is not None
    return f"{round(b / (unit.value or 1), rounded)} {unit.name}"


def read_size(b, rounded: Optional[int] = None):
    """From B to readable string.

    Args:
        b: B
        rounded (int, optional): arg for round. Defaults to None.

    Returns:
        str

    ::
        >>> read_size(0)
        '0 B'
        >>> for i in range(0, 10):
        ...     [1 * 1024**i, read_size(1 * 1024**i, rounded=1)]
        ...
        [1, '1.0 B']
        [1024, '1.0 KB']
        [1048576, '1.0 MB']
        [1073741824, '1.0 GB']
        [1099511627776, '1.0 TB']
        [1125899906842624, '1.0 PB']
        [1152921504606846976, '1.0 EB']
        [1180591620717411303424, '1.0 ZB']
        [1208925819614629174706176, '1.0 YB']
        [1237940039285380274899124224, '1024.0 YB']
    """
    return read_num(b, BytesUnit, rounded=rounded)


def read_time(secs, rounded: Optional[int] = None):
    """From secs to readable string.

    Args:
        b: seconds
        rounded (int, optional): arg for round. Defaults to None.

    Returns:
        str

    ::
        >>> read_time(0)
        '0 secs'
        >>> for i in range(0, 6):
        ...     [1.2345 * 60**i, read_time(1.2345 * 60**i, rounded=1)]
        ...
        [1.2345, '1.2 secs']
        [74.07, '1.2 mins']
        [4444.2, '1.2 hours']
        [266652.0, '3.1 days']
        [15999120.0, '6.2 mons']
        [959947200.0, '30.4 years']
    """
    return read_num(secs, TimeUnit, rounded=rounded)


class Validator:
    """
    Validator for dataclasses.

    ::
        >>> from dataclasses import dataclass, field
        >>>
        >>>
        >>> @dataclass
        ... class Person(Validator):
        ...     screen: dict = field(metadata={"callback": lambda i: i["s"]})
        ...     name: str = field(default=None, metadata={"callback": str})
        ...     age: int = field(default=0, metadata={"callback": int})
        ...
        >>>
        >>> print(Person({"s": 3}, 123, "123"))
        Person(screen=3, name='123', age=123)
    """

    def __post_init__(self):
        for f in self.__dataclass_fields__.values():
            callback = f.metadata.get("callback")
            if callback:
                setattr(self, f.name, callback(getattr(self, f.name)))

    # def to_dict(self):
    #     return {
    #         field.name: getattr(self, field.name)
    #         for field in self.__dataclass_fields__.values()
    #     }


def stagger_sort(items, group_key, sort_key=None):
    """Ensure that the same group is ordered and staggered, avoid data skew. Will not affect the original list, return as a generator.

    ::

        >>> items = [('a', 0), ('a', 2), ('a', 1), ('b', 0), ('b', 1)]
        >>> list(stagger_sort(items, sort_key=lambda i: (i[0], i[1]), group_key=lambda i: i[0]))
        [('a', 0), ('b', 0), ('a', 1), ('b', 1), ('a', 2)]

    """
    if sort_key:
        items = sorted(items, key=sort_key)
    buckets = [list(group[1]) for group in groupby(items, group_key)]
    while True:
        next_buckets = []
        for items in buckets:
            try:
                yield items.pop(0)
                next_buckets.append(items)
            except IndexError:
                pass
        if next_buckets:
            buckets = next_buckets
        else:
            break


if __name__ == "__main__":
    import doctest

    doctest.testmod()

import asyncio
import atexit
import base64
import collections.abc
import gzip
import hashlib
import inspect
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import timeit
import traceback
import types
from collections import UserDict
from datetime import datetime
from enum import IntEnum
from functools import wraps
from itertools import groupby, islice
from os.path import basename, exists
from pathlib import Path
from threading import Lock, Timer, current_thread
from time import gmtime, mktime, strftime, strptime, time, timezone
from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

__all__ = [
    "ttime",
    "ptime",
    "slice_into_pieces",
    "slice_by_size",
    "unique",
    "retry",
    "guess_interval",
    "get_hash",
    "find_jsons",
    "code_inline",
    "read_size",
    "read_time",
    "Validator",
    "stagger_sort",
    "default_dict",
    "always_return_value",
    "format_error",
    "Trie",
    "GuessExt",
    "xor_encode_decode",
    "is_running",
    "set_pid_file",
    "get_paste",
    "set_clip",
    "switch_flush_print",
    "unix_rlimit",
    "SimpleFilter",
    "FileDict",
    "PathLock",
    "i2b",
    "b2i",
    "get_hash_int",
    "iter_weights",
    "get_size",
    "base_encode",
    "base_decode",
    "gen_id",
    "timeti",
    "SnowFlake",
    "cut_file",
]


def ttime(
    timestamp: Optional[Union[float, int]] = None,
    tzone: int = int(-timezone / 3600),
    fmt="%Y-%m-%d %H:%M:%S",
) -> str:
    """Converts a timestamp to a human-readable timestring formatted as %Y-%m-%d %H:%M:%S.

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
    """Converts a timestring formatted as %Y-%m-%d %H:%M:%S back into a timestamp.

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


def slice_into_pieces(items: Sequence, n: int) -> Generator[tuple, None, None]:
    """Divides a sequence into “n” segments, returning a generator that yields “n” pieces.

    >>> for chunk in slice_into_pieces(range(10), 3):
    ...     print(chunk)
    (0, 1, 2, 3)
    (4, 5, 6, 7)
    (8, 9)

    Args:
        seq (_type_): input a sequence.
        n (_type_): split the given sequence into "n" pieces.

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
) -> Generator[tuple, None, None]:
    """Slices a sequence into chunks of a specified “size”, returning a generator that produces tuples of chunks.

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
    """Removes duplicate elements from a sequence while preserving the original order efficiently.

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
    seen: set[Any] = set()
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
    """A decorator that retries the decorated function up to “tries” times if the specified exceptions are raised.

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
            error = None
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
            error = None
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
    """Analyzes a sequence of numbers and returns the median, calculating intervals only if they are greater than or equal to the specified accuracy.

    >>> # sorted_seq: [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
    >>> # diffs: [8, 7, 10, 6, 13, 8]
    >>> # median: 8
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
    n: Optional[Union[Tuple[int, int], List[int], int]] = None,
    default: Callable = lambda obj: str(obj).encode("utf-8"),
    func=hashlib.md5,
) -> str:
    """Generates a hash string from the given input string.

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


def get_hash_int(
    string,
    n: int = 16,
    default: Callable = lambda obj: str(obj).encode("utf-8"),
    func=hashlib.md5,
) -> int:
    """Generates a int hash(like docid) from the given input bytes.

    >>> get_hash_int(1)
    2035485573088411
    >>> get_hash_int("string")
    1418352543534881
    >>> get_hash_int(b'123456', 16)
    4524183350839358
    >>> get_hash_int(b'123', 10)
    5024125808
    >>> get_hash_int(b'123', 13, func=hashlib.sha256)
    1787542395619
    >>> get_hash_int(b'123', 13, func=hashlib.sha512)
    3045057537218
    >>> get_hash_int(b'123', 13, func=hashlib.sha1)
    5537183137519
    """
    return int(get_hash(string, n=None, default=default, func=func), 16) % 10**n


def find_jsons(
    string,
    return_as: Literal["json", "object", "index"] = "json",
    json_loader=json.loads,
):
    """A generator that locates valid JSON strings, supporting only dictionaries and lists.

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
    """Minifies Python source code into a single line.

    >>> code1 = code_inline('def test_code1(): return 12345')
    >>> code1
    'import base64,gzip;exec(gzip.decompress(base64.b85decode("ABzY80RR910{=@%O;adIEiQ>q&QD1-)X=n2C`v6UEy`0cG%_|Z1psqiSP>oo000".encode("u8"))))'
    >>> exec(code1)
    >>> test_code1()
    12345
    >>> code2 = code_inline("v=12345")
    >>> code2
    'import base64,gzip;exec(gzip.decompress(base64.b85decode("ABzY80RR910{<(sH8e6dF$Dk}<L9Rb0000".encode("u8"))))'
    >>> exec(code2)
    >>> v
    12345

    Args:
        source_code (str): python original code.
        encoder (Literal['b16', 'b32', 'b64', 'b85'], optional): base64.encoder. Defaults to "b85".
    Returns:
        new source code inline.
    """
    _encoder = getattr(base64, f"{encoder}encode")
    _source = source_code.encode(encoding="u8")
    _source = gzip.compress(_source, mtime=1)
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


def read_num(
    b,
    enum: Type[IntEnum] = IntEnum,
    rounded: Optional[int] = None,
    shorten=False,
    precision=1.0,
    sep=" ",
):
    units = tuple(enum)
    unit = units[0]
    ntime = 0.5 if rounded and rounded > 0 else 1
    for _unit in units:
        if b >= _unit.value * ntime:
            unit = _unit
        else:
            break
    result = round(b / (unit.value or 1), rounded)
    if shorten and isinstance(result, float):
        int_result = int(result)
        if result:
            if int_result / result >= precision:
                result = int_result
    return f"{result}{sep}{unit.name}"


def read_size(b, rounded: Optional[int] = None, shorten=False, precision=1.0, sep=" "):
    """Converts byte counts into a human-readable string. Setting shorten=True and precision=0.99 will trim unnecessary decimal places from the tail of floating-point numbers.

    >>> (read_size(1023), read_size(1024))
    ('1023 B', '1 KB')
    >>> (read_size(400.5, 1), read_size(400.5, 1, True), read_size(400.5, 1, True, 0.99))
    ('400.5 B', '400.5 B', '400 B')
    >>> (read_size(511.55, 1, shorten=True, precision=0.999), read_size(512.55, 1, shorten=True, precision=0.999))
    ('511.6 B', '0.5 KB')
    >>> (read_size(511, 1, shorten=True), read_size(512, 1, shorten=True))
    ('511 B', '0.5 KB')
    >>> read_size(512, 1, sep='')
    '0.5KB'
    >>> read_size(1025, 1, shorten=False), read_size(1025, 1, shorten=True)
    ('1.0 KB', '1 KB')
    >>> for i in range(0, 5):
    ...     [1.1111 * 1024**i, i, read_size(1.1111 * 1024**i, rounded=i)]
    ...
    [1.1111, 0, '1.0 B']
    [1137.7664, 1, '1.1 KB']
    [1165072.7936, 2, '1.11 MB']
    [1193034540.6464, 3, '1.111 GB']
    [1221667369621.9136, 4, '1.1111 TB']

    Args:
        b: bytes
        rounded (int, optional): arg for round. Defaults to None.
        shorten (bool): shorten unnecessary tail 0.
        precision: (float): shorten precision, often set to 0.99.
        sep (str, optional): sep between result and unit.

    Returns:
        str


    """
    return read_num(
        b,
        BytesUnit,
        rounded=rounded,
        shorten=shorten,
        precision=precision,
        sep=sep,
    )


def read_time(
    secs, rounded: Optional[int] = None, shorten=False, precision=1.0, sep=" "
):
    """Converts seconds into a more readable time duration string.

    >>> read_time(0)
    '0 secs'
    >>> read_time(60)
    '1 mins'
    >>> for i in range(0, 6):
    ...     [1.2345 * 60**i, read_time(1.2345 * 60**i, rounded=1)]
    ...
    [1.2345, '1.2 secs']
    [74.07, '1.2 mins']
    [4444.2, '1.2 hours']
    [266652.0, '3.1 days']
    [15999120.0, '0.5 years']
    [959947200.0, '30.4 years']

    Args:
        b: seconds
        rounded (int, optional): arg for round. Defaults to None.
        shorten (bool): shorten unnecessary tail 0.
        precision: (float): shorten precision, often set to 0.99.
        sep (str, optional): sep between result and unit.

    Returns:
        str
    """
    return read_num(
        secs,
        TimeUnit,
        rounded=rounded,
        shorten=shorten,
        precision=precision,
        sep=sep,
    )


class Validator:
    """Validator for dataclasses.
    >>> from dataclasses import dataclass, field
    >>>
    >>>
    >>> @dataclass
    ... class Person(Validator):
    ...     screen: dict = field(metadata={"callback": lambda i: i.clear() or {'s': 4}})
    ...     name: str = field(default=None, metadata={"callback": str})
    ...     age: int = field(default=0, metadata={"callback": int})
    ...     other: str = ''
    ...
    >>>
    >>> # test type callback
    >>> print(Person({"s": 3}, 1, "1"))
    Person(screen={'s': 4}, name='1', age=1, other='')

    >>> # test Validator.STRICT = False, `other` could be int
    >>> Validator.STRICT = False
    >>> Person({"s": 3}, "1", "1", 0)
    Person(screen={'s': 4}, name='1', age=1, other=0)

    >>> # test Validator.STRICT = True, raise TypeError, `other` should not be int
    >>> Validator.STRICT = True
    >>> try:
    ...     Person({"s": 3}, "1", "1", 0)
    ... except TypeError as e:
    ...     print(e)
    `other` should be `str` but given `int`
    >>> # test class Name
    >>> @dataclass
    ... class Name(Validator):
    ...     name: str
    ...
    >>> @dataclass
    ... class Person(Validator):
    ...     name: Name
    ...
    >>> try:
    ...     print(Person('name'))
    ... except TypeError as e:
    ...     print(e)
    ...
    `name` should be `Name` but given `str`
    >>> # test typing.Dict[str, str]
    >>> import typing
    >>> @dataclass
    ... class Person(Validator):
    ...     name: typing.Dict[str, str]
    ...
    >>> try:
    ...     print(Person('name'))
    ... except TypeError as e:
    ...     print(e)
    ...
    `name` should be `dict` but given `str`
    """

    STRICT = True

    def __post_init__(self):
        for f in self.__dataclass_fields__.values():
            callback = f.metadata.get("callback")
            value = getattr(self, f.name)
            if callback:
                setattr(self, f.name, callback(value))
            if self.STRICT:
                value = getattr(self, f.name)
                if inspect.isclass(f.type):
                    tp = f.type
                elif hasattr(f.type, "__origin__"):
                    tp = f.type.__origin__
                else:
                    continue
                if not isinstance(value, tp):
                    raise TypeError(
                        f"`{f.name}` should be `{tp.__name__}` but given `{type(value).__name__}`"
                    )

    def quick_to_dict(self):
        "not very reliable"
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


def stagger_sort(items: Sequence, group_key, sort_key=None, sort_reverse=False):
    """Ensures that identical groups are ordered and evenly distributed, mitigating data skew. The function does not alter the original list and returns a generator.

    >>> items = [('a', 0), ('a', 2), ('a', 1), ('b', 0), ('b', 1)]
    >>> list(stagger_sort(items, sort_key=lambda i: (i[0], i[1]), group_key=lambda i: i[0]))
    [('a', 0), ('b', 0), ('a', 1), ('b', 1), ('a', 2)]
    >>> items = ['a-a', 'a-b', 'b-b', 'b-c', 'b-a', 'b-d', 'c-a', 'c-a']
    >>> list(stagger_sort(items, sort_key=lambda i: (i[0], i[2]), group_key=lambda i: i[0]))
    ['a-a', 'b-a', 'c-a', 'a-b', 'b-b', 'c-a', 'b-c', 'b-d']
    """
    if sort_key:
        items = sorted(items, key=sort_key, reverse=sort_reverse)
    else:
        items = list(items)
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


def iter_weights(
    weight_dict: Dict[Any, Union[int, float]], loop_length=100, round_int=round
):
    """Generates an element sequence based on weights.

    This function produces a sequence of elements where each element's frequency of occurrence
    is proportional to its weight from the provided dictionary. Elements with higher weights
    appear more frequently in the sequence. The total cycle length can be adjusted via the
    `loop_length` parameter. The `round_int` parameter allows customization of the rounding
    function to control the precision of weight calculations.

    Keys with weights greater than 0 will be yielded.

    Parameters:
    - weight_dict: A dictionary where keys are elements and values are their respective weights.
    - loop_length: An integer defining the total length of the repeating cycle, defaulting to 100.
    - round_int: A function used for rounding, defaulting to Python's built-in `round`.

    Yields:
    A generator that yields a sequence of elements distributed according to their weights.

    Examples:
    >>> list(iter_weights({"6": 6, "3": 3, "1": 0.4}, 10))
    ['6', '3', '6', '3', '6', '3', '6', '6', '6']
    >>> list(iter_weights({"6": 6, "3": 3, "1": 0.9}, 10))
    ['6', '3', '1', '6', '3', '6', '3', '6', '6', '6']
    >>> list(iter_weights({"6": 6, "3": 3, "1": 0.9}, 10, round_int=int))
    ['6', '3', '6', '3', '6', '3', '6', '6', '6']
    >>> from itertools import cycle
    >>> c = cycle(iter_weights({"6": 6, "3": 3, "1": 1}, loop_length=10))
    >>> [next(c) for i in range(20)]
    ['6', '3', '1', '6', '3', '6', '3', '6', '6', '6', '6', '3', '1', '6', '3', '6', '3', '6', '6', '6']
    """
    keys = list(weight_dict.keys())
    items = []
    total = sum(weight_dict.values())
    for key, value in weight_dict.items():
        length = round_int(loop_length * value / total) if value > 0 else 0
        items.extend([key] * length)
    for item in stagger_sort(
        items,
        group_key=lambda x: keys.index(x),
        sort_key=lambda x: weight_dict[x],
        sort_reverse=True,
    ):
        yield item


def default_dict(typeddict_class: Type[dict], **kwargs):
    """Initializes a dictionary with default zero values based on a subclass of TypedDict.

    >>> class Demo(dict):
    ...     int_obj: int
    ...     float_obj: float
    ...     bytes_obj: bytes
    ...     str_obj: str
    ...     list_obj: list
    ...     tuple_obj: tuple
    ...     set_obj: set
    ...     dict_obj: dict
    >>> item = default_dict(Demo, bytes_obj=b'1')
    >>> item
    {'int_obj': 0, 'float_obj': 0.0, 'bytes_obj': b'1', 'str_obj': '', 'list_obj': [], 'tuple_obj': (), 'set_obj': set(), 'dict_obj': {}}
    >>> type(item)
    <class 'morebuiltins.utils.Demo'>
    >>> from typing import TypedDict
    >>> class Demo(TypedDict):
    ...     int_obj: int
    ...     float_obj: float
    ...     bytes_obj: bytes
    ...     str_obj: str
    ...     list_obj: list
    ...     tuple_obj: tuple
    ...     set_obj: set
    ...     dict_obj: dict
    >>> item = default_dict(Demo, bytes_obj=b'1')
    >>> item
    {'int_obj': 0, 'float_obj': 0.0, 'bytes_obj': b'1', 'str_obj': '', 'list_obj': [], 'tuple_obj': (), 'set_obj': set(), 'dict_obj': {}}
    >>> type(item)
    <class 'dict'>
    """
    result = typeddict_class()
    built_in_types = {int, float, bytes, str, list, tuple, set, dict}
    for key, tp in typeddict_class.__annotations__.items():
        if key in kwargs:
            result[key] = kwargs[key]
        elif tp in built_in_types:
            result[key] = tp()
    return result


def always_return_value(value):
    """Got a function always return the given value.
    >>> func = always_return_value(1)
    >>> func(1, 2, 3)
    1
    >>> func(1, 2, c=3)
    1
    """

    def func(*args, **kws):
        return value

    return func


def _tb_filter(tb: traceback.FrameSummary):
    # filt tb with code-line and not excluded -packages
    return tb.line and "-packages" not in tb.filename


def format_error(
    error: BaseException,
    index: Union[int, slice] = slice(-3, None, None),
    filter: Optional[Callable] = _tb_filter,
    template="[{trace_routes}] {error_line} >>> {error.__class__.__name__}({error!s:.100})",
    filename_filter: Tuple[str, str] = ("", ""),
    **kwargs,
) -> str:
    r"""Extracts frame information from an exception, with an option to filter out “-packages” details by default. To shorten your exception message.

    Parameters:

    - `error` (`BaseException`): The exception instance for which the stack trace information is to be extracted and formatted.
    - `index` (`Union[int, slice]`, optional): Specifies which frames to include in the output. By default, it's set to `slice(-3, None, None)`, showing the last three frames. Can be an integer for a single frame or a slice object for a range of frames.
    - `filter` (`Optional[Callable]`, optional): A callable that determines whether a given frame should be included. Defaults to `_tb_filter`, which typically filters out frames from "-packages". If set to `None`, no filtering occurs.
    - `template` (`str`, optional): A string template defining how the error message should be formatted. It can include placeholders like `{trace_routes}`, `{error_line}`, and `{error.__class__.__name__}`. The default template provides a concise summary of the error location and type.
    - `filename_filter` (`Tuple[str, str]`, optional): A tuple specifying the include and exclude strings of filename. Defaults to `("", "")`, which means no filtering occurs.
    - `**kwargs`: Additional keyword arguments to be used within the formatting template.

    Returns:

    A string representing the formatted error message based on the provided parameters and template.

    Demo:

    >>> try:
    ...     # test default
    ...     1 / 0
    ... except Exception as e:
    ...     format_error(e)
    '[<doctest>:<module>(3)] 1 / 0 >>> ZeroDivisionError(division by zero)'
    >>> try:
    ...     # test in function
    ...     def func1(): 1 / 0
    ...     func1()
    ... except Exception as e:
    ...     format_error(e)
    '[<doctest>:<module>(4)|<doctest>:func1(3)] def func1(): 1 / 0 >>> ZeroDivisionError(division by zero)'
    >>> try:
    ...     # test index
    ...     def func2(): 1 / 0
    ...     func2()
    ... except Exception as e:
    ...     format_error(e, index=0)
    '[<doctest>:<module>(4)] func2() >>> ZeroDivisionError(division by zero)'
    >>> try:
    ...     # test slice index
    ...     def func2(): 1 / 0
    ...     func2()
    ... except Exception as e:
    ...     format_error(e, index=slice(-1, None, None))
    '[<doctest>:func2(3)] def func2(): 1 / 0 >>> ZeroDivisionError(division by zero)'
    """
    try:
        if filter:
            _filter = filter
        else:
            _filter = always_return_value(True)
        tbs = [tb for tb in traceback.extract_tb(error.__traceback__) if _filter(tb)]
        if isinstance(index, slice):
            tbs = tbs[index]
        elif isinstance(index, int):
            tbs = [tbs[index]]
        else:
            raise ValueError("Invalid index type")
        trace_route_list = []
        include, exclude = filename_filter
        for tb in tbs:
            filename = tb.filename
            if include and include not in filename:
                continue
            if exclude and exclude in filename:
                continue
            if exists(filename):
                _basename = basename(filename)
            elif filename[0] == "<":
                _basename = f"{filename.split()[0]}>"
            else:
                _basename = filename
            trace_route_list.append(f"{_basename}:{tb.name}({tb.lineno})")
        trace_routes = "|".join(trace_route_list)
        _kwargs = {
            "tbs": tbs,
            "error": error,
            "error_line": tbs[-1].line,
            "trace_routes": trace_routes,
        }
        _kwargs.update(kwargs)
        return template.format_map(_kwargs)
    except IndexError:
        return ""


class Trie(UserDict):
    """Transforms a standard dictionary into a trie structure that supports prefix matching.

    >>> trie = Trie({"ab": 1, "abc": 2, b"aa": 3, ("e", "e"): 4, (1, 2): 5})
    >>> trie
    {'a': {'b': {'_VALUE': 1, 'c': {'_VALUE': 2}}}, 97: {97: {'_VALUE': 3}}, 'e': {'e': {'_VALUE': 4}}, 1: {2: {'_VALUE': 5}}}
    >>> trie.match("abcde")
    (3, 2)
    >>> trie.match("abde")
    (2, 1)
    >>> trie.match(b"aabb")
    (2, 3)
    >>> trie.match("eee")
    (2, 4)
    >>> trie.match(list("eee"))
    (2, 4)
    >>> trie.match(tuple("eee"))
    (2, 4)
    >>> trie.match((1, 2, 3, 4))
    (2, 5)
    """

    def __init__(self, data: Dict[AnyStr, Any], value_node="_VALUE"):
        # data should be 1-depth dict
        super().__init__(self.parse_dict(data, value_node))
        self.value_node = value_node
        self.not_set = object()

    @staticmethod
    def parse_dict(data: dict, value_node="_VALUE") -> dict:
        "Convert a regular dictionary to a prefix tree dictionary"
        result: dict = {}
        for word, value in data.items():
            trie = result
            for char in word:
                if char not in trie:
                    trie[char] = {}
                trie = trie[char]
            trie[value_node] = value
        return result

    def match(self, seq, default: Tuple[Union[None, int], Any] = (None, None)):
        """Use the input sequence to match the hit value in the prefix tree as much as possible.

        Args:
            seq: iterable object
            default:return while not match. Defaults to (None, None).

        Returns:
            _type_: tuple
        """
        trie = self
        value_node = self.value_node
        value = self.not_set
        dep = 0
        for char in seq:
            if char in trie:
                dep += 1
                trie = trie[char]
                try:
                    value = trie[value_node]
                except KeyError:
                    pass
            else:
                break
        if value is not self.not_set:
            return (dep, value)
        return default


class GuessExt(object):
    r"""Determines whether the input bytes of a file prefix indicate a compressed file format.

    >>> cg = GuessExt()
    >>> cg.get_ext(b"PK\x05\x06zipfiledemo")
    '.zip'
    """

    MAGIC = {
        b"PK\x03\x04": ".zip",
        b"PK\x03\x06": ".zip",
        b"PK\x03\x08": ".zip",
        b"PK\x05\x04": ".zip",
        b"PK\x05\x06": ".zip",
        b"PK\x05\x08": ".zip",
        b"PK\x07\x04": ".zip",
        b"PK\x07\x06": ".zip",
        b"PK\x07\x08": ".zip",
        b'"\xb5/\xfd': ".zst",
        b"#\xb5/\xfd": ".zst",
        b"$\xb5/\xfd": ".zst",
        b"%\xb5/\xfd": ".zst",
        b"&\xb5/\xfd": ".zst",
        b"'\xb5/\xfd": ".zst",
        b"(\xb5/\xfd": ".zst",
        b"LZIP": ".lz",
        b"MSCF": ".cab",
        b"ISc(": ".cab",
        b"\x1f\x8b\x08": ".gz",
        b"BZh": ".bz2",
        b"\xfd7zXZ\x00": ".xz",
        b"7z\xbc\xaf'\x1c": ".7z",
        b"Rar!\x1a\x07\x00": ".rar",
        b"Rar!\x1a\x07\x01": ".rar",
    }

    def __init__(self):
        self.trie = Trie(self.MAGIC)

    def get_ext(self, magic_bytes: bytes, default=""):
        dep, value = self.trie.match(magic_bytes)
        if dep is None:
            return default
        else:
            return value


def xor_encode_decode(data, key):
    r"""
    Perform XOR encryption or decryption on the given data using a provided key.

    This function encrypts or decrypts the data by performing an XOR operation
    between each byte of the data and the corresponding byte of the key. The key
    is repeated if necessary to cover the entire length of the data.

    Parameters:
    - data: The byte data to be encrypted or decrypted.
    - key: The key used for encryption or decryption, also a sequence of bytes.

    Returns:
    - The resulting byte data after encryption or decryption.

    Example:

        >>> original_data = b'Hello, World!'
        >>> key = b'secret'
        >>> encrypted_data = xor_encode_decode(original_data, key)
        >>> encrypted_data
        b';\x00\x0f\x1e\nXS2\x0c\x00\t\x10R'
        >>> decrypted_data = xor_encode_decode(encrypted_data, key)
        >>> decrypted_data == original_data
        True
    """
    # Extend the key to ensure its length is at least as long as the data
    extended_key = key * (len(data) // len(key) + 1)
    # Perform XOR operation between each byte of the data and the extended key,
    # and return the new sequence of bytes
    return bytes([b ^ k for b, k in zip(data, extended_key)])


def is_running_win32(pid: int):
    with os.popen('tasklist /fo csv /fi "pid eq %s"' % int(pid)) as f:
        f.readline()
        # the second line is the result
        text = f.readline()
        return bool(text)


def is_running_linux(pid: int):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False
    except SystemError:
        return True


def is_running(pid):
    """Check if the given process ID is running.

    Parameters:
    pid -- The process ID to check.

    Returns:
    True if the process is running; False otherwise.

    Examples:
    >>> is_running(os.getpid() * 10)  # Assume process is not running
    False
    >>> current_pid = os.getpid()
    >>> # sometimes may return False on Windows due to permission issues!
    >>> is_running(current_pid) or is_running(current_pid)  # Check if the current process is running
    True
    >>> is_running("not_a_pid")  # Invalid PID input should be handled and return False
    False

    """
    try:
        pid = int(pid)
    except ValueError:
        return False
    if sys.platform == "win32":
        return is_running_win32(pid)
    else:
        return is_running_linux(pid)


def set_pid_file(
    path: Optional[Union[str, Path]] = None,
    raise_error=False,
    default_dir="",
    default_name="",
    default_level=2,
):
    """Sets a PID file to prevent multiple instances of a script or process from running concurrently.
    If no path is specified, it constructs a default based on the calling script's location and naming convention.

    The function checks for an existing PID file and verifies if the process is still running.

    If the process is running and `raise_error` is True, it raises a RuntimeError; otherwise, it returns False.
    Upon successful setup, it writes the current process ID (PID) to the file and schedules the file for deletion upon exit.

    Parameters:
    - path (Optional[Union[str, Path]]): The desired path for the PID file. Defaults to a generated path based on caller.
    - raise_error (bool): Determines whether to raise an error if the PID file is already locked. Defaults to False.
    - default_dir (str): The default directory for the PID file if none is provided. Defaults to "/dev/shm" or temp dir.
    - default_name (str): The default base name for the PID file. Automatically generated with the caller filename if not provided.
    - default_level (int): The number of directory levels to include in the default name from the caller's path.

    Returns:
    - Path: The path of the PID file if successfully created or updated.
    - False: If the PID file exists and the associated process is running, and `raise_error` is False.

    Raises:
    - ValueError: If unable to determine the caller's path or set a default name for the PID file.
    - RuntimeError: If `raise_error` is True and the PID file is locked by another process.

    Examples:

    >>> path = set_pid_file()  # Assuming this is the first run, should succeed
    >>> path.name.endswith('doctest.py.pid')
    True
    >>> set_pid_file()  # Simulating second instance trying to start, should raise error if raise_error=True
    False
    >>> path.unlink()
    """
    if path is None:
        if not default_name:
            try:
                stack = traceback.extract_stack()
                for index in range(len(stack) - 2, -1, -1):
                    caller_path = Path(stack[index].filename)
                    # print(caller_path)
                    if caller_path.is_file():
                        break
                else:
                    raise ValueError("Could not find caller path")
                if not caller_path.is_file():
                    pass
                parts: list = []
                for index, part in enumerate(caller_path.as_posix().split("/")[::-1]):
                    if index < default_level:
                        parts.insert(0, part)
                    else:
                        break
                default_name = "__".join(parts) + ".pid"
            except Exception as e:
                raise ValueError("Could not set default name for PID file: %s" % e)
        if not default_dir:
            default_dir = Path("/dev/shm")
            if not default_dir.is_dir():
                default_dir = Path(tempfile.gettempdir())
        path = Path(default_dir) / default_name
    if not isinstance(path, Path):
        path = Path(path)
    running = False
    if path.is_file():
        try:
            old_pid = int(path.read_text().strip())
            running = is_running(old_pid)
        except ValueError:
            # NaN pid
            pass

    if running:
        if raise_error:
            raise RuntimeError(f"{path.as_posix()} is locked by {old_pid}")
        else:
            return False
    else:
        pid_str = str(os.getpid())
        path.write_text(pid_str)
        new_pid = path.read_text().strip()
        if pid_str != new_pid:
            if raise_error:
                raise RuntimeError(
                    f"{path.as_posix()} is locked by other process {new_pid}"
                )
            else:
                return False

        def _release_file():
            if path.is_file() and path.read_text() == pid_str:
                path.unlink()

        atexit.register(_release_file)
        return path


def get_paste() -> Union[str, None]:
    """This module offers a simple utility for retrieving text from the system clipboard with tkinter.

    Function:
        get_paste() -> Union[str, None]

    Usage Note:
        While this function handles basic clipboard retrieval, for more advanced scenarios such as setting clipboard content or maintaining a persistent application interface, consider using libraries like `pyperclip` or running `Tkinter.mainloop` which keeps the GUI event loop active.
        Set clipboard with tkinter:
            _tk.clipboard_clear()
            _tk.clipboard_append(text)
            _tk.update()
            _tk.mainloop() # this is needed
    """
    from tkinter import TclError, Tk

    text = _tk = None
    try:
        _tk = Tk()
        _tk.withdraw()
        text = _tk.clipboard_get()
    except TclError:
        pass
    finally:
        if _tk:
            _tk.destroy()
        return text


def set_clip(text: str):
    """Copies the given text to the clipboard using a temporary file in a Windows environment.

    This function writes the provided text into a temporary file and then uses the `clip.exe` command-line utility
    to read from this file and copy its content into the clipboard.

    Parameters:
    text: str - The text content to be copied to the clipboard.
    """
    if sys.platform != "win32":
        raise RuntimeError("set_clip is only supported on Windows, you need pyperclip")
    try:
        path = Path(tempfile.gettempdir()) / "set_clip.txt"
        path.write_text(text)
        subprocess.run(
            f'clip.exe < "{path.name}"',
            cwd=path.parent.absolute(),
            shell=True,
        )
    finally:
        path.unlink(missing_ok=True)


class Clipboard:
    try:
        import pyperclip

        copy = pyperclip.copy
        paste = pyperclip.paste
    except ImportError:
        copy = set_clip
        paste = get_paste


def switch_flush_print():
    """Set builtins.print default flush=True.

    >>> print.__name__
    'print'
    >>> switch_flush_print()
    >>> print.__name__
    'flush_print'
    >>> switch_flush_print()
    >>> print.__name__
    'print'
    """
    import builtins

    if hasattr(builtins, "orignal_print"):
        orignal_print = builtins.orignal_print
        if builtins.print is not orignal_print:
            # back to orignal_print
            builtins.print = orignal_print
            return
    else:
        orignal_print = builtins.print
        builtins.orignal_print = orignal_print

    def flush_print(*args, **kwargs):
        if "flush" not in kwargs:
            kwargs["flush"] = True
        return orignal_print(*args, **kwargs)

    # set new flush_print
    builtins.print = flush_print


def unix_rlimit(max_mem: Optional[int] = None, max_file_size: Optional[int] = None):
    "Unix only. RLIMIT_RSS, RLIMIT_FSIZE to limit the max_memory and max_file_size"
    if sys.platform == "win32":
        raise RuntimeError("set_clip is only supported on Windows")
    import resource

    if max_mem is not None:
        resource.setrlimit(resource.RLIMIT_RSS, (max_mem, max_mem))
    if max_file_size is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))


class SimpleFilter:
    """Simple dup-filter with pickle file.

    >>> for r in range(1, 5):
    ...     try:
    ...         done = 0
    ...         with SimpleFilter("1.proc", 0, success_unlink=True) as sf:
    ...             for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    ...                 if sf.exist(i):
    ...                     continue
    ...                 if done > 2:
    ...                     print("crash!")
    ...                     raise ValueError
    ...                 print('[round %s]' % r, i, 'done', flush=True)
    ...                 sf.add(i)
    ...                 done += 1
    ...             break
    ...     except ValueError:
    ...         continue
    ...
    [round 1] 1 done
    [round 1] 2 done
    [round 1] 3 done
    crash!
    [round 2] 4 done
    [round 2] 5 done
    [round 2] 6 done
    crash!
    [round 3] 7 done
    [round 3] 8 done
    [round 3] 9 done
    crash!
    [round 4] 10 done
    """

    def __init__(self, path: Union[Path, str], delay=2, success_unlink=False):
        self.path = Path(path)
        self.delay = delay
        self.success_unlink = success_unlink

        self.timer = None
        self.has_new = False
        self.w_lock = Lock()
        self.current_thread = current_thread()

    def add(self, key):
        self.has_new = True
        self.cache.add(key)
        if not self.timer:
            self.timer = Timer(self.delay, self.save_cache)
            self.timer.start()

    def exist(self, key):
        return key in self.cache

    def save_cache(self):
        with self.w_lock:
            if self.has_new:
                self.path.write_bytes(pickle.dumps(self.cache))
            if self.timer:
                if current_thread() is not self.current_thread:
                    # timer finished
                    self.timer = None

    def __enter__(self):
        try:
            self.cache = pickle.loads(self.path.read_bytes())
        except Exception:
            self.cache = set()
            self.save_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer and not self.timer.is_alive():
            self.timer.cancel()
        if exc_type is None:
            # success & finish
            if self.success_unlink:
                self.has_new = False
                with self.w_lock:
                    # remove cache file
                    self.path.unlink(missing_ok=True)
                return
        self.save_cache()


class FileDict(dict):
    """A dict that can be saved to a file.

    Demo::

        >>> d = FileDict.load("test.json")
        >>> d.get('a', 'none')
        'none'
        >>> d['a'] = 1
        >>> d.save()
        >>> d2 = FileDict.load("test.json")
        >>> d2.get('a')
        1
        >>> d2.unlink()
        >>> d = FileDict.load("test.pkl")
        >>> d.get('a', 'none')
        'none'
        >>> d['a'] = 1
        >>> d.save()
        >>> d2 = FileDict.load("test.pkl")
        >>> d2.get('a')
        1
        >>> d2.unlink()
    """

    @classmethod
    def choose_handler(cls, filename):
        ext = os.path.splitext(filename)[1]
        if ext == ".json":
            return cls.json_handler
        elif ext == ".pkl":
            return cls.pickle_handler
        raise ValueError("unknown file extension")

    @staticmethod
    def json_handler(path: str, obj=None):
        if isinstance(obj, dict):
            temp = path + ".cache"
            try:
                with open(temp, "w") as f:
                    json.dump(obj, f)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                os.rename(temp, path)
            finally:
                try:
                    os.unlink(temp)
                except FileNotFoundError:
                    pass
        else:
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except FileNotFoundError:
                return {}

    @staticmethod
    def pickle_handler(path: str, obj=None):
        if isinstance(obj, dict):
            temp = path + ".cache"
            try:
                with open(temp, "wb") as f:
                    pickle.dump(obj, f)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                os.rename(temp, path)
            finally:
                try:
                    os.unlink(temp)
                except FileNotFoundError:
                    pass
        else:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return {}

    @classmethod
    def load(cls, path: str, handler=None):
        path = str(path)
        if not handler:
            handler = cls.choose_handler(str(path))
        self = cls(handler(path))
        setattr(self, "path", path)
        setattr(self, "handler", handler)
        return self

    def save(self):
        self.handler(self.path, self)

    def unlink(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def i2b(
    n: int, length=0, byteorder: Literal["litter", "big"] = "big", signed=False
) -> bytes:
    r"""Convert an int to bytes of a specified length, commonly used in TCP communication.

    Parameters:
        n: The integer to convert.
        length: The number of bytes to convert. Default is 0, which means the byte length is determined automatically based on the integer's bit length.
        byteorder: The byte order, which can be "big" or "little". Default is "big".
        signed: Whether the integer is signed. Default is False.

    Returns:
        The converted byte sequence.

    Length  Maximum::

        1   256B-1
        2    64K-1
        3    16M-1
        4     4G-1
        5     1T-1
        6    64T-1
        7    16E-1
        8   256Z-1
        9    32Y-1
        10    4P-1

    >>> i2b(0)
    b''
    >>> i2b(1)
    b'\x01'
    >>> i2b(1, length=2)
    b'\x00\x01'
    >>> i2b(255)
    b'\xff'
    >>> i2b(256)
    b'\x01\x00'
    >>> i2b(256, length=3)
    b'\x00\x01\x00'
    >>> i2b(256, byteorder="little")
    b'\x00\x01'
    >>> i2b(256, length=3, signed=True)
    b'\x00\x01\x00'
    """
    if not length:
        if signed:
            raise ValueError("signed must be False when length is 0")
        length = (n.bit_length() + 7) // 8
    return n.to_bytes(length, byteorder=byteorder, signed=signed)


def b2i(b: bytes, byteorder="big", signed=False) -> int:
    r"""Convert a byte sequence to an integer.

    Parameters:
        b: The byte sequence to convert.
        byteorder: The byte order, which can be "big" or "little". Default is "big".
        signed: Whether the integer is signed. Default is False.

    Returns:
        The converted integer.
    >>> b2i(b'\x01')
    1
    >>> b2i(b'\x00\x01')
    1
    >>> b2i(b'\x00\x01', byteorder="little")
    256
    >>> b2i(b'\xff')
    255
    >>> b2i(b'\x01\x00')
    256
    >>> b2i(b'\x00\x01\x00')
    256
    >>> b2i(b'\x00\x01\x00', signed=True)
    256
    >>> b2i(b'\x00\x01\x00', signed=False)
    256
    """
    return int.from_bytes(b, byteorder=byteorder, signed=signed)


class PathLock:
    """A Lock/asyncio.Lock of a path, and the child-path lock will block the parent-path.
    Ensure a path and its child-path are not busy. Can be used in a with statement to avoid race condition, such as rmtree.

    Demo::

        import asyncio
        import time
        from threading import Thread

        from morebuiltins.utils import Path, PathLock


        def test_sync_lock():
            parent = Path("/tmp")
            child = Path("/tmp/child")
            result = []

            def parent_job():
                with PathLock(parent):
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "parent done", flush=True)
                    result.append(parent)

            def child_job():
                with PathLock(child):
                    time.sleep(0.5)
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "child done", flush=True)
                    result.append(child)

            Thread(target=child_job).start()
            time.sleep(0.01)
            t = Thread(target=parent_job)
            t.start()
            t.join()
            # 2024-07-16 22:52:20 child done
            # 2024-07-16 22:52:20 parent done
            # child before parent
            assert result == [child, parent], result


        async def test_async_lock():
            parent = Path("/tmp")
            child = Path("/tmp/child")
            result = []

            async def parent_job():
                async with PathLock(parent):
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "parent done", flush=True)
                    result.append(parent)

            async def child_job():
                async with PathLock(child):
                    await asyncio.sleep(0.5)
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "child done", flush=True)
                    result.append(child)

            asyncio.create_task(child_job())
            await asyncio.sleep(0.01)
            await asyncio.create_task(parent_job())
            # 2024-07-16 22:52:21 child done
            # 2024-07-16 22:52:21 parent done
            # child before parent
            assert result == [child, parent], result


        test_sync_lock()
        asyncio.run(test_async_lock())
    """

    GLOBAL_LOCK: Optional[Lock] = None
    LOCKS: List[Tuple[Path, Lock]] = []
    GLOBAL_ALOCK: Optional[asyncio.Lock] = None
    ALOCKS: List[Tuple[Path, asyncio.Lock]] = []

    def __init__(self, path: Path, timeout=None):
        # ensure resolved, avoid the ../../ or softlink problem
        self.path = Path(path).resolve()
        self.timeout = timeout
        if self.GLOBAL_LOCK is None:
            self.__class__.GLOBAL_LOCK = Lock()

    @property
    def global_lock(self):
        if self.__class__.GLOBAL_LOCK is None:
            self.__class__.GLOBAL_LOCK = Lock()
        return self.__class__.GLOBAL_LOCK

    @property
    def global_alock(self):
        if self.__class__.GLOBAL_ALOCK is None:
            self.__class__.GLOBAL_ALOCK = asyncio.Lock()
        return self.__class__.GLOBAL_ALOCK

    def get_lock(self) -> Lock:
        with self.global_lock:
            for path, lock in self.LOCKS:
                if path.is_relative_to(self.path):
                    self.LOCKS.append((self.path, lock))
                    return lock
            lock = Lock()
            self.LOCKS.append((self.path, lock))
            return lock

    async def get_alock(self) -> asyncio.Lock:
        async with self.global_alock:
            for path, lock in self.ALOCKS:
                if path.is_relative_to(self.path):
                    self.ALOCKS.append((self.path, lock))
                    return lock
            lock = asyncio.Lock()
            self.ALOCKS.append((self.path, lock))
            return lock

    def __enter__(self):
        self.lock = self.get_lock()
        if self.timeout:
            self.lock.acquire(blocking=True, timeout=self.timeout)
        else:
            self.lock.acquire(blocking=True)

    def __exit__(self, *_):
        try:
            if self.GLOBAL_LOCK:
                locked = self.GLOBAL_LOCK.acquire(blocking=True)
            else:
                locked = False
            self.LOCKS.remove((self.path, self.lock))
        except ValueError:
            pass
        finally:
            if self.GLOBAL_LOCK and locked:
                self.GLOBAL_LOCK.release()
        self.lock.release()

    async def __aenter__(self):
        self.lock = await self.get_alock()
        await self.lock.acquire()

    async def __aexit__(self, *_):
        try:
            if self.GLOBAL_ALOCK:
                locked = await self.GLOBAL_ALOCK.acquire()
            else:
                locked = False
            self.ALOCKS.remove((self.path, self.lock))
        except ValueError:
            pass
        finally:
            if self.GLOBAL_ALOCK and locked:
                self.GLOBAL_ALOCK.release()
        self.lock.release()


def get_size(obj, seen=None, iterate_unsafe=False) -> int:
    """Recursively get size of objects.

    Args:
        obj: object of any type
        seen (set): set of ids of objects already seen
        iterate_unsafe (bool, optional): whether to iterate through generators/iterators. Defaults to False.

    Returns:
        int: size of object in bytes

    Examples:
    >>> get_size("") > 0
    True
    >>> get_size([]) > 0
    True
    >>> def gen():
    ...     for i in range(10):
    ...         yield i
    >>> g = gen()
    >>> get_size(g) > 0
    True
    >>> next(g)
    0
    >>> get_size(g, iterate_unsafe=True) > 0
    True
    >>> try:
    ...     next(g)
    ... except StopIteration:
    ...     "StopIteration"
    'StopIteration'
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, (str, bytes, bytearray)):
        pass
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen, iterate_unsafe) for v in obj.values()])
        size += sum([get_size(k, seen, iterate_unsafe) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen, iterate_unsafe)
    elif isinstance(obj, types.GeneratorType) or isinstance(
        obj, collections.abc.Iterator
    ):
        if iterate_unsafe:
            # Warning: this will consume the generator/iterator
            size += sum([get_size(i, seen, iterate_unsafe) for i in obj])
    elif hasattr(obj, "__iter__"):
        # Safe to iterate through containers like lists, tuples, sets, etc.
        size += sum([get_size(i, seen, iterate_unsafe) for i in obj])
    return size


def base_encode(
    num: int,
    alphabet: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
) -> str:
    """Encode a number to a base-N string.

    Args:
        num (int): The number to encode.
        alphabet (str, optional): Defaults to "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", which follows the order of ASCII characters.

    Returns:
        str: The encoded string.

    Examples:
    >>> base_encode(0)
    '0'
    >>> base_encode(1)
    '1'
    >>> base_encode(10000000000000)
    '2q3Rktoe'
    >>> base_encode(10000000000000, "0123456789")
    '10000000000000'
    >>> a = [base_encode(i).zfill(10) for i in range(10000)]
    >>> sorted(a) == a
    True
    """
    length = len(alphabet)
    result = ""
    while num:
        num, i = divmod(num, length)
        result = f"{alphabet[i]}{result}"
    return result or alphabet[0]


def base_decode(
    string: str,
    alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
) -> int:
    """Decode a base-N string to a number.

    Args:
        string (str): The string to decode.
        alphabet (str, optional): Defaults to "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".

    Returns:
        int: The decoded number.

    Examples:
    >>> base_decode("0")
    0
    >>> base_decode("1")
    1
    >>> base_decode("2Q3rKTOE")
    10000000000000
    >>> base_decode("10000000000000", "0123456789")
    10000000000000
    """
    length = len(alphabet)
    result = 0
    for char in string:
        result = result * length + alphabet.index(char)
    return result


def gen_id(rand_len=4) -> str:
    """Generate a unique ID based on the current time and random bytes

    Args:
        rand_len (int, optional): Defaults to 4.

    Returns:
        str: The generated ID.

    Examples:
    >>> a, b = gen_id(), gen_id()
    >>> a != b
    True
    >>> import time
    >>> ids = [time.sleep(0.000001) or gen_id() for _ in range(1000)]
    >>> len(set(ids))
    1000
    """
    now = datetime.now()
    s1 = now.strftime("%y%m%d_%H%M%S")
    s2 = base_encode(now.microsecond)
    s2 = f"{s2:>04}{os.urandom(rand_len // 2).hex()}"
    return f"{s1}_{s2}"


def timeti(
    stmt: Union[str, Callable] = "pass",
    setup="pass",
    timer=timeit.default_timer,
    number=1000000,
    globals=None,
) -> int:
    """Return the number of iterations per second for a given statement.

    Args:
        stmt (str, optional): Defaults to "pass".
        setup (str, optional): Defaults to "pass".
        timer (optional): Defaults to timeit.default_timer.
        number (int, optional): Defaults to 1000000.
        globals (dict, optional): Defaults to None.

    Returns:
        int: The number of iterations per second.

    Examples:
    >>> timeti("1 / 1") > 1000000
    True
    >>> timeti(lambda : 1 + 1, number=100000) > 100000
    True
    """
    result = timeit.timeit(
        stmt=stmt, setup=setup, timer=timer, number=number, globals=globals
    )
    return int(1 / (result / number))


class SnowFlake:
    def __init__(
        self,
        machine_id: int = 1,
        worker_id: int = 1,
        wait_for_next_ms=True,
        start_epoch_ms: int = -1,
        *,
        max_bits=64,
        sign_bit=1,
        machine_id_bit=5,  # between 0 and 31
        worker_id_bit=5,  # between 0 and 31
        seq_bit=12,  # between 0 and 4095
        start_date="2025-01-01",
    ):
        r"""Generate unique IDs using Twitter's Snowflake algorithm.

        The ID is composed of:
        - 41 bits timestamp in milliseconds since a custom epoch
        - 5 bits machine ID
        - 5 bits worker ID
        - 12 bits sequence number
        If you need to ensure thread safety, please lock it yourself.

            Args:
                machine_id (int): the machine_id of the SnowFlake object
                worker_id (int): the worker_id of the SnowFlake object
                wait_for_next_ms (bool, optional): whether to wait for next millisecond if sequence overflows. Defaults to True.
                start_epoch_ms (int, optional): the start epoch in milliseconds. Defaults to -1.
                *
                max_bits (int, optional): the maximum bits of the ID. Defaults to 64.
                sign_bit (int, optional): the sign bit of the ID. Defaults to 1.
                machine_id_bit (int, optional): the machine_id bit of the ID. Defaults to 5.
                worker_id_bit (int, optional): the worker_id bit of the ID. Defaults to 5.
                seq_bit (int, optional): the sequence bit of the ID. Defaults to 12.
                start_date (str): the start date of the SnowFlake object. Defaults to "2025-01-01", only used when start_epoch_ms is -1.

            Example:
            >>> import time
            >>> snowflake = SnowFlake()
            >>> ids = [snowflake.get_id() for _ in range(10000)]
            >>> len(set(ids)) == len(ids)
            True
            >>> # test timestamp overflow
            >>> snowflake = SnowFlake(1, 1, start_date=time.strftime("%Y-%m-%d"))
            >>> timeleft = snowflake.timestamp_overflow_check() // 1000 // 60 // 60 // 24 // 365
            >>> timeleft == 69
            True
            >>> snowflake = SnowFlake(1, 1, start_date=time.strftime("%Y-%m-%d"), sign_bit=0)
            >>> timeleft = snowflake.timestamp_overflow_check() // 1000 // 60 // 60 // 24 // 365
            >>> timeleft >= 138
            True
            >>> # test machine_id and worker_id overflow
            >>> try:
            ...     snowflake = SnowFlake(32, 32)
            ... except ValueError as e:
            ...     e
            ValueError('Machine ID must be between 0 and 31')
            >>> sf = SnowFlake(32, 32, machine_id_bit=6, worker_id_bit=6)
            >>> sf.max_machine_id
            63
            >>> sf = SnowFlake(32, machine_id_bit=64)
            >>> sf.timestamp_overflow_check() < 0
            True
        """
        self.max_bits = max_bits
        self.sign_bit = sign_bit
        self.machine_id_bit = machine_id_bit
        self.worker_id_bit = worker_id_bit
        self.seq_bit = seq_bit
        self.max_seq = (1 << self.seq_bit) - 1
        self.max_worker_id = (1 << self.worker_id_bit) - 1
        self.max_machine_id = (1 << self.machine_id_bit) - 1
        self.max_timestamp = self.get_max_timestamp()
        # Validate inputs
        if not 0 <= machine_id <= self.max_machine_id:
            raise ValueError(f"Machine ID must be between 0 and {self.max_machine_id}")
        if not 0 <= worker_id <= self.max_worker_id:
            raise ValueError(f"Worker ID must be between 0 and {self.max_worker_id}")
        self.timestamp_shift = self.seq_bit + self.worker_id_bit + self.machine_id_bit
        self.machine_id = machine_id
        self.worker_id = worker_id
        # Calculate parts of ID
        self.machine_id_part = machine_id << (self.seq_bit + self.worker_id_bit)
        self.worker_id_part = worker_id << self.seq_bit
        # Initialize sequence, timestamp and last timestamp
        self.seq = 0
        self.last_timestamp = -1
        if start_epoch_ms < 0:
            self.start_epoch_ms = self.str_to_ms(start_date)
        else:
            self.start_epoch_ms = start_epoch_ms
        self.wait_for_next_ms = wait_for_next_ms

    def get_max_timestamp(self):
        """Get maximum timestamp that can be represented"""
        return (1 << (self.max_bits - self.sign_bit)) - 1 >> (
            self.seq_bit + self.worker_id_bit + self.machine_id_bit
        )

    def timestamp_overflow_check(self):
        """Check how many milliseconds left until timestamp overflows"""
        return self.get_max_timestamp() - self._ms_passed()

    @staticmethod
    def str_to_ms(string: str) -> int:
        """Convert string to milliseconds since start_time"""
        return int(mktime(strptime(string, "%Y-%m-%d")) * 1000)

    def _ms_passed(self):
        """Get current timestamp in milliseconds since start_time"""
        return int(time() * 1000 - self.start_epoch_ms)

    def _wait_next_millis(self, last_timestamp):
        """Wait until next millisecond"""
        if not self.wait_for_next_ms:
            raise RuntimeError(
                f"Over {self.max_seq} IDs generated in 1ms, increase the wait_for_next_ms parameter"
            )
        timestamp = self._ms_passed()
        while timestamp <= last_timestamp:
            timestamp = self._ms_passed()
        return timestamp

    def get_id(self):
        """Generate next unique ID"""
        now = self._ms_passed()

        # Clock moved backwards, reject requests
        if now < self.last_timestamp:
            raise RuntimeError(
                f"Clock moved backwards. Refusing to generate ID for {self.last_timestamp - now} milliseconds"
            )

        # Same timestamp, increment sequence
        if now == self.last_timestamp:
            self.seq = (self.seq + 1) & self.max_seq
            # Sequence overflow, wait for next millisecond
            if self.seq == 0:
                now = self._wait_next_millis(now)
        else:
            # Reset sequence for different timestamp
            self.seq = 0

        self.last_timestamp = now
        # Compose ID from components
        return (
            (now << self.timestamp_shift)
            | (self.machine_id_part)
            | (self.worker_id_part)
            | self.seq
        )


def cut_file(path_or_file, max_bytes, remain_ratio=0.5, ensure_line_start=False):
    r"""Cut file to max_bytes, remain_ratio is the ratio of the end part to remain, ensure_line_start is to ensure the first line is a complete line

    Args:
       path_or_file (str/Path/file): input file path or file-like object
       max_bytes (int): max bytes before cut
       remain_ratio (float, optional): remain ratio of the end part. Defaults to 0.5.
       ensure_line_start (bool, optional): ensure the first line is a complete line. Defaults to False.

    Examples:
    >>> from io import BytesIO
    >>> f = BytesIO(b"1234567890\n1234567890\n1234567890\n")
    >>> cut_file(f, 30)
    >>> f.getvalue()
    b'890\n1234567890\n'
    >>> f = BytesIO(b"1234567890\n1234567890\n1234567890\n")
    >>> cut_file(f, 30, ensure_line_start=True)
    >>> f.getvalue()
    b'1234567890\n'
    >>> f = BytesIO(b"1234567890\n1234567890\n1234567890\n")
    >>> cut_file(f, 30, remain_ratio=0)
    >>> f.getvalue()
    b''
    """
    need_close = False
    try:
        if isinstance(path_or_file, (str, Path)):
            f = open(path_or_file, "r+b")
            need_close = True
        elif hasattr(path_or_file, "read"):
            f = path_or_file
        else:
            raise TypeError("path_or_file must be str, Path or file-like object")
        size = f.seek(0, 2)
        if size > max_bytes:
            f.seek(int(size - max_bytes * remain_ratio))
            left = f.read()
            if ensure_line_start:
                left = re.sub(b"^[^\n]+\n", b"", left, count=1)
            f.seek(0)
            f.truncate()
            f.write(left)
            f.flush()
    finally:
        if need_close:
            f.close()


if __name__ == "__main__":
    __name__ = "morebuiltins.utils"
    import doctest

    doctest.testmod()

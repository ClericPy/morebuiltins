import asyncio
import atexit
import base64
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from collections import UserDict
from enum import IntEnum
from functools import wraps
from itertools import groupby, islice
from os.path import basename, exists
from pathlib import Path
from time import gmtime, mktime, strftime, strptime, time, timezone
from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    Generator,
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


def slice_into_pieces(
    items: Sequence, n: int
) -> Generator[Union[tuple, Sequence], None, None]:
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
) -> Generator[Union[tuple, Sequence], None, None]:
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
    n: Union[tuple, list, int, None] = None,
    default: Callable = lambda obj: str(obj).encode("utf-8"),
    func=hashlib.md5,
) -> str:
    """Generates an MD5 hash string from the given input string.

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

    >>> code1 = ''
    >>> code2 = code_inline("variable=12345")
    >>> # import base64,gzip;exec(gzip.decompress(base64.b85decode("ABzY8mBl+`0{<&ZEXqtw%1N~~G%_|Z1ptx!(o_xr000".encode("u8"))))
    >>> exec(code2)
    >>> variable
    12345

    Args:
        source_code (str): python original code.
        encoder (Literal['b16', 'b32', 'b64', 'b85'], optional): base64.encoder. Defaults to "b85".
    Returns:
        new source code inline.
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


def read_num(
    b,
    enum=IntEnum,
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
    """Ensures that identical groups are ordered and evenly distributed, mitigating data skew. The function does not alter the original list and returns a generator.

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


def default_dict(cls: Type[dict], **kwargs) -> dict:
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
    >>> default_dict(Demo, bytes_obj=b'1')
    {'int_obj': 0, 'float_obj': 0.0, 'bytes_obj': b'1', 'str_obj': '', 'list_obj': [], 'tuple_obj': (), 'set_obj': set(), 'dict_obj': {}}
    """
    result = cls()
    built_in_types = {int, float, bytes, str, list, tuple, set, dict}
    for key, tp in cls.__annotations__.items():
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
    # filt tb with code-line and not in site-packages
    return tb.line and "site-packages" not in tb.filename


def format_error(
    error: BaseException,
    index: Union[int, slice] = slice(-3, None, None),
    filter: Optional[Callable] = _tb_filter,
    template="[{trace_routes}] {error_line} >>> {error.__class__.__name__}({error!s:.100})",
    **kwargs,
) -> str:
    r"""Extracts frame information from an exception, with an option to filter out “site-packages” details by default.

    Parameters:

    - `error` (`BaseException`): The exception instance for which the stack trace information is to be extracted and formatted.
    - `index` (`Union[int, slice]`, optional): Specifies which frames to include in the output. By default, it's set to `slice(-3, None, None)`, showing the last three frames. Can be an integer for a single frame or a slice object for a range of frames.
    - `filter` (`Optional[Callable]`, optional): A callable that determines whether a given frame should be included. Defaults to `_tb_filter`, which typically filters out frames from "site-packages". If set to `None`, no filtering occurs.
    - `template` (`str`, optional): A string template defining how the error message should be formatted. It can include placeholders like `{trace_routes}`, `{error_line}`, and `{error.__class__.__name__}`. The default template provides a concise summary of the error location and type.
    - `**kwargs`: Additional keyword arguments to be used within the formatting template.

    Returns:

    A string representing the formatted error message based on the provided parameters and template.

    Demo:

    >>> try:
    ...     # test default
    ...     1 / 0
    ... except Exception as e:
    ...     format_error(e)
    '[<doctest>:<module>:3] 1 / 0 >>> ZeroDivisionError(division by zero)'
    >>> try:
    ...     # test in function
    ...     def func1(): 1 / 0
    ...     func1()
    ... except Exception as e:
    ...     format_error(e)
    '[<doctest>:<module>:4 | <doctest>:func1:3] def func1(): 1 / 0 >>> ZeroDivisionError(division by zero)'
    >>> try:
    ...     # test index
    ...     def func2(): 1 / 0
    ...     func2()
    ... except Exception as e:
    ...     format_error(e, index=0)
    '[<doctest>:<module>:4] func2() >>> ZeroDivisionError(division by zero)'
    >>> try:
    ...     # test slice index
    ...     def func2(): 1 / 0
    ...     func2()
    ... except Exception as e:
    ...     format_error(e, index=slice(-1, None, None))
    '[<doctest>:func2:3] def func2(): 1 / 0 >>> ZeroDivisionError(division by zero)'
    >>> try:
    ...     # test with default filter(filename skip site-packages)
    ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
    ...     version_info_to_nodot(0)
    ... except Exception as e:
    ...     format_error(e)
    "[<doctest>:<module>:4] version_info_to_nodot(0) >>> TypeError('int' object is not subscriptable)"
    >>> try:
    ...     # test without filter
    ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
    ...     version_info_to_nodot(0)
    ... except Exception as e:
    ...     format_error(e, filter=None)
    '[<doctest>:<module>:4 | compatibility_tags.py:version_info_to_nodot:23] return "".join(map(str, version_info[:2])) >>> TypeError(\'int\' object is not subscriptable)'
    >>> try:
    ...     # test with custom filter.
    ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
    ...     version_info_to_nodot(0)
    ... except Exception as e:
    ...     format_error(e, filter=lambda i: '<doctest' in str(i))
    "[<doctest>:<module>:4] version_info_to_nodot(0) >>> TypeError('int' object is not subscriptable)"
    """
    try:
        filter = filter or always_return_value(True)
        tbs = [tb for tb in traceback.extract_tb(error.__traceback__) if filter(tb)]
        if isinstance(index, slice):
            tbs = tbs[index]
        elif isinstance(index, int):
            tbs = [tbs[index]]
        else:
            raise ValueError("Invalid index type")
        trace_route_list = []
        for tb in tbs:
            filename = tb.filename
            if exists(filename):
                _basename = basename(filename)
            elif filename[0] == "<":
                _basename = f"{filename.split()[0]}>"
            else:
                _basename = filename
            trace_route_list.append(f"{_basename}:{tb.name}:{tb.lineno}")
        trace_routes = " | ".join(trace_route_list)
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
    >>> is_running(os.getpid())  # Check if the current process is running
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

    _tk = Tk()
    _tk.withdraw()
    text = None
    try:
        text = _tk.clipboard_get()
    except TclError:
        pass
    finally:
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
        raise RuntimeError("set_clip is only supported on Windows")
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


if __name__ == "__main__":
    __name__ = "morebuiltins.utils"
    import doctest

    doctest.testmod()

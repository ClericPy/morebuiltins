# [morebuiltins](https://github.com/ClericPy/morebuiltins)

> I need much more built-ins

[![PyPI](https://img.shields.io/pypi/v/morebuiltins?style=plastic)](https://pypi.org/project/morebuiltins/)[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/clericpy/morebuiltins/pythonpackage.yml)](https://github.com/ClericPy/morebuiltins/actions)![PyPI - Wheel](https://img.shields.io/pypi/wheel/morebuiltins?style=plastic)![PyPI - Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fgithub.com%2FClericPy%2Fmorebuiltins%2Fraw%2Fmaster%2Fpyproject.toml
)![PyPI - Downloads](https://img.shields.io/pypi/dm/morebuiltins?style=plastic)![PyPI - License](https://img.shields.io/pypi/l/morebuiltins?style=plastic)
## Purpose

1. Collect commonly used code snippets to make up for the lack of built-in functions
2. Provide some common ideas for beginners
3. Stay Hungry, Stay Foolish

## Install

From **pypi.org**

> pip install -U morebuiltins

Or use `pyz`(pyz file may be downloaded from [releases](https://github.com/ClericPy/morebuiltins/releases))

> pip install zipapps
> 
> python -m zipapps -c -o morebuiltins.pyz morebuiltins

```python
import sys
sys.path.insert(0, 'morebuiltins.pyz')
import morebuiltins
print(morebuiltins.__file__)
# morebuiltins.pyz\morebuiltins\__init__.py
```

## Doc

---

Module Docs - https://github.com/ClericPy/morebuiltins/blob/master/doc.md

<!-- start -->
## 1. morebuiltins.utils

1.1 `ttime` - Converts a timestamp to a human-readable timestring formatted as %Y-%m-%d %H:%M:%S.

1.2 `ptime` - Converts a timestring formatted as %Y-%m-%d %H:%M:%S back into a timestamp.

1.3 `slice_into_pieces` - Divides a sequence into “n” segments, returning a generator that yields “n” pieces.

1.4 `slice_by_size` - Slices a sequence into chunks of a specified “size”, returning a generator that produces tuples of chunks.

1.5 `unique` - Removes duplicate elements from a sequence while preserving the original order efficiently.

1.6 `retry` - A decorator that retries the decorated function up to “tries” times if the specified exceptions are raised.

1.7 `guess_interval` - Analyzes a sequence of numbers and returns the median, calculating intervals only if they are greater than or equal to the specified accuracy.

1.8 `get_hash` - Generates an MD5 hash string from the given input string.

1.9 `find_jsons` - A generator that locates valid JSON strings, supporting only dictionaries and lists.

1.10 `code_inline` - Minifies Python source code into a single line.

1.11 `read_size` - Converts byte counts into a human-readable string. Setting shorten=True and precision=0.99 will trim unnecessary decimal places from the tail of floating-point numbers.

1.12 `read_time` - Converts seconds into a more readable time duration string.

1.13 `Validator` - Validator for dataclasses.

1.14 `stagger_sort` - Ensures that identical groups are ordered and evenly distributed, mitigating data skew. The function does not alter the original list and returns a generator.

1.15 `default_dict` - Initializes a dictionary with default zero values based on a subclass of TypedDict.

1.16 `always_return_value` - Got a function always return the given value.

1.17 `format_error` - Extracts frame information from an exception, with an option to filter out “site-packages” details by default.

1.18 `Trie` - Transforms a standard dictionary into a trie structure that supports prefix matching.

1.19 `GuessExt` - Determines whether the input bytes of a file prefix indicate a compressed file format.


## 2. morebuiltins.functools

2.1 `lru_cache_ttl` - A Least Recently Used (LRU) cache with a Time To Live (TTL) feature.

2.2 `threads` - Quickly convert synchronous functions to be concurrency-able. (similar to madisonmay/Tomorrow)

2.3 `background_task` - Avoid asyncio free-flying tasks, better to use the new asyncio.TaskGroup to avoid this in 3.11+. https://github.com/python/cpython/issues/91887


## 3. morebuiltins.ipc

3.1 `IPCEncoder` - An abstract base class for all encoders; implementing the necessary communication protocol requires only the definition of two abstract methods. Be mindful that varying header lengths will impact the maximum packaging size.

3.4 `SocketLogHandlerEncoder` - For a practical demonstration, refer to the test code: morebuiltins/ipc.py:_test_ipc_logging.

3.5 `SocketServer` - To see an example in action, view the test code: morebuiltins/ipc.py:_test_ipc.


## 4. morebuiltins.request

4.1 `req` - A basic mock for requests, performant albeit simplistic.

4.2 `DomainParser` - Extracts the Second-level domain (SLD) from a provided hostname or URL.

4.3 `unparse_qsl` - Provides the inverse operation of parse_qsl, converting query string lists back into a URL-encoded string.

4.4 `update_url` - Organizes the query arguments within a URL to standardize its format.


<!-- end -->

## On the way

- [x] add zipapps as a submodule(https://github.com/ClericPy/zipapps)
- [x] asyncio free-flying tasks(bg_task)
- [ ] progress_bar
- [ ] http.server (upload)
- [ ] function parser (signature.parameters)
- [ ] time reach syntax
- [ ] quick tkinter
- [ ] http request/response parser
- [ ] named lock with timeout
- [ ] TimeSizeRotatingHandler of logging.handlers

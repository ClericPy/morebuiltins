# [morebuiltins](https://github.com/ClericPy/morebuiltins)

[![PyPI](https://img.shields.io/pypi/v/morebuiltins?style=plastic)](https://pypi.org/project/morebuiltins/)[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/clericpy/morebuiltins/pythonpackage.yml)](https://github.com/ClericPy/morebuiltins/actions)![PyPI - Wheel](https://img.shields.io/pypi/wheel/morebuiltins?style=plastic)![PyPI - Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fgithub.com%2FClericPy%2Fmorebuiltins%2Fraw%2Fmaster%2Fpyproject.toml
)![PyPI - Downloads](https://img.shields.io/pypi/dm/morebuiltins?style=plastic)![PyPI - License](https://img.shields.io/pypi/l/morebuiltins?style=plastic)


## Project Overview:

This project enhances Python’s built-ins with extra functionality, prioritizing no third-party dependencies, performance, reusability, and MIT Licensing for flexible use.

- Minimal Dependencies: Designed to avoid third-party libraries for seamless integration, conflict prevention, and easy packaging via zipapp for pyz distribution.
- Performance-Driven: It prioritizes efficiency, ensuring enhanced built-ins maintain Python's performance standards.
- Self-Contained Modules: Functions and modules are standalone for simple reuse without dependencies.
- MIT Licensed: The open-source code is freely usable, modifiable, and distributable.
- Enhanced Built-ins: The project adds tools to Python's core functionality, simplifying and accelerating development.

## Install

From [pypi.org](https://pypi.org/project/morebuiltins/)

> `pip install -U morebuiltins`

From [Github Tags](https://github.com/ClericPy/morebuiltins/tags):
> 
> `pip install git+https://github.com/ClericPy/morebuiltins.git@master`
>
> `pip install git+https://github.com/ClericPy/morebuiltins.git@0.0.6`

Use `pyz`(downloaded from [releases](https://github.com/ClericPy/morebuiltins/releases))

```python
import sys
sys.path.insert(0, 'morebuiltins.pyz')
import morebuiltins
print(morebuiltins.__file__)
# morebuiltins.pyz/morebuiltins/__init__.py
```

## Doc

[Module Docs](https://github.com/ClericPy/morebuiltins/blob/master/doc.md)

[Changelog](https://github.com/ClericPy/morebuiltins/blob/master/CHANGELOG.md)


---

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

1.20 `xor_encode_decode` - Perform XOR encryption or decryption on the given data using a provided key.

1.21 `is_running` - Check if the given process ID is running.

1.22 `set_pid_file` - Sets a PID file to prevent multiple instances of a script or process from running concurrently.

1.23 `get_paste` - This module offers a simple utility for retrieving text from the system clipboard with tkinter.

1.24 `set_clip` - Copies the given text to the clipboard using a temporary file in a Windows environment.

1.25 `switch_flush_print` - Set builtins.print default flush=True.


## 2. morebuiltins.date

2.1 `ScheduleTimer` - The ScheduleTimer class facilitates the creation and evaluation of datetime patterns for scheduling purposes.

2.2 `Crontab` - Crontab python parser.


## 3. morebuiltins.functools

3.1 `lru_cache_ttl` - A Least Recently Used (LRU) cache with a Time To Live (TTL) feature.

3.2 `threads` - Quickly convert synchronous functions to be concurrency-able. (similar to madisonmay/Tomorrow)

3.3 `bg_task` - Avoid asyncio free-flying tasks, better to use the new asyncio.TaskGroup to avoid this in 3.11+. https://github.com/python/cpython/issues/91887

3.4 `NamedLock` - Reusable named locks, support for timeouts, support for multiple concurrent locks.

3.5 `FuncSchema` - Parse the parameters and types required by a function into a dictionary, and convert an incoming parameter into the appropriate type.

3.6 `InlinePB` - Inline progress bar.

3.7 `SizedTimedRotatingFileHandler` - TimedRotatingFileHandler with maxSize, to avoid files that are too large.


## 4. morebuiltins.ipc

4.1 `IPCEncoder` - An abstract base class for all encoders; implementing the necessary communication protocol requires only the definition of two abstract methods. Be mindful that varying header lengths will impact the maximum packaging size.

4.4 `SocketLogHandlerEncoder` - For a practical demonstration, refer to the test code: morebuiltins/ipc.py:_test_ipc_logging.

4.5 `SocketServer` - To see an example in action, view the test code: morebuiltins/ipc.py:_test_ipc.

4.7 `find_free_port` - Finds and returns an available port number.


## 5. morebuiltins.request

5.1 `req` - A basic mock for requests, performant albeit simplistic.

5.2 `DomainParser` - Extracts the Second-level domain (SLD) from a provided hostname or URL.

5.3 `unparse_qsl` - Provides the inverse operation of parse_qsl, converting query string lists back into a URL-encoded string.

5.4 `update_url` - Organizes the query arguments within a URL to standardize its format.

5.6 `make_response` - Generates an HTTP response based on the provided parameters.

5.7 `custom_dns` - Custom the DNS of socket.getaddrinfo, only effect current thread.


## 6. morebuiltins.download_python

6.1 `download_python` - Download python portable interpreter from https://github.com/indygreg/python-build-standalone/releases. `python -m morebuiltins.download_python`


<!-- end -->

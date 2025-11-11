# [morebuiltins](https://github.com/ClericPy/morebuiltins)

[![PyPI](https://img.shields.io/pypi/v/morebuiltins?style=plastic)](https://pypi.org/project/morebuiltins/)[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/clericpy/morebuiltins/pythonpackage.yml)](https://github.com/ClericPy/morebuiltins/actions)![PyPI - Wheel](https://img.shields.io/pypi/wheel/morebuiltins?style=plastic)![PyPI - Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fgithub.com%2FClericPy%2Fmorebuiltins%2Fraw%2Fmaster%2Fpyproject.toml
)![PyPI - Downloads](https://img.shields.io/pypi/dm/morebuiltins?style=plastic)![PyPI - License](https://img.shields.io/pypi/l/morebuiltins?style=plastic)


## Project Overview:

This project enhances Python’s built-ins with extra functionality, prioritizing no third-party dependencies, performance, reusability, and MIT Licensing for flexibl- Minima- MIT Licensed: The open-source code is freely usable, modifiable, and distributable.
- Performance-Driven: It prioritizes efficiency, ensuring enhanced built-ins maintain Python's performance standards.
- Self-Contained Modules: Functions and modules are standalone for simple reuse without dependencies.
- Well tested: All modules are thoroughly tested for stability and compatibility.
- Enhanced Built-ins: The project adds tools to Python's core functionality, simplifying and accelerating development.
- Beginner friendly: Plenty of code snippets and examples for common scenarios, so new Python users can quickly find practical solutions and learn best practices.
- Command-line ready: Includes CLI tools for downloading Python, building zipapps, running log servers, analyzing dependencies, and more.
- Minimal Dependencies: Designed to avoid third-party libraries for seamless integration, conflict prevention, and easy packaging via zipapp for pyz distribution.
ibution.

## Compatibility Break Warnings

1. **1.3.4 (2025-09-17)**
   1. Refactor of `morebuiltins.cmd.log_server`
2. **1.3.2 (2025-08-02)**
   1. move `async_logger`, `AsyncQueueListener`, `LogHelper`, `RotatingFileWriter`, and `SizedTimedRotatingFileHandler` from `morebuiltins.funcs` to `morebuiltins.logs`.
3. **1.3.0 (2025-03-08)**
   1. rename `morebuiltins.functools` to `morebuiltins.funcs` to avoid conflict with `functools` in python standard library.

## Install

- From [pypi.org](https://pypi.org/project/morebuiltins/)

> `pip install -U morebuiltins`

- From [Github Tags](https://github.com/ClericPy/morebuiltins/tags):
> 
> `pip install git+https://github.com/ClericPy/morebuiltins.git@master`
>
> `pip install git+https://github.com/ClericPy/morebuiltins.git@1.0.3`

- Use `pyz`(downloaded from [releases](https://github.com/ClericPy/morebuiltins/releases))

```python
import sys
sys.path.insert(0, 'morebuiltins.pyz')
import morebuiltins
print(morebuiltins.__file__)
# morebuiltins.pyz/morebuiltins/__init__.py
```
- **morebuiltins.pyz command-line**
> zipapps: `python morebuiltins.pyz -m morebuiltins.zipapps -o morebuiltins.pyz -c morebuiltins`
>
> download_python: `python morebuiltins.pyz -m morebuiltins.download_python`

## Doc

[Module Docs](https://github.com/ClericPy/morebuiltins/blob/master/doc.md)

[Changelog](https://github.com/ClericPy/morebuiltins/blob/master/CHANGELOG.md)

**Compatibility Warning**. rename `morebuiltins.functools` to `morebuiltins.funcs` to avoid conflict with `functools` in python standard library after 1.3.0 (2025-04-03).

## More Modules:

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

1.8 `get_hash` - Generates a hash string from the given input string.

1.9 `find_jsons` - A generator that locates valid JSON strings, supporting only dictionaries and lists.

1.10 `code_inline` - Minifies Python source code into a single line.

1.11 `read_size` - Converts byte counts into a human-readable string. Setting shorten=True and precision=0.99 will trim unnecessary decimal places from the tail of floating-point numbers.

1.12 `read_time` - Converts seconds into a more readable time duration string.

1.13 `Validator` - Validator for dataclasses.

1.14 `stagger_sort` - Ensures that identical groups are ordered and evenly distributed, mitigating data skew. The function does not alter the original list and returns a generator.

1.15 `default_dict` - Initializes a dictionary with default zero values based on a subclass of TypedDict.

1.16 `always_return_value` - Got a function always return the given value.

1.17 `format_error` - Extracts frame information from an exception, with an option to filter out “-packages” details by default. To shorten your exception message.

1.18 `Trie` - Transforms a standard dictionary into a trie structure that supports prefix matching.

1.19 `GuessExt` - Determines whether the input bytes of a file prefix indicate a compressed file format.

1.20 `xor_encode_decode` - Perform XOR encryption or decryption on the given data using a provided key.

1.21 `is_running` - Check if the given process ID is running.

1.22 `set_pid_file` - Sets a PID file to prevent multiple instances of a script or process from running concurrently.

1.23 `get_paste` - This module offers a simple utility for retrieving text from the system clipboard with tkinter.

1.24 `set_clip` - Copies the given text to the clipboard using a temporary file in a Windows environment.

1.25 `switch_flush_print` - Set builtins.print default flush=True.

1.26 `unix_rlimit` - Unix only. RLIMIT_RSS, RLIMIT_FSIZE to limit the max_memory and max_file_size

1.27 `SimpleFilter` - Simple dup-filter with pickle file.

1.28 `FileDict` - A dict that can be saved to a file.

1.29 `PathLock` - A Lock/asyncio.Lock of a path, and the child-path lock will block the parent-path.

1.30 `i2b` - Convert an int to bytes of a specified length, commonly used in TCP communication.

1.31 `b2i` - Convert a byte sequence to an integer.

1.32 `get_hash_int` - Generates a int hash(like docid) from the given input bytes.

1.33 `iter_weights` - Generates an element sequence based on weights.

1.34 `get_size` - Recursively get size of objects.

1.35 `base_encode` - Encode a number to a base-N string.

1.36 `base_decode` - Decode a base-N string to a number.

1.37 `gen_id` - Generate a readable & unique ID based on the current time(ms) and random bytes.

1.38 `timeti` - Return the number of iterations per second for a given statement.

1.40 `cut_file` - Cut file to max_bytes, remain_ratio is the ratio of the end part to remain, ensure_line_start is to ensure the first line is a complete line


## 2. morebuiltins.date

2.1 `ScheduleTimer` - The ScheduleTimer class facilitates the creation and evaluation of datetime patterns for scheduling purposes.

2.2 `Crontab` - Crontab python parser.


## 3. morebuiltins.funcs

3.1 `lru_cache_ttl` - A Least Recently Used (LRU) cache with a Time To Live (TTL) feature.

3.2 `threads` - Quickly convert synchronous functions to be concurrent. (similar to madisonmay/Tomorrow)

3.3 `bg_task` - Avoid asyncio free-flying tasks, better to use the new asyncio.TaskGroup to avoid this in 3.11+. https://github.com/python/cpython/issues/91887

3.4 `NamedLock` - Reusable named locks, support for timeouts, support for multiple concurrent locks.

3.5 `FuncSchema` - Parse the parameters and types required by a function into a dictionary, and convert an incoming parameter into the appropriate type.

3.6 `InlinePB` - Inline progress bar.

3.7 `get_type_default` - Get the default value for a type. {int: 0, float: 0.0, bytes: b"", str: "", list: [], tuple: (), set: set(), dict: {}}

3.8 `func_cmd` - Handle function with argparse, typing-hint is nessessary.

3.9 `file_import` - Import function from file path.

3.10 `get_function` - Get the function object from entrypoint.

3.11 `to_thread` - Asynchronously run function *func* in a separate thread, same as `asyncio.to_thread` in python 3.9+.

3.12 `check_recursion` - Check if a function is recursive by inspecting its AST.

3.13 `debounce` - Debounce a function, delaying its execution until after a specified wait time.


## 4. morebuiltins.ipc

4.1 `IPCEncoder` - An abstract base class for all encoders; implementing the necessary communication protocol requires only the definition of two abstract methods. Be mindful that varying header lengths will impact the maximum packaging size.

4.4 `SocketLogHandlerEncoder` - For a practical demonstration, refer to the test code: morebuiltins/ipc.py:_test_ipc_logging.

4.5 `SocketServer` - To see an example in action, view the test code: morebuiltins/ipc.py:_test_ipc.

4.7 `find_free_port` - Finds and returns an available port number.

4.8 `is_port_free` - Checks if a port is free.


## 5. morebuiltins.request

5.1 `req` - A basic mock for requests, performant albeit simplistic.

5.2 `DomainParser` - Extracts the Second-level domain (SLD) from a provided hostname or URL.

5.3 `unparse_qsl` - Provides the inverse operation of parse_qsl, converting query string lists back into a URL-encoded string.

5.4 `update_url` - Organizes the query arguments within a URL to standardize its format.

5.6 `make_response` - Generates an HTTP response based on the provided parameters.

5.7 `custom_dns` - Custom the DNS of socket.getaddrinfo, only effect current thread.


## 6. morebuiltins.download_python

6.1 `download_python` - Download python portable interpreter from https://github.com/indygreg/python-build-standalone/releases. `python -m download_python -i` or `python -m download_python -a`(auto download the latest version matched the current platform: x86_64+install_only) or `python -m download_python --auto -k 3.11 -u`


## 7. morebuiltins.tk

7.1 `TKit` - Tkinter kit for dialog usages.


## 8. morebuiltins.emails

8.1 `SimpleEmail` - SimpleEmail Sender.


## 9. morebuiltins.cmd.log_server

9.1 `LogServer` - Log server for SocketHandler, create a socket server with asyncio.start_server. Custom formatter or rotation strategy with extra in log record.


## 10. morebuiltins.cmd.proxy_checker

10.1 `ProxyChecker` - A command line toolkit to check available proxies.


## 11. morebuiltins.cmd.ui

11.2 `handle_cli` - Command Line Interface: interactive mode

11.3 `handle_web` - Function to Web UI.

11.4 `handle_tk1` - Function to tkinter UI. (interactive mode)

11.5 `handle_tk2` - Function to tkinter UI.


## 12. morebuiltins.cmd.parse_deps

12.1 `parse_deps` - Parse dependencies of a project directory, and find circular dependencies.


## 13. morebuiltins.snippets.event

13.1 `EventTemplate` - Event template for event sourcing


## 14. morebuiltins.snippets.sql

14.1 `SqliteSQL` - Sqlite SQL generator


## 15. morebuiltins.cmd.systemd.service

15.1 `service_handler` - Generate and manage systemd service files


## 16. morebuiltins.cmd.systemd.timer

16.1 `timer_handler` - Parse arguments and manage systemd timer files.


## 17. morebuiltins.sqlite


## 18. morebuiltins.shared_memory

18.1 `PLock` - A simple process lock using shared memory, for singleton control.

18.2 `SharedBytes` - Shared Memory for Python, for python 3.8+.


## 19. morebuiltins.logs

19.1 `async_logger` - Asynchronous non-blocking QueueListener that manages logger handlers.

19.2 `AsyncQueueListener` - Asynchronous non-blocking QueueListener that manages logger handlers.

19.3 `LogHelper` - Quickly bind a logging handler to a logger, with a StreamHandler or SizedTimedRotatingFileHandler.

19.4 `RotatingFileWriter` - RotatingFileWriter class for writing to a file with rotation support.

19.5 `SizedTimedRotatingFileHandler` - TimedRotatingFileHandler with maxSize, to avoid files that are too large.

19.6 `ContextFilter` - A logging filter that injects context variables into extra of log records. ContextVar is used to manage context-specific data in a thread-safe / async-safe manner.


<!-- end -->

## cmd utils

1. download_python
   1. `python -m morebuiltins.cmd.download_python -a -k 3.11 -u`
   2. `-a` will filt with current platform(x86_64+install_only), `-k` is the keywords, `-u` will unzip the tar.gz
2. zipapps
   1. `python -m morebuiltins.zipapps -h`
   2. https://github.com/ClericPy/zipapps
3. log_server
   1. `python -m morebuiltins.cmd.log_server --log-dir=./logs`
   2. client use the `logging.handlers.SocketHandler` (support python2/3)
   3. Update settings of rotation/formatter with `extra: {"max_size": 1024**2, "formatter": logging.Formatter(fmt="%(asctime)s - %(filename)s - %(message)s")}`

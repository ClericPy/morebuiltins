# morebuiltins

I need much more built-ins.

Document: https://github.com/ClericPy/morebuiltins/blob/master/doc.md

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


## 2. morebuiltins.functools

2.1 `lru_cache_ttl` - A Least Recently Used (LRU) cache with a Time To Live (TTL) feature.


## 3. morebuiltins.ipc

3.1 `IPCEncoder` - An abstract base class for all encoders; implementing the necessary communication protocol requires only the definition of two abstract methods. Be mindful that varying header lengths will impact the maximum packaging size.

3.4 `SocketLogHandlerEncoder` - For a practical demonstration, refer to the test code: morebuiltins\ipc.py:_test_ipc_logging.

3.5 `SocketServer` - To see an example in action, view the test code: morebuiltins\ipc.py:_test_ipc.


## 4. morebuiltins.request

4.1 `req` - A basic mock for requests, performant albeit simplistic.

4.2 `DomainParser` - Extracts the Second-level domain (SLD) from a provided hostname or URL.

4.3 `unparse_qsl` - Provides the inverse operation of parse_qsl, converting query string lists back into a URL-encoded string.

4.4 `update_url` - Organizes the query arguments within a URL to standardize its format.


<!-- end -->

## On the way

1. quick logger
2. threads
3. progress_bar
4. http.server (upload)
5. function parser (signature.parameters)
6. time reaches
7. quick tkinter
8. http parser (request/response)
9. asyncio free-flying tasks

## 1. morebuiltins.utils



1.1 `ttime` - Translates a timestamp to a human-readable string formatted as %Y-%m-%d %H:%M:%S.


```python

>>> ttime(1486572818.421858323, tzone=8)
'2017-02-09 00:53:38'

Args:
    timestamp (float, optional): the timestamp float. Defaults to time.time().
    tzone (int, optional): time compensation. Defaults to int(-time.timezone / 3600).
    fmt (str, optional): strftime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

Returns:
    str: time string formatted.

```


---



1.2 `ptime` - Parses a timestring formatted as %Y-%m-%d %H:%M:%S back into a timestamp.


```python

>>> ptime("2018-03-15 01:27:56", tzone=8)
1521048476

Args:
    timestring (str, optional): string like 2018-03-15 01:27:56. Defaults to ttime().
    tzone (int, optional): time compensation. Defaults to int(-timezone / 3600).
    fmt (_type_, optional): strptime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

Returns:
    str: time string formatted.

```


---



1.3 `slice_into_pieces` - Divides a sequence into “n” segments, returning a generator that yields “n” pieces.


```python

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

```


---



1.4 `slice_by_size` - Slices a sequence into chunks of a specified “size”, returning a generator that produces tuples of chunks.


```python

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

```


---



1.5 `unique` - Removes duplicate elements from a sequence while preserving the original order efficiently.


```python

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

```


---



1.6 `retry` - A decorator that retries the decorated function up to “tries” times if the specified exceptions are raised.


```python

>>> func = lambda items: 1/items.pop(0)
>>> items = [0, 1]
>>> new_func = retry(tries=2, exceptions=(ZeroDivisionError,))(func)
>>> new_func(items)
1.0

Args:
    tries (int, optional): try n times, if n==1 means no retry. Defaults to 1.
    exceptions (Tuple[Type[BaseException]], optional): only retry the given errors. Defaults to (Exception,).
    return_exception (bool, optional): raise the last exception or return it. Defaults to False.

```


---



1.7 `guess_interval` - Analyzes a sequence of numbers and returns the median, calculating intervals only if they are greater than or equal to the specified accuracy.


```python

>>> # sorted_seq: [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
>>> # diffs: [8, 7, 10, 6, 13, 8]
>>> # median: 8
>>> seq = [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
>>> guess_interval(seq, 5)
8


```


---



1.8 `get_hash` - Generates a hash string from the given input string.


```python

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

```


---



1.9 `find_jsons` - A generator that locates valid JSON strings, supporting only dictionaries and lists.


```python

>>> list(find_jsons('string["123"]123{"a": 1}[{"a": 1, "b": [1,2,3]}]'))
['["123"]', '{"a": 1}', '[{"a": 1, "b": [1,2,3]}]']
>>> list(find_jsons('string[]{}{"a": 1}'))
['[]', '{}', '{"a": 1}']
>>> list(find_jsons('string[]|{}string{"a": 1}', return_as='index'))
[(6, 8), (9, 11), (17, 25)]
>>> list(find_jsons('xxxx[{"a": 1, "b": [1,2,3]}]xxxx', return_as='object'))
[[{'a': 1, 'b': [1, 2, 3]}]]

```


---



1.10 `code_inline` - Minifies Python source code into a single line.


```python
WARNING: This function uses exec, which can pose security risks if the input is not trusted.
WARNING: exec will not update the current local scope, such as function local variables.(Use globals() instead)

>>> code1 = code_inline('def test_code1(): return 12345')
>>> code1
'import base64,zlib;exec(zlib.decompress(base64.b85decode("c-l)zO;adIEiQ>q&QD1-)X=n2C`v6UEy`0cG%_|Z1puM238e".encode("u8"))))'
>>> exec(code1)
>>> test_code1()
12345
>>> code2 = code_inline("v=12345")
>>> code2
'import base64,zlib;exec(zlib.decompress(base64.b85decode("c-kwoH8e6dF$Dkzq5-o".encode("u8"))))'
>>> exec(code2)
>>> v
12345

Args:
    source_code (str): python original code.
    encoder (Literal['b16', 'b32', 'b64', 'b85'], optional): base64.encoder. Defaults to "b85".
Returns:
    new source code inline.

```


---



1.11 `read_size` - Converts byte counts into a human-readable string. Setting shorten=True and precision=0.99 will trim unnecessary decimal places from the tail of floating-point numbers.


```python

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



```


---



1.12 `read_time` - Converts seconds into a more readable time duration string.


```python

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

```


---



1.13 `Validator` - Validator for dataclasses.


```python
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
>>> # test true / false
>>> @dataclass
... class Status(Validator):
...     ok: bool
...
>>> [Status(True), Status("true"), Status("1"), Status("yes"), Status("on"), Status("True")]
[Status(ok=True), Status(ok=True), Status(ok=True), Status(ok=True), Status(ok=True), Status(ok=True)]
>>> [Status(False), Status("false"), Status("0"), Status("no"), Status("off"), Status("False")]
[Status(ok=False), Status(ok=False), Status(ok=False), Status(ok=False), Status(ok=False), Status(ok=False)]
>>> [Status(""), Status("a")]
[Status(ok=False), Status(ok=True)]

```


---



1.14 `stagger_sort` - Ensures that identical groups are ordered and evenly distributed, mitigating data skew. The function does not alter the original list and returns a generator.


```python

>>> items = [('a', 0), ('a', 2), ('a', 1), ('b', 0), ('b', 1)]
>>> list(stagger_sort(items, sort_key=lambda i: (i[0], i[1]), group_key=lambda i: i[0]))
[('a', 0), ('b', 0), ('a', 1), ('b', 1), ('a', 2)]
>>> items = ['a-a', 'a-b', 'b-b', 'b-c', 'b-a', 'b-d', 'c-a', 'c-a']
>>> list(stagger_sort(items, sort_key=lambda i: (i[0], i[2]), group_key=lambda i: i[0]))
['a-a', 'b-a', 'c-a', 'a-b', 'b-b', 'c-a', 'b-c', 'b-d']

```


---



1.15 `default_dict` - Initializes a dictionary with default zero values based on a subclass of TypedDict.


```python

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

```


---



1.16 `always_return_value` - Got a function always return the given value.


```python
>>> func = always_return_value(1)
>>> func(1, 2, 3)
1
>>> func(1, 2, c=3)
1

```


---



1.17 `format_error` - Extracts frame information from an exception, with an option to filter out “-packages” details by default. To shorten your exception message.


```python

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

```


---



1.18 `Trie` - Transforms a standard dictionary into a trie structure that supports prefix matching.


```python

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

```


---



1.19 `GuessExt` - Determines whether the input bytes of a file prefix indicate a compressed file format.


```python

>>> cg = GuessExt()
>>> cg.get_ext(b"PK\x05\x06zipfiledemo")
'.zip'

```


---



1.20 `xor_encode_decode` - Perform XOR encryption or decryption on the given data using a provided key.


```python

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

```


---



1.21 `is_running` - Check if the given process ID is running.


```python

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


```


---



1.22 `set_pid_file` - Sets a PID file to prevent multiple instances of a script or process from running concurrently.


```python
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

```


---



1.23 `get_paste` - This module offers a simple utility for retrieving text from the system clipboard with tkinter.


```python

Function:
    get_paste() -> Union[str, None]

Usage Note:
    While this function handles basic clipboard retrieval, for more advanced scenarios such as setting clipboard content or maintaining a persistent application interface, consider using libraries like `pyperclip` or running `Tkinter.mainloop` which keeps the GUI event loop active.
    Set clipboard with tkinter:
        _tk.clipboard_clear()
        _tk.clipboard_append(text)
        _tk.update()
        _tk.mainloop() # this is needed

```


---



1.24 `set_clip` - Copies the given text to the clipboard using a temporary file in a Windows environment.


```python

This function writes the provided text into a temporary file and then uses the `clip.exe` command-line utility
to read from this file and copy its content into the clipboard.

Parameters:
text: str - The text content to be copied to the clipboard.

```


---



1.25 `switch_flush_print` - Set builtins.print default flush=True.


```python

>>> print.__name__
'print'
>>> switch_flush_print()
>>> print.__name__
'flush_print'
>>> switch_flush_print()
>>> print.__name__
'print'

```


---



1.26 `unix_rlimit` - Unix only. RLIMIT_RSS, RLIMIT_FSIZE to limit the max_memory and max_file_size





---



1.27 `SimpleFilter` - Simple dup-filter with pickle file.


```python

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

```


---



1.28 `FileDict` - A dict that can be saved to a file.


```python

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

```


---



1.29 `PathLock` - A Lock/asyncio.Lock of a path, and the child-path lock will block the parent-path.


```python
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

```


---



1.30 `i2b` - Convert an int to bytes of a specified length, commonly used in TCP communication.


```python

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

```


---



1.31 `b2i` - Convert a byte sequence to an integer.


```python

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

```


---



1.32 `get_hash_int` - Generates a int hash(like docid) from the given input bytes.


```python

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

```


---



1.33 `iter_weights` - Generates an element sequence based on weights.


```python

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

```


---



1.34 `get_size` - Recursively get size of objects.


```python

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

```


---



1.35 `base_encode` - Encode a number to a base-N string.


```python

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

```


---



1.36 `base_decode` - Decode a base-N string to a number.


```python

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

```


---



1.37 `gen_id` - Generate a readable & unique ID based on the current time(ms) and random bytes.


```python
The performance is about 400000 IDs per second.

If rand_len=0 the length of ID will be 18, rand_len=4 the length of ID will be 22.
ID format:
- {YYMMDD_HHMMSS}_{4-digit base62 of microsecond}{rand_len urandom hex string}

The following table shows the relationship between rand_len and the safe range of unique IDs per microsecond:

rand_len |  urandom_size  |    Safe Range
---------------------------------------------
     2    |         1      |       10
     4    |         2      |      100
     6    |         3      |     1000
     8    |         4      |    10000
    10    |         5      |   100000
    12    |         6      |  1000000
    14    |         7      | 10000000

Seems like rand_len -> 10 ** (rand_len // 2) safe range.

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
>>> [len(gen_id(i)[:]) for i in range(10)]
[18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
>>> # gen_id() => 251111_235204_0xYfc4f8
>>> # gen_id(sep="") => 2511112352291nTq972c

```


---



1.38 `timeti` - Return the number of iterations per second for a given statement.


```python

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

```


---



1.40 `cut_file` - Cut file to max_bytes, remain_ratio is the ratio of the end part to remain, ensure_line_start is to ensure the first line is a complete line


```python

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

```


---


## 2. morebuiltins.date



2.1 `ScheduleTimer` - The ScheduleTimer class facilitates the creation and evaluation of datetime patterns for scheduling purposes.


```python

It includes mechanisms to parse patterns involving logical operations (AND, OR) and comparison checks (equality, inequality, arithmetic, and custom range checks).

Comparison Operators:

    Equality (= or ==): Tests for equality between datetime parts.
        Example: "hour=12" checks if it's exactly 12 o'clock.
    Inequality (!=): Ensures inequality between datetime parts.
        Example: "minute!=30" for minutes not being 30.
    Less Than (<): Requires the left datetime part to be less than the right.
        Example: "day<15" for days before the 15th.
    Less Than or Equal (<=): Allows the left datetime part to be less or equal to the right.
        Example: "month<=6" covers January to June.
    Greater Than (>): Ensures the left datetime part is greater than the right.
        Example: "hour>18" for evenings after 6 PM.
    Greater Than or Equal (>=): Allows the left datetime part to be greater or equal to the right.
        Example: "weekday>=MO" starts from Monday.
    Division Modulo (/): Divisibility check for the left digit by the right digit in datetime format.
        Example: "minute/15" checks for quarter hours.
    Range Inclusion (@): Confirms if the left time falls within any defined ranges in the right string.
        Example: "hour@9-11,13-15" for office hours.

Logical Operators:
    AND (&): Both conditions must hold true.
        Example: "hour=12&minute=30" for exactly 12:30 PM.
    OR (; or |): At least one of the conditions must be true.
        Example: "hour=12;hour=18" for noon or 6 PM.
    Negation (!): Inverts the truth of the following condition. At the start of the pattern, the condition is negated.
        Example: "!hour=12" excludes noon.

Demo:

>>> start_date = datetime.strptime("2023-02-01 00:00:00", "%Y-%m-%d %H:%M:%S")
>>> list(ScheduleTimer.iter_datetimes("%M=05|%M=15", count=3, start_date=start_date, callback=str))
['2023-02-01 00:05:00', '2023-02-01 00:15:00', '2023-02-01 01:05:00']
>>> list(ScheduleTimer.iter_datetimes("%H:%M=15:30", count=3, start_date=start_date, callback=str))
['2023-02-01 15:30:00', '2023-02-02 15:30:00', '2023-02-03 15:30:00']
>>> list(ScheduleTimer.iter_datetimes("%H:%M=15:30&%d=15", count=3, start_date=start_date, callback=str))
['2023-02-15 15:30:00', '2023-03-15 15:30:00', '2023-04-15 15:30:00']
>>> list(ScheduleTimer.iter_datetimes("%H:%M=15:30&%d>=15", count=3, start_date=start_date, callback=str))
['2023-02-15 15:30:00', '2023-02-16 15:30:00', '2023-02-17 15:30:00']
>>> list(ScheduleTimer.iter_datetimes("%M@15-16&%d>=15", count=3, start_date=start_date, callback=str))
['2023-02-15 00:15:00', '2023-02-15 00:16:00', '2023-02-15 01:15:00']
>>> list(ScheduleTimer.iter_datetimes("%M/15", count=5, start_date=start_date, callback=str))
['2023-02-01 00:00:00', '2023-02-01 00:15:00', '2023-02-01 00:30:00', '2023-02-01 00:45:00', '2023-02-01 01:00:00']

```


---



2.2 `Crontab` - Crontab python parser.


```python

Demo:

>>> start_date = datetime.strptime("2023-02-01 00:00:00", "%Y-%m-%d %H:%M:%S")
>>> list(Crontab.iter_datetimes("*/15 * * * *", count=10, start_date=start_date, callback=lambda i: i.strftime("%M")))
['00', '15', '30', '45', '00', '15', '30', '45', '00', '15']
>>> list(Crontab.iter_datetimes("* * * * 2,4,6", count=10, start_date=start_date, callback=lambda i: i.strftime("%a")))
['Thu', 'Sat', 'Tue', 'Thu', 'Sat', 'Tue', 'Thu', 'Sat', 'Tue', 'Thu']
>>> list(Crontab.iter_datetimes("0 0 11-19/4,8,30 * *", count=10, start_date=start_date, callback=lambda i: i.strftime("%m-%d")))
['02-08', '02-11', '02-15', '02-19', '03-08', '03-11', '03-15', '03-19', '03-30', '04-08']
>>> list(Crontab.iter_datetimes("* * * * *", count=1, start_date=start_date))
[datetime.datetime(2023, 2, 1, 0, 0)]
>>> list(Crontab.iter_datetimes("5 4-5,6-9/2 5,6 * 3,5", count=3, start_date=start_date, callback=str))
['2023-04-05 04:05:00', '2023-04-05 05:05:00', '2023-04-05 06:05:00']
>>> list(Crontab.iter_datetimes("0 0 1 8,9,10 *", count=3, start_date=start_date, callback=str, yield_tries=True))
[(7, '2023-08-01 00:00:00'), (120, '2023-09-01 00:00:00'), (232, '2023-10-01 00:00:00')]
>>> list(Crontab.iter_datetimes("0 0 1 11 *", count=3, start_date=start_date, callback=str, yield_tries=True))
[(10, '2023-11-01 00:00:00'), (133, '2024-11-01 00:00:00'), (256, '2025-11-01 00:00:00')]

```


---


## 3. morebuiltins.funcs



3.1 `lru_cache_ttl` - A Least Recently Used (LRU) cache with a Time To Live (TTL) feature.


```python

Args:
    maxsize (int): maxsize of cache
    ttl (Optional[Union[int, float]], optional): time to live. Defaults to None.
    controls (bool, optional): set cache/ttl_clean attributes. Defaults to False.
    auto_clear (bool, optional): clear dead cache automatically. Defaults to True.
    timer (callable, optional): Defaults to time.time.

Returns:
    callable: decorator function

>>> import time
>>> # test ttl
>>> values = [1, 2]
>>> @lru_cache_ttl(1, 0.1)
... def func1(i):
...     return values.pop(0)
>>> [func1(1), func1(1), time.sleep(0.11), func1(1)]
[1, 1, None, 2]
>>> # test maxsize
>>> values = [1, 2, 3]
>>> func = lambda i: values.pop(0)
>>> func1 = lru_cache_ttl(2)(func)
>>> [func1(i) for i in [1, 1, 1, 2, 2, 2, 3, 3, 3]]
[1, 1, 1, 2, 2, 2, 3, 3, 3]
>>> # test auto_clear=True, with controls
>>> values = [1, 2, 3, 4]
>>> func = lambda i: values.pop(0)
>>> func1 = lru_cache_ttl(5, 0.1, controls=True, auto_clear=True)(func)
>>> [func1(1), func1(2), func1(3)]
[1, 2, 3]
>>> time.sleep(0.11)
>>> func1(3)
4
>>> len(func1.cache)
1
>>> # test auto_clear=False
>>> values = [1, 2, 3, 4]
>>> @lru_cache_ttl(5, 0.1, controls=True, auto_clear=False)
... def func1(i):
...     return values.pop(0)
>>> [func1(1), func1(2), func1(3)]
[1, 2, 3]
>>> time.sleep(0.11)
>>> func1(3)
4
>>> len(func1.cache)
3

```


---



3.2 `threads` - Quickly convert synchronous functions to be concurrent. (similar to madisonmay/Tomorrow)


```python

>>> @threads(10)
... def test(i):
...     time.sleep(i)
...     return i
>>> start = time.time()
>>> tasks = [test(i) for i in [0.1] * 5]
>>> len(test.pool._threads)
5
>>> len(test.tasks)
5
>>> for i in tasks:
...     i.result() if hasattr(i, 'result') else i
0.1
0.1
0.1
0.1
0.1
>>> time.time() - start < 0.2
True
>>> len(test.pool._threads)
5
>>> len(test.tasks)
0
>>> test.pool.shutdown()  # optional

```


---



3.3 `bg_task` - Avoid asyncio free-flying tasks, better to use the new asyncio.TaskGroup to avoid this in 3.11+. https://github.com/python/cpython/issues/91887


```python

Args:
    coro (Coroutine)

Returns:
    _type_: Task


```


---



3.4 `NamedLock` - Reusable named locks, support for timeouts, support for multiple concurrent locks.


```python

Demo::

    def test_named_lock():
        def test_sync():
            import time
            from concurrent.futures import ThreadPoolExecutor
            from threading import Lock, Semaphore

            def _test1():
                with NamedLock("_test1", Lock, timeout=0.05) as lock:
                    time.sleep(0.2)
                    return bool(lock)

            with ThreadPoolExecutor(10) as pool:
                tasks = [pool.submit(_test1) for _ in range(3)]
                result = [i.result() for i in tasks]
                assert result == [True, False, False], result
            assert len(NamedLock._SYNC_CACHE) == 1
            NamedLock.clear_unlocked()
            assert len(NamedLock._SYNC_CACHE) == 0

            def _test2():
                with NamedLock("_test2", lambda: Semaphore(2), timeout=0.05) as lock:
                    time.sleep(0.2)
                    return bool(lock)

            with ThreadPoolExecutor(10) as pool:
                tasks = [pool.submit(_test2) for _ in range(3)]
                result = [i.result() for i in tasks]
                assert result == [True, True, False], result

        def test_async():
            import asyncio

            async def main():
                async def _test1():
                    async with NamedLock("_test1", asyncio.Lock, timeout=0.05) as lock:
                        await asyncio.sleep(0.2)
                        return bool(lock)

                tasks = [asyncio.create_task(_test1()) for _ in range(3)]
                result = [await i for i in tasks]
                assert result == [True, False, False], result
                assert len(NamedLock._ASYNC_CACHE) == 1
                NamedLock.clear_unlocked()
                assert len(NamedLock._ASYNC_CACHE) == 0

                async def _test2():
                    async with NamedLock(
                        "_test2", lambda: asyncio.Semaphore(2), timeout=0.05
                    ) as lock:
                        await asyncio.sleep(0.2)
                        return bool(lock)

                tasks = [asyncio.create_task(_test2()) for _ in range(3)]
                result = [await i for i in tasks]
                assert result == [True, True, False], result

            asyncio.run(main())

        test_sync()
        test_async()

```


---



3.5 `FuncSchema` - Parse the parameters and types required by a function into a dictionary, and convert an incoming parameter into the appropriate type.


```python

>>> def test(a, b: str, /, c=1, *, d=["d"], e=0.1, f={"f"}, g=(1, 2), h=True, i={1}, **kws):
...     return
>>> FuncSchema.parse(test, strict=False)
{'a': {'type': <class 'str'>, 'default': <class 'inspect._empty'>}, 'b': {'type': <class 'str'>, 'default': <class 'inspect._empty'>}, 'c': {'type': <class 'int'>, 'default': 1}, 'd': {'type': <class 'list'>, 'default': ['d']}, 'e': {'type': <class 'float'>, 'default': 0.1}, 'f': {'type': <class 'set'>, 'default': {'f'}}, 'g': {'type': <class 'tuple'>, 'default': (1, 2)}, 'h': {'type': <class 'bool'>, 'default': True}, 'i': {'type': <class 'set'>, 'default': {1}}, 'kws': {'type': <class 'str'>, 'default': <class 'inspect._empty'>}}
>>> def test(a):
...     return
>>> try:FuncSchema.parse(test, strict=True)
... except TypeError as e: e
TypeError('Parameter `a` has no type and no default value.')
>>> def test(b: str):
...     return
>>> FuncSchema.parse(test, strict=True)
{'b': {'type': <class 'str'>, 'default': <class 'inspect._empty'>}}
>>> FuncSchema.parse(test, strict=True, fill_default=True)
{'b': {'type': <class 'str'>, 'default': ''}}
>>> def test(**kws):
...     return
>>> try:FuncSchema.parse(test, strict=True)
... except TypeError as e: e
TypeError('Parameter `kws` has no type and no default value.')
>>> def test(*args):
...     return
>>> try:FuncSchema.parse(test, strict=True)
... except TypeError as e: e
TypeError('Parameter `args` has no type and no default value.')
>>> FuncSchema.convert("1", int)
1
>>> FuncSchema.convert("1", str)
'1'
>>> FuncSchema.convert("1", float)
1.0
>>> FuncSchema.convert(0, bool)
False
>>> FuncSchema.convert('1', bool)
True
>>> FuncSchema.convert('[[1, 1]]', dict)
{1: 1}
>>> FuncSchema.convert('{"1": "1"}', dict)
{'1': '1'}
>>> FuncSchema.convert('[1, 1]', set)
{1}
>>> FuncSchema.convert('[1, 1]', tuple)
(1, 1)
>>> FuncSchema.convert('[1, "1"]', list)
[1, '1']
>>> FuncSchema.to_string(1)
'1'
>>> FuncSchema.to_string("1")
'1'
>>> FuncSchema.to_string(1.0, float)
'1.0'
>>> FuncSchema.to_string(False)
'false'
>>> FuncSchema.to_string(True)
'true'
>>> FuncSchema.to_string({1: 1})
'{"1": 1}'
>>> FuncSchema.to_string({'1': '1'})
'{"1": "1"}'
>>> FuncSchema.to_string({1})
'[1]'
>>> FuncSchema.to_string((1, 1))
'[1, 1]'
>>> FuncSchema.to_string([1, '1'])
'[1, "1"]'

```


---



3.6 `InlinePB` - Inline progress bar.


```python

Demo::

    with InlinePB(100) as pb:
        for i in range(100):
            pb.add(1)
            time.sleep(0.03)
    # Progress:  41 / 100  41% [||||||         ] |   33 units/s
    with InlinePB(100) as pb:
        for i in range(1, 101):
            pb.update(i)
            time.sleep(0.03)
    # Progress:  45 / 100  45% [||||||         ] |   33 units/s


```


---



3.7 `get_type_default` - Get the default value for a type. {int: 0, float: 0.0, bytes: b"", str: "", list: [], tuple: (), set: set(), dict: {}}





---



3.8 `func_cmd` - Handle function with argparse, typing-hint is nessessary.


```python

Demo::

    def test(str: str, /, int=1, *, list=["d"], float=0.1, set={"f"}, tuple=(1, 2), bool=True, dict={"k": 1}):
        """Test demo function.

        Args:
            str (str): str.
            int (int, optional): int. Defaults to 1.
            list (list, optional): list. Defaults to ["d"].
            float (float, optional): float. Defaults to 0.1.
            set (dict, optional): set. Defaults to {"f"}.
            tuple (tuple, optional): tuple. Defaults to (1, 2).
            bool (bool, optional): bool. Defaults to True.
            dict (dict, optional): dict. Defaults to {"k": 1}.
        """
        print(locals())

    # raise ValueError if auto_default is False and user do not input nessessary args.
    func_cmd(test, auto_default=False)

    CMD args:

    > python app.py
    ValueError: `str` has no default value.

    > python app.py --str 1 --int 2 --float 1.0 --list "[1,"a"]" --tuple "[2,"b"]" --set "[1,1,2]" --dict "{"k":"v"}"
    {'str': '1', 'int': 2, 'list': [1, 'a'], 'float': 1.0, 'set': {1, 2}, 'tuple': (2, 'b'), 'bool': True, 'dict': {'k': 'v'}}

    > python app.py -s 1 -i 2 -f 1.0 -l "[1,"a"]" -t "[2,"b"]" -s "[1,1,2]" -d "{"k":"v"}"
    {'str': '[1,1,2]', 'int': 2, 'list': [1, 'a'], 'float': 1.0, 'set': {'f'}, 'tuple': (2, 'b'), 'bool': True, 'dict': {'k': 'v'}}

    > python app.py -h
    usage: Test demo function.

        Args:
            str (str): str.
            int (int, optional): int. Defaults to 1.
            list (list, optional): list. Defaults to ["d"].
            float (float, optional): float. Defaults to 0.1.
            set (dict, optional): set. Defaults to {"f"}.
            tuple (tuple, optional): tuple. Defaults to (1, 2).
            bool (bool, optional): bool. Defaults to True.
            dict (dict, optional): dict. Defaults to {"k": 1}.


    options:
    -h, --help            show this help message and exit
    -s STR, --str STR     {'type': <class 'str'>, 'default': <class 'inspect._empty'>}
    -i INT, --int INT     {'type': <class 'int'>, 'default': 1}
    -l LIST, --list LIST  {'type': <class 'list'>, 'default': ['d']}
    -f FLOAT, --float FLOAT
                            {'type': <class 'float'>, 'default': 0.1}
    -se SET, --set SET    {'type': <class 'set'>, 'default': {'f'}}
    -t TUPLE, --tuple TUPLE
                            {'type': <class 'tuple'>, 'default': (1, 2)}
    -b BOOL, --bool BOOL  {'type': <class 'bool'>, 'default': True}
    -d DICT, --dict DICT  {'type': <class 'dict'>, 'default': {'k': 1}}

```


---



3.9 `file_import` - Import function from file path.


```python

Demo::
    >>> from pathlib import Path
    >>> file_path = Path(__file__).parent / "utils.py"
    >>> list(file_import(file_path, ["get_hash", "find_jsons"]).keys())
    ['get_hash', 'find_jsons']

```


---



3.10 `get_function` - Get the function object from entrypoint.


```python

Demo::

    >>> get_function("urllib.parse:urlparse").__name__
    'urlparse'

```


---



3.11 `to_thread` - Asynchronously run function *func* in a separate thread, same as `asyncio.to_thread` in python 3.9+.





---



3.12 `check_recursion` - Check if a function is recursive by inspecting its AST.


```python
Returns True if the function calls itself, otherwise False.

Demo::
    >>> def recursive_func():
    ...     return recursive_func()
    >>> check_recursion(recursive_func)
    True
    >>> def non_recursive_func():
    ...     return 1 + 1
    >>> check_recursion(non_recursive_func)
    False
    >>> # print is a std-lib function
    >>> check_recursion(print, return_error=False)
    >>> type(check_recursion(print, return_error=True))
    <class 'TypeError'>

```


---



3.13 `debounce` - Debounce a function, delaying its execution until after a specified wait time.


```python

Args:
    wait (float): The amount of time to wait before executing the function.

Demo::

    >>> @debounce(0.1)
    ... def test():
    ...     print("Function executed")
    >>> test()
    Function executed
    >>> test()
    >>> time.sleep(0.1)
    >>> test()
    Function executed
    >>> test()

```


---



3.14 `async_call` - Automatically call a function asynchronously, whether it's sync or async.


```python

If func is a coroutine function, await it directly.
If func is a regular function, run it in a thread pool executor.

Args:
    func: A callable (sync function or coroutine function)
    *args: Positional arguments for func
    **kwargs: Keyword arguments for func

Returns:
    The result of calling func with the given arguments

Example::
    import asyncio
    import time

    def sync_add(a, b):
        time.sleep(1)
        return a + b

    async def async_mul(a, b):
        await asyncio.sleep(1)
        return a * b

    async def main():
        # Call sync function
        result1 = await async_call(sync_add, 2, 3)
        assert result1 == 5

        # Call async function
        result2 = await async_call(async_mul, 4, 5)
        assert result2 == 20

    asyncio.run(main())

```


---


## 4. morebuiltins.ipc



4.1 `IPCEncoder` - An abstract base class for all encoders; implementing the necessary communication protocol requires only the definition of two abstract methods. Be mindful that varying header lengths will impact the maximum packaging size.





---



4.4 `SocketLogHandlerEncoder` - For a practical demonstration, refer to the test code: morebuiltins/ipc.py:_test_ipc_logging.


```python

Demo::

    async def _test_ipc_logging():
        import logging

        host = "127.0.0.1"
        port = 8090
        async with SocketServer(host=host, port=port, encoder=SocketLogHandlerEncoder()):
            # socket logger demo
            # ==================
            logger = logging.getLogger("test_logger")
            logger.setLevel(logging.DEBUG)
            h = SocketHandler(host, port)
            h.setLevel(logging.DEBUG)
            logger.addHandler(h)
            logger.info("test socket")
            # ==================

            # ensure test case
            await asyncio.sleep(0.1)
            assert pickle.loads(h.sock.recv(100000)[4:])["name"] == logger.name
And provide a simple implementation for generating logs for coroutine code with Client usage.

```


---



4.5 `SocketServer` - To see an example in action, view the test code: morebuiltins/ipc.py:_test_ipc.


```python

Demo::

    async def test_client(host="127.0.0.1", port=8090, encoder=None, cases=None):
        async with SocketClient(host=host, port=port, encoder=encoder) as c:
            for case in cases:
                await c.send(case)
                response = await c.recv()
                if globals().get("print_log"):
                    print("[Client]", "send:", repr(case), "=>", "recv:", repr(response))
                assert case == response or str(case) == response, [case, response]
            await c.send("[shutdown server]")


    async def _test_ipc():
        import platform

        JSONEncoder._DUMP_KWARGS["default"] = str
        for enc, cases in [
            [PickleEncoder, [123, "123", None, {"a"}, ["a"], ("a",), {"a": 1}]],
            [JSONEncoder, [123, "123", None, {"a"}, ["a"], {"a": 1}]],
        ]:
            encoder = enc()
            if platform.system() == "Linux":
                # test unix domain socket
                print("Test Linux Unix Domain Socket")
                host = "/tmp/uds.sock"
                port = None
                async with SocketServer(host=host, port=port, encoder=encoder):
                    await test_client(host, port=None, encoder=encoder, cases=cases)

            # test socket
            host = "127.0.0.1"
            port = 8090
            async with SocketServer(host=host, port=port, encoder=encoder):
                await test_client(host="127.0.0.1", port=8090, encoder=encoder, cases=cases)

```


---



4.7 `find_free_port` - Finds and returns an available port number.


```python

Parameters:
- host: The host address to bind, default is "127.0.0.1".
- port: The port number to attempt binding, default is 0 (for OS allocation).

Returns:
- If a free port is found, it returns the port number; otherwise, returns None.

Demo:

>>> free_port = find_free_port()
>>> isinstance(free_port, int)
True

```


---



4.8 `is_port_free` - Checks if a port is free.


```python

Demo:

>>> is_port_free(12345)
True

```


---


## 5. morebuiltins.request



5.1 `req` - A basic mock for requests, performant albeit simplistic.


```python

>>> import time
>>> r = req.get("https://postman-echo.com/get?a=2", headers={"User-Agent": "Chrome"}, timeout=3, params={"b": "3"})
>>> r.url
'https://postman-echo.com/get?a=2&b=3'
>>> r.ok
True
>>> r.status_code
200
>>> r.text.startswith('{')
True
>>> r = req.post("https://postman-echo.com/post?a=2", headers={"User-Agent": "Chrome"}, timeout=3, params={"b": "3"}, data=b"mock data")
>>> r.json()["data"]
'mock data'
>>> r.json()["args"]
{'a': '2', 'b': '3'}
>>> r = req.post("https://postman-echo.com/post?a=2", headers={"User-Agent": "Chrome"}, timeout=3, json={"data": "yes json"})
>>> r.json()["json"]
{'data': 'yes json'}

```


---



5.2 `DomainParser` - Extracts the Second-level domain (SLD) from a provided hostname or URL.


```python

>>> domain_parser = DomainParser()
>>> domain_parser.parse_hostname("github.com")
'github.com'
>>> domain_parser.parse_hostname("www.github.com")
'github.com'
>>> domain_parser.parse_hostname("www.api.github.com.cn")
'github.com.cn'
>>> domain_parser.parse_hostname("a.b.c.kawasaki.jp")
'c.kawasaki.jp'
>>> domain_parser.parse_hostname("a.b.c.city.kawasaki.jp")
'c.city.kawasaki.jp'
>>> domain_parser.parse_hostname("a.bbbbbb.cccccc")
''
>>> domain_parser.parse_hostname("a.bbbbbb.cccccc", default="b.c")
'b.c'
>>> domain_parser.parse_url("https://github.com/ClericPy/morebuiltins")
'github.com'


```


---



5.3 `unparse_qsl` - Provides the inverse operation of parse_qsl, converting query string lists back into a URL-encoded string.





---



5.4 `update_url` - Organizes the query arguments within a URL to standardize its format.


```python

>>> update_url('http://www.github.com?b=1&c=1&a=1', {"b": None, "c": None})  # remove params
'http://www.github.com?a=1'
>>> update_url("http://www.github.com?b=1&c=1&a=1", a="123", b=None)  # update params with kwargs
'http://www.github.com?c=1&a=123'
>>> update_url('http://www.github.com?b=1&c=1&a=1', sort=True)  # sort params
'http://www.github.com?a=1&b=1&c=1'
>>> update_url("http://www.github.com?b=1&c=1&a=1", {"a": "999"})  # update params
'http://www.github.com?b=1&c=1&a=999'
>>> update_url("http://www.github.com?b=1&c=1&a=1", replace_kwargs={"netloc": "www.new_host.com"})  # update netloc
'http://www.new_host.com?b=1&c=1&a=1'

replace_kwargs is a dict to update attributes before sorting  (such as scheme / netloc...).

```


---



5.6 `make_response` - Generates an HTTP response based on the provided parameters.


```python

:param body: The response body which can be a string, bytes, list, or dictionary. Default is an empty string.
:param status: The HTTP status code. Default is 200.
:param protocol: The HTTP protocol version. Default is "HTTP/1.1".
:param headers: A dictionary of HTTP response headers. Default is None.
:param encoding: The encoding to use. Default is "utf-8".
:return: A byte sequence representing the constructed HTTP response.

```


---



5.7 `custom_dns` - Custom the DNS of socket.getaddrinfo, only effect current thread.


```python

[WARNING] This will modify the global socket.getaddrinfo.

>>> from concurrent.futures import ThreadPoolExecutor
>>> # this only effect current thread
>>> custom_dns({"1.1.1.1": ("127.0.0.1", 80), ("1.1.1.1", 80): ("192.168.0.1", 443)})
>>> socket.getaddrinfo('1.1.1.1', 80)[0][-1]
('192.168.0.1', 443)
>>> socket.getaddrinfo('1.1.1.1', 8888)[0][-1]
('127.0.0.1', 80)
>>> ThreadPoolExecutor().submit(lambda : socket.getaddrinfo('1.1.1.1', 8888)[0][-1]).result()
('1.1.1.1', 8888)
>>> # this effect global socket.getaddrinfo
>>> custom_dns({"1.1.1.1": ("127.0.0.1", 80), ("1.1.1.1", 80): ("192.168.0.1", 443)}, thread=False)
>>> ThreadPoolExecutor().submit(lambda : socket.getaddrinfo('1.1.1.1', 8888)[0][-1]).result()
('127.0.0.1', 80)

Demo:

    custom_dns(custom={("MY_PROXY_HOST", 80): ("xxxxxxxxx", 43532)})
    print(
        requests.get(
            "https://www.github.com/", proxies={"all": "http://MY_PROXY_HOST"}
        ).text
    )

```


---


## 6. morebuiltins.download_python



6.1 `download_python` - Download python portable interpreter from https://github.com/indygreg/python-build-standalone/releases. `python -m download_python -i` or `python -m download_python -a`(auto download the latest version matched the current platform: x86_64+install_only) or `python -m download_python --auto -k 3.11 -u`


```python

λ python -m download_python -i
[10:56:17] Checking https://api.github.com/repos/indygreg/python-build-standalone/releases/latest
[10:56:19] View the rules:
https://gregoryszorc.com/docs/python-build-standalone/main/running.html#obtaining-distributions

[10:56:19] Got 290 urls from github.

[290] Enter keywords (can be int index or partial match, defaults to 0):
0. windows
1. linux
2. darwin
0
[10:56:24] Filt with keyword: "windows". 290 => 40

[40] Enter keywords (can be int index or partial match, defaults to 0):
0. 3.12.3
1. 3.11.9
2. 3.10.14
3. 3.9.19
4. 3.8.19

[10:56:25] Filt with keyword: "3.12.3". 40 => 8

[8] Enter keywords (can be int index or partial match, defaults to 0):
0. x86_64
1. i686

[10:56:28] Filt with keyword: "x86_64". 8 => 4

[4] Enter keywords (can be int index or partial match, defaults to 0):
0. shared-pgo-full.tar.zst
1. shared-install_only.tar.gz
2. pgo-full.tar.zst
3. install_only.tar.gz
3
[10:56:33] Filt with keyword: "install_only.tar.gz". 4 => 1
[10:56:33] Download URL: 39.1 MB
https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-pc-windows-msvc-install_only.tar.gz
File path to save(defaults to `./cpython-3.12.3+20240415-x86_64-pc-windows-msvc-install_only.tar.gz`)?
or `q` to exit.

[10:56:38] Start downloading...
https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-pc-windows-msvc-install_only.tar.gz
D:\github\morebuiltins\morebuiltins\download_python\cpython-3.12.3+20240415-x86_64-pc-windows-msvc-install_only.tar.gz
[10:56:44] Downloading: 39.12 / 39.12 MB | 100.00% | 11.3 MB/s | 0s
[10:56:44] Download complete.
```


---


## 7. morebuiltins.tk



7.1 `TKit` - Tkinter kit for dialog usages.


```python
Demo::

    def examples():
        while True:
            TKit.ask(0, "0")
            TKit.ask(1, "1")
            TKit.ask(2, "2")
            if TKit.ask(True, "Choose NO", default="no") is True:
                TKit.ask(0, "Wrong choice")
                continue
            if (
                TKit.ask((["1"], ["2", "3"], "4", ["5", "6", "7"]), message="Choose 3:")
                != "3"
            ):
                TKit.ask(1, "Wrong choice")
                continue
            if TKit.ask(
                [["1"], ["2", "3"], "4", ["5", "6", "7"]],
                message="Choose 3 and 6:",
                width=400,
            ) != ["3", "6"]:
                TKit.ask(2, "Wrong choice")
                continue
            result = TKit.ask("Input text 1 (press Enter to submit):")

            if result != "1":
                TKit.ask(2, "Wrong text %s" % repr(result))
                continue
            result = TKit.ask("Input text 1\\n (press Ctrl-Enter to submit):", textarea=1)

            if result != "1\n":
                TKit.ask(2, "Wrong text %s" % repr(result))
                continue

            def test_text(flush=False):
                import time

                for i in range(50):
                    print(f"Test print flush={flush} -- {i}", flush=flush)
                    time.sleep(0.02)
                return "OK"

            with TKit.text_context(
                test_text,
                flush=True,
                __resize_kwargs={"title": "The Title", "toolwindow": True},
                __text_kwargs={"font": "_ 15"},
            ) as result:
                TKit.info("result=%s" % result)

            with TKit.text_context(
                test_text,
                flush=False,
                __resize_kwargs={"title": "The Title", "toolwindow": True},
                __text_kwargs={"font": "_ 15"},
            ) as result:
                TKit.warn("result=%s" % result)
            break

    examples()

```


---


## 8. morebuiltins.emails



8.1 `SimpleEmail` - SimpleEmail Sender.


```python

Demo::

    with SimpleEmail("smtp.gmail.com", 465, "someone@gmail.com", "PASSWORD") as s:
    print(
        s.send_message(
            "This is Title",
            "This is body text or file path(.md/.txt)",
            "Author<someone@gmail.com>",
            "anybody@gmail.com",
            files="a.py,b.py,c.txt",
            filename="files.zip",
            encoding="u8",
        )
    )

```


---


## 9. morebuiltins.cmd.log_server



9.1 `LogServer` - Log server for SocketHandler, create a socket server with asyncio.start_server. Custom formatter or rotation strategy with extra in log record.


```python

[WARNING]: Ensure your log msg is "" if you only want to update settings, or the msg will be skipped.

logger.info("", extra={"log_setting": {"formatter": formatter, "max_size": 1024**2, "level_specs": [logging.ERROR]}})


### Server demo1:
    start log server in terminal, only collect logs and print to console
    > python -m morebuiltins.cmd.log_server

### Server demo2:
    custom options to log to "logs" directory, default rotates at 10MB with 5 backups, no log_stream, enable compress
    > python -m morebuiltins.cmd.log_server --log-dir=./logs --host 127.0.0.1 --port 8901 --log-stream=None --compress

### Server demo3:
    python code

```python
# Server side
import asyncio

from morebuiltins.cmd.log_server import LogServer


async def main():
    async with LogServer() as ls:
        await ls.wait_closed()


asyncio.run(main())
```

### Client demo1:

```python
# Client side(no dependency on morebuiltins)
import logging
import logging.handlers

logger = logging.getLogger("client")
logger.setLevel(logging.DEBUG)
h = logging.handlers.SocketHandler("127.0.0.1", 8901)
h.setLevel(logging.DEBUG)
logger.addHandler(h)
logger.info(
    "",
    extra={
        "log_setting": {
            "max_size": 1024**2,
            "formatter": logging.Formatter(
                fmt="%(asctime)s - %(filename)s - %(message)s"
            ),
            "level_specs": [logging.ERROR],
        }
    },
)
for _ in range(5):
    logger.info("hello world!")

# [client] 2024-08-10 19:30:07,113 - temp3.py - hello world!
# [client] 2024-08-10 19:30:07,113 - temp3.py - hello world!
# [client] 2024-08-10 19:30:07,113 - temp3.py - hello world!
# [client] 2024-08-10 19:30:07,113 - temp3.py - hello world!
# [client] 2024-08-10 19:30:07,114 - temp3.py - hello world!
```

### Client demo2:

```python
from morebuiltins.cmd.log_server import get_logger

logger = get_logger("dir/test.log")
# logger = get_logger("dir/test.log", host="localhost", port=8901)
logger.debug("debug")
logger.info("info")
logger.warning("warning")
# 2025-10-11 01:30:35,151 | DEBUG | log_server.py:416 - Set formatter for logger 'dir/test.log': %(asctime)s | %(levelname)-5s | %(filename)+8s:%(lineno)+3s - %(message)s
# 2025-10-11 01:30:35,151 | DEBUG | temp.py:  4 - debug
# 2025-10-11 01:30:35,152 | INFO  | temp.py:  5 - info
# 2025-10-11 01:30:35,152 | WARN  | temp.py:  6 - warning
```

More docs:
    > python -m morebuiltins.cmd.log_server -h

```


---


## 10. morebuiltins.cmd.proxy_checker



10.1 `ProxyChecker` - A command line toolkit to check available proxies.


```python

1. clipboard usage:
    > input-text from clipboard, and set the result to clipboard. `-l 2` means try 2 loops
        > python -m morebuiltins.cmd.proxy_checker -c -l 2

2. input-file output-file usage:
    > input-text from file, and set the result to output-file
        > python -m morebuiltins.cmd.proxy_checker -i input.txt -o output.txt
    > output to stdout
        > python -m morebuiltins.cmd.proxy_checker -i input.txt

3. stdin usage:
    > cat file.txt | python -m morebuiltins.cmd.proxy_checker > stdout.txt
    > cat file.txt | python -m morebuiltins.cmd.proxy_checker -o output.txt

python -m morebuiltins.cmd.proxy_checker -h
    options:
    -h, --help            show this help message and exit
    -t TIMEOUT, --timeout TIMEOUT
                            timeout of each request
    -l LOOP, --loop LOOP  Loop the test to validate the successful results each time until the desired number of iterations is reached.
    --retry RETRY         retry times
    -n CONCURRENCY, --concurrency CONCURRENCY
                            concurrency
    -i INPUT_FILE, --input-file INPUT_FILE
                            input text file path
    -o OUTPUT_FILE, --output-file OUTPUT_FILE
                            output text file path
    -c, --from-clipboard  text from clipboard, ignore -i. if --output-file not set, output to clipboard
    -q, --quiet           mute the progress in stderr



```


---


## 11. morebuiltins.cmd.ui



11.2 `handle_cli` - Command Line Interface: interactive mode


```python

Args:
    func: a callable function

```


---



11.3 `handle_web` - Function to Web UI.


```python

Args:
    function: callable function
    bind (str, optional): Defaults to "127.0.0.1:8080".
    open_browser (bool, optional): auto open browser. Defaults to False.
    keepalive_timeout (int, optional): shutdown if no request after timeout. Defaults to 60.

```


---



11.4 `handle_tk1` - Function to tkinter UI. (interactive mode)





---



11.5 `handle_tk2` - Function to tkinter UI.





---


## 12. morebuiltins.cmd.parse_deps



12.1 `parse_deps` - Parse dependencies of a project directory, and find circular dependencies.


```python

Args:
      project_dir (str): Path to the project directory.
      ignore_stds (bool, optional): Whether to ignore dependencies from the standard library. Defaults to True.
      format_path (bool, optional): Whether to format the paths. Defaults to True.
      pattern_list (tuple, optional): List of patterns to match files. Defaults to ("*.py",).
Returns:
        dict: A dictionary containing the project directory, circular dependencies, and dependencies.

Demo::

    import json
    import multiprocessing
    from pathlib import Path

    project_dir = Path(multiprocessing.__file__).parent
    result = parse_deps(
        project_dir,
        ignore_stds=True,
        format_path=True,
        pattern_list=("*.py",),
    )
    dependencies = sorted(
        result["dependencies"].items(),
        key=lambda i: (len(i[1]), i[0]),
        reverse=True,
    )
    print("project_dir:", project_dir.as_posix(), flush=True)
    print("circular_dependency:", result["circular_dependency"], flush=True)
    for source, deps in dependencies:
        print(source, f"({len(deps)})", flush=True)
        for i in deps:
            print("\t", i, flush=True)
    # project_dir: D:/python311/Lib/multiprocessing
    # circular_dependency: [('./connection.py', './context.py'), ('./context.py', './forkserver.py'), ('./context.py', './managers.py'), ('./context.py', './popen_forkserver.py'), ('./context.py', './popen_spawn_posix.py'), ('./context.py', './popen_spawn_win32.py'), ('./context.py', './sharedctypes.py'), ('./context.py', './spawn.py'), ('./dummy/__init__.py', './pool.py')]
    # ./context.py (13)
    #        ./connection.py
    #        ./forkserver.py
    #        ./managers.py
    #        ./pool.py
    #        ./popen_fork.py
    #        ./popen_forkserver.py
    #        ./popen_spawn_posix.py
    #        ./popen_spawn_win32.py
    #        ./queues.py
    #        ./sharedctypes.py
    #        ./spawn.py
    #        ./synchronize.py
    #        ./util.py
    # ./synchronize.py (2)
    #        ./heap.py
    #        ./resource_tracker.py
    # ./resource_sharer.py (2)
    #        ./connection.py
    #        ./context.py
    # ./queues.py (2)
    #        ./synchronize.py
    #        ./util.py
    # ./pool.py (2)
    #        ./connection.py
    #        ./dummy/__init__.py
    # ./dummy/__init__.py (2)
    #        ./dummy/connection.py
    #        ./pool.py
    # ./util.py (1)
    #        test
    # ./spawn.py (1)
    #        ./context.py
    # ./sharedctypes.py (1)
    #        ./context.py
    # ./reduction.py (1)
    #        ./resource_sharer.py
    # ./process.py (1)
    #        ./context.py
    # ./popen_spawn_win32.py (1)
    #        ./context.py
    # ./popen_spawn_posix.py (1)
    #        ./context.py
    # ./popen_forkserver.py (1)
    #        ./context.py
    # ./managers.py (1)
    #        ./context.py
    # ./heap.py (1)
    #        ./context.py
    # ./forkserver.py (1)
    #        ./context.py
    # ./connection.py (1)
    #        ./context.py


```


---


## 13. morebuiltins.snippets.event



13.1 `EventTemplate` - Event template for event sourcing





---


## 14. morebuiltins.snippets.sql



14.1 `SqliteSQL` - Sqlite SQL generator





---


## 15. morebuiltins.cmd.systemd.service



15.1 `service_handler` - Generate and manage systemd service files


```python

Example usage:

1. Create, enable and start service:
    python -m morebuiltins.cmd.systemd.service -name myservice -enable -Description "My service" -ExecStart "/bin/bash myscript.sh"
2. Stop, disable and remove service:
    python -m morebuiltins.cmd.systemd.service -name myservice -disable
3. Print service file content:
    python -m morebuiltins.cmd.systemd.service -name myservice -Description "My service" -ExecStart "/bin/bash myscript.sh" -Type simple

```


---


## 16. morebuiltins.cmd.systemd.timer



16.1 `timer_handler` - Parse arguments and manage systemd timer files.


```python
If -enable or -disable is not set, print timer and service file content.

Examples usage:

    1. enable & start timer
        - python -m morebuiltins.cmd.systemd.timer -name mytimer -enable -OnCalendar '*:0/15' -ExecStart '/bin/echo hello'
    2. disable & stop timer
        - python -m morebuiltins.cmd.systemd.timer -name mytimer -disable
    3. print timer and service file content
        - python -m morebuiltins.cmd.systemd.timer -name mytimer -OnCalendar '*:0/15' -ExecStart '/bin/echo hello'

```


---


## 17. morebuiltins.sqlite


## 18. morebuiltins.shared_memory



18.1 `PLock` - A simple process lock using shared memory, for singleton control.


```python
Use `with` context or `close_atexit` to ensure the shared memory is closed in case the process crashes.

Args:
    name (str): name of the shared memory
    force (bool, optional): whether to force rewrite the existing shared memory. Defaults to False.
    close_atexit (bool, optional): whether to close the shared memory at process exit. Defaults to False, to use __del__ or __exit__ instead.

Demo:

    >>> test_pid = 123456 # test pid, often set to None for current process
    >>> plock = PLock("test_lock", force=False, close_atexit=True, pid=test_pid)
    >>> plock.locked
    True
    >>> try:
    ...     plock2 = PLock("test_lock", force=False, close_atexit=True, pid=test_pid + 1)
    ...     raise RuntimeError("Should not be here")
    ... except RuntimeError:
    ...     True
    True
    >>> plock3 = PLock("test_lock", force=True, close_atexit=True, pid=test_pid + 1)
    >>> plock3.locked
    True
    >>> plock.locked
    False
    >>> PLock.wait_for_free(name="test_lock", timeout=0.1, interval=0.01)
    False
    >>> plock.close()
    >>> plock3.close()
    >>> PLock.wait_for_free(name="test_lock", timeout=0.1, interval=0.01)
    True

```


---



18.2 `SharedBytes` - Shared Memory for Python, for python 3.8+.


```python
This module provides a simple way to create and manage shared memory segments, shared between different processes.
Shared memory is faster than other IPC methods like pipes or queues, and it allows for direct access to the memory.

Demo:

>>> sb = SharedBytes(name="test", data=b"Hello, World!", unlink_on_exit=True)
>>> # The size of the shared memory is 18 bytes (5 bytes for header + 13 bytes for data), but mac os may return more than 18 bytes.
>>> sb.size > 10
True
>>> sb.get(name="test")
b'Hello, World!'
>>> sb.re_create(b"New Data")
>>> sb.get(name="test")
b'New Data'
>>> sb.close()
>>> sb.get(name="test", default=b"")  # This will raise ValueError since the shared memory is closed
b''

```


---


## 19. morebuiltins.logs



19.1 `async_logger` - Asynchronous non-blocking QueueListener that manages logger handlers.


```python
logger is a logging.Logger instance.
queue is a Queue or ProcessQueue instance.
respect_handler_level is a boolean that determines if the handler level should be respected.

Example:

    async def main():
        # Create logger with a blocking handler
        logger = logging.getLogger("example")
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(stream_handler)
        # Use async queue listener
        async with AsyncQueueListener(logger):
            # Log won't block the event loop
            for i in range(5):
                logger.info("log info")
                logger.debug("log debug")
                await asyncio.sleep(0.01)

```


---



19.2 `AsyncQueueListener` - Asynchronous non-blocking QueueListener that manages logger handlers.


```python
logger is a logging.Logger instance.
queue is a Queue or ProcessQueue instance.
respect_handler_level is a boolean that determines if the handler level should be respected.

Example:

    async def main():
        # Create logger with a blocking handler
        logger = logging.getLogger("example")
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(stream_handler)
        # Use async queue listener
        async with AsyncQueueListener(logger):
            # Log won't block the event loop
            for i in range(5):
                logger.info("log info")
                logger.debug("log debug")
                await asyncio.sleep(0.01)

```


---



19.3 `LogHelper` - Quickly bind a logging handler to a logger, with a StreamHandler or SizedTimedRotatingFileHandler.


```python

The default handler is a StreamHandler to sys.stderr.
The default file handler is a SizedTimedRotatingFileHandler, which can rotate logs by both time and size.

Examples::

    # 1. Bind a StreamHandler to the "mylogger" logger, output to sys.stdout
    import logging
    from morebuiltins.logs import LogHelper

    LogHelper.shorten_level()
    logger = LogHelper.bind_handler(name="mylogger", filename=sys.stdout, maxBytes=100 * 1024**2, backupCount=7)
    # use logging.getLogger to get the same logger instance
    logger2 = logging.getLogger("mylogger")
    assert logger is logger2
    logger.info("This is an info message")
    logger.fatal("This is a critical message")

    # 2. Bind file and stderr in the same logger
    import sys
    import logging
    from morebuiltins.logs import LogHelper
    LogHelper.shorten_level()
    logger = LogHelper.bind_handler(name="mylogger", filename="mylog.log", maxBytes=100 * 1024**2, backupCount=7)
    logger = LogHelper.bind_handler(name="mylogger", filename=sys.stderr)
    logger.info("This is an info message")

    # 3. Use queue=True to make logging non-blocking, both file and stderr
    import sys
    from morebuiltins.logs import LogHelper
    LogHelper.shorten_level()
    logger = LogHelper.bind_handler(name="mylogger", filename="mylog.log", maxBytes=100 * 1024**2, backupCount=7, queue=True)
    logger = LogHelper.bind_handler(name="mylogger", filename=sys.stderr, queue=True)
    logger.info("This is an info message")

```


---



19.4 `RotatingFileWriter` - RotatingFileWriter class for writing to a file with rotation support.


```python

Demo::

    >>> # test normal usage
    >>> writer = RotatingFileWriter("test.log", max_size=10 * 1024, max_backups=1)
    >>> writer.write("1" * 10)
    >>> writer.path.stat().st_size
    0
    >>> writer.flush()
    >>> writer.path.stat().st_size
    10
    >>> writer.clean_backups(writer.max_backups)
    >>> writer.unlink_file()
    >>> # test rotating
    >>> writer = RotatingFileWriter("test.log", max_size=20, max_backups=2)
    >>> writer.write("1" * 15)
    >>> writer.write("1" * 15)
    >>> writer.write("1" * 15, flush=True)
    >>> writer.path.stat().st_size
    15
    >>> len(writer.backup_path_list())
    2
    >>> writer.clean_backups(writer.max_backups)
    >>> writer.unlink_file()
    >>> # test no backups
    >>> writer = RotatingFileWriter("test.log", max_size=20, max_backups=0)
    >>> writer.write("1" * 15)
    >>> writer.write("1" * 15)
    >>> writer.write("1" * 15, flush=True)
    >>> writer.path.stat().st_size
    15
    >>> len(writer.backup_path_list())
    0
    >>> writer.clean_backups(writer.max_backups)
    >>> len(writer.backup_path_list())
    0
    >>> writer = RotatingFileWriter("test.log", max_size=20, max_backups=3)
    >>> writer.print("1" * 100)
    >>> writer.unlink(rotate=False)
    >>> len(writer.backup_path_list())
    1
    >>> writer.unlink(rotate=True)
    >>> len(writer.backup_path_list())
    0
    >>> writer = RotatingFileWriter("test.log", max_size=20, max_backups=3, compress=True)
    >>> writer.print("1" * 100)
    >>> len(writer.backup_path_list())
    1
    >>> writer.unlink(rotate=True)
    >>> len(writer.backup_path_list())
    0

```


---



19.5 `SizedTimedRotatingFileHandler` - TimedRotatingFileHandler with maxSize, to avoid files that are too large.


```python


Demo::

    import logging
    import time
    from morebuiltins.funcs import SizedTimedRotatingFileHandler

    logger = logging.getLogger("test1")
    h = SizedTimedRotatingFileHandler(
        "logs/test1.log", "d", 1, 3, maxBytes=1, ensure_dir=True
    )
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

    for i in range(5):
        logger.warning(str(i) * 102400)
        time.sleep(1)
    # 102434 test1.log
    # 102434 test1.log.20241113_231000
    # 102434 test1.log.20241113_231001
    # 102434 test1.log.20241113_231002
    logger = logging.getLogger("test2")
    h = SizedTimedRotatingFileHandler(
        "logs/test2.log", "d", 1, 3, maxBytes=1, compress=True
    )
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

    for i in range(5):
        logger.warning(str(i) * 102400)
        time.sleep(1)
    # 102434 test2.log
    #    186 test2.log.20241113_231005.gz
    #    186 test2.log.20241113_231006.gz
    #    186 test2.log.20241113_231007.gz


```


---



19.6 `ContextFilter` - A logging filter that injects context variables into extra of log records. ContextVar is used to manage context-specific data in a thread-safe / async-safe manner.


```python
RequestID / TraceID / TaskID can be used to trace logs belonging to the same request or operation across different threads or async tasks.

Example::

    import random
    import time
    import typing
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import ContextVar
    from logging import Filter, Formatter, StreamHandler, getLogger
    from threading import current_thread

    def test(trace_id: int = 0):
        trace_id_var.set(trace_id)
        for _ in range(3):
            time.sleep(random.random())
            logger.debug(f"msg from thread: {current_thread().ident}")


    trace_id_var: ContextVar = ContextVar("trace_id")
    logger = getLogger()
    logger.addFilter(ContextFilter({"trace_id": trace_id_var}))
    formatter = Formatter("%(asctime)s | [%(trace_id)s] %(message)s")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("DEBUG")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = [executor.submit(test, _) for _ in range(3)]


```


---



19.7 `LoggerStream` - LoggerStream constructor.


```python

Args:
    skip_same_head (bool, optional): Whether to skip the same log head. Defaults to True.
        True:
            24-08-10 19:30:07 This is a log message.
            This is another log message.
            24-08-10 19:30:08 This is a new log message.
        False:
            24-08-10 19:30:07 This is a log message.
            24-08-10 19:30:07 This is another log message.

Example::

    # 1. Basic usage
    logger = LoggerStream(skip_same_head=True)
    logger.write("This is a log message.\n")
    logger.write("This is another log message.\n")
    logger.write("This is a new log message.\n")

    # 2. Redirect sys.stdout
    import sys
    sys.stdout = LoggerStream(skip_same_head=False)
    print("This is a log message.")
    print("This is a log message.")
    # 24-08-10 19:30:07 This is a log message.
    # 24-08-10 19:30:07 This is a log message.

    # 3. Overwrite built-in print function
    LoggerStream.install_print()
    print(123)
    print(123)
    # 24-08-10 19:30:07 123
    # 123
    LoggerStream.restore_print()
    LoggerStream.install_print(writer=open("log.txt", "a").write)

    # 4. Subclass and override writer method
    class CustomLoggerStream(LoggerStream):
        def __init__(self, skip_same_head=True):
            super().__init__(skip_same_head=skip_same_head)
            self.logger = setup_your_logger_somehow()

        def writer(self, msg: str):
            # Custom implementation to write log message
            self.logger.info(msg)

```


---



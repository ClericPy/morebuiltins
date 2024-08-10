## 1. morebuiltins.utils



1.1 `ttime` - Converts a timestamp to a human-readable timestring formatted as %Y-%m-%d %H:%M:%S.


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



1.2 `ptime` - Converts a timestring formatted as %Y-%m-%d %H:%M:%S back into a timestamp.


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



1.8 `get_hash` - Generates an MD5 hash string from the given input string.


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
    >>> default_dict(Demo, bytes_obj=b'1')
    {'int_obj': 0, 'float_obj': 0.0, 'bytes_obj': b'1', 'str_obj': '', 'list_obj': [], 'tuple_obj': (), 'set_obj': set(), 'dict_obj': {}}
    
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



1.17 `format_error` - Extracts frame information from an exception, with an option to filter out “site-packages” details by default.


```python

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
    >>> try:
    ...     # test with default filter(filename skip site-packages)
    ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
    ...     version_info_to_nodot(0)
    ... except Exception as e:
    ...     format_error(e)
    "[<doctest>:<module>(4)] version_info_to_nodot(0) >>> TypeError('int' object is not subscriptable)"
    >>> try:
    ...     # test without filter
    ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
    ...     version_info_to_nodot(0)
    ... except Exception as e:
    ...     format_error(e, filter=None)
    '[<doctest>:<module>(4)|compatibility_tags.py:version_info_to_nodot(23)] return "".join(map(str, version_info[:2])) >>> TypeError(\'int\' object is not subscriptable)'
    >>> try:
    ...     # test with custom filter.
    ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
    ...     version_info_to_nodot(0)
    ... except Exception as e:
    ...     format_error(e, filter=lambda i: '<doctest' in str(i))
    "[<doctest>:<module>(4)] version_info_to_nodot(0) >>> TypeError('int' object is not subscriptable)"
    
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
    >>> is_running(os.getpid())  # Check if the current process is running
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


## 3. morebuiltins.functools



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



3.2 `threads` - Quickly convert synchronous functions to be concurrency-able. (similar to madisonmay/Tomorrow)


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

                asyncio.get_event_loop().run_until_complete(main())

            test_sync()
            test_async()
    
```


---



3.5 `FuncSchema` - Parse the parameters and types required by a function into a dictionary, and convert an incoming parameter into the appropriate type.


```python

    >>> def test(a, b: str, /, c=1, *, d=["d"], e=0.1, f={"f"}, g=(1, 2), h=True, i={1}, **kws):
    ...     return
    >>> FuncSchema.parse(test, strict=False)
    {'b': {'type': <class 'str'>, 'default': <class 'inspect._empty'>}, 'c': {'type': <class 'int'>, 'default': 1}, 'd': {'type': <class 'list'>, 'default': ['d']}, 'e': {'type': <class 'float'>, 'default': 0.1}, 'f': {'type': <class 'set'>, 'default': {'f'}}, 'g': {'type': <class 'tuple'>, 'default': (1, 2)}, 'h': {'type': <class 'bool'>, 'default': True}, 'i': {'type': <class 'set'>, 'default': {1}}}
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



3.7 `SizedTimedRotatingFileHandler` - TimedRotatingFileHandler with maxSize, to avoid files that are too large.


```python

    no test.

    Demo::

        import logging
        import time

        logger = logging.getLogger("test")
        h = SizedTimedRotatingFileHandler("test.log", "d", 1, 3, maxBytes=1)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)

        for i in range(5):
            logger.warning(str(i) * 100)
            time.sleep(1)
        # 2024/06/25 22:47   134     test.log.20240625_224717
        # 2024/06/25 22:47   134     test.log.20240625_224718
        # 2024/06/25 22:47   134     test.log.20240625_224719
    
```


---



3.8 `get_type_default` - Get the default value for a type. {int: 0, float: 0.0, bytes: b"", str: "", list: [], tuple: (), set: set(), dict: {}}





---



3.9 `func_cmd` - Handle function with argparse, typing-hint is nessessary.


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



3.10 `file_import` - Import function from file path.


```python

    Demo::
        >>> from pathlib import Path
        >>> file_path = Path(__file__).parent / "utils.py"
        >>> list(file_import(file_path, ["get_hash", "find_jsons"]).keys())
        ['get_hash', 'find_jsons']
    
```


---



3.11 `RotatingFileWriter` - RotatingFileWriter class for writing to a file with rotation support.


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
        >>> writer.unlink_file()
    
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



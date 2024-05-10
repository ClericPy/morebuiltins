======================

## 1. morebuiltins.utils


======================


    1.1 `ttime` - Converts a timestamp to a human-readable timestring formatted as %Y-%m-%d %H:%M:%S.

        >>> ttime(1486572818.421858323, tzone=8)
        '2017-02-09 00:53:38'

        Args:
            timestamp (float, optional): the timestamp float. Defaults to time.time().
            tzone (int, optional): time compensation. Defaults to int(-time.timezone / 3600).
            fmt (str, optional): strftime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

        Returns:
            str: time string formatted.
    

---


    1.2 `ptime` - Converts a timestring formatted as %Y-%m-%d %H:%M:%S back into a timestamp.

        >>> ptime("2018-03-15 01:27:56", tzone=8)
        1521048476

        Args:
            timestring (str, optional): string like 2018-03-15 01:27:56. Defaults to ttime().
            tzone (int, optional): time compensation. Defaults to int(-timezone / 3600).
            fmt (_type_, optional): strptime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

        Returns:
            str: time string formatted.
    

---


    1.3 `slice_into_pieces` - Divides a sequence into “n” segments, returning a generator that yields “n” pieces.

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
    

---


    1.4 `slice_by_size` - Slices a sequence into chunks of a specified “size”, returning a generator that produces tuples of chunks.

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
    

---


    1.5 `unique` - Removes duplicate elements from a sequence while preserving the original order efficiently.

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
    

---


    1.6 `retry` - A decorator that retries the decorated function up to “tries” times if the specified exceptions are raised.

        >>> func = lambda items: 1/items.pop(0)
        >>> items = [0, 1]
        >>> new_func = retry(tries=2, exceptions=(ZeroDivisionError,))(func)
        >>> new_func(items)
        1.0

        Args:
            tries (int, optional): try n times, if n==1 means no retry. Defaults to 1.
            exceptions (Tuple[Type[BaseException]], optional): only retry the given errors. Defaults to (Exception,).
            return_exception (bool, optional): raise the last exception or return it. Defaults to False.
    

---


    1.7 `guess_interval` - Analyzes a sequence of numbers and returns the median, calculating intervals only if they are greater than or equal to the specified accuracy.

        >>> # sorted_seq: [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
        >>> # diffs: [8, 7, 10, 6, 13, 8]
        >>> # median: 8
        >>> seq = [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
        >>> guess_interval(seq, 5)
        8

    

---


    1.8 `get_hash` - Generates an MD5 hash string from the given input string.

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
    

---


    1.9 `find_jsons` - A generator that locates valid JSON strings, supporting only dictionaries and lists.

        >>> list(find_jsons('string["123"]123{"a": 1}[{"a": 1, "b": [1,2,3]}]'))
        ['["123"]', '{"a": 1}', '[{"a": 1, "b": [1,2,3]}]']
        >>> list(find_jsons('string[]{}{"a": 1}'))
        ['[]', '{}', '{"a": 1}']
        >>> list(find_jsons('string[]|{}string{"a": 1}', return_as='index'))
        [(6, 8), (9, 11), (17, 25)]
        >>> list(find_jsons('xxxx[{"a": 1, "b": [1,2,3]}]xxxx', return_as='object'))
        [[{'a': 1, 'b': [1, 2, 3]}]]
    

---


    1.10 `code_inline` - Minifies Python source code into a single line.

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
    

---


    1.11 `read_size` - Converts byte counts into a human-readable string. Setting shorten=True and precision=0.99 will trim unnecessary decimal places from the tail of floating-point numbers.

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


    

---


    1.12 `read_time` - Converts seconds into a more readable time duration string.

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
    

---


    1.13 `Validator` - Validator for dataclasses.
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
    

---


    1.14 `stagger_sort` - Ensures that identical groups are ordered and evenly distributed, mitigating data skew. The function does not alter the original list and returns a generator.

        >>> items = [('a', 0), ('a', 2), ('a', 1), ('b', 0), ('b', 1)]
        >>> list(stagger_sort(items, sort_key=lambda i: (i[0], i[1]), group_key=lambda i: i[0]))
        [('a', 0), ('b', 0), ('a', 1), ('b', 1), ('a', 2)]

    

---


    1.15 `default_dict` - Initializes a dictionary with default zero values based on a subclass of TypedDict.

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
    

---


    1.16 `always_return_value` - Got a function always return the given value.
        >>> func = always_return_value(1)
        >>> func(1, 2, 3)
        1
        >>> func(1, 2, c=3)
        1
    

---


    1.17 `format_error` - Extracts frame information from an exception, with an option to filter out “site-packages” details by default.

        >>> try:
        ...     # test default
        ...     1 / 0
        ... except Exception as e:
        ...     format_error(e)
        '[<doctest morebuiltins.utils.format_error[0]>:<module>:3] 1 / 0 >>> ZeroDivisionError(division by zero)'
        >>> try:
        ...     # test in function
        ...     def func1(): 1 / 0
        ...     func1()
        ... except Exception as e:
        ...     format_error(e)
        '[<doctest morebuiltins.utils.format_error[1]>:func1:3] def func1(): 1 / 0 >>> ZeroDivisionError(division by zero)'
        >>> try:
        ...     # test index
        ...     def func2(): 1 / 0
        ...     func2()
        ... except Exception as e:
        ...     format_error(e, index=0)
        '[<doctest morebuiltins.utils.format_error[2]>:<module>:4] func2() >>> ZeroDivisionError(division by zero)'
        >>> try:
        ...     # test with default filter(filename skip site-packages)
        ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
        ...     version_info_to_nodot(0)
        ... except Exception as e:
        ...     format_error(e)
        "[<doctest morebuiltins.utils.format_error[3]>:<module>:4] version_info_to_nodot(0) >>> TypeError('int' object is not subscriptable)"
        >>> try:
        ...     # test without filter
        ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
        ...     version_info_to_nodot(0)
        ... except Exception as e:
        ...     format_error(e, filter=None)
        '[compatibility_tags.py:version_info_to_nodot:23] return "".join(map(str, version_info[:2])) >>> TypeError(\'int\' object is not subscriptable)'
        >>> try:
        ...     # test with custom filter.
        ...     from pip._internal.utils.compatibility_tags import version_info_to_nodot
        ...     version_info_to_nodot(0)
        ... except Exception as e:
        ...     format_error(e, filter=lambda i: '<doctest' in str(i))
        "[<doctest morebuiltins.utils.format_error[5]>:<module>:4] version_info_to_nodot(0) >>> TypeError('int' object is not subscriptable)"
    

---


    1.18 `Trie` - Transforms a standard dictionary into a trie structure that supports prefix matching.

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
    

---


    1.19 `GuessExt` - Determines whether the input bytes of a file prefix indicate a compressed file format.

        >>> cg = GuessExt()
        >>> cg.get_ext(b"PKzipfiledemo")
        '.zip'
    

---

======================

## 2. morebuiltins.functools


======================


    2.1 `lru_cache_ttl` - A Least Recently Used (LRU) cache with a Time To Live (TTL) feature.

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
    

---


    2.2 `threads` - Quickly convert synchronous functions to be concurrency-able. (similar to madisonmay/Tomorrow)

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
    

---


    2.3 `bg_task` - Avoid asyncio free-flying tasks, better to use the new asyncio.TaskGroup to avoid this in 3.11+. https://github.com/python/cpython/issues/91887

        Args:
            coro (Coroutine)

        Returns:
            _type_: Task

    

---


    2.4 `NamedLock` - Reusable named locks, support for timeouts, support for multiple concurrent locks.

                ```python

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


    2.5 `FuncSchema` - Parse the parameters and types required by a function into a dictionary, and convert an incoming parameter into the appropriate type.

        >>> def test(a, b: str, /, c=1, *, d=["d"], e=0.1, f={"f"}, g=(1, 2), h=True, i={1}, **kws):
        ...     return
        >>> FuncSchema.parse(test)
        {'b': {'type': <class 'str'>, 'default': <class 'inspect._empty'>}, 'c': {'type': <class 'int'>, 'default': 1}, 'd': {'type': <class 'list'>, 'default': ['d']}, 'e': {'type': <class 'float'>, 'default': 0.1}, 'f': {'type': <class 'set'>, 'default': {'f'}}, 'g': {'type': <class 'tuple'>, 'default': (1, 2)}, 'h': {'type': <class 'bool'>, 'default': True}, 'i': {'type': <class 'set'>, 'default': {1}}}
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
        >>> FuncSchema.convert('[1, 1]', set)
        {1}
        >>> FuncSchema.convert('[1, 1]', tuple)
        (1, 1)
    

---

======================

## 3. morebuiltins.ipc


======================


    3.1 `IPCEncoder` - An abstract base class for all encoders; implementing the necessary communication protocol requires only the definition of two abstract methods. Be mindful that varying header lengths will impact the maximum packaging size.


---


    3.4 `SocketLogHandlerEncoder` - For a practical demonstration, refer to the test code: morebuiltins/ipc.py:_test_ipc_logging.

        ```
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
        ```
        And provide a simple implementation for generating logs for coroutine code with Client usage.
    

---


    3.5 `SocketServer` - To see an example in action, view the test code: morebuiltins/ipc.py:_test_ipc.

            ```
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

======================

## 4. morebuiltins.request


======================


    4.1 `req` - A basic mock for requests, performant albeit simplistic.

        >>> import time
        >>> r = req.get("https://postman-echo.com/get?a=2", timeout=3, params={"b": "3"})
        >>> r.url
        'https://postman-echo.com/get?a=2&b=3'
        >>> r.ok
        True
        >>> r.status_code
        200
        >>> r.text.startswith('{')
        True
        >>> r = req.post("https://postman-echo.com/post?a=2", timeout=3, params={"b": "3"}, data=b"mock data")
        >>> r.json()["data"]
        'mock data'
        >>> r.json()["args"]
        {'a': '2', 'b': '3'}
        >>> r = req.post("https://postman-echo.com/post?a=2", timeout=3, json={"data": "yes json"})
        >>> r.json()["json"]
        {'data': 'yes json'}
    

---


    4.2 `DomainParser` - Extracts the Second-level domain (SLD) from a provided hostname or URL.

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

    

---


    4.3 `unparse_qsl` - Provides the inverse operation of parse_qsl, converting query string lists back into a URL-encoded string.


---


    4.4 `update_url` - Organizes the query arguments within a URL to standardize its format.

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
    

---

======================

## 5. morebuiltins.download_python


======================


    5.1 `download_python` - Usage: python -m morebuiltins.download_python

        λ python -m morebuiltins.download_python
        View the rules:
        https://gregoryszorc.com/docs/python-build-standalone/main/running.html#obtaining-distributions

        Got 290 urls from github.

        [290] Enter keywords (can be int index or partial match, defaults to 0):
        0. windows
        1. linux
        2. darwin

        Filt with keyword: "windows". 290 => 40

        [40] Enter keywords (can be int index or partial match, defaults to 0):
        0. 3.12.3
        1. 3.11.9
        2. 3.10.14
        3. 3.9.19
        4. 3.8.19

        Filt with keyword: "3.12.3". 40 => 8

        [8] Enter keywords (can be int index or partial match, defaults to 0):
        0. x86_64
        1. i686

        Filt with keyword: "x86_64". 8 => 4

        [4] Enter keywords (can be int index or partial match, defaults to 0):
        0. shared-pgo-full.tar.zst
        1. shared-install_only.tar.gz
        2. pgo-full.tar.zst
        3. install_only.tar.gz

        Filt with keyword: "shared-pgo-full.tar.zst". 4 => 1
        Download URL: 40.4 MB
        https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-pc-windows-msvc-shared-pgo-full.tar.zst
        File path to save(defaults to `./cpython-3.12.3+20240415-x86_64-pc-windows-msvc-shared-pgo-full.tar.zst`)?
        or `q` to exit.

        Start downloading...
        https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-pc-windows-msvc-shared-pgo-full.tar.zst
        D:\github\morebuiltins\cpython-3.12.3+20240415-x86_64-pc-windows-msvc-shared-pgo-full.tar.zst
        [Downloading]: 2.21 / 40.44 MB | 5.47%

---


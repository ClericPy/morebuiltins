======================

### 1. morebuiltins.utils

======================


    1.1 ttime - From timestamp to timestring. Translate timestamp into human-readable: %Y-%m-%d %H:%M:%S.
        >>> ttime(1486572818.421858323, tzone=8)
        '2017-02-09 00:53:38'

        Args:
            timestamp (float, optional): the timestamp float. Defaults to time.time().
            tzone (int, optional): time compensation. Defaults to int(-time.timezone / 3600).
            fmt (str, optional): strftime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

        Returns:
            str: time string formatted.
    

---


    1.2 ptime - From timestring to timestamp. Translate %Y-%m-%d %H:%M:%S into timestamp
        >>> ptime("2018-03-15 01:27:56", tzone=8)
        1521048476

        Args:
            timestring (str, optional): string like 2018-03-15 01:27:56. Defaults to ttime().
            tzone (int, optional): time compensation. Defaults to int(-timezone / 3600).
            fmt (_type_, optional): strptime fmt. Defaults to "%Y-%m-%d %H:%M:%S".

        Returns:
            str: time string formatted.
    

---


    1.3 slice_into_pieces - Slice a sequence into `n` pieces, return a generation of n pieces.
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
    

---


    1.4 slice_by_size - Slice a sequence into chunks, return as a generation of tuple chunks with `size`.
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


    1.5 get_hash - Get the md5_string from given string
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


    1.6 unique - Unique the seq and keep the order(fast).
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


    1.7 guess_interval - Given a seq of number, return the median, only calculate interval >= accuracy.
        >>> # sorted_seq: [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
        >>> # diffs: [8, 7, 10, 6, 13, 8]
        >>> # median: 8
        >>> seq = [2, 10, 12, 19, 19, 29, 30, 32, 38, 40, 41, 54, 62]
        >>> guess_interval(seq, 5)
        8

    

---


    1.8 retry - A decorator which will retry the function `tries` times while raising given exceptions.
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


    1.9 find_jsons - Generator for finding the valid JSON string, only support dict and list.
        >>> list(find_jsons('string["123"]123{"a": 1}[{"a": 1, "b": [1,2,3]}]'))
        ['["123"]', '{"a": 1}', '[{"a": 1, "b": [1,2,3]}]']
        >>> list(find_jsons('string[]{}{"a": 1}'))
        ['[]', '{}', '{"a": 1}']
        >>> list(find_jsons('string[]|{}string{"a": 1}', return_as='index'))
        [(6, 8), (9, 11), (17, 25)]
        >>> list(find_jsons('xxxx[{"a": 1, "b": [1,2,3]}]xxxx', return_as='object'))
        [[{'a': 1, 'b': [1, 2, 3]}]]
    

---


    1.10 code_inline - Make the python source code inline.
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


    1.11 stagger_sort - Ensure that the same group is ordered and staggered, avoid data skew. Will not affect the original list, return as a generator.
        >>> items = [('a', 0), ('a', 2), ('a', 1), ('b', 0), ('b', 1)]
        >>> list(stagger_sort(items, sort_key=lambda i: (i[0], i[1]), group_key=lambda i: i[0]))
        [('a', 0), ('b', 0), ('a', 1), ('b', 1), ('a', 2)]

    

---


    1.12 read_size - From bytes to readable string. shorten=True and precision=0.99 can shorten unnecessary tail floating-point numbers.
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


    1.13 read_time - From secs to readable string.
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


    1.14 default_dict - Init a default zero-value dict from the subclass of TypedDict.
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


    1.15 format_error - Get the frame info from Exception. Default filter will skip `site-packages` info.
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


    1.16 Trie - Make a normal dict to trie tree with the feature of prefix-match.
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


    1.17 GuessExt - Determine whether the input prefix bytes are compressed files,
        >>> cg = GuessExt()
        >>> cg.get_ext(b"PKzipfiledemo")
        '.zip'
    

---

======================

### 2. morebuiltins.ipc

======================


    2.1 IPCEncoder - Abstract base class for all encoders; users only need to implement two abstract methods to set up the communication protocol. Note that different header lengths affect the packaging max length.


---


    2.4 SocketLogHandlerEncoder - View the test code: morebuiltins\ipc.py:_test_ipc_logging

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
    

---


    2.5 SocketServer - View the test code: morebuiltins\ipc.py:_test_ipc
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

### 3. morebuiltins.request

======================


    3.1 req - A simple requests mock, slow but useful.
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


    3.2 DomainParser - Get the Second-level domain(SLD) from a hostname or a url.
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


    3.3 unparse_qsl - Reverse conversion for parse_qsl


---


    3.4 update_url - Sort url query args to unify format the url.
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

        `replace_kwargs` is a dict to update attributes before sorting  (such as scheme / netloc...).
    

---


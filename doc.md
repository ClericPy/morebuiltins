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


    1.12 read_size - From B to readable string.
        >>> read_size(0)
        '0 B'
        >>> for i in range(0, 10):
        ...     [1 * 1024**i, read_size(1 * 1024**i, rounded=1)]
        ...
        [1, '1.0 B']
        [1024, '1.0 KB']
        [1048576, '1.0 MB']
        [1073741824, '1.0 GB']
        [1099511627776, '1.0 TB']
        [1125899906842624, '1.0 PB']
        [1152921504606846976, '1.0 EB']
        [1180591620717411303424, '1.0 ZB']
        [1208925819614629174706176, '1.0 YB']
        [1237940039285380274899124224, '1024.0 YB']

        Args:
            b: B
            rounded (int, optional): arg for round. Defaults to None.

        Returns:
            str


    

---


    1.13 read_time - From secs to readable string.
        >>> read_time(0)
        '0 secs'
        >>> for i in range(0, 6):
        ...     [1.2345 * 60**i, read_time(1.2345 * 60**i, rounded=1)]
        ...
        [1.2345, '1.2 secs']
        [74.07, '1.2 mins']
        [4444.2, '1.2 hours']
        [266652.0, '3.1 days']
        [15999120.0, '6.2 mons']
        [959947200.0, '30.4 years']

        Args:
            b: seconds
            rounded (int, optional): arg for round. Defaults to None.

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
        ...     # test with default filter
        ...     import os
        ...     os.path.join(1, 2, 3)
        ... except Exception as e:
        ...     format_error(e)
        '[<doctest morebuiltins.utils.format_error[3]>:<module>:4] os.path.join(1, 2, 3) >>> TypeError(expected str, bytes or os.PathLike object, not int)'
        >>> try:
        ...     # test without filter
        ...     import os
        ...     os.path.join(1, 2, 3)
        ... except Exception as e:
        ...     format_error(e, filter=None)
        '[<frozen ntpath>:join:108]  >>> TypeError(expected str, bytes or os.PathLike object, not int)'
        >>> try:
        ...     # test without custom filter. refuse all, raise IndexError to return ''
        ...     import os
        ...     os.path.join(1, 2, 3)
        ... except Exception as e:
        ...     format_error(e, filter=lambda tb: False)
        ''
    

---

======================

### 2. morebuiltins.request

======================


    2.1 req - A simple requests mock, slow but useful.
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


    2.2 DomainParser - Get the Second-level domain(SLD) from a hostname or a url.
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


    2.3 unparse_qsl - Reverse conversion for parse_qsl


---


    2.4 update_url - Sort url query args to unify format the url.
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

======================

### 3. morebuiltins.ipc

======================


    3.1 IPCEncoder - Abstract base class for all encoders; users only need to implement two abstract methods to set up the communication protocol. Note that different header lengths affect the packaging max length.


---


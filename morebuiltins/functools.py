import asyncio
import importlib.util
import inspect
import json
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from itertools import chain
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from threading import Lock, Semaphore
from typing import Callable, Coroutine, Dict, Optional, OrderedDict, Set, Union
from weakref import WeakSet

__all__ = [
    "lru_cache_ttl",
    "threads",
    "bg_task",
    "NamedLock",
    "FuncSchema",
    "InlinePB",
    "SizedTimedRotatingFileHandler",
    "get_type_default",
    "func_cmd",
    "file_import",
    "RotatingFileWriter",
]


def lru_cache_ttl(
    maxsize: int,
    ttl: Optional[Union[int, float]] = None,
    controls=False,
    auto_clear=True,
    timer=time.time,
):
    """A Least Recently Used (LRU) cache with a Time To Live (TTL) feature.

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
    """
    cache: OrderedDict = OrderedDict()
    move_to_end = cache.move_to_end
    popitem = cache.popitem
    next_clear_ts = timer()

    def ttl_clean(expire):
        for k, v in tuple(cache.items()):
            if v[1] <= expire:
                cache.pop(k, None)

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            nonlocal next_clear_ts
            now = timer()
            if ttl is None:
                expire_cleared = True
            else:
                expire_cleared = auto_clear and now > next_clear_ts
                if expire_cleared:
                    ttl_clean(now)
                    next_clear_ts = now + ttl
            # key = _make_key(args, kwargs, False)
            key = hash(tuple(chain(args, kwargs.items())))
            if key in cache:
                if expire_cleared is True or now < cache[key][1]:
                    # 1. key in cache; 2. cache is valid
                    # set newest, return result
                    move_to_end(key)
                    return cache[key][0]
            # call function
            result = func(*args, **kwargs)
            # ensure size
            while len(cache) >= maxsize:
                popitem(last=False)
            if ttl is None:
                # no need save expired ts
                cache[key] = (result,)
            else:
                # setitem(key, (result, now + ttl))
                cache[key] = (result, timer() + ttl)
            # {key: (result, expired_ts)}
            return result

        if controls:
            setattr(wrapped, "cache", cache)
            setattr(wrapped, "ttl_clean", ttl_clean)
        return wrapped

    return decorator


def threads(n: Optional[int] = None, executor_class=None, **kws):
    """Quickly convert synchronous functions to be concurrency-able. (similar to madisonmay/Tomorrow)

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
    """
    pool = (executor_class or ThreadPoolExecutor)(max_workers=n, **kws)
    tasks: WeakSet = WeakSet()

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            future = pool.submit(func, *args, **kwargs)
            future.add_done_callback(lambda task: tasks.discard(task))
            tasks.add(future)
            return future

        setattr(wrapped, "pool", pool)
        setattr(wrapped, "tasks", tasks)
        return wrapped

    return decorator


_bg_tasks: Set[asyncio.Task] = set()


def bg_task(coro: Coroutine) -> asyncio.Task:
    """Avoid asyncio free-flying tasks, better to use the new asyncio.TaskGroup to avoid this in 3.11+. https://github.com/python/cpython/issues/91887

    Args:
        coro (Coroutine)

    Returns:
        _type_: Task

    """
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


LockType = Union[Lock, Semaphore, asyncio.Lock, asyncio.Semaphore]


class NamedLock:
    """Reusable named locks, support for timeouts, support for multiple concurrent locks.

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
    """

    _SYNC_CACHE: Dict[str, LockType] = {}
    _ASYNC_CACHE: Dict[str, LockType] = {}

    def __init__(self, name: str, default_factory: Callable, timeout=None):
        self.name = name
        self.default_factory = default_factory
        self.timeout = timeout
        self.lock = None

    @classmethod
    def clear_unlocked(cls):
        for cache in [cls._SYNC_CACHE, cls._ASYNC_CACHE]:
            for name, lock in list(cache.items()):
                if hasattr(lock, "locked") and not lock.locked():
                    cache.pop(name, None)
                elif isinstance(lock, Semaphore) and (
                    lock._value == 0 or not lock._cond._lock.locked()
                ):
                    cache.pop(name, None)

    def __enter__(self):
        if self.name in self._SYNC_CACHE:
            lock = self._SYNC_CACHE[self.name]
        else:
            lock = self.default_factory()
            self._SYNC_CACHE[self.name] = lock
        if lock.acquire(timeout=self.timeout):
            self.lock = lock
            return self
        else:
            return None

    def __exit__(self, *_):
        if self.lock:
            self.lock.release()

    async def __aenter__(self):
        if self.name in self._ASYNC_CACHE:
            lock = self._ASYNC_CACHE[self.name]
        else:
            lock = self.default_factory()
            self._ASYNC_CACHE[self.name] = lock
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.timeout)
            self.lock = lock
            return self
        except asyncio.TimeoutError:
            return None

    async def __aexit__(self, *_):
        if self.lock:
            self.lock.release()


class FuncSchema:
    """Parse the parameters and types required by a function into a dictionary, and convert an incoming parameter into the appropriate type.

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
    """

    ALLOW_TYPES = {int, float, str, tuple, list, set, dict, bool}
    JSON_TYPES = {tuple, list, set, dict, bool}

    @classmethod
    def parse(cls, function: Callable, strict=True, fill_default=False):
        sig = inspect.signature(function)
        result = {}
        for param in sig.parameters.values():
            if param.annotation is param.empty:
                if param.default is param.empty:
                    if strict:
                        raise TypeError(
                            f"Parameter `{param.name}` has no type and no default value."
                        )
                    else:
                        continue
                tp = type(param.default)
            else:
                tp = param.annotation
            if tp in cls.ALLOW_TYPES:
                default = param.default
                if fill_default and default is inspect._empty:
                    default = get_type_default(tp, default)
                result[param.name] = {"type": tp, "default": default}
        return result

    @classmethod
    def convert(cls, obj, target_type):
        if isinstance(obj, str) and target_type in cls.JSON_TYPES:
            return target_type(json.loads(obj))
        else:
            return target_type(obj)


class InlinePB(object):
    """Inline progress bar.

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

    """

    def __init__(
        self,
        total,
        start=0,
        maxlen=50,
        fresh_interval=0.1,
        timer=time.time,
        sig="|",
        sig_len=15,
    ):
        self.total = total
        self.done = start
        self.maxlen = maxlen
        self.cache = deque([], maxlen=maxlen)
        self.timer = timer
        self.last_fresh = self.timer()
        self.fresh_interval = fresh_interval
        self.sig = sig
        self.sig_len = sig_len

    def update(self, done):
        self.add(done - self.done)

    def add(self, num=1):
        self.done += num
        self.cache.append((self.done, self.timer()))
        if self.need_fresh():
            self.fresh()
            self.last_fresh = self.timer()

    def need_fresh(self):
        if self.timer() - self.last_fresh > self.fresh_interval:
            return True
        else:
            return False

    def speed(self) -> int:
        if len(self.cache) > 1:
            a, b = self.cache[0], self.cache[-1]
            return round((b[0] - a[0]) / (b[1] - a[1]))
        elif self.cache:
            return round(self.done / (self.timer() - self.cache[0][1]))
        else:
            return 0

    def __enter__(self):
        self._fill = len(str(self.total))
        self._end = f'{" " * 10}\r'
        return self

    def __exit__(self, *_):
        if not any(_):
            self.fresh()
            print(flush=True)

    def sig_string(self, percent):
        return f"[{self.sig * int(self.sig_len * percent / 100)}{' ' * (self.sig_len - int(self.sig_len * percent / 100))}]"

    def fresh(self):
        percent = int(100 * self.done / self.total)
        done = f"{self.done}".rjust(self._fill, " ")
        print(
            f"Progress: {done} / {self.total} {percent: >3}% {self.sig_string(percent)} | {self.speed(): >4} units/s",
            end=self._end,
            flush=True,
        )


class SizedTimedRotatingFileHandler(TimedRotatingFileHandler):
    """TimedRotatingFileHandler with maxSize, to avoid files that are too large.

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
    """

    def __init__(
        self,
        filename,
        when="h",
        interval=1,
        backupCount=0,
        maxBytes=0,
        encoding=None,
        delay=False,
        utc=False,
    ):
        """
        Initialize the timed backup file handler.

        :param filename: The name of the log file.
        :param when: The time unit for timed backups, can be "h" (hours) or "d" (days)
        :param interval: The interval for timed backups, with the unit determined by the 'when' parameter
        :param backupCount: The maximum number of backup files to keep
        :param maxBytes: The file size limit before triggering a backup (0 means no limit)
        :param encoding: The encoding of the file
        :param delay: Whether to delay opening the file until the first write
        :param utc: Whether to use UTC time for naming backups
        """
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc)
        self.maxBytes = maxBytes
        self.suffix = "%Y%m%d_%H%M%S"
        self.extMatch = re.compile(
            r"^\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}(\.\w+)?$", re.ASCII
        )

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.
        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        if super().shouldRollover(record):
            return True

        if self.maxBytes > 0:
            if self.stream.tell() >= self.maxBytes:
                return True
        return False

    def doRollover(self):
        """
        Do a rollover, as described by the base class documentation.
        However, also check for the maxBytes parameter and rollover if needed.
        """
        super().doRollover()

        if self.maxBytes > 0:
            self.stream.close()
            self.stream = self._open()


def get_type_default(tp, default=None):
    """Get the default value for a type. {int: 0, float: 0.0, bytes: b"", str: "", list: [], tuple: (), set: set(), dict: {}}"""
    return {
        int: 0,
        float: 0.0,
        bytes: b"",
        str: "",
        list: [],
        tuple: (),
        set: set(),
        dict: {},
    }.get(tp, default)


def func_cmd(function: Callable, run=True, auto_default=False):
    """Handle function with argparse, typing-hint is nessessary.

    Demo::

        def test(str: str, /, int=1, *, list=["d"], float=0.1, set={"f"}, tuple=(1, 2), bool=True, dict={"k": 1}):
            \"\"\"Test demo function.

            Args:
                str (str): str.
                int (int, optional): int. Defaults to 1.
                list (list, optional): list. Defaults to ["d"].
                float (float, optional): float. Defaults to 0.1.
                set (dict, optional): set. Defaults to {"f"}.
                tuple (tuple, optional): tuple. Defaults to (1, 2).
                bool (bool, optional): bool. Defaults to True.
                dict (dict, optional): dict. Defaults to {"k": 1}.
            \"\"\"
            print(locals())

        # raise ValueError if auto_default is False and user do not input nessessary args.
        func_cmd(test, auto_default=False)

        CMD args:

        > python app.py
        ValueError: `str` has no default value.

        > python app.py --str 1 --int 2 --float 1.0 --list "[1,\"a\"]" --tuple "[2,\"b\"]" --set "[1,1,2]" --dict "{\"k\":\"v\"}"
        {'str': '1', 'int': 2, 'list': [1, 'a'], 'float': 1.0, 'set': {1, 2}, 'tuple': (2, 'b'), 'bool': True, 'dict': {'k': 'v'}}

        > python app.py -s 1 -i 2 -f 1.0 -l "[1,\"a\"]" -t "[2,\"b\"]" -s "[1,1,2]" -d "{\"k\":\"v\"}"
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
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(usage=function.__doc__)
    schema = FuncSchema.parse(function, strict=True, fill_default=False)
    seen_shorten = {"h"}
    for key, value in schema.items():
        args = [f"--{key}"]
        for i in range(1, 4):
            if i > len(key):
                break
            short = key[:i]
            if short not in seen_shorten:
                seen_shorten.add(short)
                args.insert(0, f"-{short}")
                break
        kwargs = dict(dest=key, help=str(value))
        if value["type"] in {int, float, str}:
            kwargs["type"] = value["type"]
        else:
            kwargs["type"] = str
        if value["default"] is inspect._empty:
            kwargs["default"] = get_type_default(value["type"])
        else:
            kwargs["default"] = value["default"]
        parser.add_argument(*args, **kwargs)
    parsed = parser.parse_args()
    kwargs = {}
    args = []
    for key, value in schema.items():
        if value["default"] is inspect._empty:
            if auto_default:
                args.append(FuncSchema.convert(getattr(parsed, key), value["type"]))
            else:
                raise ValueError(f"`{key}` has no default value.")
        else:
            kwargs[key] = FuncSchema.convert(getattr(parsed, key), value["type"])
    if run:
        return function(*args, **kwargs)
    else:
        return args, kwargs


def file_import(file_path, names):
    """Import function from file path.

    Demo::
        >>> from pathlib import Path
        >>> file_path = Path(__file__).parent / "utils.py"
        >>> list(file_import(file_path, ["get_hash", "find_jsons"]).keys())
        ['get_hash', 'find_jsons']
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: getattr(module, name) for name in names}


class RotatingFileWriter:
    """RotatingFileWriter class for writing to a file with rotation support.

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
    """

    check_exist_every = 100

    def __init__(
        self,
        path: Union[Path, str],
        max_size=5 * 1024**2,
        max_backups=0,
        encoding="utf-8",
    ):
        if max_backups < 0:
            raise ValueError("max_backups must be greater than -1, 0 for itself.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.max_backups = max_backups
        self.encoding = encoding
        self.file = self.reopen_file()
        self._check_exist_count = self.check_exist_every + 1
        self._rotate_lock = Lock()

    def unlink_file(self):
        self.close_file()
        self.path.unlink(missing_ok=True)

    def close(self):
        return self.close_file()

    def close_file(self):
        file_obj = getattr(self, "file", None)
        if file_obj and not file_obj.closed:
            file_obj.close()
            self.file = None

    def reopen_file(self):
        self.close_file()
        self.file = self.path.open("a")
        return self.file

    def check_exist(self):
        return not (
            self._check_exist_count > self.check_exist_every and not self.path.is_file()
        )

    def rotate(self, new_length):
        with self._rotate_lock:
            if self.need_rotate(new_length):
                if self.max_backups > 0:
                    self.close_file()
                    now = time.strftime("%Y%m%d%H%M%S")
                    for index in range(self.max_backups):
                        suffix = f"{now}_{index}" if index else now
                        target_path = self.path.with_name(f"{self.path.name}.{suffix}")
                        if target_path.is_file():
                            # already rotated
                            continue
                        else:
                            break
                    else:
                        raise RuntimeError(
                            "max_backups is too small for writing too fast"
                        )
                    self.path.rename(target_path)
                    self.reopen_file()
                    self.clean_backups(count=None)
                else:
                    self.file.seek(0)
                    self.file.truncate()

    def need_rotate(self, new_length):
        return self.max_size and self.file.tell() + new_length > self.max_size

    def ensure_file(self, new_length=0):
        if not self.file:
            self.reopen_file()
        elif not self.check_exist():
            self.reopen_file()
        elif self.need_rotate(new_length):
            self.rotate(new_length)

    def backup_path_list(self):
        return list(self.path.parent.glob(f"{self.path.name}.*"))

    def clean_backups(self, count=None):
        """Clean oldest {count} backups, if count is None, it will clean up to max_backups."""
        path_list = self.backup_path_list()
        if path_list:
            if count is None:
                count = len(path_list) - self.max_backups
            if count > 0:
                path_list.sort(key=lambda x: x.stat().st_mtime)
                for deleted, path in enumerate(path_list, 1):
                    path.unlink(missing_ok=True)
                    if deleted >= count:
                        break

    def flush(self):
        self.file.flush()

    def write(self, text: str, flush=False):
        self._check_exist_count += 1
        self.ensure_file(len(text))
        self.file.write(text)
        if flush:
            self.file.flush()

    def print(self, *strings, end="\n", sep=" ", flush=False):
        text = f"{sep.join(map(str, strings))}{end}"
        self.write(text, flush=flush)

    def __del__(self):
        self.close_file()


def test_bg_task():
    async def _test_bg_task():
        async def coro():
            return True

        task = bg_task(coro())
        assert await task is True
        result = (task.done(), len(_bg_tasks))
        assert result == (True, 0), result

    asyncio.get_event_loop().run_until_complete(_test_bg_task())


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


def test():
    test_bg_task()
    test_named_lock()


if __name__ == "__main__":
    __name__ = "morebuiltins.functools"
    test()
    import doctest

    doctest.testmod()

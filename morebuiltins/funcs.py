import ast
import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from functools import partial, wraps
from itertools import chain
from threading import Lock, Semaphore, Timer
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from weakref import WeakSet

__all__ = [
    "lru_cache_ttl",
    "threads",
    "bg_task",
    "NamedLock",
    "FuncSchema",
    "InlinePB",
    "get_type_default",
    "func_cmd",
    "file_import",
    "get_function",
    "to_thread",
    "check_recursion",
    "debounce",
    "async_call",
    "countdown",
    "LineTime",
    "LineProfiler",
    "line_profiler",
]


T = TypeVar("T")


def _ttl_clean(cache, expire):
    for k, v in tuple(cache.items()):
        if v[1] <= expire:
            cache.pop(k, None)


def _cache_key(args, kwargs):
    # exact key is kwargs-order insensitive, so kwargs order does not matter.
    # Unhashable args can not be a dict key and a repr based pseudo key could
    # return a wrong cached value, so signal a cache miss with None.
    raw_key = tuple(chain(args, sorted(kwargs.items())))
    try:
        hash(raw_key)
    except TypeError:
        return None
    return raw_key


def _cache_store(cache, key, result, maxsize, ttl, timer):
    while len(cache) >= maxsize:
        cache.popitem(last=False)
    if ttl is None:
        # no need save expired ts
        cache[key] = (result,)
    else:
        # setitem(key, (result, timer() + ttl))
        cache[key] = (result, timer() + ttl)


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

    Notes:
        - `maxsize <= 0` disables caching and returns the function unchanged, so
          `.cache`/`.ttl_clean` are not set even with `controls=True`.
        - With `controls=True`, `.cache` maps an argument tuple (kwargs sorted by
          name) to `(result,)` or `(result, expire_ts)`; it is not the old
          hash-int-keyed dict, so keys read as tuples.
        - Unhashable arguments are not cached and fall through to the function.

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
    >>> # test kwargs order insensitive
    >>> kw_calls = []
    >>> @lru_cache_ttl(5, controls=True)
    ... def kwfunc(a=0, b=0):
    ...     kw_calls.append((a, b))
    ...     return a + b
    >>> kwfunc(a=1, b=2)
    3
    >>> kwfunc(b=2, a=1)
    3
    >>> len(kwfunc.cache)
    1
    >>> # test unhashable args: not cached, the original function is called
    >>> list_calls = []
    >>> @lru_cache_ttl(5, controls=True)
    ... def listfunc(items):
    ...     list_calls.append(items)
    ...     return sum(items)
    >>> listfunc([1, 2])
    3
    >>> listfunc([1, 2])
    3
    >>> len(list_calls)
    2
    >>> len(listfunc.cache)
    0
    >>> # test maxsize=0: short-circuit to a no-cache decorator
    >>> nocache_calls = []
    >>> @lru_cache_ttl(0)
    ... def nocache(i):
    ...     nocache_calls.append(i)
    ...     return i
    >>> [nocache(1), nocache(1)]
    [1, 1]
    >>> nocache_calls
    [1, 1]
    """
    cache: OrderedDict = OrderedDict()
    move_to_end = cache.move_to_end
    next_clear_ts = timer()

    def ttl_clean(expire):
        _ttl_clean(cache, expire)

    def decorator(func):
        if not maxsize or maxsize < 0:
            # maxsize <= 0 disables caching entirely, also avoiding the
            # KeyError of popping from an empty cache
            return func

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
            key = _cache_key(args, kwargs)
            if key is None:
                return func(*args, **kwargs)
            if key in cache and (expire_cleared is True or now < cache[key][1]):
                # 1. key in cache; 2. cache is valid
                # set newest, return result
                move_to_end(key)
                return cache[key][0]
            # call function
            result = func(*args, **kwargs)
            _cache_store(cache, key, result, maxsize, ttl, timer)
            return result

        if controls:
            # dynamic attributes on the wrapped function, exposed for callers
            # to introspect/clean the decorator state (documented in the doctest)
            wrapped.cache = cache  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]  # pyrefly: ignore[missing-attribute]
            wrapped.ttl_clean = ttl_clean  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]  # pyrefly: ignore[missing-attribute]
        return wrapped

    return decorator


def threads(n: Optional[int] = None, executor_class=None, **kws):
    """Quickly convert synchronous functions to be concurrent. (similar to madisonmay/Tomorrow)

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
    """
    pool = (executor_class or ThreadPoolExecutor)(max_workers=n, **kws)
    tasks: WeakSet = WeakSet()

    def decorator(func) -> Callable[..., Future]:
        @wraps(func)
        def wrapped(*args, **kwargs) -> Future:
            future = pool.submit(func, *args, **kwargs)
            future.add_done_callback(lambda task: tasks.discard(task))
            tasks.add(future)
            return future

        # dynamic attributes on the wrapped function, exposed for callers
        # to inspect the shared executor and in-flight task set (see doctest)
        wrapped.pool = pool  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]  # pyrefly: ignore[missing-attribute]
        wrapped.tasks = tasks  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]  # pyrefly: ignore[missing-attribute]
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


# module level lock guarding NamedLock._SYNC_CACHE check-create-write,
# avoiding two threads each creating a lock for the same name
_SYNC_CACHE_LOCK = Lock()


class NamedLock:
    """Reusable named locks, support for timeouts, support for multiple concurrent locks.

    Warning: if the lock can not be acquired within `timeout`, `__enter__`
    returns None, so `lock` in `as lock` will be None, but the with-body
    still runs - always check `if lock:` inside the body before assuming
    the lock is held.

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
    """

    _SYNC_CACHE: Dict[str, Tuple[Union[Lock, Semaphore], Optional[int]]] = {}
    _ASYNC_CACHE: Dict[str, Tuple[Union[asyncio.Lock, asyncio.Semaphore], Optional[int]]] = {}

    def __init__(self, name: str, default_factory: Callable, timeout: Optional[float] = None):
        self.name = name
        self.default_factory = default_factory
        self.timeout = timeout
        self.lock = None
        # the lock actually acquired by the last successful __enter__/__aenter__,
        # only this one is released on exit
        self._held = None

    @classmethod
    def clear_unlocked(cls):
        """Clear cache entries whose lock is fully released and not waited on.

        Only known types are inspected explicitly: a custom lock is left in the
        cache, because its release state can not be inferred from attributes.
        """
        with _SYNC_CACHE_LOCK:
            for cache in [cls._SYNC_CACHE, cls._ASYNC_CACHE]:
                for name, (lock, initial_value) in list(cache.items()):
                    if isinstance(lock, (Semaphore, asyncio.Semaphore)):
                        # a semaphore is idle when all permits are back and
                        # nobody waits. Compare against the recorded initial
                        # permit count instead of the raw _value.
                        idle = getattr(lock, "_value", None) == initial_value
                        if idle and not cls._semaphore_has_waiters(lock):
                            cache.pop(name, None)
                    elif isinstance(lock, (Lock, asyncio.Lock)):
                        # unlocked means no holder
                        if not lock.locked():
                            cache.pop(name, None)

    @staticmethod
    def _semaphore_has_waiters(lock) -> bool:
        """Check if anyone is waiting on a threading/asyncio Semaphore."""
        # asyncio.Semaphore keeps pending futures in _waiters
        waiters = getattr(lock, "_waiters", None)
        if waiters:
            return True
        # threading.Semaphore waits on its internal Condition's _waiters
        cond = getattr(lock, "_cond", None)
        return bool(cond is not None and getattr(cond, "_waiters", None))

    def __enter__(self):
        # the cache check-create-write must be atomic, or two threads missing
        # the cache at the same time would each create a lock and one would
        # overwrite the other, breaking mutual exclusion
        with _SYNC_CACHE_LOCK:
            if self.name in self._SYNC_CACHE:
                lock, initial_value = self._SYNC_CACHE[self.name]
            else:
                lock = self.default_factory()
                initial_value = getattr(lock, "_value", None)
                self._SYNC_CACHE[self.name] = (lock, initial_value)
        if self.timeout is None:
            # threading.Lock.acquire(timeout=None) raises TypeError, so only
            # pass the kwarg when a timeout was actually given
            acquired = lock.acquire()
        else:
            acquired = lock.acquire(timeout=self.timeout)
        if acquired:
            self.lock = lock
            # remember what we really hold: after a timeout self.lock may still
            # point at a lock held by another context, and __exit__ must not
            # release that one
            self._held = lock
            return self
        return None

    def __exit__(self, *_):
        held, self._held = self._held, None
        if held is not None:
            held.release()

    async def __aenter__(self):
        # no await between the cache check and write, so the check-create-write
        # is atomic inside a single event loop, no extra lock needed there
        if self.name in self._ASYNC_CACHE:
            lock, initial_value = self._ASYNC_CACHE[self.name]
        else:
            lock = self.default_factory()
            initial_value = getattr(lock, "_value", None)
            self._ASYNC_CACHE[self.name] = (lock, initial_value)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.timeout)
            self.lock = lock
            self._held = lock
            return self
        except asyncio.TimeoutError:
            return None

    async def __aexit__(self, *_):
        held, self._held = self._held, None
        if held is not None:
            held.release()


class FuncSchema:
    """Parse the parameters and types required by a function into a dictionary, and convert an incoming parameter into the appropriate type.

    Note: only parameters whose type (annotation, or inferred from the
    default value) is in ALLOW_TYPES are kept; annotations outside
    ALLOW_TYPES are silently dropped.

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
                        raise TypeError(f"Parameter `{param.name}` has no type and no default value.")
                    else:
                        tp: Any = str
                else:
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

    @classmethod
    def to_string(cls, obj, ensure_ascii=False):
        tp = type(obj)
        if isinstance(obj, str):
            return obj
        elif tp in cls.JSON_TYPES:
            if tp in {tuple, set}:
                obj = list(obj)
            return json.dumps(obj, ensure_ascii=ensure_ascii)
        elif tp in cls.ALLOW_TYPES:
            # {int, float}
            return str(obj)
        else:
            raise TypeError(f"Unsupported type: {tp}")


class InlinePB:
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
        self._end = f"{' ' * 10}\r"
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


def countdown(secs: float, interval: Union[float, int] = 1, prefix: str = "Sleeping: ") -> None:
    """A simple countdown timer that prints to the console inline.

    Demo: countdown(5)

    Args:
        secs (float): Total seconds to count down.
        interval (Union[float, int], optional): Update interval in seconds. Defaults to 1.
        prefix (str, optional): Prefix string for the countdown message. Defaults to "Sleeping: ".
    """
    str_interval = str(interval)
    rounded = len(str_interval.split(".", 1)[1]) if "." in str_interval else 0
    while secs > 0:
        print(f"{prefix}{secs:.{rounded}f}", end="\r", flush=True)
        time.sleep(interval)
        secs -= interval
    print(flush=True)


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

    Required parameters (no default value) are only reported as missing when
    the user did not pass them on the command line: argparse uses
    `default=SUPPRESS` for them, so "not passed" can be told from "passed".
    With auto_default=True, missing required parameters are silently filled
    with a zero value of the parameter type ("" for str, 0 for int, ...).

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

        # raise ValueError only if auto_default is False and a required arg
        # is missing from the command line.
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

    Doctest::

        >>> import sys
        >>> def cmd_demo(str: str, /, int=1, *, bool=True):
        ...     return str, int, bool
        >>> _argv = sys.argv
        >>> sys.argv = ["app.py", "--str", "hello", "--int", "2"]
        >>> func_cmd(cmd_demo, run=False, auto_default=False)
        (['hello'], {'int': 2, 'bool': True})
        >>> sys.argv = ["app.py"]
        >>> try:func_cmd(cmd_demo, run=False, auto_default=False)
        ... except ValueError as e: e
        ValueError('`str` has no default value.')
        >>> sys.argv = ["app.py"]
        >>> func_cmd(cmd_demo, run=False, auto_default=True)  # missing required arg is silently filled
        ([''], {'int': 1, 'bool': True})
        >>> sys.argv = _argv
    """
    from argparse import SUPPRESS, ArgumentParser

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
        kwargs: dict = dict(dest=key, help=str(value))
        if value["type"] in {int, float, str}:
            kwargs["type"] = value["type"]
        else:
            kwargs["type"] = str
        if value["default"] is inspect._empty:
            # SUPPRESS keeps the attribute absent when the user did not pass
            # the argument, so "not passed" can be told from "passed"
            kwargs["default"] = SUPPRESS
        else:
            kwargs["default"] = value["default"]
        parser.add_argument(*args, **kwargs)
    parsed = parser.parse_args()
    kwargs = {}
    args = []
    for key, value in schema.items():
        if value["default"] is inspect._empty:
            if hasattr(parsed, key):
                args.append(FuncSchema.convert(getattr(parsed, key), value["type"]))
            elif auto_default:
                # silently fill a zero value of the parameter type
                args.append(FuncSchema.convert(get_type_default(value["type"]), value["type"]))
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
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {name: getattr(module, name) for name in names}
    else:
        return {}


def get_function(entrypoint: str):
    """Get the function object from entrypoint.

    Demo::

        >>> get_function("urllib.parse:urlparse").__name__
        'urlparse'
    """
    module, _, function = entrypoint.partition(":")
    return getattr(importlib.import_module(module), function)


async def to_thread(func, /, *args, **kwargs):
    """Asynchronously run function *func* in a separate thread, same as `asyncio.to_thread` in python 3.9+."""
    func_call = partial(copy_context().run, func, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(None, func_call)


def check_recursion(function: Callable, return_error=False):
    """Check if a function is recursive by inspecting its AST.
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
    """
    try:
        source = inspect.getsource(function)
        tree = ast.parse(source)
        # functions expose __name__; a non-function callable without it fails
        # in inspect.getsource above and is handled by the except below
        func_name = function.__name__  # ty: ignore[unresolved-attribute]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
                return True
        return False
    except Exception as e:
        if return_error:
            return e
        else:
            return None


class LineTime:
    __slots__ = ("line_no", "start_time", "end_time", "code")

    def __init__(self, line_no: int, start_time: float, end_time: Optional[float], code: str):
        self.line_no = line_no
        self.start_time = start_time
        self.end_time = end_time
        self.code = code


class LineProfiler:
    """Line-by-line performance profiler."""

    stdout = sys.stdout

    def __init__(self) -> None:
        self.line_times: List[LineTime] = []
        self.start_time = 0.0
        self.total_time = 0.0
        self.line_cache: Dict[str, list] = {}
        self.filled = 0

    def trace_calls(self, frame: Any, event: str, arg: Any) -> Callable[..., Any]:
        """Trace each line of function calls"""
        current_time = time.perf_counter()
        if self.filled < len(self.line_times):
            self.filled += 1
            last_item = self.line_times[-1]
            if last_item.end_time is None:
                self.line_times[-1] = LineTime(
                    last_item.line_no,
                    last_item.start_time,
                    current_time,
                    last_item.code,
                )
        if event == "line":
            line_no = frame.f_lineno
            filename = frame.f_code.co_filename
            # Get the code content of current line
            try:
                if filename in self.line_cache:
                    lines = self.line_cache[filename]
                else:
                    with open(filename, encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    self.line_cache[filename] = lines
                if lines and line_no <= len(lines):
                    line_content = lines[line_no - 1].rstrip()  # Only remove right spaces, keep left indentation
                    # Store original code and filename for later base indentation calculation
                    if not hasattr(self, "source_lines"):
                        self.source_lines = lines
                        self.target_filename = filename
                else:
                    line_content = "-"
            except Exception:
                line_content = "-"
                self.line_cache.setdefault(filename, [])
            self.line_times.append(LineTime(line_no, current_time, None, line_content))
        return self.trace_calls

    def _get_function_base_indent(self, func_name: str):
        """Get the base indentation level of function definition"""
        if not hasattr(self, "source_lines"):
            return 0
        # Find function definition line
        for line in self.source_lines:
            if f"def {func_name}(" in line:
                # Calculate indentation level of function definition
                return len(line) - len(line.lstrip())
        return 0

    def calculate_and_print_stats(self, func_name: str):
        """Calculate and print statistics for each line"""
        if len(self.line_times) < 2:
            return
        # Calculate base indentation level of function
        base_indent = self._get_function_base_indent(func_name)
        print(f"{'=' * 95}", file=self.stdout)
        print(f"`{func_name}` profiling report:", file=self.stdout)
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Start at {start_time} | Total: {self.total_time * 1000:.3f} ms",
            file=self.stdout,
        )
        # Calculate execution time for each line
        line_durations = []
        for item in self.line_times:
            if item.end_time is None:
                continue
            line_durations.append(
                (
                    item.line_no,  # line number
                    item.end_time - item.start_time,  # duration
                    item.code,  # code content
                )
            )
        # Sort by line number and merge time for same lines
        line_stats: Dict[int, Dict[str, Any]] = {}
        for line_no, duration, code in line_durations:
            if line_no not in line_stats:
                line_stats[line_no] = {
                    "total_time": 0,
                    "count": 0,
                    "code": code,
                    "timestamps": [],
                }
            line_stats[line_no]["total_time"] += duration
            line_stats[line_no]["count"] += 1
        # Print detailed information for each line
        print(f"{'=' * 95}", file=self.stdout)
        print(
            f"{'Line':>6} {'%':>4} {'Total(ms)':>12} {'Count':>8} {'Avg(ms)':>12}  {'Source Code':<40}",
            file=self.stdout,
        )
        print(f"{'-' * 95}", file=self.stdout)
        sorted_lines = sorted(line_stats.items())
        for line_no, stats in sorted_lines:
            total_time_ms = stats["total_time"] * 1000  # convert to milliseconds
            count = stats["count"]
            avg_time_ms = total_time_ms / count if count > 0 else 0
            percentage = int(
                (stats["total_time"] / self.total_time * 100) if self.total_time > 0 else 0
            )  # convert to integer
            # Handle code indentation display
            code = stats["code"]
            # Get current line indentation
            current_indent = len(code) - len(code.lstrip())
            # First level indentation under def (usually 4 spaces)
            first_level_indent = base_indent + 4
            # Only keep spaces relative to first level indentation
            relative_indent = max(0, current_indent - first_level_indent)
            indent_spaces = " " * relative_indent
            code_with_relative_indent = indent_spaces + code.lstrip()
            code_preview = (
                code_with_relative_indent[:38] + ".."
                if len(code_with_relative_indent) > 40
                else code_with_relative_indent
            )
            print(
                f"{line_no:>6} {percentage:>4} {total_time_ms:>12.3f} {count:>8} {avg_time_ms:>12.3f}  {code_preview:<40}",
                file=self.stdout,
            )
        print(f"{'=' * 95}", file=self.stdout, flush=True)


def line_profiler(func: Callable) -> Callable:
    """Decorator to profile a function line-by-line.

        Demo usage:
            >>> import sys, io
            >>> LineProfiler.stdout = io.StringIO()  # Redirect stdout to capture print output
            >>> @line_profiler
            ... def example_function():
            ...     result = 0
            ...     for i in range(10):
            ...         result += i  # Simulate some work
            ...     return result
            >>> example_function()
            45
            >>> output = LineProfiler.stdout.getvalue()
            >>> output.splitlines()[0].startswith("=" * 95) and "`example_function` profiling report:" in output
            True
            >>> LineProfiler.stdout = sys.stdout  # Restore original stdout

    ===============================================================================================
    `example_function` profiling report:
    Start at 2025-07-26 17:09:58 | Total: 1.122 ms
    ===============================================================================================
      Line    %    Total(ms)    Count      Avg(ms)  Source Code
    -----------------------------------------------------------------------------------------------
         3   73        0.825        1        0.825  -
         4    3        0.040       11        0.004  -
         5    4        0.050       10        0.005  -
    ===============================================================================================
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        # Record start time
        start_time = time.perf_counter()
        profiler.start_time = start_time
        # Set tracer
        old_trace = sys.gettrace()
        sys.settrace(profiler.trace_calls)
        try:
            # Execute function
            result = func(*args, **kwargs)
        finally:
            # Restore original tracer
            sys.settrace(old_trace)
            # Record end time
            end_time = time.perf_counter()
            profiler.total_time = end_time - start_time
            # Calculate and print statistics
            # decorator contract is a function, which always carries __name__
            profiler.calculate_and_print_stats(func.__name__)  # ty: ignore[unresolved-attribute]
        return result

    return wrapper


def debounce(wait: float, mode: str = "leading"):
    """Debounce a function, rate-limiting (leading) or delaying (trailing) its execution.

    Two modes are supported, selected by ``mode``:

    - ``"leading"`` (**default**, the original semantics): the first call runs
      immediately and returns its real result; calls arriving within ``wait``
      seconds of that run are dropped and return None.
    - ``"trailing"``: every call resets the timer and only the last call runs,
      ``wait`` seconds after the final one. The wrapper always returns None
      immediately; the debounced call's return value is discarded.

    Trailing-mode semantics:
    - exceptions raised by the debounced function never reach the caller: the
      sync version raises inside the timer thread, the async version inside
      the task created at trigger time.
    - pending calls are dropped silently when the event loop closes or the
      process exits; nothing is flushed to the caller.
    - arguments are read from the last call when the debounced call actually
      runs, earlier calls' arguments are discarded together with their timer.

    Args:
        wait (float): The window length in seconds.
        mode (str): ``"leading"`` or ``"trailing"``; any other value raises
            ValueError. Defaults to ``"leading"`` to keep the pre-existing
            behaviour available without a migration.

    Demo::

        >>> import time
        >>> calls = []
        >>> # wide margins (0.2s window, 0.4-0.5s sleeps) keep this stable on a
        >>> # loaded machine; a tight 0.05s window makes the doctest flaky
        >>> @debounce(0.2)
        ... def lead(value):
        ...     calls.append(value)
        ...     return value
        >>> lead(0)  # window open: runs immediately and returns the real value
        0
        >>> lead(1) is None  # inside the window: dropped
        True
        >>> time.sleep(0.4)
        >>> lead(2)  # window elapsed: runs again
        2
        >>> calls
        [0, 2]
        >>> @debounce(0.2, mode="trailing")
        ... def trail(value):
        ...     calls.append(value)
        ...     return value
        >>> [trail(i) for i in range(5)]  # every call just resets the timer
        [None, None, None, None, None]
        >>> time.sleep(0.5)  # window closed: only the last call runs
        >>> calls[-1]
        4
        >>> async def main():
        ...     seen = []
        ...     @debounce(0.2)
        ...     async def alead(value):
        ...         seen.append(value)
        ...         return value
        ...     first = await alead(0)
        ...     second = await alead(1)
        ...     await asyncio.sleep(0.4)
        ...     third = await alead(2)
        ...     return first, second, third, seen
        >>> asyncio.run(main())
        (0, None, 2, [0, 2])
    """

    if mode not in ("leading", "trailing"):
        raise ValueError(f"mode must be 'leading' or 'trailing', got {mode!r}")

    def decorator(fn):
        # asyncio's version also recognizes legacy @asyncio.coroutine-marked
        # functions, so it is kept deliberately despite the 3.14 deprecation
        is_coro = asyncio.iscoroutinefunction(fn)  # pyrefly: ignore[deprecated]

        if mode == "leading":
            last_call = 0.0

            @wraps(fn)
            async def async_leading(*args, **kwargs):
                nonlocal last_call
                current_time = time.time()
                if current_time - last_call < wait:
                    return None
                last_call = current_time
                return await fn(*args, **kwargs)

            @wraps(fn)
            def sync_leading(*args, **kwargs):
                nonlocal last_call
                current_time = time.time()
                if current_time - last_call < wait:
                    return None
                last_call = current_time
                return fn(*args, **kwargs)

            return async_leading if is_coro else sync_leading

        if is_coro:
            timer_handle = None

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                nonlocal timer_handle
                loop = asyncio.get_running_loop()
                if timer_handle is not None:
                    timer_handle.cancel()
                # schedule the last call, previous pending calls are cancelled
                timer_handle = loop.call_later(
                    wait,
                    lambda: asyncio.ensure_future(fn(*args, **kwargs)),
                )
                # returns None immediately, the result is produced by the task
                return None

            return async_wrapper

        pending_timer = None
        timer_lock = Lock()

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            nonlocal pending_timer
            with timer_lock:
                if pending_timer is not None:
                    pending_timer.cancel()
                pending_timer = Timer(wait, fn, args=args, kwargs=kwargs)
                pending_timer.daemon = True
                pending_timer.start()
            # returns None immediately, the result is produced by the timer
            return None

        return sync_wrapper

    return decorator


async def async_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Automatically call a function asynchronously, whether it's sync or async.

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
    """
    # asyncio's version also recognizes legacy @asyncio.coroutine-marked
    # functions, so it is kept deliberately despite the 3.14 deprecation
    if asyncio.iscoroutinefunction(func):  # pyrefly: ignore[deprecated]
        # It's already a coroutine function, await it directly
        return await func(*args, **kwargs)
    else:
        # It's a sync function, run it in a thread pool
        return await to_thread(func, *args, **kwargs)


def test_bg_task():
    async def _test_bg_task():
        async def coro():
            return True

        task = bg_task(coro())
        if (await task) is not True:
            raise AssertionError(task)
        result = (task.done(), len(_bg_tasks))
        if result != (True, 0):
            raise AssertionError(result)

    asyncio.run(_test_bg_task())


def _test_named_lock_sync():
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock, Semaphore

    def _test1():
        # 0.4s hold against a 0.05s timeout: the margin survives a loaded
        # machine where the waiter threads start late
        with NamedLock("_test1", Lock, timeout=0.05) as lock:
            time.sleep(0.4)
            return bool(lock)

    with ThreadPoolExecutor(10) as pool:
        tasks = [pool.submit(_test1) for _ in range(3)]
        result = [i.result() for i in tasks]
        if result != [True, False, False]:
            raise AssertionError(result)
    if len(NamedLock._SYNC_CACHE) != 1:
        raise AssertionError(NamedLock._SYNC_CACHE)
    NamedLock.clear_unlocked()
    if len(NamedLock._SYNC_CACHE) != 0:
        raise AssertionError(NamedLock._SYNC_CACHE)

    def _test2():
        with NamedLock("_test2", lambda: Semaphore(2), timeout=0.05) as lock:
            time.sleep(0.4)
            return bool(lock)

    with ThreadPoolExecutor(10) as pool:
        tasks = [pool.submit(_test2) for _ in range(3)]
        result = [i.result() for i in tasks]
        if result != [True, True, False]:
            raise AssertionError(result)


def _test_named_lock_async():
    import asyncio

    async def main():
        async def _test1():
            async with NamedLock("_test1", asyncio.Lock, timeout=0.05) as lock:
                await asyncio.sleep(0.4)
                return bool(lock)

        tasks = [asyncio.create_task(_test1()) for _ in range(3)]
        result = [await i for i in tasks]
        if result != [True, False, False]:
            raise AssertionError(result)
        if len(NamedLock._ASYNC_CACHE) != 1:
            raise AssertionError(NamedLock._ASYNC_CACHE)
        NamedLock.clear_unlocked()
        if len(NamedLock._ASYNC_CACHE) != 0:
            raise AssertionError(NamedLock._ASYNC_CACHE)

        async def _test2():
            async with NamedLock("_test2", lambda: asyncio.Semaphore(2), timeout=0.05) as lock:
                await asyncio.sleep(0.4)
                return bool(lock)

        tasks = [asyncio.create_task(_test2()) for _ in range(3)]
        result = [await i for i in tasks]
        if result != [True, True, False]:
            raise AssertionError(result)

    asyncio.run(main())


def test_named_lock():
    _test_named_lock_sync()
    _test_named_lock_async()


def test_async_call():
    async def main():
        def sync_add(a, b):
            time.sleep(0.1)
            return a + b

        async def async_mul(a, b):
            await asyncio.sleep(0.1)
            return a * b

        start = time.time()
        result = await asyncio.gather(async_call(sync_add, 2, 3), async_call(async_mul, 4, 5))
        if result != [5, 20]:
            raise AssertionError(result)
        end = time.time()
        if end - start >= 0.25:
            raise AssertionError(end - start)

    asyncio.run(main())


def test_utils():
    test_bg_task()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "test_bg_task passed")
    test_named_lock()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "test_named_lock passed")
    test_async_call()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "test_async_call passed")


def test():
    global __name__
    __name__ = "morebuiltins.funcs"
    import doctest

    doctest.testmod()
    test_utils()


if __name__ == "__main__":
    test()

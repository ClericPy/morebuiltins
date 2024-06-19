import asyncio
import inspect
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from itertools import chain
from threading import Lock, Semaphore
from typing import Callable, Coroutine, Dict, Optional, OrderedDict, Set, Union
from weakref import WeakSet

__all__ = ["lru_cache_ttl", "threads", "bg_task", "NamedLock", "FuncSchema", "InlinePB"]


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
    """

    ALLOW_TYPES = {int, float, str, tuple, list, set, dict, bool}
    JSON_TYPES = {tuple, set, dict, bool}

    @classmethod
    def parse(cls, function: Callable):
        sig = inspect.signature(function)
        result = {}
        for param in sig.parameters.values():
            if param.annotation is param.empty:
                if param.default is param.empty:
                    continue
                tp = type(param.default)
            else:
                tp = param.annotation
            if tp in cls.ALLOW_TYPES:
                result[param.name] = {"type": tp, "default": param.default}
        return result

    @classmethod
    def convert(cls, obj, target_type):
        if isinstance(obj, str) and target_type in cls.JSON_TYPES:
            return target_type(json.loads(obj))
        else:
            return target_type(obj)


class InlinePB(object):
    """Inline progress bar.

    ```python
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

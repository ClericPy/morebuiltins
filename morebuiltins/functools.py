import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from itertools import chain
from threading import Lock, Semaphore
from typing import Callable, Coroutine, Dict, Optional, OrderedDict, Set, Union
from weakref import WeakSet

__all__ = ["lru_cache_ttl", "threads", "bg_task", "NamedLock"]


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
                time.sleep(0.1)
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
                time.sleep(0.1)
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
                    await asyncio.sleep(0.1)
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
                    await asyncio.sleep(0.1)
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
                time.sleep(0.1)
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
                time.sleep(0.1)
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
                    await asyncio.sleep(0.1)
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
                    await asyncio.sleep(0.1)
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

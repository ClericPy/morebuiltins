import time
from functools import wraps
from itertools import chain
from typing import Optional, OrderedDict, Union


def lru_cache_ttl(
    maxsize,
    ttl: Optional[Union[int, float]] = None,
    setattributes=False,
    auto_clear_expired=True,
    default_timer=time.time,
):
    cache: OrderedDict = OrderedDict()
    move_to_end = cache.move_to_end
    popitem = cache.popitem
    next_clear_ts = default_timer()

    def ttl_clean(expire):
        for k, v in tuple(cache.items()):
            if v[1] < expire:
                cache.pop(k, None)

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            nonlocal next_clear_ts
            now = default_timer()
            if ttl is None:
                cache_cleared = True
            else:
                cache_cleared = auto_clear_expired and now > next_clear_ts
                if cache_cleared:
                    ttl_clean(now - ttl)
                    next_clear_ts = now + ttl
            # key = _make_key(args, kwargs, False)
            key = hash(tuple(chain(args, kwargs.items())))
            if key in cache:
                # cache_cleared means not expired; or is really alive
                if cache_cleared or now < cache[key][1]:
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
                cache[key] = (result, default_timer() + ttl)
            # {key: (result, expired_ts)}
            return result

        if setattributes:
            setattr(wrapped, "cache", cache)
            setattr(wrapped, "ttl_clean", ttl_clean)
        return wrapped

    return decorator

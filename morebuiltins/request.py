import json as _json
import os
import re
import socket
import ssl
import threading
from email.utils import formatdate
from functools import lru_cache, partial
from http import HTTPStatus
from http.client import HTTPResponse
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener, urlopen

__all__ = [
    "req",
    "DomainParser",
    "unparse_qsl",
    "update_url",
    "get_lan_ip",
    "make_response",
    "custom_dns",
    "restore_dns",
]


def _merge_params(url: str, params: Optional[Union[Dict[str, Any], Sequence[Tuple[str, Any]]]]) -> str:
    """Append params to url; None values are dropped, duplicate keys are kept."""
    if not params:
        return url
    if isinstance(params, dict):
        # drop None values like requests
        params = {k: v for k, v in params.items() if v is not None}
        encoded = urlencode(params) if params else ""
    else:
        # key/value sequence: drop (key, None) items, keep duplicates
        pairs = [(k, v) for k, v in params if v is not None]
        encoded = urlencode(pairs, doseq=True) if pairs else ""
    if not encoded:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{encoded}"


def _encode_body(headers: Dict[str, str], data, json, form):
    """Encode data/json/form into a request body and set its Content-Type."""
    # empty data (b"" or {}) is not a body: it must not shadow json/form. A
    # non-empty bytes body keeps the empty Content-Type on purpose, otherwise
    # urllib injects application/x-www-form-urlencoded and servers parse the
    # raw body as a form (losing it); the caller can still override it
    if data:
        if isinstance(data, dict):
            data = urlencode(data, doseq=True).encode("utf-8")
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        headers.setdefault("Content-Type", "")
    elif json:
        data = _json.dumps(json, allow_nan=False).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    elif form:
        data = urlencode(form, doseq=True).encode("utf-8")
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    else:
        # no body at all, never send the empty placeholder value
        data = None
    return data


def _get_response_body(self):
    # fail loudly for foreign instances with no body attached by _request
    return self._response_body


def _get_response_encoding(self):
    charset = re.search(rb"charset[\s\"'=]*([a-zA-Z0-9._-]+)", self.content)
    if charset:
        encoding = charset.group(1).decode("utf-8", "ignore")
    else:
        encoding = "utf-8"
    return encoding


def _get_response_text(self):
    if self.encoding is None:
        encoding = self.get_encoding()
    else:
        encoding = self.encoding
    return self.content.decode(encoding, "replace")


def _get_response_json(self, **kwargs):
    return _json.loads(self.content, **kwargs)


def _get_response_code(self):
    return self.code


def _get_response_ok(self):
    return self.code < 400


def _ensure_response_attrs():
    """Attach the requests-like helpers to HTTPResponse/HTTPError once."""
    if hasattr(HTTPResponse, "get_encoding"):
        return

    for cls in [HTTPResponse, HTTPError]:
        # monkey patch: these attrs exist only after this call, they are not part
        # of the stdlib classes, so every assignment below is intentional
        cls.encoding = None  # ty: ignore[invalid-assignment]
        cls.status_code = property(_get_response_code)  # ty: ignore[invalid-assignment]
        cls.content = property(_get_response_body)  # ty: ignore[invalid-assignment]
        cls.text = property(_get_response_text)  # ty: ignore[invalid-assignment]
        cls.ok = property(_get_response_ok)  # ty: ignore[invalid-assignment]
        cls.json = _get_response_json  # ty: ignore[invalid-assignment]
        cls.get_encoding = _get_response_encoding  # ty: ignore[invalid-assignment]


def _open_response(
    req: Request,
    data,
    urlopen_kwargs: Dict[str, Any],
    build_opener_handlers: Optional[List[Any]],
    proxy: Optional[str],
    verify: bool,
    timeout: Optional[float],
):
    """Send req through urlopen or a custom opener; return (response, body)."""
    # local copy, never mutate the caller's list
    handlers = list(build_opener_handlers) if build_opener_handlers else []
    if proxy and handlers:
        raise ValueError("proxy and build_opener_handlers are mutually exclusive")
    context = urlopen_kwargs.pop("context", None)
    if not verify and context is None:
        # only disable validation on a context created here, never mutate
        # a context provided by the caller
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    try:
        if handlers or proxy:
            if context is not None:
                # make verify/context also work for the opener branch, which
                # otherwise silently ignores urlopen_kwargs
                handlers.append(HTTPSHandler(context=context))
            if proxy:
                handlers.append(ProxyHandler({"http": proxy, "https": proxy}))
            opener = build_opener(*handlers)
            resp = opener.open(req, data, timeout)
        else:
            if context is not None:
                urlopen_kwargs["context"] = context
            resp = urlopen(**urlopen_kwargs)  # noqa: S310  # nosec B310  # URL is supplied by the caller
        with resp as resp:
            body = resp.read()
            response = resp
    except HTTPError as error:
        body = error.read()
        response = error
    return response, body


def _request(
    url: str,
    params: Optional[Union[Dict[str, Any], Sequence[Tuple[str, Any]]]] = None,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Union[bytes, Dict[str, Any]]] = None,
    json: Optional[Any] = None,
    form: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    method: str = "GET",
    verify: bool = True,
    encoding: Optional[str] = None,
    proxy: Optional[str] = None,
    urlopen_kwargs: Optional[Dict[str, Any]] = None,
    build_opener_handlers: Optional[List[Any]] = None,
    **kwargs: Any,
) -> Union[HTTPResponse, HTTPError]:
    url = _merge_params(url, params)
    if headers is None:
        headers = {}
    data = _encode_body(headers, data, json, form)
    req = Request(  # noqa: S310  # URL is supplied by the caller; Request only builds the object
        method=method, url=url, headers=headers, data=data, **kwargs
    )
    if urlopen_kwargs is None:
        urlopen_kwargs = {}
    urlopen_kwargs["url"] = req
    if timeout is not None:
        urlopen_kwargs.setdefault("timeout", timeout)

    _ensure_response_attrs()
    response, body = _open_response(
        req,
        data,
        urlopen_kwargs,
        build_opener_handlers,
        proxy,
        verify,
        timeout,
    )
    # _ensure_response_attrs above attaches these attributes to both classes at runtime
    response._response_body = body  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]
    response.encoding = encoding  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]
    # requests-like API: headers are exposed as a plain dict, not an HTTPMessage
    response.headers = dict(response.headers)  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[bad-argument-type]
    return response


class req:
    """A basic mock for requests, performant albeit simplistic.

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

    Notes:
        - params accepts a dict (None values are dropped) or a sequence of
          (key, value) pairs (None values dropped, duplicate keys kept).
        - verify=False only takes effect when the caller does not pass its own
          `context` through urlopen_kwargs; a caller-provided context is used
          as-is and never mutated.
    """

    RequestErrors = tuple(dict.fromkeys((URLError, TimeoutError, socket.timeout)))

    request = _request
    get = partial(_request, method="GET")
    head = partial(_request, method="HEAD")
    post = partial(_request, method="POST")
    put = partial(_request, method="PUT")
    delete = partial(_request, method="DELETE")
    patch = partial(_request, method="PATCH")
    options = partial(_request, method="OPTIONS")


class DomainParser:
    """Extracts the Second-level domain (SLD) from a provided hostname or URL.

    The constructor does no network IO: it only loads the local cache file.
    When the cache file is missing, it is downloaded lazily on the first
    parse_hostname / parse_url call.

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

    """

    # default cache path, avoid request too many times.
    # you can reset this variable with new path or use $TMP path by default.
    PUBLIC_SUFFIX_CACHE_PATH = Path(gettempdir()) / "public_suffix_list.dat"
    PUBLIC_SUFFIX_API_1 = "https://publicsuffix.org/list/public_suffix_list.dat"
    PUBLIC_SUFFIX_API_2 = "https://github.com/publicsuffix/list/raw/master/public_suffix_list.dat"

    def __init__(self, cache_size=0, public_suffix_file_path=...):
        if public_suffix_file_path is ...:
            public_suffix_file_path = self.PUBLIC_SUFFIX_CACHE_PATH
        self.public_suffix_file_path = Path(public_suffix_file_path)
        # empty dict marks "cache not loaded yet"; no network IO in the
        # constructor: load the local cache only, a missing cache is
        # downloaded lazily on the first parse_hostname call
        self._suffix_trie: dict = {}
        if self.public_suffix_file_path.is_file():
            self.init_local_cache()
        # use lru_cache for performance; both branches resolve to a hostname -> str callable
        self._parse_cached: Callable[..., str] = self._parse
        if cache_size:
            self._parse_cached = lru_cache(maxsize=cache_size)(self._parse)

    def parse_url(self, url: str, default=""):
        return self.parse_hostname(urlparse(url).netloc, default=default)

    def parse_hostname(self, hostname: str, default=""):
        if not self._suffix_trie:
            # lazy download when no local cache was available at init time
            self.init_local_cache()
        return self._parse_cached(hostname=hostname, default=default)

    def _parse(self, hostname: str, default=""):
        parts = hostname.split(".")
        _suffix_trie = self._suffix_trie
        finish = False
        offset = 0
        for index in range(len(parts) - 1, -1, -1):
            part = parts[index]
            try:
                offset = index
                if finish:
                    break
                _suffix_trie = _suffix_trie[part]
            except KeyError:
                if "*" in _suffix_trie:
                    finish = True
                break
        result = parts[offset:]
        if len(result) > 1:
            return ".".join(result)
        else:
            return default

    def init_local_cache(self):
        if not self.public_suffix_file_path.is_file():
            self.public_suffix_file_path.parent.mkdir(parents=True, exist_ok=True)
            headers = {"User-Agent": ""}
            for api in [self.PUBLIC_SUFFIX_API_1, self.PUBLIC_SUFFIX_API_2]:
                try:
                    req = Request(  # noqa: S310  # constant https URL of the public suffix list
                        method="GET", url=api, headers=headers
                    )
                    with urlopen(req, timeout=10) as r:  # noqa: S310  # nosec B310  # constant public-suffix-list URL
                        text = r.read().decode("utf-8")
                        # atomic write: dump to a temp name first, then replace
                        temp_path = self.public_suffix_file_path.with_name(
                            self.public_suffix_file_path.name + f".tmp{os.getpid()}-{threading.get_ident()}"
                        )
                        with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
                            f.write(text)
                        os.replace(temp_path, self.public_suffix_file_path)
                        break
                except URLError:
                    continue
            else:
                raise ValueError("request from public suffix api failed")
        _trie = {}
        with open(self.public_suffix_file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("/"):
                    parts = line.split(".")
                    root = _trie
                    for index in range(len(parts) - 1, -1, -1):
                        part = parts[index]
                        # "!" exception rules are stored without the marker,
                        # so they match like normal suffixes
                        if part[0] == "!":
                            root = root.setdefault(part[1:], {})
                            break
                        else:
                            root = root.setdefault(part, {})
        self._suffix_trie = _trie


def unparse_qsl(qsl, sort=False, reverse=False):
    """Provides the inverse operation of parse_qsl, converting query string lists back into a URL-encoded string."""
    result = []
    if sort:
        qsl = sorted(qsl, key=lambda x: x[0], reverse=reverse)
    for key, value in qsl:
        query_name = quote_plus(key)
        result.append(query_name + "=" + quote_plus(value))
    return "&".join(result)


def update_url(
    url,
    params: Optional[dict] = None,
    replace_kwargs=None,
    sort=False,
    reverse=False,
    **kw_params,
):
    """Organizes the query arguments within a URL to standardize its format.

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
    """
    parsed = urlparse(url)
    if replace_kwargs is None:
        replace_kwargs = {}
    todo = parse_qsl(parsed.query)
    if params:
        kw_params.update(params)
    if kw_params:
        delete_indexes = set()
        replaced_names = set()
        for index, item in enumerate(todo):
            name, _ = item
            if name in kw_params:
                # replace every occurrence of the name, not only the first
                replaced_names.add(name)
                new = kw_params[name]
                if new is None:
                    delete_indexes.add(index)
                else:
                    todo[index] = (name, new)
        if delete_indexes:
            todo = [item for index, item in enumerate(todo) if index not in delete_indexes]
        for k, v in kw_params.items():
            if k not in replaced_names:
                todo.append((k, v))
    sorted_parsed = parsed._replace(query=unparse_qsl(todo, sort=sort, reverse=reverse), **replace_kwargs)
    return urlunparse(sorted_parsed)


def get_lan_ip():
    try:
        # get the local hostname and resolve its LAN IP
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except (OSError, socket.gaierror):
        # best-effort: name resolution may fail on offline hosts
        pass
    return ""


def make_response(
    body: Union[str, bytes, list, dict] = "",
    status: int = 200,
    protocol="HTTP/1.1",
    headers=None,
    encoding="utf-8",
) -> bytes:
    """Generates an HTTP response based on the provided parameters.

    :param body: The response body which can be a string, bytes, list, or dictionary. Default is an empty string.
    :param status: The HTTP status code. Default is 200.
    :param protocol: The HTTP protocol version. Default is "HTTP/1.1".
    :param headers: A dictionary of HTTP response headers. Default is None.
    :param encoding: The encoding to use. Default is "utf-8".
    :return: A byte sequence representing the constructed HTTP response.
    """
    # b'HTTP/1.1 200 OK\r\nDate: Fri, 21 Oct 2022 01:04:34 GMT\r\nContent-Length: 2\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nok'
    if headers:
        headers_string = ""
        for k, v in headers.items():
            headers_string += f"\r\n{k.title()}: {v}"
    else:
        headers_string = ""
    # locale-independent RFC 7231 date, unlike time.strftime
    now = formatdate(usegmt=True)

    if isinstance(body, bytes):
        _body = body
    elif isinstance(body, str):
        _body = body.encode(encoding, errors="replace")
    else:
        _body = _json.dumps(body, default=str, ensure_ascii=False).encode(encoding, errors="replace")
    try:
        http_status = HTTPStatus(status)
        status_line = f"{protocol} {http_status.numerator} {http_status.name}"
    except ValueError:
        # non-standard status code, fall back to an empty reason phrase
        status_line = f"{protocol} {status}"
    head = f"{status_line}\r\nDate: {now}\r\nContent-Length: {len(_body)}{headers_string}\r\n\r\n".encode(
        encoding, errors="replace"
    )
    return head + _body


original_getaddrinfo = socket.getaddrinfo


def restore_dns():
    """Restore socket.getaddrinfo to the original function frozen at import time."""
    socket.getaddrinfo = original_getaddrinfo


def custom_dns(custom: Dict[Union[str, Tuple[str, int]], Tuple[str, int]], thread=True):
    """Patch socket.getaddrinfo to translate hosts/ports through a custom map.

    [WARNING] The patch itself is process-global: it replaces
    socket.getaddrinfo for ALL threads. The ``thread`` argument only controls
    where the translation applies: with ``thread=True`` (default) the map is
    applied only when getaddrinfo is called from the installing thread, and
    calls from other threads fall through unchanged; with ``thread=False``
    the map applies to every thread.

    Installing multiple times chains wrappers (each one falls back to the
    previously installed function); call restore_dns() to restore the
    original function frozen at import time.

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> # translation only applies to the installing thread
    >>> custom_dns({"1.1.1.1": ("127.0.0.1", 80), ("1.1.1.1", 80): ("192.168.0.1", 443)})
    >>> socket.getaddrinfo('1.1.1.1', 80)[0][-1]
    ('192.168.0.1', 443)
    >>> socket.getaddrinfo('1.1.1.1', 8888)[0][-1]
    ('127.0.0.1', 80)
    >>> ThreadPoolExecutor().submit(lambda : socket.getaddrinfo('1.1.1.1', 8888)[0][-1]).result()
    ('1.1.1.1', 8888)
    >>> # translation applies to all threads
    >>> custom_dns({"1.1.1.1": ("127.0.0.1", 80), ("1.1.1.1", 80): ("192.168.0.1", 443)}, thread=False)
    >>> ThreadPoolExecutor().submit(lambda : socket.getaddrinfo('1.1.1.1', 8888)[0][-1]).result()
    ('127.0.0.1', 80)
    >>> restore_dns()
    >>> socket.getaddrinfo('1.1.1.1', 8888)[0][-1]
    ('1.1.1.1', 8888)

    Demo:

        custom_dns(custom={("MY_PROXY_HOST", 80): ("xxxxxxxxx", 43532)})
        print(
            requests.get(
                "https://www.github.com/", proxies={"all": "http://MY_PROXY_HOST"}
            ).text
        )
    """
    # chain to the currently installed function, keep restore_dns() usable
    fallback = socket.getaddrinfo
    if thread:
        thread = threading.current_thread()
    else:
        thread = None

    def wrapper(host, port=80, family=0, type=0, proto=0, flags=0):
        if not thread or thread is threading.current_thread():
            host, port = custom.get((host, port)) or custom.get(host) or (host, port)
        return fallback(host, port, family, type, proto, flags)

    # wrapper keeps the host/port translation untyped on purpose: custom keys
    # may be any hashable host/port pair accepted by the caller
    socket.getaddrinfo = wrapper  # ty: ignore[invalid-assignment]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

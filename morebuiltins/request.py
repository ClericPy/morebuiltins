import json as _json
import re
import socket
import ssl
import threading
import time
from functools import lru_cache, partial
from http import HTTPStatus
from http.client import HTTPResponse
from pathlib import Path
from tempfile import gettempdir
from typing import Dict, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

__all__ = [
    "req",
    "DomainParser",
    "unparse_qsl",
    "update_url",
    "get_lan_ip",
    "make_response",
    "custom_dns",
]


def _request(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    data: Optional[bytes] = None,
    json=None,
    form=None,
    timeout=None,
    method: str = "GET",
    verify=True,
    encoding=None,
    urlopen_kwargs: Optional[dict] = None,
    **kwargs,
):
    if params:
        if "?" in url:
            sep = "&"
        else:
            sep = "?"
        url += f"{sep}{urlencode(params)}"
    if headers is None:
        headers = {}
    if data:
        if isinstance(data, bytes):
            headers.setdefault("Content-Type", "")
        elif isinstance(data, dict):
            # as form
            data = urlencode(data, doseq=True).encode("utf-8")
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    else:
        if json:
            data = _json.dumps(json, allow_nan=False).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        elif form:
            data = urlencode(form, doseq=True).encode("utf-8")
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    req = Request(method=method, url=url, headers=headers, data=data, **kwargs)
    if urlopen_kwargs is None:
        urlopen_kwargs = {}
    urlopen_kwargs["url"] = req
    if timeout is not None:
        urlopen_kwargs.setdefault("timeout", timeout)
    if not verify:
        ctx = urlopen_kwargs.setdefault("context", ssl.create_default_context())
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    def ensure_attrs():
        if hasattr(HTTPResponse, "get_encoding"):
            return True

        def get_body(self):
            return getattr(self, "_response_body", b"")

        def get_encoding(self):
            charset = re.search(rb"charset[\s\"'=]*([a-zA-Z0-9._-]+)", self.content)
            if charset:
                encoding = charset.group(1).decode("utf-8", "ignore")
            else:
                encoding = "utf-8"
            return encoding

        def get_text(self):
            if self.encoding is None:
                encoding = self.get_encoding()
            else:
                encoding = self.encoding
            return self.content.decode(encoding, "replace")

        def get_json(self, **kwargs):
            return _json.loads(self.content, **kwargs)

        def get_code(self):
            return self.code

        def get_ok(self):
            return self.code < 400

        for cls in [HTTPResponse, HTTPError]:
            # monkey patch
            setattr(cls, "encoding", None)
            setattr(cls, "status_code", property(get_code))
            setattr(cls, "content", property(get_body))
            setattr(cls, "text", property(get_text))
            setattr(cls, "ok", property(get_ok))
            setattr(cls, "json", get_json)
            setattr(cls, "get_encoding", get_encoding)

    ensure_attrs()
    response = None
    try:
        with urlopen(**urlopen_kwargs) as resp:
            body = resp.read()
            response = resp
    except HTTPError as error:
        body = error.read()
        response = error
    setattr(response, "_response_body", body)
    setattr(response, "encoding", encoding)
    setattr(response, "headers", dict(response.headers))
    return response


class req:
    """A basic mock for requests, performant albeit simplistic.

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
    """

    RequestErrors = (URLError,)

    request = _request
    get = partial(_request, method="GET")
    head = partial(_request, method="HEAD")
    post = partial(_request, method="POST")
    put = partial(_request, method="PUT")
    delete = partial(_request, method="DELETE")
    patch = partial(_request, method="PATCH")
    options = partial(_request, method="OPTIONS")


class DomainParser(object):
    """Extracts the Second-level domain (SLD) from a provided hostname or URL.

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
    PUBLIC_SUFFIX_API_2 = (
        "https://github.com/publicsuffix/list/raw/master/public_suffix_list.dat"
    )

    def __init__(self, cache_size=0, public_suffix_file_path=...):
        if public_suffix_file_path is ...:
            public_suffix_file_path = self.PUBLIC_SUFFIX_CACHE_PATH
        self.public_suffix_file_path = Path(public_suffix_file_path)
        self.init_local_cache()
        # use lru_cache for performance
        if cache_size:
            self._parse_cached = lru_cache(maxsize=cache_size)(self._parse)
        else:
            self._parse_cached = self._parse

    def parse_url(self, url: str, default=""):
        return self.parse_hostname(urlparse(url).netloc, default=default)

    def parse_hostname(self, hostname: str, default=""):
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
                    req = Request(method="GET", url=api, headers=headers)
                    with urlopen(req, timeout=10) as r:
                        text = r.read().decode("utf-8")
                        self.public_suffix_file_path.write_text(text, encoding="utf-8")
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
                        # disable * while exist !
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
        for index, item in enumerate(todo):
            name, _ = item
            if name in kw_params:
                new = kw_params.pop(name)
                if new is None:
                    delete_indexes.add(index)
                else:
                    todo[index] = (name, new)
        if delete_indexes:
            todo = [
                item for index, item in enumerate(todo) if index not in delete_indexes
            ]
        for k, v in kw_params.items():
            todo.append((k, v))
    sorted_parsed = parsed._replace(
        query=unparse_qsl(todo, sort=sort, reverse=reverse), **replace_kwargs
    )
    return urlunparse(sorted_parsed)


def get_lan_ip():
    try:
        # 获取本机主机名
        hostname = socket.gethostname()
        # 获取本机IP
        return socket.gethostbyname(hostname)
    except Exception:
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
    struct_time = time.gmtime(time.time())
    now = time.strftime("%a, %d %b %Y %H:%M:%S GMT", struct_time)

    if isinstance(body, bytes):
        _body = body
    elif isinstance(body, str):
        _body = body.encode(encoding, errors="replace")
    else:
        _body = _json.dumps(body, default=str, ensure_ascii=False).encode(
            encoding, errors="replace"
        )
    http_status = HTTPStatus(status)
    head = f"{protocol} {http_status.numerator} {http_status.name}\r\nDate: {now}\r\nContent-Length: {len(_body)}{headers_string}\r\n\r\n".encode(
        encoding, errors="replace"
    )
    return head + _body


original_getaddrinfo = socket.getaddrinfo


def custom_dns(custom: Dict[Union[str, Tuple[str, int]], Tuple[str, int]], thread=True):
    """Custom the DNS of socket.getaddrinfo, only effect current thread.

    [WARNING] This will modify the global socket.getaddrinfo.

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> # this only effect current thread
    >>> custom_dns({"1.1.1.1": ("127.0.0.1", 80), ("1.1.1.1", 80): ("192.168.0.1", 443)})
    >>> socket.getaddrinfo('1.1.1.1', 80)[0][-1]
    ('192.168.0.1', 443)
    >>> socket.getaddrinfo('1.1.1.1', 8888)[0][-1]
    ('127.0.0.1', 80)
    >>> ThreadPoolExecutor().submit(lambda : socket.getaddrinfo('1.1.1.1', 8888)[0][-1]).result()
    ('1.1.1.1', 8888)
    >>> # this effect global socket.getaddrinfo
    >>> custom_dns({"1.1.1.1": ("127.0.0.1", 80), ("1.1.1.1", 80): ("192.168.0.1", 443)}, thread=False)
    >>> ThreadPoolExecutor().submit(lambda : socket.getaddrinfo('1.1.1.1', 8888)[0][-1]).result()
    ('127.0.0.1', 80)

    Demo:

        custom_dns(custom={("MY_PROXY_HOST", 80): ("xxxxxxxxx", 43532)})
        print(
            requests.get(
                "https://www.github.com/", proxies={"all": "http://MY_PROXY_HOST"}
            ).text
        )
    """
    if thread:
        thread = threading.current_thread()
    else:
        thread = None

    def wrapper(host, port=80, family=0, type=0, proto=0, flags=0):
        if not thread or thread is threading.current_thread():
            host, port = custom.get((host, port)) or custom.get(host) or (host, port)
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = wrapper


if __name__ == "__main__":
    import doctest

    doctest.testmod()

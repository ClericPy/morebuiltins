import json as _json
import re
import ssl
from functools import lru_cache, partial
from http.client import HTTPResponse
from pathlib import Path
from tempfile import gettempdir
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class req:
    """A simple requests mock, slow but useful.

    >>> import time
    >>> r = req.get("http://httpbin.org/get?a=2", params={"b": "3"})
    >>> r.json()["args"]
    {'a': '2', 'b': '3'}
    >>> r.ok
    True
    >>> r.status_code
    200
    >>> r.text.startswith('{')
    True
    >>> time.sleep(1)
    >>> r = req.post("http://httpbin.org/post?a=2", params={"b": "3"}, data=b"data")
    >>> r.json()["data"]
    'data'
    >>> time.sleep(1)
    >>> r = req.post("http://httpbin.org/post?a=2", params={"b": "3"}, json={"json": "yes"})
    >>> r.json()["json"]
    {'json': 'yes'}
    """

    RequestErrors = (URLError,)

    @staticmethod
    def request(
        url: str,
        params: dict = None,
        headers: dict = None,
        data: bytes = None,
        json=None,
        form=None,
        timeout=None,
        method: str = "GET",
        verify=True,
        encoding=None,
        urlopen_kwargs: dict = None,
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

    @classmethod
    def test(cls):
        import time

        r = req.get("http://httpbin.org/get?a=2", params={"b": "3"})
        assert r.json()["args"] == {"a": "2", "b": "3"}, r.json()
        assert r.ok
        assert r.status_code == 200
        assert r.text.startswith("{")
        time.sleep(1)
        r = req.post("http://httpbin.org/post?a=2", params={"b": "3"}, data=b"data")
        assert (
            r.json()["args"] == {"a": "2", "b": "3"} and r.json()["data"] == "data"
        ), r.json()
        time.sleep(1)
        r = req.post(
            "http://httpbin.org/post?a=2", params={"b": "3"}, json={"json": "yes"}
        )
        assert r.json()["json"] == {"json": "yes"}, r.json()
        time.sleep(1)

    get = partial(request, method="GET")
    head = partial(request, method="HEAD")
    post = partial(request, method="POST")
    put = partial(request, method="PUT")
    delete = partial(request, method="DELETE")
    patch = partial(request, method="PATCH")
    options = partial(request, method="OPTIONS")


class DomainParser(object):
    # default cache path, avoid request too many times
    PUBLIC_SUFFIX_CACHE_PATH = Path(gettempdir()) / "public_suffix_list.dat"
    PUBLIC_SUFFIX_API_1 = "https://publicsuffix.org/list/public_suffix_list.dat"
    PUBLIC_SUFFIX_API_2 = (
        "https://github.com/publicsuffix/list/raw/master/public_suffix_list.dat"
    )

    def __init__(self, lru_cache_size=0, public_suffix_file_path=...):
        if public_suffix_file_path is ...:
            public_suffix_file_path = self.PUBLIC_SUFFIX_CACHE_PATH
        self.public_suffix_file_path = Path(public_suffix_file_path)
        self.init_local_cache()
        # use lru_cache for performance
        if lru_cache_size:
            self._get_fld_cached = lru_cache(maxsize=lru_cache_size)(self._get_fld)
        else:
            self._get_fld_cached = self._get_fld

    def test(self):
        assert self.get_fld("github.com") == "github.com"
        assert self.get_fld("www.github.com") == "github.com"
        assert self.get_fld("www.api.github.com.cn") == "github.com.cn"
        assert self.get_fld("a.b.c.kawasaki.jp") == "c.kawasaki.jp"
        assert self.get_fld("a.b.c.city.kawasaki.jp") == "c.city.kawasaki.jp"
        assert self.get_fld("aaaaaaaaaaaa.bbbbbbbbbb.ccccccccccc") == ""

    def get_fld(self, hostname: str, default=""):
        return self._get_fld_cached(hostname=hostname, default=default)

    def _get_fld(self, hostname: str, default=""):
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


if __name__ == "__main__":
    DomainParser().test()
    # req.test()

import json as _json
import re
import ssl
from functools import lru_cache, partial
from pathlib import Path
from tempfile import gettempdir
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class Config:
    PUBLIC_SUFFIX_CACHE_PATH = Path(gettempdir()) / "public_suffix_list.dat"
    PUBLIC_SUFFIX_API_1 = "https://publicsuffix.org/list/public_suffix_list.dat"
    PUBLIC_SUFFIX_API_2 = (
        "https://github.com/publicsuffix/list/raw/master/public_suffix_list.dat"
    )


class req:
    "A simple requests mock, slow but useful."
    RequestError = (URLError,)

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
            headers.setdefault("Content-Type", "")
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

        def _mock_json(body: bytes):
            return _json.loads(body)

        def _get_text(self):
            if self.encoding is None:
                encoding = self._get_encoding()
            else:
                encoding = self.encoding
            return self.content.decode(encoding)

        def _get_encoding(self):
            charset = re.search(rb"charset[\s\"'=]*([a-zA-Z0-9._-]+)", self.content)
            if charset:
                encoding = charset.group(1).decode("utf-8", "ignore")
            else:
                encoding = "utf-8"
            return encoding

        response = None
        try:
            with urlopen(**urlopen_kwargs) as resp:
                body = resp.read()
                response = resp
        except HTTPError as error:
            body = error.read()
            response = error
        setattr(response, "content", body)
        setattr(response, "status_code", response.code)
        setattr(response, "headers", dict(response.headers))
        setattr(response, "json", partial(_mock_json, body))
        response.encoding = None
        if not hasattr(response, "get_encoding"):
            setattr(
                response.__class__,
                "get_encoding",
                _get_encoding,
            )
        if not hasattr(response, "text"):
            setattr(
                response.__class__,
                "text",
                property(_get_text),
            )
        return response

    @classmethod
    def test(cls):
        import time

        r = req.get("http://httpbin.org/get?a=2", params={"b": "3"})
        assert r.json()["args"] == {"a": "2", "b": "3"}, r.json()
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

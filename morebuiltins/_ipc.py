import asyncio
import sys
import typing
from abc import ABC, abstractmethod

# L: 2**32
# Q: 2**64


class IPCEncoder(ABC):
    @abstractmethod
    def encode(self, raw: typing.Any) -> bytes:
        pass

    @abstractmethod
    def decode(self, raw: bytes) -> typing.Any:
        pass


class JSONEncoder(IPCEncoder):
    _DUMP_KWARGS: typing.Dict = {}
    _LOAD_KWARGS: typing.Dict = {}
    _ENCODING = "utf-8"

    def __init__(self):
        import json

        self._encode = json.dumps
        self._decode = json.loads

    @abstractmethod
    def encode(self, raw: typing.Any) -> bytes:
        return self._encode(raw, **self._DUMP_KWARGS).encode(self._ENCODING)

    @abstractmethod
    def decode(self, raw: bytes) -> typing.Any:
        return self._decode(raw.decode(self._ENCODING), **self._LOAD_KWARGS)


class PickleEncoder(IPCEncoder):
    _DUMP_KWARGS: typing.Dict = {}
    _LOAD_KWARGS: typing.Dict = {}
    _ENCODING = "utf-8"

    def __init__(self):
        import pickle

        self._encode = pickle.dumps
        self._decode = pickle.loads

    @abstractmethod
    def encode(self, raw: typing.Any) -> bytes:
        return self._encode(raw, **self._DUMP_KWARGS)

    @abstractmethod
    def decode(self, raw: bytes) -> typing.Any:
        return self._decode(raw, **self._LOAD_KWARGS)


class SocketLogHandlerEncoder(IPCEncoder):
    _DUMP_KWARGS: typing.Dict = {}
    _LOAD_KWARGS: typing.Dict = {}
    _ENCODING = "utf-8"

    def __init__(self):
        import pickle

        self._encode = pickle.dumps
        self._decode = pickle.loads

    @abstractmethod
    def encode(self, raw: typing.Any) -> bytes:
        return self._encode(raw, **self._DUMP_KWARGS)

    @abstractmethod
    def decode(self, raw: bytes) -> typing.Any:
        return self._decode(raw, **self._LOAD_KWARGS)


class SocketServer(object):
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 9090,
        handler=None,
        server_kwargs: dict = None,
    ):
        if port is None and sys.platform == "win32":
            raise SystemError("not support UDS(unix domain socket) on win32 platform")
        self.host = host
        self.port = port
        self.handler = handler or self.default_handler
        self.server_kwargs = server_kwargs or {}

        self.server: asyncio.base_events.Server = None

    async def default_handler(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            if reader.at_eof():
                break
            elif reader.read:
                pass

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def start(self):
        if self.port:
            self.server = await asyncio.start_server(
                self.handle, host=self.host, port=self.port, **self.server_kwargs
            )
        else:
            self.server = await asyncio.start_unix_server(
                self.handle, path=self.host, **self.server_kwargs
            )

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_errors):
        await self.close()

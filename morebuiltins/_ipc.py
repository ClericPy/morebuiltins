import asyncio
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

# L: 2**32
# Q: 2**64


class IPCEncoder(ABC):
    # HEAD_LENGTH: Package length
    # 1: 256 B, 2: 64 KB, 3: 16 MB, 4: 4 GB, 5: 1 TB, 6: 256 TB
    HEAD_SIZE = 4

    @abstractmethod
    def _dumps(self, raw: Any) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def _loads(self, raw: bytes) -> Any:
        raise NotImplementedError

    def dumps(self, raw: Any) -> bytes:
        result = self._dumps(raw)
        head = len(result).to_bytes(self.HEAD_SIZE)
        return head + result

    def loads(self, raw: Any) -> bytes:
        return self._loads(raw)

    def get_size(self, head: bytes):
        return int.from_bytes(head)


class JSONEncoder(IPCEncoder):
    _DUMP_KWARGS: Dict = {}
    _LOAD_KWARGS: Dict = {}
    _ENCODING = "utf-8"

    def __init__(self):
        import json

        self._encoder = json.dumps
        self._decoder = json.loads
        super().__init__()

    def _dumps(self, raw: Any) -> bytes:
        return self._encoder(raw, **self._DUMP_KWARGS).encode(self._ENCODING)

    def _loads(self, raw: bytes) -> Any:
        return self._decoder(raw.decode(self._ENCODING), **self._LOAD_KWARGS)


class PickleEncoder(IPCEncoder):
    _DUMP_KWARGS: Dict = {}
    _LOAD_KWARGS: Dict = {}
    _ENCODING = "utf-8"

    def __init__(self):
        import pickle

        self._encoder = pickle.dumps
        self._decoder = pickle.loads
        super().__init__()

    def _dumps(self, raw: Any) -> bytes:
        return self._encoder(raw, **self._DUMP_KWARGS)

    def _loads(self, raw: bytes) -> Any:
        return self._decoder(raw, **self._LOAD_KWARGS)


class SocketLogHandlerEncoder(PickleEncoder):
    _DUMP_KWARGS: Dict = {}
    _LOAD_KWARGS: Dict = {}
    _ENCODING = "utf-8"

    def __init__(self):
        super().__init__()

    def _dumps(self, raw: Any) -> bytes:
        return self._encoder(raw, **self._DUMP_KWARGS)

    def _loads(self, raw: bytes) -> Any:
        return self._decoder(raw, **self._LOAD_KWARGS)


class SocketServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8090,
        handler: Optional[Callable] = None,
        encoder: Optional[IPCEncoder] = None,
        connect_kwargs: Optional[dict] = None,
    ):
        if port is None and sys.platform == "win32":
            raise SystemError("not support UDS(unix domain socket) on win32 platform")
        self.host = host
        self.port = port
        self.handler = handler or self.default_handler
        self.encoder: IPCEncoder = encoder or PickleEncoder()
        self.connect_kwargs = connect_kwargs or {}

        self.server: Optional[asyncio.base_events.Server] = None
        self._shutdown_ev: Optional[asyncio.Event] = None

    @staticmethod
    async def default_handler(self: "SocketServer", item: Any):
        print(
            time.strftime("%Y-%m-%d %H:%M:%S"), "[Server] recv:", repr(item), flush=True
        )
        if item == "[shutdown server]":
            await self.close()
        else:
            return self.encoder.dumps(item)

    async def connect_callback(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        head_size = self.encoder.HEAD_SIZE
        need_await = asyncio.iscoroutinefunction(self.handler)
        while self.is_serving() and not reader.at_eof():
            head = await reader.read(head_size)
            if len(head) < head_size:
                break
            content_length = self.encoder.get_size(head)
            # read the whole package
            head = await reader.read(content_length)
            while len(head) < content_length:
                head = head + await reader.read(content_length - len(head))
            item = self.encoder._loads(head)
            result = self.handler(self, item)
            if need_await:
                result = await result
            if isinstance(result, bytes):
                writer.write(result)
                await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def close(self):
        if self.is_serving():
            self.server.close()
            self._shutdown_ev.set()

    async def start(self):
        self._shutdown_ev = asyncio.Event()
        if self.port:
            self.server = await asyncio.start_server(
                self.connect_callback,
                host=self.host,
                port=self.port,
                **self.connect_kwargs,
            )
            await self.server.start_serving()
        else:
            self.server = await asyncio.start_unix_server(
                self.handle, path=self.host, **self.connect_kwargs
            )

    def is_serving(self):
        return self.server and self.server.is_serving()

    async def wait_closed(self):
        if self._shutdown_ev:
            await self._shutdown_ev.wait()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_errors):
        await self.close()


class SocketClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8090,
        encoder: Optional[IPCEncoder] = None,
        connect_kwargs: Optional[dict] = None,
    ):
        self.host = host
        self.port = port
        self.encoder: IPCEncoder = encoder or PickleEncoder()
        self.connect_kwargs = connect_kwargs or {}

    async def __aenter__(self):
        if self.port:
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port, **self.connect_kwargs
            )
        else:
            self.reader, self.writer = await asyncio.open_unix_connection(
                path=self.host, **self.connect_kwargs
            )
        self.lock = asyncio.Lock()
        return self

    async def __aexit__(self, *_):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()

    async def send(self, item: Any):
        assert self.writer, "use `async with`"
        async with self.lock:
            self.writer.write(self.encoder.dumps(item))
            await self.writer.drain()

    async def recv(self) -> Any:
        reader = self.reader
        assert reader, "use `async with`"
        async with self.lock:
            assert not reader.at_eof()
            head = await reader.read(self.encoder.HEAD_SIZE)
            if len(head) < self.encoder.HEAD_SIZE:
                raise ValueError()
            content_length = self.encoder.get_size(head)
            head = await reader.read(content_length)
            while len(head) < content_length:
                head = head + await reader.read(content_length - len(head))
            return self.encoder.loads(head)


async def test_server(host="127.0.0.1", port=8090):
    try:
        async with SocketServer(host=host, port=port, encoder=PickleEncoder()) as s:
            await s.wait_closed()
    except Exception:
        import traceback

        print("ERROR:::::::::::::", traceback.format_exc())
        await asyncio.sleep(2)


async def test_client():
    async with SocketClient(host="127.0.0.1", port=8090, encoder=PickleEncoder()) as c:
        for case in [123, "123", None, {"a"}, ["a"], ("a",), {"a": 1}]:
            await c.send(case)
            response = await c.recv()
            print(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "[Client]",
                repr(case),
                "=>",
                repr(response),
            )
            assert case == response
        await c.send("[shutdown server]")


async def test_ipc():
    # test normal
    task = asyncio.create_task(test_server())
    await test_client()
    await task
    # test unix domain socket
    import platform

    if platform.system() == "Linux":
        print("Test Linux Unix Domain Socket")
        task = asyncio.create_task(test_server("./uds.sock", port=None))
        await test_client()
        await task


if __name__ == "__main__":
    asyncio.run(test_ipc())

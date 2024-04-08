import asyncio
import pickle
import sys
from abc import ABC, abstractmethod
from logging import LogRecord
from logging.handlers import SocketHandler
from typing import Any, Callable, Dict, Literal, Optional, Union

__all__ = [
    "IPCEncoder",
    "JSONEncoder",
    "PickleEncoder",
    "SocketLogHandlerEncoder",
    "SocketServer",
    "SocketClient",
]


class IPCEncoder(ABC):
    """Abstract base class for all encoders; users only need to implement two abstract methods to set up the communication protocol. Note that different header lengths affect the packaging max length."""

    # HEAD_LENGTH: Package length
    # 1: 256 B, 2: 64 KB, 3: 16 MB, 4: 4 GB, 5: 1 TB, 6: 256 TB
    HEAD_SIZE = 4
    BYTEORDER: Literal["little", "big"] = "big"

    @abstractmethod
    def _dumps(self, raw: Any) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def _loads(self, raw: bytes) -> Any:
        raise NotImplementedError

    def dumps(self, raw: Any) -> bytes:
        result = self._dumps(raw)
        head = len(result).to_bytes(self.HEAD_SIZE, self.BYTEORDER)
        return head + result

    def loads(self, raw: Any) -> bytes:
        return self._loads(raw)

    def get_size(self, head: bytes):
        return int.from_bytes(head, self.BYTEORDER)


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

    def __init__(self):
        super().__init__()

    def _dumps(self, raw: Any) -> bytes:
        return pickle.dumps(raw, **self._DUMP_KWARGS)

    def _loads(self, raw: bytes) -> Any:
        return pickle.loads(raw, **self._LOAD_KWARGS)


class SocketLogHandlerEncoder(IPCEncoder):
    "View the test code: morebuiltins\ipc.py:_test_log_ipc"

    _DUMP_KWARGS: Dict = {"protocol": 1}
    _LOAD_KWARGS: Dict = {}

    def __init__(self):
        super().__init__()

    def makePickle(self, d: Union[dict, LogRecord]):
        if isinstance(d, LogRecord):
            msg = d.getMessage()
            d = dict(d.__dict__)
            d["msg"] = msg
            d["args"] = None
            d["exc_info"] = None
            d.pop("message", None)
        return pickle.dumps(d, **self._DUMP_KWARGS)

    def _dumps(self, raw: Union[LogRecord, dict]) -> bytes:
        return self.makePickle(raw)

    def _loads(self, raw: bytes) -> Any:
        return pickle.loads(raw, **self._LOAD_KWARGS)


class SocketServer:
    "View the test code: morebuiltins\ipc.py:_test_pickle_ipc"

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
        if globals().get("print_log"):
            print("[Server] recv:", repr(item), "=>", "send:", repr(item), flush=True)
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
                raise RuntimeError()
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
            await self.server.wait_closed()
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
                self.connect_callback, path=self.host, **self.connect_kwargs
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
                raise RuntimeError()
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


async def test_client(host="127.0.0.1", port=8090):
    async with SocketClient(host=host, port=port, encoder=PickleEncoder()) as c:
        for case in [123, "123", None, {"a"}, ["a"], ("a",), {"a": 1}]:
            await c.send(case)
            response = await c.recv()
            if globals().get("print_log"):
                print("[Client]", "send:", repr(case), "=>", "recv:", repr(response))
            assert case == response, [case, response]
        await c.send("[shutdown server]")


async def _test_pickle_ipc():
    import platform

    if platform.system() == "Linux":
        # test unix domain socket
        print("Test Linux Unix Domain Socket")
        host = "/tmp/uds.sock"
        port = None
        async with SocketServer(host=host, port=port, encoder=PickleEncoder()):
            await test_client(host, port=None)

    # test socket
    host = "127.0.0.1"
    port = 8090
    async with SocketServer(host=host, port=port, encoder=PickleEncoder()):
        await test_client(host="127.0.0.1", port=8090)


async def _test_log_ipc():
    import logging

    host = "127.0.0.1"
    port = 8090
    async with SocketServer(host=host, port=port, encoder=SocketLogHandlerEncoder()):
        # socket logger demo
        # ==================
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.DEBUG)
        h = SocketHandler(host, port)
        h.setLevel(logging.DEBUG)
        logger.addHandler(h)
        logger.info("test socket")
        # ==================

        # ensure test case
        await asyncio.sleep(0.1)
        assert pickle.loads(h.sock.recv(int(1e10))[4:])["name"] == logger.name


def test():
    globals().setdefault("print_log", True)  # local test show logs
    for function in [_test_pickle_ipc, _test_log_ipc]:
        asyncio.get_event_loop().run_until_complete(function())


if __name__ == "__main__":
    test()

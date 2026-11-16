import asyncio
import inspect
import pickle
import socket
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from logging import LogRecord
from logging.handlers import SocketHandler
from typing import Any, Callable, Dict, Literal, Optional, Union, cast

__all__ = [
    "IPCEncoder",
    "JSONEncoder",
    "PickleEncoder",
    "SocketLogHandlerEncoder",
    "SocketServer",
    "SocketClient",
    "find_free_port",
    "is_port_free",
]


class IPCEncoder(ABC):
    """An abstract base class for all encoders; implementing the necessary communication protocol requires only the definition of two abstract methods. Be mindful that varying header lengths will impact the maximum packaging size."""

    # HEAD_SIZE: frame length-prefix size in bytes
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
        """Serialize raw into one frame: HEAD_SIZE-byte length header + encoded body."""
        result = self._dumps(raw)
        head = len(result).to_bytes(self.HEAD_SIZE, self.BYTEORDER)
        return head + result

    def loads(self, raw: Any) -> Any:
        """Decode one frame body; the frame header is NOT stripped here.

        Frame convention: dumps() prepends the header, loads() receives the
        body only (without header). Use get_size() on the header to learn the
        body length beforehand.
        """
        return self._loads(raw)

    def get_size(self, head: bytes):
        return int.from_bytes(head, self.BYTEORDER)


class JSONEncoder(IPCEncoder):
    """JSON text encoder (UTF-8); see IPCEncoder.dumps/loads for the frame convention."""

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
    """Pickle bytes encoder; only use it between trusted peers, pickle can execute arbitrary code on load."""

    _DUMP_KWARGS: Dict = {}
    _LOAD_KWARGS: Dict = {}

    def __init__(self):
        super().__init__()

    def _dumps(self, raw: Any) -> bytes:
        return pickle.dumps(raw, **self._DUMP_KWARGS)

    def _loads(self, raw: bytes) -> Any:
        return pickle.loads(  # noqa: S301  # nosec B301  # IPC wire format: trusted peers only
            raw, **self._LOAD_KWARGS
        )


class SocketLogHandlerEncoder(IPCEncoder):
    """For a practical demonstration, refer to the test code: morebuiltins/ipc.py:_test_ipc_logging.

    Demo::

        async def _test_ipc_logging():
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
                assert pickle.loads(h.sock.recv(100000)[4:])["name"] == logger.name
    And provide a simple implementation for generating logs for coroutine code with Client usage.
    """

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
        return pickle.loads(  # noqa: S301  # nosec B301  # IPC wire format: trusted peers only
            raw, **self._LOAD_KWARGS
        )


class SocketServer:
    """To see an example in action, view the test code: morebuiltins/ipc.py:_test_ipc.

    [SECURITY WARNING]: There is no authentication on the socket protocol, and
    client payloads are deserialized (pickle by default). The "[shutdown server]"
    control message is unauthenticated too: any connected client can shut the
    server down by sending it. Only use it between trusted processes, keep
    host=127.0.0.1 and never expose the server to untrusted networks.

    Demo::

        async def test_client(host="127.0.0.1", port=8090, encoder=None, cases=None):
            async with SocketClient(host=host, port=port, encoder=encoder) as c:
                for case in cases:
                    await c.send(case)
                    response = await c.recv()
                    if globals().get("print_log"):
                        print("[Client]", "send:", repr(case), "=>", "recv:", repr(response))
                    assert case == response or str(case) == response, [case, response]
                await c.send("[shutdown server]")


        async def _test_ipc():
            import platform

            JSONEncoder._DUMP_KWARGS["default"] = str
            for enc, cases in [
                [PickleEncoder, [123, "123", None, {"a"}, ["a"], ("a",), {"a": 1}]],
                [JSONEncoder, [123, "123", None, {"a"}, ["a"], {"a": 1}]],
            ]:
                encoder = enc()
                if platform.system() == "Linux":
                    # test unix domain socket
                    print("Test Linux Unix Domain Socket")
                    host = "/tmp/uds.sock"
                    port = None
                    async with SocketServer(host=host, port=port, encoder=encoder):
                        await test_client(host, port=None, encoder=encoder, cases=cases)

                # test socket
                host = "127.0.0.1"
                port = 8090
                async with SocketServer(host=host, port=port, encoder=encoder):
                    await test_client(host="127.0.0.1", port=8090, encoder=encoder, cases=cases)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 8090,
        handler: Optional[Callable] = None,
        encoder: Optional[IPCEncoder] = None,
        connect_kwargs: Optional[dict] = None,
        start_callback: Optional[Callable] = None,
        end_callback: Optional[Callable] = None,
    ):
        if port is None and sys.platform == "win32":
            raise NotImplementedError("not support UDS(unix domain socket) on win32 platform")
        self.host = host
        self.port = port
        # handler signatures vary (sync or async, any payload), so keep the
        # callable generic; connect_callback inspects coroutine functions
        self.handler: Callable[..., Any] = handler or self.default_handler
        self.encoder: IPCEncoder = encoder or PickleEncoder()
        self.connect_kwargs = connect_kwargs or {}
        self.start_callback = start_callback
        self.end_callback = end_callback

        self.server: Optional[asyncio.base_events.Server] = None
        self._shutdown_ev: Optional[asyncio.Event] = None
        self._default_executor = ThreadPoolExecutor()
        # transports with a running connect_callback, for closing them on 3.12.x
        self._active_writers: set = set()
        # idempotency flags, close() may be called more than once
        self._started = False
        self._end_callback_called = False
        self._executor_shutdown = False

    async def default_handler(self, item: Any):
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
        self._active_writers.add(writer)
        try:
            head_size = self.encoder.HEAD_SIZE
            need_await = inspect.iscoroutinefunction(self.handler)
            while not (self._shutdown_ev and self._shutdown_ev.is_set()) and self.is_serving() and not reader.at_eof():
                try:
                    head = await reader.readexactly(head_size)
                    content_length = self.encoder.get_size(head)
                    # readexactly avoids the EOF busy-spin of a manual read loop
                    head = await reader.readexactly(content_length)
                except asyncio.IncompleteReadError:
                    break
                item = self.encoder._loads(head)
                if need_await:
                    result = await self.handler(item)
                else:
                    # must await the future, or the reply is never written back
                    result = await asyncio.get_running_loop().run_in_executor(
                        self._default_executor, self.handler, item
                    )
                if isinstance(result, bytes):
                    try:
                        writer.write(result)
                        await writer.drain()
                    except (ConnectionResetError, BrokenPipeError):
                        # close() interrupted this in-flight handler
                        break
        finally:
            self._active_writers.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def close(self):
        """Stop the server, run end_callback once and shut down the internal executor.

        Safe to call multiple times; every step is idempotent. Unlike shutdown(),
        it fully tears the server down: end_callback always runs and
        wait_closed() is awaited, no matter when close() is called.
        """
        if self._started:
            # do not gate on is_serving(): shutdown()/a previous close may have
            # stopped accepting connections while transports and wait_closed()
            # still need to be torn down
            self._started = False
            if self._shutdown_ev:
                self._shutdown_ev.set()
            server = self.server
            if server:
                server.close()
                if hasattr(server, "close_clients"):
                    # 3.13+; both methods are missing from the older asyncio stubs,
                    # so the runtime object is reached through Any
                    asyncio_313_server = cast(Any, server)
                    asyncio_313_server.close_clients()
                    asyncio_313_server.abort_clients()
            # 3.12.x has no close_clients; closing every active transport also
            # interrupts handlers blocked on reads, so wait_closed() can
            # return instead of deadlocking on the connection that triggered
            # "[shutdown server]". Idempotent where already closed.
            for writer in list(self._active_writers):
                writer.close()
            if server is not None:
                await server.wait_closed()
        if not self._end_callback_called and self.end_callback:
            self._end_callback_called = True
            if inspect.iscoroutinefunction(self.end_callback):
                await self.end_callback()
            else:
                await asyncio.get_running_loop().run_in_executor(self._default_executor, self.end_callback)
        if not self._executor_shutdown:
            self._executor_shutdown = True
            self._default_executor.shutdown(wait=False)

    async def start(self):
        # allow start() after close(): the executor was shut down and
        # end_callback already ran, both must be usable again
        if self._executor_shutdown:
            self._default_executor = ThreadPoolExecutor()
            self._executor_shutdown = False
        self._end_callback_called = False
        self._started = True
        self._shutdown_ev = asyncio.Event()
        if self.port is not None:
            self.server = await asyncio.start_server(
                self.connect_callback,
                host=self.host,
                port=self.port,
                **self.connect_kwargs,
            )
            await self.server.start_serving()
        else:
            start_unix_server = getattr(asyncio, "start_unix_server", None)
            if start_unix_server:
                self.server = await start_unix_server(self.connect_callback, path=self.host, **self.connect_kwargs)
            else:
                raise RuntimeError("asyncio.start_unix_server is not available")
        if self.start_callback:
            if inspect.iscoroutinefunction(self.start_callback):
                await self.start_callback()
            else:
                await asyncio.get_running_loop().run_in_executor(self._default_executor, self.start_callback)

    def is_serving(self):
        return self.server and self.server.is_serving()

    async def wait_closed(self):
        if self._shutdown_ev:
            await self._shutdown_ev.wait()

    def shutdown(self):
        """Schedule a shutdown from sync code (e.g. a signal handler).

        Unlike close(), it neither triggers end_callback nor awaits
        wait_closed(); it only sets the shutdown event and closes the server
        on the loop thread. Finish the teardown by awaiting close() later.
        """
        if self.server and self.is_serving():
            self.server.get_loop().call_soon_threadsafe(
                lambda: [
                    self._shutdown_ev.set() if self._shutdown_ev else None,
                    self.server.close() if self.server else None,
                ]
            )

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_errors):
        await self.close()


class SocketClient:
    """Client counterpart to SocketServer: use `async with SocketClient(host, port, encoder=...)`,
    then send/recv frames; send/recv of one instance are serialized by an asyncio.Lock."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 8090,
        encoder: Optional[IPCEncoder] = None,
        connect_kwargs: Optional[dict] = None,
    ):
        self.host = host
        self.port = port
        self.encoder: IPCEncoder = encoder or PickleEncoder()
        self.connect_kwargs = connect_kwargs or {}

    async def __aenter__(self):
        if self.port is not None:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port, **self.connect_kwargs)
        else:
            open_unix_connection = getattr(asyncio, "open_unix_connection", None)
            if not open_unix_connection:
                raise RuntimeError("asyncio.open_unix_connection is not available")
            self.reader, self.writer = await open_unix_connection(path=self.host, **self.connect_kwargs)
        self.lock = asyncio.Lock()
        return self

    async def __aexit__(self, *_):
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except OSError:
                pass

    async def send(self, item: Any):
        if not self.writer:
            raise AssertionError("use `async with`")
        async with self.lock:
            self.writer.write(self.encoder.dumps(item))
            await self.writer.drain()

    async def recv(self) -> Any:
        reader = self.reader
        if not reader:
            raise AssertionError("use `async with`")
        async with self.lock:
            if reader.at_eof():
                raise AssertionError()
            try:
                head = await reader.readexactly(self.encoder.HEAD_SIZE)
                body = await reader.readexactly(self.encoder.get_size(head))
            except asyncio.IncompleteReadError:
                raise RuntimeError("connection closed before a complete message") from None
            return self.encoder.loads(body)


def find_free_port(host="127.0.0.1", port=0):
    """Finds and returns an available port number.

    Parameters:
    - host: The host address to bind, default is "127.0.0.1".
    - port: The port number to attempt binding, default is 0 (for OS allocation).

    Returns:
    - If a free port is found, it returns the port number; otherwise, returns None.

    Demo:

    >>> free_port = find_free_port()
    >>> isinstance(free_port, int)
    True
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return s.getsockname()[1]
    except OSError:
        pass


def is_port_free(
    port: int,
    host="127.0.0.1",
    family=socket.AF_INET,
    type=socket.SOCK_STREAM,
    timeout=0.1,
):
    """Checks if a port is free.

    Demo:

    >>> is_port_free(12345)
    True
    """
    with socket.socket(family=family, type=type) as s:
        if timeout:
            s.settimeout(timeout)
        connected = s.connect_ex((host, port))
        if connected == 0:
            return False
        else:
            return True


async def test_client(host="127.0.0.1", port: Optional[int] = 8090, encoder=None, cases=None):
    if cases is None:
        raise ValueError("cases is None")
    async with SocketClient(host=host, port=port, encoder=encoder) as c:
        for case in cases:
            await c.send(case)
            response = await c.recv()
            if globals().get("print_log"):
                print("[Client]", "send:", repr(case), "=>", "recv:", repr(response))
            if not (case == response or str(case) == response):
                raise AssertionError([case, response])
        await c.send("[shutdown server]")


async def _test_ipc():
    import platform

    JSONEncoder._DUMP_KWARGS["default"] = str
    for enc, cases in [
        (PickleEncoder, [123, "123", None, {"a"}, ["a"], ("a",), {"a": 1}]),
        (JSONEncoder, [123, "123", None, {"a"}, ["a"], {"a": 1}]),
    ]:
        encoder = enc()
        if platform.system() == "Linux":
            # test unix domain socket
            print("Test Linux Unix Domain Socket")
            host = "/tmp/uds.sock"  # noqa: S108  # nosec B108  # /tmp path of this Linux UDS self-test
            port = None
            async with SocketServer(host=host, port=port, encoder=encoder):
                await test_client(host, port=None, encoder=encoder, cases=cases)

        # test socket
        host = "127.0.0.1"
        port = 8090
        async with SocketServer(host=host, port=port, encoder=encoder):
            await test_client(host="127.0.0.1", port=8090, encoder=encoder, cases=cases)


async def _test_ipc_logging():
    import logging

    host = "127.0.0.1"
    port = 8090
    if find_free_port(host, port) != port:
        raise AssertionError(f"{host}:{port} is not free")
    async with SocketServer(host=host, port=port, encoder=SocketLogHandlerEncoder()):
        if find_free_port(host, port) is not None:
            raise AssertionError(f"{host}:{port} is unexpectedly free")
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
        if not h.sock:
            raise RuntimeError("socket is None")
        record = pickle.loads(h.sock.recv(100000)[4:])  # noqa: S301  # nosec B301  # IPC wire format: loopback self-test
        if record["name"] != logger.name:
            raise AssertionError(record["name"])
        h.sock.close()


def test():
    import doctest

    doctest.testmod()

    globals().setdefault("print_log", True)  # local test show logs
    for function in [_test_ipc, _test_ipc_logging]:
        asyncio.run(function())


if __name__ == "__main__":
    test()

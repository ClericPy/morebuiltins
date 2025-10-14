import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
import traceback
import typing
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event as SyncEvent
from threading import Thread

from morebuiltins.ipc import SocketLogHandlerEncoder, SocketServer
from morebuiltins.logs import LogHelper, RotatingFileWriter
from morebuiltins.utils import Validator, format_error, read_size, ttime

__all__ = ["LogServer"]


CONNECTED_HANDLERS: typing.Dict[
    tuple, typing.Union[logging.handlers.SocketHandler, logging.NullHandler]
] = {}


class QueueMsg:
    __slots__ = ("name", "record")

    def __init__(self, name: str, record: dict):
        self.name = name
        self.record = record


class DefaultLogSetting:
    formatter = LogHelper.DEFAULT_FORMATTER
    max_size = 10 * 1024**2
    max_backups = 1

    _key_name = "log_setting"


@dataclass
class LogSetting(Validator):
    formatter: logging.Formatter = DefaultLogSetting.formatter
    max_size: int = DefaultLogSetting.max_size
    max_backups: int = DefaultLogSetting.max_backups
    level_specs: list[str] = field(default_factory=list)

    def __eq__(self, other):
        if not isinstance(other, LogSetting):
            return False
        return (
            self.formatter._fmt == other.formatter._fmt
            and str(self.formatter.datefmt) == str(other.formatter.datefmt)
            and self.max_size == other.max_size
            and self.max_backups == other.max_backups
            and self.level_specs == other.level_specs
        )


class LogServer(SocketServer):
    """Log Server for SocketHandler, create a socket server with asyncio.start_server. Update settings of rotation/formatter with extra: {"max_size": 1024**2, "formatter": logging.Formatter(fmt="%(asctime)s - %(filename)s - %(message)s")}

    ### Server demo1:
        start log server in terminal, only collect logs and print to console
        > python -m morebuiltins.cmd.log_server

    ### Server demo2:
        custom options to log to "logs" directory, default rotates at 10MB with 5 backups
        > python -m morebuiltins.cmd.log_server --log-dir=./logs --host 127.0.0.1 --port 8901

    ### Server demo3:
        python code

    ```python
    import asyncio

    from morebuiltins.cmd.log_server import LogServer


    async def main():
        async with LogServer() as ls:
            await ls.wait_closed()


    asyncio.run(main())
    ```

    ### Client demo1:

    ```python
    import logging
    import logging.handlers

    logger = logging.getLogger("client")
    logger.setLevel(logging.DEBUG)
    h = logging.handlers.SocketHandler("127.0.0.1", 8901)
    h.setLevel(logging.DEBUG)
    logger.addHandler(h)
    for _ in range(5):
        logger.info(
            "hello world!",
            extra={
                "max_size": 1024**2,
                "formatter": logging.Formatter(
                    fmt="%(asctime)s - %(filename)s - %(message)s"
                ),
            },
        )
    # [client] 2024-08-10 19:30:07,113 - temp3.py - hello world!
    # [client] 2024-08-10 19:30:07,113 - temp3.py - hello world!
    # [client] 2024-08-10 19:30:07,113 - temp3.py - hello world!
    # [client] 2024-08-10 19:30:07,113 - temp3.py - hello world!
    # [client] 2024-08-10 19:30:07,114 - temp3.py - hello world!
    ```

    ### Client demo2:

    ```python
    from morebuiltins.cmd.log_server import get_logger

    logger = get_logger("dir/test.log")
    # logger = get_logger("dir/test.log", host="localhost", port=8901)
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    # 2025-10-11 01:30:35,151 | DEBUG | log_server.py:416 - Set formatter for logger 'dir/test.log': %(asctime)s | %(levelname)-5s | %(filename)+8s:%(lineno)+3s - %(message)s
    # 2025-10-11 01:30:35,151 | DEBUG | temp.py:  4 - debug
    # 2025-10-11 01:30:35,152 | INFO  | temp.py:  5 - info
    # 2025-10-11 01:30:35,152 | WARN  | temp.py:  6 - warning
    ```

    More docs:
        > python -m morebuiltins.cmd.log_server -h
    """

    STOP_SIG = {"msg": object()}
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8901
    HANDLER_SIGNALS = (2, 15)  # SIGINT, SIGTERM

    def __init__(
        self,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        log_dir=None,
        name="log_server",
        server_log_args=(10 * 1024**2, 5),
        max_queue_size=100000,
        max_queue_buffer=20000,
        log_stream=sys.stderr,
        compress=False,
        shorten_level=False,
        idle_close_time=300,
    ):
        super().__init__(
            host,
            port,
            handler=self.default_handler,
            encoder=SocketLogHandlerEncoder(),
            start_callback=self.start_callback,
            end_callback=self.end_callback,
        )
        self.init_settings_sync(
            name=name,
            shorten_level=shorten_level,
            max_queue_size=max_queue_size,
            max_queue_buffer=max_queue_buffer,
            log_stream=log_stream,
            compress=compress,
            log_dir=log_dir,
            idle_close_time=idle_close_time,
            server_log_args=server_log_args,
        )

    def init_settings_sync(
        self,
        name: str,
        shorten_level=False,
        max_queue_size=100000,
        max_queue_buffer=20000,
        log_stream=sys.stderr,
        compress=False,
        log_dir=None,
        idle_close_time=300,
        server_log_args=(10 * 1024**2, 5),
    ):
        if shorten_level:
            LogHelper.shorten_level()
        self.name = name
        self.log_stream = log_stream
        self.compress = compress
        self._idle_close_time = idle_close_time
        self.log_dir = Path(log_dir).resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(exist_ok=True, parents=True)
        self._server_log_setting = LogSetting(
            max_size=server_log_args[0], max_backups=server_log_args[1]
        )
        self._loop: typing.Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_signals = 0
        self._lines_counter: Counter = Counter()
        self._size_counter: Counter = Counter()
        self._formatters_cache: typing.Dict[str, logging.Formatter] = {}
        self._default_formatter = LogHelper.DEFAULT_FORMATTER
        self._opened_files = typing.cast(typing.Dict[str, RotatingFileWriter], {})
        self.max_queue_size = max_queue_size
        self._write_queue: Queue = Queue(maxsize=max_queue_size)
        self.max_queue_buffer = max_queue_buffer
        self.handle_signals = self.HANDLER_SIGNALS
        for sig in self.HANDLER_SIGNALS:
            signal.signal(sig, self.handle_signal)

    async def __aenter__(self):
        await super().__aenter__()
        self._queue_consumer_task = self.loop.run_in_executor(
            self._default_executor, self.write_queue_consumer
        )
        return self

    async def __aexit__(self, *_errors):
        await asyncio.sleep(0.01)
        await super().__aexit__(*_errors)

    @property
    def loop(self):
        if not self._loop:
            if not self.server:
                raise RuntimeError("server is not started")
            self._loop = self.server.get_loop()
        return self._loop

    async def end_callback(self):
        self._write_queue.put_nowait(QueueMsg(name=self.name, record=self.STOP_SIG))
        await self._queue_consumer_task

    def start_callback(self):
        self.send_log(
            f"started log server on {self.host}:{self.port}, handle_signals={self.handle_signals}, max_queue_size={self.max_queue_size}, max_queue_buffer={self.max_queue_buffer}, log_stream={getattr(self.log_stream, 'name', None)}, compress={self.compress}, log_dir={self.log_dir}"
        )

    def send_log(
        self, msg: str, error: typing.Optional[Exception] = None, level=logging.INFO
    ):
        if error:
            msg = f"{msg} | {format_error(error)}"
        record = {
            "name": self.name,
            "filename": "log_server.py",
            "pathname": "log_server.py",
            "lineno": 0,
            "args": (),
            "msg": msg,
            "levelno": level,
            "levelname": logging.getLevelName(level),
            "exc_info": {},
        }
        record["log_setting"] = self._server_log_setting
        q_msg = QueueMsg(name=self.name, record=record)
        self._write_queue.put(q_msg)

    def get_targets(self, name: str, max_size=5 * 1024**2, max_backups=1):
        targets = []
        if self.log_stream:
            targets.append(self.log_stream)
        elif name == self.name:
            targets.append(sys.stderr)
        if self.log_dir:
            if name in self._opened_files:
                fw = self._opened_files[name]
                if fw.max_size != max_size:
                    fw.max_size = max_size
                if fw.max_backups != max_backups:
                    fw.max_backups = max_backups
                targets.append(fw)
            else:
                try:
                    path = self.log_dir.joinpath(name)
                    # fill .log suffix
                    if not path.suffix:
                        path = path.with_suffix(".log")
                    fw = RotatingFileWriter(
                        path,
                        max_size=max_size,
                        max_backups=max_backups,
                        compress=self.compress,
                    )
                except Exception as e:
                    self.send_log(
                        f"error in get_targets({name!r}, {max_size!r}, {max_backups!r})",
                        e,
                        level=logging.ERROR,
                    )
                targets.append(self._opened_files.setdefault(name, fw))
        return targets

    def write_queue_consumer(self):
        self.send_log("start write_queue_consumer daemon")
        stopped = False
        interval = 30
        last_log_time = time.time()
        while not stopped:
            try:
                if self._shutdown_ev and self._shutdown_ev.is_set():
                    self.send_log("stopping write_worker daemon(shutdown)")
                    stopped = True
                new_lines = {}
                for index in range(self.max_queue_buffer):
                    try:
                        if index == 0:
                            try:
                                q_msg = typing.cast(
                                    QueueMsg, self._write_queue.get(timeout=1)
                                )
                            except Empty:
                                break
                        else:
                            q_msg = self._write_queue.get_nowait()
                        name, record = q_msg.name, q_msg.record
                        if record is self.STOP_SIG:
                            if not stopped:
                                self.send_log("stopping write_worker daemon(signal)")
                                stopped = True
                            continue
                        if "formatter" in record:
                            formatter = record["formatter"]
                            if formatter and isinstance(formatter, logging.Formatter):
                                self._formatters_cache[name] = formatter
                            else:
                                self._formatters_cache.pop(name, None)
                                formatter = self._default_formatter
                        else:
                            formatter = self._formatters_cache.get(
                                name, self._default_formatter
                            )
                        line = formatter.format(
                            logging.LogRecord(level=record.get("levelno", 0), **record)
                        )
                        file_args = {
                            k: record.get(k) for k in {"max_size", "max_backups"}
                        }
                        file_args = {
                            k: v for k, v in file_args.items() if v is not None
                        }
                        if name in new_lines:
                            data = new_lines[name]
                            data["file_args"] = file_args
                            data["lines"].append(line)
                        else:
                            data = {"file_args": file_args, "lines": [line]}
                            new_lines[name] = data
                    except Empty:
                        break
                if new_lines:
                    for name, data in new_lines.items():
                        lines = data["lines"]
                        targets = self.get_targets(name, **data["file_args"])
                        for log_file in targets:
                            try:
                                lines_text = "\n".join(lines)
                                log_file.write(f"{lines_text}\n")
                                log_file.flush()
                            except Exception as e:
                                self.send_log(
                                    f"error in write_worker ({name})",
                                    e,
                                    level=logging.WARNING,
                                )
                        if name != self.name:
                            self._lines_counter[name] += len(data["lines"])
                            self._size_counter[name] += sum(
                                [len(line) for line in data["lines"]]
                            ) + len(data["lines"])
                if self._lines_counter:
                    now = time.time()
                    if now - last_log_time > interval:
                        start = ttime(last_log_time, fmt="%H:%M:%S")
                        end = ttime(now, fmt="%H:%M:%S")
                        last_log_time = now
                        for name in self._lines_counter:
                            try:
                                fw = self._opened_files[name]
                                fw.flush()
                                # check idle close
                                if fw.path.is_file():
                                    mtime = fw.path.stat().st_mtime
                                    if now - mtime > self._idle_close_time:
                                        fw = self._opened_files.pop(name, None)
                                        if fw:
                                            fw.close()
                                        self.send_log(
                                            f"closed idle log file: {name} (last modified: {ttime(mtime)})"
                                        )
                            except KeyError:
                                pass
                        lines_msg = json.dumps(
                            {
                                name: value
                                for name, value in self._lines_counter.most_common(30)
                            },
                            ensure_ascii=False,
                        )
                        size_msg = json.dumps(
                            {
                                name: read_size(value, 1, shorten=True)
                                for name, value in self._size_counter.most_common(30)
                            },
                            ensure_ascii=False,
                        )
                        self.send_log(
                            f"[{start} - {end}] log counter: {sum(self._lines_counter.values())} lines ({lines_msg}), {read_size(sum(self._size_counter.values()), 1, shorten=True)} ({size_msg})"
                        )
                        self._lines_counter.clear()
                        self._size_counter.clear()
            except Exception as e:
                self.send_log("error in write_queue_consumer", e, level=logging.ERROR)
                print(format_error(e), file=sys.stderr, flush=True)
                traceback.print_exc()
                self.shutdown()
                break
        self.send_log("stopped write_queue_consumer daemon")

    async def default_handler(self, record: dict):
        # record demo:
        # {"name": "test_logger", "msg": "test socket logging message", "args": null, "levelname": "INFO", "levelno": 20, "pathname": "/PATH/temp.py", "filename": "temp.py", "module": "temp", "exc_info": null, "exc_text": null, "stack_info": null, "lineno": 38, "funcName": "main", "created": 1723270162.5119407, "msecs": 511.0, "relativeCreated": 102.74338722229004, "thread": 8712, "threadName": "MainThread", "processName": "MainProcess", "process": 19104}
        try:
            if not isinstance(record, dict):
                raise TypeError("item must be a dict")
            self._write_queue.put_nowait(QueueMsg(name=record["name"], record=record))
        except Exception as e:
            self.send_log("error in default_handler", e, level=logging.WARNING)
        finally:
            del record

    def handle_signal(self, sig, frame):
        self._shutdown_signals += 1
        if self._shutdown_signals > 5:
            os._exit(1)
        msg = f"received signal: {sig}, count: {self._shutdown_signals}"
        self.send_log(msg)
        self.shutdown()
        self._write_queue.put(QueueMsg(name=self.name, record=self.STOP_SIG))
        if self._shutdown_ev:
            self.loop.call_soon_threadsafe(self._shutdown_ev.set)

    def __del__(self):
        for f in self._opened_files.values():
            f.close()

    async def run_wrapper(self):
        async with self:
            await self.wait_closed()

    def wait_close_sync(self, timeout=None):
        return self.shutdown_event.wait(timeout=timeout)

    def __enter__(self):
        "Sync entry, start log server in a thread."
        self.shutdown_event = SyncEvent()
        self._thread = Thread(
            target=asyncio.run, args=(self.run_wrapper(),), daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # delay a bit to ensure all logs are processed. or use async with LogServer(...) as ls: ...
        time.sleep(0.01)
        self._write_queue.put_nowait(QueueMsg(name=self.name, record=self.STOP_SIG))
        self.shutdown()
        self._thread.join(timeout=1)
        return self


def clear_handlers():
    for handler in CONNECTED_HANDLERS.values():
        if hasattr(handler, "close"):
            handler.close()


def create_handler(host: str, port: int, level=logging.DEBUG):
    if not CONNECTED_HANDLERS:
        import atexit

        atexit.register(clear_handlers)
        CONNECTED_HANDLERS[("", 0)] = (
            logging.NullHandler()
        )  # dummy to avoid multiple register
    h = logging.handlers.SocketHandler(host, port)
    h.createSocket()
    if not h.sock:
        raise ConnectionError(f"Cannot connect to log server at {host}:{port}")
    h.setLevel(level)
    CONNECTED_HANDLERS[(h.host, h.port)] = h
    return h


def get_logger(
    name: str,
    host: str = LogServer.DEFAULT_HOST,
    port: int = LogServer.DEFAULT_PORT,
    log_level: int = logging.DEBUG,
    socket_handler_level: int = logging.DEBUG,
    formatter: typing.Optional[logging.Formatter] = LogHelper.DEFAULT_FORMATTER,
    shorten_level: bool = True,
    # sys.stderr, sys.stdout, None
    streaming: typing.Optional[typing.TextIO] = None,
    streaming_level: int = logging.DEBUG,
) -> logging.Logger:
    "Get a singleton logger that sends logs to the LogServer."
    if shorten_level:
        LogHelper.shorten_level()
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    for h in logger.handlers:
        if isinstance(h, logging.handlers.SocketHandler):
            host_key = (h.host, h.port)
            if host_key in CONNECTED_HANDLERS and host_key == (host, port):
                # existing handler with same host and port
                if formatter:
                    CONNECTED_HANDLERS[host_key].setFormatter(formatter)
                return logger
    h = create_handler(host, port, level=socket_handler_level)
    logger.addHandler(h)
    if streaming:
        sh = logging.StreamHandler(streaming)
        sh.setLevel(streaming_level)
        logger.addHandler(sh)
    if formatter:
        _style = getattr(formatter, "_style", None)
        if _style:
            fmt = getattr(formatter, "_fmt", None)
        else:
            fmt = getattr(formatter, "_fmt", None)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
        logger.log(
            socket_handler_level,
            f"Set formatter for logger {name!r}: {fmt}",
            extra={"formatter": formatter},
        )
    return logger


async def main():
    import argparse

    parser = argparse.ArgumentParser(usage=(LogServer.__doc__ or "").replace("%", "%%"))
    parser.add_argument("--host", default=LogServer.DEFAULT_HOST)
    parser.add_argument("--port", default=LogServer.DEFAULT_PORT, type=int)
    parser.add_argument(
        "-t",
        "--log-dir",
        default="",
        dest="log_dir",
        help="log dir to save log files, if empty, log to stderr with --log-stream",
    )
    parser.add_argument("--name", default="log_server", help="log server name")
    parser.add_argument(
        "--server-log-args",
        default="10485760,5",
        dest="server_log_args",
        help="max_size,max_backups for log files, default: 10485760,5 == 10MB each log file, 1 name.log + 5 backups",
    )
    parser.add_argument(
        "--max-queue-size",
        default=100000,
        type=int,
        help="max queue size for log queue, log will be in memory queue before write to file",
    )
    parser.add_argument(
        "--max-queue-buffer",
        default=10000,
        type=int,
        help="chunk size of lines before write to file",
    )
    parser.add_argument(
        "--log-stream",
        default="sys.stderr",
        help="log to stream, if --log-stream='' or --log-stream=null will mute the stream log",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="compress log files with gzip",
    )
    parser.add_argument(
        "--origin-level",
        dest="origin_level",
        action="store_true",
        help="use original log level names, not shortened",
    )
    parser.add_argument(
        "--idle-close-time", dest="idle_close_time", type=int, default=300
    )
    args = parser.parse_args()
    stream_choices = {"sys.stdout": sys.stdout, "sys.stderr": sys.stderr}
    log_stream = stream_choices.get(args.log_stream, "null")
    async with LogServer(
        host=args.host,
        port=args.port,
        log_dir=args.log_dir,
        name=args.name,
        server_log_args=tuple(map(int, args.server_log_args.split(","))),
        max_queue_size=args.max_queue_size,
        max_queue_buffer=args.max_queue_buffer,
        log_stream=log_stream,
        compress=args.compress,
        shorten_level=not args.origin_level,
        idle_close_time=args.idle_close_time,
    ) as ls:
        await ls.wait_closed()


def sync_test():
    with LogServer() as ls:
        logger = get_logger("test_logger", host=ls.host, port=ls.port)
        for i in range(5):
            logger.info(f"test socket logging message {i + 1}")


async def async_test():
    async with LogServer() as ls:
        logger = get_logger("test_logger", host=ls.host, port=ls.port)
        for i in range(5):
            logger.info(f"test socket logging message {i + 1}")


def entrypoint():
    # return asyncio.run(main())
    return asyncio.run(async_test())
    # return sync_test()


if __name__ == "__main__":
    entrypoint()

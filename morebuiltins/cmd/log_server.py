import asyncio
import base64
import io
import ipaddress
import json
import logging
import logging.handlers
import os
import pickle
import re
import shutil
import signal
import sys
import threading
import time
import traceback
import typing
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event as SyncEvent
from threading import Thread

from morebuiltins.funcs import to_thread
from morebuiltins.ipc import SocketLogHandlerEncoder, SocketServer
from morebuiltins.logs import LogHelper, RotatingFileWriter
from morebuiltins.utils import Validator, format_error, read_size, ttime

__all__ = ["LogServer", "get_logger", "create_handler"]


class QueueMsg:
    __slots__ = ("name", "record")

    def __init__(self, name: str, record: dict):
        self.name = name
        self.record = record


class STOP_SIG(QueueMsg):
    pass


class _FormatterUnpickler(pickle.Unpickler):
    """Restricted unpickler for formatter payloads from untrusted log clients.

    Only logging's builtin formatter classes are allowed, so a malicious
    payload cannot execute arbitrary code while unpickling.
    """

    _allowed = {
        ("logging", "Formatter"),
        ("logging", "PercentStyle"),
        ("logging", "StrFormatStyle"),
        ("logging", "StringTemplateStyle"),
        # StringTemplateStyle holds a string.Template for style="$"
        ("string", "Template"),
        # protocol 0/1 pickles instances via copy_reg._reconstructor; its cls
        # argument is still a GLOBAL and goes through find_class()
        ("copy_reg", "_reconstructor"),
        # protocol 0/1 reduce() references object as the base class
        ("__builtin__", "object"),
        ("builtins", "object"),
    }

    def find_class(self, module, name):
        if (module, name) in self._allowed:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"forbidden class in formatter payload: {module}.{name}")


class DefaultLogSetting:
    formatter: logging.Formatter = LogHelper.DEFAULT_FORMATTER
    max_size: int = 10 * 1024**2
    max_backups: int = 1

    _key_name = "log_setting"

    # log server options
    host: str = "127.0.0.1"
    port: int = 8901
    log_dir: typing.Optional[str] = None
    max_queue_size: int = 100000
    max_queue_buffer: int = 20000
    handler_signals: tuple = (2, 15)  # SIGINT, SIGTERM
    log_stream: typing.Optional[typing.TextIO] = sys.stderr
    compress: bool = False
    shorten_level: bool = False
    idle_close_time: int = 60


@dataclass
class LogSetting(Validator):
    """Log setting for a logger, with a lossless asdict/from_dict round-trip.

    Demo::

        >>> from dataclasses import asdict
        >>> setting = LogSetting(level_specs=["Level 13"])
        >>> setting.level_specs
        [13]
        >>> LogSetting.from_dict(**asdict(setting)).level_specs
        [13]
        >>> LogSetting(level_specs=["debug", "Warning"]).level_specs
        [10, 30]
    """

    formatter: logging.Formatter = DefaultLogSetting.formatter
    max_size: int = DefaultLogSetting.max_size
    max_backups: int = DefaultLogSetting.max_backups
    level_specs: list = typing.cast(typing.List[int], field(default_factory=list))
    create_time: str = field(default_factory=ttime)

    @property
    def fmt(self) -> str:
        return getattr(self.formatter, "_fmt", "")

    @property
    def datefmt(self) -> str:
        return getattr(self.formatter, "datefmt", "")

    def __post_init__(self):
        formatter = self.formatter
        if isinstance(formatter, str):
            # base64 formatter
            self.formatter = self.pickle_from_base64(formatter)
        elif isinstance(formatter, logging.Formatter):
            pass
        else:
            self.formatter = DefaultLogSetting.formatter
        for index, level in enumerate(self.level_specs):
            if isinstance(level, int):
                continue
            level = str(level)
            if level not in logging._nameToLevel:
                # level names are case-insensitive: retry with the upper-cased name
                upper = level.upper()
                if upper in logging._nameToLevel:
                    level = upper
                elif re.match(r"^Level \d+$", level, re.IGNORECASE):
                    # e.g. "Level 13" produced by level_to_name for custom levels
                    self.level_specs[index] = int(level.split()[-1])
                    continue
                else:
                    raise ValueError(f"level_specs[{index}] invalid log level name: {level}")
            self.level_specs[index] = logging._nameToLevel[level]
        super().__post_init__()

    @classmethod
    def get_default(cls):
        return cls()

    @staticmethod
    def pickle_to_base64(obj) -> str:
        return base64.b64encode(pickle.dumps(obj)).decode("utf-8")

    @staticmethod
    def pickle_from_base64(data: str):
        # restricted unpickler: formatter payloads come from untrusted clients
        fp = io.BytesIO(base64.b64decode(data.encode("utf-8")))
        return _FormatterUnpickler(fp).load()

    @classmethod
    def from_dict(cls, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in cls.__annotations__}
        return cls(**kwargs)

    @staticmethod
    def level_to_name(level: int):
        levelToName = {
            logging.CRITICAL: "CRITICAL",
            logging.ERROR: "ERROR",
            logging.WARNING: "WARNING",
            logging.INFO: "INFO",
            logging.DEBUG: "DEBUG",
            logging.NOTSET: "NOTSET",
        }
        return levelToName.get(level, f"Level {level}")

    def to_dict_with_meta(self) -> dict:
        meta: dict = {
            "create_time": self.create_time,
            "fmt": self.fmt,
            "datefmt": self.datefmt,
        }
        meta.update(asdict(self))
        # base64 formatter
        meta["formatter"] = self.pickle_to_base64(self.formatter)
        # int to str
        meta["level_specs"] = [self.level_to_name(level) for level in self.level_specs]
        return meta

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

    def __repr__(self):
        return str(self)

    def __str__(self):
        fmt = getattr(self.formatter, "_fmt", "None") or ""
        datefmt = getattr(self.formatter, "datefmt", "")
        if datefmt:
            fmt = f"{fmt}+{datefmt}"
        return f"LogSetting({fmt}, max_size={read_size(self.max_size)}, max_backups={self.max_backups}, level_specs={self.level_specs})"


class LogServer(SocketServer):
    """Log server for SocketHandler, create a socket server with asyncio.start_server. Custom formatter or rotation strategy with extra in log record.

    [SECURITY WARNING]: There is no authentication on the socket protocol, and
    every incoming frame is decoded as a bare pickle (arbitrary code on load);
    the restricted unpickler only guards formatter settings nested in the
    "log_setting" payload. Binding a non-loopback host is therefore refused at
    construction: run the server on 127.0.0.1/::1 and forward logs through a
    trusted channel (SSH tunnel, trusted subnet gateway). This guards against
    misuse, not against a malicious client that can already reach the port.

    [WARNING]: Ensure your log msg is "" if you only want to update settings, or the msg will be skipped.

    logger.info("", extra={"log_setting": {"formatter": formatter, "max_size": 1024**2, "level_specs": [logging.ERROR]}})


    ### Server demo1:
        start log server in terminal, only collect logs and print to console
        > python -m morebuiltins.cmd.log_server

    ### Server demo2:
        custom options to log to "logs" directory, default rotates at 10MB with 1 backup, no log_stream, enable compress
        > python -m morebuiltins.cmd.log_server --log-dir=./logs --host 127.0.0.1 --port 8901 --log-stream=None --compress

    ### Server demo3:
        python code

    ```python
    # Server side
    import asyncio

    from morebuiltins.cmd.log_server import LogServer


    async def main():
        async with LogServer() as ls:
            await ls.wait_closed()


    asyncio.run(main())
    ```

    ### Client demo1:

    ```python
    # Client side(no dependency on morebuiltins)
    import logging
    import logging.handlers

    logger = logging.getLogger("client")
    logger.setLevel(logging.DEBUG)
    h = logging.handlers.SocketHandler("127.0.0.1", 8901)
    h.setLevel(logging.DEBUG)
    logger.addHandler(h)
    logger.info(
        "",
        extra={
            "log_setting": {
                "max_size": 1024**2,
                "formatter": logging.Formatter(
                    fmt="%(asctime)s - %(filename)s - %(message)s"
                ),
                "level_specs": [logging.ERROR],
            }
        },
    )
    for _ in range(5):
        logger.info("hello world!")

    # 2024-08-10 19:30:07,113 - temp3.py - hello world!
    # 2024-08-10 19:30:07,113 - temp3.py - hello world!
    # 2024-08-10 19:30:07,113 - temp3.py - hello world!
    # 2024-08-10 19:30:07,113 - temp3.py - hello world!
    # 2024-08-10 19:30:07,114 - temp3.py - hello world!
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

    # max concurrently opened log files, least recently used ones get closed
    MAX_OPENED_FILES = 256
    # timeout to wait for the queue consumer before closing files on shutdown
    CONSUMER_JOIN_TIMEOUT = 30

    @staticmethod
    def _is_loopback(host: str) -> bool:
        """Check whether a bind host stays on the loopback interface."""
        if host in ("localhost",):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def __init__(
        self,
        host=DefaultLogSetting.host,
        port=DefaultLogSetting.port,
        log_dir=DefaultLogSetting.log_dir,
        name="log_server",
        max_size=DefaultLogSetting.max_size,
        max_backups=DefaultLogSetting.max_backups,
        max_queue_size=DefaultLogSetting.max_queue_size,
        max_queue_buffer=DefaultLogSetting.max_queue_buffer,
        log_stream=DefaultLogSetting.log_stream,
        compress=DefaultLogSetting.compress,
        shorten_level=DefaultLogSetting.shorten_level,
        idle_close_time=DefaultLogSetting.idle_close_time,
    ):
        if not self._is_loopback(host):
            # the frame decoder is a bare pickle, so a reachable port is remote
            # code execution; refuse instead of trusting the operator's firewall
            raise ValueError(
                f"refusing to listen on non-loopback host {host!r}: the socket protocol has no "
                "authentication and every incoming frame is unpickled; bind 127.0.0.1/::1 and "
                "forward logs through a trusted channel"
            )
        super().__init__(
            host,
            port,
            handler=self.default_handler,
            encoder=SocketLogHandlerEncoder(),
            start_callback=self.start_callback,
            end_callback=self.end_callback,
        )
        self._init_settings(
            name=name,
            shorten_level=shorten_level,
            max_queue_size=max_queue_size,
            max_queue_buffer=max_queue_buffer,
            log_stream=log_stream,
            compress=compress,
            log_dir=log_dir,
            idle_close_time=idle_close_time,
            max_size=max_size,
            max_backups=max_backups,
        )

    def _init_settings(
        self,
        name: str,
        shorten_level=True,
        max_queue_size=DefaultLogSetting.max_queue_size,
        max_queue_buffer=DefaultLogSetting.max_queue_buffer,
        log_stream=DefaultLogSetting.log_stream,
        compress=DefaultLogSetting.compress,
        log_dir=DefaultLogSetting.log_dir,
        idle_close_time=DefaultLogSetting.idle_close_time,
        max_size=DefaultLogSetting.max_size,
        max_backups=DefaultLogSetting.max_backups,
    ):
        if shorten_level:
            LogHelper.shorten_level()
        self.name = name
        self.log_stream = log_stream if hasattr(log_stream, "write") else None
        self.compress = compress
        self._idle_close_time = idle_close_time
        self.log_dir = Path(log_dir).resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(exist_ok=True, parents=True)
            self.setting_path: typing.Optional[Path] = self.log_dir.joinpath(f"{self.name}_settings.jsonl")
        else:
            self.setting_path = None
        self._server_log_setting = LogSetting(max_size=max_size, max_backups=max_backups)
        self._loop: typing.Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_signals = 0
        self._lines_counter: Counter = Counter()
        self._size_counter: Counter = Counter()
        self._default_formatter = LogHelper.DEFAULT_FORMATTER
        # OrderedDict keeps write order for LRU eviction of idle files
        self._opened_files = typing.cast(typing.OrderedDict[str, RotatingFileWriter], OrderedDict())
        self.max_queue_size = max_queue_size
        self._write_queue: Queue = Queue(maxsize=max_queue_size)
        self.max_queue_buffer = max_queue_buffer
        self.handle_signals = DefaultLogSetting.handler_signals
        if threading.current_thread() is threading.main_thread():
            # signal.signal raises ValueError outside the main thread
            for sig in self.handle_signals:
                signal.signal(sig, self.handle_signal)
        self._log_settings = self.load_settings()
        if self.name in self._log_settings:
            self._log_settings[self.name] = self._server_log_setting
        self.send_log("", init_setting=True)

    def load_settings(self):
        result = typing.cast(typing.Dict[str, LogSetting], {})
        if not (self.setting_path and self.setting_path.is_file()):
            return result
        bad_lines = 0
        try:
            with self.setting_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    # one bad line must not abort the whole load
                    try:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        name = data["name"]
                        setting = LogSetting.from_dict(**data["setting"])
                        result[name] = setting
                    except Exception as e:
                        bad_lines += 1
                        self.send_log(
                            f"skip bad setting line {line_no} in {self.setting_path}: {e!r}",
                            level=logging.WARNING,
                        )
            self.send_log(f"Loaded log settings from {self.setting_path}, {len(result)} items")
            if bad_lines:
                self.send_log(
                    f"{bad_lines} bad setting lines skipped in {self.setting_path}",
                    level=logging.WARNING,
                )
        except Exception as e:
            self.send_log(f"Failed to load log settings from {self.setting_path}: {e}")
        return result

    async def __aenter__(self):
        await super().__aenter__()
        self._queue_consumer_task = self.loop.run_in_executor(self._default_executor, self.write_queue_consumer)
        return self

    async def __aexit__(self, *_errors):
        await asyncio.sleep(0.01)
        if not self.is_serving():
            # close() may already have run (e.g. from handle_signal) or the
            # server never served; join the consumer before close_opened_files
            # so no in-flight write races with closing its file. If
            # end_callback already stopped it, _stop_queue_consumer returns
            # immediately instead of enqueuing a second STOP_SIG.
            await self._stop_queue_consumer()
        await super().__aexit__(*_errors)
        await to_thread(self.close_opened_files)

    async def _stop_queue_consumer(self, timeout: typing.Optional[float] = None):
        """Stop the queue consumer: enqueue STOP_SIG unless it already exited, then join it."""
        if timeout is None:
            timeout = self.CONSUMER_JOIN_TIMEOUT
        task = getattr(self, "_queue_consumer_task", None)
        if task is not None and task.done():
            # consumer already stopped, another STOP_SIG would only linger
            return
        try:
            self._write_queue.put_nowait(STOP_SIG)
        except Full:
            # queue is full, the consumer still exits via _shutdown_ev; avoid
            # send_log here because it enqueues into the same full queue
            print("could not enqueue STOP_SIG: queue is full", file=sys.stderr, flush=True)
        if task is None:
            return
        deadline = time.time() + timeout
        while not task.done() and time.time() < deadline:
            await asyncio.sleep(0.05)
        if not task.done():
            self.send_log(
                f"queue consumer did not stop within {timeout}s, close files anyway",
                level=logging.WARNING,
            )

    async def close(self):
        await super().close()
        # unblock wait_close_sync once the server is fully closed
        if getattr(self, "shutdown_event", None) and not self.shutdown_event.is_set():
            self.shutdown_event.set()

    @staticmethod
    def default_settings():
        return DefaultLogSetting

    @property
    def loop(self):
        if not self._loop:
            if not self.server:
                raise RuntimeError("server is not started")
            self._loop = self.server.get_loop()
        return self._loop

    # SocketServer stores these hooks as Optional[Callable] instance attributes;
    # LogServer supplies them as methods and hands the bound methods to the
    # parent, so the narrower method type is the intended override
    async def end_callback(self):  # pyrefly: ignore[bad-override-mutable-attribute]
        task = getattr(self, "_queue_consumer_task", None)
        if task is None:
            return
        if not task.done():
            try:
                self._write_queue.put_nowait(STOP_SIG)
            except Full:
                # queue full, the consumer still exits via _shutdown_ev
                pass
        try:
            await asyncio.wait_for(asyncio.shield(task), self.CONSUMER_JOIN_TIMEOUT)
        except asyncio.TimeoutError:
            # a stuck consumer must not hang close() forever, or wait_close_sync
            # would never see shutdown_event; report and continue shutting down
            self.send_log(
                f"queue consumer did not stop within {self.CONSUMER_JOIN_TIMEOUT}s, closing anyway",
                level=logging.WARNING,
            )

    def start_callback(self):  # pyrefly: ignore[bad-override-mutable-attribute]  # see end_callback
        self.send_log(
            f"started log server on {self.host}:{self.port}, handle_signals={self.handle_signals}, max_queue_size={self.max_queue_size}, max_queue_buffer={self.max_queue_buffer}, log_stream={getattr(self.log_stream, 'name', None)}, compress={self.compress}, log_dir={self.log_dir}, setting={self._server_log_setting}",
        )

    def send_log(
        self,
        msg: str,
        error: typing.Optional[Exception] = None,
        level=logging.INFO,
        init_setting: bool = False,
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
            "levelname": LogSetting.level_to_name(level),
            "exc_info": {},
        }
        if init_setting:
            record[DefaultLogSetting._key_name] = asdict(self._server_log_setting)
        q_msg = QueueMsg(name=self.name, record=record)
        try:
            self._write_queue.put_nowait(q_msg)
        except Full:
            # never raise from a logging helper: send_log runs in signal
            # handlers and the consumer, a full queue must only drop the record
            print(msg, file=sys.stderr, flush=True)

    def _reuse_opened_writer(self, name, max_size, max_backups):
        """Move an open writer to the LRU tail and refresh its rotation limits."""
        fw = self._opened_files[name]
        # mark as recently used for LRU eviction
        self._opened_files.move_to_end(name)
        if fw.max_size != max_size:
            fw.max_size = max_size
        if fw.max_backups != max_backups:
            fw.max_backups = max_backups
        return fw

    def _open_writer(self, name, max_size, max_backups):
        """Validate a log name, open its writer and register it for LRU eviction."""
        if not name or name in (".", ".."):
            raise ValueError(f"invalid log name: {name!r}")
        # get_targets only calls this inside `if self.log_dir`, so log_dir is
        # set here; the cast keeps the None path unchanged (still AttributeError)
        log_dir = typing.cast(Path, self.log_dir)
        path = log_dir.joinpath(name)
        # fill .log suffix
        if not path.suffix:
            path = path.with_suffix(".log")
        # sub-directories are allowed, but the resolved path must
        # stay inside log_dir (rejects absolute paths, ".." escapes
        # and symlink tricks)
        log_dir = log_dir.resolve()
        resolved = path.resolve()
        if resolved == log_dir or not resolved.is_relative_to(log_dir):
            raise ValueError(f"invalid log name: {name!r}")
        fw = RotatingFileWriter(
            path,
            max_size=max_size,
            max_backups=max_backups,
            compress=self.compress,
        )
        self._opened_files[name] = fw
        # never evict a file this batch is still going to write to
        self._evict_opened_files(protect={name})
        return fw

    def get_targets(
        self,
        name: str,
        max_size=DefaultLogSetting.max_size,
        max_backups=DefaultLogSetting.max_backups,
        level_spec: typing.Optional[int] = None,
    ):
        targets = []
        if name == self.name:
            # server log always to stderr
            targets.append(sys.stderr)
        elif self.log_stream:
            if not level_spec:
                # spec log only to file
                targets.append(self.log_stream)
        if self.log_dir:
            if name in self._opened_files:
                targets.append(self._reuse_opened_writer(name, max_size, max_backups))
            else:
                try:
                    targets.append(self._open_writer(name, max_size, max_backups))
                except Exception as e:
                    self.send_log(
                        f"get targets error ({name!r}, {max_size!r}, {max_backups!r}) {e!r}",
                        e,
                        level=logging.ERROR,
                    )
        if level_spec is not None:
            for t in targets:
                if isinstance(t, RotatingFileWriter):
                    t.level_spec = level_spec
        return targets

    def _evict_opened_files(self, protect=()):
        """Close the least recently used files when MAX_OPENED_FILES is exceeded.

        Names in `protect` (used by the current batch) are never evicted.
        A closed RotatingFileWriter reopens itself on the next write, so an
        evicted name that gets written again is simply re-created.
        """
        evicted = []
        while len(self._opened_files) > self.MAX_OPENED_FILES:
            candidates = [n for n in self._opened_files if n not in protect]
            if not candidates:
                break
            evicted_name = candidates[0]
            fw = self._opened_files.pop(evicted_name)
            try:
                fw.close()
            except Exception as e:
                # best-effort close, keep evicting regardless
                self.send_log(
                    f"error closing evicted log file {evicted_name!r}: {e!r}",
                    level=logging.WARNING,
                )
            evicted.append(evicted_name)
        if evicted:
            # aggregate into one record: logging each eviction would feed the
            # queue that caused the eviction in the first place
            self.send_log(
                f"closed {len(evicted)} least recently used log files (limit={self.MAX_OPENED_FILES}): {evicted}",
                level=logging.DEBUG,
            )

    def save_new_setting(self, name, setting: LogSetting):
        if self._log_settings.get(name) == setting:
            return False
        self._log_settings[name] = setting
        self.send_log(f"`{name}` update setting: {setting}", level=logging.INFO)
        self.dump_settings()
        return True

    def dump_settings(self):
        """Dump & Load settings to setting_path as jsonl format, with a readable meta data."""
        if not self.setting_path:
            return True
        temp = self.setting_path.with_suffix(".tmp")
        lines = [
            json.dumps(
                {"name": name, "setting": setting.to_dict_with_meta()},
                ensure_ascii=False,
            )
            for name, setting in self._log_settings.items()
        ]
        text = "\n".join(lines) + "\n"
        try:
            temp.write_text(text, encoding="utf-8")
            shutil.move(temp.as_posix(), self.setting_path.as_posix())
        except Exception as e:
            self.send_log(
                f"error in dump_settings {traceback.format_exc()}",
                e,
                level=logging.WARNING,
            )

    def save_setting(self, name, record: dict):
        if DefaultLogSetting._key_name in record:
            data = record[DefaultLogSetting._key_name]
            if not isinstance(data, dict):
                return False
            try:
                setting = LogSetting.from_dict(**data)
                self.save_new_setting(name, setting)
                return True
            except Exception as e:
                self.send_log(
                    f"`{name}` send invalid setting, {e!r}: {repr(data)[:100]}",
                    level=logging.WARNING,
                )
        return False

    def get_setting(self, name: str):
        if name in self._log_settings:
            return self._log_settings[name]
        else:
            default = LogSetting.get_default()
            self._log_settings[name] = default
            return default

    def make_record(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ) -> logging.LogRecord:
        rv = logging.LogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if key in rv.__dict__:
                    continue
                rv.__dict__[key] = extra[key]
        return rv

    def _next_queue_msg(self, index: int):
        """Return the next queue item, or None when this batch should end."""
        if index == 0:
            try:
                return typing.cast(QueueMsg, self._write_queue.get(timeout=1))
            except Empty:
                return None
        try:
            return self._write_queue.get_nowait()
        except Empty:
            return None

    def _make_record_or_none(self, name, record_dict):
        """Build one LogRecord, reporting and dropping invalid payloads."""
        try:
            return self.make_record(
                name=name,
                level=record_dict.get("levelno", logging.INFO),
                fn=record_dict.get("pathname", ""),
                lno=record_dict.get("lineno", 0),
                msg=record_dict.get("msg", ""),
                args=record_dict.get("args", ()),
                exc_info=record_dict.get("exc_info", None),
                func=record_dict.get("funcName", None),
                extra=record_dict,
                sinfo=record_dict.get("stack_info", None),
            )
        except Exception as e:
            self.send_log(
                f"`{name}` send invalid record.__dict__, {e!r}: {repr(record_dict)[:100]}",
                level=logging.WARNING,
            )
            return None

    def _drain_write_queue(self):
        """Drain one batch from the queue, returning ``(new_lines, stopped)``."""
        stopped = False
        if self._shutdown_ev and self._shutdown_ev.is_set():
            self.send_log("stopping write_worker daemon(shutdown)")
            stopped = True
        new_lines = {}
        for index in range(self.max_queue_buffer):
            q_msg = self._next_queue_msg(index)
            if q_msg is None:
                break
            if q_msg is STOP_SIG:
                if not stopped:
                    self.send_log("stopping write_worker daemon(signal)")
                    stopped = True
                continue
            name, record_dict = q_msg.name, q_msg.record
            if self.save_setting(name, record_dict):
                # ignore msg only for setting
                continue
            log_record = self._make_record_or_none(name, record_dict)
            if log_record is not None:
                new_lines.setdefault(name, []).append(log_record)
        return new_lines, stopped

    def _format_records(self, name, record_list):
        """Format a batch into (levelno, text) pairs, dropping unformattable ones."""
        _format = self.get_setting(name).formatter.format
        lines = []
        for record in record_list:
            if getattr(record, "nofmt", False):
                lines.append((record.levelno, record.msg))
                continue
            try:
                lines.append((record.levelno, _format(record)))
            except Exception as e:
                # one bad record must not kill the whole consumer
                self.send_log(
                    f"`{name}` format error, record dropped: {e!r}",
                    level=logging.WARNING,
                )
        return lines

    def _collect_targets(self, name, setting):
        """Targets for a logger name, including per-level alias writers."""
        targets = self.get_targets(
            name,
            max_size=setting.max_size,
            max_backups=setting.max_backups,
        )
        if setting.level_specs:
            for levelno in setting.level_specs:
                levelname = LogSetting.level_to_name(levelno).lower().replace(" ", "-")
                alias_name = f"{name}_{levelname}"
                targets.extend(
                    self.get_targets(
                        alias_name,
                        max_size=setting.max_size,
                        max_backups=setting.max_backups,
                        level_spec=levelno,
                    )
                )
        return targets

    def _write_to_targets(self, name, lines, targets):
        """Write the batch to every target; returns the number of chars written."""
        text_counter = 0
        for log_file in targets:
            try:
                levelno = getattr(log_file, "level_spec", 0)
                _lines = [text for level, text in lines if level >= levelno]
                lines_text = "\n".join(_lines) + "\n"
                text_counter += len(lines_text)
                log_file.write(lines_text)
                log_file.flush()
            except Exception as e:  # noqa: PERF203  # one failing target must not abort the remaining writes in the batch
                self.send_log(
                    f"error in write_worker ({name})",
                    e,
                    level=logging.WARNING,
                )
        return text_counter

    def _write_batch(self, new_lines):
        """Format the collected records and write them to their targets."""
        for name, record_list in new_lines.items():
            setting = self.get_setting(name)
            lines = self._format_records(name, record_list)
            targets = self._collect_targets(name, setting)
            text_counter = self._write_to_targets(name, lines, targets)
            if name != self.name:
                self._lines_counter[name] += len(lines)
                self._size_counter[name] += text_counter

    def _log_counters(self, last_log_time, interval):
        """Flush idle files and emit counters; returns the new last log time."""
        if not self._lines_counter:
            return last_log_time
        now = time.time()
        if now - last_log_time <= interval:
            return last_log_time
        start = ttime(last_log_time, fmt="%H:%M:%S")
        end = ttime(now, fmt="%H:%M:%S")
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
                        self.send_log(f"closed idle log file: {name} (last modified: {ttime(mtime)})")
            except KeyError:  # noqa: PERF203  # the file may have been evicted since the counter snapshot; skip it
                pass
        lines_msg = json.dumps(
            {name: value for name, value in self._lines_counter.most_common(30)},
            ensure_ascii=False,
        )
        size_msg = json.dumps(
            {name: read_size(value, 1, shorten=True) for name, value in self._size_counter.most_common(30)},
            ensure_ascii=False,
        )
        self.send_log(
            f"[{start} - {end}] log counter: {sum(self._lines_counter.values())} lines ({lines_msg}), {read_size(sum(self._size_counter.values()), 1, shorten=True)} ({size_msg})"
        )
        self._lines_counter.clear()
        self._size_counter.clear()
        return now

    def write_queue_consumer(self):
        self.send_log("start write_queue_consumer daemon")
        stopped = False
        interval = 30
        last_log_time = time.time()
        while not stopped:
            try:
                new_lines, stopped = self._drain_write_queue()
                self._write_batch(new_lines)
                last_log_time = self._log_counters(last_log_time, interval)
            except Exception as e:  # noqa: PERF203  # one malformed batch must not kill the consumer daemon; logged then break
                self.send_log("error in write_queue_consumer", e, level=logging.ERROR)
                print(format_error(e), file=sys.stderr, flush=True)
                traceback.print_exc()
                self.shutdown()
                break
        self.send_log("stopped write_queue_consumer daemon")

    async def default_handler(self, record: dict):
        # record demo:
        # {"name": "test_logger", "msg": "log server test message", "args": null, "levelname": "INFO", "levelno": 20, "pathname": "/PATH/temp.py", "filename": "temp.py", "module": "temp", "exc_info": null, "exc_text": null, "stack_info": null, "lineno": 38, "funcName": "main", "created": 1723270162.5119407, "msecs": 511.0, "relativeCreated": 102.74338722229004, "thread": 8712, "threadName": "MainThread", "processName": "MainProcess", "process": 19104}
        try:
            if not isinstance(record, dict):
                raise TypeError("record must be a dict")
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
        try:
            self._write_queue.put_nowait(STOP_SIG)
        except Full:
            # queue full, the consumer still exits via _shutdown_ev below
            pass
        if self._shutdown_ev:
            self.loop.call_soon_threadsafe(self._shutdown_ev.set)
        # unblock wait_close_sync in sync mode (shutdown_event may not exist
        # when the server was never entered, hence the getattr guard)
        if getattr(self, "shutdown_event", None) and not self.shutdown_event.is_set():
            self.shutdown_event.set()

    def __del__(self):
        # __init__ may raise before _init_settings created the file map (e.g.
        # the non-loopback host refusal), and a half-built instance still gets
        # finalized: only close what actually exists
        if getattr(self, "_opened_files", None):
            self.close_opened_files()

    def close_opened_files(self):
        while self._opened_files:
            _, fw = self._opened_files.popitem()
            try:
                fw.close()
            except Exception:  # noqa: S110  # best-effort close during shutdown; logging here could re-enter the queue being torn down
                pass

    async def run_wrapper(self):
        async with self:
            await self.wait_closed()

    def wait_close_sync(self, timeout=None):
        return self.shutdown_event.wait(timeout=timeout)

    def __enter__(self):
        "Sync entry, start log server in a thread."
        self.shutdown_event = SyncEvent()
        self._thread = Thread(target=asyncio.run, args=(self.run_wrapper(),), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # teardown order: stop queue -> shutdown server -> join the worker
        # thread (it joins the consumer before returning) -> close files.
        # or use async with LogServer(...) as ls: ...
        time.sleep(0.01)
        try:
            self._write_queue.put_nowait(STOP_SIG)
        except Full:
            # queue full, shutdown below still lets the consumer exit
            pass
        self.shutdown()
        self._thread.join(timeout=self.CONSUMER_JOIN_TIMEOUT)
        self.close_opened_files()


CONNECTED_HANDLERS: typing.Dict[tuple, typing.Union[logging.handlers.SocketHandler, logging.NullHandler]] = {}


def clear_handlers():
    for handler in CONNECTED_HANDLERS.values():
        if hasattr(handler, "close"):
            handler.close()


def create_handler(host: str, port: int, level=logging.DEBUG):
    if not CONNECTED_HANDLERS:
        import atexit

        atexit.register(clear_handlers)
        CONNECTED_HANDLERS[("", 0)] = logging.NullHandler()  # dummy to avoid multiple register
    h = logging.handlers.SocketHandler(host, port)
    h.createSocket()
    if not h.sock:
        raise ConnectionError(f"Cannot connect to log server at {host}:{port}")
    h.setLevel(level)
    CONNECTED_HANDLERS[(h.host, h.port)] = h
    return h


def get_logger(
    name: str,
    host: str = DefaultLogSetting.host,
    port: int = DefaultLogSetting.port,
    log_level: int = logging.DEBUG,
    socket_handler_level: int = logging.DEBUG,
    shorten_level: bool = True,
    streaming: typing.Optional[typing.TextIO] = None,  # sys.stderr, sys.stdout, None
    streaming_level: int = logging.DEBUG,
    # custom settings for log files
    formatter: typing.Optional[logging.Formatter] = LogHelper.DEFAULT_FORMATTER,
    max_size: int = DefaultLogSetting.max_size,
    max_backups: int = DefaultLogSetting.max_backups,
    level_specs: typing.Optional[typing.List[int]] = None,
) -> logging.Logger:
    """Get a singleton logger that sends logs to the LogServer.
    For easy use, you can use original logging.handlers.SocketHandler, but you need to manage the handler yourself.

    Demo::

        # python -m morebuiltins.cmd.log_server --host localhost --port 8901 --log-dir logs
        import logging
        import logging.handlers
        logger = logging.getLogger("client")
        logger.setLevel(logging.DEBUG)
        h = logging.handlers.SocketHandler("localhost", 8901)
        h.setLevel(logging.DEBUG)
        logger.addHandler(h)
        # Add custom settings
        # Add error log to a specific log file; use a custom formatter and max_size; set msg to ""
        formatter = logging.Formatter(fmt="%(asctime)s - %(filename)s - %(message)s")
        logger.info("", extra={"log_setting": {"formatter": formatter, "max_size": 1024**2, "level_specs": [logging.ERROR]}})
        for _ in range(5):
            logger.info("hello world!")
        # Send some error logs
        logger.error("this is an error!")
        # The remote log server will create a "client_error.log" file in log_dir if settings are applied

    """
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
        for handler in logger.handlers:
            handler.setFormatter(formatter)
        logger.log(
            socket_handler_level,
            "",
            extra={
                DefaultLogSetting._key_name: {
                    "formatter": formatter,
                    "max_size": max_size,
                    "max_backups": max_backups,
                    "level_specs": level_specs or [],
                }
            },
        )
    return logger


async def main():
    import argparse

    parser = argparse.ArgumentParser(usage=(LogServer.__doc__ or "").replace("%", "%%"))
    parser.add_argument(
        "--host", default=DefaultLogSetting.host, help="bind host; must be loopback (127.0.0.1/::1/localhost)"
    )
    parser.add_argument("--port", default=DefaultLogSetting.port, type=int)
    parser.add_argument(
        "-t",
        "--log-dir",
        default="",
        dest="log_dir",
        help="log dir to save log files, if empty, log to stderr with --log-stream",
    )
    parser.add_argument("--name", default="log_server", help="log server name")
    parser.add_argument(
        "--max-size",
        default=DefaultLogSetting.max_size,
        type=int,
        dest="max_size",
        help=f"max_size for log files, default to {DefaultLogSetting.max_size}, {round(DefaultLogSetting.max_size / 1024 / 1024)}MB each log file",
    )
    parser.add_argument(
        "--max-backups",
        default=DefaultLogSetting.max_backups,
        type=int,
        dest="max_backups",
        help=f"max_backups for log files, default to {DefaultLogSetting.max_backups}",
    )
    parser.add_argument(
        "--max-queue-size",
        default=DefaultLogSetting.max_queue_size,
        type=int,
        help="max queue size for log queue, log will be in memory queue before write to file",
    )
    parser.add_argument(
        "--max-queue-buffer",
        default=DefaultLogSetting.max_queue_buffer,
        type=int,
        help="chunk size of lines before write to file",
    )
    parser.add_argument(
        "--log-stream",
        default="sys.stderr",
        help="log to stream: 'sys.stdout' or 'sys.stderr', "
        "if --log-stream='' or --log-stream=- or --log-stream=0, will mute the stream log, "
        "unrecognized values are muted with a warning",
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
        "--idle-close-time",
        dest="idle_close_time",
        type=int,
        default=DefaultLogSetting.idle_close_time,
        help=f"close log files idle for this many seconds, default to {DefaultLogSetting.idle_close_time}",
    )
    args = parser.parse_args()
    if not LogServer._is_loopback(args.host):
        parser.error(
            f"refusing to bind non-loopback host {args.host!r}: the socket protocol has no authentication "
            "and every incoming frame is unpickled; bind 127.0.0.1/::1 and forward logs through a trusted channel"
        )
    stream_choices = {"sys.stdout": sys.stdout, "sys.stderr": sys.stderr}
    if args.log_stream in ("", "-", "0", "None"):
        log_stream = None
    elif args.log_stream in stream_choices:
        log_stream = stream_choices[args.log_stream]
    else:
        print(
            f"warning: unrecognized --log-stream={args.log_stream!r}, stream log muted",
            file=sys.stderr,
            flush=True,
        )
        log_stream = None
    async with LogServer(
        host=args.host,
        port=args.port,
        log_dir=args.log_dir,
        name=args.name,
        max_size=args.max_size,
        max_backups=args.max_backups,
        max_queue_size=args.max_queue_size,
        max_queue_buffer=args.max_queue_buffer,
        log_stream=log_stream,
        compress=args.compress,
        shorten_level=not args.origin_level,
        idle_close_time=args.idle_close_time,
    ) as ls:
        await ls.wait_closed()


async def async_test():
    async with LogServer(log_dir="logs"):
        logger = get_logger("test_async", level_specs=[logging.ERROR, logging.INFO])
        i = 0
        for i in range(5):
            logger.info(f"log server test message {i + 1}")
        logger.error(f"log server test message {i + 1}")
        logger.info("test nofmt: ", extra={"nofmt": True})
        for i in range(5):
            logger.info(f"{i} ", extra={"nofmt": True})
    shutil.rmtree("logs", ignore_errors=True)


def sync_test():
    asyncio.run(async_test())
    with LogServer(log_dir="logs"):
        logger = get_logger("test_sync", level_specs=[logging.ERROR, 13])
        for i in range(5):
            logger.info(f"log server test message {i + 1}")
        logger.log(13, "log server test message level 13")
        logger.info("test nofmt: ", extra={"nofmt": True})
        for i in range(5):
            logger.info(f"{i} ", extra={"nofmt": True})
    shutil.rmtree("logs", ignore_errors=True)


def entrypoint():
    # return sync_test()
    return asyncio.run(main())


if __name__ == "__main__":
    entrypoint()

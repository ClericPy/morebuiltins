import asyncio
import base64
import json
import logging
import logging.handlers
import os
import pickle
import re
import shutil
import signal
import sys
import time
import traceback
import typing
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event as SyncEvent
from threading import Thread

from morebuiltins.ipc import SocketLogHandlerEncoder, SocketServer
from morebuiltins.logs import LogHelper, RotatingFileWriter
from morebuiltins.utils import Validator, format_error, read_size, ttime

__all__ = ["LogServer"]


class QueueMsg:
    __slots__ = ("name", "record")

    def __init__(self, name: str, record: dict):
        self.name = name
        self.record = record


class STOP_SIG(QueueMsg):
    pass


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
    formatter: logging.Formatter = DefaultLogSetting.formatter
    max_size: int = DefaultLogSetting.max_size
    max_backups: int = DefaultLogSetting.max_backups
    level_specs: list[int] = field(default_factory=list)
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
            level = str(level).upper()
            if level not in logging._nameToLevel:
                if re.match(r"^Level \d+$", level):
                    level = level.split()[-1]
                    return int(level)
                else:
                    raise ValueError(
                        f"level_specs[{index}] invalid log level name: {level}"
                    )
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
        return pickle.loads(base64.b64decode(data.encode("utf-8")))

    @classmethod
    def from_dict(cls, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in cls.__annotations__}
        return cls(**kwargs)

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
        meta["level_specs"] = [
            logging.getLevelName(level) for level in self.level_specs
        ]
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

    [WARNING]: Ensure your log msg is "" if you only want to update settings, or the msg will be skipped.

    logger.info("", extra={"log_setting": {"formatter": formatter, "max_size": 1024**2, "level_specs": [logging.ERROR]}})


    ### Server demo1:
        start log server in terminal, only collect logs and print to console
        > python -m morebuiltins.cmd.log_server

    ### Server demo2:
        custom options to log to "logs" directory, default rotates at 10MB with 5 backups, no log_stream, enable compress
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
            self.setting_path: typing.Optional[Path] = self.log_dir.joinpath(
                f"{self.name}_settings.jsonl"
            )
        else:
            self.setting_path = None
        self._server_log_setting = LogSetting(
            max_size=max_size, max_backups=max_backups
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
        self.handle_signals = DefaultLogSetting.handler_signals
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
        try:
            with self.setting_path.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    name = data["name"]
                    setting = LogSetting.from_dict(**data["setting"])
                    result[name] = setting
            self.send_log(
                f"Loaded log settings from {self.setting_path}, {len(result)} items"
            )
        except Exception as e:
            self.send_log(f"Failed to load log settings from {self.setting_path}: {e}")
        return result

    async def __aenter__(self):
        await super().__aenter__()
        self._queue_consumer_task = self.loop.run_in_executor(
            self._default_executor, self.write_queue_consumer
        )
        return self

    async def __aexit__(self, *_errors):
        await asyncio.sleep(0.01)
        await super().__aexit__(*_errors)
        await asyncio.to_thread(self.close_opened_files)

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

    async def end_callback(self):
        self._write_queue.put_nowait(STOP_SIG)
        await self._queue_consumer_task

    def start_callback(self):
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
            "levelname": logging.getLevelName(level),
            "exc_info": {},
        }
        if init_setting:
            record[DefaultLogSetting._key_name] = asdict(self._server_log_setting)
        q_msg = QueueMsg(name=self.name, record=record)
        self._write_queue.put_nowait(q_msg)

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
                    self._opened_files[name] = fw
                    targets.append(fw)
                except Exception as e:
                    self.send_log(
                        f"get targets error ({name!r}, {max_size!r}, {max_backups!r}) {e!r}",
                        e,
                        level=logging.ERROR,
                    )
        if level_spec is not None:
            for t in targets:
                if isinstance(t, RotatingFileWriter):
                    setattr(t, "level_spec", level_spec)
        return targets

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
            except TypeError as e:
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
                        if q_msg is STOP_SIG:
                            if not stopped:
                                self.send_log("stopping write_worker daemon(signal)")
                                stopped = True
                            continue
                        name, record = q_msg.name, q_msg.record
                        try:
                            log_record = logging.LogRecord(
                                level=record.get("levelno", 0), **record
                            )
                        except Exception as e:
                            self.send_log(
                                f"`{name}` send invalid record.__dict__, {e!r}: {repr(record)[:100]}",
                                level=logging.WARNING,
                            )
                            continue
                        if self.save_setting(name, record):
                            # ignore msg only for setting
                            continue
                        if name in new_lines:
                            new_lines[name].append(log_record)
                        else:
                            new_lines[name] = [log_record]
                    except Empty:
                        break
                for name, record_list in new_lines.items():
                    setting = self.get_setting(name)
                    _format = setting.formatter.format
                    lines = [
                        (record.levelno, _format(record)) for record in record_list
                    ]
                    targets = self.get_targets(
                        name,
                        max_size=setting.max_size,
                        max_backups=setting.max_backups,
                    )
                    if setting.level_specs:
                        for levelno in setting.level_specs:
                            levelname = (
                                logging.getLevelName(levelno).lower().replace(" ", "-")
                            )
                            alias_name = f"{name}_{levelname}"
                            targets.extend(
                                self.get_targets(
                                    alias_name,
                                    max_size=setting.max_size,
                                    max_backups=setting.max_backups,
                                    level_spec=levelno,
                                )
                            )
                    text_counter = 0
                    for log_file in targets:
                        try:
                            levelno = getattr(log_file, "level_spec", 0)
                            _lines = [text for level, text in lines if level >= levelno]
                            lines_text = "\n".join(_lines) + "\n"
                            text_counter += len(lines_text)
                            log_file.write(lines_text)
                            log_file.flush()
                        except Exception as e:
                            self.send_log(
                                f"error in write_worker ({name})",
                                e,
                                level=logging.WARNING,
                            )
                    if name != self.name:
                        self._lines_counter[name] += len(lines)
                        self._size_counter[name] += text_counter
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
        # {"name": "test_logger", "msg": "log server test message", "args": null, "levelname": "INFO", "levelno": 20, "pathname": "/PATH/temp.py", "filename": "temp.py", "module": "temp", "exc_info": null, "exc_text": null, "stack_info": null, "lineno": 38, "funcName": "main", "created": 1723270162.5119407, "msecs": 511.0, "relativeCreated": 102.74338722229004, "thread": 8712, "threadName": "MainThread", "processName": "MainProcess", "process": 19104}
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
        self._write_queue.put_nowait(STOP_SIG)
        if self._shutdown_ev:
            self.loop.call_soon_threadsafe(self._shutdown_ev.set)

    def __del__(self):
        self.close_opened_files()

    def close_opened_files(self):
        while self._opened_files:
            _, fw = self._opened_files.popitem()
            try:
                fw.close()
            except Exception:
                pass

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
        self._write_queue.put_nowait(STOP_SIG)
        self.shutdown()
        self._thread.join(timeout=1)
        self.close_opened_files()


CONNECTED_HANDLERS: typing.Dict[
    tuple, typing.Union[logging.handlers.SocketHandler, logging.NullHandler]
] = {}


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
    parser.add_argument("--host", default=DefaultLogSetting.host)
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
        help="log to stream, if --log-stream='' or --log-stream=- or --log-stream=0, will mute the stream log",
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
        for i in range(5):
            logger.info(f"log server test message {i + 1}")
        logger.error(f"log server test message {i + 1}")
    # shutil.rmtree("logs", ignore_errors=True)


def sync_test():
    # return asyncio.run(async_test())
    with LogServer(log_dir="logs"):
        logger = get_logger("test_sync", level_specs=[logging.ERROR, 13])
        for i in range(5):
            logger.info(f"log server test message {i + 1}")
    # shutil.rmtree("logs", ignore_errors=True)


def entrypoint():
    # return sync_test()
    return asyncio.run(main())


if __name__ == "__main__":
    entrypoint()

import asyncio
import atexit
import logging
import re
import sys
import time
import typing
from contextvars import ContextVar, copy_context
from functools import partial
from gzip import GzipFile
from io import TextIOBase
from logging import Filter
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from multiprocessing import Queue as ProcessQueue
from pathlib import Path
from queue import Queue
from threading import RLock, Thread
from typing import Dict, List, Optional, TextIO, Tuple, Union
from weakref import WeakSet

__all__ = [
    "async_logger",
    "AsyncQueueListener",
    "LogHelper",
    "RotatingFileWriter",
    "SizedTimedRotatingFileHandler",
    "ContextFilter",
]


async def to_thread(func, /, *args, **kwargs):
    """Asynchronously run function *func* in a separate thread, same as `asyncio.to_thread` in python 3.9+."""
    func_call = partial(copy_context().run, func, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(None, func_call)


class LogHelper:
    """Quickly bind a logging handler to a logger, with a StreamHandler or SizedTimedRotatingFileHandler.

    The default handler is a StreamHandler to sys.stderr.
    The default file handler is a SizedTimedRotatingFileHandler, which can rotate logs by both time and size.

    Examples::

        # 1. Bind a StreamHandler to the "mylogger" logger, output to sys.stdout
        import logging
        from morebuiltins.logs import LogHelper

        LogHelper.shorten_level()
        logger = LogHelper.bind_handler(name="mylogger", filename=sys.stdout, maxBytes=100 * 1024**2, backupCount=7)
        # use logging.getLogger to get the same logger instance
        logger2 = logging.getLogger("mylogger")
        assert logger is logger2
        logger.info("This is an info message")
        logger.fatal("This is a critical message")

        # 2. Bind file and stderr in the same logger
        import sys
        import logging
        from morebuiltins.logs import LogHelper
        LogHelper.shorten_level()
        logger = LogHelper.bind_handler(name="mylogger", filename="mylog.log", maxBytes=100 * 1024**2, backupCount=7)
        logger = LogHelper.bind_handler(name="mylogger", filename=sys.stderr)
        logger.info("This is an info message")

        # 3. Use queue=True to make logging non-blocking, both file and stderr
        import sys
        from morebuiltins.logs import LogHelper
        LogHelper.shorten_level()
        logger = LogHelper.bind_handler(name="mylogger", filename="mylog.log", maxBytes=100 * 1024**2, backupCount=7, queue=True)
        logger = LogHelper.bind_handler(name="mylogger", filename=sys.stderr, queue=True)
        logger.info("This is an info message")
    """

    DEFAULT_FORMAT = (
        "%(asctime)s | %(levelname)-5s | %(filename)+8s:%(lineno)+3s - %(message)s"
    )
    DEFAULT_FORMATTER = logging.Formatter(DEFAULT_FORMAT)
    FILENAME_HANDLER_MAP: Dict[str, logging.Handler] = {}

    @classmethod
    def close_all_handlers(cls) -> List[Tuple[str, bool, Optional[Exception]]]:
        """Close all handlers in the FILENAME_HANDLER_MAP."""
        result: List[Tuple[str, bool, Optional[Exception]]] = []
        for key, handler in list(cls.FILENAME_HANDLER_MAP.items()):
            try:
                handler.close()
                cls.FILENAME_HANDLER_MAP.pop(key, None)
                result.append((key, True, None))
            except Exception as e:
                result.append((key, False, e))
        return result

    @classmethod
    def bind_handler(
        cls,
        name: Optional[str] = "main",
        filename: Union[
            TextIO, TextIOBase, None, logging.Handler, str, Path
        ] = sys.stderr,
        when="h",
        interval=1,
        backupCount=0,
        maxBytes=0,
        encoding=None,
        delay=False,
        utc=False,
        compress=False,
        formatter: Union[str, logging.Formatter, None] = DEFAULT_FORMAT,
        handler_level: Union[None, str, int] = "INFO",
        logger_level: Union[None, str, int] = "INFO",
        queue: Union[bool, None, Queue, ProcessQueue] = False,
    ):
        """Bind a logging handler to the specified logger name, with support for file, stream, or custom handler.
        This sets up the logger with the desired handler, formatter, and log levels.

        Args:
            name (str, optional): The logger name. Defaults to "main".
            filename (Union[TextIO, TextIOWrapper, None, logging.Handler, str, Path], optional):
                The log destination. Can be a file path, stream, handler, or None to clear handlers. Defaults to sys.stderr.
            when (str, optional): Time interval for log rotation (if file handler). Defaults to "h".
            interval (int, optional): Rotation interval. Defaults to 1.
            backupCount (int, optional): Number of backup files to keep. Defaults to 0.
            maxBytes (int, optional): Maximum file size before rotation. Defaults to 0 (no size-based rotation).
            encoding (str, optional): File encoding. Defaults to None.
            delay (bool, optional): Delay file opening until first write. Defaults to False.
            utc (bool, optional): Use UTC time for file rotation. Defaults to False.
            compress (bool, optional): Compress rotated log files. Defaults to False.
            formatter (Union[str, logging.Formatter, None], optional): Formatter or format string. Defaults to DEFAULT_FORMAT.
            handler_level (Union[None, str, int], optional): Log level for the handler. Defaults to "INFO".
            logger_level (Union[None, str, int], optional): Log level for the logger. Defaults to "INFO".
            queue (Union[bool, None, Queue, ProcessQueue], optional): If True, use a Queue for async logging.
                If a Queue or ProcessQueue is provided, it will be used directly. Defaults to False.

        Raises:
            TypeError: If filename is not a supported type.

        Returns:
            logging.Logger: The configured logger instance.

        Demo::
            >>> logger = LogHelper.bind_handler(name="mylogger", filename=sys.stdout)
            >>> len(logger.handlers)
            1
            >>> logger = LogHelper.bind_handler(name="mylogger", filename=sys.stderr)
            >>> logger2 = logging.getLogger("mylogger")
            >>> assert logger is logger2
            >>> len(logger.handlers)
            2
            >>> bool(LogHelper.close_all_handlers() or True)
            True
            >>> logger = LogHelper.bind_handler(name="mylogger1", queue=True)
            >>> logger.handlers[0]
            <QueueHandler (NOTSET)>
            >>> logger = LogHelper.bind_handler(name="mylogger2", queue=False)
            >>> logger.handlers[0]
            <StreamHandler <stderr> (INFO)>
        """
        logger = logging.getLogger(name)
        if filename is None:
            logger.handlers.clear()
            return logger
        # Check if handler already exists for the given filename
        elif isinstance(filename, TextIOBase):
            key = str(id(filename))
            if key in cls.FILENAME_HANDLER_MAP:
                handler: logging.Handler = cls.FILENAME_HANDLER_MAP[key]
            else:
                handler = logging.StreamHandler(filename)
                cls.FILENAME_HANDLER_MAP[key] = handler
        elif isinstance(filename, logging.Handler):
            key = str(id(filename))
            if key in cls.FILENAME_HANDLER_MAP:
                handler = cls.FILENAME_HANDLER_MAP[key]
            else:
                handler = filename
                cls.FILENAME_HANDLER_MAP[key] = handler
        elif isinstance(filename, str) or isinstance(filename, Path):
            key = Path(filename).resolve().as_posix()
            if key in cls.FILENAME_HANDLER_MAP:
                handler = cls.FILENAME_HANDLER_MAP[key]
            else:
                handler = SizedTimedRotatingFileHandler(
                    key,
                    when=when,
                    interval=interval,
                    backupCount=backupCount,
                    maxBytes=maxBytes,
                    encoding=encoding,
                    delay=delay,
                    utc=utc,
                    compress=compress,
                )
                cls.FILENAME_HANDLER_MAP[key] = handler
        else:
            raise TypeError(
                f"filename must be str, Path, TextIO, or logging.Handler, not {type(filename)}"
            )
        # Update the levels of the handler and logger
        if handler_level is not None:
            handler.setLevel(handler_level)
        if logger_level is not None:
            logger.setLevel(logger_level)
        # Set the formatter for the handler
        if isinstance(formatter, str):
            formatter = logging.Formatter(formatter)
        elif isinstance(formatter, logging.Formatter):
            formatter = formatter
        else:
            formatter = logging.Formatter(cls.DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        # Add the handler to the logger
        if queue:
            if queue is True:
                queue = Queue()
            elif not isinstance(queue, (Queue, ProcessQueue)):
                raise TypeError("queue must be a Queue, ProcessQueue, True, or False")
            if (
                len(logger.handlers) == 1
                and isinstance(logger.handlers[0], QueueHandler)
                and isinstance(logger.handlers[0].listener, AsyncQueueListener)
            ):
                # already a QueueHandler
                async_listener: AsyncQueueListener = logger.handlers[0].listener
                async_listener.bind_new_handler(handler)
                return logger
            else:
                logger.addHandler(handler)
                async_listener = AsyncQueueListener(logger, queue=queue)
                async_listener.start()
                atexit.register(async_listener.stop)
        else:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        return logger

    @classmethod
    def shorten_level(
        cls,
        mapping: Dict[int, str] = {logging.WARNING: "WARN", logging.CRITICAL: "FATAL"},
    ):
        """Shorten the level names less than 5 chars: WARNING to WARN, CRITICAL to FATAL."""
        for level, name in mapping.items():
            logging.addLevelName(level, name)

    @classmethod
    def handle_crash(cls, logger: logging.Logger, msg="[Uncaught Exception]"):
        sys.excepthook = lambda exctype, value, tb: logger.critical(
            msg, exc_info=(exctype, value, tb)
        )


class RotatingFileWriter:
    """RotatingFileWriter class for writing to a file with rotation support.

    Demo::

        >>> # test normal usage
        >>> writer = RotatingFileWriter("test.log", max_size=10 * 1024, max_backups=1)
        >>> writer.write("1" * 10)
        >>> writer.path.stat().st_size
        0
        >>> writer.flush()
        >>> writer.path.stat().st_size
        10
        >>> writer.clean_backups(writer.max_backups)
        >>> writer.unlink_file()
        >>> # test rotating
        >>> writer = RotatingFileWriter("test.log", max_size=20, max_backups=2)
        >>> writer.write("1" * 15)
        >>> writer.write("1" * 15)
        >>> writer.write("1" * 15, flush=True)
        >>> writer.path.stat().st_size
        15
        >>> len(writer.backup_path_list())
        2
        >>> writer.clean_backups(writer.max_backups)
        >>> writer.unlink_file()
        >>> # test no backups
        >>> writer = RotatingFileWriter("test.log", max_size=20, max_backups=0)
        >>> writer.write("1" * 15)
        >>> writer.write("1" * 15)
        >>> writer.write("1" * 15, flush=True)
        >>> writer.path.stat().st_size
        15
        >>> len(writer.backup_path_list())
        0
        >>> writer.clean_backups(writer.max_backups)
        >>> len(writer.backup_path_list())
        0
        >>> writer = RotatingFileWriter("test.log", max_size=20, max_backups=3)
        >>> writer.print("1" * 100)
        >>> writer.unlink(rotate=False)
        >>> len(writer.backup_path_list())
        1
        >>> writer.unlink(rotate=True)
        >>> len(writer.backup_path_list())
        0
        >>> writer = RotatingFileWriter("test.log", max_size=20, max_backups=3, compress=True)
        >>> writer.print("1" * 100)
        >>> len(writer.backup_path_list())
        1
        >>> writer.unlink(rotate=True)
        >>> len(writer.backup_path_list())
        0
    """

    check_exist_every = 100

    def __init__(
        self,
        path: Union[Path, str],
        max_size=5 * 1024**2,
        max_backups=0,
        encoding="utf-8",
        errors=None,
        buffering=-1,
        newline=None,
        compress=False,
    ):
        if max_backups < 0:
            raise ValueError("max_backups must be greater than -1, 0 for itself.")
        self._compress_threads: WeakSet = WeakSet()
        self._rotate_lock = RLock()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.max_backups = max_backups
        self.encoding = encoding
        self.errors = errors
        self.buffering = buffering
        self.newline = newline
        self.compress = compress
        self.file = self.reopen_file()
        self._check_exist_count = self.check_exist_every + 1

    def get_suffix(self):
        return time.strftime("%Y%m%d%H%M%S")

    def unlink_file(self):
        return self.unlink(rotate=False)

    def unlink(self, rotate=True, parent=False):
        self.close_file()
        self.path.unlink(missing_ok=True)
        if rotate:
            self.clean_backups(count=self.max_backups + 1)
        if parent:
            for _ in self.path.parent.iterdir():
                return
            else:
                self.path.parent.rmdir()

    def close(self):
        return self.close_file()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()

    def shutdown(self):
        self.close_file()
        if self.compress:
            for t in self._compress_threads:
                t.join()

    def close_file(self):
        file_obj = getattr(self, "file", None)
        if file_obj and not file_obj.closed:
            file_obj.close()
            self.file = None

    def reopen_file(self):
        self.close_file()
        self.file = self.path.open(
            "a",
            encoding=self.encoding,
            errors=self.errors,
            buffering=self.buffering,
            newline=self.newline,
        )
        return self.file

    def check_exist(self):
        return not (
            self._check_exist_count > self.check_exist_every and not self.path.is_file()
        )

    def rotate(self, new_length):
        with self._rotate_lock:
            if self.need_rotate(new_length):
                if self.max_backups > 0:
                    self.close_file()
                    _suffix = self.get_suffix()
                    for index in range(self.max_backups):
                        if index:
                            suffix = f"{_suffix}_{index}"
                        else:
                            suffix = _suffix
                        target_path = self.path.with_name(f"{self.path.name}.{suffix}")
                        if target_path.is_file():
                            # already rotated
                            continue
                        else:
                            break
                    else:
                        raise RuntimeError(
                            "max_backups is too small for writing too fast"
                        )
                    self.path.rename(target_path)
                    self.reopen_file()
                    if not self.compress:
                        self.clean_backups(count=None)
                elif self.file:
                    self.file.seek(0)
                    self.file.truncate()

    def do_compress(self):
        with self._rotate_lock:
            for path in self.path.parent.glob(f"{self.path.name}.*"):
                if path.name == self.path.name:
                    continue
                elif path.suffix == ".gz":
                    continue
                temp_path = path.with_name(path.name + ".gz")
                with GzipFile(temp_path, "wb") as gzip_file:
                    with path.open("rb") as src_file:
                        for line in src_file:
                            gzip_file.write(line)
                path.unlink(missing_ok=True)
            self.clean_backups()

    def need_rotate(self, new_length):
        return self.max_size and self.file.tell() + new_length > self.max_size

    def ensure_file(self, new_length=0):
        if not self.file:
            self.reopen_file()
        elif not self.check_exist():
            self.reopen_file()
        elif self.need_rotate(new_length):
            self.rotate(new_length)
            if self.compress:
                t = Thread(target=self.do_compress)
                t.start()
                self._compress_threads.add(t)

    def backup_path_list(self):
        return list(self.path.parent.glob(f"{self.path.name}.*"))

    def clean_backups(self, count=None):
        """Clean oldest {count} backups, if count is None, it will clean up to max_backups."""
        with self._rotate_lock:
            path_list = self.backup_path_list()
            if path_list:
                if count is None:
                    count = len(path_list) - self.max_backups
                if count > 0:
                    path_list.sort(key=lambda x: x.stat().st_mtime)
                    for deleted, path in enumerate(path_list, 1):
                        path.unlink(missing_ok=True)
                        if deleted >= count:
                            break

    def flush(self):
        self.file.flush()

    def write(self, text: str, flush=False):
        self._check_exist_count += 1
        self.ensure_file(len(text))
        self.file.write(text)
        if flush:
            self.file.flush()

    def print(self, *strings, end="\n", sep=" ", flush=False):
        text = f"{sep.join(map(str, strings))}{end}"
        self.write(text, flush=flush)

    def __del__(self):
        self.shutdown()


class SizedTimedRotatingFileHandler(TimedRotatingFileHandler):
    """TimedRotatingFileHandler with maxSize, to avoid files that are too large.


    Demo::

        import logging
        import time
        from morebuiltins.funcs import SizedTimedRotatingFileHandler

        logger = logging.getLogger("test1")
        h = SizedTimedRotatingFileHandler(
            "logs/test1.log", "d", 1, 3, maxBytes=1, ensure_dir=True
        )
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)

        for i in range(5):
            logger.warning(str(i) * 102400)
            time.sleep(1)
        # 102434 test1.log
        # 102434 test1.log.20241113_231000
        # 102434 test1.log.20241113_231001
        # 102434 test1.log.20241113_231002
        logger = logging.getLogger("test2")
        h = SizedTimedRotatingFileHandler(
            "logs/test2.log", "d", 1, 3, maxBytes=1, compress=True
        )
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)

        for i in range(5):
            logger.warning(str(i) * 102400)
            time.sleep(1)
        # 102434 test2.log
        #    186 test2.log.20241113_231005.gz
        #    186 test2.log.20241113_231006.gz
        #    186 test2.log.20241113_231007.gz

    """

    do_compress_delay = 0.1

    def __init__(
        self,
        filename,
        when="h",
        interval=1,
        backupCount=0,
        maxBytes=0,
        encoding=None,
        delay=False,
        utc=False,
        compress=False,
        ensure_dir=True,
    ):
        """
        Initialize the timed backup file handler.

        :param filename: The name of the log file.
        :param when: The time unit for timed backups, can be "h" (hours) or "d" (days)
        :param interval: The interval for timed backups, with the unit determined by the 'when' parameter
        :param backupCount: The maximum number of backup files to keep
        :param maxBytes: The file size limit before triggering a backup (0 means no limit)
        :param encoding: The encoding of the file
        :param delay: Whether to delay opening the file until the first write
        :param utc: Whether to use UTC time for naming backups
        """
        self.log_path = Path(filename)
        if ensure_dir:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc)
        self.maxBytes = maxBytes
        self.suffix = "%Y%m%d_%H%M%S"
        self.compress = compress
        self.need_compress = False
        self.comress_chunk_size = 64 * 1024
        self.extMatch = re.compile(r"^\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}$", re.ASCII)

    def do_compress_async(self):
        if getattr(self, "_background_compress", None):
            return
        self._compressing = Thread(
            target=self.do_compress, daemon=False, name="do_compress_async"
        )
        self._compressing.start()

    def gzip_log(self, path: Path):
        try:
            temp_path = path.with_suffix(".tmp.gz")
            with open(path, "rb") as f:
                with GzipFile(temp_path, "wb") as gz:
                    size = 0
                    lines = []
                    for line in f:
                        size += len(line)
                        lines.append(line)
                        if size > self.comress_chunk_size:
                            gz.writelines(lines)
                            size = 0
                            lines = []
                    if lines:
                        gz.writelines(lines)
            if path.is_file():
                try:
                    path.unlink()
                    temp_path.rename(path.with_suffix(f"{path.suffix}.gz"))
                except OSError:
                    pass
        except OSError:
            pass
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def do_compress(self):
        try:
            while self.need_compress:
                self.need_compress = False
                time.sleep(self.do_compress_delay)
                now_suffix = f".{time.strftime(self.suffix)}"
                path_list = []
                for path in self.log_path.parent.glob(f"{self.log_path.name}.*"):
                    if path.suffix == now_suffix:
                        self.need_compress = True
                        continue
                    for time_suffix in path.name.split(".")[-2:]:
                        if self.extMatch.match(time_suffix):
                            path_list.append((time_suffix, path))
                path_list.sort()
                target_index = len(path_list) - self.backupCount
                for index, (_, path) in enumerate(path_list):
                    if index < target_index:
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        continue
                    elif path.suffix == ".gz":
                        continue
                    else:
                        self.gzip_log(path)
        finally:
            self._compressing = None

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.
        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        if super().shouldRollover(record):
            return True

        if self.maxBytes > 0:
            if self.stream.tell() >= self.maxBytes:
                return True
        return False

    def doRollover(self):
        """
        Do a rollover, as described by the base class documentation.
        However, also check for the maxBytes parameter and rollover if needed.
        """
        try:
            super().doRollover()
        except OSError:
            pass
        finally:
            self.need_compress = True
        if self.compress:
            self.do_compress_async()

    def getFilesToDelete(self):
        # always return null list
        if self.compress:
            return []
        else:
            return super().getFilesToDelete()

    def __del__(self):
        if self.compress:
            self.do_compress()


class AsyncQueueListener(QueueListener):
    """Asynchronous non-blocking QueueListener that manages logger handlers.
    logger is a logging.Logger instance.
    queue is a Queue or ProcessQueue instance.
    respect_handler_level is a boolean that determines if the handler level should be respected.

    Example:

        async def main():
            # Create logger with a blocking handler
            logger = logging.getLogger("example")
            logger.setLevel(logging.INFO)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(stream_handler)
            # Use async queue listener
            async with AsyncQueueListener(logger):
                # Log won't block the event loop
                for i in range(5):
                    logger.info("log info")
                    logger.debug("log debug")
                    await asyncio.sleep(0.01)
    """

    def __init__(
        self,
        logger: logging.Logger,
        queue: Optional[Union[Queue, ProcessQueue]] = None,
        respect_handler_level=True,
    ):
        self.logger = logger
        self.queue_id = id(queue)
        self.queue = queue or Queue()
        # Store original handlers to restore later
        self.original_handlers = list(logger.handlers)
        # Get handlers that might block, send to parent class
        self.blocking_handlers = [
            h for h in self.original_handlers if not isinstance(h, QueueHandler)
        ]
        # Initialize parent with queue and blocking handlers
        super().__init__(
            self.queue,
            *self.blocking_handlers,
            respect_handler_level=respect_handler_level,
        )
        self._started = self._stopped = False

    def bind_new_handler(self, handler: logging.Handler):
        if isinstance(handler, QueueHandler):
            raise TypeError("handler cannot be a QueueHandler")
        self.original_handlers.append(handler)
        self.blocking_handlers.append(handler)
        self.handlers = tuple(self.blocking_handlers)

    def _switch_to_queue_handler(self):
        """Switch handlers in a blocking context"""
        # Remove original handlers
        for handler in self.original_handlers:
            self.logger.removeHandler(handler)
        # Add queue handler
        queue_handler = QueueHandler(self.queue)
        queue_handler.listener = self
        self.logger.addHandler(queue_handler)

    def _restore_original_handlers(self):
        """Restore original handlers in a blocking context"""
        # Remove queue handler
        self.logger.removeHandler(self.logger.handlers[0])
        # Restore original handlers
        for handler in self.original_handlers:
            self.logger.addHandler(handler)

    def stop(self):
        if self._started and not self._stopped:
            self._stopped = True
            super().stop()
            self._restore_original_handlers()

    def start(self):
        if not self._started:
            self._started = True
            self._switch_to_queue_handler()
            super().start()

    async def __aenter__(self):
        await to_thread(self.start)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await to_thread(self.stop)


# alias for AsyncQueueListener
async_logger = AsyncQueueListener


class ContextFilter(Filter):
    """A logging filter that injects context variables into extra of log records. ContextVar is used to manage context-specific data in a thread-safe / async-safe manner.
    RequestID / TraceID / TaskID can be used to trace logs belonging to the same request or operation across different threads or async tasks.

    Example::

        import random
        import time
        import typing
        from concurrent.futures import ThreadPoolExecutor
        from contextvars import ContextVar
        from logging import Filter, Formatter, StreamHandler, getLogger
        from threading import current_thread

        def test(trace_id: int = 0):
            trace_id_var.set(trace_id)
            for _ in range(3):
                time.sleep(random.random())
                logger.debug(f"msg from thread: {current_thread().ident}")


        trace_id_var: ContextVar = ContextVar("trace_id")
        logger = getLogger()
        logger.addFilter(ContextFilter({"trace_id": trace_id_var}))
        formatter = Formatter("%(asctime)s | [%(trace_id)s] %(message)s")
        handler = StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel("DEBUG")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future = [executor.submit(test, _) for _ in range(3)]

    """

    def __init__(self, context_vars: typing.Dict[str, ContextVar], name: str = ""):
        super().__init__(name)
        self._context_vars = context_vars

    def filter(self, record):
        record_dict = record.__dict__
        for key, var in self._context_vars.items():
            record_dict.setdefault(key, var.get(None))
        return True


def test_LogHelper():
    logger = LogHelper.bind_handler(
        "app_test", filename="app_test.log", maxBytes=1, backupCount=2
    )
    for i in range(3):
        logger.info(str(i))
    LogHelper.close_all_handlers()
    count = 0
    for path in Path(".").glob("app_test.log*"):
        path.unlink(missing_ok=True)
        count += 1
    assert count == 2, count


def test_AsyncQueueListener():
    """Test AsyncQueueListener logs without blocking"""

    async def _test():
        from io import StringIO

        with StringIO() as mock_stdout:
            # Create logger with a blocking handler
            logger = logging.getLogger("example")
            logger.handlers.clear()
            logger.setLevel(logging.INFO)
            stream_handler = logging.StreamHandler(stream=mock_stdout)
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(stream_handler)
            # Use async queue listener
            async with AsyncQueueListener(logger):
                # Log won't block the event loop
                for _ in range(5):
                    logger.info("log info")
                    logger.debug("log debug")
            text = mock_stdout.getvalue()
            assert text.count("log info") == 5, text
            assert text.count("log debug") == 0, text
            return True

    assert asyncio.run(_test()) is True


def test_utils():
    test_LogHelper()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "test_LogHelper passed")
    test_AsyncQueueListener()
    print(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "test_AsyncQueueListener passed",
    )


def test():
    global __name__
    __name__ = "morebuiltins.logs"
    import doctest

    doctest.testmod()
    test_utils()


if __name__ == "__main__":
    test()

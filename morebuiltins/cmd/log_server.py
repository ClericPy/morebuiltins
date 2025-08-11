import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
import typing
from collections import Counter, namedtuple
from pathlib import Path
from queue import Empty, Queue

from ..logs import RotatingFileWriter
from ..ipc import SocketLogHandlerEncoder, SocketServer
from ..utils import format_error, read_size, ttime

__all__ = ["LogServer"]

QueueMsg = namedtuple("QueueMsg", ["name", "text", "file_args"])


class LogServer(SocketServer):
    """Log Server for SocketHandler, create a socket server with asyncio.start_server. Update settings of rotation/formatter with extra: {"max_size": 1024**2, "formatter": logging.Formatter(fmt="%(asctime)s - %(filename)s - %(message)s")}

    Server side:
        python -m morebuiltins.cmd.log_server --log-dir=./logs --host 127.0.0.1 --port 8901

    Client side:

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

    More docs:
        python -m morebuiltins.cmd.log_server -h
        usage: log_server.py [-h] [--host HOST] [--port PORT] [--log-dir LOG_DIR] [--name NAME] [--server-log-args SERVER_LOG_ARGS] [--handle-signals HANDLE_SIGNALS] [--max-queue-size MAX_QUEUE_SIZE]
                            [--max-queue-buffer MAX_QUEUE_BUFFER] [--log-stream LOG_STREAM]

        options:
        -h, --help            show this help message and exit
        --host HOST
        --port PORT
        --log-dir LOG_DIR     log dir to save log files, if empty, log to stdout with --log-stream
        --name NAME           log server name
        --server-log-args SERVER_LOG_ARGS
                                max_size,max_backups for log files, default: 10485760,5 == 10MB each log file, 1 name.log + 5 backups
        --handle-signals HANDLE_SIGNALS
        --max-queue-size MAX_QUEUE_SIZE
                                max queue size for log queue, log will be in memory queue before write to file
        --max-queue-buffer MAX_QUEUE_BUFFER
                                chunk size of lines before write to file
        --log-stream LOG_STREAM
                                log to stream, if --log-stream='' will mute the stream log

    """

    STOP_SIG = object()

    def __init__(
        self,
        host="127.0.0.1",
        port=8901,
        log_dir=None,
        name="log_server",
        server_log_args=(10 * 1024**2, 5),
        handle_signals=(2, 15),
        max_queue_size=100000,
        max_queue_buffer=20000,
        log_stream=sys.stdout,
        compress=False,
    ):
        super().__init__(
            host,
            port,
            handler=self.default_handler,
            encoder=SocketLogHandlerEncoder(),
            start_callback=self.start_callback,
            end_callback=self.end_callback,
        )
        # ensure has running loop
        self.loop = asyncio.get_running_loop()
        self.name = name
        self.server_log_args = {
            "max_size": server_log_args[0],
            "max_backups": server_log_args[1],
        }
        self.log_stream = log_stream
        self.compress = compress
        self.log_dir = Path(log_dir).resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(exist_ok=True, parents=True)
        self.handle_signals = handle_signals
        self.max_queue_size = max_queue_size
        self.max_queue_buffer = max_queue_buffer

        self._opened_files: typing.Dict[str, RotatingFileWriter] = {}
        self._write_queue = Queue(maxsize=max_queue_size)
        self._shutdown_signals = 0
        self._write_chunks = []
        self._lines_counter = Counter()
        self._size_counter = Counter()
        self._queue_consumer_task = self.loop.run_in_executor(
            self._default_executor, self.write_queue_consumer
        )
        for sig in handle_signals:
            signal.signal(sig, self.handle_signal)

    async def end_callback(self):
        await self.loop.run_in_executor(
            self._default_executor, self._write_queue.put, self.STOP_SIG
        )
        await self._queue_consumer_task

    def start_callback(self):
        self.send_log(
            f"started log server on {self.host}:{self.port}, handle_signals={self.handle_signals}, max_queue_size={self.max_queue_size}, max_queue_buffer={self.max_queue_buffer}, log_stream={getattr(self.log_stream, 'name', None)}, compress={self.compress}, log_dir={self.log_dir}"
        )

    def send_log(
        self, msg: str, error: typing.Optional[Exception] = None, level="INFO"
    ):
        now = time.time()
        ms = str(now).split(".", 1)[1][:3]
        if error:
            msg = f"{msg} | {format_error(error)}"
        msg = f"{ttime(now)},{ms} | {level: >5} | {msg}"
        self._write_queue.put((self.name, msg, self.server_log_args))

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
                    fw = RotatingFileWriter(
                        self.log_dir.joinpath(f"{name}.log"),
                        max_size=max_size,
                        max_backups=max_backups,
                        compress=self.compress,
                    )
                except Exception as e:
                    self.send_log(
                        f"error in get_targets({name!r}, {max_size!r}, {max_backups!r})",
                        e,
                        level="ERROR",
                    )
                targets.append(
                    self._opened_files.setdefault(
                        name,
                        fw,
                    )
                )
        return targets

    def write_queue_consumer(self):
        self.send_log("started write_worker daemon")
        stopped = False
        interval = 30
        last_log_time = time.time()
        while not stopped:
            try:
                new_lines = {}
                for index in range(self.max_queue_buffer):
                    try:
                        if index == 0:
                            try:
                                q_msg: QueueMsg = self._write_queue.get(timeout=1)
                            except Empty:
                                if self._write_queue.qsize():
                                    continue
                                else:
                                    break
                        else:
                            q_msg = self._write_queue.get_nowait()
                        if q_msg is self.STOP_SIG:
                            if not stopped:
                                self.send_log("stopping write_worker daemon")
                                new_lines[q_msg] = q_msg
                                stopped = True
                            continue
                        name, line, file_args = q_msg
                        if name in new_lines:
                            data = new_lines[name]
                            data["file_args"] = file_args
                        else:
                            data = {"file_args": file_args, "lines": []}
                            new_lines[name] = data
                        data["lines"].append(line)
                    except Empty:
                        break
                if new_lines:
                    self._write_chunks.append(new_lines)
                    for name, data in new_lines.items():
                        if name is self.STOP_SIG:
                            stopped = True
                            continue
                        file_args = data["file_args"]
                        lines = data["lines"]
                        targets = self.get_targets(name, **file_args)
                        for log_file in targets:
                            try:
                                if log_file is self.log_stream:
                                    head = f"[{name}] "
                                    body = f"\n{head}".join(lines)
                                    lines_text = f"{head}{body}\n"
                                else:
                                    lines_text = "\n".join(lines)
                                log_file.write(f"{lines_text}\n")
                                log_file.flush()
                            except Exception as e:
                                self.send_log(
                                    f"error in write_worker ({name})", e, level="WARN"
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
                self.send_log("error in write_queue_consumer", e, level="ERROR")
                print(format_error(e), file=sys.stderr, flush=True)
                traceback.print_exc()
                self.shutdown()
                break

    async def default_handler(self, record: dict):
        # record demo:
        # {"name": "test_logger", "msg": "test socket logging message", "args": null, "levelname": "INFO", "levelno": 20, "pathname": "/PATH/temp.py", "filename": "temp.py", "module": "temp", "exc_info": null, "exc_text": null, "stack_info": null, "lineno": 38, "funcName": "main", "created": 1723270162.5119407, "msecs": 511.0, "relativeCreated": 102.74338722229004, "thread": 8712, "threadName": "MainThread", "processName": "MainProcess", "process": 19104}
        try:
            if not isinstance(record, dict):
                raise TypeError("item must be a dict")
            name = record["name"]
            if "formatter" in record:
                text = record["formatter"].format(logging.LogRecord(level=0, **record))
            else:
                text = record["msg"]
            file_args = {
                k: v for k, v in record.items() if k in {"max_size", "max_backups"}
            }
            q_msg = QueueMsg(name=name, text=text, file_args=file_args)
            self._write_queue.put_nowait(q_msg)
        except Exception as e:
            self.send_log("error in default_handler", e, level="WARN")
        finally:
            del record

    def handle_signal(self, sig, frame):
        self._shutdown_signals += 1
        if self._shutdown_signals > 5:
            os._exit(1)
        msg = f"received signal: {sig}, count: {self._shutdown_signals}"
        self.send_log(msg)
        self.shutdown()
        self._write_queue.put(self.STOP_SIG)
        if self._shutdown_ev:
            self.loop.call_soon_threadsafe(self._shutdown_ev.set)

    def __del__(self):
        for f in self._opened_files.values():
            f.close()


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8901, type=int)
    parser.add_argument(
        "-t",
        "--log-dir",
        default="",
        dest="log_dir",
        help="log dir to save log files, if empty, log to stdout with --log-stream",
    )
    parser.add_argument("--name", default="log_server", help="log server name")
    parser.add_argument(
        "--server-log-args",
        default="10485760,5",
        dest="server_log_args",
        help="max_size,max_backups for log files, default: 10485760,5 == 10MB each log file, 1 name.log + 5 backups",
    )
    parser.add_argument("--handle-signals", default="2,15", dest="handle_signals")
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
        default="sys.stdout",
        help="log to stream, if --log-stream='' will mute the stream log",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="compress log files with gzip",
    )
    args = parser.parse_args()
    log_stream = {"sys.stdout": sys.stdout, "sys.stderr": sys.stderr, "": ""}[
        args.log_stream
    ]
    async with LogServer(
        host=args.host,
        port=args.port,
        log_dir=args.log_dir,
        name=args.name,
        server_log_args=tuple(map(int, args.server_log_args.split(","))),
        handle_signals=tuple(map(int, args.handle_signals.split(","))),
        max_queue_size=args.max_queue_size,
        max_queue_buffer=args.max_queue_buffer,
        log_stream=log_stream,
        compress=args.compress,
    ) as ls:
        await ls.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())

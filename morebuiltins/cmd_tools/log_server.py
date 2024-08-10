import asyncio
import logging
import logging.handlers
import os
import signal
import sys
import time
import typing
from collections import namedtuple
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from morebuiltins.functools import RotatingFileWriter
from morebuiltins.ipc import SocketLogHandlerEncoder, SocketServer
from morebuiltins.utils import format_error, ttime

QueueMsg = namedtuple("QueueMsg", ["name", "text", "file_args"])


class LogServer(SocketServer):
    def __init__(
        self,
        host="127.0.0.1",
        port=8901,
        log_dir=None,
        name="log_server",
        server_log_args=(10 * 1024**2, 5),
        handle_signals=(2, 15),
        max_queue_size=100000,
        max_queue_buffer=1000,
        log_stream=sys.stdout,
    ):
        super().__init__(
            host,
            port,
            handler=self.default_handler,
            encoder=SocketLogHandlerEncoder(),
            start_callback=self.start_callback,
        )
        self.name = name
        self.server_log_args = {
            "max_size": server_log_args[0],
            "max_backups": server_log_args[1],
        }
        self.log_stream = log_stream
        self.log_dir = Path(log_dir).resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(exist_ok=True, parents=True)
        self.opened_files: typing.Dict[str, RotatingFileWriter] = {}
        self.handle_signals = handle_signals
        for sig in handle_signals:
            signal.signal(sig, self.handle_signal)
        self.max_queue_size = max_queue_size
        self.write_queue = Queue(maxsize=max_queue_size)
        self.max_queue_buffer = max_queue_buffer
        self.file_writer_thread = Thread(target=self.write_worker)

        self._shutdown_signals = 0

    def start_callback(self):
        self.file_writer_thread.start()
        self.send_log(
            f"started log server on {self.host}:{self.port}, handle_signals={self.handle_signals}, max_queue_size={self.max_queue_size}, max_queue_buffer={self.max_queue_buffer}, log_stream={getattr(self.log_stream, 'name', None)}, log_dir={self.log_dir}"
        )

    def send_log(
        self, msg: str, error: typing.Optional[Exception] = None, level="INFO"
    ):
        now = time.time()
        ms = str(now).split(".", 1)[1][:3]
        if error:
            msg = f"{msg} | {format_error(error)}"
        msg = f"{ttime(now)},{ms} | {level: >5} | {msg}"
        self.write_queue.put((self.name, msg, self.server_log_args))

    def get_targets(self, name: str, max_size=5 * 1024**2, max_backups=1):
        result = []
        if self.log_stream:
            result.append(self.log_stream)
        if self.log_dir:
            if name in self.opened_files:
                rfw = self.opened_files[name]
                if rfw.max_size != max_size:
                    rfw.max_size = max_size
                if rfw.max_backups != max_backups:
                    rfw.max_backups = max_backups
                result.append(rfw)
            else:
                try:
                    rfw = RotatingFileWriter(
                        self.log_dir.joinpath(f"{name}.log"),
                        max_size=max_size,
                        max_backups=max_backups,
                    )
                except Exception as e:
                    self.send_log(
                        f"error in get_targets({name!r}, {max_size!r}, {max_backups!r})",
                        e,
                        level="ERROR",
                    )
                result.append(
                    self.opened_files.setdefault(
                        name,
                        rfw,
                    )
                )
        return result

    def write_worker(self):
        while not self._shutdown_ev.is_set():
            new_lines = {}
            for _ in range(self.max_queue_buffer):
                try:
                    name, line, file_args = self.write_queue.get(timeout=1)
                    if name in new_lines:
                        data = new_lines[name]
                        data["file_args"] = file_args
                    else:
                        data = {"file_args": file_args, "lines": []}
                        new_lines[name] = data
                    data["lines"].append(line)
                except Empty:
                    break
            for name, data in new_lines.items():
                file_args = data["file_args"]
                targets = self.get_targets(name, **file_args)
                for log_file in targets:
                    if log_file == self.log_stream:
                        head = f"[{name}] "
                    else:
                        head = ""
                    try:
                        for line in data["lines"]:
                            log_file.write(f"{head}{line}\n")
                        # only flush once per file
                        log_file.flush()
                    except Exception as e:
                        self.send_log(
                            f"error in write_worker ({name})", e, level="WARN"
                        )

    async def default_handler(self, item: dict):
        # item demo:
        # {"name": "test_logger", "msg": "test socket logging message", "args": null, "levelname": "INFO", "levelno": 20, "pathname": "/PATH/temp.py", "filename": "temp.py", "module": "temp", "exc_info": null, "exc_text": null, "stack_info": null, "lineno": 38, "funcName": "main", "created": 1723270162.5119407, "msecs": 511.0, "relativeCreated": 102.74338722229004, "thread": 8712, "threadName": "MainThread", "processName": "MainProcess", "process": 19104}
        try:
            if not isinstance(item, dict):
                raise TypeError("item must be a dict")
            name = item["name"]
            if "formatter" in item:
                text = item["formatter"].format(logging.LogRecord(level=0, **item))
            else:
                text = item["msg"]
            file_args = {
                k: v for k, v in item.items() if k in {"max_size", "max_backups"}
            }
            q_msg = QueueMsg(name=name, text=text, file_args=file_args)
            self.write_queue.put_nowait(q_msg)
        except Exception as e:
            self.send_log("error in default_handler", e, level="WARN")

    def handle_signal(self, sig, frame):
        self._shutdown_signals += 1
        if self._shutdown_signals > 5:
            os._exit(1)
        self.shutdown()
        self.send_log(f"received signal: {sig}, count: {self._shutdown_signals}")

    def __del__(self):
        for f in self.opened_files.values():
            f.close()


async def main():
    async with LogServer(log_dir="./logs") as server:
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())

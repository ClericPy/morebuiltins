import atexit
import os
import sys
import time
import typing
import warnings
from contextlib import closing, suppress
from multiprocessing.shared_memory import SharedMemory

__all__ = ["PLock", "SharedBytes"]


class PLock:
    """A simple process lock using shared memory, for singleton control.
    Use `with` context or `close_atexit` to clean up the shared memory when the process exits normally.

    If the process crashes (or is killed), that cleanup does not run. On POSIX, `SharedMemory` registers the
    segment with the `resource_tracker` by default, and that tracker is a separate process which may outlive
    the holder and still unlink the segment at its own exit, so "leaks until manually removed" only holds when
    tracking is disabled (`track=False`) or the tracker itself is gone (e.g. the whole process group was
    killed). On Windows the segment is freed by the kernel once every handle is closed, and `unlink()` is a
    no-op, so a crash does not leave the name behind either. Use `force=True` or `kill_with_name` to take over
    or clear leftovers.

    Args:
        name (str): name of the shared memory
        force (bool, optional): when True and the segment already exists, the stored pid is unconditionally
            rewritten to self.pid without signaling the holder — racy by design: the previous holder may still
            be alive and believe it owns the lock, so only force-rewrite when that process is known dead.
            Defaults to False.
        force_signum (int, optional): signal number sent to the current holder before the pid is rewritten,
            e.g. 15 (SIGTERM); the rewrite is skipped when the signal cannot be delivered, and this only takes
            effect together with `force=True`. `force` alone never signals the holder; `force_signum` alone
            (without force) is ignored. Defaults to None (no signal).
        close_atexit (bool, optional): whether to close the shared memory at process exit. Defaults to False, to use __del__ or __exit__ instead.

    Demo:

        >>> test_pid = 123456 # test pid, often set to None for current process
        >>> plock = PLock("test_lock", force=False, close_atexit=True, pid=test_pid)
        >>> plock.locked
        True
        >>> try:
        ...     plock2 = PLock("test_lock", force=False, close_atexit=True, pid=test_pid + 1)
        ...     raise RuntimeError("Should not be here")
        ... except RuntimeError:
        ...     True
        True
        >>> plock3 = PLock("test_lock", force=True, close_atexit=True, pid=test_pid + 1)
        >>> plock3.locked
        True
        >>> plock.locked
        False
        >>> PLock.wait_for_free(name="test_lock", timeout=0.1, interval=0.01)
        False
        >>> plock.close()
        >>> plock3.close()
        >>> PLock.wait_for_free(name="test_lock", timeout=0.1, interval=0.01)
        True
    """

    DEFAULT_SIZE = 4  # 4 bytes, means 2^32 = 4GB
    DEFAULT_BYTEORDER: typing.Literal["little", "big"] = "little"

    def __init__(
        self,
        name: str,
        force=False,
        close_atexit=False,
        pid=None,
        force_signum: typing.Optional[int] = None,
    ):
        self.name = name
        self.pid = pid or os.getpid()
        self.force = force
        self.force_signum = force_signum
        self.shm = None
        # whether the shared memory is closed
        self._closed = False
        # whether this instance ever acquired the lock; a failed init never
        # did, so its close() must not re-check ownership or unlink
        self._locked = False
        if close_atexit:
            atexit.register(self.close)
        self.init()
        if self.shm is None:
            raise RuntimeError(f"Failed to create shared memory {name}")

    @staticmethod
    def is_free(name: str) -> bool:
        """Check if the shared memory is free."""
        try:
            with closing(SharedMemory(name=name)):
                return False
        except FileNotFoundError:
            return True

    @classmethod
    def kill_with_name(cls, name: str, sig_num=15):
        """Kill the process that holds the shared memory.

        The stored pid is validated and probed before signaling, so garbage
        content (e.g. pid 0, which on POSIX means the whole process group of
        the caller) never reaches os.kill.
        """
        if cls.is_free(name):
            return False
        else:
            with closing(SharedMemory(name=name)) as shm:
                buf = shm.buf
                # an attached segment always exposes its buffer; guard so a
                # platform that reports None fails loudly instead of as a TypeError
                if buf is None:
                    raise RuntimeError(f"Shared memory {name} has no buffer")
                mem_pid = int.from_bytes(buf[: cls.DEFAULT_SIZE], byteorder=cls.DEFAULT_BYTEORDER)
            # POSIX pid_max defaults to 2**22; Windows pids are DWORDs, so
            # applying 2**22 there would reject every valid high pid
            max_pid = 2**32 - 1 if sys.platform == "win32" else 2**22
            if not (0 < mem_pid <= max_pid):
                # anything beyond the platform pid range is garbage data
                return False
            try:
                if sys.platform != "win32":
                    # probe existence: on POSIX os.kill(pid, 0) sends no signal
                    os.kill(mem_pid, 0)
                # no probe on win32: os.kill(pid, 0) TerminateProcess'es the
                # target there, a stale pid surfaces as OSError below instead
                os.kill(mem_pid, sig_num)
                return True
            except OSError:
                # ProcessLookupError on POSIX; on Windows error 87
                # (ERROR_INVALID_PARAMETER) for a nonexistent pid, or
                # PermissionError for a pid we are not allowed to signal
                return False

    @classmethod
    def wait_for_free(cls, name: str, timeout=3, interval=0.1):
        """Wait for the shared memory to be free."""
        start = time.time()
        while True:
            free = cls.is_free(name)
            if free:
                return True
            elif time.time() - start > timeout:
                return False
            else:
                time.sleep(interval)
                continue

    def init(self):
        if self._closed:
            raise RuntimeError("Already closed")
        ok = True
        try:
            self.shm = SharedMemory(name=self.name, create=True, size=self.DEFAULT_SIZE)
        except FileExistsError:
            self.shm = SharedMemory(name=self.name)
            if self.force:
                if self.force_signum is not None:
                    ok = self.kill_with_name(self.name, sig_num=self.force_signum)
            else:
                ok = False
        if not ok:
            mem_pid = self.mem_pid
            self.close()
            raise RuntimeError(f"Locked by another process: {mem_pid} != {self.pid}(self)")
        self.set_mem_pid()
        if not self.locked:
            raise ValueError(f"Failed to write PID to shared memory. {self.mem_pid} != {self.pid}(self)")
        self._locked = True

    def set_mem_pid(self, pid=None):
        if pid is None:
            pid = self.pid
        self.buf[: self.DEFAULT_SIZE] = pid.to_bytes(self.DEFAULT_SIZE, byteorder=self.DEFAULT_BYTEORDER)

    @property
    def buf(self) -> memoryview:
        if self.shm is None:
            raise RuntimeError("Shared memory is not initialized")
        buf = self.shm.buf
        # an attached segment always exposes its buffer; guard so a platform
        # that reports None fails loudly instead of as a TypeError
        if buf is None:
            raise RuntimeError("Shared memory has no buffer")
        return buf

    def get_mem_pid(self):
        return int.from_bytes(self.buf[: self.DEFAULT_SIZE], byteorder=self.DEFAULT_BYTEORDER)

    @property
    def mem_pid(self):
        return self.get_mem_pid()

    @property
    def locked(self):
        return self.mem_pid == self.pid

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.shm:
            # close() is best-effort teardown: a failed close/unlink must not
            # break __del__ or the caller's cleanup path
            with suppress(Exception):
                # only a lock holder may unlink; an instance whose init failed
                # (e.g. it lost the lock race and never owned the segment) must
                # just detach, otherwise this branch would warn on every
                # normal failure path
                locked = self._locked and self.get_mem_pid() == self.pid
                self.shm.close()
                if locked:
                    # only unlink if self.pid is the owner of the shared memory
                    self.shm.unlink()
                elif self._locked:
                    # another process force-rewrote the pid after `locked` was
                    # computed; unlinking now would leave that owner with a
                    # vanished segment
                    warnings.warn(
                        f"PLock {self.name}: holder pid changed before unlink, "
                        "skip unlink to avoid removing another owner's segment",
                        stacklevel=2,
                    )

    def __enter__(self):
        return self

    def __exit__(self, *_, **_kwargs):
        self.close()

    def __del__(self):
        self.close()


class SharedBytes:
    """Shared Memory for Python, for python 3.8+.
    This module provides a simple way to create and manage shared memory segments, shared between different processes.
    Shared memory is faster than other IPC methods like pipes or queues, and it allows for direct access to the memory.

    Demo:

    >>> sb = SharedBytes(name="test", data=b"Hello, World!", unlink_on_exit=True)
    >>> # The size of the shared memory is 18 bytes (5 bytes for header + 13 bytes for data), but mac os may return more than 18 bytes.
    >>> sb.size > 10
    True
    >>> sb.get(name="test")
    b'Hello, World!'
    >>> sb.re_create(b"New Data")
    >>> sb.get(name="test")
    b'New Data'
    >>> sb.close()
    >>> sb.get(name="test", default=b"")  # close() unlinked the segment, so the default is returned
    b''
    """

    closed: bool = True
    # max_size: 2 ** (i * 8). 1: 256 B, 2: 64 KB, 3: 16 MB, 4: 4 GB, 5: 1 TB, 6: 256 TB, 7: 64 PB, 8: 16 EB, defaults to 5(1TB).
    head_length: int = 5
    byteorder: typing.Literal["little", "big"] = "little"

    def __init__(self, name: str, data: bytes, head_length=None, unlink_on_exit=False):
        self.name = name
        self.head_length = head_length or self.head_length
        if unlink_on_exit:
            atexit.register(self.close)
        self.create(data)

    @property
    def size(self) -> int:
        if self.closed:
            raise ValueError("Shared memory is closed")
        return self.shm.size

    def create(self, data: bytes):
        if not self.closed:
            raise ValueError("Shared memory is already created, please use re_create")
        size = len(data)
        head_length = self.head_length
        # >= : size == 2 ** (head_length * 8) does not fit into the header either
        if size >= 2 ** (head_length * 8):
            right_head_length = 0
            for i in range(head_length, 15):
                if size < 2 ** (i * 8):
                    right_head_length = i
                    break
            raise ValueError(
                f"data size {size} is too large, max size is {2 ** (head_length * 8) - 1}, "
                f"raise head_length at least {right_head_length}"
            )
        total_size = head_length + len(data)
        self.shm = SharedMemory(name=self.name, create=True, size=total_size)
        head = size.to_bytes(head_length, byteorder=self.byteorder)
        self.buf[:total_size] = head + data
        self.closed = False

    @property
    def buf(self) -> memoryview:
        if self.shm is None:
            raise RuntimeError("Shared memory is not initialized")
        buf = self.shm.buf
        # an attached segment always exposes its buffer; guard so a platform
        # that reports None fails loudly instead of as a TypeError
        if buf is None:
            raise RuntimeError("Shared memory has no buffer")
        return buf

    def re_create(self, data: bytes):
        if self.closed:
            raise ValueError("Shared memory is closed")
        self.close()
        self.create(data)

    def close(self):
        if self.closed:
            return
        shm = self.shm
        # mark closed first, so a failure mid-way cannot leave close() half-done
        self.closed = True
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            # the segment was already removed (e.g. unlinked by another process)
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def get(cls, name: str, head_length=None, default=...) -> bytes:
        """Read back the data stored in the shared memory segment name.

        Args:
            name (str): name of the shared memory segment.
            head_length (int, optional): header length in bytes. Defaults to cls.head_length.
            default (optional): returned when the segment does not exist; if omitted (the ... sentinel), raise KeyError.
        """
        try:
            head_length = head_length or cls.head_length
            with closing(SharedMemory(name=name)) as shm:
                buf = shm.buf
                # an attached segment always exposes its buffer; guard so a
                # platform that reports None fails loudly instead of as a TypeError
                if buf is None:
                    raise RuntimeError(f"Shared memory {name} has no buffer")
                head = bytes(buf[:head_length])
                body_length = int.from_bytes(head, byteorder=cls.byteorder)
                data = bytes(buf[head_length : head_length + body_length])
                return data
        except FileNotFoundError as err:
            if default is ...:
                raise KeyError(f"Shared memory {name} not found") from err
            return default


if __name__ == "__main__":
    import doctest

    doctest.testmod()

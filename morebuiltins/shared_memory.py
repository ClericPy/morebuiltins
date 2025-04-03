import atexit
import os
import time
from multiprocessing import shared_memory


__all__ = [
    "PLock",
]


class PLock:
    """A simple process lock using shared memory, for singleton control.
    Use `with` context or `close_atexit` to ensure the shared memory is closed in case the process crashes.

    Args:
        name (str): name of the shared memory
        force (bool, optional): whether to force rewrite the existing shared memory. Defaults to False.
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
    DEFAULT_BYTEORDER = "little"

    def __init__(self, name: str, force=False, close_atexit=False, pid=None):
        self.name = name
        self.pid = pid or os.getpid()
        self.force = force
        self.shm = None
        # whether the shared memory is closed
        self._closed = False
        if close_atexit:
            atexit.register(self.close)
        self.init()

    @staticmethod
    def wait_for_free(name: str, timeout=3, interval=0.1):
        """Wait for the shared memory to be free."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                shared_memory.SharedMemory(name=name).close()
                time.sleep(interval)
            except FileNotFoundError:
                return True
        return False

    def init(self):
        if self._closed:
            raise RuntimeError("Already closed")
        ok = True
        try:
            self.shm = shared_memory.SharedMemory(
                name=self.name, create=True, size=self.DEFAULT_SIZE
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name)
            if not self.force:
                ok = False
        if not ok:
            mem_pid = self.mem_pid
            self.close()
            raise RuntimeError(
                f"Locked by another process: {mem_pid} != {self.pid}(self)"
            )
        self.set_mem_pid()
        if not self.locked:
            raise ValueError(
                f"Failed to write PID to shared memory. {self.mem_pid} != {self.pid}(self)"
            )

    def set_mem_pid(self, pid=None):
        if pid is None:
            pid = self.pid
        self.shm.buf[: self.DEFAULT_SIZE] = pid.to_bytes(
            self.DEFAULT_SIZE, byteorder=self.DEFAULT_BYTEORDER
        )

    def get_mem_pid(self):
        return int.from_bytes(
            self.shm.buf[: self.DEFAULT_SIZE], byteorder=self.DEFAULT_BYTEORDER
        )

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
            try:
                locked = self.locked
                self.shm.close()
                if locked:
                    # only unlink if self.pid is the owner of the shared memory
                    self.shm.unlink()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_, **_kwargs):
        self.close()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

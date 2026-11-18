## Changelog

### 1.4.1 (2026-11-18)
1. `morebuiltins.cmd.ui`: the Host/Origin allowlist follows the bind address (loopback IPv4/IPv6 plus the bind host; `[::1]:port` bind strings parse), the new `--web-allowed-hosts` supplies extra hosts for remote access, a rejected request answers 403 and prints the live allowlist to stderr, output produced on a non-owner thread is written back to the real stdout instead of being dropped, `SimpleHTTPRequestHandler` is kept as an alias of `FuncWebHandler`, `TKUI.get_result()` works without arguments, and an exception returns only `repr(err)` to the browser while the traceback goes to the process stderr.
2. `morebuiltins.cmd.log_server`: `end_callback` joins the consumer with `CONSUMER_JOIN_TIMEOUT` instead of hanging `close()`, `default_handler` names its parameter `record` again, and `__del__` tolerates a half-constructed instance.
   1. [**Compatibility Warnings**] binding a non-loopback host now raises ValueError at construction (`--host` exits with a parser error): the socket protocol has no authentication and every frame is unpickled; run on 127.0.0.1/::1 and forward logs over a trusted channel.
3. `morebuiltins.cmd.systemd.timer` and `morebuiltins.cmd.parse_deps`: timer parameters given as a list are emitted one line per value instead of a Python repr; relative imports whose base directory escapes `project_dir` are skipped.
4. `morebuiltins.zipapps`: `download_python` downloads to a temp file and replaces the target only after the digest verifies, so a failed download keeps the previous good file; unversioned assets sort after versioned ones; the interactive and auto selection paths share one matcher; MD5 fingerprinting falls back to SHA-256 when the interpreter is in FIPS mode or lacks `usedforsecurity`; `freezing` computes the UTC offset from the target timestamp, so the DST of that moment applies.
   1. [**Compatibility Warnings**] a failed or interrupted download keeps the previous tarball instead of deleting it.
5. `morebuiltins.request`: the `req` and `DomainParser` doctests run against a loopback echo server and a temporary public suffix list instead of `postman-echo.com` / `publicsuffix.org`, so `test.py` no longer needs outbound network.
6. `morebuiltins.shared_memory`: `PLock.close()` logs the "holder pid changed before unlink" race at debug level instead of emitting a `RuntimeWarning`, and `logs.test_LogHelper` clears stale `app_test.log*` files before counting so an interrupted earlier run no longer fails the suite.
   1. [**Compatibility Warnings**] `PLock.close()` no longer emits the `RuntimeWarning` for that race; rely on the `morebuiltins.shared_memory` debug log instead.

### 1.4.0 (2026-11-16)
1. `morebuiltins.date.Crontab`: invalid/out-of-range fields raise ValueError; fields may be separated by any whitespace; day-of-month/day-of-week follow Vixie cron (ORed only when both restricted); ValueError is raised when no match is found within max_tries instead of silently yielding nothing; `count` now means the number of distinct callback values (dedup uses a private sentinel, so falsy and `None` values are still deduplicated).
2. `morebuiltins.date.ScheduleTimer`: fix step lookup for negated patterns; dedup never worked because `last_item` was never assigned; dedup sentinel broke with falsy callback values.
3. `morebuiltins.utils.is_running`: fix UnicodeDecodeError on CJK Windows (tasklist output is locale-encoded, now read as bytes); `is_running_linux` treats EPERM as running instead of dead.
4. `morebuiltins.utils.base_decode`: default alphabet now matches `base_encode` (uppercase before lowercase).
   1. [**Compatibility Warnings**] decoding old mixed-case strings produced with the previous decode alphabet returns different values.
5. `morebuiltins.utils.find_jsons`: fix infinite loop when the first left bracket has no matched right bracket.
6. `morebuiltins.utils.base_encode`: negative input raises ValueError (used to loop forever); `get_hash` raises ValueError for `n` outside `1..len(digest)` (used to return an empty string silently).
7. `morebuiltins.utils.PathLock`: `timeout` contention raises TimeoutError only with the new `raise_on_timeout=True`; the default keeps the historical behaviour (the body still runs, without the lock, and no lock is released on exit); a timed-out waiter no longer removes another holder's entry or releases another holder's lock; the path matching keeps blocking a parent request while a child path is locked; the docstring no longer claims a whole-hierarchy guarantee.
   1. [**Compatibility Warnings**] `PathLock` is not reentrant, and holding a parent lock does not block new child requests (only the reverse direction is guarded); pass `raise_on_timeout=True` to get a TimeoutError instead of a block that runs unprotected.
8. `morebuiltins.utils.FileDict`: json/pickle handlers write via a temp file plus `os.replace`, closing the window where a crash between unlink and rename lost the file.
9. `morebuiltins.utils.SimpleFilter`: timer creation is under the lock; the pending timer is really cancelled on exit and is a daemon thread.
10. `morebuiltins.utils` docstrings corrected: `ptime` return type, `retry` default 2 (and `tries<1` now raises ValueError), `unique`/`slice_*` yield description, `i2b` upper-bound table, `gen_id` microsecond/entropy notes, `unix_rlimit` error message, `Validator` PEP 563 limitation; removed dead code in `set_pid_file`/`switch_flush_print` and the `return` inside `get_paste`'s finally.
11. `morebuiltins.logs`: `gzip_log` no longer loses both the original file and the compressed copy when rename fails; `backupCount<=0` keeps all backups (stdlib contract) instead of deleting them; rollover/compress failures are logged instead of swallowed; `do_compress_async` re-entry guard reads the right attribute; listener restore removes the tracked QueueHandler by identity; `bind_handler(filename=None)` detaches handlers without closing them and keeps their cache entries, so `close_all_handlers` can still reclaim the file handles; binding a second handler for the same file only closes handlers this class created, a user-attached `FileHandler` is left in place with a warning; `__del__` tolerates half-constructed instances; `stop()` unregisters its atexit hook; `comress_chunk_size` renamed to `compress_chunk_size` with the old name kept as a property alias.
   1. [**Compatibility Warnings**] `backupCount<=0` now keeps every rotated file (previously the extras were removed); detaching a logger no longer drops its handlers from `FILENAME_HANDLER_MAP`, so `close_all_handlers` closes handlers that are already detached; a second `FileHandler` on the same path that LogHelper did not create is no longer closed, only warned about.
12. `morebuiltins.funcs.NamedLock`: cache creation is race-free; `clear_unlocked` no longer drops a Semaphore that still holds permits; docstring warns that a timed-out acquire leaves the block unprotected.
   1. [**Compatibility Warnings**] the private `_SYNC_CACHE`/`_ASYNC_CACHE` values are now `(lock, initial_value)` tuples instead of bare locks, so direct readers must unpack.
13. `morebuiltins.funcs.func_cmd`: required parameters are only rejected when actually missing from the command line (previously every call with a required parameter raised); `auto_default=True` still fills `''`/`0` and is documented.
14. `morebuiltins.funcs.debounce`: `mode="trailing"` gives a real trailing debounce (window resets per call, only the last call's arguments run) and wrappers keep metadata via `functools.wraps`; the default `mode="leading"` keeps the previous behaviour (the first call runs immediately, calls inside the window return None), so existing callers do not change.
    1. [**Compatibility Warnings**] `mode="trailing"` does not run the decorated function immediately on first call; sync calls return None and the result of the final run is discarded; exceptions raised by the debounced function do not reach the caller (they surface via the thread/task exception handlers); a call whose window has not elapsed by loop close or interpreter exit is dropped. Callers must flush themselves before shutdown (no built-in flush API).
15. `morebuiltins.funcs`: `__all__` includes `countdown`/`LineTime`/`LineProfiler`/`line_profiler`; `lru_cache_ttl` keys on real arguments (kwargs order insensitive) and `maxsize<=0` no longer raises KeyError.
    1. [**Compatibility Warnings**] `lru_cache_ttl` cache keys are the argument tuple instead of the previous hash int, so code that inspects `<func>.cache` keys must change; `maxsize<=0` returns the function unchanged, so `.cache`/`.ttl_clean` are not set even with `controls=True`.
16. `morebuiltins.cmd.log_server`: `level_specs` survives a dump/from_dict round trip (custom `Level N` values were left as strings and broke reload); `wait_close_sync` no longer blocks forever; the queue consumer stops before files are closed; opened log files are LRU-capped (`MAX_OPENED_FILES`) to bound handles; `--idle-close-time` defaults to 60 like the class and has help text; unknown `--log-stream` values warn instead of silently muting; `__all__` exports `get_logger`/`create_handler`; docstring demo matches real output.
17. `morebuiltins.cmd.log_server`: client log names are validated (the resolved path must stay inside `log_dir`; absolute paths and `..` escapes are rejected, sub-directories are still allowed), client formatter payloads use a restricted unpickler (logging's builtin styles and `string.Template` are allowed), an invalid settings record or a single unformattable record no longer shuts the server down; queue-full no longer raises out of `close()`.
   1. [**Compatibility Warnings**] client formatter classes other than logging's builtin ones are rejected on the base64 settings path; log names escaping `log_dir` are rejected.
18. `morebuiltins.cmd.ui`: bool answers from `TKit.ask` are honoured (they were always False); each POST uses its own queue so a broken connection cannot leak output into the next request; worker output and window title are marshalled to the Tk main thread; POST validates Host/Origin, Content-Length and body size; `EOFError` exits cleanly; handler renamed to `FuncWebHandler`.
19. `morebuiltins.ipc`: sync handler replies are sent back (the executor future was not awaited); half-closed connections no longer busy-spin (readexactly); `close()` always runs the end callback and shuts the executor down; active writers are closed on Python 3.12-style teardown so `wait_closed` cannot deadlock; unsupported UDS on win32 raises NotImplementedError; `inspect.iscoroutinefunction` replaces the deprecated asyncio one.
    1. [**Compatibility Warnings**] `port=0` now means "let the OS pick a TCP port" instead of falling into the UDS branch.
20. `morebuiltins.sqlite`: docstring encoder sample passes callables; `count`/`items`/`keys`/`values`/`popitem` accept `literal=True` to match the prefix literally (`%`/`_` escaped), while the default keeps the historical LIKE wildcard behaviour; `recreate_table` honours `key_type`/`value_type`; table/column identifiers are backtick-escaped; `close()` tolerates a failed constructor; `copy`/`update`/`items` docstrings match real behaviour.
    1. [**Compatibility Warnings**] prefix matching still treats `%`/`_` as LIKE wildcards by default (the 1.4.0 draft escaped them unconditionally and changed `count(prefix="a_b")` from 2 rows to 1); pass `literal=True` for a literal prefix.
21. `morebuiltins.shared_memory`: `kill_with_name` validates the pid range and probes liveness before signalling (a zero/garbage prefix could kill the caller's process group on POSIX); `close()` re-checks ownership before unlinking; `SharedBytes.close()` survives an external unlink; size boundary off-by-one fixed; docstrings describe crash leftovers and `force` semantics honestly.
    1. [**Compatibility Warnings**] operations that detect a stale or foreign owner now emit a `RuntimeWarning` instead of proceeding silently.
22. `morebuiltins.tk`: `TextWindow` survives the user closing the window (TclError no longer replaces the body's exception); `max()` on string-only rows works; `tk.Tk()` failures no longer raise UnboundLocalError; the patched `print` forwards to the original when `file` is given.
23. `morebuiltins.request`: `verify`/`context` are honoured on the `build_opener_handlers`/`proxy` paths (`verify=False` only applies when the caller did not pass its own `context`); `custom_dns` chains installs and gains `restore_dns()`; `DomainParser` does no network IO in its constructor and downloads the PSL lazily with an atomic cache write; foreign `HTTPResponse` objects raise AttributeError instead of returning `b""`; `params` drops None values and still accepts key/value sequences; `data` empty values fall through to `json=`/`form=`; `RequestErrors` includes read timeouts (3.8/3.9 included); `update_url` replaces every duplicate key; Date header is locale-independent.
24. `morebuiltins.emails`: a failed login quits the SMTP session before re-raising; an unreadable body file raises instead of sending the path as the body; attachment paths are stripped; zero-byte attachments are still sent; body files default to utf-8.
    1. [**Compatibility Warnings**] a GBK body file that used to be read with the locale encoding now needs an explicit `encoding`; `prepare_attach("")` returns None instead of raising.
25. `morebuiltins.snippets.sql`: `delete(limit=...)` uses a `rowid IN (SELECT rowid ... ORDER BY rowid LIMIT n)` form that runs on stock CPython builds (it needs a real rowid: `WITHOUT ROWID` tables and tables with a plain `rowid` column are not supported); `clear_free_pages` vacuums non-incremental databases instead of doing nothing; `create_table` rejects `autoincrement` on composite/non-integer primary keys; `upsert` drops `OR REPLACE` so a UNIQUE conflict raises IntegrityError instead of silently deleting the conflicting row.
    1. [**Compatibility Warnings**] `upsert` on a non-target UNIQUE column now raises IntegrityError; `clear_free_pages` returns `(freelist_before, freelist_after)` and commits + VACUUMs non-incremental databases.
26. `morebuiltins.zipapps`: layer archives honour `compressed` (it was inverted); `--dump-config`/`--load-config` keep `rm_patterns`; freezing decodes with `errors="ignore"` and computes the timezone offset at call time; `._pth` handling matches mid-file `import site` and CRLF; pip probing uses subprocess; `activate_zipapps` uses the importlib spec API and registers the module in `sys.modules`; `download_python` fixes `-i -k` (empty filter exits cleanly), version ordering (`3.13.0` vs `3.12.10`), failure exit codes, progress division by zero, tar extraction filter, temp-file collisions, and verifies the asset's `digest` field (the releases do not publish `.sha256` sidecars).
    1. [**Compatibility Warnings**] `--unzip` refuses to overwrite an existing target dir without `--force`; download failures exit non-zero; a digest mismatch aborts (skipped only when the release provides no digest).
27. `morebuiltins.cmd.systemd`: systemctl return codes propagate (exit code reflects failure); `-Service__X=A=B` keeps the full value; `-RequiredBy/-Also/-Alias` are emitted; custom sections merge into existing ones; unit files are written as utf-8; missing `-ExecStart` prints help instead of a traceback; `-Environment` is repeatable on timers.
    1. [**Compatibility Warnings**] commands no longer prefix `sudo` (dry-run now matches what is executed).
28. `morebuiltins.cmd.parse_deps`: `import x` honours `ignore_stds` like `from x import y`; `from . import sibling` is resolved when the sibling module exists.
29. `morebuiltins.cmd.proxy_checker`: `-i` without `-o` writes to stdout as documented; `-c` takes precedence over `-i`; `-u` is actually used to build the probe request; `-r` help states it applies to the last loop only.
    1. [**Compatibility Warnings**] `-u` now sends an origin-form request line (`GET /path`) instead of an absolute-form one.
30. `publish.py` refuses to publish unless `--yes` is passed, the git tree is clean, CHANGELOG has the version section and the version is not already on PyPI; `--dry-run` runs the checks without uploading.
    1. [**Compatibility Warnings**] `python publish.py` no longer publishes; it prints usage and exits with code 2.

### 1.3.9 (2026-11-15) — withdrawn, do not use
1. Published by accident from an inconsistent working tree (some modules fixed, some still pre-fix). Superseded by 1.4.0; `publish.py` now refuses this class of mistake.

### 1.3.8 (2026-01-12)
1. add AUTO_CAST to `morebuiltins.utils.Validator` for automatic type conversion
2. add --dry-run/--test to cmd.systemd

### 1.3.7 (2025-12-05)
1. fix `morebuiltins.utils.Validator` while type is Union / Optional

### 1.3.6 (2025-11-26)
1. add `morebuiltins.funcs.async_call` to automatically call a function asynchronously, whether it's sync or async.
2. add `morebuiltins.funcs.countdown` to print a countdown timer inline in the console.
3. update `morebuiltins.utils.code_inline` using zlib instead of gzip.
4. add `morebuiltins.utils.Validator` string alias for bool type
   1. `Validator` now supports string values like "true", "false", "1", "0", "yes", "no", "on", "off" for boolean fields.

### 1.3.5 (2025-10-15)
1. fix writer.wait_closed() ConnectionResetError in ipc.py, proxy_checker.py, log_server.py
2. add `morebuiltins.funcs.debounce` decorator to debounce function calls.
3. add `sep` arg to `morebuiltins.utils.gen_id` to customize the separator between date and time.
4. add `LoggerStream` to `morebuiltins.logs` to redirect sys.stdout/sys.stderr with prefix.
   1. Use `sys.stdout = LoggerStream()` to redirect all stdout to LoggerStream, which sends logs to the logging system / file / custom writer.
   2. `LoggerStream.replace_print_ctx`: A context manager to temporarily replace the print function within a block of code.
      1. add time prefix to each print line.
5. add `nofmt` extra field support to `morebuiltins.cmd.log_server` to skip formatting for high-frequency/thinking logs.
6. add `scripts` entry to `pyproject.toml` for `morebuiltins` command line tool.
   1. now you can run `uvx morebuiltins morebuiltins.cmd.log_server` to start log server with uvx.

### 1.3.4 (2025-10-15)
1. fix `morebuiltins.utils.gen_id` wrong length issue.
2. fix `morebuiltins.funcs.LineProfiler` to handle nested function calls correctly.
3. update `morebuiltins.cmd.log_server` to support close idle file handlers after 300 seconds.
4. add `get_logger` to `morebuiltins.cmd.log_server` to get a **singleton** logger with `SocketHandler` to send logs to `LogServer`.
5. [**Compatibility Warnings**] Refactor of `morebuiltins.cmd.log_server`:
   1. use `LogSetting` struct instead of untyped dict, client send dict with `LogSetting` structure.
   2. add `level_specs` filter to specify log levels of files, like `[logging.ERROR, logging.CRITICAL]`.
   3. add cache file for log settings, readable by both server and human.
   4. THIS IS A BREAKING CHANGE, old clients may not work with new server.

### 1.3.3 (2025-09-01)
1. add `morebuiltins.logs.LogHelper.handle_crash` to log uncaught exceptions.
2. fix `morebuiltins.logs.LogHelper.bind_handler` to avoid adding duplicate queue handlers.
3. update default format of `morebuiltins.logs.LogHelper` to `"%(asctime)s | %(levelname)-5s | %(filename)+8s:%(lineno)+3s - %(message)s"`.

### 1.3.2 (2025-08-02)
1. fix typing-hint for `morebuiltins.funcs.threads` decorator, now it returns `Callable[..., Future]`.
2. add force kill methods to `morebuiltins.shared_memory.PLock`:
   1. add `force_signum` to `morebuiltins.shared_memory.PLock`, to specify the signal number to force kill the process if it exists.
   2. add `is_free` to `morebuiltins.shared_memory.PLock`, to check if the lock is free.
   3. add `kill_with_name` to `morebuiltins.shared_memory.PLock`, to kill the process with the given name and signal number.
3. add `morebuiltins.logs.LogHelper` to quickly bind a logging handler to a logger, with a StreamHandler or SizedTimedRotatingFileHandler.
4. **Compatibility Warnings**:
   1. move `async_logger`, `AsyncQueueListener`, `LogHelper`, `RotatingFileWriter`, and `SizedTimedRotatingFileHandler` from `morebuiltins.funcs` to `morebuiltins.logs`.
5. add `ContextFilter` for logging, to add context variables to log records.
6. add `queue` arg to `morebuiltins.logs.LogHelper.bind_handler`, to enable async logging with `AsyncQueueListener`. (like loguru's queue=True)

### 1.3.1 (2025-06-19)
1. add `check_recursion` to `morebuiltins.funcs`, to check if a function is recursive.
2. add `SharedBytes` to `morebuiltins.shared_memory`, a shared memory for bytes, similar to `multiprocessing.shared_memory.SharedMemory` with a header for size.
3. add `line_profiler` decorator to `morebuiltins.funcs`, to profile function execution time line by line.

### 1.3.0 (2025-03-08)
1. **Compatibility Warning**. rename `morebuiltins.functools` to `morebuiltins.funcs` to avoid conflict with `functools` in python standard library.
2. add `key_type` arg to `sqlite.KV`, to support `int` key.
3. add `utils.cut_file` to cut file with `a+b` mode to limit the file size
4. add recheck for `utils.set_pid_file`
5. add `shared_memory.PLock` for singleton process with `multiprocessing.shared_memory`, support linux and windows.
6. add `--StandardOutput` and `--StandardError` to `morebuiltins.cmd.systemd.service`, and update `--Environment` to list type.
7. add `is_port_free` to `morebuiltins.ipc`.


### 1.2.3 (2025-03-07)
1. add `sqlite.KV`: A key-value store using sqlite3, lightweight `sqlitedict`.

### 1.2.2 (2025-03-04)
1. Change the default alphabet in `utils.base_encode` to "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", which follows the order of ASCII characters.
2. add `utils.SnowFlake` for id generator

### 1.2.1 (2025-03-03)
1. add `utils.get_size` to get size of objects recursively.
2. add `base_encode, base_decode, gen_id, timeti` to `utils`
3. add alias for AsyncQueueListener: `functools.async_logger`

### 1.1.9 (2025-01-16)
1. add `snippets.sql.SqliteSQL` as Sqlite SQL generator
2. add `cmd.parse_deps` to parse dependencies of a project directory, and find circular dependencies.
3. add `cmd.os.linux.systemd.service` and `cmd.os.linux.systemd.timer`
4. add `functools.to_thread`, same as `asyncio.to_thread` in python 3.9+.
5. add `functools.AsyncQueueListener` for asyncio non-block logging.
6. fix systemd.timer typing hint error

### 1.1.8 (2024-12-11)
1. add `utils.i2b` and `utils.b2i`, integer and fixed-length byte strings conversion.
2. add `--compress` to `cmd.log_server`
3. add `snippets.event.EventTemplate`

### 1.1.7 (2024-11-19)
1. fix `utils.code_inline` gzip compress `mtime` to 1, to ensure the same compressed data
2. fix `utils.FileDict.save`
3. add `ipc.QueueManager` based on BaseManager, add JSON listener
4. add `compress`, `ensure_dir` arg to `funtools.SizedTimedRotatingFileHandler`
5. add `compress` arg to `funtools.RotatingFileWriter`
6. update `utils.format_error` default filter skip from "site-packages" to "-packages"

### 1.1.6 (2024-09-09)
1. add filename_filter to utils.format_error
2. add `functools.get_function` to find function from entrypoint
   1. set the type default to str while strict=False and no default/annotation
3. add `default` `button_text` to `TKit.ask_text`
4. add `morebuiltins.cmd.ui` with `--cli`/`--gui`, `--web`
   1. try `python -m morebuiltins.cmd.ui -e re:findall --cli`
   2. try `python -m morebuiltins.cmd.ui -e re:findall --gui1`
   3. try `python -m morebuiltins.cmd.ui -e re:findall --gui2`
   4. try `python -m morebuiltins.cmd.ui -e test_func_web --web --web-open --web-timeout=30`
5. add default doc for `morebuiltins.cmd.ui` --gui
6. `morebuiltins.functools.FuncSchema` changed
   1. `parse` will see arg type as `str` while strict=False and no default/annotation
   2. add `to_string` to FuncSchema

### 1.1.5 (2024-08-29)
1. add `utils.get_hash_int`

### 1.1.4 (2024-08-25)
1. modify default args for `cmd.proxy_checker`

### 1.1.3 (2024-08-12)
1. add `morebuiltins.cmd.proxy_checker` --max-result for quick return
2. fix `morebultins.cmd.log_server` lost log in high frequency writing

### 1.1.2 (2024-08-11)
1. add `cmd.proxy_checker`
   1. `python -m morebuiltins.cmd.proxy_checker -c` to check proxy, input from clipboard and output to clipboard
2. add utils.Clipboard as same api as pyperclip

### 1.1.1 (2024-08-10)
1. add `cmd.log_server`
2. add start_callback end_callback to `ipc.SocketServer`
   1. ipc: self.handler(self, item) -> self.handler(item)
3. add `RotatingFileWriter.flush`, and default flush=False
4. fix `StreamWriter.__del__` fails if event loop is already closed

### 1.1.0 (2024-08-04)
1. update 2024.08.07 zipapps https://github.com/ClericPy/zipapps/releases/tag/2024.08.07
2. add `functools.RotatingFileWriter`
   1. prepare for `log_server` with `asyncio` + `ipc.SocketLogHandlerEncoder`

### 1.0.9 (2024-08-04)
1. fix default_dict type-hint
2. fix read num zero div
3. add build_opener_handlers+proxy to request.req

### 1.0.8 (2024-07-16)
1. add `utils.PathLock`

### 1.0.7 (2024-07-15)
1. add `functools.file_import`

### 1.0.6 (2024-07-10)
1. fix `utils.Validator` typing-hint class

### 1.0.5 (2024-07-09)
1. add `add py.typed`
2. update `utils.Validator` default to `STRICT=True`

### 1.0.4 (2024-07-04)
1. fix `functools.func_cmd` multi-line docstring
2. add `utils.FileDict`

### 1.0.3 (2024-07-03)
1. add `functools.get_type_default`
2. add `functools.func_cmd`
3. add `emails.SimpleEmail`

### 1.0.2 (2024-07-01)
1. update `utils.unix_rlimit` default to None
2. add `utils.weight_dict`
3. add `utils.SimpleFilter`

### 1.0.1 (2024-06-30)
1. `functools.FuncSchema.parse` default strict=True

### 1.0.0 (2024-06-29)
1. add `tk.TextWindow`
2. add `utils.unix_rlimit`
3. release as the first stable version

### 0.0.9 (2024-06-26)
1. add `tk.TKit.ask_text`
2. focus_force for `tk.TKit`

### 0.0.8 (2024-06-25)
1. update default format of `utils.format_error`
2. add `functools.SizedTimedRotatingFileHandler`
3. add `utils.switch_flush_print`
4. add `tk.TKit`

### 0.0.7 (2024-06-19)
1. add `utils.is_running_linux`, `utils.is_running_win32`
2. add `functools.InlinePB`
3. add `date.ScheduleTimer`, `date.Crontab`

### 0.0.6 (2024-06-19)
1. add `utils.xor_encode_decode`
2. add `utils.is_running`, `utils.lock_pid_file`
3. add `request.make_response`
4. add `utils.get_paste`(tk), `utils.set_clip`(win32)
5. add `utils.custom_dns`
6. update `utils.format_error` support slice index

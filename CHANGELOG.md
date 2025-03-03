### 1.2.0 (2025-03-03)
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

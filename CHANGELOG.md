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

import argparse
import inspect
import json
import os
import sys
import time
import tkinter as tk
import traceback
import typing
import webbrowser
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from string import Template
from threading import Thread, Timer
from urllib.parse import parse_qsl, unquote_plus

from ..funcs import FuncSchema, get_function

__all__ = ["launch_ui", "handle_cli", "handle_web", "handle_tk1", "handle_tk2"]


class TKUI:
    def __init__(self, function, kwargs):
        self.function = function
        self.func_name = getattr(function, "__name__", str(function))
        self.root = tk.Tk()
        self.root.title(self.func_name)
        self.center_window()
        self.schema = kwargs
        self.inputs = {}
        self.create_inputs()
        self.origin_std = (sys.stdout, sys.stderr)
        sys.stdout = TextRedirector(self.text_output, "stdout")
        sys.stderr = TextRedirector(self.text_output, "stderr")
        doc = getattr(self.function, "__doc__", "")
        if doc:
            print(doc)

    def __del__(self):
        sys.stdout, sys.stderr = self.origin_std

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = screen_width // 3
        window_height = int(screen_height * 0.6)
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        self.root.geometry(
            "{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate)
        )

    def create_inputs(self):
        for key, value in self.schema.items():
            frame = tk.Frame(self.root)
            frame.pack(anchor="w", padx=10, pady=(5, 0))
            tp = getattr(value["type"], "__name__", value["type"])
            default = value["default"]
            if default is inspect._empty:
                default_msg = ", required"
            else:
                default_msg = f", default={default!r}"
            text = f"{key}\t[type={tp}{default_msg}]"
            label = tk.Label(
                frame,
                text=text,
                anchor="w",
                justify="left",
                fg="#111111",
            )
            label.config(font=("", 12))
            label.pack(side=tk.LEFT)
            if value["type"] is bool:
                var = tk.BooleanVar()
                checkbox = tk.Checkbutton(frame, variable=var)
                var.set(default)
                checkbox.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X)
                self.inputs[key] = (value["type"], default, var)
            else:
                entry = tk.Text(self.root, height=3)
                if default is not inspect._empty:
                    entry.insert(tk.END, FuncSchema.to_string(value["default"]))
                entry.pack(anchor="w", padx=10, pady=(0, 5), fill=tk.X)
                self.inputs[key] = (value["type"], default, entry)
        self.button = tk.Button(self.root, text="Start", command=self.on_button_click)
        self.button.pack(pady=5, fill=tk.X)
        self.text_output = tk.Text(self.root)
        scrollbar = tk.Scrollbar(self.root, command=self.text_output.yview)
        self.text_output.config(yscrollcommand=scrollbar.set)
        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def get_input_values(self):
        values = {}
        missing = []
        for key, (tp, default, widget) in self.inputs.items():
            if isinstance(widget, tk.BooleanVar):
                values[key] = widget.get()
            else:
                text = widget.get("1.0", tk.END)
                if text.endswith("\n"):
                    text = text[:-1]
                value = FuncSchema.convert(text, tp) if text else default
                if value is inspect._empty:
                    missing.append(f"`{key}`")
                values[key] = value
        if missing:
            return ValueError(f"{', '.join(missing)} required")
        return values

    def get_result(self):
        ok = "OK"
        size = 0
        start = time.time()
        head = ""
        try:
            func = self.function
            kwargs = self.get_input_values()
            if isinstance(kwargs, dict):
                head = f">>> {self.func_name}(**{kwargs})"[:200]
                result = func(**kwargs)
                if not isinstance(result, str):
                    result = repr(result)
            else:
                ok = "ERROR"
                result = repr(kwargs)
                head = result
            size = len(result)
            print(result, flush=True)
        except Exception:
            ok = "ERROR"
            traceback.print_exc(limit=1)
        finally:
            end = time.time()
            elapsed = int(1000 * (end - start))
            self.root.title(f"[{ok}, output: {size}, elapsed: {elapsed}ms] {head}")

    def on_button_click(self):
        self.text_output.delete(1.0, tk.END)
        self.button.config(state=tk.DISABLED)
        thread = Thread(target=self.get_result, daemon=True)
        thread.start()
        self.root.after(100, lambda: self.check_thread(thread))

    def check_thread(self, thread):
        if not thread.is_alive() and self.button.cget("state") == "disabled":
            self.button.config(state=tk.NORMAL)
        else:
            self.root.after(100, lambda: self.check_thread(thread))

    def run(self):
        self.root.mainloop()


class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.insert(tk.END, str, (self.tag,))
        self.widget.see(tk.END)

    def flush(self):
        pass


def get_result(func, kwargs):
    result = func(**kwargs)
    if not isinstance(result, str):
        result = repr(result)
    return result


def handle_cli(func):
    """Command Line Interface: interactive mode

    Args:
        func: a callable function
    """
    kwargs = FuncSchema.parse(func, strict=False)
    _kws: dict = {}
    for k, v in kwargs.items():
        tp = getattr(v["type"], "__name__", v["type"])
        default = v["default"]
        if default is inspect._empty:
            default_msg = ""
        else:
            default_msg = f", default={default!r}"
        while True:
            msg = f"Input the value of {k}(type={tp}{default_msg}):\n"
            value = input(msg) or v["default"]
            if value is inspect._empty:
                print(ValueError(f"{k} is required"))
                continue
            break
        _kws[k] = FuncSchema.convert(value, v["type"])
        print(">>>", k, "=", repr(_kws[k]), flush=True)
    print("Result:")
    print(">" * 50)
    print(get_result(func, _kws))
    print(">" * 50)


def handle_tk1(func):
    """Function to tkinter UI. (interactive mode)"""
    kwargs = FuncSchema.parse(func, strict=False)
    TKUI(func, kwargs).run()


def handle_tk2(func):
    """Function to tkinter UI."""
    from ..tk import TKit

    kwargs = FuncSchema.parse(func, strict=False)
    _kws: dict = {}
    for k, v in kwargs.items():
        tp = getattr(v["type"], "__name__", v["type"])
        default = v["default"]
        _kwargs: dict = {}
        if default is inspect._empty:
            default_msg = ""
        else:
            default_msg = f", default={default!r}"
        msg = f"Input the value of {k}\n(type={tp}{default_msg}):\n"
        _kwargs["message"] = msg
        arg: typing.Any = ""
        if v["type"] is bool:
            arg = True
            if default_msg:
                _kwargs["default"] = "yes" if default else "no"
        elif v["type"] is str:
            _kwargs["textarea"] = 1
            if default_msg:
                _kwargs["default"] = FuncSchema.to_string(v["default"])
        else:
            msg += 'JSON supported, like: [1, 2, 3] or {"a": 2}'
        while True:
            value = TKit.ask(arg, **_kwargs) or default
            if value is inspect._empty:
                TKit.error(f"{k} is required")
                continue
            break
        if v["type"] is bool:
            _kws[k] = value == "yes"
        else:
            _kws[k] = FuncSchema.convert(value, v["type"])
        print(">>>", k, "=", repr(_kws[k]), flush=True)
    print("Result:")
    print(">" * 50)
    result = get_result(func, _kws)
    print(result, flush=True)
    TKit.ask_text("Result:", textarea=1, button_text="OK", default=result)
    print(">" * 50, flush=True)


class QueueStd:
    stdout = sys.stdout
    stderr = sys.stderr

    def __init__(self, queue: Queue, function, time_prefix=False):
        self.queue = queue
        self.time_prefix = time_prefix
        self.f_code = function.__code__
        self.newline = True

    def write(self, string):
        for stack in inspect.stack():
            # self.stdout.write(f"{stack.function}\n")
            # self.stdout.flush()
            if stack.frame.f_code == self.f_code:
                break
        else:
            self.stdout.write(string)
            return
        if self.time_prefix:
            if self.newline:
                string = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {string}'
            if string.endswith("\n"):
                self.newline = True
            else:
                self.newline = False
        self.queue.put(string)

    def flush(self):
        pass


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    template = Template(r"""<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Web UI: $title</title>
    <style>
        body {
            width: 95%;
            height: 95%;
        }
        #container {
            width: 90%;
            margin: 50px auto;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        @media (min-width: 768px) {
            #container {
                width: 30%;
            }
        }
        label,
        textarea {
            display: block;
            width: 100%;
            box-sizing: border-box;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        textarea,
        input[type="checkbox"],
        input[type="submit"] {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        input[type="submit"] {
            width: 100%;
            background-color: #0D7C66;
            color: white;
            cursor: pointer;
        }
        .inline-input {
            display: inline-block;
            width: 100%;
            margin-right: 10px;
            display: flex;
            align-items: center;
        }
        .inline-input input { margin: 0 10px 0 10px }
        .inline-input label { max-width: 30%; }
    </style>
</head>
<body>
    <div id="container">
        <form target="_blank" action="/submit" method="post">
            $labels
            <input type="submit" value="Submit">
            <div class="inline-input"><label>stdout: <input type="checkbox" name="__handle_stdout__" value="1" checked></label>  <label>time: <input type="checkbox" name="__print_time__" value="1" checked></label></div>
            <hr>
            <b>schema:</b>
            <pre>$schema</pre>
        </form>
    </div>
</body>
</html>
""")
    function: typing.Optional[typing.Callable] = None
    title = ""
    schema: dict = {}
    schema_string = ""
    queue: Queue = Queue()
    done = object()
    httpd: typing.Optional[HTTPServer] = None
    last_alive = time.time()
    keepalive_timeout = 60

    @classmethod
    def setup_function(cls, function):
        cls.function = function
        cls.title = function.__name__
        cls.schema = FuncSchema.parse(function, strict=False)
        cls.schema_string = escape(
            json.dumps(
                cls.schema,
                default=lambda obj: None
                if obj is inspect._empty
                else getattr(obj, "__name__", str(obj)),
                # default=lambda obj: None if obj is inspect._empty else str(obj),
                ensure_ascii=False,
                indent=2,
            )
        )

    def make_labels(self):
        # {'string': {'type': <class 'str'>, 'default': <class 'inspect._empty'>}, 'encoding': {'type': <class 'str'>, 'default': 'utf-8'}, 'errors': {'type': <class 'str'>, 'default': 'replace'}}
        # <label>Name1:<textarea name="text 1" rows="2"></textarea></label>
        # <label class="inline"><span>ChooseYes</span> <input type="checkbox" name="bool" value="1" checked> </label>
        result = []
        for key, value in self.schema.items():
            if value["type"] is bool:
                default = value["default"]
                default_true = default_false = False
                if default is not inspect._empty:
                    if default:
                        default_true = True
                    else:
                        default_false = True
                line = f"""<b>{escape(key)}</b>:<div class="inline-input"><label><input type="radio" name="{key}" value="true" {"checked" if default_true else ""}> True </label><label><input type="radio" name="{key}" value="false" {"checked" if default_false else ""}> False</label></div>"""
                result.append(line)
            else:
                if value["default"] is inspect._empty:
                    default = ""
                else:
                    default = escape(FuncSchema.to_string(value["default"]))
                line = f'<label><b>{escape(key)}</b>:<textarea name="{key}" rows="3">{default}</textarea></label>'
                result.append(line)
        return "\n".join(result)

    def do_GET(self):
        if self.__class__.last_alive:
            self.__class__.last_alive = time.time()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(
            self.template.substitute(
                title=self.title,
                labels=self.make_labels(),
                schema=self.schema_string,
            ).encode("utf-8")
        )

    def decode_value(self, key, value):
        value = unquote_plus(value).replace(os.linesep, "\n")
        if key in self.schema:
            return FuncSchema.convert(value, self.schema[key]["type"])
        else:
            return value

    def setup_std(self, function, time_prefix=False):
        sys.stdout = QueueStd(self.queue, function=function, time_prefix=time_prefix)
        sys.stderr = QueueStd(self.queue, function=function, time_prefix=time_prefix)

    def restore_std(self):
        sys.stdout = QueueStd.stdout
        sys.stderr = QueueStd.stderr

    def run_function(self, kwargs: dict):
        __handle_stdout__ = kwargs.pop("__handle_stdout__", False)
        __print_time__ = kwargs.pop("__print_time__", False)
        function = self.__class__.function
        if not function:
            self.queue.put(f"No function to run {kwargs}")
            return
        result = ""
        try:
            self.__class__.last_alive = 0
            if __handle_stdout__:
                self.setup_std(function=function, time_prefix=__print_time__)
            result = function(**kwargs)
            if not isinstance(result, str):
                result = repr(result)
        except Exception as err:
            result = repr(err)
        finally:
            if __print_time__:
                self.queue.put(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {result}")
            else:
                self.queue.put(result)
            self.queue.put(self.done)
            if __handle_stdout__:
                self.restore_std()
            self.__class__.last_alive = time.time()

    def do_POST(self):
        if self.path == "/submit":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8", "replace")
            kwargs = {
                key: self.decode_value(key, value)
                for key, value in parse_qsl(post_data)
            }
            t = Thread(target=self.run_function, args=(kwargs,), daemon=True)
            t.start()
            self.send_response(200)
            self.end_headers()
            self.wfile.write("<pre>".encode("utf-8"))
            while True:
                line = self.queue.get()
                if line is self.done:
                    break
                self.wfile.write(escape(line).encode("utf-8"))
                self.wfile.flush()
            self.wfile.write("</pre>".encode("utf-8"))

    @classmethod
    def keepalive_daemon(cls):
        while True:
            if cls.last_alive:
                if time.time() - cls.last_alive > cls.keepalive_timeout:
                    cls.httpd.shutdown()
                    print("keepalive timeout", cls.keepalive_timeout, flush=True)
                    break
            else:
                # 程序执行中，不判断存活
                pass
            time.sleep(1)


def handle_web(
    function, bind="127.0.0.1:8080", open_browser=False, keepalive_timeout=60
):
    """Function to Web UI.

    Args:
        function: callable function
        bind (str, optional): Defaults to "127.0.0.1:8080".
        open_browser (bool, optional): auto open browser. Defaults to False.
        keepalive_timeout (int, optional): shutdown if no request after timeout. Defaults to 60.
    """
    host, port = bind.split(":")
    port = int(port)
    server_address = (host, port)
    SimpleHTTPRequestHandler.setup_function(function)
    SimpleHTTPRequestHandler.keepalive_timeout = keepalive_timeout
    with HTTPServer(server_address, SimpleHTTPRequestHandler) as httpd:
        SimpleHTTPRequestHandler.httpd = httpd
        print(f"Starting httpd on port http://{host}:{port}", flush=True)
        if open_browser:
            Timer(1, webbrowser.open_new_tab, (f"http://{host}:{port}",)).start()
        t = Timer(3, SimpleHTTPRequestHandler.keepalive_daemon)
        t.daemon = True
        t.start()
        httpd.serve_forever()


def test_func_web(ok: bool = True, a={"<b>123</b>123\n&gt;	&#62;111\n111\n\n": 123}):
    "test string: html escape, html tag, unicode, enter, tab. type: dict, bool"
    for _ in range(5):
        print(str(_) * 100)
        time.sleep(0.2)
    return f"a={a} ok={ok!r}"


def launch_ui():
    r"""Interactive mode, value can be parsed by JSON parser.
    
Demo::

    > python -m morebuiltins.cmd.ui -e urllib.parse:urlparse --cli

    Input the value of url(type=str):
    http://github.com
    >>> url = 'http://github.com'
    Input the value of scheme(type=str, default=''):

    >>> scheme = ''
    Input the value of allow_fragments(type=bool, default=True):
    false
    >>> allow_fragments = False
    result:
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ParseResult(scheme='http', netloc='github.com', path='', params='', query='', fragment='')
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    1. try `python -m morebuiltins.cmd.ui -e re:findall --cli`
    2. try `python -m morebuiltins.cmd.ui -e re:findall --gui1`
    3. try `python -m morebuiltins.cmd.ui -e re:findall --gui2`
    4. try `python -m morebuiltins.cmd.ui -e test_func_web --web --web-open --web-timeout=30`

    try a custom function:

    def test_func_web(ok: bool = True, a={"<b>123</b>123\n&gt;	&#62;111\n111\n\n": 123}):
        "test string: html escape, html tag, unicode, enter, tab. type: dict, bool"
        for _ in range(5):
            print(str(_) * 100)
            time.sleep(0.2)
        return f"a={a} ok={ok!r}"

    """.strip()
    parser = argparse.ArgumentParser(usage=launch_ui.__doc__)
    parser.add_argument(
        "--func",
        "--entrypoint",
        "-e",
        dest="entrypoint",
        help="entrypoint, like `package.module:func` or `module:func`",
    )
    parser.add_argument("--cli", "-cli", action="store_true", help="run in cli mode")
    parser.add_argument("--web", "-web", action="store_true", help="run in web mode")
    parser.add_argument(
        "--web-timeout",
        "-web-timeout",
        type=int,
        default=60,
        help="web mode keepalive timeout",
    )
    parser.add_argument(
        "--web-bind", "-bind", default="127.0.0.1:8080", help="web mode bind address"
    )
    parser.add_argument(
        "--web-browser",
        "--web-open",
        "-browser",
        action="store_true",
        help="web mode open browser",
    )
    parser.add_argument(
        "--gui",
        "-gui",
        "--gui1",
        "-gui1",
        action="store_true",
        help="run in gui mode(tkinter)",
    )
    parser.add_argument(
        "--gui2", "-gui2", action="store_true", help="run in gui mode(TKit)"
    )
    parser.add_argument(
        "--path", dest="path", help="insert to index 0 of sys.path", default=""
    )
    args = parser.parse_args()
    if not args.entrypoint:
        raise ValueError("entrypoint is required")
    if args.path:
        import sys

        sys.path.insert(0, args.path)
    try:
        func = get_function(args.entrypoint)
    except ModuleNotFoundError:
        func = globals()[args.entrypoint]
    if args.cli:
        handle_cli(func)
    elif args.gui:
        handle_tk1(func)
    elif args.gui2:
        handle_tk2(func)
    elif args.web:
        handle_web(
            func,
            bind=args.web_bind,
            open_browser=args.web_browser,
            keepalive_timeout=args.web_timeout,
        )
    else:
        handle_cli(func)


if __name__ == "__main__":
    launch_ui()

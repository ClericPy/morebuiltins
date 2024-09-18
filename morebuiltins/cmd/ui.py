import argparse
import inspect
import sys
import time
import tkinter as tk
import traceback
import typing
from threading import Thread

from ..functools import FuncSchema, get_function


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
                    entry.insert(tk.END, value["default"])
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


def handle_cli(func, kwargs: dict):
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


def handle_tk1(func, kwargs: dict):
    TKUI(func, kwargs).run()


def handle_tk2(func, kwargs: dict):
    from ..tk import TKit

    _kws: dict = {}
    for k, v in kwargs.items():
        tp = getattr(v["type"], "__name__", v["type"])
        default = v["default"]
        _kwargs: dict = {}
        if default is inspect._empty:
            default_msg = ""
        else:
            default_msg = f", default={default!r}"
        msg = f"Input the value of {k}(type={tp}{default_msg}):\n"
        _kwargs["message"] = msg
        arg: typing.Any = ""
        if v["type"] is bool:
            arg = True
            if default_msg:
                _kwargs["default"] = "yes" if default else "no"
        elif v["type"] is str:
            _kwargs["textarea"] = 1
            if default_msg:
                _kwargs["default"] = v["default"]
        else:
            msg += 'JSON supported, like: [1, 2, 3] or {"a": 2}'
        while True:
            value = TKit.ask(arg, **_kwargs) or default
            if value is inspect._empty:
                TKit.error(f"{k} is required")
                continue
            break
        _kws[k] = FuncSchema.convert(value, v["type"])
        print(">>>", k, "=", repr(_kws[k]), flush=True)
    print("Result:")
    print(">" * 50)
    result = get_result(func, _kws)
    print(result, flush=True)
    TKit.ask_text("Result:", textarea=1, button_text="OK", default=result)
    print(">" * 50, flush=True)


def main():
    usage = """Interactive mode, value can be parsed by JSON parser.
    
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
    """.strip()
    parser = argparse.ArgumentParser(usage=usage)
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
        "--gui", "-gui", action="store_true", help="run in gui mode(tkinter)"
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
    func = get_function(args.entrypoint)
    kwargs = FuncSchema.parse(func, strict=False)
    print("FuncSchema:", kwargs, flush=True)
    if args.cli:
        handle_cli(func, kwargs)
    elif args.gui:
        handle_tk1(func, kwargs)
    elif args.gui2:
        handle_tk2(func, kwargs)
    elif args.web:
        raise NotImplementedError("web mode is not supported yet")
    else:
        handle_cli(func, kwargs)


if __name__ == "__main__":
    main()

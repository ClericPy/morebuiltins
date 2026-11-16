import builtins
import tkinter as tk
import tkinter.messagebox
from threading import Thread
from tkinter import scrolledtext
from typing import Any, Dict, cast

__all__ = ["TKit"]


class TKit(tk.Tk):
    r"""Tkinter kit for dialog usages.
    Demo::

        def examples():
            while True:
                TKit.ask(0, "0")
                TKit.ask(1, "1")
                TKit.ask(2, "2")
                if TKit.ask(True, "Choose NO", default="no") is True:
                    TKit.ask(0, "Wrong choice")
                    continue
                if (
                    TKit.ask((["1"], ["2", "3"], "4", ["5", "6", "7"]), message="Choose 3:")
                    != "3"
                ):
                    TKit.ask(1, "Wrong choice")
                    continue
                if TKit.ask(
                    [["1"], ["2", "3"], "4", ["5", "6", "7"]],
                    message="Choose 3 and 6:",
                    width=400,
                ) != ["3", "6"]:
                    TKit.ask(2, "Wrong choice")
                    continue
                result = TKit.ask("Input text 1 (press Enter to submit):")

                if result != "1":
                    TKit.ask(2, "Wrong text %s" % repr(result))
                    continue
                result = TKit.ask("Input text 1\\n (press Ctrl-Enter to submit):", textarea=1)

                if result != "1\n":
                    TKit.ask(2, "Wrong text %s" % repr(result))
                    continue

                def test_text(flush=False):
                    import time

                    for i in range(50):
                        print(f"Test print flush={flush} -- {i}", flush=flush)
                        time.sleep(0.02)
                    return "OK"

                with TKit.text_context(
                    test_text,
                    flush=True,
                    __resize_kwargs={"title": "The Title", "toolwindow": True},
                    __text_kwargs={"font": "_ 15"},
                ) as result:
                    TKit.info("result=%s" % result)

                with TKit.text_context(
                    test_text,
                    flush=False,
                    __resize_kwargs={"title": "The Title", "toolwindow": True},
                    __text_kwargs={"font": "_ 15"},
                ) as result:
                    TKit.warn("result=%s" % result)
                break

        examples()
    """

    DEFAULT_WIDTH_RATIO = 4
    DEFAULT_HEIGHT_RATIO = 4

    def resize(
        self,
        width=0,
        height=0,
        pos=(None, None),
        topmost=False,
        alpha=1.0,
        fullscreen=False,
        toolwindow=False,
        disabled=False,
    ):
        return self.resize_tk(
            self,
            width=width,
            height=height,
            pos=pos,
            topmost=topmost,
            alpha=alpha,
            fullscreen=fullscreen,
            toolwindow=toolwindow,
            disabled=disabled,
        )

    @classmethod
    def resize_tk(
        cls,
        tk_obj,
        width=0,
        height=0,
        pos=(None, None),
        topmost=False,
        alpha=1.0,
        fullscreen=False,
        toolwindow=False,
        disabled=False,
        title=None,
    ):
        """Resize and Move the window to the given position, default to the center of the screen."""
        screen_width = tk_obj.winfo_screenwidth()
        screen_height = tk_obj.winfo_screenheight()
        width = width or (screen_width // cls.DEFAULT_WIDTH_RATIO)
        height = height or (screen_height // cls.DEFAULT_HEIGHT_RATIO)
        x, y = pos
        if x is None or y is None:
            x = (screen_width // 2) - (width // 2)
            y = (screen_height // 2) - (height // 2)
        tk_obj.geometry(f"{width}x{height}+{x}+{y}")
        tk_obj.attributes("-topmost", topmost)
        tk_obj.attributes("-alpha", alpha)
        tk_obj.attributes("-fullscreen", fullscreen)
        tk_obj.attributes("-toolwindow", toolwindow)
        tk_obj.attributes("-disabled", disabled)
        if title is not None:
            tk_obj.title(title)

    @classmethod
    def ask(cls, arg: Any, message="", title="", **kwargs):
        if isinstance(arg, bool) or arg is bool:
            return cls.ask_yesno(arg, message, title, **kwargs)
        elif isinstance(arg, int):
            return cls.ask_msgbox(arg, message, title, **kwargs)
        elif isinstance(arg, tuple):
            return cls.ask_choice(arg, message, title, **kwargs)
        elif isinstance(arg, list):
            return cls.ask_checkbox(arg, message, title, **kwargs)
        elif isinstance(arg, str):
            return cls.ask_text(arg, message, title, **kwargs)
        else:
            raise TypeError()

    @classmethod
    def info(cls, message="", title="", **kwargs):
        return cls.ask_msgbox(0, message, title, **kwargs)

    @classmethod
    def warn(cls, message="", title="", **kwargs):
        return cls.ask_msgbox(1, message, title, **kwargs)

    @classmethod
    def error(cls, message="", title="", **kwargs):
        return cls.ask_msgbox(2, message, title, **kwargs)

    @classmethod
    def ask_msgbox(cls, arg: int, message="", title="", **kwargs):
        methods = (
            tkinter.messagebox.showinfo,
            tkinter.messagebox.showwarning,
            tkinter.messagebox.showerror,
        )
        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.focus_force()
            return methods[arg](title=title, message=message)
        except IndexError:
            return methods[-1](title=title, message=message)
        finally:
            if root is not None:
                root.destroy()

    @classmethod
    def ask_yesno(cls, arg: Any, message="Confirm(YES/NO)", title="", **kwargs: Any):
        # arg only drives dispatch in ask(); both True and the bool type are
        # accepted there, and this dialog does not render it
        # ask YES/NO
        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.focus_force()
            # default is forwarded verbatim from the caller's kwargs; tkinter
            # itself decides how to handle an absent/None value
            default: Any = kwargs.get("default")
            return tkinter.messagebox.askyesno(title=title, message=message, default=default)
        finally:
            if root is not None:
                root.destroy()

    @classmethod
    def ask_choice(cls, arg: tuple, message="Make your choice", title="", **kwargs):
        # Choose one
        root = cls()
        root.title(title)
        root.resize(topmost=True, toolwindow=True, **kwargs)
        results = [None]
        # default=1: rows may all be plain strings, so no sequence row exists;
        # max(1, ...) keeps an empty tuple/list row from yielding columnspan=0
        max_col = max(1, max((len(i) for i in arg if isinstance(i, (tuple, list))), default=1))
        tk.Label(root, text=message, background="#ffffff").grid(row=0, column=0, sticky="ewns", columnspan=max_col)
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        def cb(result):
            def add_result():
                results.append(result)
                root.destroy()

            return add_result

        for row, texts in enumerate(arg, 1):
            root.grid_rowconfigure(row, weight=1)
            if not isinstance(texts, (tuple, list)):
                texts = [texts]
            for col, text in enumerate(texts, 0):
                root.grid_columnconfigure(col, weight=1)
                b = tk.Button(
                    root,
                    text=str(text),
                    borderwidth=1,
                    relief="ridge",
                    command=cb(text),
                )
                b.grid(row=row, column=col, sticky="ewns")
                if (row, col) == (1, 0):
                    b.focus_force()
        root.mainloop()
        return results[-1]

    @classmethod
    def ask_checkbox(cls, arg: list, message="Make your choices", title="", **kwargs):
        # Choose many
        root = cls()
        root.title(title)
        root.resize(topmost=True, toolwindow=True, **kwargs)
        # default=1: rows may all be plain strings, so no sequence row exists;
        # max(1, ...) keeps an empty tuple/list row from yielding columnspan=0
        max_col = max(1, max((len(i) for i in arg if isinstance(i, (tuple, list))), default=1))
        tk.Label(root, text=message, background="#ffffff").grid(row=0, column=0, sticky="ewns", columnspan=max_col)
        root.grid_columnconfigure(0, weight=1)
        results: Dict[str, tk.IntVar] = {}
        for row, texts in enumerate(arg, 1):
            root.grid_rowconfigure(row, weight=1)
            if not isinstance(texts, (tuple, list)):
                texts = [texts]
            for col, text in enumerate(texts, 0):
                root.grid_columnconfigure(col, weight=1)
                c = tk.Checkbutton(
                    root,
                    text=str(text),
                    variable=results.setdefault(text, tk.IntVar()),
                    borderwidth=1,
                    relief="ridge",
                )
                c.grid(row=row, column=col, sticky="ewns")
                if (row, col) == (1, 0):
                    c.focus_force()
        tk.Button(
            root,
            text="Submit",
            background="#ffffff",
            command=lambda: root.destroy(),
            height=2,
        ).grid(row=len(arg) + 1, column=0, columnspan=max_col, sticky="ewns")
        root.mainloop()
        return sorted((k for k, v in results.items() if v.get()))

    @classmethod
    def ask_text(cls, arg: str, message="", title="", **kwargs):
        textarea = bool(kwargs.pop("textarea", None))
        default = kwargs.pop("default", "")
        button_text = kwargs.pop("button_text", "Submit")
        message = message or arg
        root = cls()
        root.title(title)
        root.resize(topmost=True, toolwindow=True, **kwargs)

        tk.Label(root, text=message, background="#ffffff").pack(expand=True, fill="both")

        text_var = tk.StringVar()

        # textbox is either tk.Text or tk.Entry; Any keeps the shared
        # call/bind surface usable without lying about either concrete widget
        text_box: Any
        if textarea:
            text_box = tk.Text(root, height=2)
            if default:
                text_box.insert("1.0", default)

            def submit(event=None):
                # remove \n
                # text_box is Text or Entry; only this branch builds a Text,
                # so the shared variable is narrowed back to its real widget
                text = cast(Any, text_box).get("1.0", tk.END)
                if text[-1] == "\n":
                    text = text[:-1]
                text_var.set(text)
                root.destroy()

        else:
            text_box = tk.Entry(root, textvariable=text_var)
            if default:
                text_box.insert(0, default)

            def submit(event=None):
                root.destroy()

            text_box.bind("<Return>", submit)
        text_box.bind("<Control-Return>", submit)
        text_box.pack(expand=True, fill="both")
        text_box.focus_force()
        tk.Button(root, text=button_text, background="#ffffff", command=submit).pack(expand=True, fill="both")
        root.mainloop()
        return text_var.get()

    @classmethod
    def text_context(cls, function, *args, **kwargs):
        return TextWindow(function, *args, **kwargs)


class TextWindow:
    def __init__(self, function, *args, **kwargs):
        self.print = print
        self.function = function
        self.args = args
        self.kwargs = kwargs

        self.result = None
        self.error = None
        self._shutdown = False

    def print_text(self, *args, end="\n", sep=" ", file=None, flush=False):
        """Replacement for builtins.print while the context is active.

        NOTE: this is an intentional redefinition: `flush=True` does not flush
        a stream, it clears the text box and redraws the line; `file` is not
        rendered here, it is forwarded to the original print unchanged.
        """
        if file is not None:
            return self.print(*args, end=end, sep=sep, file=file, flush=flush)
        if flush:
            self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, f"{sep.join(map(str, args))}{end}")
        self.text_box.see(tk.END)

    def patch_print(self):
        # deliberate monkeypatch: print_text mirrors print's public signature
        builtins.print = self.print_text  # ty: ignore[invalid-assignment]

    def restore_print(self):
        if builtins.print is not self.print:
            # restores the print captured in __init__; the narrowed value is a
            # checker artifact of the intentional monkeypatch above
            builtins.print = self.print  # ty: ignore[invalid-assignment]

    def run(self):
        try:
            self.result = self.function(*self.args, **self.kwargs)
        except Exception as e:
            self.error = e
        finally:
            self.shutdown()

    def __enter__(self):
        try:
            self.root = tk.Tk()
            __resize_kwargs = self.kwargs.pop("__resize_kwargs", {})
            TKit.resize_tk(self.root, **__resize_kwargs)
            __text_kwargs = self.kwargs.pop("__text_kwargs", {})
            self.text_box = scrolledtext.ScrolledText(self.root, **__text_kwargs)
            self.text_box.pack(expand=True, fill="both")
            self.text_box.focus_force()
            # title bar close (X) should run shutdown (quit) instead of the
            # default destroy, which would break the __exit__ withdraw path
            self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
            self.patch_print()
            t = Thread(target=self.run, daemon=True)
            t.start()
            self.root.mainloop()
        finally:
            self.shutdown()
            if self.error:
                raise self.error

        return self.result

    def shutdown(self):
        try:
            if not self._shutdown:
                self._shutdown = True
                self.restore_print()
                self.root.quit()
        except (tk.TclError, AttributeError):
            pass

    def __exit__(self, *_):
        try:
            self.root.withdraw()
        except tk.TclError:
            # the root may already be destroyed, e.g. closed by the title bar X
            pass
        self.shutdown()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def examples():
    while True:
        TKit.ask(0, "0")
        TKit.ask(1, "1")
        TKit.ask(2, "2")
        if TKit.ask(True, "Choose NO", default="no") is True:
            TKit.ask(0, "Wrong choice")
            continue
        if TKit.ask((["1"], ["2", "3"], "4", ["5", "6", "7"]), message="Choose 3:") != "3":
            TKit.ask(1, "Wrong choice")
            continue
        if TKit.ask(
            [["1"], ["2", "3"], "4", ["5", "6", "7"]],
            message="Choose 3 and 6:",
            width=400,
        ) != ["3", "6"]:
            TKit.ask(2, "Wrong choice")
            continue
        result = TKit.ask("Input text 1 (press Enter to submit):")
        if result != "1":
            TKit.ask(2, f"Wrong text {result!r}")
            continue
        result = TKit.ask("Input text 1\\n (press Ctrl-Enter to submit):", textarea=1)
        if result != "1\n":
            TKit.ask(2, f"Wrong text {result!r}")
            continue

        def test_text(flush=False):
            import time

            for i in range(50):
                print(f"Test print flush={flush} -- {i}", flush=flush)
                time.sleep(0.02)
            return "OK"

        with TKit.text_context(
            test_text,
            flush=True,
            __resize_kwargs={"title": "The Title", "toolwindow": True},
            __text_kwargs={"font": "_ 15"},
        ) as result:
            TKit.info(f"result={result}")

        with TKit.text_context(
            test_text,
            flush=False,
            __resize_kwargs={"title": "The Title", "toolwindow": True},
            __text_kwargs={"font": "_ 15"},
        ) as result:
            TKit.warn(f"result={result}")
        break


if __name__ == "__main__":
    examples()

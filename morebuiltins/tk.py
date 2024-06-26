import tkinter as tk
import tkinter.messagebox
from typing import Any, Dict

__all__ = ["TKit"]


class TKit(tk.Tk):
    """Tkinter kit for dialog usages."""

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
    def ask_msgbox(cls, arg: int, message="", title="", **kwargs):
        methods = (
            tkinter.messagebox.showinfo,
            tkinter.messagebox.showwarning,
            tkinter.messagebox.showerror,
        )
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.focus_force()
            return methods[arg](title=title, message=message)
        except IndexError:
            return methods[-1](title=title, message=message)
        finally:
            root.destroy()

    @classmethod
    def ask_yesno(cls, arg: bool, message="Confirm(YES/NO)", title="", **kwargs):
        # ask YES/NO
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.focus_force()
            return tkinter.messagebox.askyesno(
                title=title, message=message, default=kwargs.get("default")
            )
        finally:
            root.destroy()

    @classmethod
    def ask_choice(cls, arg: tuple, message="Make your choice", title="", **kwargs):
        # Choose one
        root = cls()
        root.title(title)
        root.resize(topmost=True, toolwindow=True, **kwargs)
        results = [None]
        max_col = max([len(i) for i in arg if isinstance(i, (tuple, list))])
        tk.Label(root, text=message, background="#ffffff").grid(
            row=0, column=0, sticky="ewns", columnspan=max_col
        )
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
        max_col = max([len(i) for i in arg if isinstance(i, (tuple, list))])
        tk.Label(root, text=message, background="#ffffff").grid(
            row=0, column=0, sticky="ewns", columnspan=max_col
        )
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
        message = message or arg
        root = cls()
        root.title(title)
        root.resize(topmost=True, toolwindow=True, **kwargs)

        tk.Label(root, text=message, background="#ffffff").pack(
            expand=True, fill="both"
        )

        text_var = tk.StringVar()

        if textarea:
            text_box: Any = tk.Text(root, height=2)

            def submit(event=None):
                # remove \n
                text = text_box.get("1.0", tk.END)
                if text[-1] == "\n":
                    text = text[:-1]
                text_var.set(text)
                root.destroy()

        else:
            text_box = tk.Entry(root, textvariable=text_var)

            def submit(event=None):
                root.destroy()

            text_box.bind("<Return>", submit)
        text_box.bind("<Control-Return>", submit)
        text_box.pack(expand=True, fill="both")
        text_box.focus_force()
        tk.Button(root, text="Submit", background="#ffffff", command=submit).pack(
            expand=True, fill="both"
        )
        root.mainloop()
        return text_var.get()


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
        break


if __name__ == "__main__":
    examples()

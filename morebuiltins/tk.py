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
        arg_list = list(arg)
        max_col = max([len(i) for i in arg_list if isinstance(i, (tuple, list))])
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

        for row, texts in enumerate(arg_list, 1):
            root.grid_rowconfigure(row, weight=1)
            if not isinstance(texts, (tuple, list)):
                texts = [texts]
            for col, text in enumerate(texts, 0):
                root.grid_columnconfigure(col, weight=1)
                tk.Button(
                    root,
                    text=str(text),
                    borderwidth=1,
                    relief="ridge",
                    command=cb(text),
                ).grid(row=row, column=col, sticky="ewns")
        root.mainloop()
        return results[-1]

    @classmethod
    def ask_checkbox(cls, arg: list, message="Make your choices", title="", **kwargs):
        # Choose many
        root = cls()
        root.title(title)
        root.resize(topmost=True, toolwindow=True, **kwargs)
        arg_list = list(arg)
        max_col = max([len(i) for i in arg_list if isinstance(i, (tuple, list))])
        tk.Label(root, text=message, background="#ffffff").grid(
            row=0, column=0, sticky="ewns", columnspan=max_col
        )
        root.grid_columnconfigure(0, weight=1)
        results: Dict[str, tk.IntVar] = {}
        for row, texts in enumerate(arg_list, 1):
            root.grid_rowconfigure(row, weight=1)
            if not isinstance(texts, (tuple, list)):
                texts = [texts]
            for col, text in enumerate(texts, 0):
                root.grid_columnconfigure(col, weight=1)
                tk.Checkbutton(
                    root,
                    text=str(text),
                    variable=results.setdefault(text, tk.IntVar()),
                    borderwidth=1,
                    relief="ridge",
                ).grid(row=row, column=col, sticky="ewns")
        tk.Button(
            root,
            text="Submit",
            background="#ffffff",
            command=lambda: root.destroy(),
            height=2,
        ).grid(row=len(arg_list) + 1, column=0, columnspan=max_col, sticky="ewns")
        root.mainloop()
        return sorted((k for k, v in results.items() if v.get()))

    @classmethod
    def ask_text(cls, arg: str, message="", title="", **kwargs):
        message = message or arg
        return tkinter.messagebox.showinfo(title=title, message=message)


def examples():
    while True:
        TKit.ask(0, "0")
        TKit.ask(1, "1")
        TKit.ask(2, "2")
        if TKit.ask(True, "Choose NO") is True:
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
        break


if __name__ == "__main__":
    examples()

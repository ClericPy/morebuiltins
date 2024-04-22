import tkinter as tk
from pathlib import Path
from queue import Queue

import zipapps


class GUI(object):
    window_width = 300
    window_height = 300

    def __init__(self):
        self.root = tk.Tk()
        self.q = Queue()
        self.kwargs = self.get_default_args()
        self.init()

    def init(self):
        self.init_position()

    @staticmethod
    def get_default_args():
        return {
            "includes": "",
            "cache_path": "None",
            "main": "",
            "output": (Path.cwd() / "app.pyz").absolute().as_posix(),
            "interpreter": None,
            "compressed": True,
            "shell": False,
            "unzip": "",
            "unzip_path": "",
            "ignore_system_python_path": False,
            "main_shell": False,
            "pip_args": [],
            "compiled": False,
            "build_id": "",
            "env_paths": "",
            "lazy_install": True,
            "sys_paths": "",
            "python_version_slice": 2,
            "ensure_pip": False,
            "layer_mode": False,
            "layer_mode_prefix": "python",
            "clear_zipapps_cache": False,
            "unzip_exclude": "",
            "chmod": "",
            "clear_zipapps_self": False
        }

    def init_position(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = int((screen_width / 2) - (self.window_width / 2))
        y = int((screen_height / 3) - (self.window_height / 2))
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

    def choose_enable(self):
        print({k: v.get() for k, v in self.enables.items()}, flush=True)

    def run(self):
        tk.Label(self.root,
                 text=f'$CWD={Path.cwd().absolute().as_posix()}').pack()
        self.enables = {}
        for index, k in enumerate(['-c', '-d', '-s', '-ss', '-a'], 1):

            cb = tk.Checkbutton(self.root,
                                text=k,
                                variable=self.enables.setdefault(
                                    k, tk.IntVar(self.root)),
                                onvalue=1,
                                offvalue=0,
                                command=self.choose_enable)
            cb.pack(side='left')

        inp = tk.Entry(self.root)
        inp.pack(fill='both', side='top', expand=True)

        def print_contents(_):
            print(inp.get(), flush=True)

        # 绑定输入框的值变化事件
        inp.bind('<KeyRelease>', print_contents)

        self.root.mainloop()


def main():
    GUI().run()


if __name__ == "__main__":
    main()

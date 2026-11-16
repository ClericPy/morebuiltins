__version__ = "1.4.0"

# this __all__ is used for documentation generation, not for imports: every
# entry is a module path (e.g. "morebuiltins.utils") intentionally absent as a
# name in this package, so __all__ checkers cannot validate it by design.
__all__ = [
    "morebuiltins.utils",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.date",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.funcs",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.ipc",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.request",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.download_python",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.tk",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.emails",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.cmd.log_server",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.cmd.proxy_checker",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.cmd.ui",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.cmd.parse_deps",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.snippets.event",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.snippets.sql",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.cmd.systemd.service",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.cmd.systemd.timer",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.sqlite",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.shared_memory",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
    "morebuiltins.logs",  # pyright: ignore[reportUnsupportedDunderAll]  # pyrefly: ignore[bad-dunder-all]
]

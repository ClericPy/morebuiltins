import argparse
import inspect
import typing

from ..functools import FuncSchema, get_function


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


def handle_tk(func, kwargs: dict, inline=False):
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
            if not inline:
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

    > python -m morebuiltins.cli -e urllib.parse:urlparse --cli

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
    """.strip()
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument(
        "--func",
        "--entrypoint",
        "-e",
        dest="entrypoint",
        help="entrypoint, like `package.module:func` or `module:func`",
    )
    parser.add_argument("--cli", action="store_true", help="run in cli mode")
    parser.add_argument("--web", action="store_true", help="run in web mode")
    parser.add_argument("--gui", action="store_true", help="run in gui mode(tkinter)")
    parser.add_argument(
        "--path", dest="path", help="insert to index 0 of sys.path", default=""
    )
    parser.add_argument(
        "--inline", action="store_true", help="inline text for gui mode"
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
        handle_tk(func, kwargs, inline=args.inline)
    elif args.web:
        raise NotImplementedError("web mode is not supported yet")
    else:
        handle_cli(func, kwargs)


if __name__ == "__main__":
    main()

import asyncio
import re
import sys
import time
import typing
from collections import namedtuple
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["ProxyChecker"]


proxy_info = namedtuple("proxy_info", "proxy ok cost error")


class ProxyChecker:
    """A command line toolkit to check available proxies.

    1. clipboard usage:
        > input-text from clipboard, and set the result to clipboard. `-l 2` means try 2 loops
            > python -m morebuiltins.cmd.proxy_checker -c -l 2

    2. input-file output-file usage:
        > input-text from file, and set the result to output-file
            > python -m morebuiltins.cmd.proxy_checker -i input.txt -o output.txt
        > output to stdout
            > python -m morebuiltins.cmd.proxy_checker -i input.txt

    3. stdin usage:
        > cat file.txt | python -m morebuiltins.cmd.proxy_checker > stdout.txt
        > cat file.txt | python -m morebuiltins.cmd.proxy_checker -o output.txt

    python -m morebuiltins.cmd.proxy_checker -h
        options:
        -h, --help            show this help message and exit
        -t TIMEOUT, --timeout TIMEOUT
                                timeout of each request
        -l LOOP, --loop LOOP  Loop the test to validate the successful results each time until the desired number of iterations is reached.
        --retry RETRY         retry times
        -n CONCURRENCY, --concurrency CONCURRENCY
                                concurrency
        -i INPUT_FILE, --input-file INPUT_FILE
                                input text file path
        -o OUTPUT_FILE, --output-file OUTPUT_FILE
                                output text file path
        -c, --from-clipboard  text from clipboard, ignore -i. if --output-file not set, output to clipboard
        -q, --quiet           mute the progress in stderr


    """

    target = re.compile(rb"HTTP/1.1 20\d [^\r\n]+\r\n")
    request = b"GET http://www.gstatic.com/generate_204 HTTP/1.1\r\nHost: www.gstatic.com\r\nConnection: close\r\n\r\n"
    ip_regex = re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?:\s+|:)(\d+)")

    @staticmethod
    def build_request(url: str) -> bytes:
        """Build the probe request for the given url in absolute-form.

        A forward proxy needs the request target in absolute-form
        (RFC 7230 section 5.3.2), i.e. `GET http://host/path HTTP/1.1`, so
        that it knows which origin to forward to; origin-form (`GET /path`)
        is only for requests sent directly to an origin server. The `Host`
        header still carries the plain authority.

        >>> ProxyChecker.build_request("http://www.gstatic.com/generate_204").split(b"\\r\\n")[0]
        b'GET http://www.gstatic.com/generate_204 HTTP/1.1'
        >>> ProxyChecker.build_request("http://example.com:8080/x?y=1").split(b"\\r\\n")[:2]
        [b'GET http://example.com:8080/x?y=1 HTTP/1.1', b'Host: example.com:8080']
        """
        split = urlsplit(url)
        if not split.scheme or not split.netloc:
            raise ValueError(f"url must be absolute: {url!r}")
        return (f"GET {url} HTTP/1.1\r\nHost: {split.netloc}\r\nConnection: close\r\n\r\n").encode()

    def __init__(self, concurrency=20, timeout=3.0, retry=0):
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.retry = retry

    @classmethod
    def find_proxies(cls, text: str):
        return [f"http://{host}:{port}" for host, port in cls.ip_regex.findall(text)]

    async def check_proxies(self, proxy_list: typing.List[str]) -> typing.AsyncGenerator[proxy_info, None]:
        tasks = [asyncio.create_task(self.check(proxy)) for proxy in proxy_list]
        for task in asyncio.as_completed(tasks):
            item = await task
            yield item

    async def send_request(self, host: str, port: int):
        writer: typing.Optional[asyncio.StreamWriter] = None
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(self.request)
            await writer.drain()
            return await reader.readline()
        finally:
            if writer:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass

    async def check(self, proxy: str):
        ok = cost = error = None
        for _ in range(self.retry + 1):
            start = time.time()
            try:
                match = self.ip_regex.search(proxy)
                if not match:
                    error = 'ValueError("invalid proxy")'
                    break
                host, port = match.groups()
                async with self.sem:
                    start = time.time()
                    result = await asyncio.wait_for(self.send_request(host, int(port)), timeout=self.timeout)
                ok = bool(self.target.search(result))
                if ok:
                    break
            except Exception as e:
                error = repr(e)
            finally:
                cost = round(time.time() - start, 3)

        return proxy_info(proxy, ok, cost, error)


# usage example
def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timeout", type=float, default=3.0, help="timeout of each request")
    parser.add_argument(
        "-l",
        "--loop",
        type=int,
        default=1,
        help="Loop the test to validate the successful results each time until the desired number of iterations is reached.",
    )
    parser.add_argument("--retry", type=int, default=0, help="retry times")
    parser.add_argument("-n", "--concurrency", type=int, default=20, help="concurrency")
    parser.add_argument(
        "-r",
        "--max-result",
        type=int,
        default=0,
        help="quick return when result count >= max-result (only takes effect on the last loop)",
    )
    parser.add_argument("-i", "--input-file", default="", help="input text file path")
    parser.add_argument("-o", "--output-file", default="", help="output text file path")
    parser.add_argument(
        "-c",
        "--from-clipboard",
        action="store_true",
        help="text from clipboard, ignore -i. if --output-file not set, output to clipboard",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="mute the progress in stderr")
    parser.add_argument(
        "-u",
        "--url",
        default="http://www.gstatic.com/generate_204",
        help="default check url",
    )
    parser.add_argument("-ii", "--stdin", action="store_true", help="set stdin as input")
    return parser.parse_args()


def _read_clipboard():
    """Return pasted text and the default copy target from the clipboard helpers."""
    from morebuiltins.utils import Clipboard

    # paste/copy are plain functions stored as class attributes (pyperclip
    # or the set_clip/get_paste fallback), so they are not bound methods
    clipboard = typing.cast(typing.Any, Clipboard)
    return clipboard.paste(), clipboard.copy


def _read_stdin():
    """Read stdin until four consecutive blank lines (or EOF)."""
    text = ""
    null_lines = 0
    for line in sys.stdin:
        if not line.strip():
            null_lines += 1
        if null_lines > 3:
            break
        text += line
    return text


def _read_input(args):
    """Read the proxy text and resolve the output destination."""
    quiet = args.quiet
    text = ""
    output_file = None
    from_clipboard = args.from_clipboard
    target_path = Path(args.output_file) if args.output_file else None
    if from_clipboard:
        # -c explicitly reads from clipboard, overriding -i and stdin
        text, copy_target = _read_clipboard()
        target_path = target_path or copy_target
    elif args.input_file:
        text = Path(args.input_file).read_bytes().decode("utf-8", "ignore")
        if not args.output_file:
            output_file = sys.stdout
            quiet = True
    elif args.stdin:
        text = _read_stdin()
        if not args.output_file:
            output_file = sys.stdout
            quiet = True
    else:
        text, copy_target = _read_clipboard()
        target_path = target_path or copy_target
    return text, target_path, output_file, quiet, from_clipboard


async def _check_loops(pc, proxies, args, quiet):
    result_list = []
    for loop in range(args.loop):
        proxies = set(result_list or proxies)
        result_list.clear()
        done = 0
        oks = 0
        last_loop = args.loop == loop + 1
        async for item in pc.check_proxies(proxies):
            done += 1
            if item.ok:
                oks += 1
                result_list.append(item.proxy)
                if last_loop and args.max_result and len(result_list) >= args.max_result:
                    break
            if not quiet:
                print(
                    f"[{loop + 1}] {done} / {len(proxies)} (oks={oks})".ljust(21),
                    item[0].ljust(28),
                    *item[1:],
                    flush=True,
                    file=sys.stderr,
                )
    return result_list


def _write_result(result_list, from_clipboard, output_file, target_path):
    if result_list or from_clipboard:
        result = "\n".join(result_list)
        if output_file:
            print(result, file=output_file, flush=True)
        elif target_path:
            target_path(result)


async def main():
    args = _parse_args()
    text, target_path, output_file, quiet, from_clipboard = _read_input(args)
    pc = ProxyChecker(concurrency=args.concurrency, timeout=args.timeout, retry=args.retry)
    # build the probe request from the given url
    pc.request = ProxyChecker.build_request(args.url)
    proxies = pc.find_proxies(text)
    try:
        if isinstance(target_path, Path):
            target_path = Path(args.output_file)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = open(target_path, "w")
        result_list = await _check_loops(pc, proxies, args, quiet)
        _write_result(result_list, from_clipboard, output_file, target_path)
    finally:
        if output_file and output_file is not sys.stdout:
            output_file.close()


if __name__ == "__main__":
    asyncio.run(main())

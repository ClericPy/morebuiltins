import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import tarfile
import time
import traceback
import typing
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Asset:
    # cpython-3.9.19+20240726-x86_64_v4-unknown-linux-musl-lto-full.tar.zst
    name: str
    # keywords=['cpython', '3.9.19', 'x86_64_v4', 'unknown', 'linux', 'musl', 'noopt-full.tar.zst']
    keywords: typing.List[str]
    # https://github.com/indygreg/python-build-standalone/releases/download/20240726/cpython-3.9.19%2B20240726-x86_64_v4-unknown-linux-musl-noopt-full.tar.zst
    url: str
    # 27204188
    size: int
    # "sha256:..." served by the GitHub release API, empty when not provided
    digest: str = ""


def get_assets():
    api = "https://api.github.com/repos/indygreg/python-build-standalone/releases/latest"
    req = urllib.request.Request(api, headers={"User-Agent": "Chrome"})  # noqa: S310  # fixed https api.github.com endpoint
    with urllib.request.urlopen(url=req, timeout=10) as resp:  # noqa: S310  # nosec B310  # fixed https api.github.com endpoint
        data = json.loads(resp.read())
        # print(data)
        urls = [
            Asset(
                i["name"],
                [k.replace(f"+{data['name']}", "") for k in i["name"].split("-", 6)],
                i["browser_download_url"],
                i["size"],
                i.get("digest") or "",
            )
            for i in data["assets"]
            if not i["name"].endswith(".sha256") and i["name"].count("-") >= 6
        ]
        return urls


def get_time():
    return time.strftime("%H:%M:%S")


def _auto_filter_sets(keywords):
    """Add platform keywords for --auto and return the excluded substrings."""
    import platform

    system = platform.system().lower()
    keywords.add(system)
    not_include = set()
    if system == "windows":
        keywords.add("msvc-shared")
    elif system == "linux":
        keywords.add("gnu")
        keywords.add("unknown")
        not_include.update({"v2", "v3", "v4"})
    keywords.update({"install_only", "tar.gz", "cpython", "x86_64"})
    not_include.add("stripped")
    return not_include


def _asset_matches(url: str, keywords, not_include) -> bool:
    if any(k in url for k in not_include):
        return False
    return all(k in url for k in keywords)


def _asset_sort_key(asset: Asset):
    # cpython-3.9.19, weights keep 3.13.0 sorting above 3.12.10
    m = re.search(r"-(\d+)\.(\d+)\.(\d+)", asset.name)
    if m:
        return int(m[1]) * 10**6 + int(m[2]) * 10**3 + int(m[3])
    return asset.size


def choose_asset(assets: list, keywords, auto=False):
    keywords = {k for k in keywords if k}
    not_include = set()
    if auto:
        not_include = _auto_filter_sets(keywords)

    result = [a for a in assets if _asset_matches(a.url, keywords, not_include)]
    if not result:
        raise ValueError("No asset found.")
    if len(result) > 1:
        result.sort(key=_asset_sort_key, reverse=True)
        for i in result:
            print(i.url)
        print(
            f"[{get_time()}] Got {len(result)} urls from github, will choose the first one and start download in 3 seconds. {result[0].name}"
        )
        for _ in range(3):
            print(3 - _, flush=True)
            time.sleep(1)
    return result[0]


def verify_digest(digest, target_path: Path):
    """Verify a downloaded file against the release asset's `digest` field.

    GitHub exposes the checksum as `sha256:<hex>` on the asset itself; there
    is no ".sha256" sidecar asset to download. An empty digest means the
    release did not provide one, and verification is skipped.

    Raises ValueError on an unknown digest format or a checksum mismatch.
    """
    if not digest:
        return
    algorithm, _, expected = digest.partition(":")
    if algorithm.lower() != "sha256" or len(expected) != 64:
        raise ValueError(f"unsupported digest: {digest}")
    expected = expected.lower()
    hasher = hashlib.sha256()
    with open(target_path.as_posix(), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected:
        raise ValueError(f"sha256 mismatch for {target_path.name}: expected {expected}, got {actual}")
    print(f"[{get_time()}] sha256 verified: {actual}", flush=True)


def _format_speed(speed) -> str:
    if speed > 1024**2:
        return f"{round(speed / 1024**2, 1)} MB/s"
    if speed > 1024:
        return f"{round(speed / 1024, 1)} KB/s"
    return f"{round(speed, 1)} B/s"


def _format_timeleft(totalsize, done, speed) -> str:
    secs = (totalsize - done) / speed
    if secs > 60:
        return f"{int(secs / 60)}:{int(secs % 60):02}"
    return f"{int(secs)}s"


def _calc_timeleft_speed(records, now, done, totalsize):
    """Return ``(timeleft, speed)`` from the recent ``(timestamp, bytes)`` records."""
    timeleft = "-"
    speed_bps = 0
    if len(records) >= 2:
        # at least one record exists, so seed with the newest one
        record = records[0]
        for record in records:
            if now - record[0] > 1:
                break
        time_diff = now - record[0]
        if time_diff:
            speed_bps = round((done - record[1]) / time_diff, 1)
            if speed_bps:
                timeleft = _format_timeleft(totalsize, done, speed_bps)
    return timeleft, _format_speed(speed_bps)


def download_python():  # noqa: C901  # CLI flow (interactive filters + best-effort download/unzip), splitting it would scatter argparse state
    """Download python portable interpreter from https://github.com/indygreg/python-build-standalone/releases.

    The downloaded file is verified against the release asset's `digest` field
    (``sha256:<hex>``) before it is renamed into place; releases without a
    digest are downloaded unverified.

    Usage:
        python -m morebuiltins.download_python -i                    # interactive mode
        python -m morebuiltins.download_python -a              # auto choose the best match for the current platform
        python -m morebuiltins.download_python --auto -k 3.11 -u     # auto mode with keyword, unzip after download
        python -m morebuiltins.download_python -k 3.11,gnu -t ./tmp  # filter by keywords, save under the target dir
        python -m morebuiltins.download_python -a -k 3.11 -u -f    # unzip the tar.gz, overwrite an existing target dir

    Always combine -u/-f with -a or -k: using "-u -f" alone downloads the
    first asset of any platform.

    Options: -a/--auto, -k/--keywords (comma split), -i/--interactive,
    -t/--target (save dir, defaults to current dir), -u/--unzip, -f/--force.

    λ python -m morebuiltins.download_python -i
    [10:56:17] Checking https://api.github.com/repos/indygreg/python-build-standalone/releases/latest
    [10:56:19] View the rules:
    https://gregoryszorc.com/docs/python-build-standalone/main/running.html#obtaining-distributions

    [10:56:19] Got 290 urls from github.

    [290] Enter keywords (can be int index or partial match, defaults to 0):
    0. windows
    1. linux
    2. darwin
    0
    [10:56:24] Filt with keyword: "windows". 290 => 40

    [40] Enter keywords (can be int index or partial match, defaults to 0):
    0. 3.12.3
    1. 3.11.9
    2. 3.10.14
    3. 3.9.19
    4. 3.8.19

    [10:56:25] Filt with keyword: "3.12.3". 40 => 8

    [8] Enter keywords (can be int index or partial match, defaults to 0):
    0. x86_64
    1. i686

    [10:56:28] Filt with keyword: "x86_64". 8 => 4

    [4] Enter keywords (can be int index or partial match, defaults to 0):
    0. shared-pgo-full.tar.zst
    1. shared-install_only.tar.gz
    2. pgo-full.tar.zst
    3. install_only.tar.gz
    3
    [10:56:33] Filt with keyword: "install_only.tar.gz". 4 => 1
    [10:56:33] Download URL: 39.1 MB
    https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-pc-windows-msvc-install_only.tar.gz
    [10:56:38] Start downloading...
    https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-pc-windows-msvc-install_only.tar.gz
    <target_dir>/cpython-3.12.3+20240415-x86_64-pc-windows-msvc-install_only.tar.gz
    [10:56:44] Downloading: 39.12 / 39.12 MB | 100.00% | 11.3 MB/s | 0s
    [10:56:44] Download complete."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto",
        "-a",
        help="auto choose the best choice(like platform), such as: -k 3.11 --auto",
        action="store_true",
    )
    parser.add_argument("--target", "-t", help="target dir path", default=".")
    parser.add_argument(
        "--keywords",
        "-k",
        help="keywords to filt, split by , for many keywords.",
        default="",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="interactive mode, will ask for filt keywords.",
    )
    parser.add_argument(
        "-u",
        "--unzip",
        action="store_true",
        help="unzip the tar.gz file to target/short_name/. only for .tar.gz",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="with --unzip: overwrite the existing target dir (otherwise exit with error)",
    )
    args, _ = parser.parse_known_args()
    print(
        f"[{get_time()}] Checking https://api.github.com/repos/indygreg/python-build-standalone/releases/latest",
        flush=True,
    )
    assets = get_assets()
    print(
        f"[{get_time()}] View the rules:\nhttps://gregoryszorc.com/docs/python-build-standalone/main/running.html#obtaining-distributions\n",
        flush=True,
    )
    print(f"[{get_time()}] Got {len(assets)} urls from github.")
    keywords = args.keywords.split(",")
    if args.interactive:
        if args.keywords:
            old = len(assets)
            assets = [a for a in assets if all(k in a.name for k in keywords)]
            print(
                f'[{get_time()}] Filt with keywords: "{args.keywords}".',
                old,
                "=>",
                len(assets),
                flush=True,
            )
            if not assets:
                # an empty list would make to_filt[0] raise IndexError later
                print(
                    f"[{get_time()}] No asset matched keywords: {args.keywords}",
                    flush=True,
                )
                raise SystemExit(1)

        def sort_key(s):
            try:
                return tuple(map(int, s.split(".")))
            except ValueError:
                return s

        indexs = [4, 1, 5, 2, 3, 0, 6]
        for index in indexs:
            try:
                to_filt = sorted({i.keywords[index] for i in assets}, key=sort_key, reverse=True)
            except IndexError:
                continue
            if not to_filt:
                continue
            if len(to_filt) == 1:
                continue
            choices = "\n".join((f"{idx}. {ii}" for idx, ii in enumerate(to_filt, 0)))
            arg = input(
                f"\n[{len(assets)}] Enter keywords (can be int index or partial match, defaults to 0):\n{choices}\n"
            )
            if not arg:
                arg = to_filt[0]
            elif arg.isdigit():
                arg = to_filt[int(arg)]
            old = len(assets)
            temp = [i for i in assets if arg == i.keywords[index]]
            if temp:
                assets = temp
            else:
                assets = [i for i in assets if arg in i.keywords[index]]
            print(
                f'[{get_time()}] Filt with keyword: "{arg}".',
                old,
                "=>",
                len(assets),
                flush=True,
            )
        while len(assets) > 1:
            names = "\n".join(i.name for i in assets)
            arg = input(f"Enter keyword to reduce the list (partial match):\n{names}\n")
            assets = [i for i in assets if arg in i.name]
        if not assets:
            input("No match, press enter to exit.")
            return
        asset = assets[0]
    else:
        asset = choose_asset(assets, keywords, args.auto)
    download_url = asset.url
    total_size = asset.size
    print(
        f"[{get_time()}] Download URL:",
        round(total_size / 1024**2, 1),
        "MB",
    )
    print(download_url, flush=True)
    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    target_path = target / asset.name
    # pid suffix avoids concurrent runs clobbering the same temp file
    temp_path = target_path.with_name(f"python_downloading.tmp{os.getpid()}")
    for _path in [target_path, temp_path]:
        if _path.is_file():
            _path.unlink()
    print(f"[{get_time()}] Start downloading...")
    print(download_url)
    print(target_path.resolve(), flush=True)
    records = deque(maxlen=1000)
    last_print = time.time()

    def reporthook(blocknum, blocksize, totalsize):
        nonlocal last_print
        if totalsize < 0:
            totalsize = total_size
        _done = blocknum * blocksize
        if not _done:
            return
        percent = 100.0 * _done / totalsize if totalsize else 0.0
        total = totalsize / 1024 / 1024
        done = _done / 1024 / 1024 or total
        if percent > 100:
            percent = 100
        now = time.time()
        record = (now, _done)
        records.appendleft(record)
        timeleft, speed = _calc_timeleft_speed(records, now, _done, totalsize)
        if time.time() - last_print > 1:
            print(
                f"[{get_time()}] Downloading: {done:.2f} / {total:.2f} MB | {percent:.2f}% | {speed} | {timeleft} {' ' * 10}",
                end="\r",
                flush=True,
            )
            last_print = time.time()

    downloaded = False
    for _ in range(3):
        try:
            last_print = time.time()
            urllib.request.urlretrieve(  # noqa: S310  # nosec B310  # download_url comes from the GitHub release API response
                download_url, temp_path.absolute().as_posix(), reporthook=reporthook
            )
            # verify the temp file first: a corrupt download never replaces an
            # existing good file and the partial temp is removed below
            verify_digest(asset.digest, temp_path)
            temp_path.rename(target_path)
            downloaded = True
            print(
                f"\n[{get_time()}] Download complete.",
                flush=True,
            )
            break
        except http.client.RemoteDisconnected:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            continue
        except KeyboardInterrupt:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            print()
            print(f"\n[{get_time()}] Download canceled.", flush=True)
            return
        except Exception:
            print()
            traceback.print_exc()
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            # do not exit 0 after a failed download
            raise SystemExit(1) from None
    if not downloaded:
        # retries exhausted without success
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise SystemExit(1)
    if args.interactive:
        print("Press enter to exit.", flush=True)
        return input()
    if args.unzip and target_path.name.endswith(".tar.gz"):
        if not target_path.is_file():
            # never silently skip unzip when the target file is missing
            print(
                get_time(),
                f"target file not found, can not unzip: {target_path.as_posix()}",
                flush=True,
            )
            raise SystemExit(1)
        target_dir = target_path.with_name(target_path.stem.split("+", 1)[0])
        if target_dir.is_dir():
            if not args.force:
                # rmtree is irreversible, never delete an existing dir silently
                print(
                    get_time(),
                    f"target dir already exists, use --force to overwrite: {target_dir.as_posix()}",
                    flush=True,
                )
                raise SystemExit(1)
            shutil.rmtree(target_dir.as_posix())
        print(get_time(), f"start unzip file to {target_dir.as_posix()}", flush=True)
        with tarfile.open(target_path.as_posix(), "r:gz") as tar:
            try:
                # filter="data" defends against path traversal members
                tar.extractall(path=target_dir.as_posix(), filter="data")
            except TypeError:
                # legacy fallback: Python < 3.10.12 has no filter argument
                tar.extractall(path=target_dir.as_posix())  # noqa: S202  # nosec B202
    print(get_time(), "bye~")


if __name__ == "__main__":
    download_python()

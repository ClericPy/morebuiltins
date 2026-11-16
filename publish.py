"""Publish morebuiltins to PyPI.

!!! DANGER: running this file uploads the current working tree to PyPI, and a
!!! PyPI release cannot be deleted safely (only yanked). Never run it as a
!!! shortcut to check the CLI, never pass it a stray flag to "see if flit is
!!! installed", never run it while other tools/agents are still editing files.

Publishing is irreversible in practice, so this script refuses to run unless all
of the following hold:

1. `--yes` is passed explicitly (any other invocation only prints usage).
2. The git working tree is clean: an uncommitted or half-edited tree would be
   published as-is, producing an inconsistent release.
3. CHANGELOG.md has a section for the current `morebuiltins.__version__`.
4. The version is not already on PyPI.

Usage:
    py -3 publish.py --yes            # really publish
    py -3 publish.py --yes --dry-run  # run every check, upload nothing

Exit codes:
    0  published (or every check passed with --dry-run)
    1  a guard or a subprocess check failed
    2  usage error: no `--yes`, unknown flags, or a failing `--help` probe

A previous incident: `py -3 publish.py --some-flag` was used to probe whether
flit was installed. The flag was ignored and a real release was uploaded from a
dirty tree. That is why the guard below exists.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = "morebuiltins"


def _fail(msg: str) -> None:
    print(f"[publish] ABORT: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def get_version() -> str:
    init_path = ROOT / PACKAGE / "__init__.py"
    for line in init_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    _fail(f"no __version__ found in {init_path}")
    raise AssertionError("unreachable: _fail always raises")


def check_clean_tree() -> None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607  # git is a required system CLI for the clean-tree guard
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        _fail("git executable not found; a clean tree cannot be verified")
        raise AssertionError("unreachable: _fail always raises") from None
    if result.returncode != 0:
        _fail(f"git status failed: {result.stderr.strip()}")
    dirty = result.stdout.strip()
    if dirty:
        preview = "\n".join(dirty.splitlines()[:10])
        _fail(f"git working tree is not clean, commit the release first:\n{preview}")


def check_changelog(version: str) -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"### {version}" not in changelog:
        _fail(f"CHANGELOG.md has no '### {version}' section")


def check_not_published(version: str) -> None:
    url = f"https://pypi.org/pypi/{PACKAGE}/json"
    data: dict = {}
    try:
        with urllib.request.urlopen(  # noqa: S310  # nosec B310  # constant https PyPI JSON endpoint
            url, timeout=10
        ) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return  # package not on PyPI yet
        _fail(f"cannot query PyPI: {exc}")
    except urllib.error.URLError as exc:
        _fail(f"cannot reach PyPI: {exc.reason}")

    published = data.get("releases", {}).get(version)
    if published:
        _fail(
            f"{PACKAGE} {version} is already on PyPI "
            f"({', '.join(f['filename'] for f in published)}); "
            "bump __version__ instead of re-uploading"
        )


def main(argv: list[str]) -> None:
    if "--help" in argv or "-h" in argv or "--yes" not in argv:
        print(__doc__)
        print("[publish] nothing uploaded (pass --yes to publish)", file=sys.stderr)
        raise SystemExit(2)

    unknown = [a for a in argv if a not in ("--yes", "--dry-run")]
    if unknown:
        _fail(f"unknown arguments {unknown}; this script only publishes")

    version = get_version()
    print(f"[publish] preparing {PACKAGE} {version}", flush=True)
    check_clean_tree()
    check_changelog(version)
    check_not_published(version)

    if "--dry-run" in argv:
        print(f"[publish] dry-run: all checks passed, {version} would be uploaded")
        return

    print(f"[publish] uploading {PACKAGE} {version} to PyPI", flush=True)
    result = subprocess.run([sys.executable, "-m", "flit", "publish"], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"[publish] done: {PACKAGE} {version}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

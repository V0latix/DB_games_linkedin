#!/usr/bin/env python3
"""Environment checks for the Zip extraction pipeline."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Iterable

REQUIRED_BINARIES = ("yt-dlp", "ffmpeg", "ffprobe")


def ok_mark(is_ok: bool) -> str:
    return "OK" if is_ok else "MISSING"


def check_python() -> tuple[bool, str]:
    is_ok = sys.version_info >= (3, 11)
    return is_ok, f"Python {platform.python_version()} (required: >=3.11)"


def check_pip() -> tuple[bool, str]:
    try:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            text=True,
            capture_output=True,
            check=True,
        )
        return True, output.stdout.strip()
    except subprocess.CalledProcessError:
        return False, "pip is not available from this interpreter"


def check_binaries(commands: Iterable[str]) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    for cmd in commands:
        path = shutil.which(cmd)
        results.append((cmd, path is not None, path or "not found in PATH"))
    return results


def check_imports() -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    try:
        import numpy as _np  # noqa: F401

        checks.append(("numpy", True, "import ok"))
    except Exception as exc:  # pragma: no cover - defensive
        checks.append(("numpy", False, f"import failed: {exc}"))

    try:
        import cv2 as _cv2  # noqa: F401

        checks.append(("opencv-python (cv2)", True, "import ok"))
    except Exception as exc:  # pragma: no cover - defensive
        checks.append(("opencv-python (cv2)", False, f"import failed: {exc}"))

    return checks


def main() -> int:
    print("== Zip pipeline setup check ==")

    py_ok, py_msg = check_python()
    print(f"[python] {ok_mark(py_ok)} - {py_msg}")

    pip_ok, pip_msg = check_pip()
    print(f"[pip] {ok_mark(pip_ok)} - {pip_msg}")

    bin_results = check_binaries(REQUIRED_BINARIES)
    for cmd, is_ok, detail in bin_results:
        print(f"[{cmd}] {ok_mark(is_ok)} - {detail}")

    import_results = check_imports()
    for name, is_ok, detail in import_results:
        print(f"[{name}] {ok_mark(is_ok)} - {detail}")

    all_ok = py_ok and pip_ok and all(item[1] for item in bin_results) and all(
        item[1] for item in import_results
    )

    if all_ok:
        print("\nEnvironment looks good. You can run the pipeline.")
        return 0

    print("\nSome requirements are missing.")
    print("Install guide (macOS):")
    print("  brew install yt-dlp ffmpeg")
    print("  python3.11 -m pip install -r requirements.txt")
    print("Install guide (Ubuntu/Debian):")
    print("  sudo apt update && sudo apt install -y ffmpeg yt-dlp python3-pip")
    print("  python3 -m pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

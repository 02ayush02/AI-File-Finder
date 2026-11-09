"""Open a file (or reveal it in the file manager) with the OS default application. Local use only."""
from __future__ import annotations

import os
import subprocess
import sys


def open_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def reveal_in_folder(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

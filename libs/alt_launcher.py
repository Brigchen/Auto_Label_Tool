# -*- coding: utf-8 -*-
"""Launch Auto Label Tool (ALT) from FishVision Trainer or CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from libs.repo_paths import repo_root


def find_alt_python() -> str:
    override = os.environ.get("AUTO_LABEL_PYTHON", "").strip()
    if override and os.path.isfile(override):
        return override
    candidates = [
        r"C:\ProgramData\Anaconda3\envs\py313\python.exe",
        sys.executable,
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return sys.executable


def launch_alt(
    dataset_root: Optional[str] = None,
    *,
    split: str = "auto",
    new_console: bool = True,
) -> subprocess.Popen:
    """Start ALT.py; optional --datasets opens images/labels automatically."""
    repo = repo_root()
    alt_py = os.path.join(repo, "src", "ALT.py")
    if not os.path.isfile(alt_py):
        raise FileNotFoundError(f"未找到 ALT: {alt_py}")

    cmd = [find_alt_python(), alt_py]
    if dataset_root:
        cmd.extend(["--datasets", os.path.abspath(dataset_root)])
        if split:
            cmd.extend(["--split", split])

    kwargs = {"cwd": repo}
    if sys.platform == "win32" and new_console:
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    return subprocess.Popen(cmd, **kwargs)

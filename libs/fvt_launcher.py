# -*- coding: utf-8 -*-
"""Launch FishVision Train GUI from ALT or CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from libs.alt_launcher import find_alt_python
from libs.repo_paths import repo_root


def launch_train_gui(
    data_yaml: Optional[str] = None,
    *,
    new_console: bool = True,
) -> subprocess.Popen:
    """Start FishVision_Train_GUI.py; optional --data pre-fills the data yaml path."""
    repo = repo_root()
    gui_py = os.path.join(repo, "src", "FishVision_Train_GUI.py")
    if not os.path.isfile(gui_py):
        raise FileNotFoundError(f"未找到训练控制台: {gui_py}")

    cmd = [find_alt_python(), gui_py]
    if data_yaml and os.path.isfile(data_yaml):
        cmd.extend(["--data", os.path.abspath(data_yaml)])

    kwargs = {"cwd": repo}
    if sys.platform == "win32" and new_console:
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    return subprocess.Popen(cmd, **kwargs)

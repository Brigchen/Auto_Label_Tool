# -*- coding: utf-8 -*-
"""Windows: register <repo>/libs (and subdirs that contain openh264*.dll) for DLL load before OpenCV."""

from __future__ import annotations

import os
import sys


def register_repo_libs_dll_path():
    """Call before ``import cv2`` so optional native DLLs under ``libs/`` are discoverable."""
    if sys.platform != "win32":
        return
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(lib_dir):
        return

    dirs_to_add = [lib_dir]
    try:
        for root, _, files in os.walk(lib_dir):
            for name in files:
                low = name.lower()
                if low.startswith("openh264") and low.endswith(".dll"):
                    if root not in dirs_to_add:
                        dirs_to_add.append(root)
                    break
    except OSError:
        pass

    for d in dirs_to_add:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass

    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    for d in reversed(dirs_to_add):
        if d and d not in parts:
            parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(parts)

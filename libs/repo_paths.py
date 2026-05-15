# -*- coding: utf-8 -*-
"""Project root (repository root = parent of libs/). Works when apps run from src/."""
import os

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_LIB_DIR, ".."))


def repo_root():
    return _REPO_ROOT


def ensure_repo_on_sys_path():
    import sys

    r = repo_root()
    if r not in sys.path:
        sys.path.insert(0, r)
    return r


def configs_dir():
    """Application ini / pickle configs live under <repo>/configs/."""
    import os

    p = os.path.join(repo_root(), "configs")
    os.makedirs(p, exist_ok=True)
    return p

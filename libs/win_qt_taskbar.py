# -*- coding: utf-8 -*-
"""Windows taskbar / title bar: avoid generic python.exe icon for PyQt apps."""
from __future__ import annotations

import os
import sys


def set_windows_app_user_model_id(app_id: str) -> None:
    """Call before QApplication(). Required for correct taskbar icon when running as python.exe."""
    if sys.platform != "win32" or not app_id:
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except Exception:
        pass


def load_brand_qicon(repo_root: str, stem: str = "app"):
    """Load QIcon from resources/icons/{stem}.ico then .png (multi-size .ico is best on Windows)."""
    from PyQt5.QtGui import QIcon

    d = os.path.join(repo_root, "resources", "icons")
    for ext in (".ico", ".png"):
        p = os.path.join(d, stem + ext)
        if os.path.isfile(p):
            ic = QIcon(p)
            if not ic.isNull():
                return ic
    return QIcon()

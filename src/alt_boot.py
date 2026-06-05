#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Show splash immediately, then load ALT (avoids frozen window during import)."""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libs.win_dll_search_path import register_repo_libs_dll_path

register_repo_libs_dll_path()


def run(argv=None):
    from PyQt5.QtWidgets import QApplication

    from libs.constants import APP_VERSION
    from libs.startup_ui import StartupSplash
    from libs.win_qt_taskbar import set_windows_app_user_model_id

    if argv is None:
        argv = sys.argv

    set_windows_app_user_model_id('Brigchen.AutoLabelTool.ALT.1')
    app = QApplication(argv)
    app.setApplicationName('Auto_Label_Tool')

    splash = StartupSplash(
        'Auto Label Tool',
        version=APP_VERSION,
        repo_root=_REPO_ROOT,
        initial_message='正在启动...',
    )
    splash.show()
    splash.set_message('正在加载程序模块...')
    splash.process_events()

    # Import after splash is visible (torch/ultralytics are lazy-loaded inside ALT).
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    import ALT as alt_module  # noqa: E402

    splash.set_message('正在构建主窗口...')
    splash.process_events()
    _app, win = alt_module.get_main_app(argv, splash=splash)
    splash.finish(win)
    return _app.exec_()


if __name__ == '__main__':
    sys.exit(run())

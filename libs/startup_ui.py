# -*- coding: utf-8 -*-
"""Splash screen and startup progress feedback for ALT."""
from __future__ import annotations

import os

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
    from PyQt5.QtWidgets import QApplication, QSplashScreen
except ImportError:
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui import QColor, QFont, QIcon, QPainter, QPixmap, QSplashScreen
    from PyQt4.QtGui import QApplication


def _make_splash_pixmap(title: str, version: str, repo_root: str) -> QPixmap:
    w, h = 480, 260
    pm = QPixmap(w, h)
    pm.fill(QColor(32, 44, 58))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QColor(220, 230, 240))
    title_font = QFont()
    title_font.setPointSize(16)
    title_font.setBold(True)
    p.setFont(title_font)
    p.drawText(28, 56, title)
    sub_font = QFont()
    sub_font.setPointSize(10)
    p.setFont(sub_font)
    p.setPen(QColor(160, 175, 190))
    if version:
        p.drawText(28, 82, 'v%s' % version)
    p.setPen(QColor(120, 180, 220))
    p.drawRect(28, h - 58, w - 56, 3)
    try:
        from libs.win_qt_taskbar import load_brand_qicon
        ic = load_brand_qicon(repo_root, 'app')
        if not ic.isNull():
            p.drawPixmap(w - 108, 24, ic.pixmap(72, 72))
    except Exception:
        pass
    p.end()
    return pm


class StartupSplash:
    """Non-blocking splash; call set_message + process_events during heavy init."""

    def __init__(self, title: str, version: str = '', repo_root: str = '', initial_message: str = ''):
        self._repo_root = repo_root or os.getcwd()
        self._pix = _make_splash_pixmap(title, version, self._repo_root)
        self._splash = QSplashScreen(self._pix, Qt.WindowStaysOnTopHint)
        self._message = initial_message or '正在启动...'
        self._splash.showMessage(
            self._message,
            Qt.AlignBottom | Qt.AlignHCenter,
            QColor(200, 220, 235),
        )

    def show(self):
        self._splash.show()
        self.process_events()

    def set_message(self, message: str):
        self._message = message
        self._splash.showMessage(
            message,
            Qt.AlignBottom | Qt.AlignHCenter,
            QColor(200, 220, 235),
        )
        self.process_events()

    def process_events(self):
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def finish(self, window):
        if window is not None:
            self._splash.finish(window)
        else:
            self._splash.close()
        self.process_events()

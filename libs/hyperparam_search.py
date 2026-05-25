# -*- coding: utf-8 -*-
"""
HyperparamSearch — 自动调参面板 (QWidget)。

封装 Ultralytics model.tune() 的搜索空间配置，
提供 UI 选择要搜索的参数及其范围。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# 默认搜索空间 (Ultralytics model.tune() 推荐范围)
_DEFAULT_SEARCH_SPACE = {
    "lr0": (1e-5, 1e-1, True),
    "lrf": (1e-3, 1e-1, True),
    "momentum": (0.8, 0.98, False),
    "weight_decay": (0.0, 0.01, True),
    "warmup_epochs": (0.0, 5.0, False),
    "hsv_h": (0.0, 0.1, False),
    "hsv_s": (0.0, 0.9, False),
    "hsv_v": (0.0, 0.5, False),
    "degrees": (0.0, 45.0, False),
    "translate": (0.0, 0.5, False),
    "scale": (0.0, 1.5, False),
    "shear": (0.0, 10.0, False),
    "perspective": (0.0, 0.001, False),
    "flipud": (0.0, 0.5, False),
    "fliplr": (0.0, 1.0, False),
    "mosaic": (0.0, 1.0, False),
    "mixup": (0.0, 1.0, False),
    "copy_paste": (0.0, 1.0, False),
}


class HyperparamSearchPanel(QWidget):
    """自动调参配置面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict = {}
        self._checkboxes: dict = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── 启用开关 ──
        self.enable_cb = QCheckBox("启用自动调参 (Auto Tune)")
        self.enable_cb.setChecked(False)
        layout.addWidget(self.enable_cb)

        # ── 基本参数 ──
        basic_group = QGroupBox("调参基本设置")
        basic_form = QFormLayout()

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(100)
        self.iterations_spin.setToolTip("搜索迭代次数（每次迭代训练若干轮）")
        basic_form.addRow("搜索迭代次数:", self.iterations_spin)

        self.tune_epochs_spin = QSpinBox()
        self.tune_epochs_spin.setRange(1, 500)
        self.tune_epochs_spin.setValue(50)
        self.tune_epochs_spin.setToolTip("每次迭代的训练轮数（建议 30-100）")
        basic_form.addRow("每次迭代 Epochs:", self.tune_epochs_spin)

        basic_group.setLayout(basic_form)
        layout.addWidget(basic_group)

        # ── 搜索空间 ──
        space_group = QGroupBox("搜索空间（勾选 = 加入搜索，留空保持默认值）")
        space_form = QFormLayout()

        for key, (lo, hi, log_scale) in _DEFAULT_SEARCH_SPACE.items():
            row = self._make_param_row(key, lo, hi, log_scale)
            space_form.addRow(f"{key}:", row)

        space_group.setLayout(space_form)
        layout.addWidget(space_group)

        # ── 说明 ──
        tip = QLabel(
            "说明：勾选参数后将在指定范围内搜索最优值。"
            "未勾选的参数将使用当前选项卡中的设定值。"
            "搜索方式为 Ultralytics 内置随机搜索 + 遗传算法。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #666;")
        layout.addWidget(tip)

        layout.addStretch()

    def _make_param_row(self, key: str, lo: float, hi: float, log_scale: bool):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        cb = QCheckBox()
        cb.setChecked(False)
        rl.addWidget(cb)
        self._checkboxes[key] = cb

        rl.addWidget(QLabel("min:"))
        lo_spin = QDoubleSpinBox()
        lo_spin.setRange(-1e6, 1e6)
        lo_spin.setValue(lo)
        lo_spin.setFixedWidth(80)
        if log_scale:
            lo_spin.setDecimals(6)
        rl.addWidget(lo_spin)

        rl.addWidget(QLabel("max:"))
        hi_spin = QDoubleSpinBox()
        hi_spin.setRange(-1e6, 1e6)
        hi_spin.setValue(hi)
        hi_spin.setFixedWidth(80)
        if log_scale:
            hi_spin.setDecimals(6)
        rl.addWidget(hi_spin)

        if log_scale:
            lbl = QLabel("(log)")
            lbl.setStyleSheet("color: #888; font-size: 10px;")
            rl.addWidget(lbl)

        self._widgets[key] = (lo_spin, hi_spin, log_scale)
        return row

    def is_enabled(self) -> bool:
        return self.enable_cb.isChecked()

    def get_search_space(self) -> dict:
        """返回 model.tune() 的 space 参数。"""
        space = {}
        for key, (lo_spin, hi_spin, log_scale) in self._widgets.items():
            if self._checkboxes.get(key, False).isChecked():
                lo = lo_spin.value()
                hi = hi_spin.value()
                if log_scale:
                    space[key] = (lo, hi, "log")
                else:
                    space[key] = (lo, hi)
        return space

    def get_tune_params(self) -> dict:
        """返回调参配置（iterations, epochs 等）。"""
        if not self.is_enabled():
            return {}
        return {
            "tune_enabled": True,
            "iterations": self.iterations_spin.value(),
            "tune_epochs": self.tune_epochs_spin.value(),
            "space": self.get_search_space(),
        }

    def set_enabled(self, enabled: bool):
        self.enable_cb.setChecked(enabled)

    def get_save_state(self) -> dict:
        """Return tune tab state for session persistence."""
        return {
            "tune_enabled": self.enable_cb.isChecked(),
            "iterations": self.iterations_spin.value(),
            "tune_epochs": self.tune_epochs_spin.value(),
        }

    def set_save_state(self, d: dict) -> None:
        if "tune_enabled" in d:
            self.enable_cb.setChecked(
                str(d["tune_enabled"]).lower() in ("1", "true", "yes")
            )
        if d.get("iterations") is not None:
            self.iterations_spin.setValue(int(d["iterations"]))
        if d.get("tune_epochs") is not None:
            self.tune_epochs_spin.setValue(int(d["tune_epochs"]))

# -*- coding: utf-8 -*-
"""
AugmentPanel — 数据增强参数面板 (QWidget)。

提供 Ultralytics 支持的全部数据增强参数的滑块/数值输入控件，
一键恢复 Ultralytics 默认值。
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
    QVBoxLayout,
    QWidget,
)

# Ultralytics 默认增强参数 (YOLO detect/segment/pose 通用)
_ULTRALYTICS_AUG_DEFAULTS = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "auto_augment": "",
}


class AugmentPanel(QWidget):
    """数据增强参数选项卡面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── 颜色抖动 ──
        gb_color = QGroupBox("颜色抖动 (Color Jitter)")
        fl_color = QFormLayout()
        self._add_spin(gb_color, fl_color, "hsv_h", "HSV-Hue", 0.0, 1.0, 0.005, 4)
        self._add_spin(gb_color, fl_color, "hsv_s", "HSV-Saturation", 0.0, 1.0, 0.01, 4)
        self._add_spin(gb_color, fl_color, "hsv_v", "HSV-Value", 0.0, 1.0, 0.01, 4)
        gb_color.setLayout(fl_color)
        layout.addWidget(gb_color)

        # ── 几何变换 ──
        gb_geo = QGroupBox("几何变换 (Geometric)")
        fl_geo = QFormLayout()
        self._add_spin(gb_geo, fl_geo, "degrees", "Degrees (旋转)", 0.0, 180.0, 1.0, 1)
        self._add_spin(gb_geo, fl_geo, "translate", "Translate (平移)", 0.0, 1.0, 0.01, 2)
        self._add_spin(gb_geo, fl_geo, "scale", "Scale (缩放)", 0.0, 10.0, 0.1, 2)
        self._add_spin(gb_geo, fl_geo, "shear", "Shear (剪切)", 0.0, 180.0, 1.0, 1)
        self._add_spin(gb_geo, fl_geo, "perspective", "Perspective (透视)", 0.0, 1.0, 0.01, 3)
        gb_geo.setLayout(fl_geo)
        layout.addWidget(gb_geo)

        # ── 翻转 ──
        gb_flip = QGroupBox("翻转 (Flip)")
        fl_flip = QFormLayout()
        self._add_spin(gb_flip, fl_flip, "flipud", "Flip Up-Down", 0.0, 1.0, 0.05, 2)
        self._add_spin(gb_flip, fl_flip, "fliplr", "Flip Left-Right", 0.0, 1.0, 0.05, 2)
        gb_flip.setLayout(fl_flip)
        layout.addWidget(gb_flip)

        # ── 混合增强 ──
        gb_mix = QGroupBox("混合增强 (Mosaic / MixUp / Copy-Paste)")
        fl_mix = QFormLayout()
        self._add_spin(gb_mix, fl_mix, "mosaic", "Mosaic", 0.0, 1.0, 0.05, 2)
        self._add_spin(gb_mix, fl_mix, "mixup", "MixUp", 0.0, 1.0, 0.05, 2)
        self._add_spin(gb_mix, fl_mix, "copy_paste", "Copy-Paste", 0.0, 1.0, 0.05, 2)
        self._add_spin(gb_mix, fl_mix, "erasing", "Erasing (随机擦除)", 0.0, 1.0, 0.05, 2)
        gb_mix.setLayout(fl_mix)
        layout.addWidget(gb_mix)

        # ── 恢复默认 ──
        reset_btn = QPushButton("全部恢复 Ultralytics 默认值")
        reset_btn.clicked.connect(self._reset_all)
        layout.addWidget(reset_btn, alignment=Qt.AlignLeft)

        layout.addStretch()

    def _add_spin(self, groupbox, form, key: str, label: str,
                  min_val: float, max_val: float, step: float, decimals: int):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(_ULTRALYTICS_AUG_DEFAULTS.get(key, 0.0))
        spin.setFixedWidth(120)

        reset_btn = QPushButton("R")
        reset_btn.setFixedWidth(30)
        reset_btn.setToolTip("恢复默认值")
        default_val = _ULTRALYTICS_AUG_DEFAULTS.get(key, 0.0)
        reset_btn.clicked.connect(lambda: spin.setValue(default_val))

        row_layout.addWidget(spin)
        row_layout.addWidget(reset_btn)
        row_layout.addStretch()

        form.addRow(QLabel(f"{label}:"), row)
        self._widgets[key] = spin

    def _reset_all(self):
        for key, default in _ULTRALYTICS_AUG_DEFAULTS.items():
            w = self._widgets.get(key)
            if w is not None:
                w.setValue(default)

    def get_values(self) -> dict:
        """返回当前增强参数字典，供 model.train() 使用。"""
        out = {}
        for key, widget in self._widgets.items():
            val = widget.value()
            if abs(val) < 1e-9:
                val = 0.0
            out[key] = val
        return out

    def set_values(self, values: dict):
        """从 dict 恢复控件值。"""
        for key, val in values.items():
            w = self._widgets.get(key)
            if w is not None:
                try:
                    w.setValue(float(val))
                except (ValueError, TypeError):
                    pass

# -*- coding: utf-8 -*-
"""
AugmentPanel — 数据增强参数面板 (QWidget)。

提供 Ultralytics 支持的全部数据增强参数的滑块/数值输入控件，
一键恢复 Ultralytics 默认值；水下场景预设 + 在线水下光度增强。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from libs.underwater_augment import UNDERWATER_YOLO_PRESET

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

_UW_ONLINE_DEFAULTS = {
    "uw_augment": False,
    "uw_augment_p": 0.5,
    "uw_augment_strength": "medium",
    "uw_include_enhance": True,
}


class AugmentPanel(QWidget):
    """数据增强参数选项卡面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── 水下场景 ──
        gb_uw = QGroupBox("水下场景增强 (Underwater)")
        fl_uw = QFormLayout()
        self.uw_augment_cb = QCheckBox("训练时在线水下光度增强 (RandomUnderwater)")
        self.uw_augment_cb.setChecked(_UW_ONLINE_DEFAULTS["uw_augment"])
        self.uw_p_spin = QDoubleSpinBox()
        self.uw_p_spin.setRange(0.0, 1.0)
        self.uw_p_spin.setDecimals(2)
        self.uw_p_spin.setSingleStep(0.05)
        self.uw_p_spin.setValue(_UW_ONLINE_DEFAULTS["uw_augment_p"])
        self.uw_strength_combo = QComboBox()
        self.uw_strength_combo.addItems(["light", "medium", "strong"])
        self.uw_strength_combo.setCurrentText(_UW_ONLINE_DEFAULTS["uw_augment_strength"])
        self.uw_enhance_cb = QCheckBox("在线增强含 CLAHE 恢复分支")
        self.uw_enhance_cb.setChecked(_UW_ONLINE_DEFAULTS["uw_include_enhance"])
        preset_btn = QPushButton("应用水下 YOLO 预设 (HSV/Mosaic/…)")
        preset_btn.setToolTip(
            "设置适合水下鱼类检测的 Ultralytics 颜色/几何参数，并默认启用水下在线增强"
        )
        preset_btn.clicked.connect(self.apply_underwater_preset)
        fl_uw.addRow("", self.uw_augment_cb)
        fl_uw.addRow("在线概率:", self.uw_p_spin)
        fl_uw.addRow("强度:", self.uw_strength_combo)
        fl_uw.addRow("", self.uw_enhance_cb)
        fl_uw.addRow("", preset_btn)
        uw_hint = QLabel(
            "离线扩增：ALT → Make Datasets 勾选 Underwater offline augment。\n"
            "在线增强：每个 batch 随机施加衰减/浊度/色偏/CLAHE 等（不改变 bbox）。"
        )
        uw_hint.setWordWrap(True)
        uw_hint.setStyleSheet("color: gray; font-size: 11px;")
        fl_uw.addRow("", uw_hint)
        gb_uw.setLayout(fl_uw)
        layout.addWidget(gb_uw)

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

    def apply_underwater_preset(self):
        """Apply recommended underwater YOLO hyperparams + enable online UW augment."""
        for key, val in UNDERWATER_YOLO_PRESET.items():
            w = self._widgets.get(key)
            if w is not None:
                w.setValue(float(val))
        self.uw_augment_cb.setChecked(True)
        self.uw_p_spin.setValue(0.5)
        self.uw_strength_combo.setCurrentText("medium")
        self.uw_enhance_cb.setChecked(True)

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
        self.uw_augment_cb.setChecked(_UW_ONLINE_DEFAULTS["uw_augment"])
        self.uw_p_spin.setValue(_UW_ONLINE_DEFAULTS["uw_augment_p"])
        self.uw_strength_combo.setCurrentText(_UW_ONLINE_DEFAULTS["uw_augment_strength"])
        self.uw_enhance_cb.setChecked(_UW_ONLINE_DEFAULTS["uw_include_enhance"])

    def get_values(self) -> dict:
        """返回当前增强参数字典，供 model.train() 使用。"""
        out = {}
        for key, widget in self._widgets.items():
            val = widget.value()
            if abs(val) < 1e-9:
                val = 0.0
            out[key] = val
        out["uw_augment"] = self.uw_augment_cb.isChecked()
        out["uw_augment_p"] = self.uw_p_spin.value()
        out["uw_augment_strength"] = self.uw_strength_combo.currentText()
        out["uw_include_enhance"] = self.uw_enhance_cb.isChecked()
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
        if "uw_augment" in values:
            self.uw_augment_cb.setChecked(str(values["uw_augment"]).lower() in ("1", "true", "yes"))
        if values.get("uw_augment_p") is not None:
            try:
                self.uw_p_spin.setValue(float(values["uw_augment_p"]))
            except (ValueError, TypeError):
                pass
        if values.get("uw_augment_strength"):
            idx = self.uw_strength_combo.findText(str(values["uw_augment_strength"]))
            if idx >= 0:
                self.uw_strength_combo.setCurrentIndex(idx)
        if "uw_include_enhance" in values:
            self.uw_enhance_cb.setChecked(
                str(values["uw_include_enhance"]).lower() in ("1", "true", "yes"))

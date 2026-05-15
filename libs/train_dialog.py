# -*- coding: utf-8 -*-
"""Simplified Train Model dialog: yaml, weights, auto task, epochs / imgsz / VRAM-suggested batch, run name."""
from __future__ import annotations

import configparser
import os
from typing import Any, Callable, Dict, Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from libs.repo_paths import configs_dir, repo_root
from libs.yolo_weights import resolve_yolo_checkpoint


def _train_config_path() -> str:
    return os.path.join(configs_dir(), "config_FVT.ini")


def infer_task_from_weights(weights_str: str, app_weights_dir: str) -> str:
    """Best-effort task from Ultralytics checkpoint (user may override in UI)."""
    if not (weights_str or "").strip():
        return "detect"
    try:
        from ultralytics import YOLO

        w = resolve_yolo_checkpoint(weights_str.strip(), app_weights_dir)
        model = YOLO(w)
        t = getattr(model, "task", None)
        if t is None and hasattr(model, "model"):
            t = getattr(model.model, "task", None)
        if not t:
            return "detect"
        t = str(t).lower()
        if t in ("det", "detect"):
            return "detect"
        if t in ("segment", "seg"):
            return "segment"
        if t == "pose":
            return "pose"
        if t == "obb":
            return "obb"
        if t in ("classify", "cls"):
            return "classify"
        return "detect"
    except Exception:
        return "detect"


def load_train_dialog_defaults() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    path = _train_config_path()
    if not os.path.isfile(path):
        return out
    cfg = configparser.ConfigParser()
    try:
        cfg.read(path, encoding="utf-8")
    except Exception:
        try:
            cfg.read(path)
        except Exception:
            return out
    if not cfg.has_section("settings"):
        return out

    def _s(key: str, default: str = "") -> str:
        if cfg.has_option("settings", key):
            return cfg.get("settings", key).strip()
        return default

    def _i(key: str, default: int) -> int:
        if cfg.has_option("settings", key):
            try:
                return int(cfg.get("settings", key))
            except Exception:
                return default
        return default

    out["yaml_file"] = _s("yaml_file")
    out["weights_file"] = _s("weights_file")
    out["task"] = _s("task", "detect") or "detect"
    out["imgsz"] = _i("imgsz", 1280)
    out["batch"] = _i("batch", 8)
    out["epochs"] = _i("epochs", 100)
    out["run_name"] = _s("train_name", _s("name", "train"))
    return out


def save_train_dialog_values(values: Dict[str, Any]) -> None:
    path = _train_config_path()
    cfg = configparser.ConfigParser()
    if os.path.isfile(path):
        try:
            cfg.read(path, encoding="utf-8")
        except Exception:
            cfg.read(path)
    if not cfg.has_section("settings"):
        cfg.add_section("settings")

    def _set(key: str, val: Any):
        cfg.set("settings", key, str(val))

    _set("yaml_file", values.get("yaml_file", ""))
    _set("weights_file", values.get("weights_file", ""))
    _set("task", values.get("task", "detect"))
    _set("imgsz", int(values.get("imgsz", 1280)))
    _set("batch", int(values.get("batch", 8)))
    _set("epochs", int(values.get("epochs", 100)))
    _set("name", values.get("run_name", "train"))
    _set("train_name", values.get("run_name", "train"))

    with open(path, "w", encoding="utf-8") as f:
        cfg.write(f)


class TrainModelDialog(QDialog):
    """
    Single compact dialog: data yaml, weights, task (auto-filled from weights, editable),
    epochs / imgsz / batch (VRAM estimate via callback), run_name.
    """

    def __init__(
        self,
        parent=None,
        defaults: Optional[Dict[str, Any]] = None,
        app_weights_dir: str = "",
        estimate_batch_fn: Optional[Callable[[str, int], int]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("训练模型 (Train Model)")
        self.setMinimumWidth(560)
        self._app_weights_dir = app_weights_dir or ""
        self._estimate_batch = estimate_batch_fn

        merged = load_train_dialog_defaults()
        if defaults:
            merged.update({k: v for k, v in defaults.items() if v is not None and v != ""})

        root = QVBoxLayout(self)

        gb_data = QGroupBox("数据集")
        fd = QFormLayout()
        self.yaml_edit = QLineEdit()
        self.yaml_edit.setText(merged.get("yaml_file", ""))
        ybtn = QPushButton("浏览…")
        yrow = QWidget()
        yh = QHBoxLayout(yrow)
        yh.setContentsMargins(0, 0, 0, 0)
        yh.addWidget(self.yaml_edit)
        yh.addWidget(ybtn)
        ybtn.clicked.connect(self._browse_yaml)
        fd.addRow("data yaml:", yrow)
        gb_data.setLayout(fd)
        root.addWidget(gb_data)

        gb_model = QGroupBox("预训练权重与任务")
        fm = QFormLayout()
        self.weights_edit = QLineEdit()
        self.weights_edit.setText(merged.get("weights_file", ""))
        wbtn = QPushButton("浏览…")
        wrow = QWidget()
        wh = QHBoxLayout(wrow)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.weights_edit)
        wh.addWidget(wbtn)
        wbtn.clicked.connect(self._browse_weights)

        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "pose", "segment", "obb", "classify"])
        task = merged.get("task", "detect") or "detect"
        self.task_combo.setCurrentIndex(max(0, self.task_combo.findText(task)))

        fm.addRow("weights (.pt):", wrow)
        fm.addRow("task (随权重自动识别，可改):", self.task_combo)
        gb_model.setLayout(fm)
        root.addWidget(gb_model)

        gb_hyp = QGroupBox("训练参数")
        fh = QFormLayout()
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(int(merged.get("imgsz", 1280)))

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(int(merged.get("batch", 8)))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(int(merged.get("epochs", 100)))

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("run 名称 (Ultralytics name=…)")
        self.name_edit.setText(merged.get("run_name", "train") or "train")

        fh.addRow("imgsz:", self.imgsz_spin)
        fh.addRow("batch (按显存估算，可改):", self.batch_spin)
        fh.addRow("epochs:", self.epochs_spin)
        fh.addRow("run name:", self.name_edit)
        gb_hyp.setLayout(fh)
        root.addWidget(gb_hyp)

        self.batch_hint_label = QLabel("")
        self.batch_hint_label.setWordWrap(True)
        self.batch_hint_label.setStyleSheet("color: #555;")
        root.addWidget(self.batch_hint_label)

        tip = QLabel(
            "说明：更换权重或 imgsz 后会按显存重新估算 batch；optimizer / lr 等与历史版本一致 (SGD, lr0=0.001)。"
        )
        tip.setWordWrap(True)
        root.addWidget(tip)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.weights_edit.editingFinished.connect(self._on_weights_changed)
        self.imgsz_spin.valueChanged.connect(self._on_imgsz_changed)
        self._apply_weights_side_effects()

    def _browse_yaml(self):
        start = os.path.join(repo_root(), "datasets")
        if not os.path.isdir(start):
            start = repo_root()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 data yaml", start, "YAML (*.yaml *.yml);;All files (*.*)")
        if path:
            self.yaml_edit.setText(path)

    def _browse_weights(self):
        start = os.path.join(repo_root(), "weights")
        if not os.path.isdir(start):
            start = repo_root()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择预训练权重", start, "Weights (*.pt *.pth);;All files (*.*)")
        if path:
            self.weights_edit.setText(path)
            self._apply_weights_side_effects()

    def _on_weights_changed(self):
        self._apply_weights_side_effects()

    def _on_imgsz_changed(self, _v: int):
        self._refresh_batch_suggestion()

    def _apply_weights_side_effects(self):
        w = self.weights_edit.text().strip()
        if w and self._app_weights_dir:
            guessed = infer_task_from_weights(w, self._app_weights_dir)
            ix = self.task_combo.findText(guessed)
            if ix >= 0:
                self.task_combo.setCurrentIndex(ix)
        self._refresh_batch_suggestion()

    def _refresh_batch_suggestion(self):
        w = self.weights_edit.text().strip()
        imgsz = int(self.imgsz_spin.value())
        if self._estimate_batch is None or not w:
            self.batch_hint_label.setText("")
            return
        try:
            sug = int(self._estimate_batch(w, imgsz))
            sug = max(1, sug)
            self.batch_spin.setValue(sug)
            self.batch_hint_label.setText("根据当前显存与权重估算的 batch ≈ %d（已填入，可自行修改）" % sug)
        except Exception as e:
            self.batch_hint_label.setText("无法估算 batch：%s" % e)

    def values(self) -> Dict[str, Any]:
        return {
            "yaml_file": self.yaml_edit.text().strip(),
            "weights_file": self.weights_edit.text().strip(),
            "task": self.task_combo.currentText(),
            "imgsz": self.imgsz_spin.value(),
            "batch": self.batch_spin.value(),
            "epochs": self.epochs_spin.value(),
            "run_name": self.name_edit.text().strip() or "train",
        }

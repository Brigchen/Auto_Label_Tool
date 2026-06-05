# -*- coding: utf-8 -*-
"""Annotate-Tools dialogs: Auto Label Batch, Make Datasets."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def _browse_dir(line: QLineEdit, parent: QWidget, title: str):
    start = line.text().strip() if line.text().strip() else os.path.abspath('.')
    d = QFileDialog.getExistingDirectory(parent, title, start)
    if d:
        line.setText(d)


def _browse_file(line: QLineEdit, parent: QWidget, title: str, filt: str):
    start = os.path.dirname(line.text()) if line.text().strip() else os.path.abspath('.')
    path, _ = QFileDialog.getOpenFileName(parent, title, start, filt)
    if path:
        line.setText(path)


class _DirRow(QWidget):
    def __init__(self, placeholder: str, browse_title: str, parent=None):
        super().__init__(parent)
        self._browse_title = browse_title
        lo = QHBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        btn = QPushButton('Browse…')
        btn.clicked.connect(self._pick)
        lo.addWidget(self.edit)
        lo.addWidget(btn)

    def _pick(self):
        _browse_dir(self.edit, self, self._browse_title)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str):
        self.edit.setText(t or '')


class ChooseAutoLabelModelDialog(QDialog):
    """Choose Auto Label weights and detection thresholds in one step."""

    def __init__(
        self,
        parent=None,
        defaults: Optional[Dict[str, Any]] = None,
        weights_dir: str = '',
        prefer_model: str = 'yolo26n.pt',
    ):
        super().__init__(parent)
        self.setWindowTitle('Auto Label Model & Thresholds')
        defaults = defaults or {}
        self._weights_dir = weights_dir or ''
        self._prefer_model = prefer_model or 'yolo26n.pt'
        self._preset_paths = {}

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.preset_combo = QComboBox()
        builtin_label = '[Builtin] %s (recommended)' % self._prefer_model
        self.preset_combo.addItem(builtin_label)
        self._preset_paths[0] = ('builtin', self._prefer_model)
        idx = 1
        if self._weights_dir and os.path.isdir(self._weights_dir):
            for item in sorted(os.listdir(self._weights_dir)):
                if item.lower().endswith(('.pt', '.pth', '.h5')):
                    self.preset_combo.addItem(item)
                    self._preset_paths[idx] = ('local', os.path.join(self._weights_dir, item))
                    idx += 1
        self.preset_combo.addItem('[Browse local file…]')
        self._preset_paths[idx] = ('browse', '')
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self.weights_edit = QLineEdit()
        wbtn = QPushButton('Browse…')
        wrow = QWidget()
        wh = QHBoxLayout(wrow)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.weights_edit)
        wh.addWidget(wbtn)
        wbtn.clicked.connect(self._pick_weights)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(defaults.get('conf', 0.25)))

        self.pred_iou_spin = QDoubleSpinBox()
        self.pred_iou_spin.setRange(0.1, 0.99)
        self.pred_iou_spin.setDecimals(2)
        self.pred_iou_spin.setSingleStep(0.05)
        self.pred_iou_spin.setValue(float(defaults.get('pred_iou', 0.5)))

        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.1, 0.99)
        self.dedup_spin.setDecimals(2)
        self.dedup_spin.setSingleStep(0.05)
        self.dedup_spin.setValue(float(defaults.get('dedup_iou', 0.7)))

        form.addRow('Quick pick:', self.preset_combo)
        form.addRow('Model weights:', wrow)
        form.addRow('Confidence (conf):', self.conf_spin)
        form.addRow('Predict IoU:', self.pred_iou_spin)
        form.addRow('Dedup IoU:', self.dedup_spin)
        root.addLayout(form)

        hint = QLabel(
            'Confidence: minimum detection score. '
            'Predict IoU: NMS overlap. Dedup IoU: drop duplicate same-class boxes after predict.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color: gray; font-size: 11px;')
        root.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        cur = (defaults.get('weights') or '').strip()
        self.weights_edit.setText(cur)
        self.preset_combo.blockSignals(True)
        try:
            self._sync_preset_from_weights(cur)
        finally:
            self.preset_combo.blockSignals(False)

    def _on_preset_changed(self, index: int):
        kind, path = self._preset_paths.get(index, ('', ''))
        if kind == 'builtin':
            self.weights_edit.setText(self._prefer_model)
        elif kind == 'local':
            self.weights_edit.setText(path)
        elif kind == 'browse':
            self._pick_weights()

    def _sync_preset_from_weights(self, path: str):
        path = (path or '').strip()
        if not path:
            return
        base = os.path.basename(path.replace('\\', '/'))
        if base == self._prefer_model or path == self._prefer_model:
            self.preset_combo.setCurrentIndex(0)
            return
        for i, (kind, p) in self._preset_paths.items():
            if kind == 'local' and (
                os.path.normcase(p) == os.path.normcase(path)
                or os.path.basename(p) == base
            ):
                self.preset_combo.setCurrentIndex(i)
                return

    def _pick_weights(self):
        _browse_file(
            self.weights_edit, self, 'Select Model Weights',
            'Model weights (*.pt *.pth *.h5);;All files (*.*)')
        self.preset_combo.blockSignals(True)
        try:
            self._sync_preset_from_weights(self.weights_edit.text())
        finally:
            self.preset_combo.blockSignals(False)

    def _on_accept(self):
        if not self.weights_edit.text().strip():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Auto Label', 'Please select model weights.')
            return
        self.accept()

    def values(self) -> Dict[str, Any]:
        return {
            'weights': self.weights_edit.text().strip(),
            'conf': self.conf_spin.value(),
            'pred_iou': self.pred_iou_spin.value(),
            'dedup_iou': self.dedup_spin.value(),
        }


class AutoLabelBatchDialog(QDialog):
    """Folder batch auto-label to YOLO txt (detect / segment / pose by model)."""

    def __init__(self, parent=None, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle('Auto Label Batch')
        defaults = defaults or {}
        root = QVBoxLayout(self)

        form = QFormLayout()
        self.images_row = _DirRow('Images folder', 'Select Images Folder')
        self.labels_row = _DirRow('Labels output folder', 'Select Labels Output Folder')
        self.weights_edit = QLineEdit()
        wbtn = QPushButton('Browse…')
        wrow = QWidget()
        wh = QHBoxLayout(wrow)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.weights_edit)
        wh.addWidget(wbtn)
        wbtn.clicked.connect(self._pick_weights)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(defaults.get('conf', 0.25)))

        self.pred_iou_spin = QDoubleSpinBox()
        self.pred_iou_spin.setRange(0.1, 0.99)
        self.pred_iou_spin.setDecimals(2)
        self.pred_iou_spin.setValue(float(defaults.get('pred_iou', 0.5)))

        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.1, 0.99)
        self.dedup_spin.setDecimals(2)
        self.dedup_spin.setValue(float(defaults.get('dedup_iou', 0.7)))

        self.task_combo = QComboBox()
        self.task_combo.addItems(['Auto (from model)', 'Detect', 'Segment', 'Pose'])
        self.save_score_check = QCheckBox('Append detection score to each YOLO line (for filtering)')
        self.save_score_check.setChecked(True)

        form.addRow('Images:', self.images_row)
        form.addRow('Labels:', self.labels_row)
        form.addRow('Model weights:', wrow)
        form.addRow('Confidence:', self.conf_spin)
        form.addRow('Predict IoU:', self.pred_iou_spin)
        form.addRow('Dedup IoU:', self.dedup_spin)
        form.addRow('Task:', self.task_combo)
        form.addRow('', self.save_score_check)
        root.addLayout(form)

        self.images_row.setText(defaults.get('image_dir', ''))
        self.labels_row.setText(defaults.get('label_dir', ''))
        self.weights_edit.setText(defaults.get('weights', ''))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _pick_weights(self):
        _browse_file(
            self.weights_edit, self, 'Select Model Weights',
            'Model weights (*.pt *.pth);;All files (*.*)')

    def values(self) -> Dict[str, Any]:
        task_map = {
            0: 'auto', 1: 'detect', 2: 'segment', 3: 'pose',
        }
        return {
            'image_dir': self.images_row.text(),
            'label_dir': self.labels_row.text(),
            'weights': self.weights_edit.text().strip(),
            'conf': self.conf_spin.value(),
            'pred_iou': self.pred_iou_spin.value(),
            'dedup_iou': self.dedup_spin.value(),
            'task_hint': task_map.get(self.task_combo.currentIndex(), 'auto'),
            'save_score': self.save_score_check.isChecked(),
        }


def _browse_files_multi(parent: QWidget, title: str, filt: str) -> List[str]:
    paths, _ = QFileDialog.getOpenFileNames(parent, title, os.path.abspath('.'), filt)
    return list(paths) if paths else []


class ExtractVideosDialog(QDialog):
    """Settings for extracting frames from selected video file(s)."""

    def __init__(self, parent=None, n_videos: int = 1, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle('Extract Videos')
        defaults = defaults or {}
        root = QVBoxLayout(self)
        root.addWidget(QLabel('%d video file(s) selected.' % max(1, n_videos)))

        form = QFormLayout()
        self.output_row = _DirRow('Output folder (images/ created inside)', 'Select Output Folder')
        self.frame_gap_spin = QSpinBox()
        self.frame_gap_spin.setRange(1, 9999)
        self.frame_gap_spin.setValue(int(defaults.get('frame_gap', 1)))
        self.frame_gap_spin.setToolTip('Save every N-th frame (1 = all frames).')

        form.addRow('Output:', self.output_row)
        form.addRow('Frame interval:', self.frame_gap_spin)
        root.addLayout(form)

        self.output_row.setText(defaults.get('output_dir', ''))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> Dict[str, Any]:
        return {
            'output_dir': self.output_row.text(),
            'frame_gap': self.frame_gap_spin.value(),
        }


class AnnotateVideosDialog(QDialog):
    """Extract frames from video(s) and auto-label to images/ + labels/ (YOLO)."""

    def __init__(self, parent=None, n_videos: int = 1, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle('Annotate Videos')
        defaults = defaults or {}
        root = QVBoxLayout(self)

        root.addWidget(QLabel('%d video file(s) selected.' % max(1, n_videos)))

        form = QFormLayout()
        self.output_row = _DirRow('Output folder (images/ + labels/)', 'Select Output Folder')
        self.weights_edit = QLineEdit()
        wbtn = QPushButton('Browse…')
        wrow = QWidget()
        wh = QHBoxLayout(wrow)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.weights_edit)
        wh.addWidget(wbtn)
        wbtn.clicked.connect(self._pick_weights)

        self.frame_gap_spin = QSpinBox()
        self.frame_gap_spin.setRange(1, 9999)
        self.frame_gap_spin.setValue(int(defaults.get('frame_gap', 10)))
        self.frame_gap_spin.setToolTip('Label every N-th frame (higher = fewer frames).')

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setValue(float(defaults.get('conf', 0.25)))

        self.pred_iou_spin = QDoubleSpinBox()
        self.pred_iou_spin.setRange(0.1, 0.99)
        self.pred_iou_spin.setDecimals(2)
        self.pred_iou_spin.setValue(float(defaults.get('pred_iou', 0.5)))

        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.1, 0.99)
        self.dedup_spin.setDecimals(2)
        self.dedup_spin.setValue(float(defaults.get('dedup_iou', 0.7)))

        self.task_combo = QComboBox()
        self.task_combo.addItems(['Auto (from model)', 'Detect', 'Segment', 'Pose'])
        self.save_score_check = QCheckBox('Append score to each YOLO line (for filtering)')
        self.save_score_check.setChecked(True)

        form.addRow('Output:', self.output_row)
        form.addRow('Model weights:', wrow)
        form.addRow('Frame interval:', self.frame_gap_spin)
        form.addRow('Confidence:', self.conf_spin)
        form.addRow('Predict IoU:', self.pred_iou_spin)
        form.addRow('Dedup IoU:', self.dedup_spin)
        form.addRow('Task:', self.task_combo)
        form.addRow('', self.save_score_check)
        root.addLayout(form)

        self.output_row.setText(defaults.get('output_dir', ''))
        self.weights_edit.setText(defaults.get('weights', ''))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _pick_weights(self):
        _browse_file(
            self.weights_edit, self, 'Select Model Weights',
            'Model weights (*.pt *.pth);;All files (*.*)')

    def values(self) -> Dict[str, Any]:
        task_map = {0: 'auto', 1: 'detect', 2: 'segment', 3: 'pose'}
        return {
            'output_dir': self.output_row.text(),
            'weights': self.weights_edit.text().strip(),
            'frame_gap': self.frame_gap_spin.value(),
            'conf': self.conf_spin.value(),
            'pred_iou': self.pred_iou_spin.value(),
            'dedup_iou': self.dedup_spin.value(),
            'task_hint': task_map.get(self.task_combo.currentIndex(), 'auto'),
            'save_score': self.save_score_check.isChecked(),
        }


class MakeDatasetsDialog(QDialog):
    """Build train/val YOLO dataset from labeled images (VOC or YOLO labels)."""

    def __init__(self, parent=None, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle('Make Datasets')
        defaults = defaults or {}
        root = QVBoxLayout(self)

        form = QFormLayout()
        self.images_row = _DirRow('Labeled images folder', 'Select Images Folder')
        self.labels_row = _DirRow('Label files folder', 'Select Labels Folder')
        self.output_row = _DirRow('Dataset output folder', 'Select Output Folder')

        self.format_combo = QComboBox()
        self.format_combo.addItems(['YOLO (txt)', 'VOC (xml)'])

        self.single_class_check = QCheckBox('Convert all classes to a single class')
        self.single_class_edit = QLineEdit()
        self.single_class_edit.setPlaceholderText('Single class name (e.g. object)')
        self.single_class_edit.setEnabled(False)
        self.single_class_check.toggled.connect(self.single_class_edit.setEnabled)

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText('classes.txt for multi-class VOC (optional)')
        cbtn = QPushButton('Browse…')
        crow = QWidget()
        ch = QHBoxLayout(crow)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.addWidget(self.classes_edit)
        ch.addWidget(cbtn)
        cbtn.clicked.connect(self._pick_classes)

        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.5, 0.95)
        self.split_spin.setDecimals(2)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setValue(float(defaults.get('split', 0.7)))

        form.addRow('Images:', self.images_row)
        form.addRow('Labels:', self.labels_row)
        form.addRow('Output:', self.output_row)
        form.addRow('Label format:', self.format_combo)
        form.addRow('', self.single_class_check)
        form.addRow('Single class:', self.single_class_edit)
        form.addRow('Class list (VOC):', crow)
        form.addRow('Train ratio:', self.split_spin)
        root.addLayout(form)

        self.images_row.setText(defaults.get('image_dir', ''))
        self.labels_row.setText(defaults.get('label_dir', ''))
        self.output_row.setText(defaults.get('output_dir', 'datasets'))
        if defaults.get('single_class'):
            self.single_class_check.setChecked(True)
            self.single_class_edit.setText(defaults['single_class'])

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _pick_classes(self):
        _browse_file(self.classes_edit, self, 'Select classes.txt', 'Text (*.txt);;All files (*.*)')

    def values(self) -> Dict[str, Any]:
        return {
            'image_dir': self.images_row.text(),
            'label_dir': self.labels_row.text(),
            'output_dir': self.output_row.text(),
            'voc': self.format_combo.currentIndex() == 1,
            'single_class': self.single_class_check.isChecked(),
            'single_class_name': self.single_class_edit.text().strip(),
            'classes_file': self.classes_edit.text().strip(),
            'split': self.split_spin.value(),
        }

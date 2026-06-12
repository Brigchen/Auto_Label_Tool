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

        self.cross_class_dedup_check = QCheckBox(
            'Cross-class dedup (overlapping different classes → keep highest score)')
        self.cross_class_dedup_check.setChecked(bool(defaults.get('cross_class_dedup', True)))
        self.cross_class_dedup_check.setToolTip(
            'After same-class dedup: if two boxes of different classes overlap (IoU > Dedup IoU), '
            'keep only the higher-confidence box.'
        )

        form.addRow('Quick pick:', self.preset_combo)
        form.addRow('Model weights:', wrow)
        form.addRow('Confidence (conf):', self.conf_spin)
        form.addRow('Predict IoU:', self.pred_iou_spin)
        form.addRow('Dedup IoU:', self.dedup_spin)
        form.addRow('', self.cross_class_dedup_check)
        root.addLayout(form)

        hint = QLabel(
            'Confidence: minimum detection score. '
            'Predict IoU: NMS overlap. Dedup IoU: drop duplicate same-class boxes after predict. '
            'Cross-class dedup removes overlapping boxes of different classes (keeps highest score).'
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
            'cross_class_dedup': self.cross_class_dedup_check.isChecked(),
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

        self.cross_class_dedup_check = QCheckBox(
            'Cross-class dedup (different classes, same overlap → keep highest score)')
        self.cross_class_dedup_check.setChecked(bool(defaults.get('cross_class_dedup', True)))

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
        form.addRow('', self.cross_class_dedup_check)
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
            'cross_class_dedup': self.cross_class_dedup_check.isChecked(),
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

        self.cross_class_dedup_check = QCheckBox(
            'Cross-class dedup (different classes, same overlap → keep highest score)')
        self.cross_class_dedup_check.setChecked(bool(defaults.get('cross_class_dedup', True)))

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
        form.addRow('', self.cross_class_dedup_check)
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
            'cross_class_dedup': self.cross_class_dedup_check.isChecked(),
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

        self.test_check = QCheckBox('Create test split (for FishVision test report)')
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.05, 0.40)
        self.test_spin.setDecimals(2)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(float(defaults.get('test_split', 0.10)))
        self.test_spin.setEnabled(False)
        self.test_check.toggled.connect(self._on_test_toggled)
        if defaults.get('create_test'):
            self.test_check.setChecked(True)
            self.test_spin.setEnabled(True)
            self._refresh_train_ratio_max()

        test_row = QWidget()
        test_layout = QHBoxLayout(test_row)
        test_layout.setContentsMargins(0, 0, 0, 0)
        test_layout.addWidget(self.test_spin)
        test_layout.addStretch(1)
        self.split_spin.valueChanged.connect(lambda _v: self._refresh_train_ratio_max())
        self.test_spin.valueChanged.connect(lambda _v: self._refresh_train_ratio_max())

        form.addRow('Images:', self.images_row)
        form.addRow('Labels:', self.labels_row)
        form.addRow('Output:', self.output_row)
        form.addRow('Label format:', self.format_combo)
        form.addRow('', self.single_class_check)
        form.addRow('Single class:', self.single_class_edit)
        form.addRow('Class list (VOC):', crow)
        form.addRow('Train ratio:', self.split_spin)
        form.addRow('', self.test_check)
        form.addRow('Test ratio:', test_row)

        self.uw_offline_check = QCheckBox('Underwater offline augment (train split only)')
        self.uw_copies_spin = QSpinBox()
        self.uw_copies_spin.setRange(0, 5)
        self.uw_copies_spin.setValue(int(defaults.get('uw_offline_copies', 0)))
        self.uw_copies_spin.setEnabled(False)
        self.uw_strength_combo = QComboBox()
        self.uw_strength_combo.addItems(['light', 'medium', 'strong'])
        idx = self.uw_strength_combo.findText(str(defaults.get('uw_offline_strength', 'medium')))
        if idx >= 0:
            self.uw_strength_combo.setCurrentIndex(idx)
        self.uw_strength_combo.setEnabled(False)
        self.uw_offline_check.toggled.connect(self._on_uw_offline_toggled)
        if defaults.get('uw_offline'):
            self.uw_offline_check.setChecked(True)
            self.uw_copies_spin.setEnabled(True)
            self.uw_strength_combo.setEnabled(True)
            if self.uw_copies_spin.value() == 0:
                self.uw_copies_spin.setValue(2)

        uw_row = QWidget()
        uw_layout = QHBoxLayout(uw_row)
        uw_layout.setContentsMargins(0, 0, 0, 0)
        uw_layout.addWidget(QLabel('copies:'))
        uw_layout.addWidget(self.uw_copies_spin)
        uw_layout.addWidget(QLabel('strength:'))
        uw_layout.addWidget(self.uw_strength_combo)
        uw_layout.addStretch(1)

        form.addRow('', self.uw_offline_check)
        form.addRow('UW augment:', uw_row)

        hint = QLabel(
            'Images and labels in all subfolders are included. '
            'Nested pairs match by relative path first, then by file name. '
            'Duplicate names in different folders are renamed (e.g. a/b/c.jpg → a_b_c.jpg). '
            'When test split is enabled, val ratio = 1 − train − test; data.yaml includes test:. '
            'Underwater offline augment adds photometric variants (attenuation/haze/turbidity/…) '
            'to train images only; labels are copied unchanged.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color: gray; font-size: 11px;')
        root.addLayout(form)
        root.addWidget(hint)

        self.images_row.setText(defaults.get('image_dir', ''))
        self.labels_row.setText(defaults.get('label_dir', ''))
        self.output_row.setText(defaults.get('output_dir', 'datasets'))
        if defaults.get('voc'):
            self.format_combo.setCurrentIndex(1)
        if defaults.get('single_class'):
            self.single_class_check.setChecked(True)
        if defaults.get('single_class_name'):
            self.single_class_edit.setText(defaults.get('single_class_name', ''))
        if defaults.get('classes_file'):
            self.classes_edit.setText(defaults.get('classes_file', ''))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _pick_classes(self):
        _browse_file(self.classes_edit, self, 'Select classes.txt', 'Text (*.txt);;All files (*.*)')

    def _on_test_toggled(self, checked: bool):
        self.test_spin.setEnabled(checked)
        self._refresh_train_ratio_max()

    def _on_uw_offline_toggled(self, checked: bool):
        self.uw_copies_spin.setEnabled(checked)
        self.uw_strength_combo.setEnabled(checked)
        if checked and self.uw_copies_spin.value() == 0:
            self.uw_copies_spin.setValue(2)

    def _refresh_train_ratio_max(self):
        if self.test_check.isChecked():
            max_train = max(0.5, 0.95 - self.test_spin.value())
            if self.split_spin.value() > max_train:
                self.split_spin.setValue(max_train)
            self.split_spin.setMaximum(max_train)
        else:
            self.split_spin.setMaximum(0.95)

    def values(self) -> Dict[str, Any]:
        create_test = self.test_check.isChecked()
        return {
            'image_dir': self.images_row.text(),
            'label_dir': self.labels_row.text(),
            'output_dir': self.output_row.text(),
            'voc': self.format_combo.currentIndex() == 1,
            'single_class': self.single_class_check.isChecked(),
            'single_class_name': self.single_class_edit.text().strip(),
            'classes_file': self.classes_edit.text().strip(),
            'split': self.split_spin.value(),
            'create_test': create_test,
            'test_split': self.test_spin.value() if create_test else 0.0,
            'uw_offline': self.uw_offline_check.isChecked(),
            'uw_offline_copies': self.uw_copies_spin.value() if self.uw_offline_check.isChecked() else 0,
            'uw_offline_strength': self.uw_strength_combo.currentText(),
        }


class LabelSelfRefineDialog(QDialog):
    """Self-refine dataset labels using a trained model (GT vs predict)."""

    def __init__(self, parent=None, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle('标注自修正 (Label Self-Refine)')
        defaults = defaults or {}
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.weights_edit = QLineEdit(defaults.get('weights', ''))
        wbtn = QPushButton('Browse…')
        wrow = QWidget()
        wh = QHBoxLayout(wrow)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.weights_edit)
        wh.addWidget(wbtn)
        wbtn.clicked.connect(lambda: _browse_file(
            self.weights_edit, self, 'Select weights',
            'Weights (*.pt);;All files (*.*)'))

        self.yaml_edit = QLineEdit(defaults.get('data_yaml', ''))
        ybtn = QPushButton('Browse…')
        yrow = QWidget()
        yh = QHBoxLayout(yrow)
        yh.setContentsMargins(0, 0, 0, 0)
        yh.addWidget(self.yaml_edit)
        yh.addWidget(ybtn)
        ybtn.clicked.connect(lambda: _browse_file(
            self.yaml_edit, self, 'Select data.yaml',
            'YAML (*.yaml *.yml);;All files (*.*)'))

        self.out_edit = QLineEdit(defaults.get('out_dir', ''))
        obtn = QPushButton('Browse…')
        orow = QWidget()
        oh = QHBoxLayout(orow)
        oh.setContentsMargins(0, 0, 0, 0)
        oh.addWidget(self.out_edit)
        oh.addWidget(obtn)
        obtn.clicked.connect(lambda: _browse_dir(self.out_edit, self, 'Report output folder'))

        self.split_train = QCheckBox('train')
        self.split_val = QCheckBox('val')
        self.split_test = QCheckBox('test')
        self.split_train.setChecked(defaults.get('split_train', True))
        self.split_val.setChecked(defaults.get('split_val', True))
        self.split_test.setChecked(defaults.get('split_test', False))
        split_row = QWidget()
        sh = QHBoxLayout(split_row)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(self.split_train)
        sh.addWidget(self.split_val)
        sh.addWidget(self.split_test)
        sh.addStretch(1)

        self.dry_run_check = QCheckBox('Dry-run only (report, do not write labels)')
        self.dry_run_check.setChecked(defaults.get('dry_run', True))

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.3, 0.95)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(float(defaults.get('iou_match', 0.5)))

        self.fix_conf_spin = QDoubleSpinBox()
        self.fix_conf_spin.setRange(0.3, 1.0)
        self.fix_conf_spin.setDecimals(2)
        self.fix_conf_spin.setValue(float(defaults.get('min_fix_conf', 0.65)))

        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 1.0)
        self.min_score_spin.setDecimals(3)
        self.min_score_spin.setValue(float(defaults.get('min_score_keep', 0.35)))

        self.drop_score_check = QCheckBox('Drop GT lines with score column below threshold')
        self.drop_score_check.setChecked(defaults.get('drop_low_score', True))
        self.fix_class_check = QCheckBox('Fix class when IoU match + pred conf high')
        self.fix_class_check.setChecked(defaults.get('fix_wrong_class', True))
        self.cross_dedup_check = QCheckBox('Cross-class dedup after refine')
        self.cross_dedup_check.setChecked(defaults.get('cross_class_dedup', True))
        self.review_check = QCheckBox('Export uncertain samples to review_queue/')
        self.review_check.setChecked(defaults.get('export_review', True))

        self.review_only_check = QCheckBox(
            'Review queue only (reuse report CSV, skip inference)')
        self.review_only_check.setChecked(defaults.get('review_only', False))
        self.review_only_check.toggled.connect(self._on_review_only_toggled)

        self.html_only_check = QCheckBox(
            'HTML summary only (from existing CSV, skip inference & review export)')
        self.html_only_check.setChecked(defaults.get('html_only', False))
        self.html_only_check.toggled.connect(self._on_html_only_toggled)

        self.export_html_check = QCheckBox('Generate HTML summary (index.html)')
        self.export_html_check.setChecked(defaults.get('export_html', True))

        self.interactive_review_check = QCheckBox(
            'Open interactive review in browser after run (confirm & apply fixes)')
        self.interactive_review_check.setChecked(defaults.get('open_interactive_review', False))

        form.addRow('Weights (best.pt):', wrow)
        form.addRow('data.yaml:', yrow)
        form.addRow('Report output:', orow)
        form.addRow('Splits:', split_row)
        form.addRow('Match IoU:', self.iou_spin)
        form.addRow('Min conf to fix class:', self.fix_conf_spin)
        form.addRow('Min score to keep:', self.min_score_spin)
        form.addRow('', self.dry_run_check)
        form.addRow('', self.drop_score_check)
        form.addRow('', self.fix_class_check)
        form.addRow('', self.cross_dedup_check)
        form.addRow('', self.review_check)
        form.addRow('', self.review_only_check)
        form.addRow('', self.html_only_check)
        form.addRow('', self.export_html_check)
        form.addRow('', self.interactive_review_check)
        root.addLayout(form)

        hint = QLabel(
            'Compares each label file with model predictions. '
            'Conservative auto-fix: drop low-score auto labels, fix class on high-conf overlaps. '
            'Apply mode backs up labels/ to labels_backup_<timestamp>/ before writing. '
            'Review-only: point Report output to a folder that already contains '
            'label_self_refine_report.csv (no weights / no GPU inference). '
            'HTML-only: build index.html from CSV in seconds.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color: gray; font-size: 11px;')
        root.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        if defaults.get('review_only'):
            self.review_only_check.setChecked(True)
            self._on_review_only_toggled(True)
        if defaults.get('html_only'):
            self.html_only_check.setChecked(True)
            self._on_html_only_toggled(True)

    def _refine_widgets_full_run(self):
        return (
            self.split_train, self.split_val, self.split_test,
            self.dry_run_check, self.iou_spin, self.fix_conf_spin, self.min_score_spin,
            self.drop_score_check, self.fix_class_check, self.cross_dedup_check,
            self.review_check,
        )

    def _on_review_only_toggled(self, checked: bool):
        if checked and self.html_only_check.isChecked():
            self.html_only_check.setChecked(False)
        self._sync_offline_modes()

    def _on_html_only_toggled(self, checked: bool):
        if checked and self.review_only_check.isChecked():
            self.review_only_check.setChecked(False)
        self._sync_offline_modes()

    def _sync_offline_modes(self):
        offline = self.review_only_check.isChecked() or self.html_only_check.isChecked()
        for w in self._refine_widgets_full_run():
            w.setEnabled(not offline)
        self.weights_edit.setEnabled(True)
        if self.review_only_check.isChecked():
            self.review_check.setChecked(True)
        if self.html_only_check.isChecked():
            self.export_html_check.setChecked(True)
            self.export_html_check.setEnabled(False)
            self.interactive_review_check.setEnabled(True)
        elif not offline:
            self.export_html_check.setEnabled(True)
            self.interactive_review_check.setEnabled(True)

    def _report_csv_path(self) -> str:
        out = self.out_edit.text().strip()
        if not out:
            return ""
        return os.path.join(out, "label_self_refine_report.csv")

    def _on_accept(self):
        from PyQt5.QtWidgets import QMessageBox
        if not self.yaml_edit.text().strip():
            QMessageBox.warning(self, 'Label Self-Refine', 'Please select data.yaml.')
            return
        if not self.out_edit.text().strip():
            QMessageBox.warning(self, 'Label Self-Refine', 'Please select report output folder.')
            return
        if self.html_only_check.isChecked() or self.review_only_check.isChecked():
            csv_path = self._report_csv_path()
            if not csv_path or not os.path.isfile(csv_path):
                QMessageBox.warning(
                    self, 'Label Self-Refine',
                    'Offline mode requires an existing report CSV:\n'
                    f'{csv_path or "(empty)"}')
                return
            self.accept()
            return
        if not self.weights_edit.text().strip():
            QMessageBox.warning(self, 'Label Self-Refine', 'Please select model weights.')
            return
        if not any([self.split_train.isChecked(), self.split_val.isChecked(),
                    self.split_test.isChecked()]):
            QMessageBox.warning(self, 'Label Self-Refine', 'Select at least one split.')
            return
        self.accept()

    def values(self) -> Dict[str, Any]:
        splits = []
        if self.split_train.isChecked():
            splits.append('train')
        if self.split_val.isChecked():
            splits.append('val')
        if self.split_test.isChecked():
            splits.append('test')
        return {
            'weights': self.weights_edit.text().strip(),
            'data_yaml': self.yaml_edit.text().strip(),
            'out_dir': self.out_edit.text().strip(),
            'splits': splits,
            'dry_run': self.dry_run_check.isChecked(),
            'iou_match': self.iou_spin.value(),
            'min_fix_conf': self.fix_conf_spin.value(),
            'min_score_keep': self.min_score_spin.value(),
            'drop_low_score': self.drop_score_check.isChecked(),
            'fix_wrong_class': self.fix_class_check.isChecked(),
            'cross_class_dedup': self.cross_dedup_check.isChecked(),
            'export_review': self.review_check.isChecked(),
            'review_only': self.review_only_check.isChecked(),
            'html_only': self.html_only_check.isChecked(),
            'export_html': self.export_html_check.isChecked(),
            'open_interactive_review': self.interactive_review_check.isChecked(),
        }


class UnderwaterOfflineAugmentDialog(QDialog):
    """Add photometric underwater variants to an existing YOLO dataset."""

    def __init__(self, parent=None, defaults: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle('Underwater Offline Augment')
        defaults = defaults or {}
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.dataset_row = _DirRow('YOLO dataset root', 'Select Dataset Root')
        self.dataset_row.setText(defaults.get('dataset_root', ''))

        self.copies_spin = QSpinBox()
        self.copies_spin.setRange(0, 5)
        self.copies_spin.setValue(int(defaults.get('copies', 2)))

        self.strength_combo = QComboBox()
        self.strength_combo.addItems(['light', 'medium', 'strong'])
        idx = self.strength_combo.findText(str(defaults.get('strength', 'medium')))
        if idx >= 0:
            self.strength_combo.setCurrentIndex(idx)

        self.split_train = QCheckBox('train')
        self.split_val = QCheckBox('val')
        self.split_test = QCheckBox('test')
        self.split_train.setChecked(defaults.get('split_train', True))
        self.split_val.setChecked(defaults.get('split_val', False))
        self.split_test.setChecked(defaults.get('split_test', False))
        split_row = QWidget()
        sh = QHBoxLayout(split_row)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(self.split_train)
        sh.addWidget(self.split_val)
        sh.addWidget(self.split_test)
        sh.addStretch(1)

        form.addRow('Dataset root:', self.dataset_row)
        form.addRow('Extra copies / image:', self.copies_spin)
        form.addRow('Strength:', self.strength_combo)
        form.addRow('Splits:', split_row)
        root.addLayout(form)

        hint = QLabel(
            'Writes new image/label pairs next to originals (suffix __uwN_<type>). '
            'Labels are copied unchanged. Skips images already augmented. '
            'Recommended: train only; keep val/test untouched for fair evaluation.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color: gray; font-size: 11px;')
        root.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_accept(self):
        from PyQt5.QtWidgets import QMessageBox
        root_path = self.dataset_row.text().strip()
        if not root_path or not os.path.isdir(root_path):
            QMessageBox.warning(self, 'Underwater Offline Augment', 'Dataset root is invalid.')
            return
        if self.copies_spin.value() <= 0:
            QMessageBox.warning(self, 'Underwater Offline Augment', 'Copies must be at least 1.')
            return
        if not any([self.split_train.isChecked(), self.split_val.isChecked(),
                    self.split_test.isChecked()]):
            QMessageBox.warning(self, 'Underwater Offline Augment', 'Select at least one split.')
            return
        self.accept()

    def values(self) -> Dict[str, Any]:
        splits = []
        if self.split_train.isChecked():
            splits.append('train')
        if self.split_val.isChecked():
            splits.append('val')
        if self.split_test.isChecked():
            splits.append('test')
        return {
            'dataset_root': self.dataset_row.text().strip(),
            'copies': self.copies_spin.value(),
            'strength': self.strength_combo.currentText(),
            'splits': splits,
        }

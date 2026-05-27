# -*- coding: utf-8 -*-
"""
FishVision_Train_GUI — 选项卡式完整训练控制台。

提供:
  - 5 选项卡: 基本参数 / 优化器 / 数据增强 / 硬件高级 / 自动调参
  - 训练线程管理 (普通训练 + 自动调参)
  - TensorBoard 集成 + results.csv 实时监控
  - 配置持久化 (退出自动保存 → configs/config_FVT_last.ini，下次启动优先加载)
"""
from __future__ import annotations

import configparser
import os
import queue
import subprocess
import sys
import threading
import time as _time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Ultralytics default deterministic=True needs this before torch loads
if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

try:
    from ultralytics.utils import SETTINGS
    SETTINGS["tensorboard"] = True
except Exception:
    pass

import torch
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox, QAbstractSpinBox,
)
from ultralytics import YOLO

from libs.repo_paths import repo_root, configs_dir
from libs.lr_schedules import SCHEDULE_CHOICES
from libs.yolo_weights import resolve_yolo_checkpoint
from libs.fvt_train_runner import (
    run_training, audit_cuda_environment, format_cuda_audit_message,
    get_last_exported_weight, checkpoint_has_nonfinite,
)
from libs.train_monitor import (
    TensorBoardServer, ResultsWatcher, collect_metrics,
    find_latest_run_dir, list_matching_run_dirs, resolve_run_dir,
)
from libs.train_augment_panel import AugmentPanel
from libs.hyperparam_search import HyperparamSearchPanel


# ══════════════════════════════════════════════════════════════
#  训练工作线程
# ══════════════════════════════════════════════════════════════


class TrainingWorker(QThread):
    """执行 model.train() 或 model.tune() 的后台线程。
    
    自动捕获 stdout/stderr 到 log_queue，供 GUI 实时显示。
    """

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)  # 单行实时训练进度 (s/it, ETA)
    epoch_signal = pyqtSignal(dict)  # 指标 dict
    heartbeat_signal = pyqtSignal()  # 定时心跳, 供 GUI 检测是否卡死

    def __init__(self, train_params: dict, parent=None):
        super().__init__(parent)
        self.train_params = train_params
        self.log_queue: queue.Queue = queue.Queue()

    def run(self):
        # ── 重定向 stdout/stderr 到 log_queue ──
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        class _QueueWriter:
            def __init__(self, q, orig):
                self._q = q
                self._orig = orig
            def write(self, text):
                if text and text.strip():
                    self._q.put_nowait(text)
                self._orig.write(text)
                self._orig.flush()
            def flush(self):
                self._orig.flush()

            def isatty(self):
                return False

        sys.stdout = _QueueWriter(self.log_queue, old_stdout)
        sys.stderr = _QueueWriter(self.log_queue, old_stderr)

        try:
            kw = dict(self.train_params)
            tune_enabled = kw.pop("tune_enabled", False)

            if tune_enabled:
                wdir = os.path.join(repo_root(), "weights")
                weights = kw.get("weights_file", "")
                w = resolve_yolo_checkpoint(weights, app_weights_dir=wdir)
                model = YOLO(w)
                kw.pop("weights_file", None)
                self._run_tune(model, kw)
                status = "finished"
            else:
                def _on_live(line: str) -> None:
                    self.progress_signal.emit(line)

                status = run_training(
                    self.train_params,
                    log=print,
                    on_live_progress=_on_live,
                )

            self.finished_signal.emit(status)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] {e}")
            import traceback
            tb = traceback.format_exc()
            self.log_signal.emit(tb)
            print(f"[FVT][ERROR] {e}\n{tb}")
            self.finished_signal.emit("error")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _log_ts(self, msg: str):
        """带时间戳的日志。"""
        ts = _time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    def _run_tune(self, model, kw: dict):
        task = kw.pop("task", "detect")
        if task == "track":
            task = "detect"
        if task and task not in ("detect", "det", ""):
            kw["task"] = task

        # device 修正
        dev = kw.get("device", "auto")
        if dev == "auto":
            kw["device"] = 0 if torch.cuda.is_available() else "cpu"

        iterations = kw.pop("iterations", 100)
        tune_epochs = kw.pop("tune_epochs", 50)
        space = kw.pop("space", {})
        kw["epochs"] = tune_epochs

        self.log_signal.emit(f"Starting auto tune: iterations={iterations}, "
                             f"epochs_per_iter={tune_epochs}")
        model.tune(
            data=kw.get("data"),
            epochs=tune_epochs,
            iterations=iterations,
            batch=kw.get("batch", 8),
            imgsz=kw.get("imgsz", 640),
            space=space or None,
            optimizer=kw.get("optimizer", "AdamW"),
            save_dir=os.path.join(repo_root(), "runs", "tune"),
        )


# ══════════════════════════════════════════════════════════════
#  UI 子面板
# ══════════════════════════════════════════════════════════════

class _BasicTab(QWidget):
    """基本参数选项卡。包含基于显存+权重的自动 batch 估算。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    # ── 显存与 batch 估算 ──

    @staticmethod
    def _estimate_batch_from_vram(weights_path: str, imgsz: int) -> int:
        """根据 GPU 显存 + 权重文件大小估算 batch 值。"""
        free_mem_mb = 0
        # 尝试 pynvml (NVIDIA GPU)
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                best_free = 0
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    free = int(info.free / (1024 * 1024))
                    if free > best_free:
                        best_free = free
                free_mem_mb = best_free
        except Exception:
            pass

        # 无 GPU / pynvml 不可用 → 用系统内存粗略估算
        if free_mem_mb == 0:
            try:
                import psutil
                free_mem_mb = int(psutil.virtual_memory().available / (1024 * 1024))
                free_mem_mb = int(free_mem_mb * 0.3)
            except Exception:
                free_mem_mb = 8000

        # 权重文件大小 (MB)
        fsize_mb = 1
        try:
            if weights_path and os.path.isfile(weights_path):
                fsize_mb = max(1, int(os.path.getsize(weights_path) / (1024 * 1024)))
        except Exception:
            pass

        # 保守估算: 留出 35% 显存余量给 CUDA 上下文和中间变量
        # 公式: (free_mem * safety * 35000) / (fsize * imgsz^2)
        safety = 0.65
        if fsize_mb > 0 and imgsz > 0:
            raw = int((free_mem_mb * 35000 * safety) / (fsize_mb * imgsz * imgsz))
        else:
            raw = int((free_mem_mb * safety) / (imgsz * imgsz / 35))

        return max(1, min(raw, 128))

    def _refresh_batch_estimate(self):
        """根据当前权重路径和 imgsz 重新估算 batch，更新提示（不自动填入）。"""
        w = self.weights_edit.text().strip()
        imgsz = self.imgsz_spin.value()
        if not w:
            self.batch_hint.setText("")
            return
        try:
            sug = self._estimate_batch_from_vram(w, imgsz)
            cur = self.batch_spin.value()
            tip = (f"根据显存估算建议 batch ≤ {sug}（当前={cur}，"
                   f"请自行按需调整以避免 OOM）")
            self.batch_hint.setText(tip)
        except Exception as e:
            self.batch_hint.setText(f"无法估算 batch: {e}")

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 数据集
        gb_data = QGroupBox("数据集")
        fl_data = QFormLayout()
        self.yaml_edit = QLineEdit()
        yaml_btn = QPushButton("浏览...")
        yaml_btn.clicked.connect(self._browse_yaml)
        yaml_row = QWidget()
        yh = QHBoxLayout(yaml_row)
        yh.setContentsMargins(0, 0, 0, 0)
        yh.addWidget(self.yaml_edit)
        yh.addWidget(yaml_btn)
        fl_data.addRow("data yaml:", yaml_row)
        gb_data.setLayout(fl_data)
        layout.addWidget(gb_data)

        # 模型
        gb_model = QGroupBox("预训练模型与任务")
        fl_model = QFormLayout()
        self.weights_edit = QLineEdit()
        w_btn = QPushButton("浏览...")
        w_btn.clicked.connect(self._browse_weights)
        w_row = QWidget()
        wh = QHBoxLayout(w_row)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.weights_edit)
        wh.addWidget(w_btn)
        fl_model.addRow("权重文件:", w_row)

        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "pose", "segment", "obb", "classify", "track"])
        task_tip = QLabel("(track 使用 detect 权重训练，追踪在推理阶段启用)")
        task_tip.setStyleSheet("color: #888; font-size: 10px;")
        fl_model.addRow("任务类型:", self.task_combo)
        fl_model.addRow("", task_tip)
        gb_model.setLayout(fl_model)
        layout.addWidget(gb_model)

        # 训练基本参数
        gb_hyp = QGroupBox("训练参数")
        fh = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(100)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 4096)
        self.batch_spin.setSpecialValueText("auto")
        self.batch_spin.setValue(8)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(1280)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("run name")
        self.name_edit.setText("Training")

        self.project_edit = QLineEdit()
        self.project_edit.setPlaceholderText("runs/  (留空默认)")
        self.project_edit.setText("")

        fh.addRow("Epochs:", self.epochs_spin)
        fh.addRow("Batch:", self.batch_spin)
        fh.addRow("Imgsz:", self.imgsz_spin)
        fh.addRow("Run Name:", self.name_edit)
        fh.addRow("Project:", self.project_edit)
        gb_hyp.setLayout(fh)
        layout.addWidget(gb_hyp)

        # batch 估算提示
        self.batch_hint = QLabel("")
        self.batch_hint.setWordWrap(True)
        self.batch_hint.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(self.batch_hint)

        layout.addStretch()

        # 信号连接：权重 / imgsz 变化时自动估算 batch
        self.weights_edit.editingFinished.connect(self._refresh_batch_estimate)
        self.imgsz_spin.valueChanged.connect(self._refresh_batch_estimate)

    def _browse_yaml(self):
        start = os.path.join(repo_root(), "datasets")
        path, _ = QFileDialog.getOpenFileName(self, "选择 data yaml", start, "YAML (*.yaml *.yml);;All (*.*)")
        if path:
            self.yaml_edit.setText(path)

    def _browse_weights(self):
        start = os.path.join(repo_root(), "weights")
        path, _ = QFileDialog.getOpenFileName(self, "选择预训练权重", start, "Weights (*.pt *.pth);;All (*.*)")
        if path:
            self.weights_edit.setText(path)

    def get_values(self) -> dict:
        return {
            "data": self.yaml_edit.text().strip(),
            "weights_file": self.weights_edit.text().strip(),
            "task": self.task_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "name": self.name_edit.text().strip() or "train",
            "project": self.project_edit.text().strip(),
        }

    def set_values(self, d: dict):
        if d.get("data"):
            self.yaml_edit.setText(d["data"])
        if d.get("weights_file"):
            self.weights_edit.setText(d["weights_file"])
        if d.get("task"):
            idx = self.task_combo.findText(d["task"])
            if idx >= 0:
                self.task_combo.setCurrentIndex(idx)
        if d.get("epochs"):
            self.epochs_spin.setValue(int(d["epochs"]))
        if d.get("batch") is not None:
            self.batch_spin.setValue(int(d["batch"]))
        if d.get("imgsz"):
            self.imgsz_spin.setValue(int(d["imgsz"]))
        if d.get("name"):
            self.name_edit.setText(d["name"])
        if "project" in d:
            self.project_edit.setText(d["project"])


class _SciFloatEdit(QLineEdit):
    """可自由输入的小数字段（支持 0.005、1e-3）；读取时再校验范围。"""

    def __init__(self, value: float = 0.001, minimum: float = 1e-7, maximum: float = 1.0, parent=None):
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self.setAlignment(Qt.AlignRight)
        self.setValue(value)

    def setValue(self, value) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = self._minimum
        v = max(self._minimum, min(self._maximum, v))
        self.setText(f"{v:g}")

    def value(self) -> float:
        text = self.text().strip().replace(",", "")
        if not text:
            return self._minimum
        try:
            v = float(text)
        except ValueError:
            return self._minimum
        return max(self._minimum, min(self._maximum, v))


def _tune_double_spin(
    minimum: float,
    maximum: float,
    value: float,
    decimals: int,
    step: float,
) -> QDoubleSpinBox:
    """DoubleSpinBox that allows typing full number before committing."""
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setAlignment(Qt.AlignRight)
    # PyQt5: UpDownArrows (Qt6 renamed to UpDownButtons)
    spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
    return spin


class _OptimizerTab(QWidget):
    """优化器参数选项卡。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── 优化器 ──
        gb_opt = QGroupBox("优化器")
        fm_opt = QFormLayout()
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(
            ["AdamW", "SGD", "Adam", "NAdam", "RAdam", "RMSProp", "Adamax", "MuSGD", "auto"]
        )
        self.momentum_spin = _tune_double_spin(0.0, 1.0, 0.937, 3, 0.01)
        self.weight_decay_spin = _tune_double_spin(0.0, 1.0, 0.0005, 6, 0.0001)
        fm_opt.addRow("Optimizer:", self.optimizer_combo)
        fm_opt.addRow("Momentum:", self.momentum_spin)
        fm_opt.addRow("Weight Decay:", self.weight_decay_spin)
        gb_opt.setLayout(fm_opt)
        layout.addWidget(gb_opt)

        # ── 学习率调度 ──
        gb_lr = QGroupBox("学习率调度")
        fm_lr = QFormLayout()

        self.lr_schedule_combo = QComboBox()
        for key, label in SCHEDULE_CHOICES:
            self.lr_schedule_combo.addItem(label, key)
        self.lr_schedule_combo.setToolTip(
            "Linear: lr0 → lr0×lrf 线性\n"
            "Cosine: 全周期余弦降至 lr0×lrf\n"
            "OneCycleLR: 先升至 lr0 再降至 lr0×lrf（按 epoch 包络）\n"
            "Truncated Cosine: 前若干 epoch 余弦衰减，之后保持 lr0×lrf"
        )

        self.lr0_edit = _SciFloatEdit(0.001, 1e-7, 1.0)
        self.lr0_edit.setToolTip("初始学习率 lr0（AdamW 常用 1e-3 ~ 1e-2）")

        self.lrf_edit = _SciFloatEdit(0.01, 1e-7, 1.0)
        self.lrf_edit.setToolTip("最低 LR 比例：lr_min = lr0 × lrf")

        self.lr_cos_tmax_spin = _tune_double_spin(0.05, 1.0, 0.75, 2, 0.05)
        self.lr_cos_tmax_spin.setToolTip(
            "截断余弦：在前 N% 的总 epoch 内完成余弦衰减，其余 epoch 保持 lr0×lrf"
        )
        self.lr_cos_tmax_label = QLabel("余弦阶段占比:")
        self.lr_cos_tmax_row = QWidget()
        _tmax_row = QHBoxLayout(self.lr_cos_tmax_row)
        _tmax_row.setContentsMargins(0, 0, 0, 0)
        _tmax_row.addWidget(self.lr_cos_tmax_spin)
        _tmax_row.addWidget(QLabel("(占总 epochs)"))
        _tmax_row.addStretch()

        self.final_lr_label = QLabel("")
        self.final_lr_label.setStyleSheet("color: #555; font-size: 10px;")
        self._refresh_final_lr_hint()

        fm_lr.addRow("调度方式:", self.lr_schedule_combo)
        fm_lr.addRow("lr0 (初始):", self.lr0_edit)
        fm_lr.addRow("lrf (最低比例):", self.lrf_edit)
        fm_lr.addRow(self.lr_cos_tmax_label, self.lr_cos_tmax_row)
        fm_lr.addRow("", self.final_lr_label)
        gb_lr.setLayout(fm_lr)
        layout.addWidget(gb_lr)

        self.lr0_edit.editingFinished.connect(self._refresh_final_lr_hint)
        self.lrf_edit.editingFinished.connect(self._refresh_final_lr_hint)
        self.lr_schedule_combo.currentIndexChanged.connect(self._on_lr_schedule_changed)
        self.lr_cos_tmax_spin.valueChanged.connect(self._refresh_final_lr_hint)
        self._on_lr_schedule_changed()

        # ── Warmup ──
        gb_warm = QGroupBox("Warmup 预热")
        fm_warm = QFormLayout()
        self.warmup_epochs_spin = _tune_double_spin(0.0, 50.0, 3.0, 1, 0.5)
        self.warmup_momentum_spin = _tune_double_spin(0.0, 1.0, 0.8, 3, 0.05)
        self.warmup_bias_lr_edit = _SciFloatEdit(0.1, 1e-7, 1.0)
        self.warmup_bias_lr_edit.setToolTip("warmup 阶段 bias 参数使用的学习率（Ultralytics warmup_bias_lr）")
        fm_warm.addRow("Warmup Epochs:", self.warmup_epochs_spin)
        fm_warm.addRow("Warmup Momentum:", self.warmup_momentum_spin)
        fm_warm.addRow("Warmup Bias LR:", self.warmup_bias_lr_edit)
        gb_warm.setLayout(fm_warm)
        layout.addWidget(gb_warm)

        # ── 早停 ──
        gb_other = QGroupBox("早停")
        fm_other = QFormLayout()
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 10000)
        self.patience_spin.setValue(15)
        self.patience_spin.setSpecialValueText("off")
        self.patience_spin.setKeyboardTracking(False)
        fm_other.addRow("Patience:", self.patience_spin)
        gb_other.setLayout(fm_other)
        layout.addWidget(gb_other)

        tip = QLabel(
            "说明：lrf 为比例，lr_min = lr0×lrf。"
            "OneCycleLR 在 epoch 粒度先升至 lr0 再降至 lr_min；"
            "截断余弦可在前 N% epoch 余弦衰减后保持 lr_min。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(tip)
        layout.addStretch()

    def _on_lr_schedule_changed(self):
        sched = self.lr_schedule_combo.currentData() or "linear"
        show_tmax = sched == "truncated_cosine"
        self.lr_cos_tmax_label.setVisible(show_tmax)
        self.lr_cos_tmax_row.setVisible(show_tmax)
        self._refresh_final_lr_hint()

    def _refresh_final_lr_hint(self):
        lr0 = self.lr0_edit.value()
        lrf = self.lrf_edit.value()
        sched = self.lr_schedule_combo.currentData() or "linear"
        lr_min = lr0 * lrf
        if sched == "onecycle":
            hint = f"→ OneCycle: lr0×lrf={lr_min:.6g} ↑ lr0={lr0:g} ↓ lr0×lrf={lr_min:.6g}"
        elif sched == "truncated_cosine":
            pct = self.lr_cos_tmax_spin.value()
            hint = (
                f"→ 前 {pct:.0%} epochs 余弦降至 lr0×lrf={lr_min:.6g}，"
                f"之后保持 {lr_min:.6g}"
            )
        elif sched == "cosine":
            hint = f"→ 余弦衰减至 lr0×lrf = {lr0:g} × {lrf:g} = {lr_min:.6g}"
        else:
            hint = f"→ 线性衰减至 lr0×lrf = {lr0:g} × {lrf:g} = {lr_min:.6g}"
        self.final_lr_label.setText(hint)

    def get_values(self) -> dict:
        sched = self.lr_schedule_combo.currentData() or "linear"
        return {
            "optimizer": self.optimizer_combo.currentText(),
            "lr0": self.lr0_edit.value(),
            "lrf": self.lrf_edit.value(),
            "lr_schedule": sched,
            "lr_cos_tmax": self.lr_cos_tmax_spin.value(),
            "cos_lr": sched == "cosine",
            "momentum": self.momentum_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "warmup_epochs": self.warmup_epochs_spin.value(),
            "warmup_momentum": self.warmup_momentum_spin.value(),
            "warmup_bias_lr": self.warmup_bias_lr_edit.value(),
            "patience": self.patience_spin.value(),
        }

    def set_values(self, d: dict):
        if d.get("optimizer"):
            idx = self.optimizer_combo.findText(d["optimizer"])
            if idx >= 0:
                self.optimizer_combo.setCurrentIndex(idx)
        if d.get("lr0") is not None:
            self.lr0_edit.setValue(d["lr0"])
        if d.get("lrf") is not None:
            self.lrf_edit.setValue(d["lrf"])
        sched = d.get("lr_schedule")
        if sched:
            idx = self.lr_schedule_combo.findData(str(sched).lower())
            if idx >= 0:
                self.lr_schedule_combo.setCurrentIndex(idx)
        elif d.get("cos_lr") is not None:
            want_cos = str(d["cos_lr"]).lower() in ("1", "true", "yes")
            idx = self.lr_schedule_combo.findData("cosine" if want_cos else "linear")
            if idx >= 0:
                self.lr_schedule_combo.setCurrentIndex(idx)
        if d.get("lr_cos_tmax") is not None:
            self.lr_cos_tmax_spin.setValue(float(d["lr_cos_tmax"]))
        if d.get("momentum") is not None:
            self.momentum_spin.setValue(float(d["momentum"]))
        if d.get("weight_decay") is not None:
            self.weight_decay_spin.setValue(float(d["weight_decay"]))
        if d.get("warmup_epochs") is not None:
            self.warmup_epochs_spin.setValue(float(d["warmup_epochs"]))
        if d.get("warmup_momentum") is not None:
            self.warmup_momentum_spin.setValue(float(d["warmup_momentum"]))
        if d.get("warmup_bias_lr") is not None:
            self.warmup_bias_lr_edit.setValue(d["warmup_bias_lr"])
        if d.get("patience") is not None:
            self.patience_spin.setValue(int(d["patience"]))
        self._on_lr_schedule_changed()


class _HardwareTab(QWidget):
    """硬件与高级设置选项卡。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        gb = QGroupBox("硬件与高级选项")
        fm = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(["0", "auto", "1", "2", "3", "cpu"])
        self.device_combo.setCurrentText("0")
        self.device_combo.setToolTip("默认 GPU 0；如果无 GPU 请选 cpu")

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(0)
        self.workers_spin.setToolTip("Windows 建议 = 0（免多进程），有把握再调到 2")

        self.pretrained_cb = QCheckBox("使用预训练权重")
        self.pretrained_cb.setChecked(True)

        self.resume_cb = QCheckBox("恢复中断训练")
        self.resume_cb.setChecked(False)
        self.resume_cb.setToolTip(
            "从同名 run 下最新的 last.pt 继续训练（不会新建 fish_one-2 目录）"
        )

        self.exist_ok_cb = QCheckBox("覆盖同名 Run (exist_ok)")
        self.exist_ok_cb.setChecked(False)
        self.exist_ok_cb.setToolTip(
            "勾选后复用 runs/<task>/<name> 目录；未勾选且目录已存在时会自动递增为 name-2"
        )

        self.val_cb = QCheckBox("训练中验证")
        self.val_cb.setChecked(True)

        self.plots_cb = QCheckBox("生成训练曲线图")
        self.plots_cb.setChecked(True)

        self.amp_cb = QCheckBox("AMP 混合精度 (amp)")
        self.amp_cb.setChecked(True)
        self.amp_cb.setToolTip(
            "开启时 Ultralytics 会加载 yolo26n.pt 做 AMP 检测，"
            "首次可能等待数分钟；若长时间卡住可取消勾选"
        )

        self.save_period_spin = QSpinBox()
        self.save_period_spin.setRange(-1, 1000)
        self.save_period_spin.setValue(-1)
        self.save_period_spin.setSpecialValueText("off")

        self.cache_combo = QComboBox()
        self.cache_combo.addItems(["", "ram", "disk"])

        fm.addRow("Device:", self.device_combo)
        fm.addRow("Workers:", self.workers_spin)
        fm.addRow("", self.pretrained_cb)
        fm.addRow("", self.resume_cb)
        fm.addRow("", self.exist_ok_cb)
        fm.addRow("", self.val_cb)
        fm.addRow("", self.plots_cb)
        fm.addRow("", self.amp_cb)
        fm.addRow("Save Period:", self.save_period_spin)
        fm.addRow("Cache:", self.cache_combo)
        gb.setLayout(fm)
        layout.addWidget(gb)
        layout.addStretch()

    def get_values(self) -> dict:
        vals = {
            "device": self.device_combo.currentText(),
            "workers": self.workers_spin.value(),
            "pretrained": self.pretrained_cb.isChecked(),
            "resume": self.resume_cb.isChecked(),
            "exist_ok": self.exist_ok_cb.isChecked(),
            "val": self.val_cb.isChecked(),
            "plots": self.plots_cb.isChecked(),
            "amp": self.amp_cb.isChecked(),
        }
        if self.save_period_spin.value() > 0:
            vals["save_period"] = self.save_period_spin.value()
        cache = self.cache_combo.currentText()
        if cache:
            vals["cache"] = cache
        return vals

    def set_values(self, d: dict):
        if d.get("device"):
            idx = self.device_combo.findText(str(d["device"]))
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
        if d.get("workers") is not None:
            self.workers_spin.setValue(int(d["workers"]))
        if "pretrained" in d:
            self.pretrained_cb.setChecked(str(d["pretrained"]).lower() in ("1", "true", "yes"))
        if "resume" in d:
            self.resume_cb.setChecked(str(d["resume"]).lower() in ("1", "true", "yes"))
        if "exist_ok" in d:
            self.exist_ok_cb.setChecked(str(d["exist_ok"]).lower() in ("1", "true", "yes"))
        if "val" in d:
            self.val_cb.setChecked(str(d["val"]).lower() in ("1", "true", "yes"))
        if "plots" in d:
            self.plots_cb.setChecked(str(d["plots"]).lower() in ("1", "true", "yes"))
        if "amp" in d:
            self.amp_cb.setChecked(str(d["amp"]).lower() in ("1", "true", "yes"))
        if "cache" in d:
            idx = self.cache_combo.findText(str(d["cache"]))
            if idx >= 0:
                self.cache_combo.setCurrentIndex(idx)
        if d.get("save_period") is not None:
            self.save_period_spin.setValue(int(d["save_period"]))


class TabbedTrainGUI(QWidget):
    """选项卡式训练控制台主窗口。"""

    def __init__(self):
        super().__init__()
        self._worker: TrainingWorker = None
        self._tensorboard = TensorBoardServer()
        self._watcher: ResultsWatcher = None
        self._current_run_dir: str = ""  # 最新训练的输出目录
        self._training_workers_warned = False  # 只提醒一次
        self._last_train_activity = 0.0
        self._pending_progress_line = ""

        self._log_timer = QTimer(self)
        self._log_timer.setInterval(300)
        self._log_timer.timeout.connect(self._poll_log_queue)

        self._progress_throttle = QTimer(self)
        self._progress_throttle.setSingleShot(True)
        self._progress_throttle.setInterval(150)
        self._progress_throttle.timeout.connect(self._flush_progress_label)

        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.setInterval(30000)  # 30 秒无输出即警告
        self._watchdog_timer.timeout.connect(self._watchdog_check)

        self._init_ui()
        self._load_config()
        self._show_cuda_status()

    def _show_cuda_status(self):
        """Warn on startup if GPU and PyTorch versions mismatch (e.g. RTX 50 + torch 1.x)."""
        try:
            audit = audit_cuda_environment(timeout_sec=8.0)
            if audit.get("warnings"):
                self.msg_label.setText(format_cuda_audit_message(audit)[:500])
                self.msg_label.setStyleSheet(
                    "border: 1px solid #e6a700; padding: 6px; "
                    "font: 10pt 'Microsoft YaHei'; background: #fffbe6;"
                )
        except Exception:
            pass

    def _check_cuda_before_train(self, params: dict) -> bool:
        """Return False if user cancels after CUDA compatibility warning."""
        audit = audit_cuda_environment(timeout_sec=10.0)
        if not audit.get("warnings"):
            return True

        msg = format_cuda_audit_message(audit)
        if not audit.get("compatible", True):
            QMessageBox.critical(
                self,
                "GPU / PyTorch 不兼容",
                msg + "\n\n请升级 PyTorch 后再训练，或将 Device 设为 cpu。",
            )
            return False

        if audit.get("force_amp_off"):
            params["amp"] = False
            self.tab_hardware.amp_cb.setChecked(False)
            reply = QMessageBox.warning(
                self,
                "GPU / PyTorch 警告",
                msg + "\n\n将自动关闭 AMP 并继续。建议仍升级 PyTorch 以获得完整 GPU 支持。",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            return reply == QMessageBox.Ok
        return True

    def _init_ui(self):
        self.setWindowTitle("FishVision Trainer - 完整训练控制台")
        self.setMinimumSize(800, 620)

        layout = QVBoxLayout(self)

        # ── 选项卡 ──
        self.tabs = QTabWidget()
        self.tab_basic = _BasicTab()
        self.tab_optimizer = _OptimizerTab()
        self.tab_augment = AugmentPanel()
        self.tab_hardware = _HardwareTab()
        self.tab_tune = HyperparamSearchPanel()

        self.tabs.addTab(self.tab_basic, "基本参数")
        self.tabs.addTab(self.tab_optimizer, "优化器")
        self.tabs.addTab(self.tab_augment, "数据增强")
        self.tabs.addTab(self.tab_hardware, "硬件/高级")
        self.tabs.addTab(self.tab_tune, "自动调参")

        layout.addWidget(self.tabs)

        # ── 状态消息 ──
        self.msg_label = QLabel("就绪")
        self.msg_label.setStyleSheet(
            "border: 1px solid #ccc; padding: 6px; font: 10pt 'Microsoft YaHei';"
        )
        self.msg_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.msg_label)

        # ── 实时指标显示 ──
        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet("color: #333; font: 10pt monospace;")
        self.metrics_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.metrics_label)

        # ── 进度条 ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ── 日志面板 ──
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(160)
        self.log_box.setStyleSheet(
            "font: 9pt 'Consolas','Courier New',monospace;"
            "background: #1e1e1e; color: #d4d4d4;"
        )
        self.log_box.setVisible(False)
        layout.addWidget(self.log_box)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        self.tb_btn = QPushButton("TensorBoard")
        self.tb_btn.clicked.connect(self._open_tensorboard)

        self.terminal_btn = QPushButton("终端运行")
        self.terminal_btn.setToolTip("在独立 cmd 窗口执行训练（可看到完整输出）")
        self.terminal_btn.clicked.connect(self._run_in_terminal)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.terminal_btn)
        btn_layout.addWidget(self.tb_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    # ── 参数收集 ──

    def _collect_all_params(self) -> dict:
        """从所有选项卡收集参数。"""
        params = {}
        params.update(self.tab_basic.get_values())
        params.update(self.tab_optimizer.get_values())
        params.update(self.tab_augment.get_values())
        params.update(self.tab_hardware.get_values())

        # 自动调参参数
        tune_params = self.tab_tune.get_tune_params()
        params.update(tune_params)

        # 过滤空值
        return {k: v for k, v in params.items() if v is not None and v != ""}

    def _collect_save_params(self) -> dict:
        """Collect all UI values for session persistence (keep empty/false fields)."""
        params = {}
        params.update(self.tab_basic.get_values())
        params.update(self.tab_optimizer.get_values())
        params.update(self.tab_augment.get_values())
        params.update(self.tab_hardware.get_values())
        params["save_period"] = self.tab_hardware.save_period_spin.value()
        params["cache"] = self.tab_hardware.cache_combo.currentText()
        params.update(self.tab_tune.get_save_state())
        return params

    # ── 训练控制 ──

    def _start_training(self):
        params = self._collect_all_params()

        # 基本校验
        if not params.get("data"):
            QMessageBox.warning(self, "参数错误", "请选择 data yaml 文件")
            return
        if not params.get("weights_file"):
            QMessageBox.warning(self, "参数错误", "请选择预训练权重文件")
            return

        task = params.get("task", "detect")
        project = params.get("project", "")
        name = params.get("name", "train")
        if not params.get("resume") and not params.get("exist_ok"):
            existing = list_matching_run_dirs(task, project, name)
            if existing:
                latest = os.path.basename(find_latest_run_dir(task, project, name))
                reply = QMessageBox.question(
                    self,
                    "Run 目录已存在",
                    f"已存在 run「{latest}」。\n"
                    f"继续将自动递增目录名（如 {name}-2、{name}-3 …）。\n\n"
                    f"若要继续同一 run，请勾选「恢复中断训练」或「覆盖同名 Run」。",
                    QMessageBox.Ok | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if reply != QMessageBox.Ok:
                    return

        weights = params.get("weights_file", "")
        bad, reason = checkpoint_has_nonfinite(weights)
        if bad:
            QMessageBox.critical(
                self,
                "权重文件损坏",
                f"权重含 NaN/Inf（{reason}）。\n\n"
                "请改用干净预训练权重（如 weights/yolo26n.pt），"
                "或删除损坏的 runs/.../weights/*.pt 后重新训练。",
            )
            return

        wf = str(weights).replace("\\", "/")
        if (
            "/runs/" in wf
            and float(params.get("lr0", 0) or 0) >= 0.005
            and params.get("amp", True)
        ):
            reply = QMessageBox.warning(
                self,
                "微调易 NaN",
                "当前：runs/ 下 checkpoint + lr0≥0.005 + AMP 开启，"
                "训练易出现 loss NaN。\n\n"
                "建议：lr0 改为 0.001、关闭 AMP，或换 weights/yolo26n.pt。\n\n"
                "仍按当前参数开始？",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Ok:
                return

        if not self._check_cuda_before_train(params):
            return

        self._save_config(self._collect_save_params())

        # 禁用 UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setEnabled(False)
        # 进度由 msg_label 单行刷新；跑马灯 progress_bar 会与 QLabel 争用绘制导致 QPainter 警告
        self.progress_bar.setVisible(False)
        self.metrics_label.setText("")
        self.msg_label.setText("训练中...")

        # 显示日志面板，启动定时轮询 + 看门狗
        self.log_box.clear()
        self.log_box.setVisible(True)
        self._log_timer.start()
        self._watchdog_timer.start()
        self._training_workers_warned = False
        self._last_train_activity = _time.time()

        # 启动工作线程
        self._worker = TrainingWorker(params)
        self._worker.log_signal.connect(self._on_log)
        self._worker.progress_signal.connect(
            self._on_train_progress, Qt.QueuedConnection,
        )
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

        # 启动 results.csv 监控（动态跟踪 fish_one / fish_one-N 实际目录）
        project = params.get("project", "")
        name = params.get("name", "train")
        task = params.get("task", "detect")
        run_dir = resolve_run_dir(task, project, name)
        self._current_run_dir = run_dir
        self._watcher = ResultsWatcher(
            "",
            callback=self._on_metrics,
            interval=5.0,
            task=task,
            project=project,
            name=name,
        )
        self._watcher.start()

    def _stop_training(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
            self.msg_label.setText("训练已终止")

        self._stop_watcher()
        self._enable_ui()

    def _on_log(self, msg: str):
        self.msg_label.setText(msg)

    def _on_train_progress(self, line: str):
        """单行刷新：batch 进度、s/it、ETA（限频，避免 QPainter 冲突）。"""
        if not line:
            return
        self._pending_progress_line = line
        self._last_train_activity = _time.time()
        self._training_workers_warned = False
        if not self._progress_throttle.isActive():
            self._flush_progress_label()
            self._progress_throttle.start()

    def _flush_progress_label(self):
        if self._pending_progress_line:
            self.msg_label.setText(self._pending_progress_line)

    def _on_finished(self, status: str):
        self._stop_watcher()
        self._enable_ui()
        if status == "finished":
            exported = get_last_exported_weight()
            if exported:
                self.msg_label.setText(f"训练完成，权重已保存: {exported}")
                QMessageBox.information(
                    self,
                    "训练完成",
                    f"训练已完成。\n\nbest.pt 已复制至:\n{exported}",
                )
            else:
                self.msg_label.setText("训练完成！可在 runs/ 下查看结果")
                QMessageBox.information(
                    self, "训练完成", "训练已完成。点击「TensorBoard」查看曲线。"
                )
        else:
            self.msg_label.setText("训练出错，请查看控制台日志")

    def _on_metrics(self, metrics: dict):
        """ResultsWatcher 回调：更新指标显示。"""
        epoch = int(metrics.get("epoch", 0))
        m50 = metrics.get("metrics/mAP50(B)", None)
        m50_95 = metrics.get("metrics/mAP50-95(B)", None)
        loss = metrics.get("train/box_loss", None)
        lr = metrics.get("x/lr0", None)

        parts = [f"Epoch: {epoch}"]
        if m50 is not None:
            parts.append(f"mAP50: {m50:.4f}")
        if m50_95 is not None:
            parts.append(f"mAP50-95: {m50_95:.4f}")
        if loss is not None:
            parts.append(f"Loss: {loss:.4f}")
        if lr is not None:
            parts.append(f"LR: {lr:.6f}")

        self.metrics_label.setText(" | ".join(parts))

        # 训练中 progress_bar 已隐藏；此处仅更新 metrics_label，避免与 msg_label 争用绘制

    def _stop_watcher(self):
        if self._watcher:
            self._watcher.stop()
            self._watcher = None

    def _enable_ui(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.tabs.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._log_timer.stop()
        self._progress_throttle.stop()
        self._watchdog_timer.stop()

    def _poll_log_queue(self):
        """从训练线程的 log_queue 读取并追加到日志面板。"""
        if self._worker is None:
            return
        try:
            lines = []
            while True:
                text = self._worker.log_queue.get_nowait()
                lines.append(text)
        except queue.Empty:
            pass
        if lines:
            text = "".join(lines)
            self._last_train_activity = _time.time()
            self.log_box.setUpdatesEnabled(False)
            try:
                cursor = self.log_box.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(text)
                self.log_box.setTextCursor(cursor)
            finally:
                self.log_box.setUpdatesEnabled(True)
            scrollbar = self.log_box.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            for line in text.splitlines():
                if "Epoch" in line and ("完成" in line or "mAP50" in line):
                    self._pending_progress_line = line.strip()
                    self._flush_progress_label()
                    break

    def _watchdog_check(self):
        """检测训练线程是否长时间无输出（数据加载卡死）。"""
        if self._worker is None or not self._worker.isRunning():
            return
        if self._training_workers_warned:
            return
        idle_sec = _time.time() - self._last_train_activity
        if idle_sec < 45:
            return
        # 检查 log_queue 最近是否有活动（实时进度走 progress_signal，不算 queue）
        if self._worker.log_queue.empty() and self.log_box.toPlainText().strip():
            self._training_workers_warned = True
            warn = (
                "\n[WARNING] 训练进程长时间无输出。常见原因:\n"
                "  1. AMP 检测 / 数据集扫描（大 imgsz 或 batch=-1 时可能需 5–10 分钟）\n"
                "  2. workers 设置过大 (请到「硬件/高级」选项卡调低 Workers)\n"
                "  3. 数据集路径或标注文件存在问题\n"
                "  4. 若持续卡住，可取消「AMP 混合精度」后重试\n"
            )
            self.log_box.append(warn)
            self.msg_label.setText("⚠ 训练可能卡死，请检查日志")

    # ── TensorBoard ──

    def _open_tensorboard(self):
        runs_root = os.path.join(repo_root(), "runs")
        task = self.tab_basic.task_combo.currentText()
        project = self.tab_basic.project_edit.text().strip()
        name = self.tab_basic.name_edit.text().strip() or "train"
        cur_run = find_latest_run_dir(task, project, name)

        class _TBThread(QThread):
            result = pyqtSignal(str)
            def run(self):
                self.msgs = []
                def _cb(m):
                    self.msgs.append(m)
                # 同时跟踪当前训练目录和所有历史记录
                if self._cur_run and os.path.isdir(self._cur_run):
                    spec = f"current:{self._cur_run},all:{self._runs_root}"
                    self._tb.start(logdir_spec=spec, callback=_cb)
                else:
                    self._tb.start(logdir=self._runs_root, callback=_cb)

        self._current_run_dir = cur_run
        self._tb_thread = _TBThread()
        self._tb_thread._tb = self._tensorboard
        self._tb_thread._cur_run = self._current_run_dir
        self._tb_thread._runs_root = runs_root
        self._tb_thread.result = self._tb_thread.result  # keep ref
        self._tb_thread.finished.connect(self._on_tb_ready)
        self._tb_thread.start()
        self.msg_label.setText("正在启动 TensorBoard...")

    def _run_in_terminal(self):
        """在独立 cmd 窗口执行训练（崩溃后窗口保持打开，显示错误信息）。"""
        import json

        params = self._collect_all_params()
        if not params.get("data") or not params.get("weights_file"):
            QMessageBox.warning(self, "参数错误", "请先选择 data yaml 与权重文件")
            return

        params_blob = json.dumps(params, ensure_ascii=False)

        script = (
            f"import json, os, sys, traceback\n"
            f"os.chdir({repr(repo_root())})\n"
            f"sys.path.insert(0, {repr(repo_root())})\n"
            f"from multiprocessing import freeze_support\n"
            f"freeze_support()\n"
            f"if __name__ == '__main__':\n"
            f"    try:\n"
            f"        from ultralytics.utils import SETTINGS\n"
            f"        SETTINGS['tensorboard'] = True\n"
            f"        from libs.fvt_train_runner import run_training\n"
            f"        params = json.loads({repr(params_blob)})\n"
            f"        print('>>> FishVision terminal training started', flush=True)\n"
            f"        status = run_training(params)\n"
            f"        print('>>> Training finished:', status, flush=True)\n"
            f"    except Exception:\n"
            f"        traceback.print_exc()\n"
            f"    except BaseException as e:\n"
            f"        print(f'>>> Fatal: {{e}}', flush=True)\n"
            f"        traceback.print_exc()\n"
            f"    finally:\n"
            f"        print('>>> Window closes in 30 seconds...', flush=True)\n"
            f"        import time; time.sleep(30)\n"
        )
        tmp_script = os.path.join(repo_root(), "_fvt_debug_run.py")
        with open(tmp_script, "w", encoding="utf-8") as f:
            f.write(script)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        cmd = [sys.executable, "-u", tmp_script]
        subprocess.Popen(
            cmd,
            cwd=repo_root(),
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        self.msg_label.setText("已在独立终端启动训练（AMP/数据集初始化可能需数分钟）")

    def _on_tb_ready(self):
        msgs = getattr(self._tb_thread, 'msgs', [])
        if msgs:
            self.msg_label.setText(msgs[-1])
        else:
            self.msg_label.setText("TensorBoard 启动完成")
        self._tb_thread = None

    # ── 配置持久化 ──

    _SESSION_CONFIG = "config_FVT_last.ini"
    _DEFAULT_CONFIG = "config_FVT.ini"
    _AUG_SAVE_KEYS = (
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
        "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
        "copy_paste", "erasing",
    )
    _BOOL_INI_KEYS = (
        "pretrained", "resume", "exist_ok", "val", "plots", "amp", "tune_enabled", "cos_lr",
    )

    def _session_config_path(self) -> str:
        return os.path.join(configs_dir(), self._SESSION_CONFIG)

    def _default_config_path(self) -> str:
        return os.path.join(configs_dir(), self._DEFAULT_CONFIG)

    def _resolve_config_path(self) -> Optional[str]:
        last = self._session_config_path()
        if os.path.isfile(last):
            return last
        default = self._default_config_path()
        if os.path.isfile(default):
            return default
        return None

    @staticmethod
    def _read_ini(path: str) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        try:
            cfg.read(path, encoding="utf-8")
        except Exception:
            cfg.read(path)
        return cfg

    def _load_config(self):
        path = self._resolve_config_path()
        if not path:
            return

        cfg = self._read_ini(path)
        if not cfg.has_section("settings"):
            return

        s = cfg["settings"]
        self.tab_basic.set_values({
            "data": s.get("yaml_file", ""),
            "weights_file": s.get("weights_file", ""),
            "task": s.get("task", "detect"),
            "epochs": s.get("epochs", "100"),
            "batch": s.get("batch", "-1"),
            "imgsz": s.get("imgsz", "1280"),
            "name": s.get("name", s.get("train_name", "train")),
            "project": s.get("train_project", ""),
        })
        self.tab_optimizer.set_values({
            "optimizer": s.get("optimizer", "AdamW"),
            "lr0": s.get("lr0", "0.001"),
            "lrf": s.get("lrf", "0.01"),
            "cos_lr": s.get("cos_lr", "false"),
            "lr_schedule": s.get("lr_schedule", "linear"),
            "lr_cos_tmax": s.get("lr_cos_tmax", "0.75"),
            "momentum": s.get("momentum", "0.937"),
            "weight_decay": s.get("weight_decay", "0.0005"),
            "warmup_epochs": s.get("warmup_epochs", "3.0"),
            "warmup_momentum": s.get("warmup_momentum", "0.8"),
            "warmup_bias_lr": s.get("warmup_bias_lr", "0.1"),
            "patience": s.get("patience", "15"),
        })
        aug_vals = {k: s.get(k) for k in self._AUG_SAVE_KEYS if s.get(k) is not None}
        if aug_vals:
            self.tab_augment.set_values(aug_vals)
        self.tab_hardware.set_values({
            "device": s.get("device", "0"),
            "workers": s.get("workers", "0"),
            "cache": s.get("cache", ""),
            "amp": s.get("amp", "true"),
            "pretrained": s.get("pretrained", "true"),
            "resume": s.get("resume", "false"),
            "exist_ok": s.get("exist_ok", "false"),
            "val": s.get("val", "true"),
            "plots": s.get("plots", "true"),
            "save_period": s.get("save_period", "-1"),
        })
        self.tab_tune.set_save_state({
            "tune_enabled": s.get("tune_enabled", "false"),
            "iterations": s.get("iterations", "100"),
            "tune_epochs": s.get("tune_epochs", "50"),
        })

    def _save_config(self, params: Optional[dict] = None):
        if params is None:
            params = self._collect_save_params()

        path = self._session_config_path()
        cfg = configparser.ConfigParser()
        cfg["settings"] = {
            "yaml_file": str(params.get("data", "")),
            "weights_file": str(params.get("weights_file", "")),
            "task": str(params.get("task", "detect")),
            "epochs": str(params.get("epochs", 100)),
            "batch": str(params.get("batch", -1)),
            "imgsz": str(params.get("imgsz", 1280)),
            "name": str(params.get("name", "train")),
            "train_name": str(params.get("name", "train")),
            "train_project": str(params.get("project", "")),
            "optimizer": str(params.get("optimizer", "AdamW")),
            "lr0": str(params.get("lr0", 0.001)),
            "lrf": str(params.get("lrf", 0.01)),
            "lr_schedule": str(params.get("lr_schedule", "linear")),
            "lr_cos_tmax": str(params.get("lr_cos_tmax", 0.75)),
            "momentum": str(params.get("momentum", 0.937)),
            "weight_decay": str(params.get("weight_decay", 0.0005)),
            "warmup_epochs": str(params.get("warmup_epochs", 3.0)),
            "warmup_momentum": str(params.get("warmup_momentum", 0.8)),
            "warmup_bias_lr": str(params.get("warmup_bias_lr", 0.1)),
            "patience": str(params.get("patience", 15)),
            "device": str(params.get("device", "0")),
            "workers": str(params.get("workers", 0)),
            "cache": str(params.get("cache", "")),
            "save_period": str(params.get("save_period", -1)),
            "iterations": str(params.get("iterations", 100)),
            "tune_epochs": str(params.get("tune_epochs", 50)),
        }
        for key in self._BOOL_INI_KEYS:
            cfg["settings"][key] = str(bool(params.get(key, False))).lower()
        for key in self._AUG_SAVE_KEYS:
            if key in params:
                cfg["settings"][key] = str(params[key])

        with open(path, "w", encoding="utf-8") as f:
            cfg.write(f)

    # ── 窗口关闭 ──

    def closeEvent(self, event):
        self._stop_training()
        self._tensorboard.stop()
        try:
            self._save_config()
        except Exception:
            pass
        event.accept()


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════

def main():
    from libs.win_qt_taskbar import load_brand_qicon, set_windows_app_user_model_id
    set_windows_app_user_model_id("Brigchen.AutoLabelTool.FishVisionTrain.2")
    app = QApplication(sys.argv)
    icon = load_brand_qicon(repo_root(), "app")
    if not icon.isNull():
        app.setWindowIcon(icon)
    gui = TabbedTrainGUI()
    if not icon.isNull():
        gui.setWindowIcon(icon)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""FishVision Trainer — shared training entry for GUI thread and terminal subprocess."""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Must be set before import torch when Ultralytics deterministic=True
if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from ultralytics import YOLO

from libs.repo_paths import repo_root
from libs.train_monitor import (
    enable_ultralytics_tensorboard,
    find_latest_last_pt,
    find_latest_run_dir,
    resolve_run_base,
)
from libs.yolo_weights import resolve_yolo_checkpoint

LogFn = Callable[[str], None]

_AUG_KEYS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
    "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
    "copy_paste", "erasing", "auto_augment",
)

PYTORCH_CU128_UPGRADE_CMD = (
    "pip install -U --pre torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu128"
)

# Ultralytics tqdm: "  1/120  5.7G  loss...  429/1819 3.4s/it" or "... 100%|..."
_TRAIN_EPOCH_BAR = re.compile(
    r"^\s*(\d+)/(\d+)\s+[\d.]+\s*G.*?(\d+)/(\d+)",
)
# Validation tqdm: "... 154/159 2.3it/s"
_VAL_PROGRESS = re.compile(r"(\d+)/(\d+)\s+[\d.]+it/s")
_BATCH_PROGRESS = re.compile(r"(\d+)/(\d+)\s+[\d.]+s/it")
_LIVE_PROGRESS_HINT = re.compile(
    r"(Caching images|s/it|it/s|\d+%\||\d+/\d+\s+[\d.]+(?:s/it|it/s))",
    re.I,
)
# Ultralytics 表头 / 验证行 / 启动横幅（epoch 回调已汇总指标）
_NOISE_LINE = re.compile(
    r"^(Epoch\s+GPU_mem\b|Logging results to\b|Starting training for\b|"
    r"Class\s+Images\s+Instances\b|Speed:\s)",
    re.I,
)
_VAL_METRICS_ROW = re.compile(
    r"^\s*(all|\S+)\s+\d+\s+\d+\s+\d+\.\d",
    re.I,
)


class TrainProgressStream:
    """TTY: in-place tqdm (\\r). GUI/non-TTY: on_live callback only; epoch summaries go to log."""

    def __init__(
        self,
        training_started: threading.Event,
        on_live: Optional[Callable[[str], None]] = None,
    ):
        self.training_started = training_started
        self.on_live = on_live
        self._carry = ""

    @staticmethod
    def _is_tty(inner) -> bool:
        try:
            return bool(getattr(inner, "isatty", lambda: False)())
        except Exception:
            return False

    @staticmethod
    def _last_segment(text: str) -> str:
        if "\r" in text:
            text = text.split("\r")[-1]
        return text.strip("\033[K ")

    @staticmethod
    def _is_live_progress(line: str) -> bool:
        if not line:
            return False
        if _LIVE_PROGRESS_HINT.search(line):
            return True
        if _TRAIN_EPOCH_BAR.search(line):
            return True
        if "Class" in line and "Instances" in line and _VAL_PROGRESS.search(line):
            return True
        return False

    @staticmethod
    def _skip_noise_line(line: str) -> bool:
        if _NOISE_LINE.search(line):
            return True
        if _VAL_METRICS_ROW.match(line):
            return True
        return False

    @staticmethod
    def _skip_logged_line(line: str) -> bool:
        """Drop mid-epoch train/val bars that would spam the log as separate lines."""
        m = _TRAIN_EPOCH_BAR.search(line)
        if m and int(m.group(3)) != int(m.group(4)):
            return True
        bm = _BATCH_PROGRESS.search(line)
        if bm and int(bm.group(1)) != int(bm.group(2)):
            return True
        vm = _VAL_PROGRESS.search(line)
        if vm and int(vm.group(1)) != int(vm.group(2)):
            return True
        return False

    def _emit_live(self, line: str, inner, raw: str = "") -> None:
        if not line or not self._is_live_progress(line):
            return
        self.training_started.set()
        if self.on_live:
            try:
                self.on_live(line)
            except Exception:
                pass
        if self._is_tty(inner):
            inner.write(raw if raw else f"\r\033[K{line}")
            inner.flush()

    def feed(self, text: str, inner) -> str:
        """Return text to append to a line-oriented log (non-live lines only)."""
        if not text:
            return ""
        is_tty = self._is_tty(inner)

        # Pure tqdm carriage-return update
        if "\r" in text and "\n" not in text.strip():
            self._emit_live(self._last_segment(text), inner, raw=text)
            return ""

        if "\r" in text:
            self._emit_live(self._last_segment(text), inner)
            text = self._last_segment(text)

        self._carry += text
        out: List[str] = []
        while "\n" in self._carry:
            nl = self._carry.find("\n")
            line = self._carry[:nl].strip("\033[K ")
            self._carry = self._carry[nl + 1:]
            if not line:
                continue
            if self._skip_noise_line(line):
                continue
            if self._is_live_progress(line):
                self._emit_live(line, inner)
                if not is_tty and self._skip_logged_line(line):
                    continue
            if self._skip_logged_line(line):
                continue
            if "仍在初始化训练" in line and self.training_started.is_set():
                continue
            out.append(line + "\n")
        return "".join(out)

    def flush(self, inner) -> str:
        if not self._carry.strip():
            self._carry = ""
            return ""
        line = self._carry.strip("\033[K ")
        self._carry = ""
        if self._skip_noise_line(line):
            return ""
        if self._skip_logged_line(line):
            return ""
        if "仍在初始化训练" in line and self.training_started.is_set():
            return ""
        return line + "\n" if line else ""


def _rebind_logging_streams(stream) -> None:
    """Point ultralytics/root StreamHandlers at the filtered stdout wrapper."""
    import logging

    for logger_name in ("", "ultralytics"):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(stream)


def install_train_stream_filters(
    training_started: threading.Event,
    on_live: Optional[Callable[[str], None]] = None,
) -> tuple:
    """Wrap stdout/stderr; live tqdm -> single-line refresh; epoch metrics -> log lines."""
    import sys

    router = TrainProgressStream(training_started, on_live=on_live)
    prev_out, prev_err = sys.stdout, sys.stderr

    class _ProgressStream:
        def __init__(self, inner):
            self._inner = inner

        def write(self, text):
            out = router.feed(text, self._inner)
            if out:
                self._inner.write(out)
            return len(text)

        def flush(self):
            out = router.flush(self._inner)
            if out:
                self._inner.write(out)
            self._inner.flush()

        def isatty(self):
            return router._is_tty(self._inner)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    sys.stdout = _ProgressStream(prev_out)
    sys.stderr = _ProgressStream(prev_err)
    _rebind_logging_streams(sys.stdout)
    return prev_out, prev_err, router


def install_train_stdout_filter(
    training_started: threading.Event,
    on_live: Optional[Callable[[str], None]] = None,
) -> tuple:
    """Backward-compatible alias."""
    prev_out, prev_err, router = install_train_stream_filters(training_started, on_live)
    return prev_out, router


def _parse_torch_version() -> Tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _gpu_arch_supported(major: int, minor: int) -> bool:
    """True if current PyTorch build includes kernels for this SM version."""
    try:
        arch_list = torch.cuda.get_arch_list()
        token = f"sm_{major}{minor}"
        return token in arch_list
    except Exception:
        return False


def cuda_smoke_test(timeout_sec: float = 20.0) -> Tuple[bool, str]:
    """Quick GPU matmul; returns False on error or timeout (incompatible stack)."""
    if not torch.cuda.is_available():
        return True, "cpu"
    result = {"ok": False, "err": ""}

    def _run() -> None:
        try:
            x = torch.randn(256, 256, device="cuda")
            _ = x @ x
            torch.cuda.synchronize()
            result["ok"] = True
        except Exception as exc:
            result["err"] = str(exc)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout_sec)
    if th.is_alive():
        return False, f"GPU 计算超时 ({timeout_sec:.0f}s)"
    if result["err"]:
        return False, result["err"]
    return True, torch.cuda.get_device_name(0)


def audit_cuda_environment(timeout_sec: float = 20.0) -> dict:
    """Inspect torch/CUDA/GPU compatibility; RTX 50 (sm_120) needs torch 2.7+ cu128."""
    info: dict = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "gpu_name": "",
        "sm": (0, 0),
        "compatible": True,
        "force_amp_off": False,
        "warnings": [],
        "upgrade_cmd": PYTORCH_CU128_UPGRADE_CMD,
    }
    if not info["cuda_available"]:
        return info

    info["gpu_name"] = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    info["sm"] = (major, minor)
    tv = _parse_torch_version()
    arch_ok = _gpu_arch_supported(major, minor)

    # Blackwell RTX 50xx = sm_120; older torch wheels only ship up to sm_90
    if major >= 12 and not arch_ok:
        info["force_amp_off"] = True
        info["warnings"].append(
            f"显卡 {info['gpu_name']} (sm_{major}.{minor}) 与当前 PyTorch {info['torch_version']} "
            f"不兼容。Ultralytics AMP 检测会在 GPU 上卡死，已强制关闭 AMP。"
        )
        info["warnings"].append(
            f"建议升级 PyTorch（cu128 / sm_120）:\n  {PYTORCH_CU128_UPGRADE_CMD}"
        )
    elif major >= 9 and tv < (2, 1) and not arch_ok:
        info["force_amp_off"] = True
        info["warnings"].append(
            f"PyTorch {info['torch_version']} 可能不支持 {info['gpu_name']}，已自动关闭 AMP。"
        )

    if info["cuda_available"]:
        ok, msg = cuda_smoke_test(timeout_sec)
        if not ok:
            info["compatible"] = False
            info["force_amp_off"] = True
            info["warnings"].append(f"GPU 自检失败: {msg}")

    return info


def format_cuda_audit_message(audit: dict) -> str:
    """Human-readable audit summary for GUI dialogs."""
    lines: List[str] = []
    if audit.get("gpu_name"):
        sm = audit.get("sm", (0, 0))
        lines.append(f"GPU: {audit['gpu_name']} (sm_{sm[0]}.{sm[1]})")
    lines.append(f"PyTorch: {audit.get('torch_version', '?')}")
    lines.extend(audit.get("warnings") or [])
    if not audit.get("compatible", True):
        lines.append(
            "\n在升级 PyTorch 之前，即使关闭 AMP 也可能无法正常 GPU 训练。"
        )
    return "\n".join(lines)


def _log_ts(msg: str, log: LogFn = print) -> None:
    log(f"[{time.strftime('%H:%M:%S')}] {msg}")


class _SuppressUltralyticsTqdm:
    """Deprecated no-op kept for compatibility."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_epoch_summary_callback(log: LogFn, training_started: threading.Event):
    def _on_fit_epoch_end(trainer) -> None:
        training_started.set()
        import sys

        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass

        epoch = trainer.epoch + 1
        total = trainer.epochs
        parts = [f"Epoch {epoch}/{total} 完成"]

        if hasattr(trainer, "_get_memory"):
            try:
                parts.append(f"GPU {trainer._get_memory():.1f}G")
            except Exception:
                pass

        if trainer.tloss is not None:
            try:
                vals = trainer.tloss.tolist() if hasattr(trainer.tloss, "tolist") else trainer.tloss
                if isinstance(vals, (list, tuple)):
                    parts.append("loss=" + "/".join(f"{float(v):.4g}" for v in vals))
            except Exception:
                pass

        metrics = trainer.metrics or {}
        seen: set = set()
        for key, label in (
            ("metrics/precision(B)", "P"),
            ("metrics/precision", "P"),
            ("metrics/recall(B)", "R"),
            ("metrics/recall", "R"),
            ("metrics/mAP50(B)", "mAP50"),
            ("metrics/mAP50", "mAP50"),
            ("metrics/mAP50-95(B)", "mAP50-95"),
            ("metrics/mAP50-95", "mAP50-95"),
        ):
            if label in seen:
                continue
            if key in metrics and metrics[key] is not None:
                try:
                    parts.append(f"{label}={float(metrics[key]):.4f}")
                    seen.add(label)
                except (TypeError, ValueError):
                    pass

        _log_ts(" | ".join(parts), log)

    return _on_fit_epoch_end


def ensure_amp_probe_weights(app_weights_dir: str, log: LogFn = print) -> None:
    """Ultralytics check_amp() always loads YOLO('yolo26n.pt'); prefetch to avoid silent hang."""
    audit = audit_cuda_environment(timeout_sec=10.0)
    if audit.get("force_amp_off"):
        _log_ts("跳过 AMP 权重预下载（当前 GPU/PyTorch 组合不安全）", log)
        return
    name = "yolo26n.pt"
    local = os.path.join(app_weights_dir, name)
    if os.path.isfile(local):
        return
    try:
        from ultralytics.utils import WEIGHTS_DIR

        if os.path.isfile(os.path.join(WEIGHTS_DIR, name)):
            return
    except Exception:
        pass
    _log_ts("正在准备 AMP 检测权重 yolo26n.pt（首次可能需下载，请稍候）...", log)
    try:
        YOLO(name)
        _log_ts("AMP 检测权重已就绪", log)
    except Exception as exc:
        _log_ts(f"AMP 检测权重准备失败（训练仍会继续）: {exc}", log)


def normalize_train_kwargs(raw: dict, model=None) -> dict:
    """Build kwargs for model.train() from GUI-collected params."""
    kw = dict(raw)
    kw.pop("weights_file", None)
    kw.pop("tune_enabled", None)
    kw.pop("iterations", None)
    kw.pop("tune_epochs", None)
    kw.pop("space", None)

    task = kw.pop("task", "detect")
    if task == "track":
        task = "detect"
    if task and task not in ("detect", "det", ""):
        kw["task"] = task

    for k in _AUG_KEYS:
        if k in kw and kw[k] == "":
            kw.pop(k)

    dev = kw.get("device", "auto")
    if dev == "auto":
        kw["device"] = 0 if torch.cuda.is_available() else "cpu"

    project = kw.pop("project", "") or ""
    kw["project"] = resolve_run_base(task, project)

    resume = bool(kw.get("resume"))
    if resume and model is not None and model.ckpt and model.ckpt.get("epoch", -1) >= 0:
        kw["resume"] = True
        kw["exist_ok"] = True

    return kw


def resolve_training_weights(params: dict, app_weights_dir: str) -> str:
    """Pick checkpoint path, honoring resume -> latest last.pt."""
    weights = params.get("weights_file", "")
    if params.get("resume"):
        last_pt = find_latest_last_pt(
            params.get("task", "detect"),
            params.get("project", ""),
            params.get("name", "train"),
        )
        if last_pt:
            return last_pt
    return resolve_yolo_checkpoint(weights, app_weights_dir=app_weights_dir)


_last_exported_weight: Optional[str] = None


def get_last_exported_weight() -> Optional[str]:
    """Path copied to weights/ after the most recent successful training."""
    return _last_exported_weight


def _safe_name_part(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", (text or "").strip())
    return s.strip("_") or "train"


def _resolve_best_pt(model, params: dict) -> Optional[str]:
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        best = getattr(trainer, "best", None)
        if best is not None and Path(best).is_file():
            return str(Path(best).resolve())
    run_dir = find_latest_run_dir(
        params.get("task", "detect"),
        params.get("project", ""),
        params.get("name", "train"),
    )
    candidate = os.path.join(run_dir, "weights", "best.pt")
    return candidate if os.path.isfile(candidate) else None


def export_best_weight(params: dict, model=None, log: LogFn = print) -> Optional[str]:
    """Copy run best.pt to weights/ as {run_name}_{base_weight}_{date}.pt."""
    global _last_exported_weight

    best_src = _resolve_best_pt(model, params)
    if not best_src:
        _log_ts("未找到 best.pt，跳过复制到 weights/", log)
        _last_exported_weight = None
        return None

    run_name = _safe_name_part(params.get("name", "train"))
    weight_file = params.get("weights_file", "")
    base_weight = _safe_name_part(os.path.splitext(os.path.basename(weight_file))[0])
    date_str = time.strftime("%Y%m%d")

    weights_dir = os.path.join(repo_root(), "weights")
    os.makedirs(weights_dir, exist_ok=True)

    dest_name = f"{run_name}_{base_weight}_{date_str}.pt"
    dest_path = os.path.join(weights_dir, dest_name)
    if os.path.isfile(dest_path):
        n = 2
        while os.path.isfile(dest_path):
            dest_name = f"{run_name}_{base_weight}_{date_str}-{n}.pt"
            dest_path = os.path.join(weights_dir, dest_name)
            n += 1

    shutil.copy2(best_src, dest_path)
    _last_exported_weight = dest_path
    _log_ts(f"best.pt 已复制至 weights/: {dest_path}", log)
    return dest_path


def run_training(
    params: dict,
    log: LogFn = print,
    on_live_progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Execute one training job. Returns 'finished' or 'error'."""
    wdir = os.path.join(repo_root(), "weights")
    try:
        audit = audit_cuda_environment()
        for w in audit.get("warnings") or []:
            _log_ts(f"WARNING: {w}", log)

        params = dict(params)
        if audit.get("force_amp_off") and params.get("amp", True):
            params["amp"] = False
            _log_ts(
                "已自动关闭 AMP（避免 Ultralytics AMP 检测在 incompatible GPU 上卡死）",
                log,
            )

        if (
            not audit.get("compatible", True)
            and str(params.get("device", "0")).lower() != "cpu"
        ):
            log(
                "[ERROR] 当前 PyTorch 不支持本机 GPU，无法安全进行 CUDA 训练。\n"
                f"请执行:\n  {PYTORCH_CU128_UPGRADE_CMD}\n"
                "或在 GUI 中将 Device 设为 cpu（极慢，仅用于测试）。"
            )
            return "error"

        weights = resolve_training_weights(params, wdir)
        if params.get("resume") and weights != params.get("weights_file"):
            _log_ts(f"resume from: {weights}", log)

        resolved = resolve_yolo_checkpoint(weights, app_weights_dir=wdir)
        if resolved != weights:
            _log_ts(f"resolved weights: {resolved}", log)

        if params.get("amp", True):
            ensure_amp_probe_weights(wdir, log=log)
        else:
            _log_ts("AMP 已关闭，跳过 AMP 检测", log)

        if not enable_ultralytics_tensorboard():
            _log_ts("TensorBoard 未启用（pip install tensorboard）", log)

        model = YOLO(resolved)
        kw = normalize_train_kwargs(params, model=model)
        if audit.get("force_amp_off"):
            kw["amp"] = False

        _log_ts(
            f"train() 参数: epochs={kw.get('epochs')}, batch={kw.get('batch')}, "
            f"imgsz={kw.get('imgsz')}, workers={kw.get('workers')}, device={kw.get('device')}",
            log,
        )
        _log_ts(f"数据集: {kw.get('data')}", log)
        if kw.get("batch") == -1:
            _log_ts("batch=-1：AMP 检测后将进行显存探测(auto-batch)，可能较慢", log)
        _log_ts(
            "AMP 检测 / 数据集扫描 / 首次 batch 可能需要数分钟且无新输出，请耐心等待",
            log,
        )

        training_started = threading.Event()
        stop_hb = threading.Event()

        def _heartbeat() -> None:
            while not stop_hb.wait(30.0):
                if training_started.is_set():
                    break
                _log_ts("仍在初始化训练（AMP / 数据集 / dataloader）...", log)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        prev_out, prev_err, _router = install_train_stream_filters(
            training_started, on_live=on_live_progress,
        )
        import sys

        model.add_callback(
            "on_fit_epoch_end",
            _make_epoch_summary_callback(log, training_started),
        )
        try:
            _log_ts(">>> 调用 model.train() >>>", log)
            model.train(**kw)
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass
            _log_ts("model.train() 正常返回", log)
            export_best_weight(params, model=model, log=log)
        finally:
            sys.stdout = prev_out
            sys.stderr = prev_err
            stop_hb.set()
            training_started.set()

        return "finished"
    except Exception as exc:
        log(f"[ERROR] {exc}")
        log(traceback.format_exc())
        return "error"

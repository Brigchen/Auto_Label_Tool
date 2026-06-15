# -*- coding: utf-8 -*-
"""
TrainMonitor — 训练状态监控 + TensorBoard 进程管理。

提供:
  1. TensorBoardServer: 启动/停止 TensorBoard 子进程，打开浏览器
  2. ResultsWatcher: 定期读取 Ultralytics results.csv 获取当前指标
  3. collect_metrics(): 从 CSV 末行解析关键指标
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_task(task: str) -> str:
    """Map GUI task names to Ultralytics run subfolder (runs/<task>/)."""
    t = (task or "detect").strip().lower()
    if t in ("track", "det", ""):
        return "detect"
    return t


def resolve_run_base(task: str = "detect", project: str = "", repo: str = "") -> str:
    """Parent directory for YOLO runs (Ultralytics default: <repo>/runs/<task>/)."""
    from libs.repo_paths import repo_root

    root = repo or repo_root()
    task = normalize_task(task)
    project = (project or "").strip()
    if not project:
        return os.path.join(root, "runs", task)
    if os.path.isabs(project):
        return os.path.abspath(project)
    p = project.replace("\\", "/").strip("/")
    if p.startswith("runs/"):
        return os.path.join(root, *p.split("/"))
    return os.path.join(root, "runs", p)


def resolve_run_dir(task: str, project: str, name: str, repo: str = "") -> str:
    """Expected run directory before increment_path (runs/detect/<name>)."""
    return os.path.join(resolve_run_base(task, project, repo), name or "train")


def list_matching_run_dirs(task: str, project: str, name: str, repo: str = "") -> List[str]:
    """All run dirs named <name>, <name>-2, … under the project base."""
    base = resolve_run_base(task, project, repo)
    if not os.path.isdir(base):
        return []
    name = (name or "train").strip()
    out: List[str] = []
    exact = os.path.join(base, name)
    if os.path.isdir(exact):
        out.append(exact)
    prefix = name + "-"
    try:
        for entry in os.listdir(base):
            if entry.startswith(prefix) and os.path.isdir(os.path.join(base, entry)):
                out.append(os.path.join(base, entry))
    except OSError:
        pass
    return out


def find_latest_run_dir(task: str, project: str, name: str, repo: str = "") -> str:
    """Newest matching run directory (handles fish_one, fish_one-2, …)."""
    dirs = list_matching_run_dirs(task, project, name, repo)
    if not dirs:
        return resolve_run_dir(task, project, name, repo)
    return max(dirs, key=lambda p: os.path.getmtime(p))


def normalize_tb_logdir(path: str) -> str:
    """Forward-slash absolute path (TensorBoard on Windows is picky about backslashes)."""
    if not path:
        return ""
    return str(Path(path).expanduser().resolve()).replace("\\", "/")


def has_tensorboard_events(logdir: str) -> bool:
    """True if directory tree contains TensorBoard event files."""
    root = Path(logdir)
    if not root.is_dir():
        return False
    for pattern in ("events.out.tfevents.*", "*.tfevents.*"):
        if any(root.rglob(pattern)):
            return True
    return False


def pick_tensorboard_logdir(
    task: str = "detect",
    project: str = "",
    name: str = "train",
    repo: str = "",
) -> Tuple[str, str]:
    """Choose logdir with event files; return (path, user hint).

    Strategy (all platforms):
      1. Prefer the specific current run directory (e.g. runs/detect/fish/train3)
         if it exists or has event files — most reliable for immediate dashboards.
      2. Fall back to project base directory (runs/detect/fish) — allows TB to
         auto-discover all sub-runs (train, train2, …) in the left panel.
      3. Fall back to runs/ root as last resort.
    
    On Windows, --logdir_spec is NOT used because colon conflicts with drive
    letters (C:). Use --logdir with the chosen path instead.
    """
    cur_run = find_latest_run_dir(task, project, name, repo)
    run_base = resolve_run_base(task, project, repo)
    runs_root = os.path.join(repo or _repo_root_fallback(), "runs")

    # ── Priority 1: specific current run directory ──
    if has_tensorboard_events(cur_run):
        return normalize_tb_logdir(cur_run), f"当前 run: {os.path.basename(cur_run)}"
    # ── Priority 2: project base directory (allows multi-run discovery) ──
    if has_tensorboard_events(run_base):
        return normalize_tb_logdir(run_base), f"项目目录（含多个run）: {os.path.basename(run_base)}"
    # ── Priority 3: runs root ──
    if has_tensorboard_events(runs_root):
        return normalize_tb_logdir(runs_root), f"runs 根目录: {runs_root}"
    # ── No events yet: use most specific available path ──
    if os.path.isdir(cur_run):
        return normalize_tb_logdir(cur_run), (
            f"当前 run（尚无 TensorBoard 事件文件）: {os.path.basename(cur_run)}"
        )
    if os.path.isdir(run_base):
        return normalize_tb_logdir(run_base), f"项目目录: {os.path.basename(run_base)}"
    return normalize_tb_logdir(runs_root), f"runs 根目录: {runs_root}"


def _repo_root_fallback() -> str:
    from libs.repo_paths import repo_root

    return repo_root()


def find_latest_last_pt(task: str, project: str, name: str, repo: str = "") -> Optional[str]:
    """Newest last.pt among runs sharing the same base name."""
    best: Optional[str] = None
    best_mtime = 0.0
    for run_dir in list_matching_run_dirs(task, project, name, repo):
        last = os.path.join(run_dir, "weights", "last.pt")
        if os.path.isfile(last):
            mtime = os.path.getmtime(last)
            if mtime > best_mtime:
                best_mtime = mtime
                best = last
    return best


def enable_ultralytics_tensorboard() -> bool:
    """Turn on Ultralytics TensorBoard callback (off by default in settings.json)."""
    try:
        import importlib
        from ultralytics.utils import SETTINGS

        SETTINGS["tensorboard"] = True
        from ultralytics.utils.callbacks import tensorboard as tb_mod

        importlib.reload(tb_mod)
        return tb_mod.SummaryWriter is not None
    except Exception:
        return False


def _find_tensorboard() -> Optional[str]:
    """检查 tensorboard 是否可用（import 检测，不产生子进程）。"""
    # 1. Python 模块导入检测（最快，不阻塞）
    try:
        import tensorboard  # noqa: F401
        return f"{sys.executable} -m tensorboard.main"
    except ImportError:
        pass
    # 2. 系统命令回退
    try:
        import shutil
        if shutil.which("tensorboard") is not None:
            return "tensorboard"
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════
#  TensorBoard 进程管理
# ══════════════════════════════════════════════════════════════


class TensorBoardServer:
    """管理 TensorBoard 子进程的生命周期。"""

    def __init__(self, logdir: str = "", port: int = 6006):
        self.logdir = os.path.abspath(logdir) if logdir else ""
        self.port = port
        self._process: Optional[subprocess.Popen] = None
        self._cmd: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, logdir: str = "", port: Optional[int] = None,
              callback=None, logdir_spec: str = "", *, force_restart: bool = False) -> str:
        """启动 TensorBoard（非阻塞）。

        支持 --logdir 或 --logdir_spec 两种模式。
        logdir_spec 格式: "name1:path1,name2:path2"（Windows 下不推荐，盘符冒号会干扰解析）

        若提供 callback(msg: str)，实际启动完成后会异步调用它。
        """
        if port is not None:
            self.port = port
        if logdir:
            self.logdir = normalize_tb_logdir(logdir)
        self._logdir_spec = logdir_spec.strip()

        dir_ok = (
            self._logdir_spec != ""
            or (self.logdir and os.path.isdir(self.logdir.replace("/", os.sep)))
        )
        if not dir_ok:
            return f"日志目录不存在: {self.logdir}"

        if self.is_running:
            if force_restart:
                self.stop()
            else:
                msg = f"TensorBoard 已在运行: http://localhost:{self.port}"
                if callback:
                    callback(msg)
                return msg

        tb = _find_tensorboard()
        if tb is None:
            msg = "未找到 TensorBoard。请执行: pip install tensorboard"
            if callback:
                callback(msg)
            return msg

        # 异步启动
        def _launch():
            try:
                args = tb.split() + ["--port", str(self.port), "--bind_all"]
                # Windows: --logdir_spec 与 C:\ 盘符冲突，统一用 --logdir
                if self._logdir_spec and os.name != "nt":
                    args += ["--logdir_spec", self._logdir_spec]
                else:
                    args += ["--logdir", self.logdir or normalize_tb_logdir(self._logdir_spec)]
                self._process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(1.5)
                url = f"http://localhost:{self.port}"
                if self.is_running:
                    webbrowser.open(url)
                    msg = f"TensorBoard 已启动: {url}  (logdir={self.logdir})"
                else:
                    msg = "TensorBoard 启动失败，请检查安装 (pip install tensorboard)"
            except Exception as e:
                msg = f"TensorBoard 启动异常: {e}"
            if callback:
                try:
                    callback(msg)
                except Exception:
                    pass

        thread = threading.Thread(target=_launch, daemon=True)
        thread.start()

        return "正在启动 TensorBoard..."

    def stop(self):
        """停止 TensorBoard 进程。"""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                self._process.kill()
            self._process = None

    def open_browser(self):
        """打开已运行的 TensorBoard 页面。"""
        if self.is_running:
            webbrowser.open(f"http://localhost:{self.port}")

    def __del__(self):
        self.stop()


# ══════════════════════════════════════════════════════════════
#  results.csv 监控
# ══════════════════════════════════════════════════════════════


def collect_metrics(csv_path: str) -> Dict[str, float]:
    """从 Ultralytics results.csv 末行解析关键指标。

    CSV 列: epoch, train/box_loss, train/cls_loss, train/dfl_loss,
             metrics/precision, metrics/recall, metrics/mAP50, metrics/mAP50-95,
             val/box_loss, val/cls_loss, val/dfl_loss, x/lr0, x/lr1, x/lr2

    Returns:
        dict, 包含当前指标字段；空 dict 表示文件不可读或尚无数据。
    """
    if not csv_path or not os.path.isfile(csv_path):
        return {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {}
            row = None
            for row in reader:
                pass  # 遍历到最后一行
            if row is None:
                return {}
            out: Dict[str, float] = {}
            for key, val in row.items():
                key = key.strip()
                val = val.strip()
                if val == "":
                    continue
                try:
                    out[key] = float(val)
                except ValueError:
                    pass
            return out
    except Exception:
        return {}


class ResultsWatcher(threading.Thread):
    """后台线程：周期读取 results.csv，通过回调推送最新指标。

    用法:
        watcher = ResultsWatcher(csv_path, callback=my_fn, interval=5.0)
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(
        self,
        csv_path: str,
        callback,
        interval: float = 5.0,
        *,
        task: str = "",
        project: str = "",
        name: str = "",
    ):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.callback = callback
        self.interval = interval
        self._task = task
        self._project = project
        self._name = name
        self._dynamic = bool(task and name)
        self._stop_event = threading.Event()
        self._last_epoch = -1

    def _csv_path(self) -> str:
        if self._dynamic:
            run_dir = find_latest_run_dir(self._task, self._project, self._name)
            return os.path.join(run_dir, "results.csv")
        return self.csv_path

    def run(self):
        while not self._stop_event.is_set():
            csv_path = self._csv_path()
            if os.path.isfile(csv_path):
                metrics = collect_metrics(csv_path)
                epoch = int(metrics.get("epoch", -1))
                if epoch > self._last_epoch:
                    self._last_epoch = epoch
                    try:
                        self.callback(metrics)
                    except Exception:
                        pass
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()

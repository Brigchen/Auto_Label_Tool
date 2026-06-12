# -*- coding: utf-8 -*-
"""Background worker for label self-refine (keep GUI responsive)."""
from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from libs.label_self_refine import (
    RefineRules,
    run_export_review_only,
    run_label_self_refine,
    run_refine_html_only,
)


class LabelSelfRefineWorker(QThread):
    """Run label self-refine off the GUI thread."""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(str, str, dict)  # status, report_dir, opts

    def __init__(self, opts: dict, parent=None):
        super().__init__(parent)
        self.opts = opts

    def run(self):
        try:
            def _log(msg: str) -> None:
                self.log_signal.emit(msg)

            def _prog(cur: int, total: int, name: str) -> None:
                self.progress_signal.emit(cur, total, name)

            if self.opts.get("html_only"):
                run_refine_html_only(
                    self.opts["out_dir"],
                    data_yaml=self.opts.get("data_yaml", ""),
                    weights=self.opts.get("weights", ""),
                    log=_log,
                    progress=_prog,
                    build_interactive=bool(self.opts.get("open_interactive_review", False)),
                )
            elif self.opts.get("review_only"):
                run_export_review_only(
                    data_yaml=self.opts["data_yaml"],
                    out_dir=self.opts["out_dir"],
                    log=_log,
                    progress=_prog,
                )
            else:
                rules = RefineRules(
                    iou_match=float(self.opts.get("iou_match", 0.5)),
                    min_fix_conf=float(self.opts.get("min_fix_conf", 0.65)),
                    min_score_keep=float(self.opts.get("min_score_keep", 0.35)),
                    drop_low_score=bool(self.opts.get("drop_low_score", True)),
                    fix_wrong_class=bool(self.opts.get("fix_wrong_class", True)),
                    cross_class_dedup=bool(self.opts.get("cross_class_dedup", True)),
                    export_review=bool(self.opts.get("export_review", True)),
                )
                run_label_self_refine(
                    weights=self.opts["weights"],
                    data_yaml=self.opts["data_yaml"],
                    out_dir=self.opts["out_dir"],
                    splits=self.opts.get("splits") or ["train", "val"],
                    rules=rules,
                    dry_run=bool(self.opts.get("dry_run", True)),
                    device=str(self.opts.get("device", "0")),
                    export_html=bool(self.opts.get("export_html", True)),
                    build_interactive=bool(self.opts.get("open_interactive_review", False)),
                    log=_log,
                    progress=_prog,
                )
            self.finished_signal.emit("finished", self.opts["out_dir"], dict(self.opts))
        except Exception as exc:
            import traceback
            self.log_signal.emit(f"[ERROR] {exc}")
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit("error", "", {})

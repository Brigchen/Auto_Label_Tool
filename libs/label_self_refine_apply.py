# -*- coding: utf-8 -*-
"""Apply confirmed label self-refine actions to YOLO txt files."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

_APPLICABLE = frozenset({
    "fix_class", "drop_low_score", "drop_unmatched_gt", "add_pred", "review_class",
})

_BACKED_UP: Set[str] = set()


def is_applicable_action(action: str) -> bool:
    return action in _APPLICABLE


def backup_label_file(label_path: str, backup_root: Optional[str] = None) -> str:
    """Copy label file once per path to backup_root (default: sibling .backup/)."""
    label_path = str(Path(label_path).resolve())
    if label_path in _BACKED_UP:
        return label_path
    if not os.path.isfile(label_path):
        return label_path
    root = backup_root or str(Path(label_path).parent / ".label_refine_backup")
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(root, f"{Path(label_path).stem}_{stamp}{Path(label_path).suffix}")
    n = 2
    while os.path.isfile(dst):
        dst = os.path.join(root, f"{Path(label_path).stem}_{stamp}_{n}{Path(label_path).suffix}")
        n += 1
    shutil.copy2(label_path, dst)
    _BACKED_UP.add(label_path)
    return dst


def reset_backup_cache() -> None:
    _BACKED_UP.clear()


def _norm_line(line: str) -> str:
    return " ".join((line or "").strip().split())


def apply_line_change(
    label_path: str,
    action: str,
    old_line: str,
    new_line: str,
    *,
    backup_root: Optional[str] = None,
) -> bool:
    """Apply one line-level change. Returns True if file was modified."""
    if not is_applicable_action(action):
        return False
    if not label_path or not os.path.isfile(label_path):
        return False

    with open(label_path, encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    old_n = _norm_line(old_line)
    out_lines: List[str] = []
    changed = False

    if action in ("drop_low_score", "drop_unmatched_gt"):
        for ln in raw_lines:
            if _norm_line(ln) == old_n and old_n:
                changed = True
                continue
            if ln.strip():
                out_lines.append(ln.strip())
    elif action in ("fix_class", "review_class"):
        if not new_line.strip():
            return False
        replaced = False
        for ln in raw_lines:
            if not replaced and old_n and _norm_line(ln) == old_n:
                out_lines.append(new_line.strip())
                replaced = True
                changed = True
            elif ln.strip():
                out_lines.append(ln.strip())
        if not replaced and old_n:
            return False
    elif action == "add_pred":
        if not new_line.strip():
            return False
        out_lines = [ln.strip() for ln in raw_lines if ln.strip()]
        if _norm_line(new_line) not in {_norm_line(x) for x in out_lines}:
            out_lines.append(new_line.strip())
            changed = True
    else:
        return False

    if not changed:
        return False

    backup_label_file(label_path, backup_root)
    body = ("\n".join(out_lines) + "\n") if out_lines else ""
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(body)
    return True

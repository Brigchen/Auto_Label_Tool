# -*- coding: utf-8 -*-
"""Strip optional detection-score column from YOLO txt (6 cols -> 5 for Ultralytics train)."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from libs.yolo_line_parse import DEFAULT_MANUAL_SCORE, normalize_detection_score

LogFn = Callable[[str], None]

CACHE_DIRNAME = '.fvt_ultra_label_cache'


def _is_normalized_bbox_coords(coords: Sequence[float]) -> bool:
    if len(coords) != 4:
        return False
    return all(-0.01 <= float(c) <= 1.01 for c in coords)


def strip_detection_score_parts(parts: Sequence[str]) -> List[str]:
    """If line looks like ``cls cx cy w h score``, drop trailing score."""
    if len(parts) != 6:
        return list(parts)
    try:
        int(parts[0])
        coords = [float(x) for x in parts[1:5]]
        score = float(parts[5])
    except (TypeError, ValueError):
        return list(parts)
    if not (0.0 <= score <= 1.0):
        return list(parts)
    if not _is_normalized_bbox_coords(coords):
        return list(parts)
    return list(parts[:5])


def pad_five_col_with_manual_score_parts(
    parts: Sequence[str],
    default_score: float = DEFAULT_MANUAL_SCORE,
) -> List[str]:
    """Upgrade ``cls cx cy w h`` to 6 columns with *default_score* (manual GT)."""
    if len(parts) != 5:
        return list(parts)
    try:
        int(parts[0])
        coords = [float(x) for x in parts[1:5]]
    except (TypeError, ValueError):
        return list(parts)
    if not _is_normalized_bbox_coords(coords):
        return list(parts)
    return list(parts) + [f"{float(default_score):.6f}"]


def fix_zero_placeholder_score_parts(
    parts: Sequence[str],
    default_score: float = DEFAULT_MANUAL_SCORE,
) -> List[str]:
    """Replace erroneous 6th-column ``0.0`` placeholder with *default_score*."""
    if len(parts) != 6:
        return list(parts)
    try:
        int(parts[0])
        coords = [float(x) for x in parts[1:5]]
        score = float(parts[5])
    except (TypeError, ValueError):
        return list(parts)
    if not _is_normalized_bbox_coords(coords):
        return list(parts)
    if not (0.0 <= score <= 1.0):
        return list(parts)
    if score != 0.0:
        return list(parts)
    return list(parts[:5]) + [f"{float(default_score):.6f}"]


def pad_label_text_with_manual_score(
    text: str,
    default_score: float = DEFAULT_MANUAL_SCORE,
) -> Tuple[str, int]:
    """5-col detect lines → 6-col with manual default score."""
    changed = 0
    out_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out_lines.append('')
            continue
        parts = line.split()
        new_parts = pad_five_col_with_manual_score_parts(parts, default_score)
        if len(new_parts) != len(parts):
            changed += 1
        out_lines.append(' '.join(new_parts))
    body = '\n'.join(out_lines)
    if out_lines:
        body += '\n'
    return body, changed


def fix_zero_placeholder_scores_in_text(
    text: str,
    default_score: float = DEFAULT_MANUAL_SCORE,
) -> Tuple[str, int]:
    """Fix 6-col lines where score=0.0 was written as a manual placeholder."""
    changed = 0
    out_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out_lines.append('')
            continue
        parts = line.split()
        new_parts = fix_zero_placeholder_score_parts(parts, default_score)
        if new_parts != list(parts):
            changed += 1
        out_lines.append(' '.join(new_parts))
    body = '\n'.join(out_lines)
    if out_lines:
        body += '\n'
    return body, changed


def clean_label_text(text: str) -> Tuple[str, int]:
    """Return cleaned file body and number of lines changed."""
    changed = 0
    out_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out_lines.append('')
            continue
        parts = line.split()
        new_parts = strip_detection_score_parts(parts)
        if len(new_parts) != len(parts):
            changed += 1
        out_lines.append(' '.join(new_parts))
    body = '\n'.join(out_lines)
    if out_lines:
        body += '\n'
    return body, changed


def clean_label_file(path: str, *, dry_run: bool = False) -> int:
    """Rewrite one label file; return count of lines stripped."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return 0
    new_text, n = clean_label_text(text)
    if n and not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_text)
    return n


def iter_label_dirs_from_data_yaml(yaml_path: str) -> List[Path]:
    """Resolve ``labels/...`` directories from a YOLO ``data.yaml``."""
    try:
        import yaml
    except ImportError:
        return []

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    root = Path(str(data.get('path', '.'))).expanduser()
    dirs: List[Path] = []
    seen = set()
    for key in ('train', 'val', 'test'):
        rel = data.get(key)
        if not rel:
            continue
        rel_s = str(rel).replace('\\', '/')
        if rel_s.lower().endswith('.txt'):
            continue
        if 'images' in rel_s:
            lbl_rel = rel_s.replace('images', 'labels', 1)
        else:
            lbl_rel = rel_s
        p = (root / lbl_rel).resolve()
        key_id = str(p).lower()
        if key_id not in seen and p.is_dir():
            seen.add(key_id)
            dirs.append(p)
    labels_root = (root / 'labels').resolve()
    if labels_root.is_dir() and str(labels_root).lower() not in seen:
        dirs.append(labels_root)
    return dirs


def sanitize_label_dir(label_dir: str | Path, *, log: Optional[LogFn] = None) -> Tuple[int, int]:
    """Walk *label_dir* and strip 6-col detection lines. Returns (files_changed, lines_changed)."""
    root = Path(label_dir)
    if not root.is_dir():
        return 0, 0
    files_changed = 0
    lines_changed = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith('.txt') or name.lower() == 'classes.txt':
                continue
            path = os.path.join(dirpath, name)
            n = clean_label_file(path)
            if n:
                files_changed += 1
                lines_changed += n
                if log:
                    log('[yolo_label_clean] %s (%d lines)' % (path, n))
    return files_changed, lines_changed


def sanitize_dataset_yaml(
    yaml_path: str,
    *,
    log: Optional[LogFn] = None,
) -> Tuple[int, int]:
    """Clean all label folders referenced by *yaml_path*."""
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.isfile(yaml_path):
        return 0, 0
    total_files = 0
    total_lines = 0
    dirs = iter_label_dirs_from_data_yaml(yaml_path)
    if not dirs:
        if log:
            log('[yolo_label_clean] no label dirs resolved from %s' % yaml_path)
        return 0, 0
    for d in dirs:
        fc, lc = sanitize_label_dir(d, log=None)
        total_files += fc
        total_lines += lc
    if log and total_files:
        log(
            '[yolo_label_clean] stripped score column in %d file(s), %d line(s) under dataset yaml %s'
            % (total_files, total_lines, yaml_path)
        )
    return total_files, total_lines


def copy_label_file_train_ready(src: str, dst: str) -> None:
    """Copy a label file to *dst*, stripping detection score columns."""
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    try:
        with open(src, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return
    new_text, _ = clean_label_text(text)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(new_text)


def label_file_has_six_col_detection(path: str) -> bool:
    """True if file contains detection lines ``cls cx cy w h score``."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(strip_detection_score_parts(parts)) != len(parts):
                    return True
    except OSError:
        return False
    return False


class UltralyticsLabelCache:
    """Cache 5-column label views; originals on disk keep the score column."""

    def __init__(self) -> None:
        self._resolved: dict[str, str] = {}

    def resolve(self, lb_file: str) -> str:
        lb_file = os.path.abspath(lb_file)
        cached = self._resolved.get(lb_file)
        if cached:
            return cached
        if not label_file_has_six_col_detection(lb_file):
            self._resolved[lb_file] = lb_file
            return lb_file
        out = self._ensure_cache(lb_file)
        self._resolved[lb_file] = out
        return out

    @staticmethod
    def _cache_path_for(lb_file: str) -> Path:
        src = Path(lb_file)
        parts = src.parts
        for i, part in enumerate(parts):
            if part == 'labels':
                rel = Path(*parts[i + 1:])
                cache_base = Path(*parts[: i + 1]) / CACHE_DIRNAME
                return cache_base / rel
        return src.parent / CACHE_DIRNAME / src.name

    def _ensure_cache(self, lb_file: str) -> str:
        src = Path(lb_file)
        cache_path = self._cache_path_for(lb_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            src_mtime = src.stat().st_mtime
            if cache_path.is_file() and cache_path.stat().st_mtime >= src_mtime:
                return str(cache_path)
        except OSError:
            pass
        try:
            with open(src, 'r', encoding='utf-8') as f:
                text = f.read()
        except OSError:
            return lb_file
        new_text, _ = clean_label_text(text)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(new_text)
        return str(cache_path)


def _patch_verify_image_label(orig_verify, cache: UltralyticsLabelCache):
    def verify_image_label_compat(args):
        im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
        if lb_file and not keypoint:
            lb_file = cache.resolve(lb_file)
        return orig_verify((im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls))

    return verify_image_label_compat


@contextmanager
def ultralytics_six_column_label_compat(*, log: Optional[LogFn] = None) -> Iterator[UltralyticsLabelCache]:
    """Let Ultralytics train/val read 6-col detect labels without modifying originals."""
    import ultralytics.data.dataset as ds
    import ultralytics.data.utils as du

    cache = UltralyticsLabelCache()
    orig = du.verify_image_label
    patched = _patch_verify_image_label(orig, cache)
    du.verify_image_label = patched
    ds.verify_image_label = patched
    if log:
        log(
            '[yolo_label_clean] 6-column label compat on '
            '(Ultralytics uses cache under labels/%s; originals keep score)'
            % CACHE_DIRNAME
        )
    try:
        yield cache
    finally:
        du.verify_image_label = orig
        ds.verify_image_label = orig

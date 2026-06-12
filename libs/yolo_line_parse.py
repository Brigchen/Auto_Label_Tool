# -*- coding: utf-8 -*-
"""Parse YOLO label lines (detect / pose / seg) and optional trailing score."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# Manual labels upgraded 5→6 cols wrongly used 0.0; real autolabel scores are in (0, 1].
DEFAULT_MANUAL_SCORE = 1.0


def normalize_detection_score(score: Optional[float]) -> Optional[float]:
    """Treat score=0.0 as missing (manual placeholder), not a low-confidence detection."""
    if score is None:
        return None
    fv = float(score)
    if fv == 0.0:
        return None
    return fv


def extract_trailing_score(values: Sequence[str]) -> Tuple[List[str], Optional[float]]:
    """If the last token is a confidence in [0, 1], peel it off."""
    if len(values) < 2:
        return list(values), None
    try:
        sc = float(values[-1])
        if 0.0 <= sc <= 1.0:
            return list(values[:-1]), sc
    except (TypeError, ValueError):
        pass
    return list(values), None


def _is_normalized_bbox_coords(coords: Sequence[float]) -> bool:
    if len(coords) != 4:
        return False
    return all(-0.01 <= float(c) <= 1.01 for c in coords)


def is_pose_values(values: Sequence[str], posed_kpts_count: Optional[int] = None) -> bool:
    """True when values look like ``cls cx cy w h (x y v)...`` (no trailing score)."""
    if len(values) < 8:
        return False
    if (len(values) - 5) % 3 != 0:
        return False
    nk = (len(values) - 5) // 3
    if posed_kpts_count is not None and nk != int(posed_kpts_count):
        return False
    for i in range(7, len(values), 3):
        try:
            v = int(float(values[i]))
        except (TypeError, ValueError):
            return False
        if v not in (0, 1, 2):
            return False
    return True


def is_seg_values(values: Sequence[str]) -> bool:
    """True when values look like ``cls x1 y1 x2 y2 ...`` polygon."""
    if len(values) < 7:
        return False
    return (len(values) - 1) % 2 == 0


def parse_yolo_line_score(
    line: str,
    *,
    posed_kpts_count: Optional[int] = None,
) -> Tuple[Optional[float], str]:
    """Return ``(score_or_none, kind)`` where score is only set when confidently parsed.

    kind is one of: ``detect``, ``pose``, ``seg``, ``unknown``, ``invalid``.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None, "invalid"

    try:
        int(float(parts[0]))
        coords = [float(x) for x in parts[1:5]]
    except (TypeError, ValueError):
        return None, "invalid"

    if not _is_normalized_bbox_coords(coords):
        return None, "unknown"

    if len(parts) == 5:
        return None, "detect"

    if len(parts) == 6:
        try:
            sc = float(parts[5])
            if 0.0 <= sc <= 1.0:
                return normalize_detection_score(sc), "detect"
        except (TypeError, ValueError):
            pass
        return None, "detect"

    vals, score = extract_trailing_score(parts)
    if is_pose_values(vals, posed_kpts_count):
        return normalize_detection_score(score), "pose"
    if is_seg_values(vals):
        return normalize_detection_score(score), "seg"
    return None, "unknown"


def parse_yolo_bbox_line(
    line: str,
    *,
    posed_kpts_count: Optional[int] = None,
) -> Optional[Tuple[int, float, float, float, float, Optional[float]]]:
    """Parse bbox fields + optional score (pose/seg safe)."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except (TypeError, ValueError):
        return None
    score, _kind = parse_yolo_line_score(line, posed_kpts_count=posed_kpts_count)
    return cls, cx, cy, w, h, score

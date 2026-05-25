# -*- coding: utf-8 -*-
"""Resolve hub-style YOLO weight names to local *.pt paths before YOLO() (reuse cache, fewer downloads)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

# Matches ultralytics.engine.model default pretrained detect stub for recent releases.
DEFAULT_DETECT_PT = "yolo26n.pt"

# Requested hub basename (lowercase) -> try these filenames under each search root.
_DETECT_BUILTIN_ALIASES: dict[str, Tuple[str, ...]] = {
    # yolo11 series
    "yolo11n.pt": ("yolo11n.pt", "yolo26n.pt"),
    "yolo11s.pt": ("yolo11s.pt", "yolo26s.pt", "yolo11n.pt"),
    "yolo11m.pt": ("yolo11m.pt", "yolo26m.pt", "yolo11s.pt"),
    "yolo11l.pt": ("yolo11l.pt", "yolo26l.pt", "yolo11m.pt"),
    "yolo11x.pt": ("yolo11x.pt", "yolo26x.pt", "yolo11l.pt"),
    # yolo26 series
    "yolo26n.pt": ("yolo26n.pt", "yolo11n.pt"),
    "yolo26s.pt": ("yolo26s.pt", "yolo11s.pt", "yolo26n.pt"),
    "yolo26m.pt": ("yolo26m.pt", "yolo11m.pt", "yolo26s.pt"),
    "yolo26l.pt": ("yolo26l.pt", "yolo11l.pt", "yolo26m.pt"),
    "yolo26x.pt": ("yolo26x.pt", "yolo11x.pt", "yolo26l.pt"),
}

# Known task suffixes appended to model names (e.g. yolo26n-seg.pt, yolo11s-pose.pt)
_TASK_SUFFIXES = ("-seg", "-pose", "-obb", "-cls")


def _ultralytics_weights_dir() -> Optional[Path]:
    try:
        from ultralytics.utils import WEIGHTS_DIR

        p = Path(WEIGHTS_DIR).expanduser().resolve()
        return p if p.is_dir() else None
    except Exception:
        return None


def _names_to_try(basename: str) -> Sequence[str]:
    key = os.path.basename(basename).replace("\\", "/").lower()

    # 1. Exact match in alias table
    explicit = _DETECT_BUILTIN_ALIASES.get(key)
    if explicit is not None:
        return explicit

    # 2. Strip task suffix (e.g. yolo26s-pose.pt -> yolo26s.pt),
    #    look up base alias and re-apply suffix
    for suffix in _TASK_SUFFIXES:
        if suffix in key:
            # key = "yolo26s-pose.pt", suffix = "-pose"
            # Strip suffix from stem: "yolo26s-pose" -> strip "-pose" -> "yolo26s"
            stem = key.replace(".pt", "")  # "yolo26s-pose"
            base_stem = stem.replace(suffix, "")  # "yolo26s"
            base_key = base_stem + ".pt"
            base_aliases = _DETECT_BUILTIN_ALIASES.get(base_key)
            if base_aliases:
                return tuple(
                    a.replace(".pt", f"{suffix}.pt") if a.endswith(".pt") else a
                    for a in base_aliases
                )
            break

    # 3. Fallback: return original basename (Ultralytics hub will download)
    return (os.path.basename(basename),)


def resolve_yolo_checkpoint(weights: str, app_weights_dir: Optional[str] = None) -> str:
    """Return an absolute path to a local checkpoint if found, else the original string (hub id or yaml)."""
    if not weights:
        return weights
    s = str(weights).strip()
    if not s:
        return s
    if os.path.isfile(s):
        return str(Path(s).resolve())
    if os.path.isdir(s):
        return str(Path(s).resolve())
    if not s.lower().endswith((".pt", ".pth", ".onnx", ".engine", ".h5")):
        return s

    names = tuple(_names_to_try(s))
    roots: list[Path] = []
    if app_weights_dir:
        roots.append(Path(app_weights_dir).expanduser().resolve())
    try:
        from libs.repo_paths import repo_root

        rw = (Path(repo_root()) / "weights").resolve()
        roots.append(rw)
    except Exception:
        pass
    uw = _ultralytics_weights_dir()
    if uw is not None:
        roots.append(uw)

    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        for name in names:
            cand = root / name
            if cand.is_file():
                return str(cand.resolve())
    return s

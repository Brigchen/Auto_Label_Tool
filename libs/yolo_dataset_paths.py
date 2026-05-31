# -*- coding: utf-8 -*-
"""Resolve YOLO dataset image/label directories (shared by test report & ALT)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple


def ensure_classes_txt(label_dir: str, names: Dict[int, str]) -> str:
    """Write classes.txt under label_dir if missing."""
    os.makedirs(label_dir, exist_ok=True)
    path = os.path.join(label_dir, "classes.txt")
    if os.path.isfile(path):
        return path
    items = [names[i] for i in sorted(names)]
    if not items:
        items = ["0"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(items) + "\n")
    return path


def resolve_yolo_split_paths(
    dataset_root: str,
    split: str = "auto",
) -> Tuple[str, str, str, Dict[int, str]]:
    """Return (image_dir, label_dir, split_used, class_names)."""
    from libs.test_report import format_inspect_report, inspect_dataset_directory

    insp = inspect_dataset_directory(dataset_root)
    if not insp.valid:
        raise ValueError(format_inspect_report(insp))

    names = insp.names or {0: "0"}
    root = Path(insp.root)

    if insp.structure == "images_split":
        img_dir = Path(insp.selected_image_dir)
        split_used = img_dir.name
        lbl_dir = root / "labels" / split_used
    elif insp.structure == "flat_images":
        img_dir = Path(insp.root)
        lbl_dir = img_dir
        split_used = "custom"
    else:
        pick = split
        if pick == "auto":
            pick = next((s for s in ("test", "val", "train") if s in insp.splits_found), "")
            if not pick and insp.splits_found:
                pick = insp.splits_found[0]
        elif pick not in insp.splits_found and insp.splits_found:
            pick = insp.splits_found[0]
        if not pick:
            raise ValueError("数据集中无可用划分")
        split_used = pick
        if split_used == "all":
            img_dir = root / "images"
            lbl_dir = root / "labels"
        else:
            img_dir = root / "images" / split_used
            lbl_dir = root / "labels" / split_used

    if not img_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return str(img_dir.resolve()), str(lbl_dir.resolve()), split_used, names

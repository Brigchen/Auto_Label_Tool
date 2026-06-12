# -*- coding: utf-8 -*-
"""Collect image/label paths under a root, including nested subfolders."""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def flat_output_stem(relative_path: str) -> str:
    """Flat unique stem for nested files, e.g. ``a/b/c.jpg`` -> ``a_b_c``."""
    rel = relative_path.replace('\\', '/')
    base, _ext = os.path.splitext(os.path.basename(rel))
    sub = os.path.dirname(rel)
    if not sub or sub == '.':
        return base
    prefix = sub.replace('/', '_').replace('\\', '_')
    return '%s_%s' % (prefix, base)


def iter_files_with_ext(
    root: str,
    ext: str,
    *,
    skip_basenames: Sequence[str] = ('classes.txt',),
) -> Iterable[Tuple[str, str]]:
    """Yield ``(relative_path, absolute_path)`` for files with extension under *root*."""
    root = os.path.abspath(root)
    ext = ext.lower()
    skip = {s.lower() for s in skip_basenames}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower() in skip:
                continue
            if not name.lower().endswith(ext):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            yield rel, full


def iter_image_files(root: str) -> Iterable[Tuple[str, str]]:
    """Yield ``(relative_path, absolute_path)`` for image files under *root*."""
    root = os.path.abspath(root)
    for dirpath, _, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            yield rel, full


def build_basename_index(root: str, ext: str) -> Dict[str, List[str]]:
    """Map label stem -> list of absolute label paths (all subfolders)."""
    index: Dict[str, List[str]] = {}
    for rel, full in iter_files_with_ext(root, ext):
        stem = os.path.splitext(os.path.basename(rel))[0]
        index.setdefault(stem, []).append(full)
    return index


def resolve_label_for_image(
    label_root: str,
    image_rel: str,
    label_ext: str = '.txt',
    basename_index: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """Find a label file for an image relative path (mirror subpath, then basename)."""
    label_root = os.path.abspath(label_root)
    label_ext = label_ext.lower()
    mirror = os.path.join(label_root, os.path.splitext(image_rel)[0] + label_ext)
    if os.path.isfile(mirror):
        return mirror

    stem = os.path.splitext(os.path.basename(image_rel))[0]
    rel_dir = os.path.dirname(image_rel.replace('\\', '/'))
    index = basename_index if basename_index is not None else build_basename_index(label_root, label_ext)
    candidates = index.get(stem) or []
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    for path in candidates:
        lbl_rel = os.path.relpath(path, label_root).replace('\\', '/')
        if os.path.dirname(lbl_rel) == rel_dir:
            return path
    return candidates[0]


def collect_image_label_pairs(
    path_img: str,
    path_label: str,
    label_ext: str = '.txt',
) -> List[Tuple[str, str, str]]:
    """Return list of ``(image_rel, image_abs, label_abs)`` for all nested pairs."""
    path_img = os.path.abspath(path_img)
    path_label = os.path.abspath(path_label)
    index = build_basename_index(path_label, label_ext)
    pairs: List[Tuple[str, str, str]] = []
    for rel, img_path in iter_image_files(path_img):
        lbl = resolve_label_for_image(path_label, rel, label_ext, index)
        if lbl:
            pairs.append((rel, img_path, lbl))
    return pairs

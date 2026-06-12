# -*- coding: utf-8 -*-
"""Underwater photometric augmentation for fish detection (offline + shared core)."""
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LogFn = Callable[[str], None]

STRENGTH_LEVELS = ("light", "medium", "strong")

# Recommended Ultralytics color/geometry preset for underwater fish scenes.
UNDERWATER_YOLO_PRESET: Dict[str, float] = {
    "hsv_h": 0.012,
    "hsv_s": 0.65,
    "hsv_v": 0.50,
    "degrees": 3.0,
    "translate": 0.10,
    "scale": 0.50,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.70,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
}

_TRANSFORM_NAMES = (
    "attenuation",
    "haze",
    "turbidity",
    "color_cast",
    "vignette",
    "spotlight",
    "clahe",
    "gamma",
)


@dataclass
class UnderwaterAugmentConfig:
    strength: str = "medium"
    include_enhance: bool = True


def _strength_scale(strength: str) -> float:
    return {"light": 0.6, "medium": 1.0, "strong": 1.4}.get(strength, 1.0)


def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
    out = np.asarray(img)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    return out


def _attenuation(img: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    """Simulate wavelength-dependent light loss (R > G > B)."""
    out = img.astype(np.float32)
    r_f = rng.uniform(0.45, 1.0) ** scale
    g_f = rng.uniform(0.60, 1.0) ** scale
    b_f = rng.uniform(0.80, 1.0) ** scale
    out[:, :, 2] *= r_f
    out[:, :, 1] *= g_f
    out[:, :, 0] *= b_f
    return np.clip(out, 0, 255).astype(np.uint8)


def _backscatter_haze(img: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    """Underwater haze: I = J*t + A*(1-t)."""
    out = img.astype(np.float32)
    t = rng.uniform(0.35, 0.92) ** (1.0 / max(scale, 0.1))
    ambient = np.array([
        rng.uniform(0.55, 0.85),
        rng.uniform(0.70, 0.95),
        rng.uniform(0.85, 1.0),
    ], dtype=np.float32)
    out = out * t + ambient * (1.0 - t) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _turbidity(img: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    """Lower contrast, blur, sensor/particle noise."""
    out = img.astype(np.float32)
    alpha = rng.uniform(0.55, 0.95) ** scale
    beta = rng.uniform(-18.0, 18.0) * scale
    out = np.clip(out * alpha + beta, 0, 255)

    k = int(rng.choice([3, 5, 7]))
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.4, 1.6) * scale)

    sigma = rng.uniform(2.0, 10.0) * scale
    out = out + np.random.normal(0, sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _color_cast(img: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    """Shift white balance toward blue/green underwater cast."""
    out = img.astype(np.float32)
    gains = np.array([
        rng.uniform(0.85, 1.05),
        rng.uniform(0.90, 1.12),
        rng.uniform(0.75, 1.0),
    ], dtype=np.float32)
    bias = (scale - 1.0) * 8.0
    gains[0] *= 1.0 + 0.08 * scale
    gains[1] *= 1.0 + 0.04 * scale
    gains[2] *= 1.0 - 0.06 * scale
    out *= gains
    out += np.array([bias * 0.2, bias * 0.5, bias], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _vignette(img: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    cy = h * rng.uniform(0.45, 0.55)
    cx = w * rng.uniform(0.45, 0.55)
    yy, xx = np.indices((h, w), dtype=np.float32)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = np.sqrt(float(h * h + w * w)) * 0.5
    amount = rng.uniform(0.25, 0.65) * scale
    mask = 1.0 - amount * (dist / (max_dist + 1e-6)) ** 2
    mask = np.clip(mask, 0.35, 1.0)[..., None]
    out = img.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def _spotlight(img: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    """Uneven artificial lighting (diver lamp / cage light)."""
    h, w = img.shape[:2]
    cy = h * rng.uniform(0.15, 0.85)
    cx = w * rng.uniform(0.15, 0.85)
    yy, xx = np.indices((h, w), dtype=np.float32)
    sigma = rng.uniform(0.18, 0.45) * min(h, w) * scale
    spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma + 1e-6))
    gain = 1.0 + rng.uniform(0.15, 0.55) * scale * spot[..., None]
    out = img.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def _clahe_enhance(img: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    """Local contrast recovery (CPU; matches fish_enhance fallback)."""
    clip = rng.uniform(1.5, 3.5) * min(scale, 1.2)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def _gamma(img: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    gamma = rng.uniform(0.65, 1.45) ** (1.0 / max(scale, 0.1))
    inv = 1.0 / max(gamma, 1e-3)
    table = ((np.arange(256) / 255.0) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)


_NAMED_FN = {
    "attenuation": _attenuation,
    "haze": _backscatter_haze,
    "turbidity": _turbidity,
    "color_cast": _color_cast,
    "vignette": _vignette,
    "spotlight": _spotlight,
    "clahe": _clahe_enhance,
    "gamma": _gamma,
}

_DEGRADE = ("attenuation", "haze", "turbidity", "color_cast", "vignette", "spotlight", "gamma")
_ENHANCE = ("clahe",)


def apply_underwater_named(
    img: np.ndarray,
    name: str,
    *,
    strength: str = "medium",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply one named underwater transform."""
    fn = _NAMED_FN.get(name)
    if fn is None:
        raise ValueError(f"unknown underwater transform: {name}")
    rng = random.Random(seed)
    scale = _strength_scale(strength)
    return fn(_ensure_bgr_uint8(img), rng, scale)


def apply_underwater_random(
    img: np.ndarray,
    *,
    strength: str = "medium",
    include_enhance: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """Apply one random underwater transform; returns (image, transform_name)."""
    rng = random.Random(seed)
    scale = _strength_scale(strength)
    pool = list(_DEGRADE)
    if include_enhance:
        pool.extend(_ENHANCE)
    name = rng.choice(pool)
    out = _NAMED_FN[name](_ensure_bgr_uint8(img), rng, scale)
    return out, name


def offline_variant_names(copies: int) -> List[str]:
    """Cycle transform names for deterministic offline copies."""
    copies = max(0, int(copies))
    if copies == 0:
        return []
    names: List[str] = []
    cycle = list(_DEGRADE) + list(_ENHANCE)
    for i in range(copies):
        names.append(cycle[i % len(cycle)])
    return names


def write_offline_variants(
    img_path: str,
    label_path: str,
    dest_img_dir: str,
    dest_lbl_dir: str,
    stem: str,
    img_ext: str,
    *,
    copies: int = 1,
    strength: str = "medium",
    seed: int = 0,
    log: LogFn = print,
) -> int:
    """Write ``copies`` augmented image/label pairs next to the original stem."""
    copies = max(0, int(copies))
    if copies == 0:
        return 0

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        log(f"[uw_aug] skip unreadable image: {img_path}")
        return 0

    written = 0
    names = offline_variant_names(copies)
    for i, tname in enumerate(names):
        out_img = apply_underwater_named(
            img, tname, strength=strength, seed=seed + i * 9973 + hash(stem) % 10000,
        )
        suffix = f"__uw{i + 1}_{tname}"
        out_img_path = os.path.join(dest_img_dir, stem + suffix + img_ext)
        out_lbl_path = os.path.join(dest_lbl_dir, stem + suffix + ".txt")
        cv2.imwrite(out_img_path, out_img)
        shutil.copy2(label_path, out_lbl_path)
        written += 1
    return written


def augment_train_split_offline(
    dataset_root: str,
    *,
    copies: int = 2,
    strength: str = "medium",
    seed: int = 42,
    splits: Sequence[str] = ("train",),
    log: LogFn = print,
) -> int:
    """Augment existing YOLO split folders under ``images/{split}`` + ``labels/{split}``."""
    copies = max(0, int(copies))
    if copies == 0:
        return 0
    if strength not in STRENGTH_LEVELS:
        strength = "medium"

    root = os.path.abspath(dataset_root)
    total = 0
    for split in splits:
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if name.startswith("."):
                continue
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
                continue
            if "__uw" in stem:
                continue
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.isfile(lbl_path):
                continue
            img_path = os.path.join(img_dir, name)
            n = write_offline_variants(
                img_path, lbl_path, img_dir, lbl_dir, stem, ext,
                copies=copies, strength=strength, seed=seed, log=log,
            )
            total += n
    log(f"[uw_aug] offline: wrote {total} augmented pairs under {root} ({splits})")
    return total

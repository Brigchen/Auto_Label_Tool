# -*- coding: utf-8 -*-
"""Inject underwater photometric augment into Ultralytics v8 training pipeline."""
from __future__ import annotations

from typing import Any, Dict, Optional

from libs.underwater_augment import UnderwaterAugmentConfig, apply_underwater_random

_ORIGINAL_V8_TRANSFORMS = None
_PATCHED = False

_UW_CFG: Dict[str, Any] = {
    "enabled": False,
    "p": 0.5,
    "strength": "medium",
    "include_enhance": True,
}


def configure_underwater_train(
    enabled: bool,
    *,
    p: float = 0.5,
    strength: str = "medium",
    include_enhance: bool = True,
) -> None:
    """Enable/disable online underwater augment for the next training run."""
    _UW_CFG["enabled"] = bool(enabled)
    _UW_CFG["p"] = max(0.0, min(1.0, float(p)))
    _UW_CFG["strength"] = strength if strength in ("light", "medium", "strong") else "medium"
    _UW_CFG["include_enhance"] = bool(include_enhance)
    _ensure_patch()


def restore_underwater_train() -> None:
    """Disable online underwater augment after training."""
    _UW_CFG["enabled"] = False


def _ensure_patch() -> None:
    global _ORIGINAL_V8_TRANSFORMS, _PATCHED
    if _PATCHED:
        return
    from ultralytics.data import augment as ultra_aug

    _ORIGINAL_V8_TRANSFORMS = ultra_aug.v8_transforms

    def _patched_v8_transforms(dataset, imgsz, hyp, stretch=False):
        compose = _ORIGINAL_V8_TRANSFORMS(dataset, imgsz, hyp, stretch)
        if _UW_CFG.get("enabled"):
            compose.insert(
                -1,
                RandomUnderwater(
                    p=float(_UW_CFG.get("p", 0.5)),
                    strength=str(_UW_CFG.get("strength", "medium")),
                    include_enhance=bool(_UW_CFG.get("include_enhance", True)),
                ),
            )
        return compose

    ultra_aug.v8_transforms = _patched_v8_transforms
    _PATCHED = True


class RandomUnderwater:
    """Ultralytics-compatible photometric augment (bbox unchanged)."""

    def __init__(
        self,
        p: float = 0.5,
        strength: str = "medium",
        include_enhance: bool = True,
    ) -> None:
        self.p = max(0.0, min(1.0, float(p)))
        self.cfg = UnderwaterAugmentConfig(
            strength=strength if strength in ("light", "medium", "strong") else "medium",
            include_enhance=include_enhance,
        )

    def __call__(self, labels: dict) -> dict:
        import random

        if random.random() > self.p:
            return labels
        img = labels.get("img")
        if img is None or getattr(img, "size", 0) == 0:
            return labels
        if img.ndim != 3 or img.shape[2] != 3:
            return labels
        out, _name = apply_underwater_random(
            img,
            strength=self.cfg.strength,
            include_enhance=self.cfg.include_enhance,
        )
        labels["img"] = out
        return labels

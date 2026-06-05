# -*- coding: utf-8 -*-
"""Defer torch / ultralytics import until Auto Label or training needs them."""
from __future__ import annotations

_torch = None
_yolo_cls = None


def get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


def get_yolo_model(weights):
    global _yolo_cls
    if _yolo_cls is None:
        from ultralytics import YOLO as _Y
        _yolo_cls = _Y
    return _yolo_cls(weights)

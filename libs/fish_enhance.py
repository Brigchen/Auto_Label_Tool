# -*- coding: utf-8 -*-
"""Underwater-style enhancement (ported from legacy fish_utils, no track_libs)."""
import time

import cv2
import numpy as np
import torch


def img_resize(image, n_w, n_h):
    height, width = image.shape[0], image.shape[1]
    width_new = n_w
    height_new = n_h
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new


def _f(A=None):
    a = 1 / 1.055
    b = 0.055 / 1.055
    c = 1 / 12.92
    d = 0.04045
    lambda_ = 3.5
    A[torch.logical_and(A >= 0, A < d)] = A[torch.logical_and(A >= 0, A < d)] * c
    A[A >= d] = (A[A >= d] * a + b) ** lambda_
    return A


def _gc(A=None, method="srgb"):
    if method == "srgb":
        a = 1 / 1.055
        b = 0.055 / 1.055
        c = 1 / 12.92
        d = 0.04045
        lambda_ = 3.5
        A[A < 0] = -_f(-A[A < 0])
        A[torch.logical_and(A >= 0, A < d)] = A[torch.logical_and(A >= 0, A < d)] * c
        A[A >= d] = (A[A >= d] * a + b) ** lambda_
    elif method == "A-rgb":
        lambda_ = 1.5
        A = A**lambda_
    return A


def _two_percent_linear_2(img):
    q_bottom = 0.002
    q_top = 0.9970
    bottom = torch.quantile(img, q_bottom)
    top = torch.quantile(img, q_top)
    img = img.clone()
    img[img < bottom] = bottom
    img[img > top] = top
    max_img = torch.max(img)
    min_img = torch.min(img)
    return (img - min_img) / (max_img - min_img + 1e-8)


def gc_enhance_new(img_bgr):
    """
    GPU enhancement when CUDA is available; otherwise fast CLAHE fallback on CPU.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    if not torch.cuda.is_available():
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    o_h, o_w = img_bgr.shape[:2]
    img_r = img_resize(img_bgr, 1280, 960)
    t0 = time.time()
    device = torch.device("cuda")
    img = torch.from_numpy(img_r).to(device=device, dtype=torch.float32)[None, :] / 256.0
    method = "A-rgb"
    A = _gc(img, method)
    A = _two_percent_linear_2(A)
    A = A.clamp(0, 0.98)
    A = (A * 256).to(torch.uint8)
    A = A.squeeze(0).cpu().numpy()
    A = img_resize(A, o_w, o_h)
    _ = t0  # reserved for perf logging
    return A

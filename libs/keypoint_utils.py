# -*- coding: utf-8 -*-
"""Keypoint pose helpers: fixed slot count, v=0 => file 0,0,0, UI placeholder positions."""


def pad_keypoints_shape(shape, n_slots, template_names):
    """
    Ensure keypoints shape has exactly n_slots entries (aligned with template / YOLO line).
    Missing slots get visibility 0; coordinates filled by spread_keypoint_placeholders.
    """
    if n_slots <= 0:
        return
    try:
        from PyQt5.QtCore import QPointF
    except ImportError:
        from PyQt4.QtCore import QPointF

    while len(shape.points) < n_slots:
        shape.points.append(QPointF(0.0, 0.0))
    while len(shape.keypoint_visibility) < len(shape.points):
        shape.keypoint_visibility.append(0)
    if len(shape.points) > n_slots:
        shape.points = shape.points[:n_slots]
        shape.keypoint_visibility = shape.keypoint_visibility[:n_slots]
    names = list(template_names) if template_names else []
    if names:
        out = []
        for i in range(n_slots):
            if i < len(names):
                out.append(str(names[i]))
            else:
                out.append(str(i))
        shape.keypoint_names = out


def spread_keypoint_placeholders(shape, img_w, img_h):
    """
    For each slot with visibility==0, assign a distinct canvas position inside the
    bbox of visible points (or full image) so vertices remain individually selectable.
    File output still uses 0,0,0 for v==0 (see yolo_io.Keypoints2YoloLine).
    """
    if getattr(shape, 'shape_type', '') != 'keypoints' or not shape.points:
        return
    try:
        from PyQt5.QtCore import QPointF
    except ImportError:
        from PyQt4.QtCore import QPointF

    def _v(i):
        if i >= len(shape.keypoint_visibility):
            return 0
        try:
            return int(shape.keypoint_visibility[i])
        except (TypeError, ValueError):
            return 0

    labeled_xy = []
    for i, p in enumerate(shape.points):
        if _v(i) != 0:
            labeled_xy.append((float(p.x()), float(p.y())))
    pad = 10.0
    if labeled_xy:
        xs = [t[0] for t in labeled_xy]
        ys = [t[1] for t in labeled_xy]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
    else:
        xmin, ymin = 0.0, 0.0
        xmax, ymax = float(max(img_w - 1, 1)), float(max(img_h - 1, 1))

    w = max(xmax - xmin, 1.0)
    h = max(ymax - ymin, 1.0)
    zero_idx = [i for i in range(len(shape.points)) if _v(i) == 0]
    nz = len(zero_idx)
    if nz == 0:
        return
    for k, idx in enumerate(zero_idx):
        t = (k + 1.0) / float(nz + 1.0)
        x = xmin + pad + t * (w - 2.0 * pad)
        y = ymin + max(h - pad, pad * 0.5)
        x = max(0.0, min(float(img_w - 1), x))
        y = max(0.0, min(float(img_h - 1), y))
        shape.points[idx] = QPointF(x, y)

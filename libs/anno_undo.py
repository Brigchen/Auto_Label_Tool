# -*- coding: utf-8 -*-
"""Serialize / restore canvas annotations for undo and redo."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

try:
    from PyQt5.QtCore import Qt, QPointF
    from PyQt5.QtGui import QColor
except ImportError:
    from PyQt4.QtCore import Qt, QPointF
    from PyQt4.QtGui import QColor

from libs.shape import Shape


def snapshot_shapes_from_canvas(shapes: Sequence[Shape]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in shapes:
        out.append({
            'label': s.label,
            'line_color': s.line_color.getRgb(),
            'fill_color': s.fill_color.getRgb(),
            'points': [(float(p.x()), float(p.y())) for p in s.points],
            'shape_type': getattr(s, 'shape_type', 'bbox'),
            'keypoint_visibility': list(getattr(s, 'keypoint_visibility', [])),
            'keypoint_names': list(getattr(s, 'keypoint_names', [])),
            'skeleton': [tuple(x) for x in getattr(s, 'skeleton', [])],
            'difficult': bool(getattr(s, 'difficult', False)),
            'paintLabel': bool(getattr(s, 'paintLabel', False)),
            'visible': True,
        })
    return out


def snapshot_with_visibility(main_window) -> List[Dict[str, Any]]:
    snap = snapshot_shapes_from_canvas(main_window.canvas.shapes)
    for i, shape in enumerate(main_window.canvas.shapes):
        item = main_window.shapesToItems.get(shape)
        if item is not None and i < len(snap):
            snap[i]['visible'] = item.checkState() == Qt.Checked
    return snap


def _snap_to_load_tuples(snap: Sequence[Dict[str, Any]]):
    rows = []
    for d in snap:
        rows.append((
            d['label'],
            d['points'],
            d['line_color'],
            d['fill_color'],
            d.get('difficult', False),
            d.get('shape_type', 'bbox'),
            d.get('keypoint_visibility', []),
            d.get('keypoint_names', []),
            d.get('skeleton', []),
        ))
    return rows


def restore_annotation_snapshot(main_window, snap: Sequence[Dict[str, Any]]) -> None:
    main_window.labelList.clear()
    main_window.itemsToShapes.clear()
    main_window.shapesToItems.clear()
    main_window.loadLabels(_snap_to_load_tuples(snap))
    for i, shape in enumerate(main_window.canvas.shapes):
        if i >= len(snap):
            break
        vis = snap[i].get('visible', True)
        item = main_window.shapesToItems.get(shape)
        if item is None:
            continue
        item.setCheckState(Qt.Checked if vis else Qt.Unchecked)
        main_window.canvas.setShapeVisible(shape, vis)
    main_window.canvas.deSelectShape()
    main_window.updateComboBox()
    if main_window.noShapes():
        for act in main_window.actions.onShapesPresent:
            act.setEnabled(False)
    else:
        for act in main_window.actions.onShapesPresent:
            act.setEnabled(True)


def snapshots_equal(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> bool:
    return list(a) == list(b)

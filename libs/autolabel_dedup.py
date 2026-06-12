# -*- coding: utf-8 -*-
"""Auto-label box dedup: same-class and optional cross-class (keep highest score)."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, TypeVar

from libs.iou import compute_IOU

T = TypeVar('T')


def _dedup_indices(
    n: int,
    scores: Sequence,
    labels: Sequence,
    boxes: Sequence,
    iou_thr: float,
    *,
    same_class_only: bool,
) -> List[int]:
    if n <= 1:
        return list(range(n))
    order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
    keep: List[int] = []
    for idx in order:
        suppress = False
        for kept_idx in keep:
            if same_class_only and str(labels[idx]) != str(labels[kept_idx]):
                continue
            if compute_IOU(boxes[idx], boxes[kept_idx]) > float(iou_thr):
                suppress = True
                break
        if not suppress:
            keep.append(idx)
    return sorted(keep)


def _pick_by_indices(items: Sequence[T], indices: Sequence[int]) -> List[T]:
    return [items[i] for i in indices]


def dedup_same_class(
    boxes: Sequence,
    scores: Sequence,
    labels: Sequence,
    iou_thr: float,
) -> Tuple[List, List, List]:
    """Drop lower-score boxes when same-label IoU exceeds *iou_thr*."""
    idx = _dedup_indices(len(boxes), scores, labels, boxes, iou_thr, same_class_only=True)
    return _pick_by_indices(boxes, idx), _pick_by_indices(scores, idx), _pick_by_indices(labels, idx)


def dedup_cross_class(
    boxes: Sequence,
    scores: Sequence,
    labels: Sequence,
    iou_thr: float,
) -> Tuple[List, List, List]:
    """Drop lower-score boxes when any-label IoU exceeds *iou_thr*."""
    idx = _dedup_indices(len(boxes), scores, labels, boxes, iou_thr, same_class_only=False)
    return _pick_by_indices(boxes, idx), _pick_by_indices(scores, idx), _pick_by_indices(labels, idx)


def apply_autolabel_dedup(
    boxes: Sequence,
    scores: Sequence,
    labels: Sequence,
    dedup_iou: float,
    *,
    cross_class: bool = False,
) -> Tuple[List, List, List]:
    """Same-class dedup, then optional cross-class dedup (same *dedup_iou* threshold)."""
    boxes, scores, labels = dedup_same_class(boxes, scores, labels, dedup_iou)
    if cross_class:
        boxes, scores, labels = dedup_cross_class(boxes, scores, labels, dedup_iou)
    return boxes, scores, labels


def dedup_pose(
    boxes: Sequence,
    scores: Sequence,
    labels: Sequence,
    kpts_vis_list: Sequence,
    dedup_iou: float,
    *,
    cross_class: bool = False,
) -> Tuple[List, List, List, List]:
    """Pose rows: same dedup rules, keypoints follow surviving boxes."""
    idx = _dedup_indices(len(boxes), scores, labels, boxes, dedup_iou, same_class_only=True)
    if cross_class and len(idx) > 1:
        sub_b = _pick_by_indices(boxes, idx)
        sub_s = _pick_by_indices(scores, idx)
        sub_l = _pick_by_indices(labels, idx)
        sub_k = _pick_by_indices(kpts_vis_list, idx)
        idx2 = _dedup_indices(len(sub_b), sub_s, sub_l, sub_b, dedup_iou, same_class_only=False)
        return (
            _pick_by_indices(sub_b, idx2),
            _pick_by_indices(sub_s, idx2),
            _pick_by_indices(sub_l, idx2),
            _pick_by_indices(sub_k, idx2),
        )
    return (
        _pick_by_indices(boxes, idx),
        _pick_by_indices(scores, idx),
        _pick_by_indices(labels, idx),
        _pick_by_indices(kpts_vis_list, idx),
    )


# Back-compat aliases used by autolabel_batch
dedup_boxes = dedup_same_class

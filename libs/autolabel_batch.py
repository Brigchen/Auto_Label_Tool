# -*- coding: utf-8 -*-
"""Batch YOLO auto-labeling for a folder of images (detect / segment / pose)."""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO

from libs.autolabel_dedup import apply_autolabel_dedup, dedup_pose
from libs.yolo_io import YOLOWriter
from libs.yolo_weights import resolve_yolo_checkpoint

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def list_images(folder: str) -> List[str]:
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.lower().endswith(IMG_EXTS):
            out.append(p)
    return out


def _name_at(names, idx: int) -> str:
    if isinstance(names, dict):
        return str(names[int(idx)])
    return str(names[int(idx)])


def _ordered_class_names(names) -> List[str]:
    if isinstance(names, dict):
        keys = sorted(int(k) for k in names.keys())
        return [str(names[k]) for k in keys]
    return [str(n) for n in names]


def _canonical_label(label: str, class_list: List[str]) -> str:
    if not class_list:
        return label
    if label in class_list:
        return label
    for c in class_list:
        if str(c).lower() == str(label).lower():
            return c
    return label

def _task_from_result(result) -> str:
    kp = getattr(result, 'keypoints', None)
    if kp is not None and getattr(kp, 'data', None) is not None and len(kp.data):
        return 'pose'
    masks = getattr(result, 'masks', None)
    if masks is not None and getattr(masks, 'xy', None) is not None and len(masks.xy):
        return 'segment'
    return 'detect'


def _extract_pose_rows(result, names):
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return [], [], [], []
    kp = getattr(result, 'keypoints', None)
    if kp is None:
        return [], [], [], []

    cls_idx = boxes.cls.cpu().numpy().astype(int)
    confs = [float(x) for x in boxes.conf.cpu().numpy()]
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    labels = [_name_at(names, c) for c in cls_idx]

    import numpy as np
    if hasattr(kp, 'xy') and kp.xy is not None and len(kp.xy):
        xy = kp.xy.cpu().numpy()
        kconf = kp.conf.cpu().numpy() if getattr(kp, 'conf', None) is not None else None
    elif hasattr(kp, 'xyn') and kp.xyn is not None and len(kp.xyn):
        xyn = kp.xyn.cpu().numpy()
        kconf = kp.conf.cpu().numpy() if getattr(kp, 'conf', None) is not None else None
        h, w = result.orig_shape[0], result.orig_shape[1]
        xy = np.zeros_like(xyn)
        xy[:, :, 0] = xyn[:, :, 0] * w
        xy[:, :, 1] = xyn[:, :, 1] * h
    else:
        return [], [], [], []

    kpts_vis_list = []
    for i in range(len(xyxy)):
        pts, vis = [], []
        for j in range(xy.shape[1]):
            pts.append((int(float(xy[i, j, 0])), int(float(xy[i, j, 1]))))
            if kconf is not None:
                c = float(kconf[i, j])
                v = 2 if c > 0.5 else (1 if c > 0.25 else 0)
            else:
                v = 2
            vis.append(v)
        kpts_vis_list.append((pts, vis))
    return xyxy, confs, labels, kpts_vis_list


def _write_yolo_for_image(
    result,
    names,
    image_path: str,
    label_path: str,
    class_list: List[str],
    dedup_iou: float,
    task_hint: str = 'auto',
    save_scores: bool = True,
    cross_class_dedup: bool = True,
) -> Tuple[int, str]:
    """Write one label file. Returns (object_count, task_used)."""
    img = cv2.imread(image_path)
    if img is None:
        return 0, 'skip'
    h, w = img.shape[:2]
    image_shape = [h, w, img.shape[2] if len(img.shape) > 2 else 3]
    writer = YOLOWriter(
        os.path.dirname(image_path),
        os.path.basename(image_path),
        image_shape,
        localImgPath=image_path,
    )

    task = task_hint if task_hint in ('detect', 'segment', 'pose') else _task_from_result(result)
    count = 0

    def _opt_score(sc):
        return sc if save_scores else None

    if task == 'pose':
        xyxy, scores, labels, kpts_vis = _extract_pose_rows(result, names)
        if xyxy:
            xyxy, scores, labels, kpts_vis = dedup_pose(
                xyxy, scores, labels, kpts_vis, dedup_iou, cross_class=cross_class_dedup,
            )
            for box, score, label, (pts, vis) in zip(xyxy, scores, labels, kpts_vis):
                canon = _canonical_label(label, class_list)
                writer.addKeypoints(pts, vis, canon, difficult=False, score=_opt_score(score))
                count += 1
    elif task == 'segment':
        masks = result.masks
        boxes = result.boxes
        if masks is None or boxes is None or len(boxes) == 0:
            return 0, task
        polys = masks.xy
        for i in range(len(boxes)):
            if i >= len(polys):
                break
            xy = polys[i]
            if hasattr(xy, 'cpu'):
                xy = xy.cpu().numpy()
            pts = [(float(x), float(y)) for x, y in xy]
            if len(pts) < 3:
                continue
            label = _name_at(names, int(boxes.cls[i]))
            score = float(boxes.conf[i])
            canon = _canonical_label(label, class_list)
            writer.addPolygon(pts, canon, difficult=False, score=_opt_score(score))
            count += 1
    else:
        dets = result.boxes
        if dets is None or len(dets) == 0:
            return 0, task
        labels = [_name_at(names, int(x)) for x in dets.cls]
        scores = [float(x) for x in dets.conf]
        boxes = [x.tolist() for x in dets.xyxy]
        boxes, scores, labels = apply_autolabel_dedup(
            boxes, scores, labels, dedup_iou, cross_class=cross_class_dedup,
        )
        for box, score, label in zip(boxes, scores, labels):
            canon = _canonical_label(label, class_list)
            writer.addBndBox(int(box[0]), int(box[1]), int(box[2]), int(box[3]), canon, difficult=False, score=_opt_score(score))
            count += 1

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    writer.save(targetFile=label_path, classList=class_list)
    return count, task


def run_autolabel_batch(
    image_dir: str,
    label_dir: str,
    weights: str,
    conf: float,
    pred_iou: float,
    dedup_iou: float,
    app_weights_dir: Optional[str] = None,
    class_list: Optional[Sequence[str]] = None,
    task_hint: str = 'auto',
    save_scores: bool = True,
    cross_class_dedup: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[int, int, int]:
    """
    Returns (images_processed, images_with_labels, total_objects).
    """
    images = list_images(image_dir)
    if not images:
        return 0, 0, 0

    w = resolve_yolo_checkpoint(weights, app_weights_dir=app_weights_dir)
    model = YOLO(w)
    names = model.names
    if not class_list:
        class_list = _ordered_class_names(names)
    else:
        class_list = list(class_list)

    os.makedirs(label_dir, exist_ok=True)
    classes_path = os.path.join(label_dir, 'classes.txt')
    with open(classes_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_list) + '\n')

    n_img = len(images)
    n_labeled = 0
    n_obj = 0
    for i, img_path in enumerate(images):
        if cancel_cb and cancel_cb():
            break
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        if progress_cb:
            progress_cb(i, n_img, os.path.basename(img_path))
        result = model.predict(source=img_path, conf=conf, iou=pred_iou, verbose=False)[0]
        cnt, _ = _write_yolo_for_image(
            result, names, img_path, label_path, class_list, dedup_iou,
            task_hint=task_hint, save_scores=save_scores,
            cross_class_dedup=cross_class_dedup,
        )
        if cnt > 0:
            n_labeled += 1
            n_obj += cnt

    if progress_cb:
        progress_cb(n_img, n_img, 'done')
    return len(images), n_labeled, n_obj

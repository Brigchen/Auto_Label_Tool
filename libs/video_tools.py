# -*- coding: utf-8 -*-
"""Extract frames from video files; annotate videos to YOLO images + labels."""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO

from libs.autolabel_batch import _write_yolo_for_image, _ordered_class_names
from libs.yolo_weights import resolve_yolo_checkpoint

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.wmv', '.m4v', '.webm')


def _imwrite(path: str, frame) -> bool:
    try:
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return cv2.imwrite(path, frame)


def _video_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def extract_frames_from_video(
    video_path: str,
    images_dir: str,
    frame_gap: int,
    name_prefix: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> int:
    """Save every frame_gap-th frame. Returns number of images written."""
    frame_gap = max(1, int(frame_gap))
    prefix = name_prefix or _video_stem(video_path)
    os.makedirs(images_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    saved = 0
    index = 0
    while cap.isOpened():
        if cancel_cb and cancel_cb():
            break
        ret, frame = cap.read()
        if not ret:
            break
        index += 1
        if progress_cb:
            progress_cb(index, frame_total, os.path.basename(video_path))
        if index % frame_gap != 0:
            continue
        out_name = '%s_%06d.jpg' % (prefix, index)
        if _imwrite(os.path.join(images_dir, out_name), frame):
            saved += 1
    cap.release()
    return saved


def extract_videos_to_images(
    video_paths: Sequence[str],
    output_root: str,
    frame_gap: int,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[int, int]:
    """Extract frames from one or more videos into output_root/images/. Returns (videos_ok, frames_saved)."""
    images_dir = os.path.join(output_root, 'images')
    os.makedirs(images_dir, exist_ok=True)
    videos_ok = 0
    frames_saved = 0
    paths = [p for p in video_paths if p and os.path.isfile(p)]
    n = len(paths)
    for vi, vp in enumerate(paths):
        if cancel_cb and cancel_cb():
            break

        def _cb(cur, tot, _name):
            if progress_cb:
                base = int(100 * vi / max(n, 1))
                span = int(100 / max(n, 1))
                progress_cb(base + int(span * cur / max(tot, 1)), 100, _name)

        cnt = extract_frames_from_video(
            vp, images_dir, frame_gap, name_prefix=_video_stem(vp),
            progress_cb=_cb, cancel_cb=cancel_cb)
        if cnt > 0:
            videos_ok += 1
            frames_saved += cnt
    if progress_cb:
        progress_cb(100, 100, 'done')
    return videos_ok, frames_saved


def annotate_videos_to_yolo(
    video_paths: Sequence[str],
    output_root: str,
    weights: str,
    frame_gap: int,
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
    Extract frames to images/ and write YOLO labels/ per frame.
    Returns (frames_saved, frames_labeled, total_objects).
    """
    images_dir = os.path.join(output_root, 'images')
    labels_dir = os.path.join(output_root, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    w = resolve_yolo_checkpoint(weights, app_weights_dir=app_weights_dir)
    model = YOLO(w)
    names = model.names
    if not class_list:
        class_list = _ordered_class_names(names)
    else:
        class_list = list(class_list)

    classes_path = os.path.join(labels_dir, 'classes.txt')
    with open(classes_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_list) + '\n')

    frame_gap = max(1, int(frame_gap))
    paths = [p for p in video_paths if p and os.path.isfile(p)]
    frames_saved = 0
    frames_labeled = 0
    total_objects = 0
    step = 0
    # rough step budget: sum of estimated frames per video
    budget = 0
    for vp in paths:
        cap = cv2.VideoCapture(vp)
        if cap.isOpened():
            budget += max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) // frame_gap)
            cap.release()
    budget = max(budget, 1)

    for vp in paths:
        if cancel_cb and cancel_cb():
            break
        prefix = _video_stem(vp)
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        index = 0
        while cap.isOpened():
            if cancel_cb and cancel_cb():
                break
            ret, frame = cap.read()
            if not ret:
                break
            index += 1
            if index % frame_gap != 0:
                continue
            step += 1
            if progress_cb:
                progress_cb(step, budget, '%s #%d' % (prefix, index))
            img_name = '%s_%06d.jpg' % (prefix, index)
            img_path = os.path.join(images_dir, img_name)
            if not _imwrite(img_path, frame):
                continue
            frames_saved += 1
            result = model.predict(source=frame, conf=conf, iou=pred_iou, verbose=False)[0]
            label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
            cnt, _ = _write_yolo_for_image(
                result, names, img_path, label_path, class_list, dedup_iou,
                task_hint=task_hint, save_scores=save_scores,
                cross_class_dedup=cross_class_dedup,
            )
            if cnt > 0:
                frames_labeled += 1
                total_objects += cnt
        cap.release()

    if progress_cb:
        progress_cb(budget, budget, 'done')
    return frames_saved, frames_labeled, total_objects

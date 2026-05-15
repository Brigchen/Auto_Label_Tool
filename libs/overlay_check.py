# -*- coding: utf-8 -*-
"""YOLO txt overlay checks and class remaps (from legacy box_overlay_check_yolo)."""


def xywh2xyxy(box):
    x, y, w, h = [float(x) for x in box]
    x2 = x + w
    y2 = y + h
    return [x, y, x2, y2]


def compute_iou_yolo_line(box1, box2):
    """IoU for two YOLO lines (cls xywhn...), uses box coords after class index."""
    b1 = box1[1:]
    b2 = box2[1:]
    rec1 = xywh2xyxy(b1)
    rec2 = xywh2xyxy(b2)
    left_column_max = max(float(rec1[0]), float(rec2[0]))
    right_column_min = min(float(rec1[2]), float(rec2[2]))
    up_row_max = max(float(rec1[1]), float(rec2[1]))
    down_row_min = min(float(rec1[3]), float(rec2[3]))
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0.0
    s1 = (float(rec1[2]) - float(rec1[0])) * (float(rec1[3]) - float(rec1[1]))
    s2 = (float(rec2[2]) - float(rec2[0])) * (float(rec2[3]) - float(rec2[1]))
    s_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
    return s_cross / (s1 + s2 - s_cross + 1e-8)


def remove_overlay(yolo_path, iou_threshold=0.7):
    """Remove highly overlapping boxes in one YOLO txt (in-place)."""
    with open(yolo_path, "r", encoding="utf-8") as box_file:
        boxes = [ln.rstrip().split() for ln in box_file if ln.strip()]
    if not boxes:
        return
    kept = []
    for i, box in enumerate(boxes):
        overlay = False
        for other in boxes[i + 1 :]:
            if compute_iou_yolo_line(box, other) > iou_threshold:
                overlay = True
                break
        if not overlay:
            kept.append(box)
    with open(yolo_path, "w", encoding="utf-8") as out_file:
        for box in kept:
            out_file.write(" ".join(box) + "\n")


def clamp_class_ids(yolo_path, max_class_exclusive):
    """Clamp any class id >= max to 0 (in-place)."""
    changed = False
    with open(yolo_path, "r", encoding="utf-8") as box_file:
        lines = box_file.readlines()
    out_lines = []
    for boxline in lines:
        if not boxline.strip():
            continue
        box = boxline.rstrip().split()
        if int(box[0]) >= max_class_exclusive:
            changed = True
            box[0] = "0"
        out_lines.append(" ".join(box) + "\n")
    if changed:
        with open(yolo_path, "w", encoding="utf-8") as out_file:
            out_file.writelines(out_lines)


def merge_class_to_binary(yolo_path, other_class_id=88, positive_class_id=1):
    """Map class other_class_id -> positive_class_id, all others -> 0."""
    with open(yolo_path, "r", encoding="utf-8") as box_file:
        boxes = [ln.rstrip().split() for ln in box_file if ln.strip()]
    with open(yolo_path, "w", encoding="utf-8") as out_file:
        for box in boxes:
            if int(box[0]) == other_class_id:
                box[0] = str(positive_class_id)
            else:
                box[0] = "0"
            out_file.write(" ".join(box) + "\n")


def process_yolo_tree(root, do_overlay=True, do_clamp=True, max_class_exclusive=1, do_merge=False, merge_other=88):
    """Walk root for *.txt labels (except classes.txt) and apply checks."""
    import os

    for path, _dirs, files in os.walk(root):
        for fl in files:
            if not fl.endswith(".txt") or fl == "classes.txt":
                continue
            p = os.path.join(path, fl)
            try:
                if do_overlay:
                    remove_overlay(p)
                if do_clamp:
                    clamp_class_ids(p, max_class_exclusive)
                if do_merge:
                    merge_class_to_binary(p, merge_other)
            except OSError:
                continue

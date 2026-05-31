# -*- coding: utf-8 -*-
"""Post-training test-set evaluation: HTML report + worst-case visual exports."""
from __future__ import annotations

import html
import json
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from libs.iou import compute_IOU
from libs.pkg_paths import resolve_cjk_plot_font
from libs.yolo_dataset_paths import ensure_classes_txt

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Box:
    cls: int
    name: str
    xyxy: Tuple[float, float, float, float]
    conf: float = 1.0


@dataclass
class ImageEval:
    image_path: str
    label_path: str
    score: float  # 0=best, 1=worst (1 - F1)
    f1: float
    tp: int
    fp: int
    fn: int
    class_errors: int
    gt_count: int
    pred_count: int
    gt_classes: List[str] = field(default_factory=list)
    pred_classes: List[str] = field(default_factory=list)
    gt_boxes: List[Box] = field(default_factory=list)
    pred_boxes: List[Box] = field(default_factory=list)
    export_name: str = ""
    detail: str = ""


def _log(msg: str, log: LogFn = print) -> None:
    ts = time.strftime("%H:%M:%S")
    log(f"[{ts}] {msg}")


def load_data_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = str(data.get("path", "") or "").strip()
    if root:
        root = str(Path(root).expanduser().resolve())
    names = data.get("names") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {
        "path": root,
        "train": data.get("train", ""),
        "val": data.get("val", ""),
        "test": data.get("test", ""),
        "names": {int(k): str(v) for k, v in names.items()},
    }


def _resolve_split_dir(root: str, split: str) -> Path:
    p = Path(split)
    if p.is_absolute():
        return p
    if root:
        return Path(root) / split
    return p


def _count_images_in_dir(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    n = 0
    for dirpath, _, files in os.walk(directory):
        for fn in files:
            if Path(fn).suffix.lower() in _IMG_EXTS:
                n += 1
    return n


def _find_data_yaml_near(root: Path) -> Optional[Path]:
    for cand in (
        root / "data.yaml",
        root / "dataset.yaml",
        root.parent / "data.yaml",
    ):
        if cand.is_file():
            return cand
    for fn in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
        if fn.is_file():
            return fn
    return None


@dataclass
class DatasetInspectResult:
    valid: bool
    root: str
    structure: str  # yolo_root | images_split | flat_images | invalid
    splits_found: List[str]
    image_count: int
    label_count: int
    names: Dict[int, str]
    data_yaml_path: str
    issues: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    selected_image_dir: str = ""


def inspect_dataset_directory(path: str) -> DatasetInspectResult:
    """Check YOLO-style dataset layout; return issues and fix hints."""
    root = Path(path).expanduser().resolve()
    issues: List[str] = []
    hints: List[str] = []
    names: Dict[int, str] = {}
    data_yaml_path = ""
    yaml_file = _find_data_yaml_near(root)
    if yaml_file is not None:
        try:
            meta = load_data_yaml(str(yaml_file))
            names = meta.get("names") or {}
            data_yaml_path = str(yaml_file)
        except Exception as exc:
            issues.append(f"读取 {yaml_file.name} 失败: {exc}")

    if not root.is_dir():
        return DatasetInspectResult(
            valid=False,
            root=str(root),
            structure="invalid",
            splits_found=[],
            image_count=0,
            label_count=0,
            names=names,
            data_yaml_path=data_yaml_path,
            issues=[f"目录不存在: {root}"],
            hints=["请选择有效的数据集根目录或 images/<split> 目录"],
        )

    # Selected: .../images/{train|val|test}
    if root.name in ("train", "val", "test") and root.parent.name == "images":
        split = root.name
        dataset_root = root.parent.parent
        labels_dir = dataset_root / "labels" / split
        img_count = _count_images_in_dir(root)
        lbl_count = sum(1 for _ in labels_dir.glob("*.txt")) if labels_dir.is_dir() else 0
        if img_count == 0:
            issues.append(f"images/{split}/ 下无图像文件")
        if not labels_dir.is_dir():
            issues.append(f"缺少 labels/{split}/ 目录")
            hints.append(f"请创建: {labels_dir}")
        elif lbl_count == 0:
            issues.append(f"labels/{split}/ 下无 .txt 标注")
            hints.append("标注文件名需与图像 stem 一致，如 img001.jpg → img001.txt")
        if not names and yaml_file is None:
            hints.append("建议在数据集根目录放置 data.yaml 并配置 names 类别名")
        return DatasetInspectResult(
            valid=img_count > 0 and not any("缺少 labels" in x for x in issues),
            root=str(dataset_root),
            structure="images_split",
            splits_found=[split],
            image_count=img_count,
            label_count=lbl_count,
            names=names,
            data_yaml_path=data_yaml_path,
            issues=issues,
            hints=hints,
            selected_image_dir=str(root),
        )

    # YOLO root: images/ + labels/
    images_root = root / "images"
    labels_root = root / "labels"
    if images_root.is_dir():
        splits_found: List[str] = []
        total_img = total_lbl = 0
        for sp in ("train", "val", "test"):
            sp_img = images_root / sp
            if sp_img.is_dir() and _count_images_in_dir(sp_img) > 0:
                splits_found.append(sp)
                total_img += _count_images_in_dir(sp_img)
                sp_lbl = labels_root / sp
                if sp_lbl.is_dir():
                    total_lbl += sum(1 for _ in sp_lbl.glob("*.txt"))
        if not splits_found:
            flat_img = _count_images_in_dir(images_root)
            if flat_img > 0:
                splits_found = ["all"]
                total_img = flat_img
                if labels_root.is_dir():
                    total_lbl = sum(1 for _ in labels_root.glob("*.txt"))
        if not labels_root.is_dir():
            issues.append("缺少 labels/ 目录")
            hints.append(f"请创建: {labels_root}")
            hints.append("标准结构: images/<split>/ 与 labels/<split>/ 一一对应")
        elif total_img > 0 and total_lbl == 0:
            issues.append("未找到任何 .txt 标注文件")
            hints.append("labels/<split>/ 下放置与图像同名的 YOLO 标注")
        if not splits_found:
            issues.append("images/ 下无图像（需 train/val/test 子目录或 images/ 内直接放图）")
            hints.append("示例:\n  dataset/images/train/\n  dataset/labels/train/")
        if not names:
            hints.append("建议在根目录添加 data.yaml，包含 path、names、nc")
        valid = bool(splits_found) and total_img > 0 and "缺少 labels/" not in "".join(issues)
        return DatasetInspectResult(
            valid=valid,
            root=str(root),
            structure="yolo_root",
            splits_found=splits_found,
            image_count=total_img,
            label_count=total_lbl,
            names=names,
            data_yaml_path=data_yaml_path,
            issues=issues,
            hints=hints,
            selected_image_dir=str(images_root),
        )

    # Flat folder: image files directly in selected dir
    flat_imgs = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    if flat_imgs:
        lbl_same = sum(1 for p in flat_imgs if (root / f"{p.stem}.txt").is_file())
        issues.append("当前为扁平目录，非标准 YOLO 结构")
        hints.append("推荐结构:\n  <dataset>/images/test/\n  <dataset>/labels/test/")
        hints.append("或选择数据集根目录（含 images/ 与 labels/）")
        if lbl_same == 0:
            issues.append("同目录下未找到与图像同名的 .txt 标注")
        return DatasetInspectResult(
            valid=len(flat_imgs) > 0,
            root=str(root),
            structure="flat_images",
            splits_found=["all"],
            image_count=len(flat_imgs),
            label_count=lbl_same,
            names=names,
            data_yaml_path=data_yaml_path,
            issues=issues,
            hints=hints,
            selected_image_dir=str(root),
        )

    issues.append("无法识别为 YOLO 数据集目录")
    hints.extend(
        [
            "请选择以下之一:",
            "  1) 数据集根目录（含 images/、labels/）",
            "  2) 划分目录 images/train|val|test",
            "  3) 含图像与同名 .txt 的文件夹",
        ]
    )
    return DatasetInspectResult(
        valid=False,
        root=str(root),
        structure="invalid",
        splits_found=[],
        image_count=0,
        label_count=0,
        names=names,
        data_yaml_path=data_yaml_path,
        issues=issues,
        hints=hints,
    )


def format_inspect_report(result: DatasetInspectResult) -> str:
    lines = [f"目录: {result.root}", f"结构: {result.structure}"]
    if result.splits_found:
        lines.append(f"可用划分: {', '.join(result.splits_found)}")
    lines.append(f"图像: {result.image_count} | 标注: {result.label_count}")
    if result.data_yaml_path:
        lines.append(f"data yaml: {result.data_yaml_path}")
    if result.issues:
        lines.append("\n问题:")
        lines.extend(f"  · {x}" for x in result.issues)
    if result.hints:
        lines.append("\n建议:")
        lines.extend(f"  · {x}" for x in result.hints)
    return "\n".join(lines)


def _collect_pairs_from_image_dir(img_dir: Path, labels_dir: Optional[Path] = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for dirpath, _, files in os.walk(img_dir):
        for fn in sorted(files):
            if Path(fn).suffix.lower() not in _IMG_EXTS:
                continue
            ip = Path(dirpath) / fn
            lp = _label_path_for_image(ip, labels_dir)
            pairs.append((str(ip.resolve()), str(lp.resolve()) if lp.is_file() else ""))
    return pairs


def list_pairs_from_dataset_dir(dataset_dir: str, split: str) -> Tuple[List[Tuple[str, str]], str, Dict[int, str], str]:
    """List image/label pairs from an inspected custom directory."""
    insp = inspect_dataset_directory(dataset_dir)
    if not insp.valid:
        msg = format_inspect_report(insp)
        raise ValueError(msg)

    root = Path(insp.root)
    names = insp.names or {0: "0"}
    yaml_path = insp.data_yaml_path

    if insp.structure == "images_split":
        img_dir = Path(insp.selected_image_dir)
        split_used = img_dir.name
        labels_dir = root / "labels" / split_used
        pairs = _collect_pairs_from_image_dir(img_dir, labels_dir if labels_dir.is_dir() else None)
    elif insp.structure == "flat_images":
        split_used = "custom"
        pairs = _collect_pairs_from_image_dir(Path(insp.selected_image_dir or insp.root), None)
    else:
        if split in insp.splits_found:
            split_used = split
        elif insp.splits_found:
            split_used = insp.splits_found[0]
        else:
            raise FileNotFoundError("数据集中无可用划分")
        if split_used == "all":
            img_dir = root / "images"
            labels_dir = root / "labels" if (root / "labels").is_dir() else None
        else:
            img_dir = root / "images" / split_used
            labels_dir = root / "labels" / split_used
        if not img_dir.is_dir():
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        pairs = _collect_pairs_from_image_dir(img_dir, labels_dir if labels_dir and labels_dir.is_dir() else None)

    if not pairs:
        raise FileNotFoundError(f"目录内无图像: {dataset_dir}")
    return pairs, split_used, names, yaml_path


def _label_path_for_image(img_path: Path, labels_root: Optional[Path] = None) -> Path:
    if labels_root is not None:
        rel = img_path.name
        cand = labels_root / (img_path.stem + ".txt")
        if cand.is_file():
            return cand
    # images/... -> labels/...
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def list_split_images(
    data_yaml: str,
    split: str = "test",
    *,
    strict: bool = False,
) -> Tuple[List[Tuple[str, str]], str]:
    """Return [(image_path, label_path), ...] and split name actually used."""
    meta = load_data_yaml(data_yaml)
    root = meta["path"]
    if meta.get(split):
        key = split
    elif strict:
        raise FileNotFoundError(f"data yaml 未配置 split={split}，请在 yaml 中添加 {split}: images/{split}")
    elif split == "test" and meta.get("val"):
        key = "val"
    elif meta.get("val"):
        key = "val"
    elif meta.get("train"):
        key = "train"
    else:
        raise FileNotFoundError(f"data yaml 中未找到 {split}/val/train 路径")

    split_rel = meta.get(key)
    if not split_rel:
        raise FileNotFoundError(f"data yaml 中未找到 {key} 路径")

    img_dir = _resolve_split_dir(root, str(split_rel))
    if not img_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")

    labels_dir = None
    img_str = str(img_dir).replace("\\", "/")
    if "/images/" in img_str:
        labels_dir = Path(img_str.replace("/images/", "/labels/"))

    pairs = _collect_pairs_from_image_dir(img_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError(f"目录内无图像: {img_dir}")
    return pairs, key


def resolve_report_pairs(
    *,
    source_mode: str = "yaml",
    data_yaml: str = "",
    dataset_dir: str = "",
    split: str = "test",
) -> Tuple[List[Tuple[str, str]], str, Dict[int, str], str]:
    if source_mode == "dir":
        if not dataset_dir:
            raise ValueError("请指定数据集目录")
        return list_pairs_from_dataset_dir(dataset_dir, split)
    if not data_yaml or not os.path.isfile(data_yaml):
        raise FileNotFoundError("data yaml 无效")
    pairs, split_used = list_split_images(data_yaml, split=split, strict=True)
    meta = load_data_yaml(data_yaml)
    return pairs, split_used, meta["names"], data_yaml


def _yolo_line_to_xyxy(
    parts: Sequence[str],
    w: int,
    h: int,
) -> Tuple[int, Tuple[float, float, float, float]]:
    cls = int(float(parts[0]))
    cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return cls, (x1, y1, x2, y2)


def load_gt_boxes(label_path: str, names: Dict[int, str], img_w: int, img_h: int) -> List[Box]:
    if not label_path or not os.path.isfile(label_path):
        return []
    boxes: List[Box] = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, xyxy = _yolo_line_to_xyxy(parts, img_w, img_h)
            boxes.append(Box(cls=cls, name=names.get(cls, str(cls)), xyxy=xyxy, conf=1.0))
    return boxes


def _boxes_from_result(res, names: Dict[int, str]) -> List[Box]:
    boxes: List[Box] = []
    if res.boxes is None or len(res.boxes) == 0:
        return boxes
    xyxy = res.boxes.xyxy.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    for i in range(len(clss)):
        c = int(clss[i])
        boxes.append(
            Box(
                cls=c,
                name=names.get(c, str(c)),
                xyxy=tuple(float(x) for x in xyxy[i]),
                conf=float(confs[i]),
            )
        )
    return boxes


def predict_boxes(model, image_path: str, names: Dict[int, str], conf: float) -> List[Box]:
    res = model.predict(source=image_path, conf=conf, verbose=False)[0]
    return _boxes_from_result(res, names)


def predict_all_boxes(
    model,
    image_paths: Sequence[str],
    names: Dict[int, str],
    conf: float,
    *,
    batch: int = 32,
    device: str = "0",
    progress: Optional[ProgressFn] = None,
) -> Dict[str, List[Box]]:
    """Batch inference; returns {resolved_image_path: [Box, ...]}."""
    pred_map: Dict[str, List[Box]] = {}
    n = len(image_paths)
    chunk_size = max(1, int(batch))
    for start in range(0, n, chunk_size):
        chunk = list(image_paths[start : start + chunk_size])
        results = model.predict(source=chunk, conf=conf, device=device, verbose=False)
        if not isinstance(results, list):
            results = [results]
        for idx, res in enumerate(results):
            if idx >= len(chunk):
                break
            key = str(Path(chunk[idx]).resolve())
            pred_map[key] = _boxes_from_result(res, names)
        if progress:
            done = min(start + len(chunk), n)
            progress(done, n, f"推理 {done}/{n}")
    return pred_map


def match_boxes(
    gts: List[Box],
    preds: List[Box],
    iou_thr: float,
) -> Tuple[List[Tuple[int, int, float, bool]], List[int], List[int]]:
    """Greedy match by confidence. Returns matches, unmatched_gt_idx, unmatched_pred_idx."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    matched_gt: set = set()
    matches: List[Tuple[int, int, float, bool]] = []
    for pi in order:
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = compute_IOU(preds[pi].xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_gi)
            cls_ok = gts[best_gi].cls == preds[pi].cls
            matches.append((best_gi, pi, best_iou, cls_ok))
    matched_pred = {m[1] for m in matches}
    unmatched_gt = [i for i in range(len(gts)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(preds)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def eval_image(
    gts: List[Box],
    preds: List[Box],
    iou_thr: float = 0.5,
) -> Tuple[float, float, int, int, int, int, str]:
    """Return error_score(1-F1), f1, tp, fp, fn, class_errors, detail_text."""
    matches, unmatched_gt, unmatched_pred = match_boxes(gts, preds, iou_thr)
    tp = sum(1 for m in matches if m[3])
    class_err = sum(1 for m in matches if not m[3])
    fn = len(unmatched_gt) + class_err
    fp = len(unmatched_pred) + class_err
    if tp + fp + fn == 0:
        return 0.0, 1.0, 0, 0, 0, 0, "无目标"
    f1 = 2 * tp / (2 * tp + fp + fn)
    err = 1.0 - f1
    detail_parts = [f"TP={tp} FP={fp} FN={fn}"]
    if class_err:
        detail_parts.append(f"类错={class_err}")
    return err, f1, tp, fp, fn, class_err, ", ".join(detail_parts)


_CJK_FONT_CACHE: Dict[int, ImageFont.ImageFont] = {}


def _cjk_font(size: int = 20) -> ImageFont.ImageFont:
    if size not in _CJK_FONT_CACHE:
        path = resolve_cjk_plot_font()
        if path:
            _CJK_FONT_CACHE[size] = ImageFont.truetype(path, size, encoding="utf-8")
        else:
            _CJK_FONT_CACHE[size] = ImageFont.load_default()
    return _CJK_FONT_CACHE[size]


def _text_bbox(font: ImageFont.ImageFont, text: str) -> Tuple[int, int, int, int]:
    if hasattr(font, "getbbox"):
        return font.getbbox(text)
    w, h = font.getsize(text)
    return 0, 0, w, h


def draw_comparison(
    image_path: str,
    gts: List[Box],
    preds: List[Box],
    title: str = "",
) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _cjk_font(20)
    title_font = _cjk_font(24)

    for gt in gts:
        x1, y1, x2, y2 = map(int, gt.xyxy)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 200, 0), width=2)
        lbl = f"GT:{gt.name}"
        _, _, _, th = _text_bbox(font, lbl)
        ty = max(0, y1 - th - 4)
        draw.text((x1, ty), lbl, font=font, fill=(0, 200, 0))

    for pred in preds:
        x1, y1, x2, y2 = map(int, pred.xyxy)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        lbl = f"P:{pred.name} {pred.conf:.2f}"
        _, _, _, th = _text_bbox(font, lbl)
        ty = min(pil.height - th - 2, y2 + 4)
        draw.text((x1, ty), lbl, font=font, fill=(255, 0, 0))

    if title:
        draw.text((8, 6), title, font=title_font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _imwrite(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)


def _is_low_score(item: ImageEval) -> bool:
    """True when F1 < 1 (any detection mismatch)."""
    return item.score > 0


def _sample_pairs(
    pairs: Sequence[Tuple[str, str]],
    limit: int,
    *,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    if limit <= 0 or limit >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    idx = rng.sample(range(len(pairs)), limit)
    return [pairs[i] for i in idx]


def _export_split_folder(split_used: str) -> str:
    if split_used in ("train", "val", "test", "all"):
        return split_used
    return "custom"


def _class_stats(items: List[ImageEval]) -> Dict[str, dict]:
    """Per GT class: how many images contain it and how often F1 is low."""
    stats: Dict[str, dict] = {}
    for it in items:
        classes = list(dict.fromkeys(it.gt_classes)) if it.gt_classes else ["(无标注)"]
        for cn in classes:
            st = stats.setdefault(cn, {"count": 0, "bad": 0, "f1_sum": 0.0})
            st["count"] += 1
            st["f1_sum"] += it.f1
            if it.f1 < 0.5:
                st["bad"] += 1
    return stats


def _write_html(
    out_dir: str,
    report_title: str,
    split_used: str,
    overall: dict,
    items: List[ImageEval],
    worst: List[ImageEval],
    class_stats: Dict[str, dict],
    weights: str,
    data_yaml: str,
    export_dataset_dir: str = "",
    export_split: str = "",
) -> str:
    index_path = os.path.join(out_dir, "index.html")
    rel = lambda p: html.escape(os.path.relpath(p, out_dir).replace("\\", "/"))
    exp_split = export_split or split_used

    rows = []
    for i, w in enumerate(worst):
        gt_txt = ", ".join(w.gt_classes) if w.gt_classes else "(无)"
        pred_txt = ", ".join(f"{c}" for c in w.pred_classes) if w.pred_classes else "(无)"
        if w.export_name:
            img_rel = rel(os.path.join(out_dir, "images", w.export_name))
            vis_cell = (
                f"<td><a href='{img_rel}' target='_blank'>"
                f"<img src='{img_rel}' width='240'/></a></td>"
            )
        else:
            vis_cell = "<td><span style='color:#888;'>—</span></td>"
        rows.append(
            f"<tr><td>{i+1}</td>{vis_cell}"
            f"<td>{html.escape(os.path.basename(w.image_path))}</td>"
            f"<td>{w.f1:.3f}</td><td>{w.score:.3f}</td>"
            f"<td>{html.escape(gt_txt)}</td><td>{html.escape(pred_txt)}</td>"
            f"<td>{html.escape(w.detail)}</td></tr>"
        )

    cls_rows = []
    for cn, st in sorted(class_stats.items(), key=lambda x: (-x[1]["bad"], x[0])):
        avg_f1 = st["f1_sum"] / max(st["count"], 1)
        cls_rows.append(
            f"<tr><td>{html.escape(cn)}</td><td>{st['count']}</td>"
            f"<td>{avg_f1:.3f}</td><td>{st['bad']}</td></tr>"
        )

    ov = overall or {}
    summary = (
        f"<p>权重: <code>{html.escape(weights)}</code></p>"
        f"<p>数据集: <code>{html.escape(data_yaml)}</code> | 划分: <b>{html.escape(split_used)}</b></p>"
        f"<p>评估图像数: {len(items)} | 低分样本 (F1&lt;1): {len(worst)}</p>"
    )
    if ov:
        metrics_line = ", ".join(f"{k}={v}" for k, v in ov.items() if v is not None)
        summary += f"<p>Ultralytics 指标: {html.escape(metrics_line)}</p>"
    if export_dataset_dir:
        summary += (
            f"<p>低分样本数据集: <code>{html.escape(export_dataset_dir)}</code>"
            f"（images/{html.escape(exp_split)} + labels/{html.escape(exp_split)}，"
            f"全部 F1&lt;1 样本已导出，可用标注工具打开修正）</p>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>{html.escape(report_title)}</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; background: #f5f5f5; }}
h1,h2 {{ color: #333; }}
.card {{ background: #fff; padding: 16px 20px; margin: 16px 0; border-radius: 8px;
         box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #eee; }}
.legend {{ font-size: 13px; color: #555; }}
.legend span {{ display: inline-block; margin-right: 16px; }}
.gt {{ color: #0a0; font-weight: bold; }} .pred {{ color: #c00; font-weight: bold; }}
</style></head><body>
<h1>{html.escape(report_title)}</h1>
<div class="card">{summary}
<p class="legend"><span class="gt">■ 绿框 = GT 实际</span>
<span class="pred">■ 红框 = 预测 (P:类名 score)</span></p>
<p>score = 1 − F1（越大越差）；优先查看 score 高 / F1 低的样本。</p></div>
<div class="card"><h2>各类别统计（bad = F1&lt;0.5 的图像数）</h2>
<table><tr><th>类别</th><th>图像数</th><th>平均 F1</th><th>bad</th></tr>
{"".join(cls_rows) if cls_rows else "<tr><td colspan='4'>无</td></tr>"}
</table></div>
<div class="card"><h2>低分样本（检测效果差）</h2>
<table><tr><th>#</th><th>可视化</th><th>文件</th><th>F1</th><th>score</th>
<th>GT 类别</th><th>预测类别</th><th>详情</th></tr>
{"".join(rows) if rows else "<tr><td colspan='8'>无</td></tr>"}
</table></div>
<p style="color:#888;font-size:12px;">生成时间 {html.escape(time.strftime("%Y-%m-%d %H:%M:%S"))}</p>
</body></html>"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return index_path


def export_low_score_dataset(
    worst: List[ImageEval],
    export_root: str,
    names: Dict[int, str],
    *,
    split_folder: str = "test",
    log: LogFn = print,
) -> Tuple[str, int, int]:
    """Copy low-score originals to images/<split> + labels/<split> for relabeling."""
    root = Path(export_root).resolve()
    img_dir = root / "images" / split_folder
    lbl_dir = root / "labels" / split_folder
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[dict] = []
    img_n = lbl_n = 0
    for rank, item in enumerate(worst, start=1):
        src_img = Path(item.image_path)
        if not src_img.is_file():
            continue
        dst_img = img_dir / src_img.name
        shutil.copy2(src_img, dst_img)
        img_n += 1

        dst_lbl = lbl_dir / f"{src_img.stem}.txt"
        if item.label_path and os.path.isfile(item.label_path):
            shutil.copy2(item.label_path, dst_lbl)
            lbl_n += 1
        else:
            dst_lbl.write_text("", encoding="utf-8")

        manifest.append(
            {
                "rank": rank,
                "file": src_img.name,
                "f1": round(item.f1, 4),
                "score": round(item.score, 4),
                "detail": item.detail,
                "source_image": str(Path(item.image_path).resolve()),
                "source_label": str(Path(item.label_path).resolve()) if item.label_path else "",
            }
        )

    manifest_path = root / "low_score_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    name_list = [names[i] for i in sorted(names)]
    data_yaml = root / "data.yaml"
    yaml_body = {
        "path": str(root).replace("\\", "/"),
        "train": f"images/{split_folder}",
        "val": f"images/{split_folder}",
        "test": f"images/{split_folder}",
        "nc": len(name_list),
        "names": name_list,
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_body, f, allow_unicode=True, sort_keys=False)

    ensure_classes_txt(str(lbl_dir), names)

    readme = root / "README.txt"
    readme.write_text(
        "低分样本导出包（用于修正标注）\n"
        f"  images/{split_folder}/  原始图像\n"
        f"  labels/{split_folder}/  YOLO 标注\n"
        "  low_score_manifest.json  F1/score 与源路径对照\n"
        "  data.yaml                可用 FishVision/ALT 打开\n",
        encoding="utf-8",
    )
    _log(f"低分数据集: {root}（图像 {img_n}，标注 {lbl_n}）", log)
    return str(root), img_n, lbl_n


def run_test_report(
    weights: str,
    data_yaml: str,
    out_dir: str,
    *,
    source_mode: str = "yaml",
    dataset_dir: str = "",
    split: str = "test",
    conf: float = 0.25,
    iou_match: float = 0.5,
    sample_limit: int = 0,
    max_report_images: int = 100,
    device: str = "0",
    run_ultralytics_val: bool = True,
    predict_batch: int = 32,
    export_dataset: bool = False,
    export_dataset_dir: Optional[str] = None,
    log: LogFn = print,
    progress: Optional[ProgressFn] = None,
) -> Tuple[str, str]:
    """Run eval, export worst images, write index.html. Returns (html_path, exported_dataset_dir)."""
    out_dir = str(Path(out_dir).resolve())
    img_out = os.path.join(out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    pairs, split_used, names, yaml_for_val = resolve_report_pairs(
        source_mode=source_mode,
        data_yaml=data_yaml,
        dataset_dir=dataset_dir,
        split=split,
    )
    total_in_split = len(pairs)
    if sample_limit > 0 and sample_limit < total_in_split:
        pairs = _sample_pairs(pairs, sample_limit)
        _log(f"随机抽样评估: {len(pairs)} / {total_in_split} 张", log)
    report_yaml = yaml_for_val or data_yaml or dataset_dir
    img_paths = [p for p, _ in pairs]
    export_split = _export_split_folder(split_used)

    src_desc = dataset_dir if source_mode == "dir" else data_yaml
    _log(f"测试报告: 来源={source_mode} | split={split_used} | 评估 {len(pairs)} 张", log)
    _log(f"数据: {src_desc}", log)
    model = YOLO(weights)

    overall: dict = {}
    can_ultra_val = run_ultralytics_val and yaml_for_val and os.path.isfile(yaml_for_val)
    if run_ultralytics_val and not can_ultra_val:
        _log("Ultralytics 整集验证跳过（自定义目录模式或无 data.yaml）", log)
    if can_ultra_val:
        try:
            _log("Ultralytics 整集验证…", log)
            metrics = model.val(
                data=yaml_for_val,
                split=split_used if split_used in ("train", "val", "test") else "val",
                conf=conf,
                iou=iou_match,
                device=device,
                verbose=False,
            )
            if hasattr(metrics, "results_dict"):
                rd = metrics.results_dict
                for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                    if k in rd:
                        overall[k.replace("metrics/", "").replace("(B)", "")] = f"{float(rd[k]):.4f}"
        except Exception as exc:
            _log(f"Ultralytics val 跳过 ({exc})，继续逐图分析", log)

    _log("批量推理…", log)
    pred_map = predict_all_boxes(
        model,
        img_paths,
        names,
        conf,
        batch=predict_batch,
        device=device,
        progress=progress,
    )

    results: List[ImageEval] = []
    n = len(pairs)
    for i, (img_path, label_path) in enumerate(pairs):
        key = str(Path(img_path).resolve())
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt_boxes(label_path, names, w, h)
        preds = pred_map.get(key, [])
        err, f1, tp, fp, fn, cls_err, detail = eval_image(gts, preds, iou_match)
        results.append(
            ImageEval(
                image_path=img_path,
                label_path=label_path,
                score=err,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
                class_errors=cls_err,
                gt_count=len(gts),
                pred_count=len(preds),
                gt_classes=[b.name for b in gts],
                pred_classes=[f"{b.name}({b.conf:.2f})" for b in preds],
                gt_boxes=gts,
                pred_boxes=preds,
                detail=detail,
            )
        )
        if progress:
            progress(i + 1, n, f"评估 {os.path.basename(img_path)}")

    results.sort(key=lambda x: (-x.score, x.f1))
    low_score = [r for r in results if _is_low_score(r)]
    preview_n = max_report_images if max_report_images > 0 else len(low_score)
    preview = low_score[:preview_n]

    for j, item in enumerate(preview):
        safe = f"{j+1:03d}_f1{item.f1:.2f}_{Path(item.image_path).stem}.jpg"
        item.export_name = safe
        title = f"F1={item.f1:.2f} {item.detail}"
        vis = draw_comparison(item.image_path, item.gt_boxes, item.pred_boxes, title)
        _imwrite(os.path.join(img_out, safe), vis)

    exported_dataset_dir = ""
    if export_dataset and low_score:
        ds_root = export_dataset_dir or os.path.join(out_dir, "low_score_dataset")
        exported_dataset_dir, _, _ = export_low_score_dataset(
            low_score, ds_root, names, split_folder=export_split, log=log
        )

    class_stats = _class_stats(results)
    html_path = _write_html(
        out_dir,
        report_title="FishVision 测试集诊断报告",
        split_used=split_used,
        overall=overall,
        items=results,
        worst=low_score,
        class_stats=class_stats,
        weights=weights,
        data_yaml=report_yaml,
        export_dataset_dir=exported_dataset_dir,
        export_split=export_split,
    )
    _log(f"报告已生成: {html_path}", log)
    _log(f"低分样本: {len(low_score)} 张（报告可视化前 {len(preview)} 张）", log)
    if exported_dataset_dir:
        _log(f"低分数据集目录: {exported_dataset_dir}", log)
    if progress:
        progress(n, n, "done")
    return html_path, exported_dataset_dir

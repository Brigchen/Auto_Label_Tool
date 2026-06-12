# -*- coding: utf-8 -*-
"""Self-refine YOLO labels: compare GT with model predictions and apply conservative fixes."""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO

from libs.autolabel_dedup import apply_autolabel_dedup
from libs.dataset_paths import flat_output_stem
from libs.yolo_line_parse import parse_yolo_bbox_line, normalize_detection_score
from libs.test_report import (
    Box,
    _read_image_size,
    boxes_to_yolo_text,
    load_data_yaml,
    load_gt_boxes,
    match_boxes,
    list_split_images,
    predict_all_boxes,
)

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]


@dataclass
class RefineRules:
    """Conservative auto-fix rules (high-confidence disagreements only)."""

    iou_match: float = 0.5
    min_fix_conf: float = 0.65
    min_score_keep: float = 0.0
    drop_low_score: bool = True
    fix_wrong_class: bool = True
    drop_unmatched_gt: bool = False
    add_missing_pred: bool = False
    min_add_conf: float = 0.85
    cross_class_dedup: bool = True
    dedup_iou: float = 0.7
    export_review: bool = True


@dataclass
class RefineAction:
    image_path: str
    label_path: str
    split: str
    action: str
    detail: str
    old_line: str = ""
    new_line: str = ""


@dataclass
class RefineSummary:
    images: int = 0
    labels_changed: int = 0
    lines_dropped: int = 0
    lines_class_fixed: int = 0
    lines_added: int = 0
    review_images: int = 0
    actions: List[RefineAction] = field(default_factory=list)


def _win_long_path(path: str) -> str:
    """Enable extended-length paths on Windows when needed."""
    if os.name != "nt" or not path:
        return path
    p = os.path.normpath(os.path.abspath(path))
    if p.startswith("\\\\?\\"):
        return p
    if len(p) < 240:
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p.lstrip("\\")
    return "\\\\?\\" + p


def _safe_copy2(src: str, dst: str) -> None:
    src = _win_long_path(src)
    dst = _win_long_path(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)


def _export_stem_from_abs(abs_path: str, used: set) -> str:
    """Unique flat stem for nested dataset files (avoid basename collisions)."""
    p = Path(abs_path)
    parts = p.parts
    if len(parts) >= 2:
        tail = parts[-2:]
        rel = "/".join(tail)
    else:
        rel = p.name
    stem = flat_output_stem(rel)
    base = stem
    n = 2
    while stem.lower() in used:
        stem = f"{base}_{n}"
        n += 1
    used.add(stem.lower())
    return stem


def _safe_resolve(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve())
    except OSError:
        return os.path.abspath(path)


def _log(msg: str, log: LogFn = print) -> None:
    ts = time.strftime("%H:%M:%S")
    log(f"[{ts}] {msg}")


def _parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float, Optional[float]]]:
    """Parse YOLO detect or pose line; score only from trailing column (not keypoint x)."""
    return parse_yolo_bbox_line(line)


def _gt_line_to_box(parsed, names: Dict[int, str], img_w: int, img_h: int) -> Box:
    cls, cx, cy, w, h, score = parsed
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return Box(
        cls=cls,
        name=names.get(cls, str(cls)),
        xyxy=(x1, y1, x2, y2),
        conf=float(score) if score is not None else 1.0,
    )


def _boxes_to_lines(boxes: List[Box], img_w: int, img_h: int, save_score: bool) -> List[str]:
    text = boxes_to_yolo_text(boxes, img_w, img_h, save_score=save_score)
    return [ln for ln in text.splitlines() if ln.strip()]


def refine_one_image(
    image_path: str,
    label_path: str,
    preds: List[Box],
    names: Dict[int, str],
    rules: RefineRules,
    *,
    split: str = "",
) -> Tuple[List[str], List[RefineAction], bool]:
    """Return (output YOLO lines, actions, needs_review)."""
    actions: List[RefineAction] = []
    needs_review = False
    img_w, img_h = _read_image_size(image_path)
    if img_w <= 0:
        return [], actions, False

    raw_lines: List[str] = []
    parsed_rows: List[Tuple[int, float, float, float, float, Optional[float]]] = []
    if label_path and os.path.isfile(label_path):
        with open(label_path, encoding="utf-8") as f:
            for raw in f.read().splitlines():
                line = raw.strip()
                if not line:
                    continue
                p = _parse_yolo_line(line)
                if p is None:
                    continue
                raw_lines.append(line)
                parsed_rows.append(p)

    gts = [_gt_line_to_box(p, names, img_w, img_h) for p in parsed_rows]
    matches, unmatched_gt, unmatched_pred = match_boxes(gts, preds, rules.iou_match)
    match_by_gt = {gi: (pi, iou, cls_ok) for gi, pi, iou, cls_ok in matches}

    keep_boxes: List[Box] = []
    for gi, p in enumerate(parsed_rows):
        cls, cx, cy, w, h, score = p
        score = normalize_detection_score(score)
        old_line = raw_lines[gi] if gi < len(raw_lines) else ""
        gt_box = gts[gi]

        if rules.drop_low_score and score is not None and score < rules.min_score_keep:
            actions.append(RefineAction(
                image_path, label_path, split, "drop_low_score",
                f"score {score:.3f} < {rules.min_score_keep:.3f}",
                old_line, "",
            ))
            continue

        if gi in unmatched_gt:
            if rules.drop_unmatched_gt:
                actions.append(RefineAction(
                    image_path, label_path, split, "drop_unmatched_gt",
                    "no matching prediction", old_line, "",
                ))
                continue
            needs_review = True
            actions.append(RefineAction(
                image_path, label_path, split, "review_fn",
                "GT box unmatched (possible漏标 or hard sample)", old_line, old_line,
            ))
            keep_boxes.append(gt_box)
            continue

        pi, iou, cls_ok = match_by_gt[gi]
        pred = preds[pi]
        if not cls_ok and rules.fix_wrong_class and pred.conf >= rules.min_fix_conf:
            new_box = Box(
                cls=pred.cls,
                name=pred.name,
                xyxy=gt_box.xyxy,
                conf=pred.conf,
            )
            keep_boxes.append(new_box)
            new_line = _boxes_to_lines([new_box], img_w, img_h, save_score=True)
            actions.append(RefineAction(
                image_path, label_path, split, "fix_class",
                f"{gt_box.name} -> {pred.name} (IoU={iou:.2f}, conf={pred.conf:.3f})",
                old_line,
                new_line[0] if new_line else "",
            ))
            continue

        if not cls_ok:
            needs_review = True
            proposed = Box(
                cls=pred.cls,
                name=pred.name,
                xyxy=gt_box.xyxy,
                conf=pred.conf,
            )
            prop_lines = _boxes_to_lines([proposed], img_w, img_h, save_score=True)
            actions.append(RefineAction(
                image_path, label_path, split, "review_class",
                f"class mismatch, pred conf {pred.conf:.3f} < fix threshold",
                old_line,
                prop_lines[0] if prop_lines else old_line,
            ))
        keep_boxes.append(gt_box)

    if rules.add_missing_pred:
        for pi in unmatched_pred:
            pred = preds[pi]
            if pred.conf < rules.min_add_conf:
                continue
            keep_boxes.append(pred)
            add_lines = _boxes_to_lines([pred], img_w, img_h, save_score=True)
            actions.append(RefineAction(
                image_path, label_path, split, "add_pred",
                f"add {pred.name} conf={pred.conf:.3f}",
                "", add_lines[0] if add_lines else "",
            ))

    if rules.cross_class_dedup and len(keep_boxes) > 1:
        xyxy = [list(b.xyxy) for b in keep_boxes]
        scores = [float(b.conf) for b in keep_boxes]
        labels = [b.name for b in keep_boxes]
        xyxy, scores, labels = apply_autolabel_dedup(
            xyxy, scores, labels, rules.dedup_iou, cross_class=True,
        )
        name_to_cls = {v: k for k, v in names.items()}
        deduped: List[Box] = []
        for box, sc, nm in zip(xyxy, scores, labels):
            cid = name_to_cls.get(nm, keep_boxes[0].cls if keep_boxes else 0)
            deduped.append(Box(cls=int(cid), name=nm, xyxy=tuple(box), conf=float(sc)))
        if len(deduped) < len(keep_boxes):
            actions.append(RefineAction(
                image_path, label_path, split, "cross_class_dedup",
                f"{len(keep_boxes)} -> {len(deduped)} boxes",
                "", "",
            ))
        keep_boxes = deduped

    out_lines = _boxes_to_lines(keep_boxes, img_w, img_h, save_score=True)
    return out_lines, actions, needs_review


def _collect_pairs(data_yaml: str, splits: Sequence[str]) -> List[Tuple[str, str, str]]:
    """Return [(image_path, label_path, split_name), ...]."""
    rows: List[Tuple[str, str, str]] = []
    for sp in splits:
        try:
            pairs, used = list_split_images(data_yaml, split=sp, strict=True)
        except FileNotFoundError:
            continue
        for img, lbl in pairs:
            rows.append((img, lbl, used))
    return rows


def _backup_label_roots(data_yaml: str, splits: Sequence[str], backup_root: Path) -> None:
    meta = load_data_yaml(data_yaml)
    root = Path(meta.get("path") or Path(data_yaml).parent)
    seen = set()
    for sp in splits:
        try:
            pairs, _ = list_split_images(data_yaml, split=sp, strict=True)
        except FileNotFoundError:
            continue
        for _img, lbl in pairs:
            lbl_p = Path(lbl)
            if not lbl_p.is_file():
                continue
            split_dir = lbl_p.parent
            key = str(split_dir.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                rel = split_dir.relative_to(root.resolve())
            except ValueError:
                rel = Path(split_dir.name)
            dst = backup_root / rel
            if split_dir.is_dir():
                shutil.copytree(split_dir, dst, dirs_exist_ok=True)


def save_refine_report(out_dir: str, summary: RefineSummary, rules: RefineRules) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "label_self_refine_report.csv"
    json_path = out / "label_self_refine_summary.json"

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "label", "action", "detail", "old_line", "new_line"])
        for a in summary.actions:
            w.writerow([
                a.split,
                a.image_path,
                a.label_path,
                a.action,
                a.detail,
                a.old_line,
                a.new_line,
            ])

    payload = {
        "images": summary.images,
        "labels_changed": summary.labels_changed,
        "lines_dropped": summary.lines_dropped,
        "lines_class_fixed": summary.lines_class_fixed,
        "lines_added": summary.lines_added,
        "review_images": summary.review_images,
        "rules": rules.__dict__,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}


def load_review_items_from_report_csv(csv_path: str) -> List[Tuple[str, str, str]]:
    """Parse review_* rows from an existing label_self_refine_report.csv."""
    csv_path = str(Path(csv_path).resolve())
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Report CSV not found: {csv_path}")

    by_image: Dict[str, Tuple[str, str, str]] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = (row.get("action") or "").strip()
            if not action.startswith("review"):
                continue
            img = (row.get("image") or "").strip()
            if not img:
                continue
            lbl = (row.get("label") or "").strip()
            split = (row.get("split") or "").strip()
            if img not in by_image:
                by_image[img] = (img, lbl, split)
    return list(by_image.values())


def run_export_review_only(
    data_yaml: str,
    out_dir: str,
    *,
    report_csv: Optional[str] = None,
    log: LogFn = print,
    progress: Optional[ProgressFn] = None,
) -> int:
    """Re-export review_queue from an existing report CSV (skip model inference)."""
    out_dir = str(Path(out_dir).resolve())
    csv_path = str(Path(report_csv).resolve()) if report_csv else os.path.join(
        out_dir, "label_self_refine_report.csv",
    )
    review_items = load_review_items_from_report_csv(csv_path)
    if not review_items:
        raise ValueError("No review_* rows in report CSV; nothing to export.")

    meta = load_data_yaml(data_yaml)
    names = meta["names"]
    _log(
        f"Review-only export: {len(review_items)} images from {csv_path}",
        log,
    )

    review_root = os.path.join(out_dir, "review_queue")
    rq = _export_review_queue(
        review_items,
        review_root,
        names,
        "review",
        log=log,
        progress=progress,
    )
    if not rq:
        raise RuntimeError("review_queue export produced no files (check source paths).")
    _log(f"Review queue: {rq} ({len(review_items)} images)", log)
    try:
        from libs.label_self_refine_report import write_refine_html_report
        write_refine_html_report(out_dir, data_yaml=data_yaml, log=log)
    except Exception as exc:
        _log(f"[WARN] HTML 报告生成失败: {exc}", log)
    return len(review_items)


def run_refine_html_only(
    out_dir: str,
    *,
    data_yaml: str = "",
    weights: str = "",
    report_csv: Optional[str] = None,
    build_interactive: bool = False,
    log: LogFn = print,
    progress: Optional[ProgressFn] = None,
) -> str:
    """Build index.html from existing CSV (no full re-inference)."""
    from libs.label_self_refine_report import write_refine_html_report
    out_dir = str(Path(out_dir).resolve())
    csv_path = report_csv or os.path.join(out_dir, "label_self_refine_report.csv")
    html_path = write_refine_html_report(
        out_dir,
        csv_path=csv_path,
        data_yaml=data_yaml,
        weights=weights,
        dry_run=True,
        log=log,
    )
    if build_interactive:
        try:
            from libs.label_self_refine_review import build_review_manifest, _write_review_html
            build_review_manifest(
                out_dir,
                data_yaml=data_yaml,
                weights=weights,
                csv_path=csv_path,
                max_previews=200,
                log=log,
                progress=progress,
            )
            _write_review_html(out_dir)
        except Exception as exc:
            log(f"[WARN] 复核清单: {exc}")
    return html_path


def _export_review_queue(
    review_items: List[Tuple[str, str, str]],
    out_dir: str,
    names: Dict[int, str],
    split_folder: str,
    log: LogFn = print,
    progress: Optional[ProgressFn] = None,
) -> str:
    """Copy uncertain samples for manual review (robust paths, unique stems)."""
    if not review_items:
        return ""

    root = Path(out_dir).resolve()
    img_dir = root / "images" / split_folder
    lbl_dir = root / "labels" / split_folder
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    used_stems: set = set()
    manifest: List[dict] = []
    img_n = lbl_n = skipped = 0
    total = len(review_items)

    for rank, (img, lbl, sp) in enumerate(review_items, start=1):
        if progress and (rank == 1 or rank % 25 == 0 or rank == total):
            progress(rank, total, os.path.basename(img))
        src_img = Path(img)
        if not src_img.is_file():
            skipped += 1
            continue
        stem = _export_stem_from_abs(str(src_img), used_stems)
        ext = src_img.suffix if src_img.suffix else ".jpg"
        dst_img = img_dir / f"{stem}{ext}"
        try:
            _safe_copy2(str(src_img), str(dst_img))
        except OSError as exc:
            skipped += 1
            if skipped <= 5:
                log(f"[review_queue] skip image: {src_img.name} ({exc})")
            continue
        img_n += 1

        dst_lbl = lbl_dir / f"{stem}.txt"
        try:
            if lbl and os.path.isfile(lbl):
                _safe_copy2(lbl, str(dst_lbl))
                lbl_n += 1
            else:
                dst_lbl.write_text("", encoding="utf-8")
        except OSError as exc:
            skipped += 1
            if skipped <= 5:
                log(f"[review_queue] skip label for {stem}: {exc}")
            try:
                dst_lbl.write_text("", encoding="utf-8")
            except OSError:
                pass

        manifest.append({
            "rank": rank,
            "file": f"{stem}{ext}",
            "split": sp,
            "source_image": _safe_resolve(str(src_img)),
            "source_label": _safe_resolve(lbl) if lbl else "",
        })

    if img_n == 0:
        log(f"[review_queue] no images exported ({skipped} skipped)")
        return ""

    manifest_path = root / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    name_list = [names[i] for i in sorted(names)] if names else ["0"]
    data_yaml = root / "data.yaml"
    yaml_body = {
        "path": str(root).replace("\\", "/"),
        "train": f"images/{split_folder}",
        "val": f"images/{split_folder}",
        "test": f"images/{split_folder}",
        "nc": len(name_list),
        "names": name_list,
    }
    import yaml as _yaml
    with open(data_yaml, "w", encoding="utf-8") as f:
        _yaml.safe_dump(yaml_body, f, allow_unicode=True, sort_keys=False)

    from libs.yolo_dataset_paths import ensure_classes_txt
    ensure_classes_txt(str(lbl_dir), names)

    readme = root / "README.txt"
    readme.write_text(
        "Label self-refine review queue\n"
        f"  images/{split_folder}/  uncertain samples\n"
        f"  labels/{split_folder}/  original GT labels\n"
        "  review_manifest.json      source path mapping\n"
        "  data.yaml                 open in ALT / FishVision\n",
        encoding="utf-8",
    )
    if skipped:
        log(f"[review_queue] exported {img_n} images, {lbl_n} labels ({skipped} skipped)")
    return str(root)


def run_label_self_refine(
    weights: str,
    data_yaml: str,
    out_dir: str,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    rules: Optional[RefineRules] = None,
    dry_run: bool = True,
    device: str = "0",
    predict_batch: int = 32,
    export_html: bool = True,
    build_interactive: bool = False,
    log: LogFn = print,
    progress: Optional[ProgressFn] = None,
) -> RefineSummary:
    """Compare labels with model predictions; optionally write refined labels (with backup)."""
    rules = rules or RefineRules()
    data_yaml = str(Path(data_yaml).resolve())
    out_dir = str(Path(out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)

    meta = load_data_yaml(data_yaml)
    names = meta["names"]
    pairs = _collect_pairs(data_yaml, splits)
    if not pairs:
        raise FileNotFoundError("data yaml 中未找到可处理的 train/val/test 图像对")

    _log(f"标注自修正: {len(pairs)} 张 | splits={list(splits)} | dry_run={dry_run}", log)
    model = YOLO(weights)
    img_paths = [p[0] for p in pairs]
    _log("批量推理…", log)
    pred_map = predict_all_boxes(
        model, img_paths, names, conf=min(rules.min_fix_conf, rules.min_add_conf, 0.25),
        batch=predict_batch, device=device, progress=progress,
    )

    summary = RefineSummary()
    review_items: List[Tuple[str, str, str]] = []
    changed_files: Dict[str, List[str]] = {}

    n = len(pairs)
    for i, (img_path, label_path, split) in enumerate(pairs):
        key = str(Path(img_path).resolve())
        preds = pred_map.get(key, [])
        out_lines, actions, needs_review = refine_one_image(
            img_path, label_path, preds, names, rules, split=split,
        )
        summary.actions.extend(actions)
        summary.images += 1
        for a in actions:
            if a.action == "drop_low_score" or a.action == "drop_unmatched_gt":
                summary.lines_dropped += 1
            elif a.action == "fix_class":
                summary.lines_class_fixed += 1
            elif a.action == "add_pred":
                summary.lines_added += 1
        if needs_review and rules.export_review:
            review_items.append((img_path, label_path, split))

        old_body = ""
        if label_path and os.path.isfile(label_path):
            old_body = Path(label_path).read_text(encoding="utf-8")
        new_body = ("\n".join(out_lines) + "\n") if out_lines else ""

        if old_body.strip() != new_body.strip():
            summary.labels_changed += 1
            changed_files[label_path] = out_lines

        if progress:
            progress(i + 1, n, os.path.basename(img_path))

    if progress:
        progress(n, n, "done")

    summary.review_images = len({x[0] for x in review_items})

    report_paths = save_refine_report(out_dir, summary, rules)
    _log(f"报告: {report_paths['csv']}", log)

    if export_html:
        try:
            from libs.label_self_refine_report import write_refine_html_report
            _log("生成 HTML 摘要…", log)
            write_refine_html_report(
                out_dir,
                data_yaml=data_yaml,
                weights=weights,
                dry_run=dry_run,
                log=log,
            )
        except Exception as exc:
            _log(f"[WARN] HTML 报告生成失败: {exc}", log)

    if build_interactive:
        try:
            from libs.label_self_refine_review import build_review_manifest, _write_review_html
            _log("生成交互复核清单…", log)
            build_review_manifest(
                out_dir,
                data_yaml=data_yaml,
                weights=weights,
                pred_map=pred_map,
                max_previews=200,
                log=log,
                progress=progress,
            )
            _write_review_html(out_dir)
        except Exception as exc:
            _log(f"[WARN] 交互复核清单失败: {exc}", log)

    if rules.export_review and review_items:
        try:
            review_root = os.path.join(out_dir, "review_queue")
            rq = _export_review_queue(
                review_items, review_root, names, "review", log=log,
            )
            if rq:
                _log(f"待人工复核: {rq} ({summary.review_images} 张)", log)
        except Exception as exc:
            _log(f"[WARN] review_queue 导出失败（报告已保存）: {exc}", log)

    if dry_run:
        _log(
            f"[dry-run] 将修改 {summary.labels_changed} 个标注文件 "
            f"(删 {summary.lines_dropped} 行, 改类 {summary.lines_class_fixed}, "
            f"新增 {summary.lines_added})",
            log,
        )
        return summary

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = Path(meta.get("path") or Path(data_yaml).parent) / f"labels_backup_{stamp}"
    _log(f"备份原标注 -> {backup_root}", log)
    _backup_label_roots(data_yaml, splits, backup_root)

    for label_path, lines in changed_files.items():
        os.makedirs(os.path.dirname(label_path) or ".", exist_ok=True)
        body = ("\n".join(lines) + "\n") if lines else ""
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(body)

    _log(
        f"已写入 {summary.labels_changed} 个标注文件 "
        f"(删 {summary.lines_dropped} 行, 改类 {summary.lines_class_fixed}, 新增 {summary.lines_added})",
        log,
    )
    return summary

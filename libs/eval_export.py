# -*- coding: utf-8 -*-
"""Export per-class metrics and confusion matrix (Excel/CSV/JSON) for training & test report."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LogFn = Callable[[str], None]


def _log(msg: str, log: LogFn = print) -> None:
    log(msg)


def _cm_labels_from_names(names: Dict[int, str], nc: int) -> List[str]:
    labels = [str(names.get(i, str(i))) for i in range(nc)]
    return labels + ["background"]


def normalize_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    """Column-normalize like Ultralytics (axis = true class)."""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return m
    return m / (m.sum(0).reshape(1, -1) + 1e-9)


def build_confusion_matrix_from_boxes(
    pairs: Sequence[Tuple[str, str]],
    pred_map: Dict[str, list],
    names: Dict[int, str],
    iou_thr: float,
    *,
    load_gt_boxes,
    load_fn=None,
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Build (nc+1)² confusion matrix and per-class box counts from GT/pred pairs."""
    from libs.test_report import match_boxes

    nc = max(len(names), max(names.keys(), default=-1) + 1)
    labels = _cm_labels_from_names(names, nc)
    cm = np.zeros((nc + 1, nc + 1), dtype=float)
    box_stats = {i: {"tp": 0, "fp": 0, "fn": 0, "instances": 0} for i in range(nc)}

    for img_path, label_path in pairs:
        import cv2

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt_boxes(label_path, names, w, h)
        preds = pred_map.get(str(Path(img_path).resolve()), [])

        for gt in gts:
            if 0 <= gt.cls < nc:
                box_stats[gt.cls]["instances"] += 1

        matches, unmatched_gt, unmatched_pred = match_boxes(gts, preds, iou_thr)
        matched_pred = {m[1] for m in matches}

        for gi, pi, _iou, cls_ok in matches:
            gc = gts[gi].cls
            dc = preds[pi].cls
            if not (0 <= gc < nc and 0 <= dc < nc + 1):
                continue
            if cls_ok:
                cm[dc, gc] += 1
                box_stats[gc]["tp"] += 1
            else:
                cm[dc, gc] += 1
                box_stats[gc]["fn"] += 1
                if dc < nc:
                    box_stats[dc]["fp"] += 1

        for gi in unmatched_gt:
            gc = gts[gi].cls
            if 0 <= gc < nc:
                cm[nc, gc] += 1
                box_stats[gc]["fn"] += 1

        for pi in unmatched_pred:
            dc = preds[pi].cls
            if 0 <= dc < nc:
                cm[dc, nc] += 1
                box_stats[dc]["fp"] += 1

    rows: List[dict] = []
    for i in range(nc):
        tp = box_stats[i]["tp"]
        fp = box_stats[i]["fp"]
        fn = box_stats[i]["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append(
            {
                "Class": labels[i],
                "Instances": box_stats[i]["instances"],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "F1": round(f1, 4),
            }
        )
    return cm, labels, rows


def per_class_rows_from_image_stats(class_stats: Dict[str, dict]) -> List[dict]:
    rows = []
    for cn, st in sorted(class_stats.items(), key=lambda x: (-x[1].get("bad", 0), x[0])):
        cnt = max(int(st.get("count", 0)), 1)
        avg_f1 = float(st.get("f1_sum", 0.0)) / cnt
        rows.append(
            {
                "Class": cn,
                "Images_with_class": st.get("count", 0),
                "Avg_image_F1": round(avg_f1, 4),
                "Bad_images_F1_lt_0.5": st.get("bad", 0),
            }
        )
    return rows


def _dataframe_confusion(matrix: np.ndarray, labels: List[str]) -> "Any":
    import pandas as pd

    n = min(len(labels), matrix.shape[0], matrix.shape[1])
    labels = labels[:n]
    sub = matrix[:n, :n]
    df = pd.DataFrame(sub, index=labels, columns=labels)
    return df.reset_index().rename(columns={"index": "Predicted"})


def _save_workbook(
    xlsx_path: str,
    sheets: Dict[str, "Any"],
    log: LogFn = print,
) -> str:
    import pandas as pd

    path = Path(xlsx_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
            for name, df in sheets.items():
                safe = str(name)[:31]
                if hasattr(df, "to_excel"):
                    df.to_excel(writer, sheet_name=safe, index=False)
                else:
                    pd.DataFrame(df).to_excel(writer, sheet_name=safe, index=False)
        return str(path)
    except ImportError:
        base = path.with_suffix("")
        _log(f"openpyxl 未安装，改为 CSV: {base}_*.csv", log)
        for name, df in sheets.items():
            out = f"{base}_{name}.csv"
            if hasattr(df, "to_csv"):
                df.to_csv(out, index=False, encoding="utf-8-sig")
        return str(base)


def save_confusion_sidecars(
    out_dir: str,
    labels: List[str],
    matrix: np.ndarray,
    *,
    basename: str = "confusion_matrix",
    log: LogFn = print,
) -> Dict[str, str]:
    """Save normalized/raw CM as csv + json for replotting."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = np.asarray(matrix, dtype=float)
    norm = normalize_confusion_matrix(m)
    n = min(len(labels), m.shape[0], m.shape[1])
    labels = labels[:n]
    m = m[:n, :n]
    norm = norm[:n, :n]

    csv_norm = out / f"{basename}_normalized.csv"
    csv_raw = out / f"{basename}.csv"
    json_path = out / f"{basename}_normalized.json"

    import pandas as pd

    pd.DataFrame(norm, index=labels, columns=labels).to_csv(
        csv_norm, encoding="utf-8-sig"
    )
    pd.DataFrame(m, index=labels, columns=labels).to_csv(
        csv_raw, encoding="utf-8-sig"
    )
    payload = {
        "class_names": labels,
        "matrix": m.tolist(),
        "matrix_normalized": norm.tolist(),
        "normalized": True,
        "normalize_axis": "columns",
        "note": "Rows=Predicted, Columns=True (Ultralytics convention)",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = {
        "confusion_csv": str(csv_raw),
        "confusion_normalized_csv": str(csv_norm),
        "confusion_normalized_json": str(json_path),
    }
    _log(f"混淆矩阵: {csv_norm.name}, {json_path.name}", log)
    return paths


def export_ultralytics_val_results(
    metrics,
    out_dir: str,
    *,
    basename: str = "eval_metrics",
    split: str = "",
    overall_extra: Optional[dict] = None,
    image_class_rows: Optional[List[dict]] = None,
    log: LogFn = print,
) -> Dict[str, str]:
    """Export Ultralytics val metrics + confusion matrix to Excel and sidecar files."""
    import pandas as pd

    out_dir = str(Path(out_dir).resolve())
    paths: Dict[str, str] = {}

    overall = dict(getattr(metrics, "results_dict", {}) or {})
    if overall_extra:
        overall.update(overall_extra)
    if split:
        overall["split"] = split

    per_class = []
    if hasattr(metrics, "summary"):
        try:
            per_class = metrics.summary(normalize=True, decimals=4)
        except Exception as exc:
            _log(f"per-class summary 跳过: {exc}", log)

    cm_obj = getattr(metrics, "confusion_matrix", None)
    matrix = np.array(getattr(cm_obj, "matrix", np.zeros((0, 0))), dtype=float)
    labels: List[str] = []
    if cm_obj is not None and hasattr(cm_obj, "names"):
        nc = len(cm_obj.names)
        labels = _cm_labels_from_names(dict(cm_obj.names), nc)
    elif matrix.size:
        n = matrix.shape[0]
        labels = [str(i) for i in range(n)]

    sheets: Dict[str, Any] = {}
    if overall:
        sheets["overall"] = pd.DataFrame([overall])
    if per_class:
        sheets["per_class"] = pd.DataFrame(per_class)
    if image_class_rows:
        sheets["per_class_by_image"] = pd.DataFrame(image_class_rows)
    if matrix.size and labels:
        sheets["confusion_raw"] = _dataframe_confusion(matrix, labels)
        sheets["confusion_normalized"] = _dataframe_confusion(
            normalize_confusion_matrix(matrix), labels
        )

    xlsx = os.path.join(out_dir, f"{basename}.xlsx")
    paths["excel"] = _save_workbook(xlsx, sheets, log=log)
    if matrix.size and labels:
        paths.update(
            save_confusion_sidecars(out_dir, labels, matrix, basename=basename, log=log)
        )
    _log(f"类群评估 Excel: {paths.get('excel', xlsx)}", log)
    return paths


def export_custom_eval_results(
    out_dir: str,
    *,
    pairs: Sequence[Tuple[str, str]],
    pred_map: Dict[str, list],
    names: Dict[int, str],
    iou_thr: float,
    load_gt_boxes,
    basename: str = "eval_metrics",
    split: str = "",
    overall: Optional[dict] = None,
    image_class_rows: Optional[List[dict]] = None,
    log: LogFn = print,
) -> Dict[str, str]:
    """Export box-level per-class stats and confusion matrix without Ultralytics val."""
    import pandas as pd

    cm, labels, per_class = build_confusion_matrix_from_boxes(
        pairs, pred_map, names, iou_thr, load_gt_boxes=load_gt_boxes
    )
    ov = dict(overall or {})
    if split:
        ov["split"] = split
    sheets: Dict[str, Any] = {}
    if ov:
        sheets["overall"] = pd.DataFrame([ov])
    if per_class:
        sheets["per_class"] = pd.DataFrame(per_class)
    if image_class_rows:
        sheets["per_class_by_image"] = pd.DataFrame(image_class_rows)
    if cm.size:
        sheets["confusion_raw"] = _dataframe_confusion(cm, labels)
        sheets["confusion_normalized"] = _dataframe_confusion(
            normalize_confusion_matrix(cm), labels
        )

    out_dir = str(Path(out_dir).resolve())
    xlsx = os.path.join(out_dir, f"{basename}.xlsx")
    paths = {"excel": _save_workbook(xlsx, sheets, log=log)}
    paths.update(
        save_confusion_sidecars(out_dir, labels, cm, basename=basename, log=log)
    )
    _log(f"类群评估 Excel: {paths['excel']}", log)
    return paths


def export_post_train_val_metrics(
    model,
    params: dict,
    run_dir: str,
    *,
    split: str = "val",
    log: LogFn = print,
) -> Optional[Dict[str, str]]:
    """Run val on best weights and export per-class Excel + confusion matrix."""
    data = params.get("data") or params.get("yaml_file") or params.get("data_yaml")
    if not data or not os.path.isfile(str(data)):
        _log("训练后评估跳过：无有效 data yaml", log)
        return None
    device = params.get("device", "0")
    try:
        metrics = model.val(
            data=str(data),
            split=split,
            device=device,
            verbose=False,
            plots=False,
        )
    except Exception as exc:
        _log(f"训练后 val 导出跳过: {exc}", log)
        return None
    return export_ultralytics_val_results(
        metrics,
        run_dir,
        basename=f"eval_{split}",
        split=split,
        overall_extra={
            "weights": params.get("weights_file", ""),
            "data_yaml": str(data),
            "task": "post_train_val",
        },
        log=log,
    )

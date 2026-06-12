# -*- coding: utf-8 -*-
"""Training diagnosis: data gaps, weak classes, confusion, actionable suggestions."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from libs.eval_export import normalize_confusion_matrix

MIN_INSTANCES_RECOMMENDED = 50
MIN_INSTANCES_WARN = 15
MIN_TEST_INSTANCES = 5
LOW_F1_THRESHOLD = 0.50
LOW_RECALL_THRESHOLD = 0.50
LOW_PRECISION_THRESHOLD = 0.50
CONFUSION_RATE_THRESHOLD = 0.12


def _name_for_id(names: Dict[int, str], cid: int) -> str:
    return str(names.get(cid, names.get(int(cid), str(cid))))


def _labels_dir_for_split(meta: dict, split_key: str, resolve_split_dir) -> Optional[Path]:
    rel = meta.get(split_key)
    if not rel:
        return None
    root = meta.get("path") or ""
    img_dir = resolve_split_dir(root, str(rel))
    img_s = str(img_dir).replace("\\", "/")
    if "/images/" in img_s:
        return Path(img_s.replace("/images/", "/labels/"))
    if img_dir.is_dir():
        sibling = img_dir.parent.parent / "labels" / img_dir.name
        if sibling.is_dir():
            return sibling
    return None


def count_instances_in_label_dir(label_dir: Path) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    if not label_dir.is_dir():
        return counts
    for p in label_dir.rglob("*.txt"):
        if p.name.lower() == "classes.txt":
            continue
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    counts[int(float(parts[0]))] += 1
        except (OSError, ValueError):
            continue
    return dict(counts)


def scan_dataset_class_counts(
    data_yaml: str,
    names: Dict[int, str],
    *,
    load_data_yaml,
    resolve_split_dir,
) -> Dict[str, Any]:
    """Count GT instances per class in train/val/test from data.yaml."""
    out: Dict[str, Any] = {
        "splits": {},
        "per_class": {},
        "total_by_split": {},
    }
    if not data_yaml or not os.path.isfile(data_yaml):
        return out
    meta = load_data_yaml(data_yaml)
    split_counts: Dict[str, Dict[int, int]] = {}
    for key in ("train", "val", "test"):
        lbl_dir = _labels_dir_for_split(meta, key, resolve_split_dir)
        if lbl_dir is None:
            continue
        split_counts[key] = count_instances_in_label_dir(lbl_dir)
        out["total_by_split"][key] = sum(split_counts[key].values())
    out["splits"] = {k: {str(cid): v for cid, v in d.items()} for k, d in split_counts.items()}

    all_ids = set(names.keys())
    for d in split_counts.values():
        all_ids.update(d.keys())
    for cid in sorted(all_ids):
        cn = _name_for_id(names, cid)
        tr = split_counts.get("train", {}).get(cid, 0)
        va = split_counts.get("val", {}).get(cid, 0)
        te = split_counts.get("test", {}).get(cid, 0)
        out["per_class"][cn] = {
            "class_id": cid,
            "train": tr,
            "val": va,
            "test": te,
            "total": tr + va + te,
        }
    return out


def find_confusion_pairs(
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    top_n: int = 12,
    min_rate: float = CONFUSION_RATE_THRESHOLD,
) -> List[dict]:
    """Top predicted≠true class pairs (column-normalized CM)."""
    if matrix is None or matrix.size == 0 or not labels:
        return []
    m = np.asarray(matrix, dtype=float)
    norm = normalize_confusion_matrix(m)
    n = min(len(labels), m.shape[0], m.shape[1])
    labels = list(labels[:n])
    has_bg = n > 0 and labels[-1] == "background"
    nc = n - 1 if has_bg else n
    pairs: List[dict] = []
    for pred_i in range(nc):
        for true_i in range(nc):
            if pred_i == true_i:
                continue
            rate = float(norm[pred_i, true_i])
            cnt = float(m[pred_i, true_i])
            if cnt <= 0:
                continue
            if rate >= min_rate:
                pairs.append(
                    {
                        "predicted": labels[pred_i],
                        "true_class": labels[true_i],
                        "confusion_rate": round(rate, 4),
                        "count": int(cnt),
                    }
                )
    pairs.sort(key=lambda x: (-x["confusion_rate"], -x["count"]))
    return pairs[:top_n]


def _parse_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_training_diagnosis(
    *,
    names: Dict[int, str],
    per_class_rows: List[dict],
    class_stats: Dict[str, dict],
    results: Sequence[Any],
    split_used: str,
    overall: dict,
    dataset_counts: Dict[str, Any],
    confusion_pairs: List[dict],
    eval_image_count: int,
    low_score_count: int,
) -> dict:
    """Build structured diagnosis dict + markdown/html fragments."""
    findings: List[dict] = []
    suggestions: List[str] = []

    low_score_rate = low_score_count / max(eval_image_count, 1)
    map50 = _parse_float(overall.get("mAP50", overall.get("metrics/mAP50(B)", -1)), -1)

    if eval_image_count < 30:
        findings.append(
            {
                "level": "warn",
                "title": "评估样本偏少",
                "detail": f"当前 {split_used} 仅评估 {eval_image_count} 张，结论可能不稳定。",
            }
        )
        suggestions.append(
            "建议划分独立 test 集（Make Datasets 勾选 test）或扩大评估样本量后再做结论。"
        )

    if low_score_rate > 0.5:
        findings.append(
            {
                "level": "error",
                "title": "整体检测质量偏低",
                "detail": f"{low_score_count}/{eval_image_count} 张图像 F1<1（{low_score_rate:.0%}）。",
            }
        )
    elif low_score_rate > 0.25:
        findings.append(
            {
                "level": "warn",
                "title": "存在一定比例困难样本",
                "detail": f"{low_score_count}/{eval_image_count} 张图像存在漏检/误检/类错。",
            }
        )

    if 0 <= map50 < 0.35:
        findings.append(
            {
                "level": "error",
                "title": "mAP50 较低",
                "detail": f"整集 mAP50≈{map50:.3f}，模型尚未学好主要类别。",
            }
        )
        suggestions.append(
            "优先检查数据标注质量与类别定义；延长训练 epoch 或换更大 backbone；"
            "确认 train/val 分布与测试场景一致。"
        )
    elif 0 <= map50 < 0.55:
        findings.append(
            {
                "level": "warn",
                "title": "mAP50 有提升空间",
                "detail": f"整集 mAP50≈{map50:.3f}。",
            }
        )

    # Data sufficiency
    sparse_data: List[dict] = []
    no_train: List[str] = []
    per_cls_ds = dataset_counts.get("per_class") or {}
    for cn, row in sorted(per_cls_ds.items(), key=lambda x: x[1].get("train", 0)):
        tr = int(row.get("train", 0))
        te = int(row.get("test", 0))
        tot = int(row.get("total", 0))
        if tr == 0 and tot > 0:
            no_train.append(cn)
        elif 0 < tr < MIN_INSTANCES_WARN:
            sparse_data.append({"class": cn, "train": tr, "total": tot, "test": te})
        elif MIN_INSTANCES_WARN <= tr < MIN_INSTANCES_RECOMMENDED:
            sparse_data.append({"class": cn, "train": tr, "total": tot, "test": te, "mild": True})

    if no_train:
        findings.append(
            {
                "level": "error",
                "title": "部分类别训练集无样本",
                "detail": "、".join(no_train[:8]) + ("…" if len(no_train) > 8 else ""),
            }
        )
        suggestions.append(
            "训练集缺少上述类别标注：补充图像并重新划分 train/val/test，或从 val/test 挪部分样本到 train。"
        )

    severe_sparse = [x for x in sparse_data if not x.get("mild")]
    if severe_sparse:
        top = severe_sparse[:10]
        detail = "；".join(f"{x['class']}(train={x['train']})" for x in top)
        findings.append(
            {
                "level": "warn",
                "title": "训练样本不足的类群",
                "detail": detail + ("…" if len(severe_sparse) > 10 else ""),
            }
        )
        suggestions.append(
            f"稀有类建议每类至少 {MIN_INSTANCES_RECOMMENDED}+ 个实例："
            "补拍/爬虫增广、Make Datasets 前合并子文件夹、对少数类复制+轻度增强。"
        )

    weak_test = [
        cn for cn, row in per_cls_ds.items()
        if int(row.get("test", 0)) > 0 and int(row.get("test", 0)) < MIN_TEST_INSTANCES
    ]
    if weak_test and split_used == "test":
        findings.append(
            {
                "level": "info",
                "title": "测试集部分类别样本过少",
                "detail": "、".join(weak_test[:10]),
            }
        )

    # Per-class eval performance
    weak_recall: List[dict] = []
    weak_precision: List[dict] = []
    weak_f1: List[dict] = []
    for row in per_class_rows or []:
        cn = str(row.get("Class", ""))
        inst = int(row.get("Instances", 0))
        if inst < 3:
            continue
        p = _parse_float(row.get("Precision", 0))
        r = _parse_float(row.get("Recall", 0))
        f1 = _parse_float(row.get("F1", 0))
        if f1 < LOW_F1_THRESHOLD:
            weak_f1.append({"class": cn, "f1": f1, "instances": inst, "precision": p, "recall": r})
        if r < LOW_RECALL_THRESHOLD:
            weak_recall.append({"class": cn, "recall": r, "instances": inst, "f1": f1})
        if p < LOW_PRECISION_THRESHOLD:
            weak_precision.append({"class": cn, "precision": p, "instances": inst, "f1": f1})

    weak_f1.sort(key=lambda x: (x["f1"], -x["instances"]))
    weak_recall.sort(key=lambda x: (x["recall"], -x["instances"]))
    weak_precision.sort(key=lambda x: (x["precision"], -x["instances"]))

    if weak_f1:
        top = weak_f1[:8]
        findings.append(
            {
                "level": "warn",
                "title": "检测 F1 偏低的类群",
                "detail": "；".join(
                    f"{x['class']}(F1={x['f1']:.2f}, n={x['instances']})" for x in top
                ),
            }
        )

    if weak_recall:
        suggestions.append(
            "漏检偏多类群（Recall 低）：增加遮挡/小目标/远距离样本；检查是否漏标；"
            "可尝试略降推理 conf、提高 imgsz、加强 mosaic/scale 增强。"
        )
    if weak_precision:
        suggestions.append(
            "误检偏多类群（Precision 低）：增加难负样本与背景图；略提高 conf；"
            "审查与相似类的标注边界是否一致。"
        )

    # Image-level bad classes
    image_weak = sorted(
        (
            (cn, st)
            for cn, st in (class_stats or {}).items()
            if cn != "(无标注)" and st.get("count", 0) >= 3
        ),
        key=lambda x: (-x[1].get("bad", 0), x[0]),
    )[:10]
    image_weak_out = []
    for cn, st in image_weak:
        cnt = max(int(st.get("count", 0)), 1)
        avg_f1 = float(st.get("f1_sum", 0)) / cnt
        bad = int(st.get("bad", 0))
        if bad > 0 or avg_f1 < 0.6:
            image_weak_out.append(
                {"class": cn, "images": cnt, "avg_image_f1": round(avg_f1, 3), "bad_images": bad}
            )

    # Confusion
    if confusion_pairs:
        top = confusion_pairs[:6]
        detail = "；".join(
            f"{p['true_class']}→被识为→{p['predicted']}({p['confusion_rate']:.0%})"
            for p in top
        )
        findings.append(
            {
                "level": "warn",
                "title": "类间易混淆（区分度不高）",
                "detail": detail,
            }
        )
        suggestions.append(
            "易混淆类对：增加二者同框/相似姿态对比样本；统一标注规范；"
            "若形态极相似可考虑合并类别或做二阶段细分类。"
        )

    # Error pattern from image eval
    total_fn = sum(getattr(r, "fn", 0) for r in results)
    total_fp = sum(getattr(r, "fp", 0) for r in results)
    total_cls_err = sum(getattr(r, "class_errors", 0) for r in results)
    if total_fn + total_fp > 0:
        fn_ratio = total_fn / (total_fn + total_fp)
        if fn_ratio > 0.65:
            suggestions.append(
                "整体以漏检(FN)为主：补充小目标与困难场景数据，或略降低 conf / 增大输入尺寸。"
            )
        elif fn_ratio < 0.35:
            suggestions.append(
                "整体以误检(FP)为主：增加背景与负样本，训练时提高 conf 阈值筛选自动标注。"
            )
        if total_cls_err > total_fp * 0.3:
            suggestions.append(
                "类间混淆错误占比较高：重点复核易混淆类标注，并参考本报告混淆矩阵。"
            )

    # Deduplicate suggestions
    seen = set()
    uniq_suggestions: List[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            uniq_suggestions.append(s)

    if not uniq_suggestions:
        uniq_suggestions.append(
            "当前指标整体尚可：可导出低分样本集做定向复核，并持续按类补充困难样本。"
        )

    return {
        "summary": {
            "eval_images": eval_image_count,
            "low_score_images": low_score_count,
            "low_score_rate": round(low_score_rate, 4),
            "split": split_used,
            "map50": map50 if map50 >= 0 else None,
            "finding_count": len(findings),
        },
        "findings": findings,
        "data_sufficiency": {
            "sparse_train": severe_sparse,
            "mild_sparse_train": [x for x in sparse_data if x.get("mild")],
            "no_train_classes": no_train,
            "sparse_test_classes": weak_test,
        },
        "weak_classes": {
            "low_f1": weak_f1[:15],
            "low_recall": weak_recall[:15],
            "low_precision": weak_precision[:15],
            "by_image": image_weak_out,
        },
        "confusion_pairs": confusion_pairs,
        "error_totals": {
            "fn": total_fn,
            "fp": total_fp,
            "class_errors": total_cls_err,
        },
        "suggestions": uniq_suggestions,
    }


def diagnosis_to_markdown(diagnosis: dict) -> str:
    lines = ["# FishVision 训练诊断摘要", ""]
    sm = diagnosis.get("summary") or {}
    lines.append(f"- 评估划分: **{sm.get('split', '')}**")
    lines.append(f"- 评估图像: **{sm.get('eval_images', 0)}**")
    lines.append(f"- 低分图像 (F1<1): **{sm.get('low_score_images', 0)}** ({sm.get('low_score_rate', 0):.0%})")
    if sm.get("map50") is not None:
        lines.append(f"- mAP50: **{sm['map50']:.4f}**")
    lines.append("")

    lines.append("## 发现的问题")
    for f in diagnosis.get("findings") or []:
        lines.append(f"- **[{f.get('level', 'info').upper()}]** {f.get('title', '')}: {f.get('detail', '')}")
    lines.append("")

    wp = diagnosis.get("weak_classes") or {}
    if wp.get("low_f1"):
        lines.append("## 薄弱类群 (F1)")
        for x in wp["low_f1"][:10]:
            lines.append(
                f"- {x['class']}: F1={x['f1']:.3f}, P={x.get('precision', 0):.3f}, "
                f"R={x.get('recall', 0):.3f}, instances={x['instances']}"
            )
        lines.append("")

    if diagnosis.get("confusion_pairs"):
        lines.append("## 易混淆类群")
        for p in diagnosis["confusion_pairs"][:10]:
            lines.append(
                f"- GT **{p['true_class']}** 常被预测为 **{p['predicted']}** "
                f"({p['confusion_rate']:.0%}, n={p['count']})"
            )
        lines.append("")

    lines.append("## 改进建议")
    for i, s in enumerate(diagnosis.get("suggestions") or [], 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    return "\n".join(lines)


def render_diagnosis_html(diagnosis: dict) -> str:
    import html as html_lib

    level_style = {
        "error": "background:#fdecea;color:#b71c1c;border-left:4px solid #c62828;",
        "warn": "background:#fff8e1;color:#6d4c00;border-left:4px solid #f9a825;",
        "info": "background:#e8f4fd;color:#0d47a1;border-left:4px solid #1976d2;",
    }

    parts: List[str] = []
    sm = diagnosis.get("summary") or {}
    parts.append("<div class='card'><h2>训练诊断与改进建议</h2>")
    parts.append(
        f"<p>基于 <b>{html_lib.escape(str(sm.get('split', '')))}</b> 集 "
        f"{sm.get('eval_images', 0)} 张评估；"
        f"低分样本 {sm.get('low_score_images', 0)} 张 "
        f"({float(sm.get('low_score_rate', 0)):.0%})。"
    )
    if sm.get("map50") is not None:
        parts.append(f" 整集 mAP50=<b>{sm['map50']:.4f}</b>。")
    parts.append("</p>")

    parts.append("<h3>主要发现</h3><ul class='findings'>")
    for f in diagnosis.get("findings") or []:
        lv = f.get("level", "info")
        style = level_style.get(lv, level_style["info"])
        parts.append(
            f"<li style='{style}padding:8px 12px;margin:8px 0;list-style:none;border-radius:4px;'>"
            f"<b>{html_lib.escape(str(f.get('title', '')))}</b> — "
            f"{html_lib.escape(str(f.get('detail', '')))}</li>"
        )
    if not diagnosis.get("findings"):
        parts.append("<li>未发现显著结构性问题。</li>")
    parts.append("</ul>")

    wp = diagnosis.get("weak_classes") or {}
    ds = diagnosis.get("data_sufficiency") or {}
    if ds.get("sparse_train") or ds.get("no_train_classes"):
        parts.append("<h3>数据量不足</h3><table><tr><th>类别</th><th>train 实例</th><th>合计</th></tr>")
        for x in (ds.get("no_train_classes") or [])[:5]:
            parts.append(
                f"<tr><td>{html_lib.escape(x)}</td><td>0</td><td>—</td></tr>"
            )
        for x in (ds.get("sparse_train") or [])[:12]:
            parts.append(
                f"<tr><td>{html_lib.escape(x['class'])}</td>"
                f"<td>{x['train']}</td><td>{x.get('total', '')}</td></tr>"
            )
        parts.append("</table>")

    if wp.get("low_f1"):
        parts.append(
            "<h3>检测薄弱类群 (按 F1)</h3>"
            "<table><tr><th>类别</th><th>F1</th><th>Precision</th><th>Recall</th><th>Instances</th></tr>"
        )
        for x in wp["low_f1"][:12]:
            parts.append(
                f"<tr><td>{html_lib.escape(x['class'])}</td>"
                f"<td>{x['f1']:.3f}</td><td>{x.get('precision', 0):.3f}</td>"
                f"<td>{x.get('recall', 0):.3f}</td><td>{x['instances']}</td></tr>"
            )
        parts.append("</table>")

    if diagnosis.get("confusion_pairs"):
        parts.append(
            "<h3>易混淆类群（GT 列 → 被预测为）</h3>"
            "<table><tr><th>真实类</th><th>误判为</th><th>混淆比例</th><th>次数</th></tr>"
        )
        for p in diagnosis["confusion_pairs"][:12]:
            parts.append(
                f"<tr><td>{html_lib.escape(p['true_class'])}</td>"
                f"<td>{html_lib.escape(p['predicted'])}</td>"
                f"<td>{p['confusion_rate']:.1%}</td><td>{p['count']}</td></tr>"
            )
        parts.append("</table>")

    parts.append("<h3>如何改进</h3><ol class='suggestions'>")
    for s in diagnosis.get("suggestions") or []:
        parts.append(f"<li>{html_lib.escape(s)}</li>")
    parts.append("</ol>")
    parts.append(
        "<p style='font-size:12px;color:#666;'>详细数据见同目录 "
        "<code>training_diagnosis.json</code> / <code>training_diagnosis.md</code>，"
        "以及 Excel 中 per_class、混淆矩阵表。</p>"
    )
    parts.append("</div>")
    return "".join(parts)


def save_diagnosis_sidecars(out_dir: str, diagnosis: dict) -> Dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "training_diagnosis.json"
    md_path = root / "training_diagnosis.md"
    json_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(diagnosis_to_markdown(diagnosis), encoding="utf-8")
    return {"diagnosis_json": str(json_path), "diagnosis_md": str(md_path)}

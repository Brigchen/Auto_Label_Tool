# -*- coding: utf-8 -*-
"""HTML summary for label self-refine reports (from CSV + optional summary JSON)."""
from __future__ import annotations

import csv
import html
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from libs.test_report import (
    _imwrite,
    _read_image_size,
    draw_comparison,
    load_data_yaml,
    load_gt_boxes,
)

LogFn = Callable[[str], None]

_ACTION_LABELS = {
    "fix_class": "修正类别",
    "drop_low_score": "删除低分标注",
    "drop_unmatched_gt": "删除无匹配 GT",
    "add_pred": "新增预测框",
    "cross_class_dedup": "跨类 dedup",
    "review_class": "待复核：类别不一致",
    "review_fn": "待复核：漏检/难样本",
}

_CLASS_FIX_RE = re.compile(r"^(.+?)\s*->\s*(.+?)\s*\(IoU=", re.I)


@dataclass
class CsvAction:
    image_path: str
    label_path: str
    split: str
    action: str
    detail: str
    old_line: str = ""
    new_line: str = ""


@dataclass
class RefineReportAnalysis:
    actions: List[CsvAction] = field(default_factory=list)
    images_in_csv: int = 0
    images_changed: int = 0
    images_review: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    split_images: Dict[str, int] = field(default_factory=dict)
    split_actions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    class_fix_pairs: Dict[Tuple[str, str], int] = field(default_factory=dict)
    summary_meta: Dict = field(default_factory=dict)
    rules: Dict = field(default_factory=dict)


def load_actions_from_csv(csv_path: str) -> List[CsvAction]:
    csv_path = str(Path(csv_path).resolve())
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Report CSV not found: {csv_path}")
    actions: List[CsvAction] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            actions.append(CsvAction(
                image_path=(row.get("image") or "").strip(),
                label_path=(row.get("label") or "").strip(),
                split=(row.get("split") or "").strip(),
                action=(row.get("action") or "").strip(),
                detail=(row.get("detail") or "").strip(),
                old_line=(row.get("old_line") or "").strip(),
                new_line=(row.get("new_line") or "").strip(),
            ))
    return actions


def _load_summary_json(out_dir: str) -> Dict:
    p = Path(out_dir) / "label_self_refine_summary.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def analyze_refine_report(
    actions: List[CsvAction],
    summary_meta: Optional[Dict] = None,
) -> RefineReportAnalysis:
    summary_meta = summary_meta or {}
    analysis = RefineReportAnalysis(
        actions=actions,
        summary_meta=summary_meta,
        rules=dict(summary_meta.get("rules") or {}),
    )

    images: Set[str] = set()
    changed_images: Set[str] = set()
    review_images: Set[str] = set()
    split_img: Dict[str, Set[str]] = defaultdict(set)

    for a in actions:
        if not a.image_path:
            continue
        images.add(a.image_path)
        split_img[a.split or "unknown"].add(a.image_path)
        analysis.action_counts[a.action] = analysis.action_counts.get(a.action, 0) + 1
        sp = a.split or "unknown"
        analysis.split_actions.setdefault(sp, {})
        analysis.split_actions[sp][a.action] = analysis.split_actions[sp].get(a.action, 0) + 1

        if a.action.startswith("review"):
            review_images.add(a.image_path)
        if a.action in (
            "fix_class", "drop_low_score", "drop_unmatched_gt",
            "add_pred", "cross_class_dedup",
        ):
            changed_images.add(a.image_path)

        if a.action == "fix_class":
            m = _CLASS_FIX_RE.match(a.detail)
            if m:
                pair = (m.group(1).strip(), m.group(2).strip())
                analysis.class_fix_pairs[pair] = analysis.class_fix_pairs.get(pair, 0) + 1

    analysis.images_in_csv = len(images)
    analysis.images_review = len(review_images)
    analysis.images_changed = len(changed_images)
    analysis.split_images = {k: len(v) for k, v in split_img.items()}

    if summary_meta.get("images"):
        pass  # prefer summary totals in HTML
    return analysis


def _action_label(action: str) -> str:
    return _ACTION_LABELS.get(action, action)


def _pick_samples(
    actions: List[CsvAction],
    action_prefix: str,
    limit: int,
) -> List[CsvAction]:
    seen: Set[str] = set()
    out: List[CsvAction] = []
    for a in actions:
        if action_prefix.endswith("*"):
            ok = a.action.startswith(action_prefix[:-1])
        else:
            ok = a.action == action_prefix
        if not ok or not a.image_path or a.image_path in seen:
            continue
        seen.add(a.image_path)
        out.append(a)
        if len(out) >= limit:
            break
    return out


def _render_comparison_preview(
    action: CsvAction,
    names: Dict[int, str],
    preview_dir: Path,
    used_names: Set[str],
    preds: Optional[List] = None,
) -> str:
    """Draw GT + pred boxes; return relative path for HTML."""
    if not action.image_path or not os.path.isfile(action.image_path):
        return ""
    stem = Path(action.image_path).stem
    base = stem
    n = 2
    while stem.lower() in used_names:
        stem = f"{base}_{n}"
        n += 1
    used_names.add(stem.lower())

    w, h = _read_image_size(action.image_path)
    gts = load_gt_boxes(action.label_path, names, w, h) if w > 0 else []
    pred_list = preds or []
    title = f"{_action_label(action.action)}"
    try:
        img = draw_comparison(action.image_path, gts, pred_list, title=title[:80])
    except Exception:
        return ""
    rel_name = f"{stem}.jpg"
    out_path = preview_dir / rel_name
    _imwrite(str(out_path), img)
    return f"html_preview/{rel_name}"


def _render_gt_preview(
    action: CsvAction,
    names: Dict[int, str],
    preview_dir: Path,
    used_names: Set[str],
) -> str:
    """Draw GT boxes only; return relative path for HTML or empty."""
    if not action.image_path or not os.path.isfile(action.image_path):
        return ""
    stem = Path(action.image_path).stem
    base = stem
    n = 2
    while stem.lower() in used_names:
        stem = f"{base}_{n}"
        n += 1
    used_names.add(stem.lower())

    w, h = _read_image_size(action.image_path)
    gts = []
    if action.label_path and os.path.isfile(action.label_path) and w > 0:
        gts = load_gt_boxes(action.label_path, names, w, h)

    title = f"{_action_label(action.action)}"
    try:
        img = draw_comparison(action.image_path, gts, [], title=title[:80])
    except Exception:
        return ""
    rel_name = f"{stem}.jpg"
    out_path = preview_dir / rel_name
    _imwrite(str(out_path), img)
    return f"html_preview/{rel_name}"


def write_refine_html_report(
    out_dir: str,
    *,
    csv_path: Optional[str] = None,
    data_yaml: str = "",
    weights: str = "",
    dry_run: Optional[bool] = None,
    max_preview: int = 36,
    log: LogFn = print,
) -> str:
    """Build index.html from label_self_refine_report.csv. Returns html path."""
    out_dir = str(Path(out_dir).resolve())
    csv_path = csv_path or os.path.join(out_dir, "label_self_refine_report.csv")
    actions = load_actions_from_csv(csv_path)
    summary_meta = _load_summary_json(out_dir)
    analysis = analyze_refine_report(actions, summary_meta)

    names: Dict[int, str] = {}
    if data_yaml and os.path.isfile(data_yaml):
        try:
            names = load_data_yaml(data_yaml).get("names") or {}
        except Exception:
            names = {}

    preview_dir = Path(out_dir) / "html_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    used_preview: Set[str] = set()

    def _samples_html(sample_actions: List[CsvAction], limit: int) -> str:
        rows = []
        for i, a in enumerate(sample_actions[:limit], start=1):
            rel = _render_gt_preview(a, names, preview_dir, used_preview)
            if rel:
                vis = (
                    f"<a href='{html.escape(rel)}' target='_blank'>"
                    f"<img src='{html.escape(rel)}' width='220'/></a>"
                )
            else:
                vis = "<span style='color:#888;'>—</span>"
            rows.append(
                f"<tr><td>{i}</td><td>{vis}</td>"
                f"<td>{html.escape(os.path.basename(a.image_path))}</td>"
                f"<td>{html.escape(a.split)}</td>"
                f"<td>{html.escape(_action_label(a.action))}</td>"
                f"<td>{html.escape(a.detail)}</td></tr>"
            )
        return "".join(rows) if rows else "<tr><td colspan='6'>无</td></tr>"

    review_samples = _pick_samples(actions, "review", max_preview)
    fix_samples = _pick_samples(actions, "fix_class", min(24, max_preview))
    drop_samples = _pick_samples(actions, "drop_low_score", min(12, max_preview))

    total_images = int(summary_meta.get("images") or analysis.images_in_csv)
    labels_changed = int(summary_meta.get("labels_changed") or analysis.images_changed)
    review_n = int(summary_meta.get("review_images") or analysis.images_review)
    lines_dropped = int(summary_meta.get("lines_dropped") or 0)
    lines_class_fixed = int(summary_meta.get("lines_class_fixed") or 0)
    lines_added = int(summary_meta.get("lines_added") or 0)

    if dry_run is None:
        dry_run = True

    action_rows = []
    for act, cnt in sorted(analysis.action_counts.items(), key=lambda x: (-x[1], x[0])):
        action_rows.append(
            f"<tr><td>{html.escape(_action_label(act))}</td>"
            f"<td><code>{html.escape(act)}</code></td><td>{cnt}</td></tr>"
        )

    split_rows = []
    for sp in sorted(analysis.split_images.keys()):
        acts = analysis.split_actions.get(sp, {})
        act_txt = ", ".join(
            f"{_action_label(k)}={v}" for k, v in sorted(acts.items(), key=lambda x: -x[1])
        )
        split_rows.append(
            f"<tr><td>{html.escape(sp)}</td><td>{analysis.split_images[sp]}</td>"
            f"<td>{html.escape(act_txt or '—')}</td></tr>"
        )

    pair_rows = []
    for (src, dst), cnt in sorted(
        analysis.class_fix_pairs.items(), key=lambda x: (-x[1], x[0][0]),
    ):
        pair_rows.append(
            f"<tr><td>{html.escape(src)}</td><td>{html.escape(dst)}</td><td>{cnt}</td></tr>"
        )

    rules = analysis.rules
    rules_txt = ""
    if rules:
        rules_txt = "<ul>" + "".join(
            f"<li><code>{html.escape(str(k))}</code> = {html.escape(str(v))}</li>"
            for k, v in rules.items()
        ) + "</ul>"

    review_queue = os.path.join(out_dir, "review_queue")
    has_review = os.path.isdir(review_queue)

    findings = []
    if review_n > 0:
        pct = 100.0 * review_n / max(total_images, 1)
        findings.append(
            f"约 <b>{review_n}</b> 张（{pct:.1f}%）建议人工复核，"
            "主要为类别不一致或未匹配预测。"
        )
    if lines_class_fixed > 0:
        findings.append(
            f"模型高置信度建议修正类别 <b>{lines_class_fixed}</b> 处"
            f"（涉及约 {analysis.action_counts.get('fix_class', 0)} 条 action 记录）。"
        )
    if lines_dropped > 0:
        findings.append(f"低 score 标注行建议删除 <b>{lines_dropped}</b> 处。")
    if labels_changed == 0 and not review_n:
        findings.append("未发现需修改或复核项；标注与模型预测整体一致。")
    if dry_run:
        findings.append(
            "<b>dry-run</b>：以下为建议变更，尚未写入磁盘。"
            "取消 dry-run 后将备份 labels/ 再应用。"
        )

    findings_html = (
        "<ul class='findings'>" + "".join(f"<li>{x}</li>" for x in findings) + "</ul>"
        if findings else "<p>无</p>"
    )

    index_path = os.path.join(out_dir, "index.html")
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>标注自修正报告</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; background: #f5f5f5; }}
h1,h2 {{ color: #333; }}
.card {{ background: #fff; padding: 16px 20px; margin: 16px 0; border-radius: 8px;
         box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.metric {{ flex: 1 1 140px; background: #fafafa; border: 1px solid #eee;
           border-radius: 6px; padding: 12px; text-align: center; }}
.metric b {{ display: block; font-size: 22px; color: #1565c0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #eee; }}
.legend {{ font-size: 13px; color: #555; }}
.gt {{ color: #0a0; font-weight: bold; }}
.findings li {{ line-height: 1.55; }}
code {{ font-size: 12px; word-break: break-all; }}
</style></head><body>
<h1>标注自修正报告 (Label Self-Refine)</h1>
<div class="card">
<p>权重: <code>{html.escape(weights or '—')}</code></p>
<p>数据集: <code>{html.escape(data_yaml or '—')}</code></p>
<p>报告目录: <code>{html.escape(out_dir)}</code></p>
<p>CSV: <code>{html.escape(os.path.basename(csv_path))}</code>
 | JSON: <code>label_self_refine_summary.json</code>
{" | review_queue: 已导出" if has_review else ""}</p>
<div class="metrics">
  <div class="metric"><b>{total_images}</b>评估图像</div>
  <div class="metric"><b>{labels_changed}</b>将改/已改文件</div>
  <div class="metric"><b>{review_n}</b>待复核图像</div>
  <div class="metric"><b>{lines_class_fixed}</b>改类行数</div>
  <div class="metric"><b>{lines_dropped}</b>删除行数</div>
  <div class="metric"><b>{lines_added}</b>新增行数</div>
</div>
<p class="legend"><span class="gt">■ 绿框 = GT</span>
<span class="pred" style="color:#c00;font-weight:bold">■ 红框 = 预测</span>
 | 交互复核请运行 FishVision/ALT「Open Interactive Review」打开 <code>review.html</code></p>
</div>
<div class="card"><h2>诊断摘要</h2>{findings_html}</div>
<div class="card"><h2>动作统计</h2>
<table><tr><th>说明</th><th>action</th><th>次数</th></tr>
{"".join(action_rows) if action_rows else "<tr><td colspan='3'>无 action 记录（标注与预测完全一致）</td></tr>"}
</table></div>
<div class="card"><h2>划分 (split)</h2>
<table><tr><th>split</th><th>涉及图像数</th><th>动作分布</th></tr>
{"".join(split_rows) if split_rows else "<tr><td colspan='3'>无</td></tr>"}
</table></div>
<div class="card"><h2>类别修正对 (GT → 预测)</h2>
<table><tr><th>原类别</th><th>建议类别</th><th>次数</th></tr>
{"".join(pair_rows) if pair_rows else "<tr><td colspan='3'>无</td></tr>"}
</table></div>
{"<div class='card'><h2>规则参数</h2>" + rules_txt + "</div>" if rules_txt else ""}
<div class="card"><h2>待复核样本预览</h2>
<table><tr><th>#</th><th>预览</th><th>文件</th><th>split</th><th>类型</th><th>详情</th></tr>
{_samples_html(review_samples, max_preview)}
</table></div>
<div class="card"><h2>类别修正样本预览</h2>
<table><tr><th>#</th><th>预览</th><th>文件</th><th>split</th><th>类型</th><th>详情</th></tr>
{_samples_html(fix_samples, min(24, max_preview))}
</table></div>
<div class="card"><h2>低分删除样本预览</h2>
<table><tr><th>#</th><th>预览</th><th>文件</th><th>split</th><th>类型</th><th>详情</th></tr>
{_samples_html(drop_samples, min(12, max_preview))}
</table></div>
</body></html>"""

    Path(index_path).write_text(doc, encoding="utf-8")
    log(f"HTML 报告: {index_path}")
    return index_path


def run_refine_html_from_csv(
    out_dir: str,
    *,
    data_yaml: str = "",
    weights: str = "",
    csv_path: Optional[str] = None,
    log: LogFn = print,
) -> str:
    """Standalone: build index.html from existing CSV (no inference)."""
    return write_refine_html_report(
        out_dir,
        csv_path=csv_path,
        data_yaml=data_yaml,
        weights=weights,
        log=log,
    )

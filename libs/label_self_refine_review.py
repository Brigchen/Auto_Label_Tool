# -*- coding: utf-8 -*-
"""Interactive HTML review + local server to confirm and apply label fixes."""
from __future__ import annotations

import json
import os
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

from libs.label_self_refine_apply import apply_line_change, is_applicable_action, reset_backup_cache
from libs.label_self_refine_report import load_actions_from_csv
from libs.test_report import (
    Box,
    _imwrite,
    _read_image_size,
    draw_comparison,
    load_data_yaml,
    load_gt_boxes,
    match_boxes,
    predict_all_boxes,
)

LogFn = Callable[[str], None]

MANIFEST_NAME = "label_self_refine_review.json"
REVIEW_HTML = "review.html"
_CLASS_FIX_RE = re.compile(r"^(.+?)\s*->\s*(.+?)\s*\(IoU=", re.I)


def _box_to_dict(b: Box) -> dict:
    return {
        "cls": int(b.cls),
        "name": str(b.name),
        "xyxy": [float(x) for x in b.xyxy],
        "conf": float(b.conf),
    }


def _line_for_box(b: Box, img_w: int, img_h: int) -> str:
    from libs.test_report import boxes_to_yolo_text
    return boxes_to_yolo_text([b], img_w, img_h, save_score=True).strip()


def _highlight_boxes(
    image_path: str,
    gts: List[Box],
    preds: List[Box],
    *,
    highlight_gt: Optional[Box] = None,
    highlight_pred: Optional[Box] = None,
) -> Optional[str]:
    if not os.path.isfile(image_path):
        return None
    hg = [highlight_gt] if highlight_gt else []
    hp = [highlight_pred] if highlight_pred else (preds if not highlight_gt else [])
    if highlight_gt and preds:
        hp = preds
    try:
        img = draw_comparison(image_path, gts if not highlight_gt else gts, hp, title="")
        return _encode_preview(img, image_path)
    except Exception:
        return None


def _encode_preview(img, image_path: str) -> str:
    import base64
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build_review_manifest(
    out_dir: str,
    *,
    data_yaml: str = "",
    weights: str = "",
    csv_path: Optional[str] = None,
    pred_map: Optional[Dict[str, List[Box]]] = None,
    max_previews: int = 0,  # 0 means unlimited (all applicable items get preview)
    log: LogFn = print,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Build or refresh interactive review manifest (GT+pred, full new_line).
    
    Args:
        max_previews: Maximum number of preview images to generate.
            0 or negative means unlimited (generate preview for ALL applicable items).
            This is important for interactive review - users need to see all samples.
    """
    out_dir = str(Path(out_dir).resolve())
    csv_path = csv_path or os.path.join(out_dir, "label_self_refine_report.csv")
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)

    existing: Dict[str, Any] = {}
    if os.path.isfile(manifest_path):
        try:
            existing = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    status_by_id = {
        int(it.get("id", -1)): str(it.get("status", "pending"))
        for it in (existing.get("items") or [])
        if it.get("id") is not None
    }

    names: Dict[int, str] = {}
    if data_yaml and os.path.isfile(data_yaml):
        names = load_data_yaml(data_yaml).get("names") or {}

    actions = load_actions_from_csv(csv_path)
    candidates = [a for a in actions if is_applicable_action(a.action) or a.action.startswith("review")]

    pred_cache: Dict[str, List[Box]] = dict(pred_map or {})
    if not pred_cache:
        need_predict = weights and os.path.isfile(weights)
        if need_predict:
            unique_imgs = sorted({
                a.image_path for a in candidates
                if a.image_path and os.path.isfile(a.image_path)
            })
            if unique_imgs:
                log(f"加载预测框用于对比图 ({len(unique_imgs)} 张)…")
                from ultralytics import YOLO
                model = YOLO(weights)
                pred_cache = predict_all_boxes(
                    model, unique_imgs, names, conf=0.25, batch=32, device="0",
                    progress=progress,
                )
    else:
        log(f"复用已有预测结果 ({len(pred_cache)} 张)")

    items: List[Dict[str, Any]] = []
    preview_dir = Path(out_dir) / "html_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_count = 0
    # 0 or negative means unlimited previews for interactive review
    preview_cap = max_previews if max_previews > 0 else 999999

    for idx, a in enumerate(candidates):
        if progress and idx % 50 == 0:
            progress(idx, len(candidates), os.path.basename(a.image_path or ""))

        applicable = is_applicable_action(a.action)
        new_line = (a.new_line or "").strip()
        old_line = (a.old_line or "").strip()

        img_w, img_h = _read_image_size(a.image_path) if a.image_path else (0, 0)
        gts = load_gt_boxes(a.label_path, names, img_w, img_h) if img_w > 0 else []
        preds = pred_cache.get(str(Path(a.image_path).resolve()), []) if a.image_path else []

        if applicable and (not new_line or new_line.endswith("...")) and img_w > 0:
            if a.action in ("fix_class", "review_class") and old_line and preds and gts:
                from libs.test_report import boxes_to_yolo_text, match_boxes
                old_parts = old_line.split()
                matched_gt = None
                for gt in gts:
                    gt_ln = boxes_to_yolo_text([gt], img_w, img_h, save_score=True).strip()
                    if gt_ln.split()[:5] == old_parts[:5]:
                        matched_gt = gt
                        break
                if matched_gt is None and len(old_parts) >= 5:
                    for gt in gts:
                        if str(int(gt.cls)) == old_parts[0]:
                            matched_gt = gt
                            break
                if matched_gt is not None:
                    mlist, _, _ = match_boxes([matched_gt], preds, 0.5)
                    if mlist:
                        pred = preds[mlist[0][1]]
                        nb = Box(
                            cls=pred.cls, name=pred.name,
                            xyxy=matched_gt.xyxy, conf=pred.conf,
                        )
                        new_line = boxes_to_yolo_text([nb], img_w, img_h, save_score=True).strip()
            elif a.action == "add_pred" and preds:
                new_line = _line_for_box(preds[0], img_w, img_h)

        # Generate preview for ALL applicable items (interactive review needs full preview)
        preview_rel = ""
        want_preview = (
            applicable
            and preview_count < preview_cap
            and a.image_path
            and os.path.isfile(a.image_path)
        )
        if want_preview:
            try:
                img = draw_comparison(
                    a.image_path, gts, preds,
                    title=f"{a.action}",
                )
                pname = f"review_{idx:05d}.jpg"
                _imwrite(str(preview_dir / pname), img)
                preview_rel = f"html_preview/{pname}"
                preview_count += 1
            except Exception:
                preview_rel = ""

        item = {
            "id": idx,
            "image_path": a.image_path,
            "label_path": a.label_path,
            "split": a.split,
            "action": a.action,
            "detail": a.detail,
            "old_line": old_line,
            "new_line": new_line,
            "applicable": applicable and bool(old_line or a.action == "add_pred"),
            "status": status_by_id.get(idx, "pending"),
            "preview_rel": preview_rel,
        }
        items.append(item)

    if progress:
        progress(len(candidates), max(len(candidates), 1), "done")

    manifest = {
        "out_dir": out_dir,
        "data_yaml": data_yaml,
        "weights": weights,
        "names": {str(k): v for k, v in names.items()},
        "items": items,
        "preview_count": preview_count,  # Track how many previews generated
    }
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log(f"复核清单: {manifest_path} ({len(items)} 条, {preview_count} 张预览)")
    return manifest


def _write_review_html(out_dir: str) -> str:
    html_path = os.path.join(out_dir, REVIEW_HTML)
    doc = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>标注自修正 · 交互复核</title>
<style>
body { font-family: "Microsoft YaHei", sans-serif; margin: 16px; background: #f0f2f5; }
h1 { margin: 0 0 8px; }
.toolbar { background: #fff; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08); display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
button { cursor: pointer; padding: 6px 14px; border-radius: 4px; border: 1px solid #ccc; background: #fff; }
button.primary { background: #1565c0; color: #fff; border-color: #1565c0; }
button:disabled { opacity: .5; cursor: not-allowed; }
.legend { font-size: 13px; color: #555; }
.legend .gt { color: #0a0; font-weight: bold; }
.legend .pred { color: #c00; font-weight: bold; }
#status { font-size: 13px; color: #333; }
.card { background: #fff; border-radius: 8px; margin-bottom: 12px; padding: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,.06); display: grid; grid-template-columns: 390px 1fr; gap: 16px; }
.card.applied { opacity: .65; background: #f1f8e9; }
.card.skipped { opacity: .55; background: #fafafa; }
.preview-wrap { position: relative; }
.card img.preview-thumb {
  width: 100%; max-width: 390px; border-radius: 4px; border: 1px solid #ddd;
  cursor: zoom-in; display: block; transition: box-shadow .15s, transform .15s;
}
.card img.preview-thumb:hover {
  box-shadow: 0 4px 14px rgba(0,0,0,.18); transform: scale(1.01);
}
#lightbox {
  display: none; position: fixed; inset: 0; z-index: 10000;
  background: rgba(0,0,0,.88); align-items: center; justify-content: center;
  padding: 24px; box-sizing: border-box;
}
#lightbox.open { display: flex; }
#lightbox img {
  max-width: 96vw; max-height: 92vh; object-fit: contain;
  border-radius: 6px; box-shadow: 0 8px 32px rgba(0,0,0,.55);
  cursor: zoom-out; background: #111;
}
#lightbox .lb-close {
  position: fixed; top: 16px; right: 20px; z-index: 10001;
  width: 40px; height: 40px; border: none; border-radius: 50%;
  background: rgba(255,255,255,.92); color: #222; font-size: 22px; line-height: 1;
  cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,.25);
}
#lightbox .lb-close:hover { background: #fff; }
#lightbox .lb-hint {
  position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
  color: rgba(255,255,255,.75); font-size: 13px; pointer-events: none;
}
.meta { font-size: 13px; line-height: 1.5; }
.meta code { display: block; background: #f5f5f5; padding: 4px 6px; margin: 4px 0;
  word-break: break-all; font-size: 11px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; background: #e3f2fd; }
.badge.applied { background: #c8e6c9; }
.badge.skipped { background: #eee; }
.filters label { margin-right: 12px; font-size: 13px; }
</style></head><body>
<h1>标注自修正 · 交互复核</h1>
<p class="legend"><span class="gt">■ 绿框 = GT</span> &nbsp; <span class="pred">■ 红框 = 预测</span>
 &nbsp; 点击「确认修正」将写回对应 labels/*.txt（自动备份到 .label_refine_backup/）</p>
<div class="toolbar">
  <span id="counts"></span>
  <button class="primary" id="btn-refresh">刷新列表</button>
  <button id="btn-apply-all">确认全部待处理 (fix_class)</button>
  <span class="filters">
    <label><input type="checkbox" id="f-fix" checked> fix_class</label>
    <label><input type="checkbox" id="f-drop" checked> drop_*</label>
    <label><input type="checkbox" id="f-review" checked> review_class</label>
    <label><input type="checkbox" id="f-hide-done" checked> 隐藏已处理</label>
  </span>
  <span id="status"></span>
</div>
<div id="list"></div>
<div id="lightbox" role="dialog" aria-label="放大预览" aria-modal="true">
  <button type="button" class="lb-close" title="关闭 (Esc)">&times;</button>
  <img id="lightbox-img" alt="放大预览"/>
  <div class="lb-hint">点击空白处或按 Esc 关闭</div>
</div>
<script>
let manifest = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function openLightbox(src) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lightbox-img').src = src;
  lb.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.getElementById('lightbox-img').removeAttribute('src');
  document.body.style.overflow = '';
}
document.getElementById('lightbox').addEventListener('click', (e) => {
  if (e.target.id === 'lightbox' || e.target.id === 'lightbox-img') closeLightbox();
});
document.querySelector('#lightbox .lb-close').addEventListener('click', (e) => {
  e.stopPropagation(); closeLightbox();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

function actionLabel(a) {
  const m = {fix_class:'修正类别',drop_low_score:'删除低分',drop_unmatched_gt:'删除无匹配',
    add_pred:'新增框',review_class:'复核改类',review_fn:'复核漏检'};
  return m[a] || a;
}
function filterItems(items) {
  const showFix = document.getElementById('f-fix').checked;
  const showDrop = document.getElementById('f-drop').checked;
  const showReview = document.getElementById('f-review').checked;
  const hideDone = document.getElementById('f-hide-done').checked;
  return items.filter(it => {
    if (hideDone && it.status !== 'pending') return false;
    if (!it.applicable) return false;
    if (it.action === 'fix_class' || it.action === 'add_pred') return showFix;
    if (it.action.startsWith('drop')) return showDrop;
    if (it.action.startsWith('review')) return showReview;
    return true;
  });
}
function render() {
  if (!manifest) return;
  const items = filterItems(manifest.items || []);
  const pending = (manifest.items||[]).filter(x => x.status==='pending' && x.applicable).length;
  const applied = (manifest.items||[]).filter(x => x.status==='applied').length;
  const totalPreview = (manifest.items||[]).filter(x => x.preview_rel).length;
  document.getElementById('counts').textContent =
    `共 ${manifest.items.length} 条 (${totalPreview} 预览) | 待处理 ${pending} | 已应用 ${applied}`;
  const list = document.getElementById('list');
  list.innerHTML = '';
  items.forEach(it => {
    const div = document.createElement('div');
    div.className = 'card' + (it.status==='applied'?' applied': it.status==='skipped'?' skipped':'');
    const img = it.preview_rel
      ? `<div class="preview-wrap"><img class="preview-thumb" src="/${it.preview_rel}" alt="preview" title="点击放大查看"/></div>`
      : '<div style="color:#888;padding:40px 0;text-align:center">无预览</div>';
    const canApply = it.status === 'pending' && it.applicable;
    div.innerHTML = `
      <div>${img}</div>
      <div class="meta">
        <div><span class="badge ${it.status}">${it.status}</span>
          <b>${actionLabel(it.action)}</b> · ${it.split} · #${it.id}</div>
        <div>${it.detail || ''}</div>
        <div><small>${(it.image_path||'').split(/[/\\\\]/).pop()}</small></div>
        ${it.old_line ? '<div>旧: <code>'+esc(it.old_line)+'</code></div>' : ''}
        ${it.new_line ? '<div>新: <code>'+esc(it.new_line)+'</code></div>' : ''}
        <div style="margin-top:8px">
          <button class="primary" data-apply="${it.id}" ${canApply?'':'disabled'}>确认修正</button>
          <button data-skip="${it.id}" ${it.status==='pending'?'':'disabled'}>跳过</button>
        </div>
      </div>`;
    list.appendChild(div);
  });
  list.querySelectorAll('[data-apply]').forEach(btn => {
    btn.onclick = () => applyOne(btn.dataset.apply);
  });
  list.querySelectorAll('[data-skip]').forEach(btn => {
    btn.onclick = () => skipOne(btn.dataset.skip);
  });
  list.querySelectorAll('img.preview-thumb').forEach(el => {
    el.onclick = (e) => { e.stopPropagation(); openLightbox(el.src); };
  });
}
function esc(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;'); }
async function load() {
  try {
    manifest = await api('/api/manifest');
  } catch(e) {
    document.getElementById('status').textContent = '加载失败: ' + e.message;
    return;
  }
  render();
}
async function applyOne(id) {
  setStatus('应用中…');
  try {
    manifest = await api('/api/apply/' + id, {method:'POST'});
    setStatus('已应用 #' + id);
    render();
  } catch(e) { setStatus('失败: ' + e.message); }
}
async function skipOne(id) {
  manifest = await api('/api/skip/' + id, {method:'POST'});
  render();
}
async function applyAllFix() {
  setStatus('批量应用中…');
  try {
    manifest = await api('/api/apply-batch', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({actions:['fix_class']})
    });
    setStatus('批量完成');
    render();
  } catch(e) { setStatus('失败: ' + e.message); }
}
function setStatus(t) { document.getElementById('status').textContent = t; }
document.getElementById('btn-refresh').onclick = load;
document.getElementById('btn-apply-all').onclick = applyAllFix;
['f-fix','f-drop','f-review','f-hide-done'].forEach(id => {
  document.getElementById(id).onchange = render;
});
load();
</script></body></html>"""
    Path(html_path).write_text(doc, encoding="utf-8")
    return html_path


class RefineReviewServer:
    """Serve interactive review UI and apply API on localhost."""

    def __init__(self, out_dir: str, port: int = 0):
        self.out_dir = str(Path(out_dir).resolve())
        self.port = port or 8765
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _load_manifest(self) -> Dict[str, Any]:
        p = Path(self.out_dir) / MANIFEST_NAME
        if not p.is_file():
            raise FileNotFoundError(f"Missing {MANIFEST_NAME}")
        return json.loads(p.read_text(encoding="utf-8"))

    def _save_manifest(self, data: Dict[str, Any]) -> None:
        Path(self.out_dir, MANIFEST_NAME).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def _find_item(self, data: Dict[str, Any], item_id: int) -> Optional[Dict[str, Any]]:
        for it in data.get("items") or []:
            if int(it.get("id", -1)) == item_id:
                return it
        return None

    def _apply_item(self, item: Dict[str, Any]) -> None:
        ok = apply_line_change(
            item.get("label_path") or "",
            item.get("action") or "",
            item.get("old_line") or "",
            item.get("new_line") or "",
        )
        if not ok:
            raise RuntimeError(f"apply failed for item {item.get('id')}")

    def _make_handler(self):
        out_dir = self.out_dir
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _json(self, code: int, payload: Any) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _file(self, path: Path, content_type: str) -> None:
                if not path.is_file():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                if path in ("/", "/review.html"):
                    return self._file(Path(out_dir) / REVIEW_HTML, "text/html; charset=utf-8")
                if path == "/api/manifest":
                    try:
                        return self._json(200, server._load_manifest())
                    except Exception as exc:
                        return self._json(500, {"error": str(exc)})
                if path.startswith("/html_preview/"):
                    rel = path.lstrip("/")
                    return self._file(Path(out_dir) / rel, "image/jpeg")
                self.send_error(404)

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    if path.startswith("/api/apply/"):
                        iid = int(path.split("/")[-1])
                        data = server._load_manifest()
                        item = server._find_item(data, iid)
                        if not item:
                            return self._json(404, {"error": "not found"})
                        server._apply_item(item)
                        item["status"] = "applied"
                        server._save_manifest(data)
                        return self._json(200, data)
                    if path.startswith("/api/skip/"):
                        iid = int(path.split("/")[-1])
                        data = server._load_manifest()
                        item = server._find_item(data, iid)
                        if item:
                            item["status"] = "skipped"
                            server._save_manifest(data)
                        return self._json(200, data)
                    if path == "/api/apply-batch":
                        length = int(self.headers.get("Content-Length", 0))
                        body = self.rfile.read(length) if length else b"{}"
                        req = json.loads(body.decode("utf-8") or "{}")
                        actions = set(req.get("actions") or ["fix_class"])
                        data = server._load_manifest()
                        for item in data.get("items") or []:
                            if item.get("status") != "pending":
                                continue
                            if item.get("action") not in actions:
                                continue
                            if not item.get("applicable"):
                                continue
                            try:
                                server._apply_item(item)
                                item["status"] = "applied"
                            except Exception:
                                pass
                        server._save_manifest(data)
                        return self._json(200, data)
                    self.send_error(404)
                except Exception as exc:
                    self._json(500, {"error": str(exc)})

        return Handler

    def start(self, open_browser: bool = True) -> str:
        reset_backup_cache()
        _write_review_html(self.out_dir)
        handler = self._make_handler()
        for port in range(self.port, self.port + 20):
            try:
                self._httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
                self.port = port
                break
            except OSError:
                continue
        if self._httpd is None:
            raise RuntimeError("无法绑定本地端口")
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://127.0.0.1:{self.port}/review.html"
        if open_browser:
            webbrowser.open(url)
        return url

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd = None


def open_interactive_review(
    out_dir: str,
    *,
    data_yaml: str = "",
    weights: str = "",
    rebuild_manifest: bool = True,
    log: LogFn = print,
) -> str:
    """Build manifest if needed and open browser review UI."""
    out_dir = str(Path(out_dir).resolve())
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    if rebuild_manifest or not os.path.isfile(manifest_path):
        build_review_manifest(out_dir, data_yaml=data_yaml, weights=weights, log=log)
    server = RefineReviewServer(out_dir)
    url = server.start(open_browser=True)
    log(f"交互复核: {url}")
    return url

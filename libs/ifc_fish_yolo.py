# -*- coding: utf-8 -*-
"""IFC fish dataset: fish_yolo.yaml (pose metadata) discovery and parsing."""
import os
import re


def discover_fish_yolo_yaml(labels_dir):
    """Search labels_dir, its parent, and grandparents for fish_yolo.yaml."""
    if not labels_dir:
        return None
    cur = os.path.abspath(labels_dir)
    for _ in range(5):
        candidate = os.path.join(cur, 'fish_yolo.yaml')
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _parse_kpt_shape(text):
    m = re.search(r'kpt_shape:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if not m:
        return 0, 3
    return int(m.group(1)), int(m.group(2))


def _parse_flip_idx(text):
    m = re.search(r'flip_idx:\s*\[(.*?)\]', text, re.S)
    if not m:
        return []
    out = []
    for part in m.group(1).replace('\n', ' ').split(','):
        part = part.strip()
        if not part or part.startswith('#'):
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _parse_names_from_text(text):
    names = {}
    for m in re.finditer(r'^\s*(\d+)\s*:\s*([^\s#]+)', text, re.M):
        try:
            names[int(m.group(1))] = m.group(2).strip()
        except ValueError:
            continue
    if not names:
        return []
    return [names[i] for i in sorted(names.keys())]


def _parse_kpt_names_regex(text):
    """Fallback: lines like '    - name' after '  0:' under kpt_names."""
    lines = text.splitlines()
    idx = None
    for i, ln in enumerate(lines):
        if re.match(r'^kpt_names\s*:', ln.strip()):
            idx = i
            break
    if idx is None:
        return []
    sub = '\n'.join(lines[idx + 1:])
    m = re.search(r'^\s*0\s*:\s*\n((?:^\s*-\s*.+\n?)+)', sub, re.M)
    if not m:
        return []
    names = []
    for ln in m.group(1).splitlines():
        mm = re.match(r'^\s*-\s*(.+)$', ln)
        if mm:
            item = mm.group(1).split('#')[0].strip().strip("'\"")
            if item:
                names.append(item)
    return names


def default_skeleton_chain(n):
    if n <= 1:
        return []
    return [(i, i + 1) for i in range(n - 1)]


def normalize_skeleton_edges(raw, n_kpts):
    """
    Parse dataset YAML `skeleton` into 0-based (i, j) pairs.

    Supported shapes:
      - [[0, 1], [1, 2], ...]   # list of edges (Ultralytics / common style)
      - [0, 1, 1, 2, 2, 3]      # flat a,b,a,b,... (COCO-style flattened)

    Invalid indices or self-loops are dropped. n_kpts is the keypoint count.
    """
    if raw is None or n_kpts < 1:
        return []
    edges = []
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (int, float)):
        for k in range(0, len(raw) - 1, 2):
            try:
                edges.append((int(raw[k]), int(raw[k + 1])))
            except (TypeError, ValueError, IndexError):
                continue
    else:
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    edges.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
    seen = set()
    out = []
    for a, b in edges:
        if a == b:
            continue
        if not (0 <= a < n_kpts and 0 <= b < n_kpts):
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def parse_fish_yolo_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    kn, kd = _parse_kpt_shape(text)
    flip_idx = _parse_flip_idx(text)
    class_names = _parse_names_from_text(text)
    kpt_names = []

    data = None
    try:
        import yaml
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            ks = data.get('kpt_shape')
            if isinstance(ks, (list, tuple)) and len(ks) >= 2:
                kn, kd = int(ks[0]), int(ks[1])
            fi = data.get('flip_idx')
            if isinstance(fi, (list, tuple)):
                flip_idx = [int(x) for x in fi]
            nm = data.get('names')
            if isinstance(nm, dict) and nm:
                class_names = [nm[k] for k in sorted(nm.keys(), key=lambda x: int(x))]
            kn_block = data.get('kpt_names') or {}
            k0 = kn_block.get(0, kn_block.get('0'))
            if isinstance(k0, list):
                kpt_names = [str(x).strip() for x in k0]
            elif isinstance(k0, dict):
                kpt_names = [str(v).strip() for _, v in sorted(k0.items(), key=lambda t: int(t[0]))]
    except Exception:
        pass

    if not kpt_names:
        kpt_names = _parse_kpt_names_regex(text)

    n_kpts = int(kn) if kn else len(kpt_names)
    if n_kpts < 1:
        n_kpts = max(len(kpt_names), 1)

    sk_raw = None
    if isinstance(data, dict):
        if 'skeleton' in data:
            sk_raw = data.get('skeleton')
        elif 'kpt_skeleton' in data:
            sk_raw = data.get('kpt_skeleton')

    # None = key absent -> caller uses default chain; [] = explicit no edges
    if sk_raw is None:
        skeleton_edges = None
    elif isinstance(sk_raw, list) and len(sk_raw) == 0:
        skeleton_edges = []
    else:
        skeleton_edges = normalize_skeleton_edges(sk_raw, n_kpts)

    return {
        'raw_path': path,
        'kpt_n': kn,
        'kpt_dims': kd,
        'class_names': class_names,
        'kpt_names': kpt_names,
        'flip_idx': flip_idx,
        'skeleton': skeleton_edges,
    }

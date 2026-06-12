#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CECO FishBase Images 批量自动标注脚本
- 使用 fish_one 权重检测鱼类
- 根据子文件夹名(物种学名)映射到 classes.txt 中的类别
- 输出 YOLO 格式标签（含 score）到 fishbase_images_labels 目录
- IOU_THR=0.5（更激进的 NMS，减少重叠大框）
"""
import os
import sys
import time
import cv2
from ultralytics import YOLO

# 项目根目录加入 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # 上一层为项目根目录
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 切换工作目录到项目根

from libs.yolo_io import YOLOWriter

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# ── 路径配置 ──────────────────────────────────────────────
IMAGE_DIR   = r"H:\Python\YOLO\Fish_Detection\CECO\CECO_datasets\fishbase_images"
LABEL_DIR   = r"H:\Python\YOLO\Fish_Detection\CECO\CECO_datasets\fishbase_images_labels"
CLASSES_FILE = r"H:\Python\YOLO\Fish_Detection\CECO\CECO_datasets\classes.txt"
WEIGHTS     = os.path.join(ROOT, "weights", "fish_one_yolo26s_20260529.pt")

# 检测参数
CONF_THR   = 0.05   # 置信度阈值
IOU_THR    = 0.50   # NMS IoU 阈值（比 Google_Images 的 0.65 更激进，减少重叠大框）
DEDUP_IOU  = 0.80   # 同类去重 IoU

# 模型中的标签
FISH_LABEL  = "鱼"
SHRIMP_LABEL = "虾"

# 特殊文件夹名 → 模型标签的映射（默认全部视为"鱼"）
SHRIMP_FOLDERS = {"Shrimp"}


def load_class_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def folder_to_species(folder_name: str) -> str:
    """子文件夹名 Genus_species → 学名 Genus species"""
    return folder_name.replace('_', ' ', 1)


def dedup_boxes(boxes, scores, labels, iou_thr):
    """同类 NMS 去重"""
    if not boxes:
        return boxes, scores, labels
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep = []
    for idx in order:
        suppress = False
        for kept in keep:
            if str(labels[idx]) != str(labels[kept]):
                continue
            # 计算 IoU
            b1, b2 = boxes[idx], boxes[kept]
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            iou = inter / (a1 + a2 - inter + 1e-9)
            if iou > iou_thr:
                suppress = True
                break
        if not suppress:
            keep.append(idx)
    keep = sorted(keep)
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]


def process_one(result, image_path, label_path, species, class_list,
                is_shrimp_folder=False, model_class_names=None):
    """对单张检测结果写入 YOLO 标签文件，返回检测目标数"""
    img = cv2.imread(image_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    image_shape = [h, w, 3]

    writer = YOLOWriter(
        os.path.dirname(image_path),
        os.path.basename(image_path),
        image_shape,
        localImgPath=image_path,
    )

    dets = result.boxes
    if dets is None or len(dets) == 0:
        return 0

    # 提取检测框
    boxes = [x.tolist() for x in dets.xyxy]
    scores = [float(x) for x in dets.conf]
    cls_ids = [int(x) for x in dets.cls]  # 模型原始类别 id

    # 根据文件夹类型过滤检测结果
    if is_shrimp_folder:
        # Shrimp 文件夹: 只保留模型检测为"虾"(cls=1)的目标
        filtered = [(b, s, c) for b, s, c in zip(boxes, scores, cls_ids)
                    if model_class_names and model_class_names.get(c) == SHRIMP_LABEL]
        if not filtered:
            return 0
        boxes, scores, cls_ids = zip(*filtered)
        boxes, scores = list(boxes), list(scores)
        labels = [species] * len(boxes)
    else:
        # 普通鱼类文件夹: 只保留模型检测为"鱼"(cls=0)的目标
        filtered = [(b, s, c) for b, s, c in zip(boxes, scores, cls_ids)
                    if model_class_names and model_class_names.get(c) == FISH_LABEL]
        if not filtered:
            return 0
        boxes, scores, cls_ids = zip(*filtered)
        boxes, scores = list(boxes), list(scores)
        labels = [species] * len(boxes)

    # 去重
    boxes, scores, labels = dedup_boxes(boxes, scores, labels, DEDUP_IOU)

    count = 0
    for box, score, label in zip(boxes, scores, labels):
        writer.addBndBox(
            int(box[0]), int(box[1]), int(box[2]), int(box[3]),
            label, difficult=False, score=score
        )
        count += 1

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    writer.save(targetFile=label_path, classList=class_list)
    return count


def main():
    print("=" * 60)
    print("CECO FishBase Images 批量自动标注")
    print("=" * 60)

    # 加载类别列表
    class_list = load_class_list(CLASSES_FILE)
    print(f"类别列表: {len(class_list)} 个类别")

    # 扫描物种文件夹
    species_folders = sorted([
        d for d in os.listdir(IMAGE_DIR)
        if os.path.isdir(os.path.join(IMAGE_DIR, d))
    ])
    print(f"物种文件夹: {len(species_folders)} 个")

    missing = []
    for folder in species_folders:
        sp = folder_to_species(folder)
        if sp not in class_list:
            missing.append((folder, sp))
    if missing:
        print(f"\n⚠ 以下 {len(missing)} 个物种不在 classes.txt 中:")
        for f, s in missing:
            print(f"  {f} → '{s}'")

    # 加载模型
    print(f"\n加载模型: {WEIGHTS}")
    model = YOLO(WEIGHTS)
    print(f"模型类别: {model.names}")

    # 创建输出目录
    os.makedirs(LABEL_DIR, exist_ok=True)

    # 写入 classes.txt 到输出目录
    out_classes = os.path.join(LABEL_DIR, 'classes.txt')
    with open(out_classes, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_list) + '\n')
    print(f"classes.txt 已写入: {out_classes}")

    # 统计
    total_images = 0
    total_labeled = 0
    total_objects = 0
    skipped_species = []
    t0 = time.time()

    for si, folder in enumerate(species_folders):
        species = folder_to_species(folder)
        is_shrimp = folder in SHRIMP_FOLDERS
        folder_path = os.path.join(IMAGE_DIR, folder)
        images = sorted([
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(IMG_EXTS)
        ])
        if not images:
            print(f"  [{si+1}/{len(species_folders)}] {folder}: 无图片，跳过")
            continue

        n_img = len(images)
        n_labeled = 0
        n_obj = 0

        for ii, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            base = os.path.splitext(img_name)[0]
            label_path = os.path.join(LABEL_DIR, base + '.txt')

            total_images += 1
            try:
                result = model.predict(
                    source=img_path, conf=CONF_THR, iou=IOU_THR,
                    max_det=300, verbose=False
                )[0]
                cnt = process_one(result, img_path, label_path, species, class_list,
                                  is_shrimp_folder=is_shrimp,
                                  model_class_names=model.names)
                if cnt > 0:
                    n_labeled += 1
                    n_obj += cnt
            except Exception as e:
                print(f"  ✗ 错误 {img_name}: {e}")

            # 进度
            if (ii + 1) % 20 == 0 or (ii + 1) == n_img:
                elapsed = time.time() - t0
                speed = total_images / elapsed if elapsed > 0 else 0
                print(f"  [{si+1}/{len(species_folders)}] {folder}: "
                      f"{ii+1}/{n_img}  标注={n_obj}  "
                      f"总进度={total_images}  速度={speed:.1f}img/s",
                      flush=True)

        if n_labeled > 0:
            total_labeled += n_labeled
            total_objects += n_obj
        else:
            skipped_species.append(folder)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"标注完成!")
    print(f"  处理图片: {total_images}")
    print(f"  有标注图片: {total_labeled}")
    print(f"  总检测目标: {total_objects}")
    print(f"  无标注物种: {len(skipped_species)}")
    if skipped_species:
        for s in skipped_species:
            print(f"    - {s}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  标签目录: {LABEL_DIR}")
    print(f"  IOU_THR: {IOU_THR}")
    print("=" * 60)


if __name__ == '__main__':
    main()

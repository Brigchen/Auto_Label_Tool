#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CECO Google Images 批量自动标注脚本
- 使用 fish_one 权重检测鱼类
- 根据子文件夹名(物种学名)映射到 classes.txt 中的类别
- 输出 YOLO 格式标签（含 score）到 Google_Images_labels 目录
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
IMAGE_DIR   = r"H:\Python\YOLO\Fish_Detection\CECO\CECO_datasets\Google_Images"
LABEL_DIR   = r"H:\Python\YOLO\Fish_Detection\CECO\CECO_datasets\Google_Images_labels"
CLASSES_FILE = r"H:\Python\YOLO\Fish_Detection\CECO\CECO_datasets\classes.txt"
WEIGHTS     = os.path.join(ROOT, "weights", "fish_one_yolo26s_20260529.pt")

# 检测参数
CONF_THR   = 0.05   # 置信度阈值
IOU_THR    = 0.65   # NMS IoU 阈值
DEDUP_IOU  = 0.80   # 同类去重 IoU

# 模型中"鱼"对应的标签 → 会被替换为物种名
FISH_LABEL = "鱼"


def load_class_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def folder_to_species(folder_name: str) -> str:
    """子文件夹名 Genus_species → 学名 Genus species"""
    # 特殊处理: Epinephelus_fuscoguttatus-lanceolatus → Epinephelus fuscoguttatus-lanceolatus
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


def process_one(result, image_path, label_path, species, class_list):
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
    labels = [species] * len(boxes)  # 全部用物种名替换

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
    print("CECO Google Images 批量自动标注")
    print("=" * 60)

    # 加载类别列表
    class_list = load_class_list(CLASSES_FILE)
    print(f"类别列表: {len(class_list)} 个类别")

    # 构建 物种名 → class_list 中是否存在的映射
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
        folder_path = os.path.join(IMAGE_DIR, folder)
        images = sorted([
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(IMG_EXTS)
        ])
        if not images:
            continue

        n_img = len(images)
        n_labeled = 0
        n_obj = 0

        for ii, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            base = os.path.splitext(img_name)[0]
            # 标签文件直接保存在 Google_Images_labels 根目录下
            label_path = os.path.join(LABEL_DIR, base + '.txt')

            total_images += 1
            try:
                result = model.predict(
                    source=img_path, conf=CONF_THR, iou=IOU_THR, verbose=False
                )[0]
                cnt = process_one(result, img_path, label_path, species, class_list)
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
                      f"总进度={total_images}/7500  速度={speed:.1f}img/s",
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
    print("=" * 60)


if __name__ == '__main__':
    main()

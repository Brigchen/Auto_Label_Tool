# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 08:45:14 2022

@author: Brig
"""

import os
import random
import shutil

from libs.dataset_paths import collect_image_label_pairs, flat_output_stem


def make_train_val(
    path_img,
    path_label,
    path_out,
    train_percent=0.7,
    test_percent=0.0,
    uw_offline_copies=0,
    uw_offline_strength="medium",
    uw_offline_seed=42,
):
    """Copy train/val splits; optional test split for FishVision test report."""
    path_img = os.path.abspath(path_img)
    path_label = os.path.abspath(path_label)
    path_out = os.path.abspath(path_out)
    if not os.path.isdir(path_img) or not os.path.isdir(path_label):
        print('[make_train_val] invalid image or label directory')
        return 0

    pairs = collect_image_label_pairs(path_img, path_label, '.txt')
    num = len(pairs)
    if num == 0:
        print('[make_train_val] no image/label pairs found under %s and %s' % (path_img, path_label))
        return 0

    train_percent = float(train_percent)
    test_percent = max(0.0, float(test_percent or 0.0))
    if test_percent > 0.0 and train_percent + test_percent >= 1.0:
        print('[make_train_val] train_percent + test_percent must be < 1.0')
        return 0

    indices = list(range(num))
    random.shuffle(indices)
    tr = int(num * train_percent)
    train_indices = set(indices[:tr])
    if test_percent > 0.0:
        te = int(num * test_percent)
        test_indices = set(indices[tr:tr + te])
        val_indices = set(indices[tr + te:])
    else:
        test_indices = set()
        val_indices = set(indices[tr:])

    splits = ('train', 'val', 'test') if test_percent > 0.0 else ('train', 'val')
    for image_set in splits:
        os.makedirs(os.path.join(path_out, 'images', image_set), exist_ok=True)
        os.makedirs(os.path.join(path_out, 'labels', image_set), exist_ok=True)

    def _split_for(i):
        if i in train_indices:
            return 'train'
        if i in test_indices:
            return 'test'
        return 'val'

    from libs.underwater_augment import write_offline_variants

    uw_offline_copies = max(0, int(uw_offline_copies or 0))
    uw_splits = {"train"} if uw_offline_copies > 0 else set()
    uw_written = 0
    copied = 0
    for i, (rel, img_path, lbl_path) in enumerate(pairs):
        split = _split_for(i)
        stem = flat_output_stem(rel)
        img_ext = os.path.splitext(img_path)[1]
        try:
            shutil.copy2(
                img_path,
                os.path.join(path_out, 'images', split, stem + img_ext),
            )
            shutil.copy2(
                lbl_path,
                os.path.join(path_out, 'labels', split, stem + '.txt'),
            )
            copied += 1
            if split in uw_splits:
                dest_img = os.path.join(path_out, 'images', split)
                dest_lbl = os.path.join(path_out, 'labels', split)
                uw_written += write_offline_variants(
                    img_path,
                    lbl_path,
                    dest_img,
                    dest_lbl,
                    stem,
                    img_ext,
                    copies=uw_offline_copies,
                    strength=uw_offline_strength,
                    seed=int(uw_offline_seed),
                    log=print,
                )
        except Exception as e:
            print('[make_train_val]', e)
    if uw_offline_copies > 0:
        print('[make_train_val] underwater offline augment: +%d train variants (strength=%s)' % (
            uw_written, uw_offline_strength))
    if test_percent > 0.0:
        print('[make_train_val] copied %d pairs (train=%d val=%d test=%d)' % (
            copied, len(train_indices), len(val_indices), len(test_indices)))
    else:
        print('[make_train_val] copied %d pairs (train=%d val=%d)' % (
            copied, len(train_indices), len(val_indices)))
    return copied

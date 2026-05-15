# -*- coding: utf-8 -*-
"""Screen / copy images and labels by taxon (YOLO or VOC XML)."""
import os
import shutil
import xml.etree.ElementTree as ET

_IMG_EXT = [".bmp", ".jpeg", ".jpg", ".png", ".tif"]


def load_predefined_classes(predef_classes_file):
    if os.path.exists(predef_classes_file):
        with open(predef_classes_file, "r", encoding="utf-8") as f:
            predefined_classes = f.read().splitlines()
        print("load class names: ", len(predefined_classes))
        return predefined_classes
    print("%s not existed" % predef_classes_file)
    return []


def _find_image_for_txt(basename, img_dir):
    for ext in _IMG_EXT:
        img_file = os.path.join(img_dir, "%s%s" % (basename, ext))
        if os.path.exists(img_file):
            return True, img_file
    return False, ""


def copy_images_for_txts(txt_dir, img_dir, save_dir):
    for path, _dirs, files in os.walk(txt_dir):
        for fl in files:
            if not fl.endswith(".txt") or fl == "classes.txt":
                continue
            basename = os.path.splitext(fl)[0]
            found, im_fl = _find_image_for_txt(basename, img_dir)
            if found:
                print(im_fl)
                shutil.copy(im_fl, save_dir)


def screen_yolo(yolo_path, img_dir, taxa, taxa_nms, sav_dir):
    fl = os.path.basename(yolo_path)
    with open(yolo_path, "r", encoding="utf-8") as box_file:
        boxes = [ln.rstrip().split() for ln in box_file if ln.strip()]
    for name in taxa:
        sav_dir_txt = os.path.join(sav_dir, str(name), "txts")
        sav_dir_img = os.path.join(sav_dir, str(name), "images")
        os.makedirs(sav_dir_txt, exist_ok=True)
        os.makedirs(sav_dir_img, exist_ok=True)
        for box in boxes:
            if int(box[0]) > 0:
                print("%s found in: %s" % (name, yolo_path))
                try:
                    found, im_fl = _find_image_for_txt(os.path.splitext(fl)[0], img_dir)
                    if found:
                        shutil.copy(yolo_path, sav_dir_txt)
                        shutil.copy(im_fl, sav_dir_img)
                    return
                except Exception as e:
                    print(e)
                    return


def screen_xml(xml_path, sav_dir, taxa_nms):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    print(xml_path)
    if size is None:
        return
    for obj in root.iter("object"):
        name = obj.find("name").text.strip()
        if name in taxa_nms:
            print("found %s in %s" % (name, xml_path))
            sav_dir_xml = os.path.join(sav_dir, str(name), "xmls")
            os.makedirs(sav_dir_xml, exist_ok=True)
            try:
                shutil.move(xml_path, os.path.join(sav_dir_xml, os.path.basename(xml_path)))
            except Exception as e:
                print(e)
            return


def main_xml(root, out_path, taxa_nms):
    for path, _dirs, files in os.walk(root):
        for fl in files:
            if ".xml" not in fl:
                continue
            try:
                xml_dir = os.path.join(path, fl)
                screen_xml(xml_dir, out_path, taxa_nms)
            except Exception as e:
                print(e)


def main_yolo(txt_dir, img_dir, save_dir, taxa, class_file):
    taxa_nms = load_predefined_classes(class_file)
    print(taxa_nms)
    for path, _dirs, files in os.walk(txt_dir):
        for fl in files:
            if not fl.endswith(".txt") or fl == "classes.txt":
                continue
            try:
                screen_yolo(os.path.join(path, fl), img_dir, taxa, taxa_nms, save_dir)
            except Exception as e:
                print(e)

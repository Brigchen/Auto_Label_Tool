# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:10:27 2022

@author: Brig
"""

# xml解析包
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil
import pandas as pd
import random

#%% load class names

def class_load(filename):
    # filename = r"C:\YOLO\Fish_Detection\CECO\CECO_labels_classes\classes.txt"
    try:
        df = pd.read_csv(filename, header = None)
        classes = df[0].tolist()
        return classes
    except Exception as e:
        print('loading class file error: %s'%e)
        return None
        
# classes = ['Fish']
#%%
# 进行归一化操作
def convert(size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[1]) / 2.0  # 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 物体在图中的中心点y坐标
    w = box[1] - box[0]  # 物体实际像素宽度

    h = box[3] - box[2]  # 物体实际像素高度
    x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
    w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
    y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
    h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)   # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]

# year ='2012', 对应图片的id（文件名）
def convert_annotation_xml_to_txt(xml_path, txt_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    if size is None:
        return
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    os.makedirs(os.path.dirname(txt_path) or '.', exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text.strip()
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def convert_annotation(xmls_path, txts_path, image_id, classes):
    convert_annotation_xml_to_txt(
        os.path.join(xmls_path, '%s.xml' % image_id),
        os.path.join(txts_path, '%s.txt' % image_id),
        classes,
    )

#%%
def voc2yolo(txts_path, xmls_path, classes):
    from libs.dataset_paths import flat_output_stem, iter_files_with_ext

    if not classes:
        return
    os.makedirs(txts_path, exist_ok=True)
    for rel, xml_path in iter_files_with_ext(xmls_path, '.xml'):
        stem = flat_output_stem(rel)
        txt_out = os.path.join(txts_path, stem + '.txt')
        try:
            convert_annotation_xml_to_txt(xml_path, txt_out, classes)
        except Exception as e:
            print('[voc2yolo]', rel, e)


#%%
txtDir = r"C:\Users\brigc\Documents\MATLAB\Python\Pytorch\ObjectiveDetection\Auto_Label_Tool\datasets\one_class\labels\0kb"
xmlDir = r"C:\Users\brigc\Documents\MATLAB\Python\Pytorch\ObjectiveDetection\Auto_Label_Tool\datasets\one_class\labels\0kb"    
classFile = r"C:\Users\brigc\Documents\MATLAB\Python\Pytorch\ObjectiveDetection\Auto_Label_Tool\datasets\fish_single_classes_eng.txt"
classes = class_load(classFile)
voc2yolo(txtDir, xmlDir, classes)

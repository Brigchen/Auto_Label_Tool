#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
from typing import Optional
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from libs.constants import DEFAULT_ENCODING

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

def split_yolo_difficult_and_score(d) -> tuple:
    """Map shape ``difficult`` field to ``(voc_difficult, detection_score)``."""
    if isinstance(d, bool):
        return d, None
    if isinstance(d, float):
        fv = float(d)
        if fv == 0.0:
            return False, None
        if 0.0 < fv <= 1.0:
            return False, fv
        return bool(d), None
    if isinstance(d, int):
        return bool(d), None
    return False, None


def resolve_yolo_write_score(difficult, score=None) -> Optional[float]:
    """Score for YOLO txt suffix; never treat VOC ``difficult=0/1`` as confidence."""
    if score is not None:
        fv = float(score)
        return fv if fv > 0.0 else None
    if isinstance(difficult, float):
        fv = float(difficult)
        return fv if fv > 0.0 else None
    return None


class YOLOWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult, score=None):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        voc_diff, _ = split_yolo_difficult_and_score(difficult)
        bndbox['difficult'] = voc_diff
        resolved = resolve_yolo_write_score(difficult, score)
        if resolved is not None:
            bndbox['score'] = resolved
        self.boxlist.append(bndbox)

    def addPolygon(self, points, name, difficult, score=None):
        polygon = {'points': points}
        polygon['name'] = name
        voc_diff, _ = split_yolo_difficult_and_score(difficult)
        polygon['difficult'] = voc_diff
        resolved = resolve_yolo_write_score(difficult, score)
        if resolved is not None:
            polygon['score'] = resolved
        self.boxlist.append(polygon)

    def addKeypoints(self, points, visibility, name, difficult, score=None):
        pose = {'points': points, 'visibility': visibility}
        pose['name'] = name
        voc_diff, _ = split_yolo_difficult_and_score(difficult)
        pose['difficult'] = voc_diff
        pose['shape_type'] = 'keypoints'
        resolved = resolve_yolo_write_score(difficult, score)
        if resolved is not None:
            pose['score'] = resolved
        self.boxlist.append(pose)

    @staticmethod
    def _line_score_suffix(box):
        sc = box.get('score')
        if sc is None:
            return ''
        try:
            return ' %.6f' % float(sc)
        except (TypeError, ValueError):
            return ''

    def BndBox2YoloLine(self, box, classList=[]):
        xmin = box['xmin']
        xmax = box['xmax']
        ymin = box['ymin']
        ymax = box['ymax']

        xcen = float((xmin + xmax)) / 2 / self.imgSize[1]
        ycen = float((ymin + ymax)) / 2 / self.imgSize[0]

        w = float((xmax - xmin)) / self.imgSize[1]
        h = float((ymax - ymin)) / self.imgSize[0]

        # PR387
        boxName = box['name']
        if boxName not in classList:
            classList.append(boxName)

        classIndex = classList.index(boxName)

        return classIndex, xcen, ycen, w, h

    def Polygon2YoloLine(self, polygon, classList=[]):
        boxName = polygon['name']
        if boxName not in classList:
            classList.append(boxName)
        classIndex = classList.index(boxName)

        points = []
        for x, y in polygon['points']:
            px = min(max(float(x) / self.imgSize[1], 0.0), 1.0)
            py = min(max(float(y) / self.imgSize[0], 0.0), 1.0)
            points.extend([px, py])
        return classIndex, points

    def Keypoints2YoloLine(self, pose, classList=[]):
        boxName = pose['name']
        if boxName not in classList:
            classList.append(boxName)
        classIndex = classList.index(boxName)

        pts = pose.get('points', [])
        vis = pose.get('visibility', [])
        if not pts:
            return None

        def _v(i):
            if i < len(vis):
                try:
                    return int(vis[i])
                except (TypeError, ValueError):
                    return 2
            return 2

        labeled = [(p[0], p[1]) for i, p in enumerate(pts) if _v(i) != 0]
        if labeled:
            xs = [p[0] for p in labeled]
            ys = [p[1] for p in labeled]
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        xcen = float((xmin + xmax)) / 2 / self.imgSize[1]
        ycen = float((ymin + ymax)) / 2 / self.imgSize[0]
        w = float((xmax - xmin)) / self.imgSize[1]
        h = float((ymax - ymin)) / self.imgSize[0]

        kpts = []
        for i, (x, y) in enumerate(pts):
            v = int(vis[i]) if i < len(vis) else 2
            if v == 0:
                kpts.extend([0.0, 0.0, 0])
            else:
                px = min(max(float(x) / self.imgSize[1], 0.0), 1.0)
                py = min(max(float(y) / self.imgSize[0], 0.0), 1.0)
                kpts.extend([px, py, v])
        return classIndex, xcen, ycen, w, h, kpts

    def save(self, classList=[], targetFile=None):

        out_file = None #Update yolo .txt
        out_class_file = None   #Update class list .txt

        if targetFile is None:
            out_file = open(
            self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
            classesFile = os.path.join(os.path.dirname(os.path.abspath(self.filename)), "classes.txt")
            out_class_file = open(classesFile, 'w', encoding=ENCODE_METHOD)

        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)
            classesFile = os.path.join(os.path.dirname(os.path.abspath(targetFile)), "classes.txt")
            out_class_file = open(classesFile, 'w', encoding=ENCODE_METHOD)


        for box in self.boxlist:
            if box.get('shape_type') == 'keypoints':
                pose_line = self.Keypoints2YoloLine(box, classList)
                if pose_line is not None:
                    classIndex, xcen, ycen, w, h, kpts = pose_line
                    out_file.write(
                        "%d %.6f %.6f %.6f %.6f %s%s\n" % (
                            classIndex, xcen, ycen, w, h,
                            " ".join(["%.6f" % p if (idx % 3) != 2 else str(int(p))
                                      for idx, p in enumerate(kpts)]),
                            self._line_score_suffix(box),
                        )
                    )
            elif 'points' in box:
                classIndex, points = self.Polygon2YoloLine(box, classList)
                if len(points) >= 6:
                    out_file.write(
                        "%d %s%s\n" % (
                            classIndex,
                            " ".join(["%.6f" % p for p in points]),
                            self._line_score_suffix(box),
                        )
                    )
            else:
                classIndex, xcen, ycen, w, h = self.BndBox2YoloLine(box, classList)
                out_file.write(
                    "%d %.6f %.6f %.6f %.6f%s\n" % (
                        classIndex, xcen, ycen, w, h, self._line_score_suffix(box))
                )

        # print (classList)
        # print (out_class_file)
        for c in classList:
            out_class_file.write(c+'\n')

        out_class_file.close()
        out_file.close()


class YoloReader:

    def __init__(self, filepath, image, classListPath=None, posed_kpts_count=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath

        if classListPath is None:
            dir_path = os.path.dirname(os.path.realpath(self.filepath))
            self.classListPath = os.path.join(dir_path, "classes.txt")
        else:
            self.classListPath = classListPath

        # print (filepath, self.classListPath)

        classesFile = open(self.classListPath, 'r', encoding='utf-8')
        self.classes = classesFile.read().strip('\n').split('\n')

        # print (self.classes)

        imgSize = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]

        self.imgSize = imgSize
        # If set (e.g. from fish_yolo.yaml kpt_shape[0]), force pose parsing when triplet count matches.
        self.posed_kpts_count = posed_kpts_count

        self.verified = False
        # try:
        self.parseYoloFormat()
        # except:
            # pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, xmin, ymin, xmax, ymax, difficult):

        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult, 'bbox', []))

    def addPolygonShape(self, label, points, difficult):
        self.shapes.append((label, points, None, None, difficult, 'polygon', []))

    def addKeypointShape(self, label, points, visibility, difficult):
        self.shapes.append((label, points, None, None, difficult, 'keypoints', visibility))

    def yoloLine2Shape(self, classIndex, xcen, ycen, w, h):
        label = self.classes[int(classIndex)]

        xmin = max(float(xcen) - float(w) / 2, 0)
        xmax = min(float(xcen) + float(w) / 2, 1)
        ymin = max(float(ycen) - float(h) / 2, 0)
        ymax = min(float(ycen) + float(h) / 2, 1)

        xmin = int(self.imgSize[1] * xmin)
        xmax = int(self.imgSize[1] * xmax)
        ymin = int(self.imgSize[0] * ymin)
        ymax = int(self.imgSize[0] * ymax)

        return label, xmin, ymin, xmax, ymax

    def yoloLine2Polygon(self, values):
        classIndex = int(values[0])
        label = self.classes[classIndex]
        coords = values[1:]

        points = []
        for i in range(0, len(coords), 2):
            x = min(max(float(coords[i]), 0.0), 1.0)
            y = min(max(float(coords[i + 1]), 0.0), 1.0)
            points.append((int(self.imgSize[1] * x), int(self.imgSize[0] * y)))
        return label, points

    def yoloLine2Keypoints(self, values):
        classIndex = int(values[0])
        label = self.classes[classIndex]
        coords = values[1:]
        xcen, ycen, w, h = coords[:4]
        kpts = coords[4:]

        points = []
        visibility = []
        for i in range(0, len(kpts), 3):
            x = min(max(float(kpts[i]), 0.0), 1.0)
            y = min(max(float(kpts[i + 1]), 0.0), 1.0)
            v = int(float(kpts[i + 2]))
            points.append((int(self.imgSize[1] * x), int(self.imgSize[0] * y)))
            visibility.append(v)
        return label, points, visibility

    def parseYoloFormat(self):
        bndBoxFile = open(self.filepath, 'r')
        for bndBox in bndBoxFile:
            values = bndBox.strip().split()
            if not values:
                continue

            def _trailing_score(vals):
                if len(vals) < 2:
                    return vals, None
                try:
                    sc = float(vals[-1])
                    if 0.0 <= sc <= 1.0:
                        return vals[:-1], sc
                except (TypeError, ValueError):
                    pass
                return vals, None

            # YOLO detection format: class cx cy w h [score]
            if len(values) in (5, 6):
                vals, score = (values, None) if len(values) == 5 else _trailing_score(values)
                if len(vals) != 5:
                    vals, score = values[:5], (float(values[5]) if len(values) == 6 else None)
                classIndex, xcen, ycen, w, h = vals
                label, xmin, ymin, xmax, ymax = self.yoloLine2Shape(classIndex, xcen, ycen, w, h)
                if score is not None and float(score) > 0.0:
                    diff = float(score)
                else:
                    diff = False
                self.addShape(label, xmin, ymin, xmax, ymax, diff)
                continue

            # YOLO pose format (Ultralytics): class cx cy w h (x y v)... [score]
            vals_pose, score_pose = _trailing_score(values)
            if len(vals_pose) >= 8 and (len(vals_pose) - 5) % 3 == 0:
                is_pose_line = True
                nk = (len(vals_pose) - 5) // 3
                if self.posed_kpts_count is not None and nk != int(self.posed_kpts_count):
                    is_pose_line = False
                for i in range(7, len(vals_pose), 3):
                    try:
                        v = int(float(vals_pose[i]))
                    except Exception:
                        is_pose_line = False
                        break
                    if v not in (0, 1, 2):
                        is_pose_line = False
                        break
                if is_pose_line:
                    label, points, visibility = self.yoloLine2Keypoints(vals_pose)
                    if len(points) >= 1:
                        diff = float(score_pose) if score_pose is not None and float(score_pose) > 0.0 else False
                        self.addKeypointShape(label, points, visibility, diff)
                    continue

            # YOLO segmentation format: class x1 y1 x2 y2 ... [score]
            vals_seg, score_seg = _trailing_score(values)
            if len(vals_seg) >= 7 and (len(vals_seg) - 1) % 2 == 0:
                label, points = self.yoloLine2Polygon(vals_seg)
                if len(points) >= 3:
                    diff = float(score_seg) if score_seg is not None and float(score_seg) > 0.0 else False
                    self.addPolygonShape(label, points, diff)

# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from base64 import b64encode, b64decode
from libs.pascal_voc_io import PascalVocWriter
from libs.yolo_io import YOLOWriter, split_yolo_difficult_and_score
from libs.pascal_voc_io import XML_EXT
import os.path
import sys

import warnings
warnings.filterwarnings('ignore')


class LabelFileError(Exception):
    pass


class LabelFile(object):
    # It might be changed as window creates. By default, using XML ext
    # suffix = '.lif'
    suffix = XML_EXT

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.verified = False

    def savePascalVocFormat(self, filename, shapes, imagePath, imageData,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        image = QImage()
        ok = image.load(imagePath)
        if not ok:
            print('image format eroor, and try to retrieve it anyway')
            with open(imagePath,'rb') as f:
                image.loadFromData(f.read())
        # image.load(imagePath)
        imageShape = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]
        writer = PascalVocWriter(imgFolderName, imgFileName,
                                 imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            # Add Chris
            # Convert difficult: score > 0.5 -> difficult=0, score <= 0.5 -> difficult=1
            raw_difficult = shape['difficult']
            if isinstance(raw_difficult, bool):
                difficult = int(raw_difficult)
            elif isinstance(raw_difficult, (int, float)):
                fv = float(raw_difficult)
                if 0.0 < fv <= 1.0:
                    # Treat as confidence score: high conf (>0.5) -> not difficult (0)
                    difficult = 0 if fv > 0.5 else 1
                else:
                    difficult = int(bool(fv))
            else:
                difficult = 0
            bndbox = LabelFile.convertPoints2BndBox(points)
            writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult)

        writer.save(targetFile=filename)
        return

    def saveYoloFormat(self, filename, shapes, imagePath, imageData, classList,
                            saveSegmentation=False,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        image = QImage()
        ok = image.load(imagePath)
        if not ok:
            print('image format eroor, and try to retrieve it anyway')
            with open(imagePath,'rb') as f:
                image.loadFromData(f.read())
        imageShape = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]
        writer = YOLOWriter(imgFolderName, imgFileName,
                                 imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            shape_type = shape.get('shape_type', 'bbox')
            voc_difficult, det_score = split_yolo_difficult_and_score(shape['difficult'])
            if shape_type == 'keypoints':
                writer.addKeypoints(
                    points, shape.get('keypoint_visibility', []), label, voc_difficult,
                    score=det_score,
                )
            elif saveSegmentation:
                writer.addPolygon(points, label, voc_difficult, score=det_score)
            else:
                bndbox = LabelFile.convertPoints2BndBox(points)
                writer.addBndBox(
                    bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, voc_difficult,
                    score=det_score,
                )

        writer.save(targetFile=filename, classList=classList)
        return
    
    # def saveDet2Yolo(self, filename, dets, imagePath, imageData, classList):
    #     imgFolderPath = os.path.dirname(imagePath)
    #     imgFolderName = os.path.split(imgFolderPath)[-1]
    #     imgFileName = os.path.basename(imagePath)
    #     #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
    #     # Read from file path because self.imageData might be empty if saving to
    #     # Pascal format
    #     image = QImage()
    #     ok = image.load(imagePath)
    #     if not ok:
    #         print('image format eroor, and try to retrieve it anyway')
    #         with open(imagePath,'rb') as f:
    #             image.loadFromData(f.read())
    #     imageShape = [image.height(), image.width(),
    #                   1 if image.isGrayscale() else 3]
    #     writer = YOLOWriter(imgFolderName, imgFileName,
    #                              imageShape, localImgPath=imagePath)
    #     writer.verified = self.verified
        
    #     for det in dets:
    #         writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult)

    #     writer.save(targetFile=filename, classList=classList)
    #     return

    def toggleVerify(self):
        self.verified = not self.verified

    ''' ttf is disable
    def load(self, filename):
        import json
        with open(filename, 'rb') as f:
                data = json.load(f)
                imagePath = data['imagePath']
                imageData = b64decode(data['imageData'])
                lineColor = data['lineColor']
                fillColor = data['fillColor']
                shapes = ((s['label'], s['points'], s['line_color'], s['fill_color'])\
                        for s in data['shapes'])
                # Only replace data after everything is loaded.
                self.shapes = shapes
                self.imagePath = imagePath
                self.imageData = imageData
                self.lineColor = lineColor
                self.fillColor = fillColor

    def save(self, filename, shapes, imagePath, imageData, lineColor=None, fillColor=None):
        import json
        with open(filename, 'wb') as f:
                json.dump(dict(
                    shapes=shapes,
                    lineColor=lineColor, fillColor=fillColor,
                    imagePath=imagePath,
                    imageData=b64encode(imageData)),
                    f, ensure_ascii=True, indent=2)
    '''

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))


#!/usr/bin/python
# -*- coding: utf-8 -*-


try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.utils import distance
import sys

DEFAULT_LINE_COLOR = QColor(128, 0, 255, 200)
DEFAULT_FILL_COLOR = QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QColor(0, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR = QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)
MIN_Y_LABEL = 10


class Shape(object):
    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    # The following class variables influence the drawing
    # of _all_ shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 12
    scale = 1.0

    def __init__(self, label=None, line_color=None, difficult=False, paintLabel=False, shape_type='bbox'):
        self.label = label
        self.points = []
        self.shape_type = shape_type
        self.keypoint_visibility = []
        self.keypoint_names = []
        self.skeleton = []
        self.fill = False
        self.selected = False
        self.difficult = difficult
        self.paintLabel = paintLabel
        # Keypoints: double-click body selects whole instance; all vertices drawn as squares for group move/delete.
        self.keypoints_bulk_select = False

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def close(self):
        self._closed = True

    def reachMaxPoints(self):
        return False

    def addPoint(self, point):
        self.points.append(point)

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def _kpt_labeled(self, i):
        """YOLO pose v: 0=not labeled, 1=occluded, 2=visible — only v==0 is hidden in the UI."""
        if self.shape_type != 'keypoints':
            return True
        if i < 0:
            return False
        if i >= len(self.keypoint_visibility):
            return True
        try:
            return int(self.keypoint_visibility[i]) != 0
        except (TypeError, ValueError):
            return True

    def paint(self, painter):
        if self.points:
            color = self.select_line_color if self.selected else self.line_color
            pen = QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QPainterPath()
            vrtx_path = QPainterPath()

            is_kpt = self.shape_type == 'keypoints'
            if not is_kpt:
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self.points[0])
            else:
                bulk = self.selected and getattr(self, 'keypoints_bulk_select', False)
                if bulk:
                    for i, p in enumerate(self.points):
                        self.drawVertex(vrtx_path, i)
                else:
                    for i, p in enumerate(self.points):
                        if self._kpt_labeled(i):
                            self.drawVertex(vrtx_path, i)

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self.vertex_fill_color)

            if self.shape_type == 'keypoints' and self.skeleton:
                skeleton_pen = QPen(color)
                skeleton_pen.setWidth(max(1, int(round(1.5 / self.scale))))
                painter.setPen(skeleton_pen)
                for a, b in self.skeleton:
                    if not (self._kpt_labeled(a) and self._kpt_labeled(b)):
                        continue
                    if 0 <= a < len(self.points) and 0 <= b < len(self.points):
                        painter.drawLine(self.points[a], self.points[b])

            # Draw text at the top-left
            if self.paintLabel:
                min_x = sys.maxsize
                min_y = sys.maxsize
                bulk_lbl = is_kpt and self.selected and getattr(self, 'keypoints_bulk_select', False)
                for i, point in enumerate(self.points):
                    if is_kpt and not bulk_lbl and not self._kpt_labeled(i):
                        continue
                    min_x = min(min_x, point.x())
                    min_y = min(min_y, point.y())
                if min_x != sys.maxsize and min_y != sys.maxsize:
                    font = QFont()
                    font.setPointSize(32)
                    font.setBold(True)
                    painter.setFont(font)
                    #painter.setPen(QColor(255,255,255,255))
                    if(self.label == None):
                        self.label = ""
                    if(min_y < MIN_Y_LABEL):
                        min_y += MIN_Y_LABEL
                    painter.drawText(int(min_x), int(min_y), self.label)

            if self.shape_type == 'keypoints' and self.keypoint_names:
                painter.setPen(QColor(255, 255, 0))
                bulk = self.selected and getattr(self, 'keypoints_bulk_select', False)
                for i, point in enumerate(self.points):
                    if not bulk and not self._kpt_labeled(i):
                        continue
                    if i < len(self.keypoint_names):
                        painter.drawText(int(point.x()) + 4, int(point.y()) - 4, self.keypoint_names[i])

            if self.fill:
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        bulk = self.shape_type == 'keypoints' and self.selected and getattr(self, 'keypoints_bulk_select', False)
        shape = self.P_SQUARE if bulk else self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, hv_shape = self._highlightSettings[self._highlightMode]
            d *= size
            shape = self.P_SQUARE if bulk else hv_shape
        if self._highlightIndex is not None:
            self.vertex_fill_color = self.hvertex_fill_color
        else:
            self.vertex_fill_color = Shape.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        for i, p in enumerate(self.points):
            # Keypoints: include v==0 slots (phantom positions) so they can be selected / deleted / moved.
            if distance(p - point) <= epsilon:
                return i
        return None

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def makePath(self):
        path = QPainterPath(self.points[0])
        for p in self.points[1:]:
            path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        shape = Shape("%s" % self.label, shape_type=self.shape_type)
        shape.points = [p for p in self.points]
        shape.keypoint_visibility = list(self.keypoint_visibility)
        shape.keypoint_names = list(self.keypoint_names)
        shape.skeleton = list(self.skeleton)
        shape.keypoints_bulk_select = False
        shape.fill = self.fill
        shape.selected = self.selected
        shape._closed = self._closed
        if self.line_color != Shape.line_color:
            shape.line_color = self.line_color
        if self.fill_color != Shape.fill_color:
            shape.fill_color = self.fill_color
        shape.difficult = self.difficult
        return shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

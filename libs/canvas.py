import pyautogui,cv2
import numpy as np
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

#from PyQt4.QtOpenGL import *

from libs.shape import Shape
from libs.utils import distance
from libs.keypoint_utils import pad_keypoints_shape, spread_keypoint_placeholders

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor

# class Canvas(QGLWidget):


class Canvas(QWidget):
    zoomRequest = pyqtSignal(int)
    scrollRequest = pyqtSignal(int, int)
    newShape = pyqtSignal()
    undoCheckpoint = pyqtSignal()
    selectionChanged = pyqtSignal(bool)
    shapeMoved = pyqtSignal()
    drawingPolygon = pyqtSignal(bool)
    # Edit mode: right-click on a keypoint vertex -> parent shows kpt_name picker (shape, point_index).
    keypointRenameRequested = pyqtSignal(object, int)

    CREATE, EDIT = list(range(2))

    epsilon = 11.0

    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.current = None
        self.selectedShape = None  # save the selected shape here
        self.selectedShapeCopy = None
        self.drawingLineColor = QColor(0, 0, 255)
        self.drawingRectColor = QColor(0, 0, 255)
        self.line = Shape(line_color=self.drawingLineColor)
        self.prevPoint = QPointF()
        self.offsets = QPointF(), QPointF()
        self.scale = 1.0
        self.pixmap = QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.hVertex = None
        self._painter = QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        self.menus = (QMenu(), QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        self.verified = False
        self._move_undo_pushed = False
        self._arrow_undo_pushed = False
        self.drawSquare = False
        self.drawingShapeMode = 'bbox'
        self.keypointNames = []
        self.keypointSkeleton = []
        self.maxKeypoints = 0

    def setDrawingColor(self, qColor):
        self.drawingLineColor = qColor
        self.drawingRectColor = qColor

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()
        self.prevPoint = QPointF()
        self.repaint()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
        self.hVertex = self.hShape = None

    def selectedVertex(self):
        return self.hVertex is not None
    
    def get_mouse_img(self):
        x,y = pyautogui.position()
        box_size=int(120)    #可调
        half_size=int(box_size/2)
        enlarge_ratio=2.5      #可调
        img = pyautogui.screenshot(region=[x-half_size,y-half_size,box_size,box_size])
        img=cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
        cv_img=cv2.resize(img, (int(enlarge_ratio*box_size), int(enlarge_ratio*box_size)), interpolation=cv2.INTER_CUBIC)
        cv_img=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
        img=QImage(cv_img[:],cv_img.shape[1], cv_img.shape[0],cv_img.shape[1] * 3, QImage.Format_RGB888)
        img=QPixmap.fromImage(img)
        return img
    
    def get_mouse_area(self,pos,np_img,area_size=60,ratio=5):
        xmin=int(max(0,pos.x()-area_size))
        xmax=int(min(np_img.shape[1],pos.x()+area_size))
        ymin=int(max(0,pos.y()-area_size))
        ymax=int(min(np_img.shape[0],pos.y()+area_size))
        np_img=np_img[ymin:ymax,xmin:xmax,:]
        enlargesize=int(ratio*area_size)
        np_img=cv2.resize(np_img, (enlargesize, enlargesize), interpolation=cv2.INTER_CUBIC)
        np_img=cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB)
        cv2.line(np_img, (0, enlargesize//2), (enlargesize, enlargesize//2), (0, 255, 0))
        cv2.line(np_img, (enlargesize//2,0), (enlargesize//2, enlargesize), (0, 255, 0))  
        qimg=QImage(np_img[:],np_img.shape[1], np_img.shape[0],np_img.shape[1] * 3, QImage.Format_RGB888)
        qimg=QPixmap.fromImage(qimg)
        return qimg
        
    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        pos = self.transformPos(ev.pos())
        # Update coordinates in status bar if image is opened
        window = self.parent().window()
        if window.useMagnifyingLens.isChecked() and window.filePath:
            # image=window.image 
            try:
                img=cv2.imread(window.filePath)
                img=self.get_mouse_area(pos,img)
                window.new_test.setPixmap(img) 
            except:
                pass
        
        if window.filePath is not None:
            self.parent().window().labelCoordinates.setText(
                'X: %d; Y: %d' % (int(pos.x()), int(pos.y())))

        # Polygon drawing.
        if self.drawing():
            self.overrideCursor(CURSOR_DRAW)
            if self.current:
                # Display annotation width and height while drawing
                currentWidth = int(abs(self.current[0].x() - pos.x()))
                currentHeight = int(abs(self.current[0].y() - pos.y()))
                self.parent().window().labelCoordinates.setText(
                        'Width: %d, Height: %d / X: %d; Y: %d' % (currentWidth, currentHeight, int(pos.x()), int(pos.y())))
                
                color = self.drawingLineColor
                if self.outOfPixmap(pos):
                    # Don't allow the user to draw outside the pixmap.
                    # Project the point to the pixmap's edges.
                    try:
                        pos = self.intersectionPoint(self.current[-1], pos)
                    except:
                        pass
                elif self.drawingShapeMode == 'polygon' and len(self.current) > 1 and self.closeEnough(pos, self.current[0]):
                    # Attract line to starting point and colorise to alert the
                    # user:
                    pos = self.current[0]
                    print('type of pos:', type(pos))
                    color = self.current.line_color
                    self.overrideCursor(CURSOR_POINT)
                    self.current.highlightVertex(0, Shape.NEAR_VERTEX)

                if self.drawingShapeMode == 'bbox' and self.drawSquare:
                    initPos = self.current[0]
                    minX = int(initPos.x())
                    minY = int(initPos.y())
                    min_size = int(min(abs(pos.x() - minX), abs(pos.y() - minY)))
                    directionX = -1 if pos.x() - minX < 0 else 1
                    directionY = -1 if pos.y() - minY < 0 else 1
                    self.line[1] = QPointF(minX + directionX * min_size, minY + directionY * min_size)
                else:
                    self.line[1] = pos

                self.line.line_color = color
                self.prevPoint = QPointF()
                self.current.highlightClear()
            else:
                self.prevPoint = pos
            self.repaint()
            return

        # Polygon copy moving.
        if Qt.RightButton & ev.buttons():
            if self.selectedShapeCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShapeCopy, pos)
                self.repaint()
            elif self.selectedShape:
                self.selectedShapeCopy = self.selectedShape.copy()
                self.repaint()
            return

        # Polygon/Vertex moving.
        if Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.shapeMoved.emit()
                self.repaint()
            elif self.selectedShape and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShape, pos)
                self.shapeMoved.emit()
                self.repaint()
            return

        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        #self.setToolTip("Image")
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            idx = shape.nearestVertex(pos, self.epsilon)
            if (
                idx is not None
                and not (
                    getattr(shape, 'shape_type', '') == 'keypoints'
                    and getattr(shape, 'keypoints_bulk_select', False)
                    and shape.selected
                )
            ):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex, self.hShape = idx, shape
                shape.highlightVertex(idx, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                #self.setToolTip("Click & drag to move point")
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex, self.hShape = None, shape
                #self.setToolTip("Click & drag to move shape '%s'" % shape.label)
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape = None, None
            self.overrideCursor(CURSOR_DEFAULT)

    def mousePressEvent(self, ev):
        pos = self.transformPos(ev.pos())

        if ev.button() == Qt.LeftButton:
            if self.drawing():
                self.handleDrawing(pos)
            else:
                self.selectShapePoint(pos)
                self.prevPoint = pos
                if self.selectedShape and not self._move_undo_pushed:
                    self.undoCheckpoint.emit()
                    self._move_undo_pushed = True
                self.repaint()
        elif ev.button() == Qt.RightButton and self.editing():
            self.selectShapePoint(pos)
            self.prevPoint = pos
            if self.selectedShape and not self._move_undo_pushed:
                self.undoCheckpoint.emit()
                self._move_undo_pushed = True
            self.repaint()

    def _keypoint_vertex_at(self, pos):
        """Topmost visible keypoints shape with a vertex near pos; returns (shape, index) or (None, None)."""
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            if getattr(shape, 'shape_type', '') != 'keypoints':
                continue
            if getattr(shape, 'keypoints_bulk_select', False) and shape.selected:
                continue
            idx = shape.nearestVertex(pos, self.epsilon)
            if idx is not None:
                return shape, idx
        return None, None

    def mouseReleaseEvent(self, ev):
        if ev.button() in (Qt.LeftButton, Qt.RightButton):
            self._move_undo_pushed = False
        if ev.button() == Qt.RightButton:
            self.restoreCursor()
            if self.editing() and not self.selectedShapeCopy and self.keypointNames:
                pos = self.transformPos(ev.pos())
                hit_shape, hit_idx = self._keypoint_vertex_at(pos)
                if hit_shape is not None:
                    if self.selectedShape is not hit_shape:
                        self.selectShape(hit_shape)
                    hit_shape.highlightVertex(hit_idx, hit_shape.MOVE_VERTEX)
                    self.hVertex, self.hShape = hit_idx, hit_shape
                    self.keypointRenameRequested.emit(hit_shape, hit_idx)
                    self.repaint()
                    return
            menu = self.menus[bool(self.selectedShapeCopy)]
            if not menu.exec_(self.mapToGlobal(ev.pos()))\
               and self.selectedShapeCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapeCopy = None
                self.repaint()
        elif ev.button() == Qt.LeftButton and self.selectedShape:
            if self.selectedVertex():
                self.overrideCursor(CURSOR_POINT)
            else:
                self.overrideCursor(CURSOR_GRAB)

    def endMove(self, copy=False):
        assert self.selectedShape and self.selectedShapeCopy
        shape = self.selectedShapeCopy
        #del shape.fill_color
        #del shape.line_color
        if copy:
            self.shapes.append(shape)
            self.selectedShape.selected = False
            self.selectedShape = shape
            self.repaint()
        else:
            self.selectedShape.points = [p for p in shape.points]
        self.selectedShapeCopy = None

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShape:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.repaint()

    def handleDrawing(self, pos):
        if self.outOfPixmap(pos):
            return

        if not self.current:
            self.current = Shape(shape_type=self.drawingShapeMode)
            self.current.addPoint(pos)
            self.line.points = [pos, pos]
            self.setHiding()
            self.drawingPolygon.emit(True)
            self.update()
            return

        if self.drawingShapeMode == 'bbox':
            initPos = self.current[0]
            minX = int(initPos.x())
            minY = int(initPos.y())
            targetPos = self.line[1]
            maxX = int(targetPos.x())
            maxY = int(targetPos.y())
            self.current.addPoint(QPointF(maxX, minY))
            self.current.addPoint(targetPos)
            self.current.addPoint(QPointF(minX, maxY))
            self.finalise()
            return

        if self.drawingShapeMode == 'polygon':
            # Polygon mode: close when clicking near the first vertex, otherwise append a new point.
            if len(self.current) > 2 and self.closeEnough(pos, self.current[0]):
                self.finalise()
            else:
                self.current.addPoint(pos)
                self.line.points = [self.current[-1], self.current[0]]
                self.update()
            return

        # Keypoints mode: click to append keypoints, finalise with Enter / double click.
        if self.drawingShapeMode == 'keypoints':
            self.current.addPoint(pos)
            self.line.points = [self.current[-1], self.current[-1]]
            if self.maxKeypoints > 0 and len(self.current.points) >= self.maxKeypoints:
                self.finalise()
                return
            self.update()
            return

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        if not (self.drawing() and self.current):
            return False
        if self.drawingShapeMode == 'polygon':
            return len(self.current) > 2
        if self.drawingShapeMode == 'keypoints':
            return len(self.current) > 0
        return False

    def mouseDoubleClickEvent(self, ev):
        pos = self.transformPos(ev.pos())
        if ev.button() == Qt.LeftButton and self.drawing():
            if self.current and self.drawingShapeMode == 'polygon' and self.canCloseShape() and len(self.current) > 3:
                self.current.popPoint()
                self.finalise()
            elif self.current and self.drawingShapeMode == 'keypoints' and len(self.current) > 0:
                self.finalise()
            ev.accept()
            return
        if ev.button() == Qt.LeftButton and self.editing():
            for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
                if getattr(shape, 'shape_type', '') != 'keypoints':
                    continue
                br = shape.boundingRect()
                if br.width() <= 0.0 or br.height() <= 0.0:
                    continue
                if not br.contains(pos):
                    continue
                self.selectShape(shape)
                shape.keypoints_bulk_select = True
                self.unHighlight()
                self.hVertex, self.hShape = None, shape
                self.update()
                ev.accept()
                return
        super(Canvas, self).mouseDoubleClickEvent(ev)

    def selectShape(self, shape):
        self.deSelectShape()
        shape.selected = True
        self.selectedShape = shape
        self.setHiding()
        self.selectionChanged.emit(True)
        self.update()

    def selectShapePoint(self, point):
        """Select the first shape created which contains this point."""
        self.deSelectShape()
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            if getattr(shape, 'shape_type', '') != 'keypoints':
                continue
            idx = shape.nearestVertex(point, self.epsilon)
            if idx is not None:
                shape.keypoints_bulk_select = False
                shape.highlightVertex(idx, shape.MOVE_VERTEX)
                self.hVertex, self.hShape = idx, shape
                self.selectShape(shape)
                return
        for shape in reversed(self.shapes):
            if self.isVisible(shape) and shape.containsPoint(point):
                if getattr(shape, 'shape_type', '') == 'keypoints':
                    shape.keypoints_bulk_select = False
                self.hVertex, self.hShape = None, shape
                self.selectShape(shape)
                self.calculateOffsets(shape, point)
                return

    def calculateOffsets(self, shape, point):
        rect = shape.boundingRect()
        x1 = int(rect.x() - point.x())
        y1 = int(rect.y() - point.y())
        x2 = int((rect.x() + rect.width()) - point.x())
        y2 = int((rect.y() + rect.height()) - point.y())
        self.offsets = QPointF(x1, y1), QPointF(x2, y2)

    def snapPointToCanvas(self, x, y):
        """
        Moves a point x,y to within the boundaries of the canvas.
        :return: (x,y,snapped) where snapped is True if x or y were changed, False if not.
        """
        if x < 0 or x > self.pixmap.width() or y < 0 or y > self.pixmap.height():
            x = max(x, 0)
            y = max(y, 0)
            x = min(x, self.pixmap.width())
            y = min(y, self.pixmap.height())
            return x, y, True

        return x, y, False

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if getattr(shape, 'shape_type', '') == 'keypoints' and getattr(shape, 'keypoints_bulk_select', False):
            shape.keypoints_bulk_select = False
        if self.outOfPixmap(pos):
            try:
                pos = self.intersectionPoint(point, pos)
            except Exception as e:
                # print(e)
                pass

        if self.drawSquare and len(shape.points) == 4:
            opposite_point_index = (index + 2) % 4
            opposite_point = shape[opposite_point_index]

            min_size = int(min(abs(pos.x() - opposite_point.x()), abs(pos.y() - opposite_point.y())))
            directionX = -1 if pos.x() - opposite_point.x() < 0 else 1
            directionY = -1 if pos.y() - opposite_point.y() < 0 else 1
            shiftPos = QPointF(int(opposite_point.x() + directionX * min_size - point.x()),
                               int(opposite_point.y() + directionY * min_size - point.y()))
        else:
            shiftPos = pos - point

        shape.moveVertexBy(index, shiftPos)

        if getattr(shape, 'shape_type', '') == 'keypoints' and self.pixmap:
            while len(shape.keypoint_visibility) < len(shape.points):
                shape.keypoint_visibility.append(2)
            if index < len(shape.keypoint_visibility) and int(shape.keypoint_visibility[index]) == 0:
                if abs(shiftPos.x()) + abs(shiftPos.y()) > 2.5:
                    shape.keypoint_visibility[index] = 2

        if len(shape.points) == 4:
            lindex = (index + 1) % 4
            rindex = (index + 3) % 4
            lshift = None
            rshift = None
            if index % 2 == 0:
                rshift = QPointF(shiftPos.x(), 0)
                lshift = QPointF(0, shiftPos.y())
            else:
                lshift = QPointF(shiftPos.x(), 0)
                rshift = QPointF(0, shiftPos.y())
            shape.moveVertexBy(rindex, rshift)
            shape.moveVertexBy(lindex, lshift)

    def boundedMoveShape(self, shape, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QPointF(min(0, int(o1.x())), min(0, int(o1.y())))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QPointF(min(0, self.pixmap.width() - int(o2.x())),
                           min(0, self.pixmap.height() - int(o2.y())))
        # The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason. XXX
        #self.calculateOffsets(self.selectedShape, pos)
        dp = pos - self.prevPoint
        if dp:
            shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShape:
            if getattr(self.selectedShape, 'shape_type', '') == 'keypoints':
                self.selectedShape.keypoints_bulk_select = False
            self.selectedShape.selected = False
            self.selectedShape = None
            self.setHiding(False)
            self.selectionChanged.emit(False)
            self.update()

    def deleteSelected(self):
        if self.selectedShape:
            self.undoCheckpoint.emit()
        if self.selectedShape and getattr(self.selectedShape, 'shape_type', '') == 'keypoints' and self.selectedVertex():
            sh = self.selectedShape
            idx = self.hVertex
            if idx is not None and 0 <= idx < len(sh.points) and self.pixmap:
                while len(sh.keypoint_visibility) < len(sh.points):
                    sh.keypoint_visibility.append(2)
                sh.keypoint_visibility[idx] = 0
                spread_keypoint_placeholders(sh, self.pixmap.width(), self.pixmap.height())
                any_visible = False
                for i in range(len(sh.points)):
                    v = int(sh.keypoint_visibility[i]) if i < len(sh.keypoint_visibility) else 0
                    if v != 0:
                        any_visible = True
                        break
                if not any_visible:
                    self.shapes.remove(sh)
                    self.selectedShape = None
                    self.setHiding(False)
                    self.selectionChanged.emit(False)
                    self.unHighlight()
                    self.update()
                    return sh
                self.shapeMoved.emit()
                self.update()
                return None
        if self.selectedShape:
            shape = self.selectedShape
            self.shapes.remove(self.selectedShape)
            self.selectedShape = None
            self.update()
            return shape

    def copySelectedShape(self):
        if self.selectedShape:
            self.undoCheckpoint.emit()
            shape = self.selectedShape.copy()
            self.deSelectShape()
            self.shapes.append(shape)
            shape.selected = True
            self.selectedShape = shape
            self.boundedShiftShape(shape)
            return shape

    def boundedShiftShape(self, shape):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shape[0]
        offset = QPointF(2.0, 2.0)
        self.calculateOffsets(shape, point)
        self.prevPoint = point
        if not self.boundedMoveShape(shape, point - offset):
            self.boundedMoveShape(shape, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapeCopy:
            self.selectedShapeCopy.paint(p)

        # Paint rect
        if self.current is not None and len(self.line) == 2:
            leftTop = self.line[0]
            rightBottom = self.line[1]
            rectWidth = int(rightBottom.x() - leftTop.x())
            rectHeight = int(rightBottom.y() - leftTop.y())
            p.setPen(self.drawingRectColor)
            brush = QBrush(Qt.BDiagPattern)
            p.setBrush(brush)
            p.drawRect(int(leftTop.x()), int(leftTop.y()), rectWidth, rectHeight)

        if self.drawing() and not self.prevPoint.isNull() and not self.outOfPixmap(self.prevPoint):
            p.setPen(QColor(0, 0, 0))
            # print(self.prevPoint.x(), ":", type(self.prevPoint.x()))
            p.drawLine(int(self.prevPoint.x()), 0, int(self.prevPoint.x()), int(self.pixmap.height()))
            p.drawLine(0, int(self.prevPoint.y()), self.pixmap.width(), int(self.prevPoint.y()))

        self.setAutoFillBackground(True)
        if self.verified:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(184, 239, 38, 128))
            self.setPalette(pal)
        else:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(232, 232, 232, 255))
            self.setPalette(pal)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical coordinates."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w and 0 <= p.y() <= h)

    def finalise(self):
        assert self.current
        # Single-point keypoints: first == last index is the same point — must not treat as "empty closed shape".
        if self.drawingShapeMode != 'keypoints' and self.current.points[0] == self.current.points[-1]:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
            return

        if self.drawingShapeMode != 'keypoints':
            self.current.close()
        else:
            m = len(self.current.points)
            n = int(self.maxKeypoints) if self.maxKeypoints > 0 else m
            if n <= 0:
                n = m
            while len(self.current.keypoint_visibility) < m:
                self.current.keypoint_visibility.append(2)
            pad_keypoints_shape(self.current, n, self.keypointNames)
            self.current.skeleton = [tuple(p) for p in self.keypointSkeleton]
            if self.pixmap:
                spread_keypoint_placeholders(self.current, self.pixmap.width(), self.pixmap.height())
        self.undoCheckpoint.emit()
        self.shapes.append(self.current)
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        #d = distance(p1 - p2)
        #m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        return distance(p1 - p2) < self.epsilon

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width(), 0),
                  (size.width(), size.height()),
                  (0, size.height())]
        x1, y1 = int(p1.x()), int(p1.y())
        x2, y2 = int(p2.x()), int(p2.y())
        # print(x1, y1)
        # print(x2, y2)
        # print(self.intersectingEdges((x1, y1), (x2, y2), points))
              
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QPointF(min(max(0, x2), max(x3, x4)), y3)

        # Ensure the labels are within the bounds of the image. If not, fix them.
        x, y, _ = self.snapPointToCanvas(x, y)

        return QPointF(x, y)

    def intersectingEdges(self, x1y1, x2y2, points):
        """For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen."""
        x1, y1 = x1y1
        x2, y2 = x2y2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            # print("denom=", denom)
            # print('nua=', nua)
            # print('nub=', nub)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = distance(m - QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        qt_version = 4 if hasattr(ev, "delta") else 5
        if qt_version == 4:
            if ev.orientation() == Qt.Vertical:
                v_delta = ev.delta()
                h_delta = 0
            else:
                h_delta = ev.delta()
                v_delta = 0
        else:
            delta = ev.angleDelta()
            h_delta = delta.x()
            v_delta = delta.y()

        mods = ev.modifiers()
        shift = bool(int(mods) & int(Qt.ShiftModifier))
        if v_delta and not shift:
            self.zoomRequest.emit(v_delta)
        else:
            v_delta and self.scrollRequest.emit(v_delta, Qt.Vertical)
            h_delta and self.scrollRequest.emit(h_delta, Qt.Horizontal)
        ev.accept()

    def keyReleaseEvent(self, ev):
        if ev.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            self._arrow_undo_pushed = False
        super(Canvas, self).keyReleaseEvent(ev)

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_Escape and self.current:
            print('ESC press')
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key in (Qt.Key_Return, Qt.Key_Enter) and self.canCloseShape():
            self.finalise()
            ev.accept()
        elif key in (Qt.Key_0, Qt.Key_1, Qt.Key_2) and self.selectedShape and self.selectedVertex():
            if getattr(self.selectedShape, 'shape_type', '') == 'keypoints':
                idx = self.hVertex
                v = int(key - Qt.Key_0)
                while len(self.selectedShape.keypoint_visibility) < len(self.selectedShape.points):
                    self.selectedShape.keypoint_visibility.append(2)
                if idx is not None and 0 <= idx < len(self.selectedShape.keypoint_visibility):
                    self.selectedShape.keypoint_visibility[idx] = v
                    self.shapeMoved.emit()
                    self.repaint()
        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and self.selectedShape:
            if not self._arrow_undo_pushed:
                self.undoCheckpoint.emit()
                self._arrow_undo_pushed = True
            if key == Qt.Key_Left:
                self.moveOnePixel('Left')
            elif key == Qt.Key_Right:
                self.moveOnePixel('Right')
            elif key == Qt.Key_Up:
                self.moveOnePixel('Up')
            else:
                self.moveOnePixel('Down')

    def moveOnePixel(self, direction):
        # print(self.selectedShape.points)
        if direction == 'Left' and not self.moveOutOfBound(QPointF(-1.0, 0)):
            step = QPointF(-1.0, 0)
        elif direction == 'Right' and not self.moveOutOfBound(QPointF(1.0, 0)):
            step = QPointF(1.0, 0)
        elif direction == 'Up' and not self.moveOutOfBound(QPointF(0, -1.0)):
            step = QPointF(0, -1.0)
        elif direction == 'Down' and not self.moveOutOfBound(QPointF(0, 1.0)):
            step = QPointF(0, 1.0)
        else:
            return

        for i in range(len(self.selectedShape.points)):
            self.selectedShape.points[i] += step
        self.shapeMoved.emit()
        self.repaint()

    def moveOutOfBound(self, step):
        points = [p + step for p in self.selectedShape.points]
        return True in map(self.outOfPixmap, points)

    def setLastLabel(self, text, line_color  = None, fill_color = None):
        assert text
        self.shapes[-1].label = text
        if line_color:
            self.shapes[-1].line_color = line_color

        if fill_color:
            self.shapes[-1].fill_color = fill_color

        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.line.points = [self.current[-1], self.current[0]]
        self.drawingPolygon.emit(True)

    def resetAllLines(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.line.points = [self.current[-1], self.current[0]]
        self.drawingPolygon.emit(True)
        self.current = None
        self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        self.repaint()

    def loadShapes(self, shapes):
        self.shapes = list(shapes)
        self.current = None
        self.repaint()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def currentCursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    def overrideCursor(self, cursor):
        self._cursor = cursor
        if self.currentCursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    def restoreCursor(self):
        QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.update()

    def setDrawingShapeToSquare(self, status):
        self.drawSquare = status

    def setDrawingShapeMode(self, mode):
        if mode in ('bbox', 'polygon', 'keypoints'):
            self.drawingShapeMode = mode

    def setKeypointTemplate(self, names, skeleton):
        self.keypointNames = list(names)
        self.keypointSkeleton = [tuple(x) for x in skeleton]
        self.maxKeypoints = len(self.keypointNames)

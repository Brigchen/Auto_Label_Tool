# -*- coding: utf-8 -*-
"""
__version__: v1.1214
Created on Mon Nov 27 08:35:35 2023
update:
20231214: counting function with half transparency table.
@author: brigc
"""

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from libs.win_dll_search_path import register_repo_libs_dll_path

register_repo_libs_dll_path()
import cv2
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal#, QPoint
from PyQt5.QtGui import QIcon, QPixmap, QImage, QFont#, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGridLayout, QLabel, QWidget, QPushButton, QTableWidget, \
    QLineEdit, QProgressBar, QCheckBox, QHBoxLayout, QRadioButton, QDesktopWidget, QDialog, QVBoxLayout, \
    QTableWidgetItem, QHeaderView
import configparser
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from ultralytics import YOLO
# from xml.dom.minidom import Document
# import csv
from datetime import datetime
# from ultralytics.data.utils import VID_FORMATS
# import qdarkstyle
# from qdarkstyle.light.palette import LightPalette
from PIL import Image, ImageDraw, ImageFont
from libs.fish_enhance import gc_enhance_new
from libs.pkg_paths import resolve_cjk_plot_font
from libs.repo_paths import configs_dir, repo_root
# import imageio as iio
import random
import numpy as np
# import pyqtgraph as pg
import pandas as pd
from ultralytics.utils.plotting import colors

#%%
# from func_timeout import func_set_timeout, exceptions

def check_rtsp_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"无法连接到 RTSP 流 {url}")
        return False
    else:
        print(f"成功连接到 RTSP 流 {url}")
        cap.release()
        return True
    
#%%
import re
from xpinyin import Pinyin

def check_chinese(fl):
    print('36: check_chinese')
    pinyin = Pinyin()
    codes = '[‘’！~#￥%……&*（）{}|；？、，,;~!@#$%^&*() -+]'
    # for fl in os.listdir(dr):
    new_fl = fl
    if not pinyin.get_pinyin(fl, '').isalpha():
        try:
            new_fl = pinyin.get_pinyin(fl)
            new_fl = re.sub(codes, '_', new_fl)
            new_fl = new_fl.replace('___','_').replace('__','_')
            # os.rename(os.path.join(dr,fl),os.path.join(dr,new_fl))
        except Exception as e:
            print(e)
    return new_fl
                
#%%
def xyxy2xywhn(x, w=640, h=640):
    print('53: xyxy2xywhn')
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format.
    x, y, width and height are normalized to image dimensions

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        # eps (float): The minimum value of the box's width and height. Defaults to 0.0
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
    """
    y = np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def plot_one_box_pil(x, img, color=None, label=None, width=4, font_path=None, font_size=30):
    print("75: plot")
    draw = ImageDraw.Draw(img)
    if font_path is None:
        font_path = resolve_cjk_plot_font(os.path.dirname(os.path.abspath(__file__)))
    if font_path and os.path.isfile(font_path):
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    else:
        font = ImageFont.load_default()
    # color = tuple(color if color is not None else [random.randint(0, 128) for _ in range(3)])
    color = tuple(color if color is not None else colors(random.randint(0,128)))

    xyxy = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
    draw.rectangle(xyxy, outline=color, width=width)
    if label is not None:
        _, _, _, char_h = font.getbbox(label)
        draw.text((xyxy[0], xyxy[1]-char_h), label, font=font, fill=color)

    return img

#%%

class VideoDetectionThread(QThread):
    logger = pyqtSignal(str)
    frameChanged = pyqtSignal(QImage)
    progressChanged = pyqtSignal(list)
    counter = pyqtSignal(dict)

    def __init__(self, videos, output_path, model_weights, fps, 
                 confidence, iou, savResult, rtsp_check = 0, rtsp = None, 
                 enhance = 0, detect = 1, tracking = 0, counting = 0):
        super().__init__()
        print('99: initiate videodetectionthread')
        self.videos = videos
        self.output_path = output_path
        self.rtsp = rtsp
        self.rtsp_check = rtsp_check
        self.model_weights = model_weights
        self.fps = fps
        self.confidence = confidence
        self.iou = iou
        self.savResult = savResult
        self.stopped = False
        self.enhance = enhance
        self.detect = detect
        self.tracking = tracking
        self.counting = counting
        self.detections = {'鱼': 0}


    def run(self):
        # 进行目标检测
        print('118: run videodetectionthread')
        from libs.yolo_weights import resolve_yolo_checkpoint
        from libs.repo_paths import repo_root

        w = resolve_yolo_checkpoint(
            self.model_weights, app_weights_dir=os.path.join(repo_root(), "weights"))
        if w != self.model_weights:
            print("[FishVision] resolved weights:", self.model_weights, "->", w)
        model = YOLO(w)
        # model.half()
        names = model.module.names if hasattr(model, 'module') else model.names
        # colors = [[random.randint(0, 100) for _ in range(3)] for _ in range(len(names))]
        class_dict = {}
        for cnt in range(len(names)):
            class_dict[str(cnt)] = names[cnt]
        
        if self.rtsp_check and self.rtsp:
            if check_rtsp_stream(self.rtsp):
                videos = [self.rtsp]
            else:
                self.logger.emit('error with opening %s'%self.rtsp)
                self.cancel()
                return
        else:
            videos = self.videos
        
        for videoDir in videos: 
            try:                    
                    # videoDir = videoDir.encode(sys.getfilesystemencoding())                   
                self.logger.emit('Initiating annotating %s'%videoDir)
                if not self.rtsp_check:
                # if videoDir.split('.')[-1].lower() in VID_FORMATS:
                    video = os.path.basename(videoDir)
                    video_name_save = check_chinese(os.path.splitext(video)[0]) + '_annotated.mp4'
                else:
                    video_name_save = r'rtsp_%s_annotated.mp4'%(datetime.now().strftime("%Y%m%d%H%M%S"))
                cap = cv2.VideoCapture(videoDir)
                
                print('Annotating:%s'%videoDir)
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc =  cv2.VideoWriter_fourcc(*'H264')
                out_fps = self.fps
                out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                count = int(cap.get(7))
                if self.rtsp_check:
                    count = 1000
            
                # 创建输出视频的对象
                if self.savResult:
                    print('182: set up save video')
                    self.out = cv2.VideoWriter(os.path.join(self.output_path, video_name_save), fourcc, out_fps, (out_width, out_height))
            
                frame_cnt = 0
                # pbar = tqdm(total=count, desc="Progressing:") 
                freq = int(fps/out_fps)
                if freq < 1:
                    freq = 1
                i = 0       
                print('191: freq = ', freq)
                fish_id_exist = {}
                df_id_cls = pd.DataFrame(columns=['id','cls'])
                plot_font = resolve_cjk_plot_font(os.path.dirname(os.path.abspath(__file__)))

                def _reset_trackers():
                    pred = getattr(model, "predictor", None)
                    if pred is None:
                        return
                    for t in getattr(pred, "trackers", None) or []:
                        reset_fn = getattr(t, "reset", None)
                        if callable(reset_fn):
                            reset_fn()

                _reset_trackers()
                while cap.isOpened() and not self.stopped:
                    i += 1
                    # Read a frame from the video
                    success, frame = cap.read()
                    if success:
                        if i%freq == 1: 
                            if self.enhance:
                                enhanced_frame = gc_enhance_new(frame)
                                annotated_frame = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
                            else:
                                enhanced_frame = frame
                                annotated_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                                                
                            if self.tracking or self.detect:            
                                res = model.track(
                                    frame,
                                    conf=self.confidence,
                                    iou=self.iou,
                                    persist=True,
                                    tracker="bytetrack.yaml",
                                    verbose=False,
                                )
                                r = res[0]
                                boxes = r.boxes
                                if boxes is not None and len(boxes) > 0:
                                    xyxy = boxes.xyxy.cpu().numpy()
                                    cls_arr = boxes.cls.cpu().numpy().astype(int)
                                    conf_arr = boxes.conf.cpu().numpy()
                                    if boxes.id is not None:
                                        tid_arr = boxes.id.cpu().numpy().astype(int)
                                    else:
                                        tid_arr = np.arange(len(xyxy), dtype=int)
                                    for j in range(len(xyxy)):
                                        cls = int(cls_arr[j])
                                        track_id = int(tid_arr[j])
                                        conf = float(conf_arr[j])
                                        fish_id_exist[track_id] = [cls, conf, colors(cls)]
                                        id_cls = {"id": track_id, "cls": cls}
                                        df_id_cls = pd.concat(
                                            [df_id_cls, pd.DataFrame([id_cls])], axis=0, ignore_index=True
                                        )
                                        if self.tracking:
                                            annotated_frame = plot_one_box_pil(
                                                xyxy[j],
                                                annotated_frame,
                                                color=fish_id_exist[track_id][2],
                                                label=f"id{track_id}:{names[cls]} {conf:.2f}",
                                            )
                                    if self.detect and not self.tracking:
                                        if self.enhance:
                                            bgr_ann = (
                                                r.plot(img=enhanced_frame, font=plot_font)
                                                if plot_font
                                                else r.plot(img=enhanced_frame)
                                            )
                                        else:
                                            bgr_ann = r.plot(font=plot_font) if plot_font else r.plot()
                                        annotated_frame = Image.fromarray(cv2.cvtColor(bgr_ann, cv2.COLOR_BGR2RGB))
                            if self.savResult:
                                save_frame = cv2.cvtColor(np.asarray(annotated_frame), cv2.COLOR_RGB2BGR)
                                self.out.write(save_frame)
                                print('246: save...')
                            if self.counting:
                                ids = np.unique(df_id_cls['id'])
                                print('248: length of ids ', len(ids))
                                detections={}
                                for pid in ids:
                                    df_id = df_id_cls[df_id_cls['id']==pid]
                                    clses, cnts = np.unique(df_id['cls'], return_counts=True)
                                    if len(clses) > 1:                                
                                        cnts_dict = dict(zip(clses, cnts))                                        
                                        non_fish_dict = {key: value for key, value in cnts_dict.items() if key != 0}
                                        # 找到数量最多的键及其对应的数量
                                        max_cls = max(non_fish_dict, key=non_fish_dict.get)
                                        detections[pid] = names[max_cls]
                                    elif len(clses) == 1:
                                        detections[pid] = names[clses[0]]

                                species, counts = np.unique(list(detections.values()), return_counts=True)
                                print('263: length of species ', len(species) )
                                self.detections = dict(zip(species, counts))
                                print(self.detections)
                                self.counter.emit(self.detections)
                                        
                            rgb_image = np.asarray(annotated_frame)
                            h, w, ch = rgb_image.shape
                            bytes_per_line = ch * w
                            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                            p = convert_to_qt_format.scaled(1280, 700, Qt.KeepAspectRatio)
                            self.frameChanged.emit(p)
                            
                            if self.rtsp_check:
                                self.progressChanged.emit([0,0])
                            else:
                                self.progressChanged.emit([count,i+1])
        
                            frame_cnt += 1
                        # i += 1
                    else:
                            # Break the loop if the end of the video is reached
                        break
                
                # Release the video capture object and close the display window
                cap.release()
                if self.savResult:
                    self.out.release()
                print('finishing with %s'%videoDir)
                log = 'finished:%s'%videoDir  # [(frame_id, bbox), ...]
                if self.rtsp_check:
                    self.progressChanged.emit([0,0])
                else:
                    self.progressChanged.emit([count,count])
                self.logger.emit(log)


            except Exception as e:
                print(e)

        log = 'Finish all videos'
        self.logger.emit(log)

    def cancel(self):
        print('306: cancel')
        self.logger.emit('stopping current treading')
        if self.savResult and self.out:
            self.out.release()
            print('310: save video closed')
        # self.progressChanged.emit([0,])
        self.stopped = True
        # cap.release()

        
    def show_raw(self):
        print('249: raw')
        self.enhance = 0

    def show_enhance(self):
        print('253: enhance')
        self.enhance = 1
        
    def show_detect(self):
        print('256: detect')
        self.detect = 1
        self.tracking = 0
        
    def show_track(self):
        print('262: track')
        self.tracking = 1
        
    def off_detect(self):
        # self.enhance = 0
        print('265: off')
        self.detect = 0
        self.tracking = 0
        
    def show_counts(self):
        print('272: count')
        self.counting = 1

    
    def off_counts(self):
        print('276: off')
        self.counting = 0
        # self.detection_dialog.hide()

#%%        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        print('282: init main')
        # 初始化变量
        self.video_path = ''
        self.videos = []
        self.output_path = ''
        self.model_weights = ''
        self.rtsp = ''
        self.fps = 30
        self.confidence = 0.25
        self.iou = 0.5
        self.savResult = 0
        self.rtsp_check = 0
        self.enhance = 0
        self.detect = 0
        self.tracking = 0
        self.counting = 0
        self.video_detection_thread = None
        self.detection_dialog = DetectionDialog()

        self.detections = {'鱼': 0}
        # 初始化界面
        
        self.init_ui()

    def init_ui(self):
        # 设置窗口标题和图标
        # print('306: init ui')
        self.setWindowTitle('Fish Video Detection')
        fv_icon = os.path.join(repo_root(), 'resources', 'icons', 'fishvision_v2_s.png')
        self.setWindowIcon(QIcon(fv_icon))

        # self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
        self.setFont(QFont("Microsoft YaHei", 9))
        # 设置窗口大小
        # Get the screen geometry
        screenGeometry = QDesktopWidget().screenGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        
        # Calculate the position for the window to be vertically at the top and horizontally centered
        windowWidth = 1280
        windowHeight = 800
        x = (screenGeometry.width() - windowWidth) // 2
        y = 25
        self.setGeometry(x, y, windowWidth, windowHeight)
        qr = self.frameGeometry()
        qr.moveCenter(cp)
        # print('392: ',qr.topLeft())
        self.move(qr.topLeft())
        self.detection_dialog.move(self.pos().x() + windowWidth - self.detection_dialog.size().width(), self.pos().y())

        # 创建主窗口的中心部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建布局
        grid_layout = QGridLayout()

        # 创建控件
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(QSize(1280, 680))
        grid_layout.addWidget(self.video_label, 0, 0, 1, 6)

        # 创建进度条
        self.status_bar = QLabel(self)
        grid_layout.addWidget(self.status_bar, 1, 0, 1, 6)

        # 创建进度条
        self.progress_bar = QProgressBar(self)
        grid_layout.addWidget(self.progress_bar, 2, 0, 1, 6)

        # First group of switches for video type
        videoTypeGroup = QGroupBox("视频增强(Enhancement)")
        videoTypeLayout = QHBoxLayout()
        # switchGroup1 = QHBoxLayout()
        self.originalVideoButton = QRadioButton('原视频(Raw)')
        self.originalVideoButton.setChecked(True)
        self.originalVideoButton.toggled.connect(self.show_raw)
        videoTypeLayout.addWidget(self.originalVideoButton)
      
        self.enhancedVideoButton = QRadioButton('增加(Enhance)')
        self.enhancedVideoButton.toggled.connect(self.show_enhance)
        videoTypeLayout.addWidget(self.enhancedVideoButton)

        videoTypeGroup.setLayout(videoTypeLayout)
        
        # Second group of switches for detection mode
        detectionModeGroup = QGroupBox("检测模式(Detection Modes)")
        detectionModeLayout = QHBoxLayout()
        self.offDetectionButton = QRadioButton('无检测(No Detect)')
        self.offDetectionButton.setChecked(True)
        self.offDetectionButton.toggled.connect(self.off_detect)
        detectionModeLayout.addWidget(self.offDetectionButton)

        self.objectDetectionButton = QRadioButton('检测(Detect)')
        self.objectDetectionButton.toggled.connect(self.show_detect)
        detectionModeLayout.addWidget(self.objectDetectionButton)

        self.objectTrackingButton = QRadioButton('追踪(Track)')
        self.objectTrackingButton.toggled.connect(self.show_track)
        detectionModeLayout.addWidget(self.objectTrackingButton)
        detectionModeGroup.setLayout(detectionModeLayout)
        # switchGroup.addWidget(detectionModeGroup)

        grid_layout.addWidget(videoTypeGroup, 3, 0, 1, 2)
        grid_layout.addWidget(detectionModeGroup, 3, 2, 1, 3)
        
        
        #创建输出结果复选框
        countModeGroup = QGroupBox("统计")
        countModeLayout = QHBoxLayout()
        self.count_checkbox = QCheckBox("计数(Counting)", self)
        self.count_checkbox.setChecked(self.counting)
        self.count_checkbox.stateChanged.connect(self.show_counts)
        countModeLayout.addWidget(self.count_checkbox)
        countModeGroup.setLayout(countModeLayout)
        grid_layout.addWidget(countModeGroup, 3, 5)

        # RTSP/RTMP
        self.rtsp_checkbox = QCheckBox(r'Open RTSP/RTMP', self)
        self.rtsp_checkbox.setChecked(self.rtsp_check)
        self.rtsp_checkbox.stateChanged.connect(self.on_rtsp_checkbox_changed)
        grid_layout.addWidget(self.rtsp_checkbox, 4, 0)

        self.rtsp_line_edit = QLineEdit(self)
        self.rtsp_line_edit.editingFinished.connect(self.rtsp_changed)
        grid_layout.addWidget(self.rtsp_line_edit, 4, 1,1,5)

        open_video_button = QPushButton('Open videos', self)
        open_video_button.clicked.connect(self.open_video)
        grid_layout.addWidget(open_video_button, 5, 0)

        self.video_path_line_edit = QLineEdit(self)
        self.video_path_line_edit.setReadOnly(True)
        grid_layout.addWidget(self.video_path_line_edit, 5, 1,1,5)

        open_output_button = QPushButton('Open save path', self)
        open_output_button.clicked.connect(self.open_output)
        grid_layout.addWidget(open_output_button, 6, 0)

        self.output_path_line_edit = QLineEdit(self)
        self.output_path_line_edit.setReadOnly(True)
        grid_layout.addWidget(self.output_path_line_edit, 6, 1,1,5)

        load_model_button = QPushButton('Load model weights', self)
        load_model_button.clicked.connect(self.load_model)
        grid_layout.addWidget(load_model_button, 7, 0)

        self.model_line_edit = QLineEdit(self)
        self.model_line_edit.setReadOnly(True)
        grid_layout.addWidget(self.model_line_edit, 7, 1, 1, 5)

        # 创建帧率输入框
        self.fps_label = QLabel("FPS(int):", self)
        grid_layout.addWidget(self.fps_label, 8, 0)
        self.fps_line_edit = QLineEdit(self)
        self.fps_line_edit.setValidator(QIntValidator(0, 1000))
        self.fps_line_edit.editingFinished.connect(self.fps_changed)
        grid_layout.addWidget(self.fps_line_edit, 8, 1)

        # 创建置信度输入框
        self.confidence_label = QLabel("Confidence(0-1):", self)
        grid_layout.addWidget(self.confidence_label, 8, 2)
        self.confidence_line_edit = QLineEdit(self)
        self.confidence_line_edit.setValidator(QDoubleValidator(0,1,2))
        self.confidence_line_edit.editingFinished.connect(self.confidence_changed)
        grid_layout.addWidget(self.confidence_line_edit, 8, 3)

        # 创建IOU输入框
        self.iou_label = QLabel("IoU(0-1):", self)
        grid_layout.addWidget(self.iou_label, 8, 4)
        self.iou_line_edit = QLineEdit(self)
        self.iou_line_edit.setValidator(QDoubleValidator(0,1,2))
        self.iou_line_edit.editingFinished.connect(self.iou_changed)
        grid_layout.addWidget(self.iou_line_edit, 8, 5)

        # 创建复选框
        # self.enable_tracking_checkbox = QCheckBox("Enable tracking", self)
        # self.enable_tracking_checkbox.setChecked(self.tracking)
        # self.enable_tracking_checkbox.stateChanged.connect(self.on_tracking_checkbox_changed)
        # grid_layout.addWidget(self.enable_tracking_checkbox, 8, 2)

        #创建输出结果复选框
        self.enable_output_checkbox = QCheckBox("输出检测视频(Save labeled video)", self)
        self.enable_output_checkbox.setChecked(self.savResult)
        self.enable_output_checkbox.stateChanged.connect(self.on_output_checkbox_changed)
        grid_layout.addWidget(self.enable_output_checkbox, 9, 0)

        self.start_detection_button = QPushButton('开始(Start)', self)
        self.start_detection_button.clicked.connect(self.start_detection)
        grid_layout.addWidget(self.start_detection_button, 9, 1, 1, 3)


        self.cancel_detection_button = QPushButton('取消(Cancel)', self)
        self.cancel_detection_button.setEnabled(False)  # Set cancel button disabled by default
        self.cancel_detection_button.clicked.connect(self.cancel_detection)
        grid_layout.addWidget(self.cancel_detection_button, 9, 4, 1, 2)

        # 设置布局
        central_widget.setLayout(grid_layout)
        self.load_config()

    def load_config(self):
        print('474: load config')
        config = configparser.ConfigParser()
        _cfg = os.path.join(configs_dir(), "config.ini")
        if os.path.exists(_cfg):
            try:
                config.read(_cfg)
                self.video_path = config.get('Settings', 'video_path')
                self.output_path = config.get('Settings', 'output_path')
                self.model_weights = config.get('Settings', 'model_weights')
                self.fps = int(config.get('Settings', 'fps'))
                self.confidence = float(config.get('Settings', 'confidence'))
                self.iou = float(config.get('Settings', 'iou'))
                self.tracking = int(config.get('Settings','tracking'))
                self.detect = int(config.get('Settings', 'detect'))
                self.enhance = int(config.get('Settings', 'enhance'))
                self.savResult = int(config.get('Settings', 'savresult'))
                self.rtsp_check = int(config.get('Settings', 'rtsp_check'))
                self.rtsp = config.get('Settings', 'rtsp')
                self.counting = int(config.get("Settings", 'counting'))
                
                self.video_path_line_edit.setText(self.video_path)
                self.output_path_line_edit.setText(self.output_path)
                self.model_line_edit.setText(self.model_weights)
                self.fps_line_edit.setText(str(self.fps))
                self.confidence_line_edit.setText(str(self.confidence))
                self.iou_line_edit.setText(str(self.iou))
                self.enable_output_checkbox.setChecked(self.savResult)
                self.objectTrackingButton.setChecked(self.tracking)
                self.objectDetectionButton.setChecked(self.detect)
                self.enhancedVideoButton.setChecked(self.enhance)
                self.rtsp_checkbox.setChecked(self.rtsp_check)
                self.rtsp_line_edit.setText(self.rtsp)
                self.count_checkbox.setChecked(self.counting)
                
            except Exception as e:
                print(u'导入配置参数失败:', e)
                self.status_bar.setText(u'导入配置参数失败')
                
    def save_config(self):
        print('512: save')
        config = configparser.ConfigParser()
        config['Settings'] = {
            'video_path': self.video_path,
            'output_path': self.output_path,
            'model_weights': self.model_weights,
            'fps': self.fps,
            'confidence': self.confidence,
            'iou': self.iou,
            'savresult':self.savResult,
            'rtsp_check':self.rtsp_check,
            'rtsp': self.rtsp,
            'enhance':self.enhance,
            'detect':self.detect,
            'tracking':self.tracking,
            'counting':self.counting
        }
        with open(os.path.join(configs_dir(), "config.ini"), "w") as configfile:
            config.write(configfile)
            

    def open_video(self):
        print('534: open')
        # 打开视频文件夹
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Video Files (*.mp4 *.avi *.mov)')
        # file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)  # 允许选择多个文件
        if not self.video_path:
            default_video_path = r"D:\Python\BJ_Fish\Videos"  # Set default path here
        else:
            default_video_path = self.video_path
        file_dialog.setDirectory(default_video_path)
        if file_dialog.exec_():
            self.videos = file_dialog.selectedFiles()
            self.video_path = os.path.dirname(file_dialog.selectedFiles()[0])
            self.video_path_line_edit.setText(self.video_path)
            try:
                video_capture = cv2.VideoCapture(self.videos[0])

                ret, frame = video_capture.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # rgb_image = np.asarray(annotated_frame)
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    image = convert_to_qt_format.scaled(1280, 700, Qt.KeepAspectRatio)
                    # image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    # pixmap = image.scaled(1280, 700, Qt.KeepAspectRatio)
                    pixmap = QPixmap.fromImage(image)
                    self.video_label.setPixmap(pixmap)
                    # self.video_label.setScaledContents(True)
            except Exception as e:
                print(e)
                self.status_bar.setText('Error for opening: %s'%self.videos[0])

    def open_output(self):
        print('551: open')
        # 打开输出文件夹
        dir_dialog = QFileDialog(self)
        dir_dialog.setFileMode(QFileDialog.Directory)
        if not self.output_path:
            default_save_path = r"D:\Python\BJ_Fish\Videos"  # Set default path here
        else:
            default_save_path = self.video_path
        dir_dialog.setDirectory(default_save_path)
        if dir_dialog.exec_():
            self.output_path = dir_dialog.selectedFiles()[0]
            self.output_path_line_edit.setText(self.output_path)

    def load_model(self):
        print('565: load')
        # 打开视频文件夹
        weight_dialog = QFileDialog(self)
        weight_dialog.setNameFilter('Pytoch model weight files (*.pt *.pth *.engine *.onnx)')
        # file_dialog.setFileMode(QFileDialog.ExistingFile)
        weight_dialog.setFileMode(QFileDialog.ExistingFile)
        default_weight_path = os.path.join(os.getcwd(), "weights")  # Set default path here
        weight_dialog.setDirectory(default_weight_path)
        
        if weight_dialog.exec_():
            self.model_weights = weight_dialog.selectedFiles()[0]
            self.model_line_edit.setText(self.model_weights)

    def fps_changed(self):
        print('579: fps')
        try:
            self.fps = int(self.fps_line_edit.text())
        except:
            print('输入错误，限于整数')

    def confidence_changed(self):
        print('586: confidence')
        try:
            self.confidence = float(self.confidence_line_edit.text())
        except:
            print('输入错误，限于0-1')

    def iou_changed(self):
        print('593: iou')
        try:
            self.iou = float(self.iou_line_edit.text())
        except:
            print('输入错误，限于0-1')
            
    def rtsp_changed(self):
        print('600: rtsp')
        try:            
            rtsp = self.rtsp_line_edit.text()
            if 'rtmp://'in rtsp or 'rstp://' in rtsp:
                self.rtsp = rtsp
            else:
                self.status_bar.setText(u'输入地址不是网络流媒体格式，请检查')
        except:
            print('输入错误, 请输入网络流媒体体链接')        

    def on_rtsp_checkbox_changed(self, state):
        print('611: rtsp box')
        if state == 2:  # 2 represents checked state
            self.rtsp_check = 1
        else:
            self.rtsp_check = 0

    def on_output_checkbox_changed(self, state):
        print('618: out box')
        if state == 2:  # 2 represents checked state
            self.savResult = 1
        else:
            self.savResult = 0

    def on_tracking_checkbox_changed(self, state):
        print('625: track box')
        if state == 2:  # 2 represents checked state
            self.tracking = 1
        else:
            self.tracking = 0

    def start_detection(self):
        print('632: det')
        if self.rtsp_check and not self.rtsp:
            print(u'请输入网络流媒体地址')
            self.status_bar.setText(u'请输入网络流媒体地址')            
            return
        
        if not self.rtsp_check and not self.videos:
            self.status_bar.setText(u'请选择视频文件，可以多选')  
            return

        if self.savResult and not self.output_path:
            print(u'请指定保存文件夹')
            self.status_bar.setText(u'请指定保存文件夹')
            return

        if not self.model_weights:
            self.status_bar.setText(u'请选择模型权重文件')  
            return

        if self.video_detection_thread and self.video_detection_thread.isRunning():
            return

        # 创建线程进行目标检测
        self.save_config()
        # self.tracking = self.enable_tracking_checkbox.isChecked()
        self.video_detection_thread = VideoDetectionThread(self.videos, self.output_path, self.model_weights, self.fps,
                                                            self.confidence, self.iou, self.savResult,
                                                            self.rtsp_check, self.rtsp, self.enhance, self.detect, self.tracking,
                                                            self.counting)
        self.video_detection_thread.frameChanged.connect(self.update_frame)
        self.video_detection_thread.logger.connect(self.show_log)
        self.video_detection_thread.progressChanged.connect(self.update_progress_bar)
        self.video_detection_thread.counter.connect(self.update_detection_results)
        # self.video_detection_thread.finished.connect(self.show_log)
        self.video_detection_thread.start()
        self.start_detection_button.setEnabled(False)  # Disable start button
        self.cancel_detection_button.setEnabled(True)  # Enable cancel button
        

    def cancel_detection(self):
        print('671: cancel')
        if self.video_detection_thread and self.video_detection_thread.isRunning():
            self.video_detection_thread.cancel()
            self.cancel_detection_button.setEnabled(False)  # Disable cancel button
            self.start_detection_button.setEnabled(True)  # Enable start button
            self.progress_bar.reset()
            
    def show_raw(self):
        print('678: raw')
        self.enhance = 0
        # self.originalVideoButton.setChecked(True)
        if self.video_detection_thread and self.video_detection_thread.isRunning():
            self.video_detection_thread.show_raw()
            
    def show_enhance(self):
        print('685: enh')
        self.enhance = 1
        # self.enhancedVideoButton.setChecked(True)
        if self.video_detection_thread and self.video_detection_thread.isRunning():
            self.video_detection_thread.show_enhance()
            
    def show_detect(self):
        print('692: det')
        self.detect = 1
        self.tracking = 0
        # self.objectDetectionButton.setChecked(True)
        if self.video_detection_thread and self.video_detection_thread.isRunning():
            self.video_detection_thread.show_detect()
            
    def show_track(self):
        print('700: track')
        self.detect = 0
        self.tracking = 1
        # self.objectTrackingButton.setChecked(True)
        if self.video_detection_thread and self.video_detection_thread.isRunning():
            self.video_detection_thread.show_track()       
        
    def off_detect(self):
        print('749: off ')
        self.detect = 0
        self.tracking = 0
        # self.offDetectionButton.setChecked(1)
        
        if self.video_detection_thread and self.video_detection_thread.isRunning():
            self.video_detection_thread.off_detect()       
        
    def show_counts(self, state):
        print('757: count')
        
        if state == 2:  # 2 represents checked state
            self.counting = 1
            self.detection_dialog.show()
            if self.video_detection_thread and self.video_detection_thread.isRunning():          
                self.video_detection_thread.show_counts()
        else:
            self.counting = 0
            self.detection_dialog.hide()
            if self.video_detection_thread and self.video_detection_thread.isRunning():
                self.video_detection_thread.off_counts()

    def update_detection_results(self, detections):
        self.detections = detections
        if self.detection_dialog.isVisible():
            self.detection_dialog.update_plot(self.detections)
            
    def update_frame(self, frame):
        print('776: update')
        self.video_label.setPixmap(QPixmap.fromImage(frame))

    def show_log(self, log):
        print('781: log')
        # ...
        self.status_bar.setText(log)
        if log == 'Finish all videos':
            self.cancel_detection_button.setEnabled(False)  # Disable cancel button
            self.start_detection_button.setEnabled(True)  # Enable start button
        
    def update_progress_bar(self, progress):
        print('789: pbar')
        self.progress_bar.setRange(0,progress[0])
        self.progress_bar.setValue(progress[1])
    
    def closeEvent(self, event):
        self.save_config()
        self.cancel_detection()
        if self.detection_dialog.isVisible():
            self.detection_dialog.close()
        event.accept()
    
class DetectionDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        # font = QFont("SimHei", 12)
        self.setWindowTitle("鱼类数量统计（Statistics）")
        self.setFont(QFont("SimHei", 12))
        self.setGeometry(500, 20, 250, 600)  # 设置弹窗初始位置和大小
        # self.main_window_pos = main_window_pos
        # 设置初始位置为屏幕的右上角
        # desktop = QDesktopWidget()
        # screen_rect = desktop.availableGeometry()
        # self.move(screen_rect.right() - self.width() - 300, 50)
        
        # 设置初始位置为主窗口的右上角
        # main_window_pos = parent.pos()
        # main_window_size = parent.size()
        # dialog_size = self.size()
        # self.move(main_window_pos.x() + main_window_size.width() - dialog_size.width(), main_window_pos.y())

        layout = QVBoxLayout()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)  # 设置无边框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置半透明效果
        self.detections = {'鱼':0}
        
        # 创建表格
        self.table_widget = QTableWidget()
        self.table_widget.setShowGrid(False)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(['鱼类', '数量'])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) # 设置列宽度自适应
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setStyleSheet(
            "QHeaderView::section { background-color: #f3f3f3; font-weight: bold; font-size: 13px; }"
            "QTableWidget { background-color: rgba(255, 255, 255, 100); font-family: SimHei; font-size: 14px; }"
            "QTableWidget::item { padding: 10px; }"
        )
        # layout.addWidget(self.table_widget)
        
        layout.addWidget(self.table_widget)
        
        self.setLayout(layout)
        # 添加鼠标跟踪事件，用于支持拖动
        self.setMouseTracking(True)
        self.old_pos = None

    def update_plot(self, detections):
        self.detections = detections
        self.table_widget.setRowCount(len(self.detections.items()))
        for i, (name, count) in enumerate(self.detections.items()):
            item_name = QTableWidgetItem(name)
            item_name.setTextAlignment(Qt.AlignCenter)
            item_count = QTableWidgetItem(str(count))
            item_count.setTextAlignment(Qt.AlignCenter)
            self.table_widget.setItem(i, 0, item_name)
            self.table_widget.setItem(i, 1, item_count)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.old_pos:
            delta = event.globalPos() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = None
            
    # def moveEvent(self, event):
    #     main_window_x, main_window_y = self.main_window_pos
    #     self.move(main_window_x + 500, main_window_y)
#%%
if __name__ == '__main__':
    from libs.win_qt_taskbar import load_brand_qicon, set_windows_app_user_model_id

    set_windows_app_user_model_id('Brigchen.AutoLabelTool.FishVision.1')
    app = QApplication(sys.argv)
    _root = repo_root()
    _icon = load_brand_qicon(_root, 'fishvision_v2_s')
    if _icon.isNull():
        _icon = QIcon(os.path.join(_root, 'resources', 'icons', 'fishvision_v2_s.png'))
    app.setWindowIcon(_icon)
    main_window = MainWindow()
    main_window.setWindowIcon(_icon)
    print("748 -->")
    main_window.show()
    print('750 -->')
    sys.exit(app.exec_())

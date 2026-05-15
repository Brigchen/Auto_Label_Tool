#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs, os, sys, platform, subprocess, random

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from libs.win_dll_search_path import register_repo_libs_dll_path

register_repo_libs_dll_path()
import natsort, cv2, easygui, json
import xml.etree.ElementTree as ET
from copy import deepcopy
from strsimpy.jaro_winkler import JaroWinkler
from functools import partial
import torch
import traceback
import glob
import shutil
import urllib.request
from urllib.error import URLError
from xpinyin import Pinyin
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_removed_sys_paths = []
for _p in list(sys.path):
    try:
        _abs_p = os.path.abspath(_p) if _p else _SCRIPT_DIR
    except Exception:
        _abs_p = _p
    if _abs_p == _SCRIPT_DIR:
        _removed_sys_paths.append(_p)
        sys.path.remove(_p)

from ultralytics import YOLO
from ultralytics.utils.torch_utils import time_sync
try:
    from ultralytics.data.loaders import LoadImages
except Exception:
    class LoadImages:
        """Compatibility fallback for ultralytics>=8.4 where old LoadImages API changed."""
        IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

        def __init__(self, path):
            self.files = []
            if os.path.isdir(path):
                for name in sorted(os.listdir(path)):
                    p = os.path.join(path, name)
                    if os.path.isfile(p) and name.lower().endswith(self.IMG_EXTS):
                        self.files.append(p)
            elif os.path.isfile(path):
                self.files = [path]

        def __len__(self):
            return len(self.files)

        def __iter__(self):
            for p in self.files:
                img = cv2.imread(p)
                if img is None:
                    continue
                # Keep shape compatible with historical usage:
                # for path, img, _, _ in dataset: path[0] ...
                yield [p], img, None, None

for _p in reversed(_removed_sys_paths):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from libs.iou import compute_IOU
from libs.combobox import ComboBox
from libs.constants import *
from libs.utils import *
from libs.resources import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.pascal_voc_io import PascalVocWriter
from libs.yolo_io import YOLOWriter
from libs.ifc_fish_yolo import discover_fish_yolo_yaml, parse_fish_yolo_yaml, default_skeleton_chain
from libs.keypoint_utils import pad_keypoints_shape, spread_keypoint_placeholders
# from libs.pascal_voc_io import XML_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem
# FILE = Path('ALT.py').resolve() #__file__
# ROOT = FILE.parents[0]
#
# sys.path.append(ROOT/r'ultralytics')

import warnings
warnings.filterwarnings('ignore')

__appname__ = 'Auto_Label_Tool'
__version__ = APP_VERSION
__update__ = APP_UPDATE_DATE
__copyright__ = "Brigchen, from Ocean-IDEA of XMU"
__contact__ = "brigchen@xmu.edu.cn"
from libs.repo_paths import repo_root
from libs.yolo_weights import DEFAULT_DETECT_PT, resolve_yolo_checkpoint

APP_ROOT = repo_root()
_version_info = {
    'app_name': __appname__,
    'version': __version__,
    'update_date': __update__,
    'changelog': list(APP_CHANGELOG) if isinstance(APP_CHANGELOG, list) else [],
}

'''
2024-5-29
auto labeling current image

2023-9-03
minor bug solved

2023-8-14
1. load label names for manual assignment
2. debug

2023-8-10
1. extract videos from one directory

2023-7-24
1. multi-class labeling function
2. train function update
3. trained model update

2023-7-4
 1. fix the bug of label setting

2023-7-1
 1. remove agument function

2023-6-30
 1. update model engine to yolov8
 2. modify VOC to YOLO error of mismatchs by image size error


2023-6-25
 1. add training single class function
 2. extract video: change extracted image name as "video filename" + frame number, to avoid repeatness of images from different videos.
'''
#%%

def check_chinese(dr):
    pinyin = Pinyin()
    for fl in os.listdir(dr):
        if not pinyin.get_pinyin(fl, '').isalpha():
            try:
                new_fl = pinyin.get_pinyin(fl)
                os.rename(os.path.join(dr,fl),os.path.join(dr,new_fl))
            except Exception as e:
                print(e)
            


#%% 检查gpu大小与确实batchsize
import pynvml
import psutil

def get_gpu_mem(gpuid):
    
    pynvml.nvmlInit()
    UNIT = 1024 * 1024
    # handle = len(pynvml.nvmlDeviceGetHandleByIndex(0))# 这里的0是GPU id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpuid))
    mem_total = (int(meminfo.total/UNIT)) 
    mem_free = (int(meminfo.free/UNIT))
    
    return [mem_total, mem_free]

def get_cpu_mem():
    UNIT = 1024 * 1024
    mem_total = psutil.virtual_memory().total/UNIT
    mem_free = psutil.virtual_memory().free/UNIT
    return [mem_total, mem_free]


def get_cuda_setup(bsize1, gpu_mem1):
    
    if torch.cuda.is_available():
        mem = get_gpu_mem(0)
        gpuid = 0
        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                # gpuid = i
                # print(get_gpu_mem(i))
                if get_gpu_mem(i)[1] > mem[1]:
                    gpuid = i
                    mem = get_gpu_mem(i)
        print("GPU id: %d free mem = %d, used %s"%(gpuid, mem[1], str(int((mem[0]-mem[1])/mem[0]*100))+'%'))
    else:
        mem = get_cpu_mem()
        # print("CPU mem = %d"%mem)
    bsize = int(bsize1/gpu_mem1 * (mem[1]/1024))
    # print(bsize)
    return gpuid, mem[1], bsize

# get_batchsize(28,24)
#%%
def print_error():
    a,b,c = sys.exc_info()
    error = '%s\n%s\n%s'%(a,b,'_'.join([str(x) for x in traceback.extract_tb(c)]))
    return error

def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/best*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''

class train_one(QThread):
    # 数据信号
    # data_signal = pyqtSignal(dict)
    str_signal = pyqtSignal(str)
    # 提示语信号
    def __init__(self, **kwargs):
        super().__init__()    
    # 线程调用类型（用于单个线程处理不同逻辑）
        self.weights = kwargs.get("weights")
        self.data = kwargs.get("data")
        self.bsize = int(kwargs.get("bsize"))
        self.imgsz = int(kwargs.get("imgsz"))
        self.epochs = int(kwargs.get("epochs"))
        self.optimizer = kwargs.get('optimizer')
        self.lr0 = float(kwargs.get('lr0'))
    def run(self):
        try:
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    gpuid, free_mem, bsize = get_cuda_setup(24,24)
                    # print(torch.cuda.get_device_capability('cuda:2'))
                    device = "%d"%gpuid
                    # print(torch.cuda.device_count())
                else:
                    device = "0"
            else:
                device = "cpu"  
            print('use device:%s for training'%device)
            w = resolve_yolo_checkpoint(self.weights, os.path.join(APP_ROOT, 'weights'))
            if w != self.weights:
                print('[train] resolved weights:', self.weights, '->', w)
            model = YOLO(w)
            model.train(data=self.data, batch=self.bsize, imgsz=self.imgsz, epochs=self.epochs, optimizer=self.optimizer, lr0=self.lr0, device=device)
            self.str_signal.emit('done')
        except Exception as e:
            self.str_signal.emit(print_error())

class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    @staticmethod
    def _autolabel_weights_usable(w):
        """True if weights string is a local file, dir, or Ultralytics model id like yolo26n.pt."""
        if not w:
            return False
        s = ustr(w).strip()
        if not s:
            return False
        if os.path.isfile(s) or os.path.isdir(s):
            return True
        base = os.path.basename(s.replace('\\', '/'))
        if base == s.replace('\\', '/') and s.lower().endswith(('.pt', '.pth', '.h5')):
            return True
        return False

    def _persist_autolabel_settings(self):
        """Write Auto Label preferences to disk (subset of closeEvent)."""
        try:
            self.settings[SETTING_AUTOLABEL_WEIGHTS] = ustr(getattr(self, 'autolabel_weights', '') or '')
            self.settings[SETTING_AUTOLABEL_CONF] = float(getattr(self, 'autoLabelConf', 0.25))
            self.settings[SETTING_AUTOLABEL_PRED_IOU] = float(getattr(self, 'autoLabelPredIou', 0.5))
            self.settings[SETTING_AUTOLABEL_DUP_IOU] = float(getattr(self, 'autoLabelDedupIou', 0.7))
            self.settings[SETTING_AUTOLABEL_SETUP_DONE] = bool(getattr(self, 'autoLabelSetupDone', False))
            self.settings[SETTING_AUTOLABEL_WEIGHTS_CONFIRMED] = bool(
                getattr(self, 'autolabelWeightsUserConfirmed', False))
            self.settings.save()
        except Exception as e:
            print('[ALT][WARN] persist autolabel settings:', e)

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)
        
        # set default label for auto labeling
        self.default_label = None

        # Save as Pascal voc xml
        self.defaultSaveDir = defaultSaveDir
        self.usingPascalVocFormat = True
        self.usingYoloFormat = False
        self.txtPath = ''
        self.xmlPath = ''
        self.lastLabelFile = ''

        #define global path for myown use
        self.img_folder_path=''
        self.xml_folder_path=''
        # For loading all image under a directory
        self.mImgList = []
        self.allImgList = []
        self.fileAnnoMeta = {}
        self.dirname = None
        self.labelHist = []
        self.predefined_classes=[]
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        # self.screencastViewer = self.getAvailableScreencastViewer()
        # self.screencast = "https://youtu.be/p0nR2YsCY_U"
        self.yolo_classes = []
        self.keypointSchemas = {}
        self.currentKeypointTemplate = ''
        self.ifc_fish_meta = None
        
        # auto labeling (weights for training use self.weights; autolabel uses self.autolabel_weights)
        self.weights = ''
        self.autolabel_weights = ''
        self.autoLabelConf = 0.25
        self.autoLabelPredIou = 0.5
        self.autoLabelSetupDone = False
        self.autolabelWeightsUserConfirmed = False

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.predefined_classes)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        
        
        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox('Default Label')
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelCombobox = QComboBox()
        self.defaultLabelCombobox.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.defaultLabelCombobox.currentIndexChanged.connect(self.update_default_class)
        self.defaultLabelButton = QPushButton('+')
        font_bt = QFont()
        font_bt.setFamily('微软雅黑')
        font_bt.setBold(True)
        font_bt.setPointSize(20)
        self.defaultLabelButton.setFont(font_bt)
        self.defaultLabelButton.setFixedSize(20,25)
        self.defaultLabelButton.clicked.connect(self.addLabelname)
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelCombobox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelButton)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        # self.diffcButton = QCheckBox(getStr('useDifficult'))
        # self.diffcButton.setChecked(False)
        # self.diffcButton.stateChanged.connect(self.btnstate)
        # self.editButton = QToolButton()
        # self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        #listLayout.addWidget(self.editButton)
        #listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Create and add combobox for showing unique labels in group 
        self.comboBox = ComboBox(self)
        listLayout.addWidget(self.comboBox)
        
        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)
        
        
        #test: show local enlarge img
        self.new_test = QLabel(self)
        self.new_test.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        self.new_test.setGeometry(0,0,360,360)
        listLayout.addWidget(self.new_test)
        # self.new_test.setText('who am I?')
        
        
        self.dock = QDockWidget('Label information', self)   #getStr('boxLabelText')
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        self.fileSpeciesFilterCombo = QComboBox()
        self.fileSpeciesFilterCombo.addItem('All species')
        self.fileSpeciesFilterCombo.currentIndexChanged.connect(self.applyFileListFilters)
        self.fileScoreFilterCheck = QCheckBox('Score <')
        self.fileScoreFilterSpin = QDoubleSpinBox()
        self.fileScoreFilterSpin.setRange(0.0, 1.0)
        self.fileScoreFilterSpin.setDecimals(3)
        self.fileScoreFilterSpin.setSingleStep(0.05)
        self.fileScoreFilterSpin.setValue(0.5)
        self.fileScoreFilterCheck.stateChanged.connect(self.applyFileListFilters)
        self.fileScoreFilterSpin.valueChanged.connect(self.applyFileListFilters)
        self.fileCountFilterCheck = QCheckBox('Box count >')
        self.fileCountFilterSpin = QSpinBox()
        self.fileCountFilterSpin.setRange(0, 100000)
        self.fileCountFilterSpin.setValue(20)
        self.fileCountFilterCheck.stateChanged.connect(self.applyFileListFilters)
        self.fileCountFilterSpin.valueChanged.connect(self.applyFileListFilters)

        speciesRow = QHBoxLayout()
        speciesRow.setContentsMargins(0, 0, 0, 0)
        speciesRow.addWidget(QLabel('Species:'))
        speciesRow.addWidget(self.fileSpeciesFilterCombo)
        scoreRow = QHBoxLayout()
        scoreRow.setContentsMargins(0, 0, 0, 0)
        scoreRow.addWidget(self.fileScoreFilterCheck)
        scoreRow.addWidget(self.fileScoreFilterSpin)
        countRow = QHBoxLayout()
        countRow.setContentsMargins(0, 0, 0, 0)
        countRow.addWidget(self.fileCountFilterCheck)
        countRow.addWidget(self.fileCountFilterSpin)
        filelistLayout.addLayout(speciesRow)
        filelistLayout.addLayout(scoreRow)
        filelistLayout.addLayout(countRow)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)




        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))
        schema_file = ustr(settings.get(SETTING_KEYPOINT_SCHEMA_FILE,
                           os.path.join(APP_ROOT, 'datasets', 'keypoints_schema.json')))
        self.loadKeypointSchemas(schema_file)
        default_template = ustr(settings.get(SETTING_KEYPOINT_TEMPLATE, 'coco17'))
        self.apply_keypoint_template(default_template)

        scroll = QScrollArea()
        scroll.setMouseTracking(True)
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.keypointRenameRequested.connect(self.rename_keypoint_at_vertex)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dockFeatures) #self.dock.features() ^ 

        # Actions
        action = partial(newAction, self)
        # action()函数的输入参数，第一个str是显示按钮的名称，第二个是对应的触发函数，第三个是快捷键，第四个是icon
        
        #增加的action：
        load_label_names=action("load label names", self.load_label_names,'Ctrl+0', 'file')
        search_system=action('Search_System', self.search_actions_info,None,'zoom-in')
        batch_rename_img=action('batch_rename_img', self.batch_rename_img, 'Ctrl+1','edit')
        rename_img_xml=action('rename_img_xml', self.rename_img_xml, 'Ctrl+Alt+1','edit')
        duplicate_xml=action('duplicate_xml', self.make_duplicate_xml,'Ctrl+2', 'copy')
        batch_duplicate=action('batch_duplicate', self.batch_duplicate_xml,'Ctrl+Alt+2', 'copy')
        label_pruning=action('label_pruning', self.prune_useless_label,'Ctrl+3', 'delete')
        file_pruning=action('file_pruning', self.remove_extra_img_xml,'Ctrl+Alt+3', 'delete')
        change_label=action('change_label', self.change_label_name, 'Ctrl+4','color_line')
        fix_property=action('fix_property', self.fix_xml_property, 'Ctrl+5','color_line')
        auto_labeling_one=action('auto_labeling_single_class', self.auto_labeling_one,'Ctrl+6', 'app')
        auto_labeling_multi=action('auto_labeling_multi_classes', self.auto_labeling_multi,'Ctrl+7', 'app')
        choose_autolabel_model = action(
            u'选择自动标注模型',
            self.choose_autolabel_model,
            'Ctrl+Shift+M',
            'open',
            u'为「Auto Label 当前图」(快捷键 Q) 选择或更换 YOLO 权重；之后按 Q 将直接使用此处所选模型。',
            enabled=True)
        # data_agument=action('data_agument', self.data_auto_agument,'Ctrl+8', 'copy')
        
        make_training_datasets_one = action('make_training_datasets_single-class', self.make_datasets_one,'Ctrl+Alt+8', 'app')
        make_training_datasets = action('make_training_datasets', self.make_datasets,'Ctrl+8', 'app')
        training_model = action('training_model', self.training_model,'Ctrl+9', 'app')
        
        folder_info=action('folder_info', self.show_folder_infor,'Alt+1', 'help')
        label_info=action('label_info', self.show_label_info,'Alt+2', 'help')
        
        extract_video=action('extract_video', self.extract_video,'Shift+1', 'new')
        extract_videos=action('extract_videos', self.extract_videos,'Shift+2', 'new')
        extract_stream=action('extract_stream', self.extract_stream,'Shift+3', 'new')
        batch_resize_img=action('batch_resize_img', self.batch_resize_img,'Shift+4', 'fit-window')
        merge_video=action('merge_video', self.merge_video,'Shift+5', 'open')
        annotation_video=action('annotation_video', self.annotation_video,'Shift+6', 'new')
        
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        open = action(getStr('openFile'), self.openFile,
                      'Ctrl+O', 'open', getStr('openFileDetail'))

        opendir = action('Open Img Dir', self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        changeSavedir = action('Open Label Dir', self.changeSavedirDialog,
                               'Ctrl+r', 'open', getStr('changeSavedAnnotationDir'))

        openAnnotation = action(getStr('openAnnotation'), self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', getStr('openAnnotationDetail'))

        openNextImg = action('Next Img', self.openNextImg,
                             'd', 'next', getStr('nextImgDetail'))

        openPrevImg = action('Prev Img', self.openPrevImg,
                             'a', 'prev', getStr('prevImgDetail'))

        verify = action('Verify', self.verifyImg,
                        'space', 'verify', getStr('verifyImgDetail'))

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+S', 'save', getStr('saveDetail'), enabled=False)

        save_format = action('VOC', self.change_format,
                      'Ctrl+', 'format_voc', getStr('changeSaveFormat'), enabled=True)

        saveAs = action(getStr('saveAs'), self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', getStr('saveAsDetail'), enabled=False)

        close = action(getStr('closeCur'), self.closeFile, 'Ctrl+W', 'close', getStr('closeCurDetail'))

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        autolabel = action('Auto Label', self.autoLabelCurrentImg,
                        'q', 'new', u'Auto Label Current Image', enabled=False)
        
        copylabel = action('Copy Labels', self.copyLastLabel,
                        'c', 'new', u'Copy Last Labels', enabled=True)
        
        create = action('Create Box', self.createShape,
                        'w', 'new', getStr('crtBoxDetail'), enabled=False)
        delete = action('Delete Box', self.deleteSelectedShape,
                        'Delete', 'delete', getStr('delBoxDetail'), enabled=False)
        copy = action('Copy Box', self.copySelectedShape,
                      'Ctrl+D', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        advancedMode = action(getStr('advancedMode'), self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', getStr('advancedModeDetail'),
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))
        checkUpdateNow = action('Check Updates', self.checkSoftwareUpdateNow, 'Ctrl+Shift+U', 'open', 'Check software update from manifest')
        setUpdateUrl = action('Set Update URL', self.setUpdateManifestUrl, None, 'edit', 'Configure update manifest URL')

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel,
                      'E', 'edit', getStr('editLabelDetail'),
                      enabled=False)
        # self.editButton.setDefaultAction(edit)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        draw_mode = action('BBox', self.change_draw_mode,
                      'Ctrl+/', 'new', 'Change draw mode (BBox/Polygon/Keypoints)', enabled=True)
        keypoint_template = action('Keypoint Template', self.select_keypoint_template,
                      'Ctrl+Shift+K', 'edit', 'Select keypoint template', enabled=True)
        load_keypoint_schema = action('Load Keypoint Schema', self.load_keypoint_schema_file,
                      None, 'open', 'Load keypoint schema json', enabled=True)

        # Explicit YOLO save mode: bbox (off) or segmentation polygon (on)
        self.yoloSegmentationSaveOption = QAction('YOLO Segmentation Save Mode', self)
        self.yoloSegmentationSaveOption.setCheckable(True)
        self.yoloSegmentationSaveOption.setChecked(settings.get(SETTING_YOLO_SAVE_SEG, False))
        self.yoloSegmentationSaveOption.triggered.connect(self.toggleYoloSegmentationSaveMode)

        # Store actions for further handling.
        self.actions = struct(save=save, save_format=save_format, saveAs=saveAs, open=open, close=close, resetAll = resetAll,
                              lineColor=color1, autolabel = autolabel, copylabel = copylabel, create=create, delete=delete, edit=edit, copy=copy,
                              draw_mode=draw_mode,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              hideAll=hideAll, showAll=showAll,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, close, resetAll,quit,duplicate_xml,batch_duplicate,label_pruning,folder_info), #,data_agument
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.drawSquaresOption),
                              beginnerContext=(autolabel, copylabel, create, edit, copy, delete),
                              advancedContext=(autolabel, copylabel, createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, autolabel, copylabel, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            annotate=self.menu('&Annotate-Tools'),
            video=self.menu('&Video-Tools'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.useMagnifyingLens = QAction('Magnifying Lens', self)
        self.useMagnifyingLens.setCheckable(True)
        self.useMagnifyingLens.setChecked(settings.get(SETTING_Magnifying_Lens, False))
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        # 设置菜单栏
        addActions(self.menus.annotate,(load_label_names, None, batch_rename_img,rename_img_xml,None,
                    duplicate_xml,batch_duplicate,None,
                    label_pruning,file_pruning,change_label,fix_property,None,
                    choose_autolabel_model, None,
                    auto_labeling_one, auto_labeling_multi,None, make_training_datasets_one, make_training_datasets, training_model, None, #data_agument,None,
                    folder_info,label_info))
        addActions(self.menus.video,(extract_video,extract_videos,extract_stream,None,batch_resize_img,merge_video,None,annotation_video))
        addActions(self.menus.file,
                   (open, opendir, changeSavedir, openAnnotation, self.menus.recentFiles, save, save_format, saveAs, close, resetAll, quit))
        addActions(self.menus.help, (help, showInfo, checkUpdateNow, setUpdateUrl, None, search_system))
        addActions(self.menus.view, (
            self.useMagnifyingLens,
            self.autoSaving,
            self.singleClassMode,
            self.displayLabelOption,
            keypoint_template,
            load_keypoint_schema,
            self.yoloSegmentationSaveOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, save_format, draw_mode, None, autolabel, copylabel, create, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, save, save_format, draw_mode, None,
            autolabel, copylabel, createMode, editMode, None,
            hideAll, showAll)

        self.set_draw_mode(ustr(settings.get(SETTING_DRAW_MODE, 'bbox')))

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False
        self._startupImagePath = None

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        self.lastWeightDir = ustr(settings.get(SETTING_LAST_WEIGHT_DIR, None))
        try:
            self.autoLabelDedupIou = float(settings.get(SETTING_AUTOLABEL_DUP_IOU, 0.7))
        except Exception:
            self.autoLabelDedupIou = 0.7
        def _as_bool(v):
            if isinstance(v, QVariant):
                return v.toBool()
            if isinstance(v, str):
                return v.lower() in ('1', 'true', 'yes', 'on')
            return bool(v)
        try:
            self.autoLabelConf = float(settings.get(SETTING_AUTOLABEL_CONF, 0.25))
        except Exception:
            self.autoLabelConf = 0.25
        try:
            self.autoLabelPredIou = float(settings.get(SETTING_AUTOLABEL_PRED_IOU, 0.5))
        except Exception:
            self.autoLabelPredIou = 0.5
        self.autoLabelSetupDone = _as_bool(settings.get(SETTING_AUTOLABEL_SETUP_DONE, False))
        sw = ustr(settings.get(SETTING_AUTOLABEL_WEIGHTS, '')).strip()
        if sw:
            self.autolabel_weights = sw
        if self.autoLabelSetupDone and not MainWindow._autolabel_weights_usable(self.autolabel_weights):
            self.autoLabelSetupDone = False
        # Old ALT_Settings.pkl had no key => False => user must pick weights in UI at least once.
        self.autolabelWeightsUserConfirmed = _as_bool(
            settings.get(SETTING_AUTOLABEL_WEIGHTS_CONFIRMED, False))
        if not MainWindow._autolabel_weights_usable(self.autolabel_weights):
            self.autolabelWeightsUserConfirmed = False
        self.autoUpdateEnabled = _as_bool(settings.get(SETTING_AUTO_UPDATE_ENABLED, True))
        self.updateManifestUrl = ustr(settings.get(SETTING_UPDATE_MANIFEST_URL, ''))
        self.lastUpdateCheckAt = ustr(settings.get(SETTING_LAST_UPDATE_CHECK, ''))
        saved_species = ustr(settings.get(SETTING_FILE_FILTER_SPECIES, 'All species'))
        saved_score_enabled = _as_bool(settings.get(SETTING_FILE_FILTER_SCORE_ENABLED, False))
        saved_score_thr = float(settings.get(SETTING_FILE_FILTER_SCORE_THRESHOLD, 0.5))
        saved_count_enabled = _as_bool(settings.get(SETTING_FILE_FILTER_COUNT_ENABLED, False))
        saved_count_thr = int(settings.get(SETTING_FILE_FILTER_COUNT_THRESHOLD, 20))
        self.fileScoreFilterCheck.setChecked(saved_score_enabled)
        self.fileScoreFilterSpin.setValue(saved_score_thr)
        self.fileCountFilterCheck.setChecked(saved_count_enabled)
        self.fileCountFilterSpin.setValue(saved_count_thr)
        idx_saved_species = self.fileSpeciesFilterCombo.findText(saved_species)
        self.fileSpeciesFilterCombo.setCurrentIndex(idx_saved_species if idx_saved_species >= 0 else 0)
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()
        if self.defaultSaveDir is not None and os.path.exists(self.defaultSaveDir):
            self.handleSavedirLabelNames(self.defaultSaveDir)

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        lastStartupImage = ustr(settings.get(SETTING_LAST_OPEN_IMAGE, None))
        if lastStartupImage and os.path.isfile(lastStartupImage):
            self._startupImagePath = lastStartupImage
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Restore startup context: prefer last opened image directory, then explicit file.
        startupImageDir = self.lastOpenDir if (self.lastOpenDir and os.path.isdir(self.lastOpenDir)) else None
        if startupImageDir is None and self.filePath and os.path.isdir(self.filePath):
            startupImageDir = self.filePath

        # Since loading may take time, queue it to run after UI setup.
        if startupImageDir:
            self.queueEvent(partial(self.importDirImages, startupImageDir))
        elif self.filePath and os.path.isfile(self.filePath):
            self.queueEvent(partial(self.loadFile, self.filePath))
        else:
            lastFile = ustr(settings.get(SETTING_FILENAME, None))
            if lastFile and os.path.isfile(lastFile):
                self.queueEvent(partial(self.loadFile, lastFile))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Image dir/file restoration is handled by startup queue logic above.
        if self.autoUpdateEnabled and self.updateManifestUrl:
            self.queueEvent(lambda: self.checkSoftwareUpdate(silent=True))

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            if self.canvas.drawingShapeMode == 'bbox':
                self.canvas.setDrawingShapeToSquare(True)

    ## Support Functions ##
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(newIcon("format_voc"))
            self.usingPascalVocFormat = True
            self.usingYoloFormat = False
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(newIcon("format_yolo"))
            self.usingPascalVocFormat = False
            self.usingYoloFormat = True
            LabelFile.suffix = TXT_EXT

    def change_format(self):
        if self.usingPascalVocFormat: self.set_format(FORMAT_YOLO)
        elif self.usingYoloFormat: self.set_format(FORMAT_PASCALVOC)

    def set_draw_mode(self, draw_mode='bbox'):
        if draw_mode not in ('bbox', 'polygon', 'keypoints'):
            draw_mode = 'bbox'
        self.canvas.setDrawingShapeMode(draw_mode)
        if draw_mode == 'bbox':
            self.actions.draw_mode.setText('BBox')
        elif draw_mode == 'polygon':
            self.actions.draw_mode.setText('Polygon')
        else:
            self.actions.draw_mode.setText('Keypoints')
        self._sync_shape_actions_to_draw_mode(draw_mode)

    def _sync_shape_actions_to_draw_mode(self, draw_mode):
        """Toolbar/context labels: Box vs Polygon vs Keypoints."""
        gs = self.stringBundle.getString

        def tip(a, text):
            if text is not None:
                a.setToolTip(text)
                a.setStatusTip(text)

        if draw_mode == 'keypoints':
            self.actions.create.setText(gs('crtKptShort'))
            tip(self.actions.create, gs('crtKptDetail'))
            self.actions.delete.setText(gs('delKptShort'))
            tip(self.actions.delete, gs('delKptDetail'))
            self.actions.copy.setText(gs('dupKptShort'))
            tip(self.actions.copy, gs('dupKptDetail'))
            self.actions.createMode.setText(gs('crtKpt'))
            tip(self.actions.createMode, gs('crtKptDetail'))
            self.actions.editMode.setText(gs('editKpt'))
            tip(self.actions.editMode, gs('editKptDetail'))
            self.actions.hideAll.setText(gs('hideKpt'))
            tip(self.actions.hideAll, gs('hideKptDetail'))
            self.actions.showAll.setText(gs('showKpt'))
            tip(self.actions.showAll, gs('showKptDetail'))
        elif draw_mode == 'polygon':
            self.actions.create.setText(gs('crtPolyShort'))
            tip(self.actions.create, gs('crtPolyDetail'))
            self.actions.delete.setText(gs('delPolyShort'))
            tip(self.actions.delete, gs('delPolyDetail'))
            self.actions.copy.setText(gs('dupPolyShort'))
            tip(self.actions.copy, gs('dupPolyDetail'))
            self.actions.createMode.setText(gs('crtPoly'))
            tip(self.actions.createMode, gs('crtPolyDetail'))
            self.actions.editMode.setText(gs('editPoly'))
            tip(self.actions.editMode, gs('editPolyDetail'))
            self.actions.hideAll.setText(gs('hidePoly'))
            tip(self.actions.hideAll, gs('hidePolyDetail'))
            self.actions.showAll.setText(gs('showPoly'))
            tip(self.actions.showAll, gs('showPolyDetail'))
        else:
            self.actions.create.setText(gs('crtBoxShort'))
            tip(self.actions.create, gs('crtBoxDetail'))
            self.actions.delete.setText(gs('delBoxShort'))
            tip(self.actions.delete, gs('delBoxDetail'))
            self.actions.copy.setText(gs('dupBoxShort'))
            tip(self.actions.copy, gs('dupBoxDetail'))
            self.actions.createMode.setText(gs('crtBox'))
            tip(self.actions.createMode, gs('crtBoxDetail'))
            self.actions.editMode.setText(gs('editBox'))
            tip(self.actions.editMode, gs('editBoxDetail'))
            self.actions.hideAll.setText(gs('hideBox'))
            tip(self.actions.hideAll, gs('hideAllBoxDetail'))
            self.actions.showAll.setText(gs('showBox'))
            tip(self.actions.showAll, gs('showAllBoxDetail'))

    def change_draw_mode(self):
        if self.canvas.drawingShapeMode == 'bbox':
            self.set_draw_mode('polygon')
        elif self.canvas.drawingShapeMode == 'polygon':
            self.set_draw_mode('keypoints')
        else:
            self.set_draw_mode('bbox')

    def loadKeypointSchemas(self, schema_path):
        self.keypointSchemas = {
            'coco17': {
                'keypoints': [
                    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                ],
                'skeleton': [
                    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
                    [1, 3], [2, 4], [3, 5], [4, 6]
                ]
            }
        }
        try:
            if schema_path and os.path.isfile(schema_path):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                templates = data.get('templates', {})
                for name, cfg in templates.items():
                    names = cfg.get('keypoints', [])
                    skeleton = cfg.get('skeleton', [])
                    if isinstance(names, list) and isinstance(skeleton, list):
                        self.keypointSchemas[name] = {'keypoints': names, 'skeleton': skeleton}
                self.settings[SETTING_KEYPOINT_SCHEMA_FILE] = schema_path
        except Exception:
            pass

    def apply_keypoint_template(self, template_name):
        if not self.keypointSchemas:
            self.canvas.setKeypointTemplate([], [])
            self.currentKeypointTemplate = ''
            return False
        if template_name not in self.keypointSchemas:
            template_name = list(self.keypointSchemas.keys())[0]
        cfg = self.keypointSchemas.get(template_name, {})
        self.canvas.setKeypointTemplate(cfg.get('keypoints', []), cfg.get('skeleton', []))
        self.currentKeypointTemplate = template_name
        self.settings[SETTING_KEYPOINT_TEMPLATE] = template_name
        return True

    def load_keypoint_schema_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Choose keypoint schema file', r'datasets', "JSON file(*.json)"
        )
        if not filename:
            return
        self.loadKeypointSchemas(filename)
        if self.keypointSchemas:
            self.apply_keypoint_template(list(self.keypointSchemas.keys())[0])
            self.statusBar().showMessage('Loaded keypoint schema: {}'.format(os.path.basename(filename)), 3000)

    def select_keypoint_template(self):
        if not self.keypointSchemas:
            QMessageBox.information(self, u'Info', u'No keypoint schema loaded.')
            return
        names = sorted(list(self.keypointSchemas.keys()))
        current = self.currentKeypointTemplate if self.currentKeypointTemplate in names else names[0]
        selected, ok = QInputDialog.getItem(self, 'Keypoint Template', 'Select template:', names, names.index(current), False)
        if ok and selected:
            self.apply_keypoint_template(selected)
            self.statusBar().showMessage('Keypoint template: {}'.format(selected), 3000)

    def rename_keypoint_at_vertex(self, shape, index):
        """Right-click keypoint: pick display name from current template kpt_names (per-point override)."""
        if shape is None or getattr(shape, 'shape_type', '') != 'keypoints':
            return
        names = list(self.canvas.keypointNames)
        if not names:
            self.statusBar().showMessage(u'无关键点模板名称，无法重命名（请加载 fish_yolo.yaml 或 keypoints schema）', 5000)
            return
        if index < 0 or index >= len(shape.points):
            return
        while len(shape.keypoint_names) < len(shape.points):
            i = len(shape.keypoint_names)
            shape.keypoint_names.append(names[i] if i < len(names) else str(i))
        cur = shape.keypoint_names[index] if index < len(shape.keypoint_names) else ''
        try:
            d0 = names.index(cur) if cur in names else min(index, len(names) - 1)
        except ValueError:
            d0 = min(max(0, index), len(names) - 1)
        selected, ok = QInputDialog.getItem(
            self, u'关键点名称', u'选择该点的 kpt_name：', names, d0, False)
        if ok and selected is not None:
            shape.keypoint_names[index] = ustr(selected)
            self.setDirty()
            self.canvas.update()

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        # self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        self.comboBox.cb.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        # subprocess.Popen(self.screencastViewer + [self.screencast])
        msg = u'Tutorial:{0} \nPlease email for assistance:{1}'.format('not available now', __contact__)
        QMessageBox.information(self, u'Information', msg)

    def showInfoDialog(self):
        changelog = _version_info.get('changelog', [])
        change_text = ''
        if isinstance(changelog, list) and changelog:
            change_text = '\nRecent changes:\n- ' + '\n- '.join([ustr(x) for x in changelog[:6]])
        msg = u'Name:{0} \nApp Version:{1} \nUpdated:{2}\nCopyright:{3}\nContact:{4}{5}'.format(
            __appname__, __version__, __update__, __copyright__, __contact__, change_text)
        QMessageBox.information(self, u'Information', msg)

    def _report_error(self, scene='unknown', with_dialog=True):
        err = print_error()
        print('[ALT][ERROR][%s] %s' % (scene, err))
        if with_dialog:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(err))

    @staticmethod
    def _parse_version_tuple(version_text):
        parts = []
        for x in str(version_text).replace('-', '.').split('.'):
            try:
                parts.append(int(x))
            except Exception:
                parts.append(0)
        return tuple(parts)

    def _append_jsonl_record(self, target_path, record):
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            print('append record failed:', target_path, e)

    def _record_upgrade_event(self, event_type, record):
        logs_dir = os.path.join(APP_ROOT, 'upgrade_logs')
        record = dict(record)
        record['event_type'] = event_type
        self._append_jsonl_record(os.path.join(logs_dir, 'upgrade_history.jsonl'), record)

    def _fetch_update_manifest(self, manifest_url):
        with urllib.request.urlopen(manifest_url, timeout=8) as resp:
            payload = resp.read().decode('utf-8')
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise RuntimeError('Invalid update manifest format.')
        return data

    def setUpdateManifestUrl(self):
        txt, ok = QInputDialog.getText(
            self, 'Update Manifest URL',
            'Input update manifest URL (json):',
            text=self.updateManifestUrl if self.updateManifestUrl else '')
        if not ok:
            return
        self.updateManifestUrl = ustr(txt).strip()
        QMessageBox.information(self, u'Info', u'更新地址已保存（关闭时写入设置）。')

    def checkSoftwareUpdateNow(self):
        self.checkSoftwareUpdate(silent=False)

    def checkSoftwareUpdate(self, silent=True):
        if not self.updateManifestUrl:
            if not silent:
                QMessageBox.information(self, u'Info', u'请先在 Help -> Set Update URL 配置更新清单地址。')
            return
        now_text = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
        self.lastUpdateCheckAt = now_text
        try:
            manifest = self._fetch_update_manifest(self.updateManifestUrl)
            remote_version = str(manifest.get('version', '')).strip()
            remote_date = str(manifest.get('update_date', '')).strip()
            notes = manifest.get('notes', [])
            download_url = str(manifest.get('download_url', '')).strip()
            if not remote_version:
                raise RuntimeError('manifest missing version')
            local_v = self._parse_version_tuple(__version__)
            remote_v = self._parse_version_tuple(remote_version)
            is_newer = remote_v > local_v
            self._record_upgrade_event('software_update_check', {
                'time': now_text,
                'local_version': __version__,
                'remote_version': remote_version,
                'manifest_url': self.updateManifestUrl,
                'has_update': bool(is_newer),
            })
            if not is_newer:
                if not silent:
                    QMessageBox.information(self, u'Info', u'已是最新版本。')
                return

            notes_text = ''
            if isinstance(notes, list) and notes:
                notes_text = '\n- ' + '\n- '.join([ustr(x) for x in notes[:8]])
            msg = u'发现新版本：{0}\n当前版本：{1}\n更新时间：{2}{3}'.format(
                remote_version, __version__, remote_date if remote_date else 'unknown', notes_text)
            if download_url:
                msg += u'\n\n是否下载更新包到本地 updates/ 目录？'
                yes = QMessageBox.Yes
                no = QMessageBox.No
                if QMessageBox.question(self, u'Update Available', msg, yes | no) == yes:
                    self.downloadSoftwareUpdatePackage(download_url, remote_version)
            else:
                QMessageBox.information(self, u'Update Available', msg + u'\n\nmanifest 未提供 download_url。')
        except Exception as e:
            self._record_upgrade_event('software_update_check_error', {
                'time': now_text,
                'manifest_url': self.updateManifestUrl,
                'error': str(e),
            })
            if not silent:
                QMessageBox.information(self, u'Error', u'检查更新失败: %s' % str(e))

    def downloadSoftwareUpdatePackage(self, download_url, remote_version):
        updates_dir = os.path.join(APP_ROOT, 'updates')
        os.makedirs(updates_dir, exist_ok=True)
        filename = os.path.basename(download_url.split('?')[0]) or ('update_%s.zip' % remote_version)
        if not (filename.lower().endswith('.zip') or filename.lower().endswith('.py')):
            filename = 'update_%s.bin' % remote_version
        local_path = os.path.join(updates_dir, filename)
        now_text = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
        try:
            urllib.request.urlretrieve(download_url, local_path)
            self._record_upgrade_event('software_update_download', {
                'time': now_text,
                'version': remote_version,
                'download_url': download_url,
                'saved_to': local_path,
            })
            QMessageBox.information(
                self, u'Done!',
                u'更新包已下载：\n%s\n\n请备份工程后手动替换，重启软件生效。' % local_path)
        except Exception as e:
            self._record_upgrade_event('software_update_download_error', {
                'time': now_text,
                'version': remote_version,
                'download_url': download_url,
                'error': str(e),
            })
            QMessageBox.information(self, u'Error', u'下载更新包失败: %s' % str(e))

    def copyLastLabel(self):
        assert self.beginner()
        print('Copy Last Labels to current image')
        # def auto_labeling_m(self):
        try:
            if self.lastLabelFile:
                if self.defaultSaveDir is not None:
                    basename = os.path.basename(
                        os.path.splitext(self.filePath)[0])
                    ext = os.path.splitext(self.lastLabelFile)[-1]
                    LabelFile = os.path.join(self.defaultSaveDir, basename + ext)
                    shutil.copy(self.lastLabelFile, LabelFile)

            self.loadFile(self.filePath)
            # self.refreshImg()
            # QMessageBox.information(self, u'Done!', u'auto labeling done,and reload img folder')
        except Exception as e:
            self._report_error('autoLabelCurrentImg')

    @staticmethod
    def _ordered_model_class_names(names):
        """Ultralytics class names in id order (0..N-1), as strings."""
        if isinstance(names, dict):
            keys = sorted(names.keys(), key=lambda k: int(k))
            return [str(names[k]) for k in keys]
        return [str(x) for x in names]

    def _canonical_label_in_yolo_class_list(self, model_label, class_list):
        """Map model output name to the exact string in class_list (same index when saving YOLO)."""
        ml = str(model_label).strip()
        for c in class_list:
            if str(c).strip() == ml:
                return c
        return ml

    def _sync_lists_after_autolabel(self, class_list):
        """YOLOWriter.save mutates class_list in place; mirror into app state."""
        self.yolo_classes = class_list
        for c in self.yolo_classes:
            if c not in self.predefined_classes:
                self.predefined_classes.append(c)
        self.defaultLabelCombobox.clear()
        self.defaultLabelCombobox.addItems(self.predefined_classes)

    def _dedup_same_class_overlaps(self, boxes, scores, labels, iou_thr):
        """Drop lower-score boxes when same-label IoU is above threshold."""
        if len(boxes) <= 1:
            return boxes, scores, labels
        order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
        keep = []
        for idx in order:
            suppress = False
            for kept_idx in keep:
                if str(labels[idx]) != str(labels[kept_idx]):
                    continue
                if compute_IOU(boxes[idx], boxes[kept_idx]) > float(iou_thr):
                    suppress = True
                    break
            if not suppress:
                keep.append(idx)
        keep = sorted(keep)
        return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]

    def _dedup_pose_same_class(self, boxes, scores, labels, kpts_vis_list, iou_thr):
        """Same as box dedup but keeps paired keypoint rows."""
        if len(boxes) <= 1:
            return boxes, scores, labels, kpts_vis_list
        order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
        keep = []
        for idx in order:
            suppress = False
            for kept_idx in keep:
                if str(labels[idx]) != str(labels[kept_idx]):
                    continue
                if compute_IOU(boxes[idx], boxes[kept_idx]) > float(iou_thr):
                    suppress = True
                    break
            if not suppress:
                keep.append(idx)
        keep = sorted(keep)
        return (
            [boxes[i] for i in keep],
            [scores[i] for i in keep],
            [labels[i] for i in keep],
            [kpts_vis_list[i] for i in keep],
        )

    def _extract_pose_rows_from_result(self, result, names):
        """Build lists for pose auto-label from Ultralytics result (boxes + keypoints)."""
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return [], [], [], []
        kp = getattr(result, 'keypoints', None)
        if kp is None:
            return [], [], [], []

        cls_idx = boxes.cls.cpu().numpy().astype(int)
        confs = [float(x) for x in boxes.conf.cpu().numpy()]
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        labels = []
        for c in cls_idx:
            if isinstance(names, dict):
                labels.append(str(names[int(c)]))
            else:
                labels.append(str(names[int(c)]))

        n = len(xyxy)
        import numpy as np
        if hasattr(kp, 'xy') and kp.xy is not None and len(kp.xy):
            xy = kp.xy.cpu().numpy()
            kconf = kp.conf.cpu().numpy() if getattr(kp, 'conf', None) is not None else None
        elif hasattr(kp, 'xyn') and kp.xyn is not None and len(kp.xyn):
            xyn = kp.xyn.cpu().numpy()
            kconf = kp.conf.cpu().numpy() if getattr(kp, 'conf', None) is not None else None
            # convert normalized to pixel using last known image size from result
            h, w = result.orig_shape[0], result.orig_shape[1]
            xy = np.zeros_like(xyn)
            xy[:, :, 0] = xyn[:, :, 0] * w
            xy[:, :, 1] = xyn[:, :, 1] * h
        else:
            return [], [], [], []

        kpts_vis_list = []
        for i in range(n):
            pts = []
            vis = []
            for j in range(xy.shape[1]):
                pts.append((int(float(xy[i, j, 0])), int(float(xy[i, j, 1]))))
                if kconf is not None:
                    c = float(kconf[i, j])
                    v = 2 if c > 0.5 else (1 if c > 0.25 else 0)
                else:
                    v = 2
                vis.append(v)
            kpts_vis_list.append((pts, vis))
        return xyxy, confs, labels, kpts_vis_list

    def _apply_ifc_fish_yolo_from_dir(self, labels_dir):
        """Load fish_yolo.yaml next to labels (IFC fish pose dataset) and apply keypoint template."""
        yaml_path = discover_fish_yolo_yaml(labels_dir)
        if not yaml_path:
            self.ifc_fish_meta = None
            return
        try:
            meta = parse_fish_yolo_yaml(yaml_path)
            self.ifc_fish_meta = meta
            nk = meta.get('kpt_n') or len(meta.get('kpt_names') or [])
            kpt_names = meta.get('kpt_names') or [str(i) for i in range(max(nk, 1))]
            sk_meta = meta.get('skeleton')
            if sk_meta is None:
                sk = default_skeleton_chain(len(kpt_names))
            else:
                sk = list(sk_meta)
            self.keypointSchemas['ifc_fish'] = {'keypoints': kpt_names, 'skeleton': sk}
            self.apply_keypoint_template('ifc_fish')
            self.set_draw_mode('keypoints')
            if meta.get('class_names') and not self.predefined_classes:
                self.predefined_classes = list(meta['class_names'])
                self.yolo_classes = list(meta['class_names'])
                self.defaultLabelCombobox.clear()
                self.defaultLabelCombobox.addItems(self.predefined_classes)
            self.statusBar().showMessage('IFC fish_yolo.yaml: %s' % yaml_path, 5000)
        except Exception as e:
            print('[ALT][WARN][ifc_fish_yolo] parse failed:', e)
            self.ifc_fish_meta = None

    def _pick_weight_file(self, title_text):
        start_dir = self.lastWeightDir if (self.lastWeightDir and os.path.isdir(self.lastWeightDir)) else os.path.abspath('.')
        weights, _ = QFileDialog.getOpenFileName(
            self,
            title_text,
            start_dir,
            "Model weights (*.pt *.pth *.h5);;All files (*.*)")
        if not weights:
            return ''
        if not (weights.endswith('.h5') or weights.endswith('.pt') or weights.endswith('.pth')):
            QMessageBox.information(self, u'Wrong!', u'权重文件扩展名应为 .pt / .pth / .h5')
            return ''
        picked_dir = os.path.dirname(weights)
        if picked_dir and os.path.isdir(picked_dir):
            self.lastWeightDir = picked_dir
        return weights

    def _ensure_preferred_weight(self, prefer_model=None):
        """Ensure preferred model exists under project weights, auto-download if missing."""
        prefer_model = prefer_model or DEFAULT_DETECT_PT
        weights_dir = os.path.join(APP_ROOT, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        prefer_local = os.path.join(weights_dir, os.path.basename(prefer_model))
        if os.path.isfile(prefer_local):
            return prefer_local

        resolved = resolve_yolo_checkpoint(prefer_model, app_weights_dir=weights_dir)
        if os.path.isfile(resolved):
            return os.path.abspath(resolved)

        # Try Ultralytics builtin download/caching pathway.
        try:
            model = YOLO(prefer_model)
            ckpt_path = getattr(model, 'ckpt_path', '') or ''
            if ckpt_path and os.path.isfile(ckpt_path):
                shutil.copy2(ckpt_path, prefer_local)
                return prefer_local
            if os.path.isfile(prefer_local):
                return prefer_local
            # Fallback: if not copied but model id works, keep using model id.
            return prefer_model
        except Exception:
            return ''

    def _select_model_weight(self, title_text, prefer_model=None, for_autolabel=False,
                             force_browse_default=False):
        """Prefer modern builtin model but keep compatibility with legacy local weights.

        for_autolabel: when True, read/write ``self.autolabel_weights`` (Auto Label flows);
        when False, read/write ``self.weights`` (training), so training never steals autolabel weights.
        force_browse_default: when True with for_autolabel, highlight Browse so Enter does not pick builtin.
        """
        prefer_model = prefer_model or DEFAULT_DETECT_PT
        attr = 'autolabel_weights' if for_autolabel else 'weights'
        weight_path = os.path.join(APP_ROOT, 'weights')
        local_weights = []
        if os.path.isdir(weight_path):
            for item in sorted(os.listdir(weight_path)):
                if item.endswith('.h5') or item.endswith('.pt') or item.endswith('.pth'):
                    local_weights.append(item)

        builtin_label = '[Builtin] %s (recommended)' % prefer_model
        browse_label = '[Browse local file...]'
        items = [builtin_label] + local_weights + [browse_label]
        cur = ustr(getattr(self, attr, '') or '').strip()
        if cur:
            default_index = 0
            current_name = os.path.basename(cur)
            if current_name in local_weights:
                default_index = items.index(current_name)
        else:
            # First-time Auto Label: avoid defaulting to built-in on Enter — user must pick explicitly.
            default_index = items.index(browse_label) if for_autolabel else 0
        if for_autolabel and force_browse_default:
            default_index = items.index(browse_label)
        selected, ok = QInputDialog.getItem(self, "Select", title_text, tuple(items), default_index, False)
        if not ok:
            return ''
        if selected == builtin_label:
            resolved = self._ensure_preferred_weight(prefer_model)
            if resolved:
                setattr(self, attr, resolved)
                return resolved
            QMessageBox.information(
                self, u'Warning',
                u'自动下载 %s 失败，请手动选择本地权重文件。' % prefer_model)
            picked = self._pick_weight_file(
                u"选择可用的本地模型权重文件（.pt/.pth/.h5）：")
            if picked:
                setattr(self, attr, picked)
            return picked
        if selected == browse_label:
            picked = self._pick_weight_file(
                u"选择本地模型权重文件（支持旧版与新版，.pt/.pth/.h5）：")
            if picked:
                setattr(self, attr, picked)
            return picked
        picked = os.path.join(weight_path, selected)
        setattr(self, attr, picked)
        return picked

    def choose_autolabel_model(self):
        u"""Annotate-Tools「选择自动标注模型」：为快捷键 Q（Auto Label 当前图）指定或更换 YOLO 权重；快捷键 Ctrl+Shift+M。"""
        prev = ustr(getattr(self, 'autolabel_weights', '') or '').strip()
        title = u'Auto Label 模型权重（当前: %s）：' % (
            os.path.basename(prev) if prev else u'(未设置)')
        wsel = self._select_model_weight(
            title,
            prefer_model=DEFAULT_DETECT_PT,
            for_autolabel=True,
            force_browse_default=False)
        if not wsel:
            return
        self.autolabelWeightsUserConfirmed = True
        self._persist_autolabel_settings()
        disp = ustr(self.autolabel_weights)
        if len(disp) > 80:
            disp = '…' + disp[-78:]
        self.statusBar().showMessage(u'Auto Label 已切换模型: %s' % disp, 8000)

    def autoLabelCurrentImg(self):
        assert self.beginner()
        print('auto label current image')
        # def auto_labeling_m(self):
        try:
            self.xml_folder_path = self.defaultSaveDir
            source = self.filePath
            xml_path = self.xmlPath

            # Ask for weights when unusable, first-run wizard not done, or user never confirmed
            # in this app generation (old pkl lacked autolabel/weights_confirmed — was skipping dialog).
            need_weight_dialog = (
                not MainWindow._autolabel_weights_usable(self.autolabel_weights)
                or not self.autoLabelSetupDone
                or not getattr(self, 'autolabelWeightsUserConfirmed', False)
            )
            if need_weight_dialog:
                wsel = self._select_model_weight(
                    "Model weight for Auto Label (default %s):" % DEFAULT_DETECT_PT,
                    prefer_model=DEFAULT_DETECT_PT,
                    for_autolabel=True,
                    force_browse_default=not getattr(self, 'autolabelWeightsUserConfirmed', False))
                if not wsel:
                    return
                self.autolabelWeightsUserConfirmed = True
                self.settings[SETTING_AUTOLABEL_WEIGHTS_CONFIRMED] = True
                self.settings[SETTING_AUTOLABEL_WEIGHTS] = ustr(self.autolabel_weights)
                self.settings.save()

            if not self.autoLabelSetupDone:
                conf_thres, ok = QInputDialog.getDouble(
                    self,
                    u'Auto Label 置信度',
                    u'检测置信度阈值 conf（首次设置后将自动沿用；关闭前会写入设置文件）:',
                    value=float(self.autoLabelConf),
                    min=0.01,
                    max=1.0,
                    decimals=3)
                if not ok:
                    return
                dedup_iou, ok = QInputDialog.getDouble(
                    self,
                    u'Auto Label 去重',
                    u'同类框重叠去重 IoU 阈值（>阈值仅保留高分框；首次设置后将自动沿用）:',
                    value=float(self.autoLabelDedupIou),
                    min=0.1,
                    max=0.99,
                    decimals=2)
                if not ok:
                    return
                self.autoLabelConf = float(conf_thres)
                self.autoLabelDedupIou = float(dedup_iou)
                self.autoLabelSetupDone = True
                self._persist_autolabel_settings()
            else:
                conf_thres = float(self.autoLabelConf)
                dedup_iou = float(self.autoLabelDedupIou)

            iou_thres = float(self.autoLabelPredIou)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    gpuid, _, _ = get_cuda_setup(24,24)
                    # print(torch.cuda.get_device_capability('cuda:2'))
                    device = torch.device("cuda:%d"%gpuid)
                    print(torch.cuda.device_count())
                else:
                    device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")  

            # Load model and label name.
            w = resolve_yolo_checkpoint(self.autolabel_weights, os.path.join(APP_ROOT, 'weights'))
            model = YOLO(w)  # load FP32 model
            names = model.module.names if hasattr(model, 'module') else model.names
            # print('888:', names)
            result = model.predict(source=source, conf=conf_thres, iou=iou_thres)[0]

            image = QImage()
            ok = image.load(self.filePath)
            imageShape = [image.height(), image.width(),
                          1 if image.isGrayscale() else 3]

            use_pose = (getattr(self.canvas, 'drawingShapeMode', '') == 'keypoints') or (self.ifc_fish_meta is not None)
            wrote_pose = False
            if use_pose:
                xyxy, scores, labels, kpts_vis = self._extract_pose_rows_from_result(result, names)
                if xyxy and kpts_vis:
                    xyxy, scores, labels, kpts_vis = self._dedup_pose_same_class(
                        xyxy, scores, labels, kpts_vis, dedup_iou)
                    if self.yolo_classes:
                        class_list = list(self.yolo_classes)
                    else:
                        class_list = self._ordered_model_class_names(names)
                    writer = YOLOWriter(self.img_folder_path, os.path.basename(self.filePath),
                                        imageShape, localImgPath=self.filePath)
                    for box, score, label, (pts, vis) in zip(xyxy, scores, labels, kpts_vis):
                        canon = self._canonical_label_in_yolo_class_list(label, class_list)
                        writer.addKeypoints(pts, vis, canon, score)
                    writer.save(targetFile=self.txtPath, classList=class_list)
                    self._sync_lists_after_autolabel(class_list)
                    wrote_pose = True

            if not wrote_pose:
                dets = result.boxes
                if dets is None or len(dets) == 0:
                    QMessageBox.information(self, u'Info', u'未检测到目标。')
                    return
                labels = [names[int(x)] for x in dets.cls]
                scores = [float(x) for x in dets.conf]
                boxes = [x.tolist() for x in dets.xyxy]
                boxes, scores, labels = self._dedup_same_class_overlaps(boxes, scores, labels, dedup_iou)
                if self.yolo_classes:
                    class_list = list(self.yolo_classes)
                else:
                    class_list = self._ordered_model_class_names(names)
                writer = YOLOWriter(self.img_folder_path, os.path.basename(self.filePath),
                                     imageShape, localImgPath=self.filePath)          
                for box, score, label in zip(boxes, scores, labels):
                    canon = self._canonical_label_in_yolo_class_list(label, class_list)
                    writer.addBndBox(int(box[0]), int(box[1]), int(box[2]), int(box[3]), canon, score)
                writer.save(targetFile=self.txtPath, classList=class_list)
                self._sync_lists_after_autolabel(class_list)

            self.loadFile(self.filePath)
            # self.refreshImg()
            # QMessageBox.information(self, u'Done!', u'auto labeling done,and reload img folder')
        except Exception as e:
            self._report_error('autoLabelCurrentImg')

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)
        self.canvas.setFocus()

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)
        if not edit:
            self.canvas.setFocus()

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.predefined_classes) > 0:
            self.labelDialog = LabelDialog(
                parent=self, listItem=self.predefined_classes)
        
        text = self.labelDialog.popUp(text=self.prevLabelText)#(item.text())

            
        if text is not None:
            self.lastLabel = text
            self.prevLabelText = text
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()
            if not text in self.predefined_classes:
                self.predefined_classes.append(text)
                self.defaultLabelCombobox.clear()
                self.defaultLabelCombobox.addItems(self.predefined_classes)
            if text not in self.yolo_classes:
                self.yolo_classes.append(text)
            if text not in self.labelHist:
                self.labelHist.append(text)
    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        if item is None:
            return
        data_path = item.data(Qt.UserRole)
        filename = ustr(data_path) if data_path else ustr(item.text())
        if filename:
            self.loadFile(filename)

    def _setFileListItems(self, file_list, reasons_map=None):
        self.fileListWidget.clear()
        self.mImgList = list(file_list)
        for imgPath in self.mImgList:
            display_text = imgPath
            if reasons_map and imgPath in reasons_map:
                tags = reasons_map.get(imgPath, [])
                if tags:
                    display_text = '%s  [%s]' % (imgPath, ', '.join(tags))
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, imgPath)
            self.fileListWidget.addItem(item)

    def _guessLabelPathForImage(self, imgPath):
        if not self.defaultSaveDir or (not os.path.isdir(self.defaultSaveDir)):
            return None
        base = os.path.splitext(os.path.basename(imgPath))[0]
        txt_path = os.path.join(self.defaultSaveDir, base + '.txt')
        xml_path = os.path.join(self.defaultSaveDir, base + '.xml')
        if os.path.isfile(txt_path):
            return txt_path
        if os.path.isfile(xml_path):
            return xml_path
        return None

    def _readFileAnnoMeta(self, imgPath):
        meta = {'species': set(), 'scores': [], 'count': 0}
        label_path = self._guessLabelPathForImage(imgPath)
        if not label_path:
            return meta

        if label_path.lower().endswith('.xml'):
            tree = ET.ElementTree(file=label_path)
            root = tree.getroot()
            count = 0
            species = set()
            for obj in root.findall('object'):
                count += 1
                name_node = obj.find('name')
                if name_node is not None and name_node.text:
                    species.add(name_node.text.strip())
            meta['species'] = species
            meta['count'] = count
            return meta

        # YOLO txt
        classes_path = os.path.join(self.defaultSaveDir, 'classes.txt')
        classes = []
        if os.path.isfile(classes_path):
            with open(classes_path, 'r', encoding='utf-8') as f:
                classes = [x.strip() for x in f.read().splitlines()]
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        species = set()
        scores = []
        count = 0
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            count += 1
            try:
                cid = int(parts[0])
                if 0 <= cid < len(classes):
                    species.add(classes[cid])
            except Exception:
                pass
            # score field exists in this project for auto-labeled YOLO lines (6th token for bbox lines)
            if len(parts) == 6:
                try:
                    scores.append(float(parts[5]))
                except Exception:
                    pass
        meta['species'] = species
        meta['scores'] = scores
        meta['count'] = count
        return meta

    def rebuildFileAnnoMetaCache(self):
        self.fileAnnoMeta = {}
        for img_path in self.allImgList:
            try:
                self.fileAnnoMeta[img_path] = self._readFileAnnoMeta(img_path)
            except Exception:
                self.fileAnnoMeta[img_path] = {'species': set(), 'scores': [], 'count': 0}

    def refreshSpeciesFilterOptions(self):
        curr = self.fileSpeciesFilterCombo.currentText()
        species_all = set()
        for meta in self.fileAnnoMeta.values():
            for s in meta.get('species', set()):
                if s:
                    species_all.add(s)
        options = ['All species'] + sorted(species_all)
        self.fileSpeciesFilterCombo.blockSignals(True)
        self.fileSpeciesFilterCombo.clear()
        self.fileSpeciesFilterCombo.addItems(options)
        idx = self.fileSpeciesFilterCombo.findText(curr)
        self.fileSpeciesFilterCombo.setCurrentIndex(idx if idx >= 0 else 0)
        self.fileSpeciesFilterCombo.blockSignals(False)

    def applyFileListFilters(self):
        if not self.allImgList:
            self._setFileListItems([])
            return

        selected_species = self.fileSpeciesFilterCombo.currentText().strip()
        species_active = selected_species and selected_species != 'All species'
        score_active = self.fileScoreFilterCheck.isChecked()
        score_thr = float(self.fileScoreFilterSpin.value())
        count_active = self.fileCountFilterCheck.isChecked()
        count_thr = int(self.fileCountFilterSpin.value())

        reasons_map = {}
        if not (species_active or score_active or count_active):
            self._setFileListItems(self.allImgList, reasons_map)
            return

        filtered = []
        for img_path in self.allImgList:
            meta = self.fileAnnoMeta.get(img_path, {'species': set(), 'scores': [], 'count': 0})
            hit_species = species_active and (selected_species in meta.get('species', set()))
            hit_score = score_active and any((s < score_thr) for s in meta.get('scores', []))
            hit_count = count_active and (int(meta.get('count', 0)) > count_thr)
            if hit_species or hit_score or hit_count:
                filtered.append(img_path)
                tags = []
                if hit_species:
                    tags.append('species')
                if hit_score:
                    tags.append('low-score')
                if hit_count:
                    tags.append('many-boxes')
                reasons_map[img_path] = tags
        self._setFileListItems(filtered, reasons_map)

    # Add chris
    # def btnstate(self, item= None):
    #     """ Function to handle difficult examples
    #     Update on each object """
    #     if not self.canvas.editing():
    #         return

    #     item = self.currentItem()
    #     if not item: # If not selected Item, take the first one
    #         item = self.labelList.item(self.labelList.count()-1)

    #     # difficult = self.diffcButton.isChecked()

    #     try:
    #         shape = self.itemsToShapes[item]
    #     except:
    #         pass
    #     # Checked and Update
    #     try:
    #         if difficult != shape.difficult:
    #             shape.difficult = difficult
    #             self.setDirty()
    #         else:  # User probably changed item visibility
    #             self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
    #     except:
    #         pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                item = self.shapesToItems.get(shape)
                if item is not None:
                    item.setSelected(True)
                else:
                    # Selection can briefly outlive label-item mappings during reset/load.
                    self.labelList.clearSelection()
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        try:
            shape.paintLabel = self.displayLabelOption.isChecked()
            item = HashableQListWidgetItem(shape.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setBackground(generateColorByText(shape.label))
            self.itemsToShapes[item] = shape
            self.shapesToItems[shape] = item
            self.labelList.addItem(item)
            for action in self.actions.onShapesPresent:
                action.setEnabled(True)
            self.updateComboBox()
        except Exception as e:
            print(e)

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]
        self.updateComboBox()

    def loadLabels(self, shapes):
        s = []
        for shape_data in shapes:
            if len(shape_data) >= 9:
                label, points, line_color, fill_color, difficult, shape_type, keypoint_visibility, keypoint_names, skeleton = shape_data[:9]
            elif len(shape_data) >= 7:
                label, points, line_color, fill_color, difficult, shape_type, keypoint_visibility = shape_data[:7]
                keypoint_names = []
                skeleton = []
            else:
                label, points, line_color, fill_color, difficult = shape_data[:5]
                shape_type = 'bbox'
                keypoint_visibility = []
                keypoint_names = []
                skeleton = []

            shape = Shape(label=label, shape_type=shape_type)
            shape.keypoint_visibility = list(keypoint_visibility)
            shape.keypoint_names = list(keypoint_names) if keypoint_names else list(self.canvas.keypointNames)
            shape.skeleton = [tuple(x) for x in skeleton] if skeleton else [tuple(x) for x in self.canvas.keypointSkeleton]
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            if shape_type != 'keypoints':
                shape.close()
            else:
                tmpl = self.canvas.keypointNames or []
                nk = max(len(shape.points), len(tmpl))
                if nk > 0:
                    pad_keypoints_shape(shape, nk, tmpl)
                    pm = self.canvas.pixmap
                    if pm is not None and not pm.isNull():
                        spread_keypoint_placeholders(shape, pm.width(), pm.height())
                    elif self.image is not None and not self.image.isNull():
                        spread_keypoint_placeholders(shape, self.image.width(), self.image.height())
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)

            self.addLabel(shape)
        self.updateComboBox()
        self.canvas.loadShapes(s)

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]
            
        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append('')
        uniqueTextList.sort()
        # self.comboBox.clear()
        self.comboBox.update_items(uniqueTextList)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        shape_type=getattr(s, 'shape_type', 'bbox'),
                        keypoint_visibility=list(getattr(s, 'keypoint_visibility', [])),
                        keypoint_names=list(getattr(s, 'keypoint_names', [])),
                        skeleton=list(getattr(s, 'skeleton', [])),
                       # add chris
                        difficult = s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                if annotationFilePath[-4:].lower() != ".xml":
                    annotationFilePath += XML_EXT
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
                
            elif self.usingYoloFormat is True:
                if annotationFilePath[-4:].lower() != ".txt":
                    annotationFilePath += TXT_EXT
                self.labelFile.saveYoloFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.yolo_classes,
                                                   saveSegmentation=self.yoloSegmentationSaveOption.isChecked(),
                                                   lineColor=self.lineColor.getRgb(), fillColor=self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            # print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            self.lastLabelFile = annotationFilePath
            print('Annotation File: %s saved'%self.lastLabelFile)
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)
    
    def comboSelectionChanged(self, index):
        text = self.comboBox.cb.itemText(index)
        for i in range(self.labelList.count()):
            if text == '':
                self.labelList.item(i).setCheckState(2) 
            elif text != self.labelList.item(i).text():
                self.labelList.item(i).setCheckState(0)
            else:
                self.labelList.item(i).setCheckState(2)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            # self.diffcButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked():# or not self.defaultLabelCombobox.currentText():
            if len(self.predefined_classes) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.predefined_classes)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelCombobox.currentText()
            self.lastLabel = text

        # Add Chris
        # self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
            if text not in self.yolo_classes:
                self.yolo_classes.append(text)
            if text not in self.predefined_classes:
                self.predefined_classes.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)
        # print(filePath)
        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Highlight the file item
        # print('unicodeFilePath:', unicodeFilePath)
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                print('1408: labelFile True')
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                    print('1411:', self.labelFile)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified
            else:
                # Load image:
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
                self.canvas.verified = False

            image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            # if self.usingPascalVocFormat is True:
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(self.filePath)[0])
                self.xmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
                self.txtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)
                # print('1455:', self.txtPath)
                """Annotation file priority:
                PascalXML > YOLO
                """
                if os.path.isfile(self.xmlPath):
                    self.loadPascalXMLByFilename(self.xmlPath)
                elif os.path.isfile(self.txtPath):
                    try:
                        self.loadYOLOTXTByFilename(self.txtPath)
                    except Exception as e:
                        QMessageBox.information(self,
                                                u'error', 'no classes.txt file in annotation folder for yolo format. ({})'.format(
                                                    print_error()))
            else:
                self.xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                self.txtPath = os.path.splitext(filePath)[0] + TXT_EXT
                if os.path.isfile(self.xmlPath):
                    self.loadPascalXMLByFilename(self.xmlPath)
                elif os.path.isfile(self.txtPath):
                    try:
                        self.loadYOLOTXTByFilename(self.txtPath)
                    except Exception as e:
                        QMessageBox.information(self,u'error!','no classes.txt in annotation folder for yolo format. ({})'.format(print_error()))

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count()-1))
                self.labelList.item(self.labelList.count()-1).setSelected(True)

            self.canvas.setFocus(True)
            # print('1487:', self.labelFile)
            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''
        settings[SETTING_LAST_OPEN_IMAGE] = self.filePath if (self.filePath and os.path.isfile(self.filePath)) else ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''
        if self.lastWeightDir and os.path.exists(self.lastWeightDir):
            settings[SETTING_LAST_WEIGHT_DIR] = self.lastWeightDir
        else:
            settings[SETTING_LAST_WEIGHT_DIR] = ''
        self._persist_autolabel_settings()
        settings[SETTING_FILE_FILTER_SPECIES] = ustr(self.fileSpeciesFilterCombo.currentText())
        settings[SETTING_FILE_FILTER_SCORE_ENABLED] = self.fileScoreFilterCheck.isChecked()
        settings[SETTING_FILE_FILTER_SCORE_THRESHOLD] = float(self.fileScoreFilterSpin.value())
        settings[SETTING_FILE_FILTER_COUNT_ENABLED] = self.fileCountFilterCheck.isChecked()
        settings[SETTING_FILE_FILTER_COUNT_THRESHOLD] = int(self.fileCountFilterSpin.value())
        settings[SETTING_AUTO_UPDATE_ENABLED] = bool(self.autoUpdateEnabled)
        settings[SETTING_UPDATE_MANIFEST_URL] = ustr(self.updateManifestUrl)
        settings[SETTING_LAST_UPDATE_CHECK] = ustr(self.lastUpdateCheckAt)

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
        settings[SETTING_DRAW_MODE] = self.canvas.drawingShapeMode
        settings[SETTING_YOLO_SAVE_SEG] = self.yoloSegmentationSaveOption.isChecked()
        settings[SETTING_Magnifying_Lens] = self.useMagnifyingLens.isChecked()
        settings[SETTING_KEYPOINT_TEMPLATE] = self.currentKeypointTemplate
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                       '%s - Save annotations to the directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                       | QFileDialog.DontResolveSymlinks))
        print(dirpath)
        if dirpath:
            self.defaultSaveDir = dirpath
            self.handleSavedirLabelNames(dirpath)
        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def detectSavedirAnnoFormat(self, dirpath):
        txt_count = 0
        xml_count = 0
        try:
            for name in os.listdir(dirpath):
                lower_name = name.lower()
                if lower_name.endswith('.xml'):
                    xml_count += 1
                elif lower_name.endswith('.txt') and lower_name != 'classes.txt':
                    txt_count += 1
        except Exception:
            return None

        if txt_count > 0 and xml_count == 0:
            return 'yolo'
        if xml_count > 0:
            return 'xml'
        if txt_count > 0:
            return 'yolo'
        return None

    def handleSavedirLabelNames(self, dirpath):
        self._apply_ifc_fish_yolo_from_dir(dirpath)
        classes_path = os.path.join(dirpath, 'classes.txt')
        anno_format = self.detectSavedirAnnoFormat(dirpath)

        if os.path.isfile(classes_path):
            self.loadPredefinedClasses(classes_path)
            self.yolo_classes = list(self.predefined_classes)
            return

        if anno_format == 'yolo':
            QMessageBox.information(
                self,
                u'提示',
                u'当前标注目录检测为 YOLO(.txt)，但未找到 classes.txt，请手动加载分类清单。'
            )
            self.load_label_names()
        elif anno_format == 'xml':
            QMessageBox.information(
                self,
                u'提示',
                u'当前标注目录检测为 XML，但未找到 classes.txt，请手动加载分类清单。'
            )
            self.load_label_names()
        # Label directory changed; refresh file filters metadata.
        if self.allImgList:
            self.rebuildFileAnnoMetaCache()
            self.refreshSpeciesFilterOptions()
            self.applyFileListFilters()

    def openAnnotationDialog(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        if self.usingPascalVocFormat:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self,'%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return
            

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent!=True :
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                         '%s - Open Directory' % __appname__, defaultOpenDirPath,
                                                         QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)
        #print(targetDirPath)
        self.img_folder_path = targetDirPath
        self.importDirImages(targetDirPath)

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.allImgList = self.scanAllImages(dirpath)
        self.rebuildFileAnnoMetaCache()
        self.refreshSpeciesFilterOptions()
        self.applyFileListFilters()

        if not self.mImgList:
            return

        startupImage = self._startupImagePath
        self._startupImagePath = None
        if startupImage and startupImage in self.mImgList:
            self.loadFile(startupImage)
        else:
            self.openNextImg()

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        self.navigateImageByStep(-1)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        self.navigateImageByStep(1)

    def navigateImageByStep(self, step):
        if len(self.mImgList) <= 0:
            return
        if step == 0:
            return

        # Keep navigation stable even if the current image failed to load.
        current_path = self.filePath
        try:
            current_index = self.mImgList.index(current_path) if current_path in self.mImgList else -1
        except Exception:
            current_index = -1

        # If there is no current image, start from one side.
        if current_index < 0:
            candidate_index = 0 if step > 0 else len(self.mImgList) - 1
        else:
            candidate_index = current_index + step

        while 0 <= candidate_index < len(self.mImgList):
            filename = self.mImgList[candidate_index]
            if filename and self.loadFile(filename):
                return
            # Skip unreadable/broken images and continue in the same direction.
            candidate_index += step

        self.status('No more valid images in this direction.')

    def refreshImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        # if self.filePath is None:
        filename = self.mImgList[0]
        # else:
        #     currIndex = self.mImgList.index(self.filePath)
        #     if currIndex + 1 < len(self.mImgList):
        #         filename = self.mImgList[currIndex + 1]
        
        if filename:
            self.loadFile(filename)


    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0]
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                #print(savedPath)
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog(removeExt=False))

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0] # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def _after_predefined_class_names_changed(self):
        """Combobox must already reflect self.predefined_classes. Rebind LabelDialog and reset
        last/prev/default strings so new boxes do not reuse a label from a previous class list."""
        classes = [c.strip() for c in self.predefined_classes if c and str(c).strip()]
        self.predefined_classes = classes
        if self.defaultLabelCombobox.count():
            self.defaultLabelCombobox.setCurrentIndex(0)
        first = self.defaultLabelCombobox.currentText().strip() if self.defaultLabelCombobox.count() else ''
        self.default_class = first
        self.prevLabelText = first
        self.lastLabel = first if first else None
        if first:
            self.default_label = first
        self.yolo_classes = list(classes)
        self.labelDialog = LabelDialog(parent=self, listItem=classes)

    def loadPredefinedClasses(self, predefClassesFile):
        if not os.path.isfile(predefClassesFile):
            return
        with open(predefClassesFile, 'r', encoding='utf-8') as f:
            raw = f.read().splitlines()
        self.predefined_classes = [x.strip() for x in raw if x.strip()]
        self.defaultLabelCombobox.clear()
        if self.predefined_classes:
            self.defaultLabelCombobox.addItems(self.predefined_classes)
        self._after_predefined_class_names_changed()
        print('Set default class name as:', self.default_class)
        print('load class names: ', len(self.predefined_classes))

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self.filePath is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        self.set_format(FORMAT_YOLO)
        self._apply_ifc_fish_yolo_from_dir(os.path.dirname(txtPath))
        posed_n = None
        if self.ifc_fish_meta and self.ifc_fish_meta.get('kpt_n'):
            posed_n = int(self.ifc_fish_meta['kpt_n'])
        tYoloParseReader = YoloReader(txtPath, self.image, posed_kpts_count=posed_n)
        shapes = tYoloParseReader.getShapes()
        # print(shapes)
        self.yolo_classes = tYoloParseReader.classes
        # print('1909:', shapes)
        self.loadLabels(shapes)
        self.canvas.verified = tYoloParseReader.verified

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())

    def toggleYoloSegmentationSaveMode(self):
        mode = 'segmentation' if self.yoloSegmentationSaveOption.isChecked() else 'bbox'
        self.statusBar().showMessage('YOLO save mode: %s' % mode, 3000)
        
    def function_test(self):
        progress = QProgressDialog(self)
        progress.setWindowTitle("请稍等")  
        progress.setLabelText("正在操作...")
        progress.setCancelButtonText("取消")
        progress.setMinimumDuration(5)
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0,100) 
        for i in range(100):
            progress.setValue(i) 
            import time
            time.sleep(0.2)
            if progress.wasCanceled():
                QMessageBox.warning(self,"提示","操作失败,请检查文件路径！") 
                break
        else:
            progress.setValue(100)
            QMessageBox.information(self,"提示","数据增强成功！")
    
    def test_act(self):
        go=True
        while go:
            if self.show_test():
                QMessageBox.information(self,u'Wrong!',u'!想多了！')
            else:
                go=False
                QMessageBox.information(self,u'Right!',u'想啥！') 
        
    def show_test(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'IDEA?'
        return no == QMessageBox.warning(self, u'Attention:', msg, yes | no)  

    def search_actions_info(self):
        """this is a action information search system.
        input actions name, ti will tell you what it is and how to use it.
        for example, input 'rename_img_xml' or 'ca1' can read actions <rename_img_xml>'s instruction.
        key_word must be actions in menu bar or its shorcut key for now, more intelligent system will update lately. 
        """
        def find_max_similarity(test_str,str_list,threshold=0.7):
            jarowinkler = JaroWinkler()
            max_simi=threshold
            max_str=None
            for string in str_list:
                simi=jarowinkler.similarity(test_str, string)
                if simi > max_simi:
                    max_str=string
                    max_simi = simi
                    
            return max_str
            
        search_key, ok=QInputDialog.getText(self, 'Text Input Dialog', 
                    "Input your search key word：\n(input nothing for search-system's own instruction)")
        if (not ok):
            return
        action_dict={'load_label_names':self.load_label_names,'c0':self.load_label_names,
                     'batch_rename_img':self.batch_rename_img,'c1':self.batch_rename_img,
                     'rename_img_xml':self.rename_img_xml,'ca1':self.rename_img_xml,
                     'duplicate_xml':self.make_duplicate_xml,'c2':self.make_duplicate_xml,
                     'batch_duplicate':self.batch_duplicate_xml,'ca2':self.batch_duplicate_xml,
                     'label_pruning':self.prune_useless_label,'c3':self.prune_useless_label,
                     'file_pruning':self.remove_extra_img_xml,'ca3':self.remove_extra_img_xml,
                     'change_label':self.change_label_name,'c4':self.change_label_name,
                     'fix_property':self.fix_xml_property,'c5':self.fix_xml_property,
                     'choose_autolabel_model':self.choose_autolabel_model,'csm':self.choose_autolabel_model,
                     'auto_labeling_one':self.auto_labeling_one,'c6':self.auto_labeling_one,
                     'auto_labeling_multi':self.auto_labeling_multi,'c7':self.auto_labeling_multi,
                     'make_training_datasets_one':self.make_datasets_one, 'ca8':self.make_datasets_one,
                     'make_training_datasets':self.make_datasets, 'c8':self.make_datasets,
                     'training_model':self.training_model,'c9':self.training_model,
                     # 'data_agument':self.data_auto_agument,'c8':self.data_auto_agument,
                     'folder_info':self.show_folder_infor,'a1':self.show_folder_infor,
                     'label_info':self.show_label_info,'a2':self.show_label_info,
                     'extract_video':self.extract_video,'s1':self.extract_video,
                     'extract_videos':self.extract_videos,'s2':self.extract_videos,
                     'extract_stream':self.extract_stream,'s3':self.extract_stream,
                     'batch_resize_img':self.batch_resize_img,'s4':self.batch_resize_img,
                     'merge_video':self.merge_video,'s5':self.merge_video,
                     'annotation_video':self.annotation_video,'s65':self.annotation_video,
                     'Search_System':self.search_actions_info
                     }
        search_key='Search_System' if search_key=='' else search_key
        if search_key in action_dict.keys():
            search_info=action_dict[search_key].__doc__
            search_info=search_info.replace('  ','')
            QMessageBox.information(self,u'Info!',search_info)
        else:
            vague_key=find_max_similarity(search_key,action_dict.keys())
            if vague_key in action_dict.keys():
                search_info=action_dict[vague_key].__doc__
                search_info=search_info.replace('  ','')
                search_info="here is info about '{}' based on your input: '{}'\n\n".format(vague_key,search_key)+search_info
                QMessageBox.information(self,u'Info!',search_info)
            else:
                QMessageBox.information(self,u'Sorry!',
                u'unkown key word, key word must in(or similar to) actions in menu bar or its shotcut key, please try again.')
        
    def batch_rename_img(self):
        """batch rename img name. 
        new name constructed by key_word, index, if_fill three prospoty,for example,'car_1.jpg'(or 'car_001.jpg' if fill to 3 digit). 
        additionally, '_' will not appear when key_word is empty. after this actions, you may need reopen img folder.
        !!makesure your new name not conflict with your old name!!
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:
            path=os.path.dirname(self.filePath)
            filelist=natsort.natsorted(os.listdir(path))
            key_word,ok=QInputDialog.getText(self, 'Text Input Dialog',"Input key word：")
            if not ok:
                return
            index,ok=QInputDialog.getInt(self, 'Text Input Dialog',"Input index：",value=1)
            if not ok:
                return
            Fill,ok=QInputDialog.getInt(self, 'Text Input Dialog',
                    'Digit Fill\n(fill means 1->001, 0 means no fill)',value=0)
            if not ok:
                return
            if 1 < Fill < len(str(index+len(filelist))):
                QMessageBox.information(self,u'Waring!',
                u"your Fill is smaller than largest index's digit, try larger Fill or input 0 to use no fill")
                return
            key_word = '' if key_word == '' else key_word+'_'
            for item in filelist:
                if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png'):
                    filepath=os.path.join(os.path.abspath(path), item)
                    if Fill > 1:
                        new_item='{}{}.jpg'.format(key_word,str(index).zfill(Fill))
                    else:
                        new_item='{}{}.jpg'.format(key_word,str(index))
                    new_filepath=os.path.join(os.path.abspath(path), new_item)
                    os.rename(filepath,new_filepath)
                    index+=1
            QMessageBox.information(self,u'Done!',u'Batch rename done.')
        except Exception as e:
            self._report_error('training_model')
        
    def rename_img_xml(self):
        """batch rename img's and its corresponding xml's name. 
        new name constructed by key_word, index, if_fill three prospoty,for example,'car_1.jpg'(or 'car_001.jpg' if fill to 3 digit). 
        additionally, '_' will not appear when key_word is empty.after this actions, you may need reopen img folder.
        !!makesure your new name not conflict with your old name!!
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:
            img_folder_path=os.path.dirname(self.filePath)
            xml_folder_path=self.defaultSaveDir
            imglist = natsort.natsorted(os.listdir(img_folder_path))
            xmllist = natsort.natsorted(os.listdir(xml_folder_path))
            key_word,ok=QInputDialog.getText(self, 'Text Input Dialog',"Input key word：")
            if not ok:
                return
            index,ok=QInputDialog.getInt(self, 'Int Input Dialog',"Input index：",value=1)
            if not ok:
                return
            Fill,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                    'Digit Fill\n(fill means 1->001, 0 means no fill)',value=0)
            if not ok:
                return
            if 1 < Fill < len(str(index+len(imglist))):
                QMessageBox.information(self,u'Waring!',
                u"your Fill is smaller than largest index's digit, try larger Fill or input 0 to use no fill")
                return
            key_word = '' if key_word == '' else key_word+'_'
            for item in xmllist:
                if item.endswith('.xml') and (item[0:-4]+'.jpg' in imglist or item[0:-4]+'.JPG' in imglist):
                    xmlPath=os.path.join(os.path.abspath(xml_folder_path), item)
                    imgPath=os.path.join(os.path.abspath(img_folder_path), item[0:-4])+'.jpg'
                    if Fill > 1:
                        new_item='{}{}'.format(key_word,str(index).zfill(Fill))
                    else:
                        new_item='{}{}'.format(key_word,str(index))
                    new_xmlPath=os.path.join(os.path.abspath(xml_folder_path), new_item+'.xml')
                    new_imgPath=os.path.join(os.path.abspath(img_folder_path), new_item+'.jpg')
                    os.rename(xmlPath,new_xmlPath)
                    os.rename(imgPath,new_imgPath)
                    index+=1
                else:
                    pass
            QMessageBox.information(self,u'Done!',u'Batch rename done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
            
    def make_duplicate_xml(self):
        """copy last xml file to local img, make sure last xml exist. 
        if local xml exist, you need confirm to overwrite it.
        """
        try:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex - 1 >= 0:
                last_filename = self.mImgList[currIndex - 1]
                imgFileName = os.path.basename(last_filename)
                last_xml = os.path.splitext(imgFileName)[0]
                last_path=os.path.join(ustr(self.defaultSaveDir),last_xml+'.xml')
                
                currfilename = self.mImgList[currIndex]
                imgFileName = os.path.basename(currfilename)
                curr_xml = os.path.splitext(imgFileName)[0]
                save_path=os.path.join(ustr(self.defaultSaveDir),curr_xml+'.xml')

                xml_info={'filename':'none','path':'none'}
                xml_info['filename']=curr_xml+'.jpg'
                xml_info['path']=str(self.filePath)
                if os.path.exists(save_path):
                    if self.question_1():
                        print('over write!')
                        pass
                    else:
                        print('cancled!')
                        return

                tree = ET.ElementTree(file=last_path)
                root=tree.getroot()
                for key in xml_info.keys():
                    root.find(key).text=xml_info[key]
                tree.write(save_path)
            else:
                QMessageBox.information(self,u'Sorry!',u'please ensure the first xml file exists.')
                return
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
            
    def question_1(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'current xml exists,procesing anyway?'
        return yes == QMessageBox.warning(self, u'Attention:', msg, yes | no)
        
    def batch_duplicate_xml(self):
        """batch copy xml file, make sure at least the first xml exist.
        this action will not overwrite xml file, if local xml exist, it will jump to next and copy local xml to next one.
        """
        if len(self.mImgList) <= 0:
            QMessageBox.information(self,u'Sorry!',u'something is wrong, try load img/xml path again.')
        else:
            for i in range(len(self.mImgList)):
                currfilename = self.mImgList[i]
                imgFileName = os.path.basename(currfilename)
                curr_xml = os.path.splitext(imgFileName)[0]
                save_path=os.path.join(ustr(self.defaultSaveDir),curr_xml+'.xml')
                if i ==0:
                    if os.path.exists(save_path):
                        pass
                    else:
                        QMessageBox.information(self,u'Sorry!',u'please ensure the first xml file exists.')
                        return
                else:
                    last_filename = self.mImgList[i - 1]
                    imgFileName = os.path.basename(last_filename)
                    last_xml = os.path.splitext(imgFileName)[0]
                    last_path=os.path.join(ustr(self.defaultSaveDir),last_xml+'.xml')
                    if os.path.exists(save_path):
                        pass
                    else:
                        xml_info={'filename':'none','path':'none'}
                        xml_info['filename']=curr_xml+'.jpg'
                        xml_info['path']=str(self.filePath)
                        tree = ET.ElementTree(file=last_path)
                        root=tree.getroot()
                        for key in xml_info.keys():
                            root.find(key).text=xml_info[key]
                        tree.write(save_path)          
            QMessageBox.information(self,u'Done!',u'batch duplicate xml file succeed, you can procesing other job now.')
        
    def prune_useless_label(self):
        """
        Label pruning (XML + YOLO txt) with UI filters:
        1) choose which label types to keep (multi-select, default: all)
        2) optional min area ratio filter (remove too-small boxes/polygons)
        3) optional min score filter (remove low-score bbox/seg lines when score column exists)
        """
        if self.filePath is None:
            QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
            return

        img_folder_path = os.path.abspath(os.path.dirname(self.filePath))
        label_folder_path = os.path.abspath(self.defaultSaveDir)
        if not os.path.isdir(label_folder_path):
            QMessageBox.information(self, u'Wrong!', u'please load label directory first.')
            return

        # Collect available label types (union of XML names + YOLO classes.txt).
        all_labels_set = set()
        has_xml = False
        for item in os.listdir(label_folder_path):
            if item.lower().endswith('.xml'):
                has_xml = True
                xml_path = os.path.join(label_folder_path, item)
                try:
                    tree = ET.ElementTree(file=xml_path)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        name_node = obj.find('name')
                        if name_node is not None and name_node.text:
                            all_labels_set.add(name_node.text)
                except Exception:
                    continue

        classes_path = os.path.join(label_folder_path, 'classes.txt')
        has_yolo = os.path.isfile(classes_path)
        if has_yolo:
            try:
                with open(classes_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        if line.strip():
                            all_labels_set.add(line.strip())
            except Exception:
                pass

        if not all_labels_set:
            QMessageBox.information(self, u'Wrong!', u'No XML objects or YOLO classes found in label folder.')
            return

        all_labels = sorted(list(all_labels_set))

        dlg = QDialog(self)
        dlg.setWindowTitle('Label pruning options')
        layout = QVBoxLayout(dlg)

        # (1) label type multi-select
        layout.addWidget(QLabel('Select label types to keep:'))
        label_list = QListWidget(dlg)
        label_list.setSelectionMode(QAbstractItemView.NoSelection)
        for lbl in all_labels:
            it = QListWidgetItem(lbl)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            label_list.addItem(it)
        layout.addWidget(label_list)

        btn_row = QHBoxLayout()
        btn_sel = QPushButton('Select All', dlg)
        btn_desel = QPushButton('Deselect All', dlg)
        btn_sel.clicked.connect(lambda: self._set_all_checks(label_list, True))
        btn_desel.clicked.connect(lambda: self._set_all_checks(label_list, False))
        btn_row.addWidget(btn_sel)
        btn_row.addWidget(btn_desel)
        layout.addLayout(btn_row)

        # (2) area filter
        area_row = QHBoxLayout()
        chk_area = QCheckBox('Enable min area ratio filter', dlg)
        chk_area.setChecked(True)
        spin_area = QDoubleSpinBox(dlg)
        spin_area.setDecimals(6)
        spin_area.setSingleStep(0.0005)
        spin_area.setRange(0.0, 1e9)
        spin_area.setValue(0.001)
        area_row.addWidget(chk_area)
        area_row.addWidget(QLabel('min_area_ratio:', dlg))
        area_row.addWidget(spin_area)
        layout.addLayout(area_row)

        # (3) score filter
        score_row = QHBoxLayout()
        chk_score = QCheckBox('Enable score filter (if score column exists)', dlg)
        chk_score.setChecked(False)
        spin_score = QDoubleSpinBox(dlg)
        spin_score.setDecimals(4)
        spin_score.setSingleStep(0.05)
        spin_score.setRange(0.0, 1e9)
        spin_score.setValue(0.5)
        score_row.addWidget(chk_score)
        score_row.addWidget(QLabel('min_score:', dlg))
        score_row.addWidget(spin_score)
        layout.addLayout(score_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec_() != QDialog.Accepted:
            return

        keep_labels = []
        for i in range(label_list.count()):
            it = label_list.item(i)
            if it.checkState() == Qt.Checked:
                keep_labels.append(it.text())

        if not keep_labels:
            QMessageBox.information(self, u'Wrong!', u'You selected no label types.')
            return

        min_area_ratio = spin_area.value() if chk_area.isChecked() else None
        min_score = spin_score.value() if chk_score.isChecked() else None

        # Confirm
        confirm = u'Keep labels ({} types): {}\n'.format(len(keep_labels), ','.join(keep_labels[:20]))
        if len(keep_labels) > 20:
            confirm += u'...(and {} more)\n'.format(len(keep_labels) - 20)
        if min_area_ratio is not None:
            confirm += u'Min area ratio: {}\n'.format(min_area_ratio)
        if min_score is not None:
            confirm += u'Min score: {}\n'.format(min_score)
        confirm += u'\nThis will delete/clean label files (and orphan images). Please back up first.'

        yes = QMessageBox.Yes
        if QMessageBox.warning(self, u'Attention:', confirm, yes | QMessageBox.No) != yes:
            return

        try:
            renamed_dups, copied_labels = self._dedupe_same_basename_images_and_sync_labels(
                img_folder_path, label_folder_path
            )

            xml_exists = has_xml
            yolo_exists = has_yolo

            xml_msg = ''
            yolo_msg = ''
            if xml_exists:
                removed_objs, removed_xmls, removed_xml_imgs = self._prune_xml_labels(
                    label_folder_path, img_folder_path, keep_labels, min_area_ratio=min_area_ratio
                )
                xml_msg = 'XML: removed {} objects, {} labels, {} images'.format(
                    removed_objs, removed_xmls, removed_xml_imgs
                )

            if yolo_exists:
                removed_lines, removed_txts, removed_txt_imgs = self._prune_yolo_labels(
                    label_folder_path, img_folder_path, keep_labels,
                    min_area_ratio=min_area_ratio, min_score=min_score
                )
                yolo_msg = 'YOLO: removed {} lines, {} labels, {} images'.format(
                    removed_lines, removed_txts, removed_txt_imgs
                )

            if not xml_exists and not yolo_exists:
                QMessageBox.information(self, u'Wrong!', u'no xml/txt labels found in current label folder.')
                return

            done_msg = u'label pruning done.'
            if renamed_dups > 0:
                done_msg += u'\nrenamed duplicate-basename images: {}'.format(renamed_dups)
            if copied_labels > 0:
                done_msg += u'\n.synced copied labels: {}'.format(copied_labels)
            if xml_msg:
                done_msg += u'\n' + xml_msg
            if yolo_msg:
                done_msg += u'\n' + yolo_msg
            QMessageBox.information(self, u'Done!', done_msg)
        except Exception:
            return

    def _set_all_checks(self, list_widget, checked):
        for i in range(list_widget.count()):
            it = list_widget.item(i)
            it.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _supported_image_exts(self):
        return ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']

    def _image_ext_priority(self, ext):
        order = {'.jpg': 0, '.jpeg': 1, '.png': 2, '.bmp': 3, '.webp': 4, '.tif': 5, '.tiff': 6}
        return order.get(ext.lower(), 999)

    def _build_image_index(self, img_folder_path):
        image_map = {}
        for name in os.listdir(img_folder_path):
            ext = os.path.splitext(name)[1].lower()
            if ext in self._supported_image_exts():
                base = os.path.splitext(name)[0]
                image_map.setdefault(base, []).append(name)
        for base in image_map:
            image_map[base] = sorted(
                image_map[base],
                key=lambda n: (self._image_ext_priority(os.path.splitext(n)[1]), n.lower())
            )
        return image_map

    def _dedupe_same_basename_images(self, img_folder_path):
        image_map = self._build_image_index(img_folder_path)
        rename_count = 0
        for base, names in image_map.items():
            if len(names) <= 1:
                continue
            # Keep one image with original basename; rename others as *_jpg.jpg, *_png.png...
            for name in names[1:]:
                ext = os.path.splitext(name)[1].lower()
                ext_tag = ext.lstrip('.')
                src = os.path.join(img_folder_path, name)
                target_name = "{}_{}{}".format(base, ext_tag, ext)
                target = os.path.join(img_folder_path, target_name)
                suffix = 1
                while os.path.exists(target):
                    target_name = "{}_{}_{:02d}{}".format(base, ext_tag, suffix, ext)
                    target = os.path.join(img_folder_path, target_name)
                    suffix += 1
                os.rename(src, target)
                rename_count += 1
        return rename_count

    def _dedupe_same_basename_images_and_sync_labels(self, img_folder_path, label_folder_path):
        """
        Rename duplicated images sharing the same basename, and also sync label filenames (xml/txt).
        This makes label basename -> image basename a one-to-one mapping.
        """
        image_map = self._build_image_index(img_folder_path)
        rename_count = 0
        copy_count = 0

        for base, names in image_map.items():
            if len(names) <= 1:
                continue

            # Keep the first image unchanged; rename the rest.
            for name in names[1:]:
                ext = os.path.splitext(name)[1].lower()
                ext_tag = ext.lstrip('.')

                src = os.path.join(img_folder_path, name)
                target_base = "{}_{}".format(base, ext_tag)

                target_name = "{}{}".format(target_base, ext)
                target = os.path.join(img_folder_path, target_name)
                suffix = 1
                while os.path.exists(target):
                    target_name = "{}_{}{}".format(target_base, suffix, ext)
                    target = os.path.join(img_folder_path, target_name)
                    suffix += 1

                # Copy labels for the renamed image, so pruning won't delete the renamed image.
                old_xml = os.path.join(label_folder_path, base + '.xml')
                new_xml = os.path.join(label_folder_path, target_base + '.xml')
                if os.path.isfile(old_xml) and (not os.path.isfile(new_xml)):
                    shutil.copy2(old_xml, new_xml)
                    copy_count += 1

                old_txt = os.path.join(label_folder_path, base + '.txt')
                new_txt = os.path.join(label_folder_path, target_base + '.txt')
                if os.path.isfile(old_txt) and (not os.path.isfile(new_txt)):
                    # classes.txt is excluded by name, but base+".txt" is safe here.
                    shutil.copy2(old_txt, new_txt)
                    copy_count += 1

                os.rename(src, target)
                rename_count += 1

        return rename_count, copy_count

    def _prune_xml_labels(self, xml_folder_path, img_folder_path, keep_labels, min_area_ratio=None):
        removed_objs = 0
        removed_labels = 0
        removed_images = 0
        image_map = self._build_image_index(img_folder_path)

        for item in os.listdir(xml_folder_path):
            if not item.lower().endswith('.xml'):
                continue
            xml_path = os.path.join(os.path.abspath(xml_folder_path), item)
            tree = ET.ElementTree(file=xml_path)
            root = tree.getroot()
            keep_any = False

            img_w = None
            img_h = None
            size_node = root.find('size')
            if size_node is not None:
                w_node = size_node.find('width')
                h_node = size_node.find('height')
                try:
                    img_w = int(w_node.text) if w_node is not None else None
                    img_h = int(h_node.text) if h_node is not None else None
                except Exception:
                    img_w, img_h = None, None

            for obj in list(root.findall('object')):
                name_node = obj.find('name')
                lbl = name_node.text if name_node is not None else ''

                if lbl not in keep_labels:
                    root.remove(obj)
                    removed_objs += 1
                    continue

                # Optional size filter (PascalVOC bndbox -> area ratio)
                if min_area_ratio is not None:
                    bndbox = obj.find('bndbox')
                    if bndbox is not None and img_w and img_h:
                        try:
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            bw = max(0.0, xmax - xmin)
                            bh = max(0.0, ymax - ymin)
                            area_ratio = (bw * bh) / float(img_w * img_h) if img_w > 0 and img_h > 0 else 1.0
                        except Exception:
                            area_ratio = 1.0
                    else:
                        # If missing size info, keep the object (safer).
                        area_ratio = 1.0

                    if area_ratio < min_area_ratio:
                        root.remove(obj)
                        removed_objs += 1
                        continue

                keep_any = True

            if keep_any:
                tree.write(xml_path)
            else:
                os.remove(xml_path)
                removed_labels += 1
                base = os.path.splitext(item)[0]
                for img_name in image_map.get(base, []):
                    img_path = os.path.join(os.path.abspath(img_folder_path), img_name)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        removed_images += 1

        return removed_objs, removed_labels, removed_images

    def _prune_yolo_labels(self, txt_folder_path, img_folder_path, keep_labels, min_area_ratio=None, min_score=None):
        classes_path = os.path.join(txt_folder_path, 'classes.txt')
        if not os.path.isfile(classes_path):
            raise RuntimeError('classes.txt not found for yolo pruning.')

        with open(classes_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]

        keep_old_ids = [i for i, c in enumerate(classes) if c in keep_labels]
        old2new = {old_id: new_id for new_id, old_id in enumerate(keep_old_ids)}
        new_classes = [classes[i] for i in keep_old_ids]

        removed_lines = 0
        removed_labels = 0
        removed_images = 0
        image_map = self._build_image_index(img_folder_path)

        def polygon_area_ratio(points):
            # points in normalized coordinates: list[(x,y)]
            n = len(points)
            if n < 3:
                return 0.0
            s = 0.0
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]
                s += x1 * y2 - x2 * y1
            return abs(s) / 2.0

        for item in os.listdir(txt_folder_path):
            lower_item = item.lower()
            if lower_item == 'classes.txt' or not lower_item.endswith('.txt'):
                continue

            txt_path = os.path.join(os.path.abspath(txt_folder_path), item)
            kept_lines = []
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    vals = line.strip().split()
                    if not vals:
                        continue
                    try:
                        old_id = int(vals[0])
                    except Exception:
                        continue

                    if old_id not in old2new:
                        removed_lines += 1
                        continue

                    # Parse optional bbox/seg size + optional score.
                    coords = vals[1:]
                    score_present = False
                    score = None
                    area_ratio = None

                    # YOLO bbox:
                    #   class cx cy w h
                    #   class cx cy w h score
                    if len(coords) == 4:
                        try:
                            w = float(coords[2])
                            h = float(coords[3])
                            area_ratio = max(0.0, w) * max(0.0, h)
                        except Exception:
                            area_ratio = None
                    elif len(coords) == 5:
                        try:
                            w = float(coords[2])
                            h = float(coords[3])
                            area_ratio = max(0.0, w) * max(0.0, h)
                            score_present = True
                            score = float(coords[4])
                        except Exception:
                            area_ratio = None
                            score_present = False

                    else:
                        # YOLO segmentation polygon:
                        #   class x1 y1 x2 y2 ...        (no score)
                        #   class x1 y1 x2 y2 ... score (score at last token)
                        # Remaining tokens after class:
                        #   even -> no score, pairs are coords
                        #   odd  -> last token is score
                        try:
                            if len(coords) % 2 == 0:
                                pts_flat = coords
                                score_present = False
                                score = None
                            else:
                                pts_flat = coords[:-1]
                                score_present = True
                                score = float(coords[-1])

                            if len(pts_flat) >= 6 and len(pts_flat) % 2 == 0:
                                points = []
                                for i in range(0, len(pts_flat), 2):
                                    x = float(pts_flat[i])
                                    y = float(pts_flat[i + 1])
                                    points.append((x, y))
                                area_ratio = polygon_area_ratio(points)
                        except Exception:
                            area_ratio = None

                    # Apply score filter if enabled and score exists in this line.
                    if min_score is not None and score_present and score is not None:
                        if score < min_score:
                            removed_lines += 1
                            continue

                    # Apply area filter if enabled.
                    if min_area_ratio is not None:
                        if area_ratio is None:
                            # If we can't compute, keep the line (safer).
                            pass
                        else:
                            if area_ratio < min_area_ratio:
                                removed_lines += 1
                                continue

                    vals[0] = str(old2new[old_id])
                    kept_lines.append(' '.join(vals))

            if kept_lines:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(kept_lines) + '\n')
            else:
                os.remove(txt_path)
                removed_labels += 1
                base = os.path.splitext(item)[0]
                for img_name in image_map.get(base, []):
                    img_path = os.path.join(os.path.abspath(img_folder_path), img_name)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        removed_images += 1

        with open(classes_path, 'w', encoding='utf-8') as f:
            if new_classes:
                f.write('\n'.join(new_classes) + '\n')
            else:
                f.write('')

        return removed_lines, removed_labels, removed_images

    def question_2(self,ls):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'these {0} labels will remain ['.format(len(ls))
        for i in range(len(ls)):
            msg=msg+str(ls[i])+'  '
        msg=msg[0:-2]+'], others will be deleted, sure to continue?(personly advise you back up xml files)'
        return yes == QMessageBox.warning(self, u'Attention:', msg, yes | no)
    
    def remove_extra_img_xml(self):
        """File pruning: keep only images that have corresponding labels.

        Supports label folder mixed types:
        - PascalVOC XML (*.xml)
        - YOLO TXT (*.txt + classes.txt)

        Also handles duplicated images with same basename but different formats by renaming
        images and syncing label filenames to keep 1:1 basename mapping.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:
            img_folder_path = os.path.abspath(os.path.dirname(self.filePath))
            label_folder_path = os.path.abspath(self.defaultSaveDir)

            # Sync renames first (so the following pruning sees consistent basenames).
            renamed_dups, copied_labels = self._dedupe_same_basename_images_and_sync_labels(
                img_folder_path, label_folder_path
            )

            has_xml = any(name.lower().endswith('.xml') for name in os.listdir(label_folder_path))
            has_yolo = any(
                name.lower().endswith('.txt') and name.lower() != 'classes.txt'
                for name in os.listdir(label_folder_path)
            )

            if not has_xml and not has_yolo:
                QMessageBox.information(self, u'Wrong!', u'No xml or yolo txt labels found in label folder.')
                return

            image_map = self._build_image_index(img_folder_path)
            image_bases = set(image_map.keys())

            # Delete orphan label files (label basenames not in images).
            if has_xml:
                for item in os.listdir(label_folder_path):
                    if not item.lower().endswith('.xml'):
                        continue
                    base = os.path.splitext(item)[0]
                    if base not in image_bases:
                        os.remove(os.path.join(label_folder_path, item))

            if has_yolo:
                for item in os.listdir(label_folder_path):
                    lower = item.lower()
                    if lower == 'classes.txt' or not lower.endswith('.txt'):
                        continue
                    base = os.path.splitext(item)[0]
                    if base not in image_bases:
                        os.remove(os.path.join(label_folder_path, item))

            # Delete orphan images (images without any corresponding label type).
            remaining_supported_images = 0
            remaining_xmls = 0
            remaining_yolo_txts = 0

            for item in os.listdir(img_folder_path):
                ext = os.path.splitext(item)[1].lower()
                if ext not in self._supported_image_exts():
                    continue
                base = os.path.splitext(item)[0]

                has_any_label = False
                if has_xml and os.path.isfile(os.path.join(label_folder_path, base + '.xml')):
                    has_any_label = True
                if has_yolo and os.path.isfile(os.path.join(label_folder_path, base + '.txt')):
                    has_any_label = True

                if not has_any_label:
                    os.remove(os.path.join(img_folder_path, item))
                else:
                    remaining_supported_images += 1

            if has_xml:
                remaining_xmls = sum(
                    1 for n in os.listdir(label_folder_path) if n.lower().endswith('.xml')
                )
            if has_yolo:
                remaining_yolo_txts = sum(
                    1 for n in os.listdir(label_folder_path) if n.lower().endswith('.txt') and n.lower() != 'classes.txt'
                )

            msg = u'file pruning done.'
            if renamed_dups > 0:
                msg += u' Renamed duplicated images: {}, copied labels: {}.'.format(renamed_dups, copied_labels)
            msg += u' Kept images: {}.'.format(remaining_supported_images)
            if has_xml:
                msg += u' Kept xmls: {}.'.format(remaining_xmls)
            if has_yolo:
                msg += u' Kept yolo txts: {}.'.format(remaining_yolo_txts)

            QMessageBox.information(self, u'Info!', msg)
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
        
    def show_folder_infor(self):
        """show current img folder path and xml folder path and img's number and xml's number.
        usually img's amount should equal to xml's amount.
        """
        try:
            imglist = os.listdir(os.path.dirname(self.filePath))
            xmllist = os.listdir(self.defaultSaveDir)
            QMessageBox.information(self,u'Haa',u'img path: {0}\nimg nums: {1} imgs\nxml path: {2}\nxml nums: {3} xmls'.format(os.path.dirname(self.filePath),len(imglist),self.defaultSaveDir,len(xmllist)))
           
        except:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            
    def show_label_info(self):
        """show all label's name and it's box amount and img amount. 
        """
        def file_name(file_dir):
            L = []
            for root, dirs, files in os.walk(file_dir):
                for file in files:
                    if os.path.splitext(file)[1] == '.xml':
                        L.append(os.path.join(root, file))
            return L
        try:
            if self.filePath == None:
                QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
                return
            xml_dirs = file_name(self.defaultSaveDir)
            total_Box = 0;total_Pic = 0
            Class = []; box_num=[]; pic_num=[]; flag=[]

            for i in range(0, len(xml_dirs)):
                total_Pic+=1
                annotation_file = open(xml_dirs[i]).read()
                root = ET.fromstring(annotation_file)
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    if label not in Class:
                        Class.append(label)
                        box_num.append(0)
                        pic_num.append(0)
                        flag.append(0)
                    for i in range(len(Class)):
                        if label == Class[i]:
                            box_num[i] += 1
                            flag[i] = 1
                            total_Box += 1
                for i in range(len(Class)):
                    if flag[i] == 1:
                        pic_num[i] += 1
                        flag[i] = 0
            result={}
            for i in range(len(Class)):
                result[Class[i]]= (pic_num[i], box_num[i])
            result['total']=(total_Pic, total_Box)
            info='label | pic_num | box_num \n'
            info += '----------------------------\n'
            for key in result.keys():
                info+='{}:  {}\n'.format(key,result[key])
            QMessageBox.information(self,u'Info',info)
            
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
    
    def extract_video(self):
        """extract imgs from video.
        'frame gap' means save img by this frequency(not save every img in video if frame_gap larger than 1).
        img will saved in the same path with video.
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            video_path,_ = QFileDialog.getOpenFileName(self,'choose video file:')
            if not video_path:
                return
            save_path_root = QFileDialog.getExistingDirectory(self,'choose a directory to save images extracted:', video_path)
            # print(save_path_root)
            # print(os.path.splitext(os.path.basename(video_path))[0])
            if not save_path_root:
                return
            # save_path
            save_path=os.path.join(save_path_root, 'images')
            # print(save_path)
            os.makedirs(save_path,exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            frame_num =int(cap.get(7))
            # print(frame_num)
            frame_gap,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input frame gap, img will extract by this frequency",value=1)
            if not ok:
                return
            progress = QProgressDialog(self)
            progress.setWindowTitle("请稍等")  
            progress.setLabelText("正在提取图像...")
            progress.setCancelButtonText("取消")
            # progress.setMinimumDuration(5)
            # progress.setWindowModality(Qt.WindowModal)
            progress.setRange(0,100)
            cap = cv2.VideoCapture(video_path)
            frame_num =int(cap.get(7))
            # print(frame_num)
            # frame_gap,ok=QInpu
            while cap.isOpened():
                
                ret, frame = cap.read()
                if ret:
                    index=int(cap.get(1))
                    if index%frame_gap!=0:
                        continue
                    cv2.imwrite(os.path.join(save_path,os.path.splitext(os.path.basename(video_path))[0]+'_%06d'%(int(cap.get(1)))+'.jpg'),frame)
                    # print(int(index/frame_num*100))
                    progress.setValue(int(index/frame_num*100))
                    QCoreApplication.processEvents()
                    if progress.wasCanceled():
                        break
                else:
                    break
            cap.release()
            progress.setValue(100)
            QMessageBox.information(self,u'Done!',u'video extract done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))

    def extract_videos(self):
        """extract imgs from videos.
        'frame gap' means save img by this frequency(not save every img in video if frame_gap larger than 1).
        img will saved in the same path with video.
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            video_root = QFileDialog.getExistingDirectory(self,'choose a directory of video:', r'./')
            if not video_root:
                return
            save_path_root = QFileDialog.getExistingDirectory(self,'choose a directory to save images extracted:', video_root)
            # print(save_path_root)
            # print(os.path.splitext(os.path.basename(video_path))[0])
            if not save_path_root:
                return
            frame_gap,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input frame gap, img will extract by this frequency",value=10)
            # save_path 
            if not ok:
                return
            progress = QProgressDialog(self)
            progress.setWindowTitle("请稍等")  
            progress.setLabelText("正在提取图像...")
            progress.setCancelButtonText("取消")
            # progress.setMinimumDuration(5)
            # progress.setWindowModality(Qt.WindowModal)
            progress.setRange(0,len(os.listdir(video_root)))
            for i, video in enumerate(os.listdir(video_root)):
                if ".mp4" in video or '.mpg' in video or '.avi' in video:
                    nm = video.split("_")[0]
                    save_path=os.path.join(save_path_root, nm, "images")
                    video_path = os.path.join(video_root, video)
                    # print(save_path)
                    os.makedirs(save_path,exist_ok=True)
                    cap = cv2.VideoCapture(video_path)
                    frame_num =int(cap.get(7))
                    # print(frame_num)
                    cap = cv2.VideoCapture(video_path)
                    # frame_num =int(cap.get(7))
                    progress.setValue(i+1)
                    if progress.wasCanceled():
                        break
                    while cap.isOpened():
                        
                        ret, frame = cap.read()
                        if ret:
                            index=int(cap.get(1))
                            if index%frame_gap!=0:
                                continue
                            save_name = os.path.splitext(video)[0]
                            save_name = remove_novel(save_name)+'_%06d'%(int(cap.get(1)))+'.jpg'
                            cv2.imencode('.jpg', frame)[1].tofile(os.path.join(save_path,save_name))
                            # cv2.imwrite(os.path.join(save_path,os.path.splitext(video)[0]+str(int(cap.get(1)))+'.jpg'),frame)
                            QCoreApplication.processEvents()

                        else:
                            break
                    cap.release()
            # progress.setValue(100)
            QMessageBox.information(self,u'Done!',u'video extract done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))

    def extract_stream(self):
        """extract imgs from stream, 'stream_path' usually start with rtsp or rtmp.
        'frame gap' means save img by this frequency(not save every img in video if frame_gap larger than 1).
        'max save number' means actions will stop after save this amount imgs.
        this action will stop after read stream path failed 3 times.
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            stream_path,ok=QInputDialog.getText(self, 'Text Input Dialog', 
                        "Input steam path(start with rtmp、rtsp...):")
            if not(stream_path and ok):
                return
            save_path = QFileDialog.getExistingDirectory()
            if not save_path:
                return
            frame_gap,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input frame gap, img will extract by this frequency",value=1)
            if not ok:
                return
            max_frame,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input max save number, process will end after save cetain number imgs",value=10)
            if not ok:
                return
            cap = cv2.VideoCapture(stream_path)
            drop_times=0
            while True:
                ret, frame = cap.read()
                if ret:
                    index=int(cap.get(1))
                    if index%frame_gap!=0:
                        continue
                    if index>(max_frame*frame_gap):
                        break
                    cv2.imwrite(save_path+'/'+str(int(cap.get(1)))+'.jpg',frame)
                else:
                    cap.release()
                    drop_times+=1
                    if drop_times>=3:
                        QMessageBox.information(self,u'Wrong!',u'stream path not useable.')
                        break
                    cap = cv2.VideoCapture(stream_path)
            cap.release()
            QMessageBox.information(self,u'Done!',u'stream extract done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
           
    def batch_resize_img(self):
        """input Wdith and Height to resize all img to one shape.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return    
        try:
            img_path = os.path.dirname(self.filePath)
            filelist = natsort.natsorted(os.listdir(img_path))
            new_W,ok=QInputDialog.getInt(self,'Integer input dialog','input img wdith :',value=1920)
            if not ok:
                return
            new_H,ok=QInputDialog.getInt(self,'Integer input dialog','input img height :',value=1080)
            if not ok:
                return
            for item in filelist:
                img=cv2.imread(os.path.join(img_path,item))
                img=cv2.resize(img,(new_W,new_H))
                cv2.imwrite(os.path.join(img_path,item),img)
                
            QMessageBox.information(self,u'Done!',u'batch resize done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
        
    def merge_video(self):
        """merge all img in one path to one video, video will saved in img's parent path.
        for some restraint, fps must be 25, you can use 'repeat times' to repeat play img if you want slower the video.
        this action may take some time, please don't click mouse too frequently. 
        you can press 'space' if you find bounding box not accurate during auto annotate.
        """
        try:
            img_path = QFileDialog.getExistingDirectory(self,'choose imgs folder:')
            if not img_path:
                return
            filelist = natsort.natsorted(os.listdir(img_path)) #获取该目录下的所有文件名
            img=cv2.imread(img_path+'/'+filelist[0])
            img_size=img.shape
            fps = 25
            repeat_time,ok = QInputDialog.getInt(self, 'Int Input Dialog',
                        "Input each img's repeat times(the bigger, the slower), usually set 1",value=1)
            if not ok:
                return
            file_path = img_path +'_result' + ".avi" #导出路径
            fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
            video = cv2.VideoWriter( file_path, fourcc, fps ,(img_size[1],img_size[0]))
            for item in filelist:
                if item.endswith('.jpg'):   #判断图片后缀是否是.png
                    item = img_path +'/'+item 
                    img = cv2.imread(item)
                    for j in range(repeat_time):
                        video.write(img)        

            video.release()
            QMessageBox.information(self,u'Done!',u'video merge done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))

    def annotation_video(self):
        """ auto annotation video file or local camera.
        select video file, cancle to use local camera.
        img and xml will saved on dir of video path unless you use local camera, and folder will be './' in which case.
        'CSRT' type means more accuracy and low speed(recommend), 'MOSSE' means high speed and low accuracy, 'KCF' is in middle.
        frames are resized for display reason, one better run 'fix_property' after this process.
        press 'space' to re-drawing bounding box during annotation if you find bounding box not accurate.
        """
        try:
            tree = ET.ElementTree(file='./datasets/origin.xml')
            root=tree.getroot()
            for child in root.findall('object'):
                template_obj=child#保存一个物体的样板
                root.remove(child)
            tree.write('./datasets/template.xml')
            trackerType_selector={'CSRT':cv2.TrackerCSRT_create,
                                  'BOOSTING':cv2.TrackerBoosting_create,
                                  'MIL':cv2.TrackerMIL_create,
                                  'KCF':cv2.TrackerKCF_create,
                                  'TLD':cv2.TrackerTLD_create,
                                  'MEDIANFLOW':cv2.TrackerMedianFlow_create,
                                  'GOTURN':cv2.TrackerGOTURN_create,
                                  'MOSSE':cv2.TrackerMOSSE_create}
            items=tuple(trackerType_selector)
            trackerType , ok = QInputDialog.getItem(self, "Select",
                "Tracker type, usually 'CSRT' is ok:", items, 0, False)
            if not ok:
                return
            videoPath ,_ = QFileDialog.getOpenFileName(self,"choose video file, cancle to use local camera:")
            if not videoPath:
                videoPath = 0
            save_gap,ok=QInputDialog.getInt(self,'Integer input dialog','input save gap, img will saved by this frenquency :',value=25)
            if not ok:
                return
            img_size,ok=QInputDialog.getInt(self,'Integer input dialog','input img size, img resized ti this shape by height:',value=900)
            if not ok:
                return
            process_shape=(int(1.777*img_size),int(img_size))
            cap = cv2.VideoCapture(videoPath)
            ret, frame = cap.read()
            height_K=frame.shape[0]/img_size
            weight_K=frame.shape[1]/(1.777*img_size)
            if not ret:
                print('Failed to read video')
                sys.exit(1)
            else:
                pass
                frame=cv2.resize(frame,process_shape)
            def init_multiTracker(frame):
                bboxes = []
                colors = []
                labels = []
                while True:
                    # 在对象上绘制边界框selectROI的默认行为是从fromCenter设置为false时从中心开始绘制框，可以从左上角开始绘制框
                    bbox = cv2.selectROI("draw box and press 'SPACE' to affirm, Press 'q' to quit draw box and start tracking-labeling", frame)
                    if min(bbox[2],bbox[3]) >= 10:
                        label_name,ok=QInputDialog.getText(self, 'Text Input Dialog', 
                        "Input label name:")
                        if not(label_name and ok):
                            return
                        labels.append(label_name)
                        bboxes.append(bbox)
                        colors.append((random.randint(30, 240), random.randint(30, 240), random.randint(30, 240)))
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv2.rectangle(frame, p1, p2, [10,250,10], 2, 1)
                    else:
                        print("bbox size small than 10, will be abandoned")
                    k=cv2.waitKey(0)
                    # print(k)
                    if k==113:
                        break
                print('Selected bounding boxes: {}'.format(bboxes))
                multiTracker = cv2.MultiTracker_create()
                # 初始化多跟踪器
                for bbox in bboxes:
                    tracker=trackerType_selector[trackerType]()
                    multiTracker.add(tracker, frame, bbox)    
                return multiTracker,colors,labels
            multiTracker,colors,labels=init_multiTracker(frame)
            cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("MultiTracker", process_shape[0], process_shape[1])
            cv2.moveWindow("MultiTracker", 10, 10)
            # 处理视频并跟踪对象
            index=0
            while cap.isOpened():
                ret, origin_frame = cap.read()
                if not ret:
                    break
                frame=cv2.resize(origin_frame,process_shape)
                draw=frame.copy()
                ret, boxes = multiTracker.update(frame)
                # 绘制跟踪的对象
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(draw, p1, p2, colors[i], 2, 1)
                    info = labels[i]
                    t_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.7 , 1)[0]
                    cv2.rectangle(draw, p1, (int(newbox[0]) + t_size[0]+3, int(newbox[1]) + t_size[1]+6), colors[i], -1)
                    cv2.putText(draw, info, (int(newbox[0])+1, int(newbox[1])+t_size[1]+2), cv2.FONT_HERSHEY_TRIPLEX, 0.7, [255,255,255], 1)
                # show frame
                cv2.imshow("MultiTracker, press 'SPACE' to redraw box, press 'q' to quit video labeling", draw)
                # quit on ESC or Q button
                if index%save_gap==0:
                    tree = ET.ElementTree(file='./datasets/template.xml')
                    root=tree.getroot()
                    for i, newbox in enumerate(boxes):
                        temp_obj = template_obj
                        temp_obj.find('name').text=str(labels[i])
                        temp_obj.find('bndbox').find('xmin').text=str(int(weight_K*newbox[0]))
                        temp_obj.find('bndbox').find('ymin').text=str(int(height_K*newbox[1]))
                        temp_obj.find('bndbox').find('xmax').text=str(int(weight_K*newbox[0]+weight_K*newbox[2]))
                        temp_obj.find('bndbox').find('ymax').text=str(int(height_K*newbox[1]+height_K*newbox[3]))
                        root.append(deepcopy(temp_obj))       #深度复制
                    if videoPath==0:
                        parent_path='./temp'
                    else:
                        parent_path=os.path.dirname(videoPath)
                    os.makedirs(os.path.join(parent_path,'JPEGImages'), exist_ok=True)
                    os.makedirs(os.path.join(parent_path,'Annotations'), exist_ok=True)
                    cv2.imwrite(os.path.join(parent_path,'JPEGImages/','{}.jpg'.format(index)),origin_frame)
                    tree.write(os.path.join(parent_path,'Annotations/','{}.xml'.format(index)))
                index+=1
                k=cv2.waitKey(1)
                if k==32: #press space to reinit box
                    cv2.destroyAllWindows()
                    multiTracker,colors,labels=init_multiTracker(frame)
                    cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("MultiTracker", process_shape[0], process_shape[1])
                    cv2.moveWindow("MultiTracker", 10, 10)
                if k== 27 or k == 113: #press q or esc to quit
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            cap.release()
            cv2.destroyAllWindows()
            QMessageBox.information(self,u'Done!',u'video auto annotation done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
    
    def load_label_names(self):
        filename, _ = QFileDialog.getOpenFileName(self,"Choosing class name list file:", r'datasets', "TXT file(*.txt)")
        if not filename:
            return
        print(filename)
        self.loadPredefinedClasses(filename)
        # f = open(filename, 'r') 
        # classes = f.readlines()                 
        # f.close()
        # print('load class names: ', len(classes)) 
    
    def update_default_class(self):
        self.default_class = self.defaultLabelCombobox.currentText()
        print('load class names: ', len(self.predefined_classes))
    
    
    def addLabelname(self):
        new_labelname, ok = QInputDialog.getText(self, "New labelname",                                                
        "add a new labelname to list:")
        if ok:
            self.predefined_classes.append(new_labelname)
            self.defaultLabelCombobox.addItem(new_labelname)
    
    def change_label_name(self):
        """Batch rename one class name in current annotation directory (XML and/or YOLO)."""
        if self.filePath is None:
            QMessageBox.information(self, u'Wrong!', u'have no loaded folder yet, please check again.')
            return
        if not self.defaultSaveDir or (not os.path.isdir(self.defaultSaveDir)):
            QMessageBox.information(self, u'Wrong!', u'label folder is invalid, please check save dir first.')
            return

        try:
            origin, ok = QInputDialog.getText(self, 'Text Input Dialog', "Input origin label name(single)：")
            if not ok:
                return
            origin = origin.strip()
            if not origin:
                QMessageBox.information(self, u'Wrong!', u'origin label name cannot be empty.')
                return

            target, ok = QInputDialog.getText(self, 'Text Input Dialog', "Input target label name(single)：")
            if not ok:
                return
            target = target.strip()
            if not target:
                QMessageBox.information(self, u'Wrong!', u'target label name cannot be empty.')
                return
            if origin == target:
                QMessageBox.information(self, u'Hint', u'origin and target are same, no change needed.')
                return

            label_dir = self.defaultSaveDir
            xml_files = [x for x in natsort.natsorted(os.listdir(label_dir)) if x.lower().endswith('.xml')]
            txt_files = [x for x in natsort.natsorted(os.listdir(label_dir))
                         if x.lower().endswith('.txt') and x.lower() != 'classes.txt']
            classes_path = os.path.join(label_dir, 'classes.txt')

            # ---- pre-scan counts ----
            xml_hit_files = 0
            xml_hit_boxes = 0
            for item in xml_files:
                xml_path = os.path.join(os.path.abspath(label_dir), item)
                tree = ET.ElementTree(file=xml_path)
                root = tree.getroot()
                hit = 0
                for obj in root.findall('object'):
                    name_node = obj.find('name')
                    if name_node is not None and (name_node.text or '').strip() == origin:
                        hit += 1
                if hit > 0:
                    xml_hit_files += 1
                    xml_hit_boxes += hit

            txt_hit_files = 0
            txt_hit_boxes = 0
            yolo_old_idx = None
            yolo_new_idx = None
            classes = []
            if os.path.isfile(classes_path):
                with open(classes_path, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f.read().splitlines()]
                if origin in classes:
                    yolo_old_idx = classes.index(origin)
                if target in classes:
                    yolo_new_idx = classes.index(target)
                else:
                    classes.append(target)
                    yolo_new_idx = len(classes) - 1

                if yolo_old_idx is not None:
                    for item in txt_files:
                        txt_path = os.path.join(os.path.abspath(label_dir), item)
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            lines = f.read().splitlines()
                        hit = 0
                        for line in lines:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            try:
                                cid = int(parts[0])
                            except Exception:
                                continue
                            if cid == yolo_old_idx:
                                hit += 1
                        if hit > 0:
                            txt_hit_files += 1
                            txt_hit_boxes += hit

            msg = (
                "Ready to rename label:\n"
                "  %s -> %s\n\n"
                "Planned changes:\n"
                "  XML files: %d (objects: %d)\n"
                "  YOLO txt files: %d (objects: %d)\n\n"
                "Continue?"
            ) % (origin, target, xml_hit_files, xml_hit_boxes, txt_hit_files, txt_hit_boxes)
            if QMessageBox.Yes != QMessageBox.question(self, u'Confirm', msg, QMessageBox.Yes | QMessageBox.No):
                return

            # ---- apply XML rename ----
            xml_changed_files = 0
            xml_changed_boxes = 0
            for item in xml_files:
                xml_path = os.path.join(os.path.abspath(label_dir), item)
                tree = ET.ElementTree(file=xml_path)
                root = tree.getroot()
                changed = 0
                for obj in root.findall('object'):
                    name_node = obj.find('name')
                    if name_node is not None and (name_node.text or '').strip() == origin:
                        name_node.text = target
                        changed += 1
                if changed > 0:
                    tree.write(xml_path)
                    xml_changed_files += 1
                    xml_changed_boxes += changed
                    print('modify label: %s' % item)

            # ---- apply YOLO rename by class id remap ----
            txt_changed_files = 0
            txt_changed_boxes = 0
            if os.path.isfile(classes_path) and (yolo_old_idx is not None) and (yolo_new_idx is not None):
                for item in txt_files:
                    txt_path = os.path.join(os.path.abspath(label_dir), item)
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        lines = f.read().splitlines()

                    new_lines = []
                    changed = 0
                    for line in lines:
                        raw = line.strip()
                        if not raw:
                            new_lines.append(line)
                            continue
                        parts = raw.split()
                        try:
                            cid = int(parts[0])
                        except Exception:
                            new_lines.append(line)
                            continue
                        if cid == yolo_old_idx:
                            parts[0] = str(yolo_new_idx)
                            changed += 1
                            new_lines.append(" ".join(parts))
                        else:
                            new_lines.append(line)

                    if changed > 0:
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
                        txt_changed_files += 1
                        txt_changed_boxes += changed

                if origin in classes:
                    old_idx = classes.index(origin)
                    classes[old_idx] = ''
                while classes and classes[-1] == '':
                    classes.pop()
                with open(classes_path, 'w', encoding='utf-8') as f:
                    for c in classes:
                        f.write(c + '\n')

                # Sync in-memory class lists for UI.
                self.yolo_classes = list(classes)
                self.predefined_classes = [c for c in classes if c]
                self.defaultLabelCombobox.clear()
                self.defaultLabelCombobox.addItems(self.predefined_classes)
                self._after_predefined_class_names_changed()

            QMessageBox.information(
                self,
                u'Done!',
                u'label rename done.\nXML: %d files / %d objects\nYOLO: %d files / %d objects'
                % (xml_changed_files, xml_changed_boxes, txt_changed_files, txt_changed_boxes)
            )
        except Exception as e:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(print_error()))
        
    def fix_xml_property(self):
        """fix xml's property such as size,folder,filename,path.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return 
        try:
            xml_folder_path=self.defaultSaveDir
            img_folder_path=os.path.dirname(self.filePath)
            xmllist = os.listdir(xml_folder_path)
            folder_info={'folder':'JPEGImages','filename':'none','path':'none'}
            for item in xmllist:
                if item.endswith('.xml'):
                    folder_info['filename']=item[0:-4]+'.jpg'
                    folder_info['path']=os.path.join(img_folder_path, item[0:-4])+'.jpg'
                    img = cv2.imread(folder_info['path'])
                    size=img.shape
                    xmlPath=os.path.join(os.path.abspath(xml_folder_path), item)
                    tree = ET.ElementTree(file=xmlPath)
                    root=tree.getroot()
                    try:
                        root.find('size').find('width').text=str(size[1])
                        root.find('size').find('height').text=str(size[0])
                        root.find('size').find('depth').text=str(size[2])
                    except:
                        print('xml has no size attribute!')
                    for key in folder_info.keys():
                        try:
                            root.find(key).text=folder_info[key]
                        except:
                            print(item,': attribute',key,'not exist!')
                            pass
                    tree.write(xmlPath)
            QMessageBox.information(self,u"Done!",u"fix xml's property done!")
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))

    def auto_labeling_s(self):
        try:
            self.xml_folder_path=self.defaultSaveDir
            tree = ET.ElementTree(file='./datasets/origin.xml')
            root=tree.getroot()
            for child in root.findall('object'):
                template_obj=child#保存一个物体的样板
                root.remove(child)

            for child_size in root.findall('size'):
                template_size = child_size
                root.remove(child_size)
                
            tree.write('./datasets/template.xml')
                
            #=====def some function=====
            def change_size_property(shape_img,template_obj):
                temp_size=template_size
                for child in temp_size:
                    key=child.tag
                    if key == 'width':
                        child.text=str(shape_img[1])
                    if key == 'height':
                        child.text=str(shape_img[0])
                    if key == 'depth':
                        child.text=str(shape_img[2])                    
                
                return temp_size
            
            def change_obj_property(detect_result,template_obj):
                temp_obj=template_obj
                for child in temp_obj:
                    key=child.tag
                    if key in detect_result.keys():
                        child.text=detect_result[key]
                    if key=='bndbox':
                        for gchild in child:
                            gkey=gchild.tag
                            gchild.text=str(detect_result[gkey])
                return temp_obj
                
            def change_result_type(boxes,scores,labels, needed_labels, conf_thres):
                result=[]
                for box, score, label in zip(boxes, scores, labels):
                    if score>conf_thres and label in needed_labels:
                        # print(label)
                        try:
                            new_obj={}
                            new_obj['name']=self.default_label
                            new_obj['xmin']=int(box[0])
                            new_obj['ymin']=int(box[1])
                            new_obj['xmax']=int(box[2])
                            new_obj['ymax']=int(box[3])
                            result.append(new_obj)
                        except:
                            print('labels_info have no label: '+str(label))
                            pass
                return result

            source=os.path.dirname(self.filePath)
            check_chinese(source)
            xml_path=self.defaultSaveDir
            
            weights = self._select_model_weight(
                "Model weight for auto-labeling single class (default %s):" % DEFAULT_DETECT_PT,
                prefer_model=DEFAULT_DETECT_PT,
                for_autolabel=True)
            if not weights:
                return
            # conf_thres = 0.25
            conf_thres, ok = QInputDialog.getDouble(self,'threshold dialog','input confidence threshold:', value=0.2, min=0.05, max=1.0)
            if not ok:
                return
            iou_thres=0.8
            # Select GPUs/CPU
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    gpuid, _, _ = get_cuda_setup(24,24)
                    # print(torch.cuda.get_device_capability('cuda:2'))
                    device = torch.device("cuda:%d"%gpuid)
                    print(torch.cuda.device_count())
                else:
                    device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")         
            # half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model and label name.
            w = resolve_yolo_checkpoint(weights, os.path.join(APP_ROOT, 'weights'))
            model = YOLO(w)  # load FP32 model
            names = model.model.names
            # names = model.module.names if hasattr(model, 'module') else model.names
            # print('ok 1')
            if len(names)== 1:
                needed_labels=list(names.values())
            else:
                needed_labels=easygui.multchoicebox(msg="select labels you want auto-labeling?",title="Select labels", choices=list(names.values()))
            new_label, ok = QInputDialog.getText(self,'Label dialog','input new label name for assignment:', text = self.default_label)
            print("needed names:", needed_labels)
            if ok:
                self.default_label = new_label
            print("new_label:", self.default_label)
            dataset = LoadImages(source)
            progress = QProgressDialog(self)
            progress.setWindowTitle(u"Waiting")  
            progress.setLabelText(u"auto-labeling now,Please wait...")
            progress.setCancelButtonText(u"Cancle it")
            progress.setMinimumDuration(1)
            progress.setWindowModality(Qt.WindowModal)
            progress.setRange(0,100)   
            # print('ok 4')
            index = -1
            # print('ok 5')
            for path, img, _, _ in dataset:
                index += 1
                # print(path[0])
                progress.setValue(int(100*index/len(dataset)))
                if progress.wasCanceled():
                    QMessageBox.warning(self,"Attention","auto-labeling canceled！") 
                    return
                try:
                    result = model.predict(source=img, conf=conf_thres, iou=iou_thres)[0]
                    # names = model.model.names
                    t2 = time_sync()
                    dets = result.boxes

                    labels = [names[int(x)] for x in dets.cls]
                    scores = [float(x) for x in dets.conf]
                    boxes = [x.tolist() for x in dets.xyxy]
                    tree = ET.ElementTree(file='./datasets/template.xml')
                    root = tree.getroot()
                    path_img = path[0]
                    common_property={'filename':path[0].split('\\')[-1],'path':source,'folder':'images'}
                    # print('ok 7')
                    for child in root:
                        key=child.tag
                        if key in common_property.keys():
                            child.text=common_property[key]
                    # print('ok 8')
                    new_size = change_size_property(img[0].shape,template_obj)
                    root.append(new_size)
    
                    result = change_result_type(boxes,scores,labels,needed_labels,conf_thres)
                    # print('ok 9')
                    if len(result)>0:
                        for j in range(len(result)):
                            new_obj = change_obj_property(result[j],template_obj)
                            root.append(deepcopy(new_obj))       #深度复制
                            #!!!这块没直接append(new_obj)是因为当增加多个节点的话，new_obj会进行覆盖，必须要用深度复制以进行区分
                    # print('ok 10')
                    # print(self.filePath)
                    if not self.xml_folder_path:
                        self.xml_folder_path = os.path.join(os.path.dirname(os.path.dirname(self.filePath)), 'xml')
                    # print(xml_path)
                    if not os.path.exists(self.xml_folder_path):
                        os.makedirs(self.xml_folder_path)
                    # print(xml_path)
                    # print(path)
                    path_write = os.path.join(self.xml_folder_path, path_img[len(os.path.dirname(path_img))+1:-4]+'.xml') #path[0:-4] + '.xml'#
                    # print(path_write)
                    tree.write(path_write)
                except Exception as e:
                    print(e)
                    continue
                # print('ok 11')
            progress.setValue(100)
            self.refreshImg()
            QMessageBox.information(self,u'Done!',u'auto labeling done,and reload img folder')  
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))        

    def auto_labeling_m(self):
        try:
            self.xml_folder_path = self.defaultSaveDir
            tree = ET.ElementTree(file='./datasets/origin.xml')
            root = tree.getroot()
            for child in root.findall('object'):
                template_obj = child  # 保存一个物体的样板
                root.remove(child)

            for child_size in root.findall('size'):
                template_size = child_size
                root.remove(child_size)

            tree.write('./datasets/template.xml')

            # =====def some function=====
            def change_size_property(shape_img, template_obj):
                temp_size = template_size
                for child in temp_size:
                    key = child.tag
                    if key == 'width':
                        child.text = str(shape_img[1])
                    if key == 'height':
                        child.text = str(shape_img[0])
                    if key == 'depth':
                        child.text = str(shape_img[2])

                return temp_size

            def change_obj_property(detect_result, template_obj):
                temp_obj = template_obj
                for child in temp_obj:
                    key = child.tag
                    if key in detect_result.keys():
                        child.text = detect_result[key]
                    if key == 'bndbox':
                        for gchild in child:
                            gkey = gchild.tag
                            gchild.text = str(detect_result[gkey])
                return temp_obj

            def change_result_type(boxes,scores,labels, needed_labels):
                result=[]
                i = 0
                for box, score, label in zip(boxes, scores, labels):
                    i+=1
                    overlay = False
                    if label in needed_labels:
                        print(label)
                        for b in boxes[i:]:
                            if compute_IOU(box,b) > 0.75:
                                overlay=True
                        if not overlay:
                            try:
                                new_obj={}
                                new_obj['name']=label
                                new_obj['xmin']=int(box[0])
                                new_obj['ymin']=int(box[1])
                                new_obj['xmax']=int(box[2])
                                new_obj['ymax']=int(box[3])
                                result.append(new_obj)
                            except:
                                print('labels_info have no label: '+str(label))
                                pass
                return result

            source = os.path.dirname(self.filePath)
            xml_path = self.defaultSaveDir

            weights = self._select_model_weight(
                "Model weight for auto-labeling multi class (default %s):" % DEFAULT_DETECT_PT,
                prefer_model=DEFAULT_DETECT_PT,
                for_autolabel=True)
            if not weights:
                return
            # conf_thres = 0.25
            conf_thres, ok = QInputDialog.getDouble(self,'threshold dialog','input confidence threshold:', value=0.2, min=0.05, max=1.0)
            if not ok:
                return
            iou_thres=0.5
            # iou_thres=0.5
            # Select GPUs/CPU
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    gpuid, _, _ = get_cuda_setup(24,24)
                    # print(torch.cuda.get_device_capability('cuda:2'))
                    device = torch.device("cuda:%d"%gpuid)
                    print(torch.cuda.device_count())
                else:
                    device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")  

            # Load model and label name.
            w = resolve_yolo_checkpoint(weights, os.path.join(APP_ROOT, 'weights'))
            model = YOLO(w)  # load FP32 model
            names = model.module.names if hasattr(model, 'module') else model.names
            # print('ok 1')
            if len(names) == 1:
                needed_labels = list(names.values())
            else:
                needed_labels = easygui.multchoicebox(msg="select labels you want auto-labeling?",
                                                      title="Select labels", choices=list(names.values()))
            # new_label, ok = QInputDialog.getText(self, 'Label dialog', 'input new label name for assignment:',
            #                                      text=self.default_label)
            # if ok:
            #     self.default_label = new_label
            if not needed_labels:
                return
            print(needed_labels)
            # if half:
            #model.half()  # to FP16
            check_chinese(source)
            dataset = LoadImages(source)
            progress = QProgressDialog(self)
            progress.setWindowTitle(u"Waiting")
            progress.setLabelText(u"auto-labeling now,Please wait...")
            progress.setCancelButtonText(u"Cancle it")
            progress.setMinimumDuration(1)
            progress.setWindowModality(Qt.WindowModal)
            progress.setRange(0, 100)
            # print('ok 4')
            index = -1
            # print('ok 5')
            for path, img, _, _ in dataset:
                index += 1
                print(path[0])
                progress.setValue(int(100 * index / len(dataset)))
                if progress.wasCanceled():
                    QMessageBox.warning(self, "Attention", "auto-labeling canceled！")
                    return
                if img:
                    result = model.predict(source=img, conf=conf_thres, iou=iou_thres)[0]
                    names = model.model.names
                    t2 = time_sync()
                    dets = result.boxes
                    labels = [names[int(x)] for x in dets.cls]
                    print(labels)
                    scores = [float(x) for x in dets.conf]
                    boxes = [x.tolist() for x in dets.xyxy]
                    
                    tree = ET.ElementTree(file='./datasets/template.xml')
                    root = tree.getroot()
                    path_img=path[0]
                    common_property = {'filename': path_img.split('\\')[-1], 'path': source, 'folder': 'images'}
                    # print('ok 7')
                    for child in root:
                        key = child.tag
                        if key in common_property.keys():
                            child.text = common_property[key]
                    # print('ok 8')
                    new_size = change_size_property(img[0].shape, template_obj)
                    root.append(new_size)
    
                    result = change_result_type(boxes, scores, labels, needed_labels)
                    print(len(result))
                    if len(result) > 0:
                        for j in range(len(result)):
                            new_obj = change_obj_property(result[j], template_obj)
                            root.append(deepcopy(new_obj))  # 深度复制
                            # !!!这块没直接append(new_obj)是因为当增加多个节点的话，new_obj会进行覆盖，必须要用深度复制以进行区分
                    # print('ok 10')
                    # print(self.filePath)
                    if not self.xml_folder_path:
                        self.xml_folder_path = os.path.join(os.path.dirname(os.path.dirname(self.filePath)), 'xmls')
                    # print(xml_path)
                    if not os.path.exists(self.xml_folder_path):
                        os.makedirs(self.xml_folder_path)
                    # print(xml_path)
                    # print(path)
                    path_write = os.path.join(self.xml_folder_path,
                                              path_img[len(os.path.dirname(path_img)) + 1:-4] + '.xml')  # path[0:-4] + '.xml'#
                    # print(path_write)
                    tree.write(path_write)
                # print('ok 11')
            progress.setValue(100)
            self.refreshImg()
            QMessageBox.information(self, u'Done!', u'auto labeling done,and reload img folder')
        except Exception as e:
            QMessageBox.information(self, u'Sorry!', u'something is wrong. ({})'.format(print_error()))


    def auto_labeling_one(self):

        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:       
            #=====choose model and input label name=====
            with torch.no_grad():
                self.auto_labeling_s()
            return
 
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))

    def auto_labeling_multi(self):

        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:       
            #=====choose model and input label name=====
            with torch.no_grad():
                self.auto_labeling_m()
            return
 
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))

    def make_datasets_one(self):
        from libs.voc_2_yolo_single import voc2yolo_one
        from libs.make_train_val_test_datasets import make_train_val
        path_out = r'datasets/one_class'
        try:
            image_dr = QFileDialog.getExistingDirectory(self,"Choosing labeled images folder:",self.img_folder_path)
            if not image_dr:
                return
            label_dr = QFileDialog.getExistingDirectory(self,"Choosing label files folder:",self.xml_folder_path)
            print('image dir:%s'%image_dr)
            print('label dir:%s'%label_dr)
            if not label_dr:
                return
            data_dr = QFileDialog.getExistingDirectory(self,"Choosing datasets folder:", 'datasets')
            if data_dr:
                path_out = data_dr
            
            label_format_list = ['VOC(xml)', 'YOLO(txt)']
            items = tuple(label_format_list)

            label_format, ok = QInputDialog.getItem(self, "Select",
            "label file's format(VOC or YOLO)':", items, 0, False)
            if not ok:
                return
            if 'VOC' in label_format:
                txt_dr = os.path.join(os.path.dirname(label_dr), 'txts')
                # print(txt_dr)
                # filename = QFileDialog.getOpenFileName(self,"Choosing class name list file:", r'datasets')
                # if not filename:
                #     return
                
                # f = open(filename, 'r') 
                # classes = f.readlines()                 
                # f.close()
                # print('load class names: ', len(classes)) 
                voc2yolo_one(txt_dr, label_dr)
                label_dr = txt_dr
            print('VOC label files converted to yolo format')
            make_train_val(image_dr, label_dr, path_out, 0.7)
            yaml_path = self._write_dataset_yaml(path_out, label_dr, yaml_name='data_one_class.yaml')
            tr_i, va_i, tr_l, va_l, out_abs = self._count_make_dataset_outputs(path_out)
            tot_i, tot_l = tr_i + va_i, tr_l + va_l
            msg = (
                u'训练/验证集已生成（单类流程）。\n\n'
                u'训练集：图片 %d 张，YOLO 标注 %d 个\n'
                u'验证集：图片 %d 张，YOLO 标注 %d 个\n'
                u'合计：图片 %d 张，标注 %d 个\n\n'
                u'已写入 classes.txt：输出根目录、labels/train、labels/val（与 data.yaml 中 names 一致）。\n\n'
                u'输出目录：\n%s\n\n'
                u'YAML：\n%s'
            ) % (tr_i, tr_l, va_i, va_l, tot_i, tot_l, out_abs, os.path.abspath(yaml_path))
            QMessageBox.information(self, u'Done!', msg)
                         
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))

    def _infer_dataset_names(self, label_dir):
        classes_path = os.path.join(label_dir, 'classes.txt')
        if os.path.isfile(classes_path):
            with open(classes_path, 'r', encoding='utf-8') as f:
                names = [x.strip() for x in f.read().splitlines() if x.strip()]
            if names:
                return names

        max_cls = -1
        for item in os.listdir(label_dir):
            if not item.lower().endswith('.txt') or item.lower() == 'classes.txt':
                continue
            txt_path = os.path.join(label_dir, item)
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    max_cls = max(max_cls, cid)
            except Exception:
                continue

        if max_cls >= 0:
            if self.predefined_classes and len(self.predefined_classes) >= (max_cls + 1):
                return list(self.predefined_classes[:max_cls + 1])
            return ['class_%d' % i for i in range(max_cls + 1)]
        return ['object']

    def _write_yolo_classes_txt_sidecars(self, dataset_root, names):
        """Write classes.txt at dataset root and under labels/train & labels/val for inspection."""
        lines = [str(n).strip() for n in names if str(n).strip()]
        if not lines:
            lines = ['object']
        body = '\n'.join(lines) + '\n'
        root = os.path.abspath(dataset_root)
        paths = (
            os.path.join(root, 'classes.txt'),
            os.path.join(root, 'labels', 'train', 'classes.txt'),
            os.path.join(root, 'labels', 'val', 'classes.txt'),
        )
        for p in paths:
            parent = os.path.dirname(p)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(body)
        print('[ALT][INFO][make_datasets] classes.txt written (%d classes) -> root, labels/train, labels/val' % len(lines))

    def _write_dataset_yaml(self, dataset_root, label_dir, yaml_name='data.yaml'):
        names = self._infer_dataset_names(label_dir)
        yaml_path = os.path.join(dataset_root, yaml_name)
        dataset_root_norm = os.path.abspath(dataset_root).replace('\\', '/')
        lines = [
            'path: %s' % dataset_root_norm,
            'train: images/train',
            'val: images/val',
            'nc: %d' % len(names),
            'names:'
        ]
        for i, name in enumerate(names):
            safe_name = str(name).replace("'", "\\'")
            lines.append("  %d: '%s'" % (i, safe_name))
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        self._write_yolo_classes_txt_sidecars(dataset_root, names)
        return yaml_path

    def _normalize_yaml_for_training_linebased(self, yaml_file, yaml_dir):
        """Legacy line-based path: rewrite when train/val exist next to yaml."""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                raw_lines = f.read().splitlines()

            key_values = {}
            for ln in raw_lines:
                s = ln.strip()
                if (not s) or s.startswith('#') or (':' not in s):
                    continue
                k, v = s.split(':', 1)
                key_values[k.strip()] = v.strip().strip("'").strip('"')

            root_path = key_values.get('path', '').replace('/', os.sep).replace('\\', os.sep).strip()
            train_rel = key_values.get('train', 'images/train').replace('/', os.sep).replace('\\', os.sep).strip()
            val_rel = key_values.get('val', 'images/val').replace('/', os.sep).replace('\\', os.sep).strip()

            train_dir = os.path.join(root_path, train_rel) if root_path else os.path.join(yaml_dir, train_rel)
            val_dir = os.path.join(root_path, val_rel) if root_path else os.path.join(yaml_dir, val_rel)
            train_ok = os.path.isdir(train_dir)
            val_ok = os.path.isdir(val_dir)
            if train_ok and val_ok:
                return yaml_file

            fallback_train = os.path.join(yaml_dir, train_rel)
            fallback_val = os.path.join(yaml_dir, val_rel)
            if os.path.isdir(fallback_train) and os.path.isdir(fallback_val):
                updated = []
                path_written = False
                for ln in raw_lines:
                    if ln.strip().startswith('path:'):
                        updated.append("path: %s" % yaml_dir.replace('\\', '/'))
                        path_written = True
                    else:
                        updated.append(ln)
                if not path_written:
                    updated.insert(0, "path: %s" % yaml_dir.replace('\\', '/'))
                with open(yaml_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(updated) + '\n')
                print('[ALT][INFO][yaml_fix] Updated yaml path to:', yaml_dir)
                return yaml_file
        except Exception as e:
            print('[ALT][WARN][yaml_fix] linebased failed:', e)
        return yaml_file

    def _normalize_yaml_for_training(self, yaml_file):
        """Fix stale Ultralytics data.yaml: wrong path: breaks test:/val: resolution (e.g. moved dataset)."""
        yaml_file = os.path.abspath(yaml_file)
        yaml_dir = os.path.dirname(yaml_file)

        try:
            import yaml as _yaml
        except ImportError:
            _yaml = None

        if _yaml is None:
            return self._normalize_yaml_for_training_linebased(yaml_file, yaml_dir)

        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = _yaml.safe_load(f) or {}
        except Exception as e:
            print('[ALT][WARN][yaml_fix] yaml parse failed:', e)
            return self._normalize_yaml_for_training_linebased(yaml_file, yaml_dir)

        def as_path(p):
            if p is None:
                return ''
            if isinstance(p, (list, tuple)):
                p = p[0] if p else ''
            return str(p).strip().strip("'").strip('"')

        def full_for_root(root, rel):
            rel = as_path(rel)
            if not rel:
                return ''
            rel_os = rel.replace('/', os.sep)
            if os.path.isabs(rel_os):
                return os.path.normpath(rel_os)
            r = root if (root and str(root).strip()) else yaml_dir
            return os.path.normpath(os.path.join(str(r), rel_os))

        def relpath_under(rroot, abs_p):
            abs_p = os.path.normpath(abs_p)
            rroot = os.path.normpath(rroot)
            try:
                return os.path.relpath(abs_p, rroot).replace('\\', '/')
            except ValueError:
                return abs_p.replace('\\', '/')

        train_s = data.get('train')
        val_s = data.get('val')
        test_s = data.get('test')
        declared_root = as_path(data.get('path'))

        seen = set()
        root_order = []
        for c in (yaml_dir, os.path.dirname(yaml_dir), declared_root):
            if not c or c in seen:
                continue
            seen.add(c)
            if os.path.isdir(c):
                root_order.append(c)

        best_root = None
        for cand in root_order:
            ta = full_for_root(cand, train_s)
            va = full_for_root(cand, val_s)
            if ta and va and os.path.isdir(ta) and os.path.isdir(va):
                best_root = cand
                break

        if not best_root:
            return self._normalize_yaml_for_training_linebased(yaml_file, yaml_dir)

        nu = dict(data)
        new_path = os.path.normpath(best_root).replace('\\', '/')
        nu['path'] = new_path

        if train_s:
            ta = full_for_root(best_root, train_s)
            if ta and os.path.isdir(ta):
                nu['train'] = relpath_under(best_root, ta)
        if val_s:
            va = full_for_root(best_root, val_s)
            if va and os.path.isdir(va):
                nu['val'] = relpath_under(best_root, va)

        if test_s is not None and str(test_s).strip() != '':
            te = full_for_root(best_root, test_s)
            if not (te and os.path.isdir(te)):
                nu.pop('test', None)
                print('[ALT][INFO][yaml_fix] Removed missing test: split (was:', as_path(test_s), ')')
        elif 'test' in nu and not as_path(nu.get('test')).strip():
            nu.pop('test', None)

        def _norm_path_field(d, k):
            v = as_path(d.get(k, ''))
            if not v:
                return ''
            return os.path.normpath(v).replace('\\', '/')

        changed = (
            _norm_path_field(nu, 'path') != _norm_path_field(data, 'path')
            or nu.get('train') != data.get('train')
            or nu.get('val') != data.get('val')
            or ('test' in data) != ('test' in nu)
        )
        if not changed:
            return yaml_file

        try:
            with open(yaml_file, 'w', encoding='utf-8') as f:
                _yaml.safe_dump(
                    nu, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
            print('[ALT][INFO][yaml_fix] Rewrote yaml for training path=%s train=%s val=%s test=%s' % (
                nu.get('path'), nu.get('train'), nu.get('val'), nu.get('test', '<none>')))
        except Exception as e:
            print('[ALT][WARN][yaml_fix] write failed:', e)
            return self._normalize_yaml_for_training_linebased(yaml_file, yaml_dir)

        return yaml_file

    @staticmethod
    def _yolo_yaml_abs_split_dir(yaml_dir, data, key):
        """Resolve Ultralytics-style train/val/test path relative to yaml dir and optional path: root."""
        v = data.get(key)
        if isinstance(v, (list, tuple)):
            v = v[0] if v else ''
        v = str(v or '').strip().strip("'\"")
        if not v:
            return ''
        v_os = v.replace('/', os.sep)
        if os.path.isabs(v_os):
            return os.path.normpath(v_os)
        r = str(data.get('path', '') or '').strip().strip("'\"")
        if r:
            r_os = r.replace('/', os.sep)
            if os.path.isabs(r_os):
                root = os.path.normpath(r_os)
            else:
                root = os.path.normpath(os.path.join(yaml_dir, r_os))
        else:
            root = yaml_dir
        return os.path.normpath(os.path.join(root, v_os))

    def _training_validate_data_yaml(self, yaml_path):
        """Pre-flight check before YOLO train: train/val image dirs must exist and contain images."""
        yaml_path = os.path.abspath(yaml_path)
        yaml_dir = os.path.dirname(yaml_path)
        try:
            import yaml as _yaml
        except ImportError:
            return True, ''
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = _yaml.safe_load(f) or {}
        except Exception as e:
            return False, u'无法读取 data yaml：%s' % e
        train_dir = self._yolo_yaml_abs_split_dir(yaml_dir, data, 'train')
        val_dir = self._yolo_yaml_abs_split_dir(yaml_dir, data, 'val')
        img_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

        def count_imgs(d):
            if not d or not os.path.isdir(d):
                return 0
            n = 0
            try:
                for name in os.listdir(d):
                    low = name.lower()
                    if low.endswith(img_ext):
                        n += 1
            except OSError:
                return 0
            return n

        nt = count_imgs(train_dir)
        nv = count_imgs(val_dir)
        parts = []
        if not train_dir:
            parts.append(u'train: 未在 yaml 中配置或为空')
        elif not os.path.isdir(train_dir):
            parts.append(u'train 目录不存在：\n%s' % train_dir)
        elif nt == 0:
            parts.append(u'train 目录下没有图片（%s）' % train_dir)
        if not val_dir:
            parts.append(u'val: 未在 yaml 中配置或为空')
        elif not os.path.isdir(val_dir):
            parts.append(u'val 目录不存在：\n%s' % val_dir)
        elif nv == 0:
            parts.append(u'val 目录下没有图片（%s）' % val_dir)
        if parts:
            hint = (
                u'\n\n请确认 data.yaml 中 path / train / val 与磁盘结构一致（常见为 path 下 images/train 与 images/val）。\n'
                u'说明见 https://docs.ultralytics.com/datasets/'
            )
            return False, '\n'.join(parts) + hint
        print('[ALT][INFO][train_check] train images=%d (%s) val images=%d (%s)' % (nt, train_dir, nv, val_dir))
        return True, ''

    @staticmethod
    def _count_make_dataset_outputs(path_out):
        """After make_train_val: count images and YOLO label txts per split (excludes classes.txt)."""
        root = os.path.abspath(path_out)
        img_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

        def count_imgs(sub):
            d = os.path.join(root, 'images', sub)
            if not os.path.isdir(d):
                return 0
            return sum(1 for f in os.listdir(d) if f.lower().endswith(img_ext))

        def count_lbl(sub):
            d = os.path.join(root, 'labels', sub)
            if not os.path.isdir(d):
                return 0
            return sum(
                1 for f in os.listdir(d)
                if f.lower().endswith('.txt') and f.lower() != 'classes.txt')

        tr_i, va_i = count_imgs('train'), count_imgs('val')
        tr_l, va_l = count_lbl('train'), count_lbl('val')
        return tr_i, va_i, tr_l, va_l, root

    def make_datasets(self):
        from libs.voc_2_yolo import voc2yolo
        from libs.make_train_val_test_datasets import make_train_val
        path_out = r'datasets'
        try:
            image_dr = QFileDialog.getExistingDirectory(self,"Choosing labeled images folder:", self.img_folder_path)
            if not image_dr:
                return
            label_dr = QFileDialog.getExistingDirectory(self,"Choosing label files folder:",self.xml_folder_path)
            print('image dir:%s'%image_dr)
            print('label dir:%s'%label_dr)
            if not label_dr:
                return
            data_dr = QFileDialog.getExistingDirectory(self,"Choosing datasets folder:", 'datasets')
            if data_dr:
                path_out = data_dr
            
            label_format_list = ['VOC(xml)', 'YOLO(txt)']
            items = tuple(label_format_list)

            label_format, ok = QInputDialog.getItem(self, "Select",
            "label file's format(VOC or YOLO)':", items, 0, False)
            if not ok:
                return
            if 'VOC' in label_format:
                txt_dr = os.path.join(os.path.dirname(label_dr), 'txts')
                # print(txt_dr)
                filename, _ = QFileDialog.getOpenFileName(self,"Choosing class name list file:", r'datasets',"TXT file(*.txt)")
                if not filename:
                    return
                f = open(filename, 'r', encoding='utf-8')
                classes = f.read().splitlines()
                # print(classes)
                f.close()
                # f = open(filename, 'r')
                # classes = f.readlines()
                # f.close()
                print('load class names: ', len(classes)) 
                voc2yolo(txt_dr, label_dr, classes)
                label_dr = txt_dr
            print('VOC label files converted to yolo format')
            make_train_val(image_dr, label_dr, path_out, 0.7)
            yaml_path = self._write_dataset_yaml(path_out, label_dr, yaml_name='data.yaml')
            tr_i, va_i, tr_l, va_l, out_abs = self._count_make_dataset_outputs(path_out)
            tot_i, tot_l = tr_i + va_i, tr_l + va_l
            msg = (
                u'训练/验证集已生成。\n\n'
                u'训练集：图片 %d 张，YOLO 标注 %d 个\n'
                u'验证集：图片 %d 张，YOLO 标注 %d 个\n'
                u'合计：图片 %d 张，标注 %d 个\n\n'
                u'已写入 classes.txt：输出根目录、labels/train、labels/val（与 data.yaml 中 names 一致）。\n\n'
                u'输出目录：\n%s\n\n'
                u'YAML：\n%s'
            ) % (tr_i, tr_l, va_i, va_l, tot_i, tot_l, out_abs, os.path.abspath(yaml_path))
            QMessageBox.information(self, u'Done!', msg)
                         
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
    
    def training_model(self):
        try:
            weights = self._select_model_weight(
                "Training base model (default %s, supports legacy .pt/.pth/.h5):" % DEFAULT_DETECT_PT,
                prefer_model=DEFAULT_DETECT_PT)
            if not weights:
                return
            weights = resolve_yolo_checkpoint(weights, os.path.join(APP_ROOT, 'weights'))
            self.weights = weights

            start_yaml_dir = os.path.join(APP_ROOT, 'datasets')
            if not os.path.isdir(start_yaml_dir):
                start_yaml_dir = APP_ROOT
            yaml_file, _ = QFileDialog.getOpenFileName(
                self,
                "Choose data yaml file:",
                start_yaml_dir,
                "YAML file (*.yaml *.yml);;All files (*.*)")
            if not yaml_file:
                return
            if not (yaml_file.endswith('.yaml') or yaml_file.endswith('.yml')):
                QMessageBox.information(self, u'Wrong!', u'data file must end with .yaml or .yml')
                return
            yaml = yaml_file
            self.yaml = self._normalize_yaml_for_training(yaml)
            print(self.yaml)
            if not os.path.isfile(self.yaml):
                QMessageBox.warning(self, u'Training', u'找不到 yaml 文件：\n%s' % self.yaml)
                return
            ok_paths, vmsg = self._training_validate_data_yaml(self.yaml)
            if not ok_paths:
                QMessageBox.warning(self, u'Training data', vmsg)
                return

            imgsz,OK=QInputDialog.getInt(self,'Image size','input img size:',value=1280)
            if not OK:
                return
        #%% calculation of max batchsize
            # weights = r"C:\Users\brigc\Documents\Python\Pytorch\ObjectiveDetection\Auto_Label_Tool\weights\fish_single_s_20230701.pt"
            # imgsz=1280
            try:
                if os.path.exists(weights):
                    fsize = int(os.path.getsize(weights) / (1024 * 1024))
                    _, free_mem, _ = get_cuda_setup(24, 24)
                    max_bsize = int(free_mem / (fsize * imgsz * imgsz / (1000 * 35)))
                    print(fsize, free_mem, max_bsize)
                else:
                    # Builtin model id (e.g. yolo26n.pt) may not exist as a local file yet.
                    # Use a conservative default max batch size and let Ultralytics handle download/load.
                    _, free_mem, _ = get_cuda_setup(24, 24)
                    max_bsize = max(1, int(free_mem / (imgsz * imgsz / 35)))
                    print('using builtin model id:', weights, 'estimated max batchsize:', max_bsize)
            except Exception:
                print('weight file is not right or broken')
                return
         #%% set hyper parameters
            # set batchsize
            bsize, OK = QInputDialog.getInt(self,'Batch size','dont over max batchsize:%d'%max_bsize, value=8)
            if not OK:
                return
            print('Batch size:%d'%bsize)
            # if bszie > max_bsize:
            #     QMessageBox.information(self,u'batch size is over the capacity of GPU right now')
            #     bsize, OK = QInputDialog.getInt(self,'Batch size','dont over max batchsize:%d'%max_bsize,value=8)
            #     if not OK:
            #         return
            # set epochs
            n_epochs, OK=QInputDialog.getInt(self,'Epochs','input epochs:',value=100)

            if not OK:
                return
            optimizer = 'SGD'
            lr0 = 0.001

            print('epochs:%d'%n_epochs, 'lr0:%f'%lr0, 'optimizer:%s'%optimizer)
            self.progress = QProgressDialog(self)
            self.progress.setWindowTitle(u"Waiting")  
            self.progress.setLabelText(u"training label model now, Please wait...")
            self.progress.setCancelButtonText(u"Cancle training")
            self.progress.setRange(0,0)
            self.progress.show()
            # Remove stale YOLO cache files near selected yaml, do not assume fixed datasets path.
            cache_search_roots = {
                os.path.dirname(os.path.abspath(self.yaml)),
                os.path.join(APP_ROOT, 'datasets')
            }
            for root_dir in list(cache_search_roots):
                if not root_dir or (not os.path.isdir(root_dir)):
                    continue
                for walk_root, _, files in os.walk(root_dir):
                    for fl in files:
                        if fl.endswith('.cache'):
                            try:
                                os.remove(os.path.join(walk_root, fl))
                            except Exception:
                                pass
            training = train_one(data=self.yaml,
                                 imgsz=imgsz,bsize=bsize, 
                                 epochs=n_epochs,
                                 weights=self.weights,
                                 optimizer=optimizer,
                                 lr0=lr0)
            training.str_signal.connect(self.train_info)
            training.start()
            QCoreApplication.processEvents()
            if self.progress.wasCanceled():
                return
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
            # self.progress.setRange(0,100) 
            # self.progress.setValue(100)
            return
    
    def train_info(self, str_info):
        from datetime import datetime
        now = datetime.now().strftime('%Y%m%d')
        if str_info == 'done':
            self.progress.setRange(0,100) 
            self.progress.setValue(100)
            QMessageBox.information(self,u'Done!',u'training finished, check result at runs')
            ckpt = get_latest_run(r'runs')
            print(ckpt)
            if ckpt:
                shutil.copy(os.path.join(os.path.dirname(ckpt), 'best.pt'), r'weights')
                model_path = os.path.join(r'weights', r'best.pt')
                model_name = os.path.join(r'weights', os.path.basename(self.weights))
                print(model_path)
                print(model_name)
                model_name = model_name.replace('.pt','')
                new_model_path = (model_name.split('2023')[0]+'_%s.pt'%now).replace('__', '_')
                update_record = {
                    'time': QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss'),
                    'source_checkpoint': ckpt,
                    'source_weight': self.weights,
                    'staged_best': model_path,
                    'target_weight': new_model_path,
                    'result': 'skipped_exists',
                }
                if not os.path.exists(new_model_path):
                    os.rename(model_path, (model_name.split('2023')[0]+'_%s.pt'%now).replace('__', '_'))
                    update_record['result'] = 'updated'
                    QMessageBox.information(self,u'Update!',u'Auto-labeling model weights updated!')
                self._append_jsonl_record(os.path.join(APP_ROOT, 'weights', 'update_history.jsonl'), update_record)
                self._record_upgrade_event('weight_update', update_record)
        else:
            print('[ALT][ERROR][train_thread] %s' % str_info)
            QMessageBox.information(self,u'Error!', str_info) 
        
    # def data_auto_agument(self):
    #     """data agument, using Affine change, intensity change, contrast change, gama change, Gaussian fillter to agument img data.
    #     you can select agument multiple(1~4).
    #     this action may take some time, please don't click mouse too frequently.
    #     """
    #     try:
    #         self.xml_folder_path=self.defaultSaveDir
    #         self.img_folder_path=os.path.dirname(self.filePath)
    #         imglist = natsort.natsorted(os.listdir(self.img_folder_path))
    #         xmllist = natsort.natsorted(os.listdir(self.xml_folder_path))
    #         # print(len(imglist),len(xmllist))
    #         if len(imglist) != len(xmllist):
    #             QMessageBox.information(self,u'Wrong!',u'lens of img and xml do not equal.')
    #             return 
    #         else:
    #             magnification,OK=QInputDialog.getInt(self,'Integer input dialog','input agument magnification(1~4):',value=4)
    #             if OK:
    #                 if magnification<1 or magnification>4:
    #                     magnification=4
    #                 img_Temp=[]
    #                 xml_Temp=[]
    #                 for i in range(magnification):
    #                     img_Temp.extend(imglist)
    #                     xml_Temp.extend(xmllist)
    #                 one_step=int(len(img_Temp)/4)
    #                 imglist1=img_Temp[0:one_step];xmllist1=xml_Temp[0:one_step]
    #                 imglist2=img_Temp[one_step:2*one_step];xmllist2=xml_Temp[one_step:2*one_step]
    #                 imglist3=img_Temp[2*one_step:3*one_step];xmllist3=xml_Temp[2*one_step:3*one_step]
    #                 imglist4=img_Temp[3*one_step:];xmllist4=xml_Temp[3*one_step:]
    #                 progress = QProgressDialog(self)
    #                 progress.setWindowTitle(u"Waiting")  
    #                 progress.setLabelText(u"Processing now,Please wait...")
    #                 progress.setCancelButtonText(u"Cancle data agument")
    #                 progress.setMinimumDuration(1)
    #                 progress.setWindowModality(Qt.WindowModal)
    #                 progress.setRange(0,100) 
                    
    #                 self.agument_A(imglist1,xmllist1,progress)
    #                 self.agument_B(imglist2,xmllist2,progress)
    #                 self.agument_C(imglist3,xmllist3,progress)
    #                 self.agument_D(imglist4,xmllist4,progress)
    #                 imglist = natsort.natsorted(os.listdir(self.img_folder_path))
    #                 xmllist = natsort.natsorted(os.listdir(self.xml_folder_path))
    #                 self.exam_agument(xmllist,progress)
                    
    #                 progress.setValue(100)
    #                 QMessageBox.information(self,"Done","data agument scuesseed！")
            
    #     except Exception as e:
    #         QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(print_error()))
            
    # def agument_A(self,imglist,xmllist,progress):
    #     print('agmt:',len(imglist))
    #     shift_info=[]
    #     basename_list = [os.path.splitext(x)[0] for x in imglist]
    #     for i in range(len(imglist)):
    #         progress.setValue(17*(i/(len(imglist)))) 
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
    #         item=imglist[i]
    #         if os.path.splitext(item)[-1] in QImageReader.supportedImageFormats():
    #             print(item)
    #             imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
    #             img = cv2.imread(imgPath)
    #             if img.shape:
    #                 size=img.shape
    #                 new_img1=cv2.flip(img,1,dst=None)
    #                 shift_X=np.random.randint(-0.15*size[1], 0.15*size[1])
    #                 shift_Y=np.random.randint(-0.15*size[0], 0.15*size[0])
    #                 shift_info.append([shift_X,shift_Y])
    #                 M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
    #                 shifted = cv2.warpAffine(new_img1, M, (new_img1.shape[1], new_img1.shape[0]),borderValue=(99,99,99))
    #                 noise=np.random.randint(-8,8,size=[size[0],size[1],3])
    #                 new_img=shifted+noise
    #                 save_path=self.img_folder_path+'/agmtA_'+item
    #                 cv2.imwrite(save_path,new_img)
    #             else:
    #                 print('error with imread:%s'%imgPath)
    #                 continue
        
    #     for i in range(len(xmllist)):
    #         progress.setValue(17+3*(i/(len(xmllist))))
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
    #         item=xmllist[i]
    #         if item.endswith('.xml'):
    #             filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
    #             item_img = imglist[basename_list.index(os.path.splitext(item)[0])]
    #             if item_img:
    #                 imgPath=os.path.join(os.path.abspath(self.img_folder_path), item_img)
    #                 img = cv2.imread(imgPath)
    #                 if img.shape:
    #                     size = img.shape
    #                     tree = ET.ElementTree(file=filePath)
    #                     root=tree.getroot()
    #                     root.find('filename').text='agmtA_'+item_img
    #                     root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtA_'+item_img
    #                     for child in root:
    #                         if child.tag=='object':
    #                             for gchild in child:
    #                                 if gchild.tag=='bndbox':
    #                                     temp=gchild[0].text
    #                                     gchild[0].text=str(size[1]-int(gchild[2].text)+shift_info[i][0])
    #                                     gchild[1].text=str(int(gchild[1].text)+shift_info[i][1])
    #                                     gchild[2].text=str(size[1]-int(temp)+shift_info[i][0])
    #                                     gchild[3].text=str(int(gchild[3].text)+shift_info[i][1])
    #                     tree.write(self.xml_folder_path+'/agmtA_'+item)
    #     print('agument_A done!')

    # def agument_B(self,imglist,xmllist,progress):
    #     shift_info=[]
    #     basename_list = [os.path.splitext(x)[0] for x in imglist]
    #     for i in range(len(imglist)):
    #         progress.setValue(20+17*(i/(len(imglist)))) 
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
    #         item=imglist[i]
    #         if os.path.splitext(item)[-1] in QImageReader.supportedImageFormats():
    #             print(item)
    #             imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
    #             img = cv2.imread(imgPath)
    #             if img.shape:
    #                 size=img.shape
    #                 result=99*np.ones(img.shape)
    #                 k=random.uniform(0.5,0.7) #根据实际需求更改范围，小于1为缩小，大于1为放大
    #                 small = cv2.resize(img, (0,0), fx = k, fy = k, interpolation = cv2.INTER_AREA)
    #                 result[0:small.shape[0],0:small.shape[1],:]=small
    #                 shift_X=np.random.randint(-0.1*size[1], 0.3*size[1])
    #                 shift_Y=np.random.randint(-0.1*size[0], 0.3*size[0])
    #                 shift_info.append([k,shift_X,shift_Y])
    #                 M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) 
    #                 shifted = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]),borderValue=(99,99,99))
    #                 noise=np.random.randint(-8,8,size=[size[0],size[1],3])
    #                 new_img=shifted+noise
    #                 save_path=self.img_folder_path+'/agmtB_'+item
    #                 cv2.imwrite(save_path,new_img)
    #             else:
    #                 print('error with imread:%s'%imgPath)
    #                 continue
                
    #     for i in range(len(xmllist)):
    #         progress.setValue(37+3*(i/(len(xmllist)))) 
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
    #         item=xmllist[i]
    #         if item.endswith('.xml'):
    #             xmlPath=os.path.join(os.path.abspath(self.xml_folder_path), item)
    #             item_img = imglist[basename_list.index(os.path.splitext(item)[0])]
    #             if item_img:
    #                 imgPath=os.path.join(os.path.abspath(self.img_folder_path), 'agmtB_'+item_img)
                
    #                 tree = ET.ElementTree(file=xmlPath)
    #                 root=tree.getroot()
    #                 root.find('filename').text='agmtB_'+item_img
    #                 root.find('path').text=imgPath
    #                 for child in root.findall('object'):
    #                     ymin=int(child.find('bndbox').find('ymin').text)
    #                     ymax=int(child.find('bndbox').find('ymax').text)
    #                     xmin=int(child.find('bndbox').find('xmin').text)
    #                     xmax=int(child.find('bndbox').find('xmax').text)
    #                     child.find('bndbox').find('ymin').text=str(int(shift_info[i][0]*ymin+shift_info[i][2]))
    #                     child.find('bndbox').find('ymax').text=str(int(shift_info[i][0]*ymax+shift_info[i][2]))
    #                     child.find('bndbox').find('xmin').text=str(int(shift_info[i][0]*xmin+shift_info[i][1]))
    #                     child.find('bndbox').find('xmax').text=str(int(shift_info[i][0]*xmax+shift_info[i][1]))
    #                 tree.write(self.xml_folder_path+'/agmtB_'+item)
    #     print('agument_B done!')

    # def agument_C(self,imglist,xmllist,progress):
    #     shift_info=[]
    #     basename_list = [os.path.splitext(x)[0] for x in imglist]
    #     for i in range(len(imglist)):
    #         progress.setValue(40+17*(i/(len(imglist)))) 
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
    #         item=imglist[i]
    #         if os.path.splitext(item)[-1] in QImageReader.supportedImageFormats():
    #             imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
    #             img = cv2.imread(imgPath)
    #             size=img.shape
    #             a=int(2*random.randint(1,3)+1)
    #             b=random.uniform(11,21)
    #             blur = cv2.GaussianBlur(img,(a,a),b)
    #             shift_X=np.random.randint(-0.1*size[1], 0.1*size[1])
    #             shift_Y=np.random.randint(-0.1*size[0], 0.1*size[0])
    #             shift_info.append([shift_X,shift_Y])
    #             M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
    #             shifted = cv2.warpAffine(blur, M, (blur.shape[1], blur.shape[0]),borderValue=(99,99,99))
    #             noise=np.random.randint(-5,5,size=[size[0],size[1],3])
    #             new_img=shifted+noise
    #             save_path=self.img_folder_path+'/agmtC_'+item
    #             cv2.imwrite(save_path,new_img)

    #     for i in range(len(xmllist)):
    #         progress.setValue(57+3*(i/(len(xmllist)))) 
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
    #         item=xmllist[i]
    #         if item.endswith('.xml'):
    #             filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
    #             item_img = imglist[basename_list.index(os.path.splitext(item)[0])]
    #             if item_img:
    #                 imgPath=os.path.join(os.path.abspath(self.img_folder_path), item_img)
    #             img = cv2.imread(imgPath)
    #             if img.shape:
    #                 size=img.shape
    #                 tree = ET.ElementTree(file=filePath)
    #                 root=tree.getroot()
    #                 root.find('filename').text='agmtC_'+item_img
    #                 root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtC_'+item_img
    #                 for child in root.findall('object'):
    #                     ymin=int(child.find('bndbox').find('ymin').text)
    #                     ymax=int(child.find('bndbox').find('ymax').text)
    #                     xmin=int(child.find('bndbox').find('xmin').text)
    #                     xmax=int(child.find('bndbox').find('xmax').text)
    #                     child.find('bndbox').find('ymin').text=str(int(ymin+shift_info[i][1]))
    #                     child.find('bndbox').find('ymax').text=str(int(ymax+shift_info[i][1]))
    #                     child.find('bndbox').find('xmin').text=str(int(xmin+shift_info[i][0]))
    #                     child.find('bndbox').find('xmax').text=str(int(xmax+shift_info[i][0]))
    #                 tree.write(self.xml_folder_path+'/agmtC_'+item)
    #     print('agument_C done!')

    # def agument_D(self,imglist,xmllist,progress):
    #     shift_info=[]
    #     basename_list = [os.path.splitext(x)[0] for x in imglist]
    #     for i in range(len(imglist)):
    #         progress.setValue(60+27*(i/(len(imglist)))) 
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
    #         item=imglist[i]
    #         if os.path.splitext(item)[-1] in QImageReader.supportedImageFormats():
    #             imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
    #             img = cv2.imread(imgPath)
    #             if img.shape:
    #                 size=img.shape
    #                 k=random.uniform(0.6,1.3)
    #                 gama=exposure.adjust_gamma(img,k)
    #                 shift_X=np.random.randint(-0.1*size[1], 0.1*size[1])
    #                 shift_Y=np.random.randint(-0.1*size[0], 0.1*size[0])
    #                 shift_info.append([shift_X,shift_Y])
    #                 M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
    #                 shifted = cv2.warpAffine(gama, M, (gama.shape[1], gama.shape[0]),borderValue=(99,99,99))
    #                 noise=np.random.randint(-5,5,size=[size[0],size[1],3])
    #                 new_img=shifted+noise
    #                 save_path=self.img_folder_path+'/agmtD_'+item
    #                 cv2.imwrite(save_path,new_img)
    #     for i in range(len(xmllist)):
    #         progress.setValue(87+3*(i/(len(xmllist)))) 
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
    #         item=xmllist[i]
    #         if item.endswith('.xml'):
    #             filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
    #             item_img = imglist[basename_list.index(os.path.splitext(item)[0])]
    #             if item_img:
    #                 imgPath=os.path.join(os.path.abspath(self.img_folder_path), item_img)
    #             img = cv2.imread(imgPath)
    #             if img.shape:
    #                 size=img.shape
    #                 tree = ET.ElementTree(file=filePath)
    #                 root=tree.getroot()
    #                 root.find('filename').text='agmtD_'+item_img
    #                 root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtD_'+item_img
    #                 for child in root.findall('object'):
    #                     ymin=int(child.find('bndbox').find('ymin').text)
    #                     ymax=int(child.find('bndbox').find('ymax').text)
    #                     xmin=int(child.find('bndbox').find('xmin').text)
    #                     xmax=int(child.find('bndbox').find('xmax').text)
    #                     child.find('bndbox').find('ymin').text=str(int(ymin+shift_info[i][1]))
    #                     child.find('bndbox').find('ymax').text=str(int(ymax+shift_info[i][1]))
    #                     child.find('bndbox').find('xmin').text=str(int(xmin+shift_info[i][0]))
    #                     child.find('bndbox').find('xmax').text=str(int(xmax+shift_info[i][0]))
    #                 tree.write(self.xml_folder_path+'/agmtD_'+item)
    #     print('agument_D done!')
        
    # def IOU(self,rectA, rectB):  
    #     W = min(rectA[2], rectB[2]) - max(rectA[0], rectB[0])
    #     H = min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])
    #     if W <= 0 or H <= 0:
    #         return 0;
    #     SA = (rectA[2] - rectA[0]) * (rectA[3] - rectA[1])
    #     SB = (rectB[2] - rectB[0]) * (rectB[3] - rectB[1])
    #     cross = W * H
    #     return cross/(SA + SB - cross)

    # def exam_bndbox_is_lawful(self,window=[0,0,100,100],bndbox=[10,10,20,30],min_IOU=0.002,max_IOU=1):
    #     '''
    #     判断bndbox是否在图片范围window内，若全部在图片范围外，删掉box；对于有部分在图片内的，将box图片范围外的部分裁剪掉，
    #     同时若裁剪后IOU过小，同样删除box
    #     '''
    #     if min_IOU<self.IOU(window,bndbox)<=max_IOU:
    #         is_lawful=True
    #         for i in range(2):
    #             if bndbox[i]<window[i]:
    #                 bndbox[i]=window[i]
    #         for j in range(2,4):
    #             if bndbox[j]>window[j]:
    #                 bndbox[j]=window[j]
    #     else:
    #         is_lawful=False
    #     return is_lawful,bndbox
        
    # def exam_agument(self,xmllist,progress):
    #     for i in range(len(xmllist)):
    #         progress.setValue(90+9*(i/(len(xmllist)))) 
    #         if progress.wasCanceled():
    #             QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
    #             break
            
    #         item=xmllist[i]
    #         if item.endswith('.xml'):
    #             xmlPath=os.path.join(os.path.abspath(self.xml_folder_path), item)
    #             tree = ET.ElementTree(file=xmlPath)
    #             root=tree.getroot()

    #             a=int(root.find('size').find('width').text)
    #             b=int(root.find('size').find('height').text)
    #             window=[0,0,a,b]

    #             for child in root.findall('object'):
    #                 a=int(child.find('bndbox').find('xmin').text)
    #                 b=int(child.find('bndbox').find('ymin').text)
    #                 c=int(child.find('bndbox').find('xmax').text)
    #                 d=int(child.find('bndbox').find('ymax').text)
    #                 bndbox=[min(a,c),min(b,d),max(a,c),max(b,d)]
    #                 is_lawful,new_bndbox=self.exam_bndbox_is_lawful(window=window,bndbox=bndbox)
    #                 if not is_lawful:
    #                     root.remove(child)
    #                 else:
    #                     child.find('bndbox').find('xmin').text=str(new_bndbox[0])
    #                     child.find('bndbox').find('ymin').text=str(new_bndbox[1])
    #                     child.find('bndbox').find('xmax').text=str(new_bndbox[2])
    #                     child.find('bndbox').find('ymax').text=str(new_bndbox[3])
                    
    #             tree.write(xmlPath)
    #     print('exam_agument done!')

def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    
    # Usage : ALT.py image predefClassFile saveDir
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(
                         os.path.dirname(sys.argv[0]),
                         'datasets', 'predefined_classes.txt'),
                     argv[3] if len(argv) >= 4 else None) #os.path.join(os.path.dirname(sys.argv[0]), 'data', 'xml'))
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    app, _win = get_main_app(sys.argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())

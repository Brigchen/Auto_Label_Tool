import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar, QGridLayout,QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import configparser
from ultralytics import YOLO
from libs.repo_paths import repo_root, configs_dir
from libs.yolo_weights import resolve_yolo_checkpoint

class training(QThread):

    finish_signal = pyqtSignal(str)
    abort_signal = pyqtSignal(str)
    # progress_signal = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(training, self).__init__()

        self.epochs = kwargs.get('epochs')
        self.batch = kwargs.get('batch')
        self.imgsz = kwargs.get('imgsz')
        self.name = kwargs.get('name')
        self.optimizer = kwargs.get('optimizer')
        self.lr0 = kwargs.get('lr0')
        self.yaml_file = kwargs.get('yaml_file')
        self.weights_file = kwargs.get('weights_file')
        self.task = kwargs.get('task', 'detect')

    def run(self):
        try:
            wdir = os.path.join(repo_root(), "weights")
            w = resolve_yolo_checkpoint(self.weights_file, app_weights_dir=wdir)
            if w != self.weights_file:
                print("[FVT][train] resolved weights:", self.weights_file, "->", w)
            model = YOLO(w)
            train_kw = dict(
                data=self.yaml_file,
                epochs=int(self.epochs),
                batch=int(self.batch),
                imgsz=int(self.imgsz),
                name=self.name,
                optimizer=self.optimizer,
                lr0=float(self.lr0),
            )
            t = (self.task or "detect").lower()
            if t not in ("detect", "det", ""):
                train_kw["task"] = t
            model.train(**train_kw)
            self.finish_signal.emit('finished')
        except Exception as e:
            print(e)
            self.abort_training()

    def abort_training(self):
        self.abort_signal.emit('aborted')

class FVTGui(QWidget):
    def __init__(self):
        super().__init__()
        self.epochs = '100'
        self.batch = '-1'
        self.imgsz = '1280'
        self.name = 'Training'
        self.optimizer = 'AdamW'
        self.lr0 = '0.001'
        self.yaml_file = ""
        self.weights_file = ""
        self.task = "detect"
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 数据集部分
        dataset_groupbox = QGroupBox('数据集')
        dataset_layout = QHBoxLayout()
        

        self.yaml_file_label = QLabel('Yaml文件:')
        self.yaml_file_lineedit = QLineEdit()
        self.yaml_file_lineedit.setText(self.yaml_file)
        self.yaml_file_button = QPushButton('浏览')
        dataset_layout.addWidget(self.yaml_file_label)
        dataset_layout.addWidget(self.yaml_file_lineedit)
        dataset_layout.addWidget(self.yaml_file_button)

        dataset_groupbox.setLayout(dataset_layout)
        layout.addWidget(dataset_groupbox)

        # 模型部分
        model_groupbox = QGroupBox('预训练模型')
        model_layout = QHBoxLayout()
        

        self.weights_label = QLabel('权重文件:')
        self.weights_lineedit = QLineEdit()
        self.weights_lineedit.setText(self.weights_file)
        self.weights_button = QPushButton('浏览')
        model_layout.addWidget(self.weights_label)
        model_layout.addWidget(self.weights_lineedit)
        model_layout.addWidget(self.weights_button)

        model_groupbox.setLayout(model_layout)
        layout.addWidget(model_groupbox)

        task_group = QGroupBox("训练任务 (Ultralytics：目标/关键点/分割等；追踪在 FishVision 推理阶段)")
        task_lo = QHBoxLayout()
        self.task_label = QLabel("task:")
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "pose", "segment", "obb", "classify"])
        self.task_combo.setCurrentText(self.task)
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        task_lo.addWidget(self.task_label)
        task_lo.addWidget(self.task_combo)
        task_group.setLayout(task_lo)
        layout.addWidget(task_group)

        # 训练参数部分
        train_groupbox = QGroupBox('训练参数')
        train_layout = QGridLayout()

        self.epochs_lineedit = QLineEdit(str(self.epochs))
        self.epochs_lineedit.textChanged.connect(self.on_epochs_changed)

        self.batch_lineedit = QLineEdit(str(self.batch))
        self.batch_lineedit.textChanged.connect(self.on_batch_changed)

        self.imgsz_lineedit = QLineEdit(str(self.imgsz))
        self.imgsz_lineedit.textChanged.connect(self.on_imgsz_changed)

        self.name_lineedit = QLineEdit(self.name)
        self.name_lineedit.textChanged.connect(self.on_name_changed)

        self.optimizer_lineedit = QLineEdit(self.optimizer)
        self.optimizer_lineedit.textChanged.connect(self.on_optimizer_changed)

        self.lr0_lineedit = QLineEdit(str(self.lr0))
        self.lr0_lineedit.textChanged.connect(self.on_lr0_changed)

        train_layout.addWidget(QLabel('Epochs:'), 0, 0)
        train_layout.addWidget(self.epochs_lineedit, 0, 1)

        train_layout.addWidget(QLabel('Batch:'), 0, 2)
        train_layout.addWidget(self.batch_lineedit, 0, 3)

        train_layout.addWidget(QLabel('Imgsz:'), 0, 4)
        train_layout.addWidget(self.imgsz_lineedit, 0, 5)

        train_layout.addWidget(QLabel('Name:'), 1, 0)
        train_layout.addWidget(self.name_lineedit, 1, 1)

        train_layout.addWidget(QLabel('Optimizer:'), 1, 2)
        train_layout.addWidget(self.optimizer_lineedit, 1, 3)

        train_layout.addWidget(QLabel('Lr0:'), 1, 4)
        train_layout.addWidget(self.lr0_lineedit, 1, 5)

        train_groupbox.setLayout(train_layout)
        layout.addWidget(train_groupbox)

        # 进度条和按钮
        self.start_button = QPushButton('开始训练')
        self.cancel_button = QPushButton('退出')
        self.message_box = QLabel('enjoy training your own yolo model')
        self.message_box.setStyleSheet("border :2px solid green;font :bold italic 12px 'Microsoft YaHei'")
        self.message_box.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar()

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)

        layout.addWidget(self.message_box)
        layout.addWidget(self.progress_bar)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.yaml_file_button.clicked.connect(self.browse_yaml_file)
        self.weights_button.clicked.connect(self.browse_weights)
        
        self.cancel_button.clicked.connect(self.exit_app)
        self.start_button.clicked.connect(self.start_training)

        self.setWindowTitle('FishVision Trainer')
        self.setGeometry(100, 100, 1000, 500)  # 设置宽高比为16:9
        self.show()
        self.load_config()

    def load_config(self):
        print('161: load config')
        config = configparser.ConfigParser()
        _cfg = os.path.join(configs_dir(), "config_FVT.ini")
        if os.path.exists(_cfg):
            try:
                config.read(_cfg)
                self.epochs = config.get('settings','epochs')
                self.batch = config.get('settings','batch')
                self.imgsz = config.get('settings','imgsz')
                self.name = config.get('settings','name')
                self.optimizer = config.get('settings','optimizer')
                self.lr0 = config.get('settings','lr0')
                self.yaml_file = config.get('settings','yaml_file')
                self.weights_file = config.get('settings','weights_file')
                self.task = config.get('settings', 'task') if config.has_option('settings', 'task') else 'detect'
                self.task_combo.setCurrentText(self.task)

                self.epochs_lineedit.setText(self.epochs)
                self.batch_lineedit.setText(self.batch)    
                self.imgsz_lineedit.setText(self.imgsz)        
                self.name_lineedit.setText(self.name)        
                self.optimizer_lineedit.setText(self.optimizer)        
                self.lr0_lineedit.setText(self.lr0)                               
                self.yaml_file_lineedit.setText(self.yaml_file)
                self.weights_lineedit.setText(self.weights_file)
            except Exception as e:
                print(u'导入配置参数失败:', e)
                self.message_box.setText(u'导入配置参数失败')

    def save_config(self):
        print('187: save config')
        config = configparser.ConfigParser()
        config['settings']= {
                "epochs":self.epochs,
                "batch":self.batch,
                "imgsz":self.imgsz,
                "name":self.name,
                "optimizer":self.optimizer,
                "lr0":self.lr0,
                "yaml_file":self.yaml_file,
                "weights_file":self.weights_file,
                "task": self.task,
        }
        with open(os.path.join(configs_dir(), "config_FVT.ini"), "w") as configfile:
            config.write(configfile)

    def browse_yaml_file(self):
        start = os.path.join(repo_root(), "datasets")
        self.yaml_file, _ = QFileDialog.getOpenFileName(self, "选择Yaml文件", start)
        print(self.yaml_file)
        self.yaml_file_lineedit.setText(self.yaml_file)

    def browse_weights(self):
        start = os.path.join(repo_root(), "weights")
        self.weights_file, _ = QFileDialog.getOpenFileName(self, "选择模型权重文件", start)
        print(self.weights_file)
        self.weights_lineedit.setText(self.weights_file)

    # 定义槽函数
    def on_task_changed(self, text):
        self.task = str(text)

    def on_epochs_changed(self, text):
        self.epochs = str(text)
        self.batch = str(text)

    def on_imgsz_changed(self, text):
        self.imgsz = str(text)

    def on_name_changed(self, text):
        self.name = str(text)

    def on_optimizer_changed(self, text):
        self.optimizer = str(text)

    def on_lr0_changed(self, text):
        self.lr0 = str(text)

    def start_training(self):
        if self.weights_file and self.yaml_file:
            self.save_config()
            self.start_button.setEnabled(False)
            self.message_box.setText(u'开始训练了，请在cmd窗口查看训练进展')
            self.progress_bar.setEnabled(True)
            self.progress_bar.setRange(0,0)
            self.thread1 = training( 
                    epochs = self.epochs,
                    batch = self.batch,
                    imgsz = self.imgsz,
                    name = self.name,
                    optimizer = self.optimizer,
                    lr0 = self.lr0,
                    yaml_file = self.yaml_file,
                    weights_file = self.weights_file,
                    task = self.task,
            )
            
            self.thread1.finish_signal.connect(self.finish_callback)
            self.thread1.abort_signal.connect(self.abort_callback)
            self.thread1.start()
        else:   
            self.message_box.setText('请设置数据集yaml文件与预训练模型权重文件')

    def finish_callback(self, text):
        if text == 'finished':
            self.start_button.setEnabled(True)
            self.progress_bar.setEnabled(False)
            self.message_box.setText(u'训练完成了，请在目录Runs下查看结果')
    
    def abort_callback(self, text):
        if text == 'aborted':
            self.start_button.setEnabled(True)
            self.message_box.setText(u'训练中止了，请在cmd窗口查看错误信息')
            
    def exit_app(self):
        # 在此处添加取消训练逻辑
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FVTGui()
    sys.exit(app.exec_())
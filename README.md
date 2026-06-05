# Auto_Label_Tool

基于 **PyQt5** 的桌面工具集：支持 **手动与自动标注**、**视频 / RTSP 检测演示**，以及基于 **Ultralytics YOLO** 的训练（检测、姿态、分割等）。适用于通用目标检测标注流程。

---

## 功能简介（简版）

| 模块 | 说明 |
|------|------|
| **ALT 标注主程序**（`src/ALT.py`） | VOC / YOLO 格式、关键点、自动标注、**Ctrl+Z 撤销**、视频追踪标注、放大镜、类别筛选与统计、标注校正；**Choose Auto-Label Model** 一次设置权重与 conf；**Train Console**；CLI **`--datasets`**；启动闪屏与大数据集加载进度。 |
| **FishVision**（`src/FishVision_GUI.py`） | 视频或 RTSP 预览、检测，可选跟踪与画面增强。 |
| **FishVision 训练界面**（`src/FishVision_Train_GUI.py`） | 五选项卡训练控制台：基本参数、优化器、数据增强、硬件/高级、自动调参；实时单行进度（s/it、ETA）、每 epoch 指标摘要；**测试报告**（train/val/test、随机抽样、低分样本导出）；**打开 ALT** 改标注；退出自动保存参数；TensorBoard / 终端训练；训练结束自动复制 `best.pt` 到 `weights/`。 |

共享逻辑在 `libs/`（IO、设置、YOLO/VOC、导出、叠加检查等）；应用配置在 `configs/`；**数据**建议放在 `datasets/`、**权重**放在 `weights/`（仓库内已有占位说明，大文件默认不提交）。

---

## 安装（简版）

1. **准备 Python 环境**（建议 3.8+），并自行安装 **PyTorch**（按显卡/CPU 从 [pytorch.org](https://pytorch.org/) 选择命令；`requirements.txt` 一般不包含 `torch`）。
2. **克隆或解压**本仓库到本地工作目录。
3. 在**仓库根目录**执行依赖安装：

```bat
py -3 -m pip install -r requirements.txt
```

4. 将所用 **`.pt` / `.pth` 权重** 放到 **`weights/`**（推荐）；图像与标注等数据放到 **`datasets/`**（子目录结构可与软件内配置一致）。

---

## 使用说明（简版）

1. **始终在仓库根目录启动**（保证 `configs/`、`datasets/`、`weights/` 等相对路径正确）。
2. **Windows**：可直接双击根目录下的批处理；或在已激活环境的终端中执行：

```bat
python src\alt_boot.py
python src\ALT.py
python src\ALT.py --datasets H:/path/to/dataset --split val
python src\FishVision_GUI.py
python src\FishVision_Train_GUI.py
python src\FishVision_Train_GUI.py --data H:/path/to/data.yaml
```

- `Auto_Label_Tool.bat` → 启动 ALT（推荐，含启动闪屏）  
- `FishVision.bat` → FishVision  
- `FishVision_trainer.bat` → 训练界面  

3. **Python 解释器选择顺序**（批处理内）：若设置了环境变量 **`AUTO_LABEL_PYTHON`** 且路径存在，则优先使用；否则依次尝试 `C:\ProgramData\Anaconda3\python.exe`、`py -3`、`python`。使用非上述路径的 Conda 环境时，请设置 `AUTO_LABEL_PYTHON` 指向该环境的 `python.exe`，或在激活环境后于终端运行上述 `python` 命令。

4. **配置**  
   - ALT 窗口与状态：`configs/` 下由程序生成的 `ALT_Settings.pkl` 等（由 `libs/settings.py` 管理）。  
   - FishVision：`configs/config.ini`  
   - 训练界面：首次模板 `configs/config_FVT.ini`；**上次会话**自动保存为 `configs/config_FVT_last.ini`（关闭 GUI 或开始训练时写入，下次启动优先加载）。  
   其他自定义 `config_*.ini` 请放在 **`configs/`**。

5. **训练（FishVision Trainer）**  
   - 在 GUI 中配置 data yaml、权重、batch/imgsz/workers/cache/AMP 等，点击「开始训练」。  
   - 底部状态栏显示实时 batch 进度；日志区每 epoch 输出一行验证指标。  
   - **「测试报告」**：对 train/val/test 或自定义目录评估，生成 HTML；可导出全部低分样本（`images/<划分>/`）并在 ALT 中改标注。  
   - **「打开 ALT」**：从训练界面启动标注工具。  
   - 「终端运行」在独立 cmd 窗口执行，便于查看完整 Ultralytics 输出。  
   - 「TensorBoard」指向 `runs/`（实际 run 在 `runs/detect/<name>` 等子目录）。  
   - 大显存 + 大数据集建议：`workers=4`、`cache=disk` 或 `ram`；RTX 50 系列需 PyTorch 2.x + cu128。

6. **标注（ALT）**  
   - 菜单 **Annotate-Tools → Train Console** 打开 FishVision 训练控制台。  
   - **Annotate-Tools → Choose Auto-Label Model**：同时设置检测模型与 **conf / IoU** 阈值。  
   - **Edit → Undo / Redo** 或 **Ctrl+Z / Ctrl+Y**：撤销误删框、Auto Label 等操作。  
   - 滚轮缩放图像，**Shift+滚轮** 平移。  
   - 文件列表右键可删除图像及标注（无二次确认）。  
   - 命令行：`Auto_Label_Tool.bat --datasets <数据集根目录> --split test`。

---

## 版本与变更记录

- 当前版本号见根目录 **`VERSION`**（程序内亦可能与 `libs/constants.py` 中版本信息对应）。  
- 变更说明见 **`CHANGELOG.md`**。

---

## 仓库结构（简表）

| 路径 | 作用 |
|------|------|
| `src/ALT.py` | 主标注程序 |
| `src/alt_boot.py` | ALT 启动入口（闪屏 + 延迟加载 ML 库） |
| `libs/anno_undo.py` | 标注撤销/重做 |
| `libs/eval_export.py` | 训练/测试 per-class Excel 与混淆矩阵导出 |
| `src/FishVision_GUI.py` | 视频 / RTSP 工具 |
| `src/FishVision_Train_GUI.py` | 训练图形界面（选项卡控制台 + 测试报告） |
| `libs/fvt_train_runner.py` | 训练执行与进度/导出逻辑 |
| `libs/test_report.py` | 测试集 HTML 报告与低分样本导出 |
| `libs/alt_launcher.py` / `libs/fvt_launcher.py` | ALT / Trainer 子进程启动 |
| `libs/train_monitor.py` | TensorBoard、results.csv 监控 |
| `libs/` | 公共库与资源加载 |
| `configs/` | 应用 ini / 本地状态 |
| `datasets/`、`weights/` | 用户数据与权重（见各目录内 `README.md`） |
| `resources/` | 图标、字体、Qt 字符串资源 |

修改 `resources/strings/*.properties` 或图标后，如需重新打包资源，可在具备 PyQt 工具的环境下执行：  
`pyrcc5 resources.qrc -o libs/resources.py`

---

## 许可证与联系

详见 **`LICENSE`** 及程序内 **关于** 信息。  
另有一份历史中文说明 **`readme_CN.md`**（部分内容可能与当前目录结构不一致，请以本 `README.md` 为准）。

---

## English (brief)

PyQt5 desktop suite for manual/auto labeling, video/RTSP demos, and Ultralytics YOLO training. Install PyTorch separately, then `pip install -r requirements.txt` from repo root. Run `python src\ALT.py` (and sibling scripts) or use the `.bat` launchers on Windows. Data → `datasets/`, weights → `weights/`, settings → `configs/`.

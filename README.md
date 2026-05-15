# Auto_Label_Tool

PyQt5 desktop tools for **manual + automatic labeling**, **video detection / RTSP demos**, and **Ultralytics YOLO training** (detection, pose, segmentation, etc.). Suitable for general object-labeling workflows, not limited to a single domain.

## Version

- Current version: see **`VERSION`** (also mirrored in `libs/constants.py` as `APP_VERSION` / `APP_UPDATE_DATE`).
- Release notes: **`CHANGELOG.md`**.

## Repository layout

| Path | Role |
|------|------|
| `src/ALT.py` | Main labeling app: VOC / YOLO, keypoints, auto-label, training hooks. |
| `src/FishVision_GUI.py` | Video / RTSP preview, detection, optional tracking & enhancement. |
| `src/FishVision_Train_GUI.py` | Training UI: yaml + weights + hyperparameters + task type (`detect`, `pose`, `segment`, …). |
| `libs/` | Shared UI, IO, settings, YOLO/VOC helpers, export (`model_export`), overlay checks (`overlay_check`), taxon screening (`taxa_screen`). |
| `configs/` | **Application configuration** — `config.ini` (FishVision), `config_FVT.ini` (trainer), `ALT_Settings.pkl` (ALT window/state), and any `config_*.ini` you add. |
| `datasets/` | User datasets (images, labels, yaml); not part of the application package. |
| `resources/` | Icons and `strings` for Qt resources (`resources.qrc` → `libs/resources.py`). |
| `weights/` | Place your `.pt` weights here (recommended). |

## How to run

1. Create / activate a Python environment with **PyTorch** (install from [PyTorch](https://pytorch.org/) for your CUDA/CPU), then install Python deps from the repo root:

```bat
py -3 -m pip install -r requirements.txt
```

(`requirements.txt` lists packages such as **natsort**, PyQt5, OpenCV, ultralytics, etc.; it does not install `torch`—use your existing PyTorch env.)

2. From the **repository root** (so `datasets/`, `configs/`, `weights/` resolve correctly):

```bat
python src\ALT.py
python src\FishVision_GUI.py
python src\FishVision_Train_GUI.py
```

On Windows you can double-click the `.bat` files in the repo root. They **`cd` to the repo folder**, then pick a Python in this order:

1. **`AUTO_LABEL_PYTHON`** if set and the file exists (for any conda env or custom install).
2. **`C:\ProgramData\Anaconda3\python.exe`** if present (typical “All Users” Anaconda).
3. **`py -3`** if the Python launcher is on `PATH`.
4. **`python`** on `PATH`.

If you use a **conda env** that is *not* the base install above, either set `AUTO_LABEL_PYTHON` to that env’s `python.exe`, or run from an activated terminal: `python src\ALT.py`. You can also create a tiny wrapper `.bat` that runs `call conda activate your_env` then calls the same `python.exe` path.

- `Auto_Label_Tool.bat` — starts ALT  
- `FishVision.bat` — FishVision GUI  
- `FishVision_trainer.bat` — FishVision trainer  

## Configuration

- **ALT**: persistent UI/state in `configs/ALT_Settings.pkl` (managed by `libs/settings.py`).
- **FishVision**: `configs/config.ini`
- **FishVision Train**: `configs/config_FVT.ini`
- Additional `config_*.ini` files belong under **`configs/`**; migrate any old copies from the repo root if you still have them locally.

## Developer notes

- After editing `resources/strings/*.properties` or icons, regenerate Qt resources:  
  `pyrcc5 resources.qrc -o libs/resources.py`
- ONNX / TensorRT helpers: `from libs.model_export import export_yolo_pt_to_onnx, build_tensorrt_engine`
- Label hygiene: `from libs.overlay_check import remove_overlay, process_yolo_tree`
- Taxon screening: `from libs.taxa_screen import main_yolo, main_xml`

## License / contact

See `setup.py` and in-app **About** for maintainer and license metadata.

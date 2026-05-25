# Changelog

All notable changes to this project are documented here.

## [3.2.0] – 2026-05-21

### Added

- **FishVision Trainer** (`src/FishVision_Train_GUI.py`): five-tab training console — basic params, optimizer, data augmentation, hardware/advanced, and hyperparameter search.
- **`libs/fvt_train_runner.py`**: shared training entry for GUI thread and terminal subprocess (CUDA audit, AMP weight prefetch, TensorBoard, live progress routing, epoch metric summaries, `best.pt` export to `weights/`).
- **`libs/train_monitor.py`**: TensorBoard server helper, dynamic run-directory resolution, `results.csv` watcher.
- **`libs/train_augment_panel.py`**, **`libs/hyperparam_search.py`**: augment and auto-tune UI panels.
- **Live training progress**: single-line tqdm refresh (`s/it`, ETA) in GUI status bar; one summary line per epoch (P/R/mAP50/mAP50-95).
- **Session persistence**: auto-save all tabs on exit to `configs/config_FVT_last.ini`; next launch loads last session (falls back to `config_FVT.ini` template).
- **Terminal training**: launch training in an independent cmd window with full Ultralytics output.
- **Run directory controls**: resume from `last.pt`, `exist_ok`, warning when run name already exists.

### Changed

- **`FishVision_trainer.bat`**: set `CUBLAS_WORKSPACE_CONFIG` before launch (suppresses deterministic CuBLAS warnings).
- **`libs/yolo_weights.py`**: improved local checkpoint resolution for training weights.
- **`configs/config_FVT.ini`**: updated recommended defaults (e.g. `workers=4`, `cache=disk`, `imgsz=640`).

### Fixed

- RTX 50-series / PyTorch compatibility warnings; auto-disable AMP when GPU kernels are missing.
- Qt `QPainter` warnings during training (throttled progress updates, hide busy progress bar while live refresh runs).
- Filter Ultralytics table/header spam from GUI log; TensorBoard now sees runs under `runs/detect/`.

---

## [3.1.0] – 2026-05-12

### Changed

- **Branding**: In-app and documentation product name is **`Auto_Label_Tool`** (no longer “for fish” / fish-only framing in the main app title and README).
- **Layout**: Main applications live under `src/` (`src/ALT.py`, `src/FishVision_GUI.py`, `src/FishVision_Train_GUI.py`). Repository root no longer ships duplicate `ALT.py` launchers; run `python src/ALT.py` (after `cd` to repo root) or use the provided `.bat` files.
- **Configuration**: GUI INI files and the ALT pickle settings file are stored under `configs/` (`config.ini`, `config_FVT.ini`, `config_auto.ini`, `ALT_Settings.pkl`). Paths are resolved via `libs.repo_paths.configs_dir()`.
- **Dependencies**: Removed vendored `ultralytics/` and `track_libs/`; use pip-installed `ultralytics`. Utilities moved into `libs/` (`overlay_check`, `taxa_screen`, `model_export`, `repo_paths`, `pkg_paths`, `fish_enhance`).

### Added

- `README.md` (this workflow and run instructions).
- `VERSION` (semantic version string for packaging and docs).

### Removed

- Non-core CLI/demo scripts from the repository root (detection/train batch tools, duplicate GUIs, etc.), per product scope; `datasets/` content kept as user data.

---

Earlier product notes (from in-app constants) are summarized in `README.md` § Version history.

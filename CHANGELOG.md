# Changelog

All notable changes to this project are documented here.

## [3.3.0] – 2026-05-29

### Added

- **Test report** (`libs/test_report.py`, FishVision Trainer): evaluate train/val/test or custom dataset directory; HTML report with GT vs pred overlays; optional Ultralytics val metrics; export **all** low-score samples (F1 &lt; 1) to `images/<split>/` + `labels/<split>/`.
- **Random sample mode**: test full split by default, or evaluate a random subset count.
- **GUI cross-launch**: Trainer → **打开 ALT**; ALT → **Train Console** (`Ctrl+Shift+T`); post-report prompt to open ALT on exported low-score dataset.
- **`libs/alt_launcher.py`**, **`libs/fvt_launcher.py`**, **`libs/yolo_dataset_paths.py`**: subprocess launchers and YOLO path resolution shared by ALT and Trainer.
- **ALT CLI**: `--datasets` / `--dataset` and `--split` to auto-load images/labels on startup.
- **`libs/env_bootstrap.py`**: suppresses deprecated `pynvml` import warning before torch.
- **`scripts/build_fish_one2_test.py`**: helper to build a held-out test split from an existing YOLO dataset.
- **Batch launchers**: `Auto_Label_Tool.bat` and `FishVision_trainer.bat` pass through `%*` (CLI args).

### Changed

- **ALT canvas zoom**: mouse wheel zooms by default; **Shift+wheel** scrolls (no Ctrl required).
- **FishVision Trainer**: `--data` CLI pre-fills data yaml path.

### Fixed

- Test report dialog `dataset_status` init order; export `lbl_dir` typo; split-aware export folders (not always `test`).
- `QScrollBar.setValue(float)` crashes on wheel zoom / scroll.
- Training exit: `_QueueWriter` / stream wrappers now implement `close()`; logging handlers restored after train (`restore_logging_streams`).

---

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

# Changelog

All notable changes to this project are documented here.

## [3.5.1] – 2026-06-12

### Fixed

- **difficult/score conversion**: score > 0.5 → difficult=0 (not difficult), score ≤ 0.5 → difficult=1 (difficult) in both `yolo_io.split_yolo_difficult_and_score()` and `labelFile.py` Pascal VOC save path.
- **Interactive review previews**: now generates unlimited preview images for ALL applicable pending items (was capped at 200). `build_review_manifest` default `max_previews=0` means unlimited.
- **Review detail field**: `review_class` action now shows `{gt_name} -> {pred_name} (IoU=..., conf=...)` instead of generic `class mismatch`; `drop_low_score` and `drop_unmatched_gt` now include class name; `review_fn` includes class name.
- **TensorBoard on Windows**: `--logdir` now points to project base directory (e.g. `runs/detect/fish`) so all sub-runs are auto-discovered and selectable in the left panel.
- **Review.html static mode**: embedded manifest JSON inline to avoid CORS `Failed to fetch` when opened directly from file system. Server mode now force-rebuilds manifest before starting to ensure all previews are generated.

### Changed

- **Label Self-Refine dialog**: mode selection changed from two checkboxes (`review_only`, `html_only`) to a dropdown with 3 clear options. All dialog parameters now persist to `config_FVT.ini` and are restored on next open.
- **Label Self-Refine HTML report**: `index.html` simplified to show statistics + diagnosis only, with 10 representative sample previews per action type. Removed interactive review link.
- **Review.html**: removed static-mode fallback detection. Pure server mode only (must be started via Trainer). Simplified JS code.
- **Preview images**: highlight box now uses **orange** border (width 4px) with `★` prefix label for easy identification in multi-target scenes.

### Added

- **Preview highlight**: `draw_comparison()` accepts new `highlight_box` parameter. `build_review_manifest` automatically locates the target box by matching `old_line`/`new_line` and passes it for highlighted rendering.

---

## [3.5.0] – 2026-06-12

### Added

- **Label Self-Refine** (`libs/label_self_refine*.py`): compare GT vs predictions, apply conservative fixes (wrong class, low-score drops, missing boxes); dry-run mode; interactive review web server; CSV/HTML report.
- **Training Diagnosis** (`libs/test_report_analysis.py`): HTML report with data gaps, weak classes, confusion pairs, actionable suggestions; per-class instance counts across train/val/test.
- **Underwater Augment** (`libs/underwater_augment.py`, `libs/uw_train_hook.py`): offline batch augment + online training hook (attenuation, haze, turbidity, color cast, vignette, spotlight, clahe, gamma).
- **Cross-class dedup** (`libs/autolabel_dedup.py`): remove overlapping boxes across different classes during auto-label (video batch and single-image).
- **Test report export Predict labels**: option to export prediction boxes (with conf) instead of GT for low-score samples; useful for correcting model errors.
- **Make Datasets upgrade**: support creating test split; settings persistence (`SETTING_MAKE_DATASETS_*`); underwater offline augment integration (`uw_offline_copies`, `uw_offline_strength`).
- **TensorBoard improvements**: auto-select directory with event files (`pick_tensorboard_logdir`); Windows path normalization; force restart option.
- **YOLO line parse** (`libs/yolo_line_parse.py`): parse YOLO txt lines with optional confidence column.
- **Dataset paths** (`libs/dataset_paths.py`): recursive file iteration utilities for large datasets.
- **YOLO label clean** (`libs/yolo_label_clean.py`): six-column YOLO label compatibility wrapper for Ultralytics val/train.

### Changed

- **YOLO IO**: `addBndBox/addKeypoints` now accept explicit `difficult` and `score` parameters; score field properly serialized.
- **Dataset YAML**: supports `test: images/test` field when test split exists.
- **classes.txt**: written to `labels/test/` directory when test split is created.
- **Train augment panel**: added underwater augment controls (`uw_augment`, `uw_augment_p`, `uw_augment_strength`, `uw_include_enhance`).
- **Test report dialog**: added label source dropdown (GT / Predict with conf).
- **ALT Make Datasets dialog**: persistent settings; test split checkbox; underwater offline augment options.
- **Label Self-Refine dialog**: simplified mode selection (dropdown instead of checkboxes); parameters now persist to `config_FVT.ini`.
- **Label Self-Refine HTML report**: simplified `index.html` (statistics summary only); added interactive review button; removed duplicate preview tables.

### Fixed

- Six-column YOLO label compatibility during Ultralytics val/train (columns: class x y w h conf).
- Windows TensorBoard `--logdir_spec` path parsing issues (backslash conflicts).
- Auto-label score field handling for keypoints and bounding boxes.
- **difficult/score conversion**: score > 0.5 → difficult=0 (not difficult), score ≤ 0.5 → difficult=1 (difficult).

---

## [3.4.0] – 2026-05-30

### Added

- **ALT startup splash** (`src/alt_boot.py`, `libs/startup_ui.py`): show progress immediately; defer heavy **torch/ultralytics** import until Auto Label / training (`libs/lazy_ml.py`).
- **Annotation undo/redo** (`libs/anno_undo.py`): **Ctrl+Z** / **Ctrl+Y** (or **Ctrl+Shift+Z**) for delete, move, edit label, Auto Label, copy labels, etc. (up to 50 steps per image).
- **Choose Auto-Label Model** dialog: set **weights + conf / predict IoU / dedup IoU** together (`ChooseAutoLabelModelDialog` in `libs/annotate_dialogs.py`).
- **Eval export** (`libs/eval_export.py`): per-class metrics Excel + normalized confusion matrix (CSV/JSON) for training val and test report; dependency **openpyxl**.

### Changed

- **Edit Label**: input on top, class list below; default focus on list; highlight **last used** class; single-click or **Enter** on list confirms selection.
- **File list**: right-click delete image + labels **without** confirmation; unsaved edits on current image are discarded when deleting.
- **ALT launch**: `Auto_Label_Tool.bat` and `libs/alt_launcher.py` use `src/alt_boot.py`; large-folder load shows progress while indexing annotations.

### Fixed

- Edit Label: **Enter** after keyboard letter-jump in class list now confirms the highlighted item.
- `requirements.txt`: restore separate **openpyxl** line (was accidentally merged with comment).

---

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

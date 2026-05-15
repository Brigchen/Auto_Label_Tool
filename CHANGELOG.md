# Changelog

All notable changes to this project are documented here.

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

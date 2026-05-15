# -*- coding: utf-8 -*-
"""Paths into the pip-installed ultralytics package (not a vendored repo copy)."""
from pathlib import Path


def ultralytics_package_dir():
    import ultralytics

    return Path(ultralytics.__file__).resolve().parent


def ultralytics_cfg_yaml(*parts):
    """Resolve cfg/yaml under site-packages, e.g. ('models', 'v8', 'yolov8s.yaml')."""
    return ultralytics_package_dir().joinpath("cfg", *parts)


def resolve_cjk_plot_font(script_dir=None):
    """Font path for OpenCV / ultralytics.plot Chinese labels; None = library default."""
    import os

    here = script_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(here, "resources", "fonts", "wqy-microhei.ttc"),
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

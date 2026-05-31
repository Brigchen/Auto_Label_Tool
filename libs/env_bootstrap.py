# -*- coding: utf-8 -*-
"""Early process setup before torch / CUDA imports."""
from __future__ import annotations

import os
import warnings

if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Deprecated meta-package `pynvml` redirects imports and warns even when `nvidia-ml-py` is installed.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated.*",
    category=FutureWarning,
)

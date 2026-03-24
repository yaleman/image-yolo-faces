"""Batch face detection with YOLOv8."""

from __future__ import annotations

import os

# Albumentations performs a PyPI version check on import unless disabled.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

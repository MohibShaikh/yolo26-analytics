"""Base types for video sources."""

from __future__ import annotations

import numpy as np

from yolo26_analytics.models import FrameMeta

FrameItem = tuple[np.ndarray, FrameMeta]

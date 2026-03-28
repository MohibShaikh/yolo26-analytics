"""Base types for video sources."""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from yolo26_analytics.models import FrameMeta

FrameItem = tuple[npt.NDArray[Any], FrameMeta]

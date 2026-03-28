"""Heatmap generation from track centroid positions."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

_NDArray = npt.NDArray[Any]


class HeatmapAccumulator:
    """Accumulates centroid positions into a 2D histogram."""

    def __init__(self, width: int = 1920, height: int = 1080, radius: int = 20) -> None:
        self._width = width
        self._height = height
        self._radius = radius
        self._accumulator: _NDArray = np.zeros((height, width), dtype=np.float32)

    def add_point(self, x: int, y: int) -> None:
        if 0 <= x < self._width and 0 <= y < self._height:
            r = self._radius
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(self._width, x + r + 1)
            y2 = min(self._height, y + r + 1)
            patch = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
            cv2.circle(patch, (x - x1, y - y1), r, 1.0, -1)
            self._accumulator[y1:y2, x1:x2] += patch

    def get_heatmap(self) -> _NDArray:
        result: _NDArray = self._accumulator.copy()
        return result

    def reset(self) -> None:
        self._accumulator[:] = 0.0


def generate_heatmap_image(
    accumulator: HeatmapAccumulator,
    reference_frame: _NDArray,
    alpha: float = 0.6,
    blur_ksize: int = 51,
) -> _NDArray:
    """Generate a colored heatmap overlay on a reference frame."""
    heatmap = accumulator.get_heatmap()
    heatmap = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    h, w = reference_frame.shape[:2]
    if colored.shape[:2] != (h, w):
        colored = cv2.resize(colored, (w, h))
    overlay: _NDArray = cv2.addWeighted(reference_frame, 1 - alpha, colored, alpha, 0)
    return overlay

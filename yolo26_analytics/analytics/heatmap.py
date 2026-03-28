"""Heatmap generation from track centroid positions."""
from __future__ import annotations
import cv2
import numpy as np

class HeatmapAccumulator:
    """Accumulates centroid positions into a 2D histogram."""
    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self._width = width
        self._height = height
        self._accumulator = np.zeros((height, width), dtype=np.float32)

    def add_point(self, x: int, y: int) -> None:
        if 0 <= x < self._width and 0 <= y < self._height:
            self._accumulator[y, x] += 1.0

    def get_heatmap(self) -> np.ndarray:
        return self._accumulator.copy()

    def reset(self) -> None:
        self._accumulator[:] = 0.0

def generate_heatmap_image(
    accumulator: HeatmapAccumulator,
    reference_frame: np.ndarray,
    alpha: float = 0.6,
    blur_ksize: int = 51,
) -> np.ndarray:
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
    overlay = cv2.addWeighted(reference_frame, 1 - alpha, colored, alpha, 0)
    return overlay

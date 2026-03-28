from __future__ import annotations

import numpy as np
import pytest

from yolo26_analytics.analytics.heatmap import HeatmapAccumulator, generate_heatmap_image


class TestHeatmapAccumulator:
    def test_accumulates_points(self) -> None:
        acc = HeatmapAccumulator(width=640, height=480)
        acc.add_point(100, 100)
        acc.add_point(100, 100)
        acc.add_point(200, 200)
        heatmap = acc.get_heatmap()
        assert heatmap.shape == (480, 640)
        assert heatmap[100, 100] > heatmap[200, 200]

    def test_generate_heatmap_image(self) -> None:
        acc = HeatmapAccumulator(width=640, height=480)
        for _ in range(50):
            acc.add_point(320, 240)
        ref_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = generate_heatmap_image(acc, ref_frame)
        assert overlay.shape == (480, 640, 3)
        assert overlay[240, 320].sum() > 0

    def test_reset(self) -> None:
        acc = HeatmapAccumulator(width=640, height=480)
        acc.add_point(100, 100)
        acc.reset()
        heatmap = acc.get_heatmap()
        assert heatmap.sum() == 0.0

"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from yolo26_analytics.models import Detection, Track


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 640x480 black frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[Detection]:
    """A set of sample detections for testing."""
    return [
        Detection(bbox=(100, 100, 200, 300), confidence=0.9, class_name="person"),
        Detection(bbox=(300, 150, 400, 350), confidence=0.8, class_name="person"),
        Detection(bbox=(120, 110, 170, 180), confidence=0.7, class_name="helmet"),
    ]


@pytest.fixture
def sample_tracks(sample_detections: list[Detection]) -> list[Track]:
    """Sample tracks with IDs assigned."""
    return [
        Track(track_id=i + 1, detection=det)
        for i, det in enumerate(sample_detections)
    ]

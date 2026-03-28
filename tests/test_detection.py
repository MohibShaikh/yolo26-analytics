from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yolo26_analytics.detection.yolo26 import YOLO26Detector


class TestYOLO26Detector:
    def _make_mock_result(self) -> MagicMock:
        """Create a mock Ultralytics result."""
        box1 = MagicMock()
        box1.xyxy = MagicMock()
        box1.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 200, 300, 400]])
        box1.conf = MagicMock()
        box1.conf.cpu.return_value.numpy.return_value = np.array([0.92])
        box1.cls = MagicMock()
        box1.cls.cpu.return_value.numpy.return_value = np.array([0])

        result = MagicMock()
        result.boxes = [box1]
        return result

    @patch("yolo26_analytics.detection.yolo26.YOLO")
    def test_predict_returns_detections(self, mock_yolo_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "helmet"}
        mock_result = self._make_mock_result()
        mock_model.return_value = [mock_result]
        mock_yolo_cls.return_value = mock_model

        detector = YOLO26Detector(weights="yolo26n.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.predict(frame)

        assert len(detections) == 1
        assert detections[0].class_name == "person"
        assert detections[0].confidence == pytest.approx(0.92, abs=0.01)
        assert detections[0].bbox == (100, 200, 300, 400)

    @patch("yolo26_analytics.detection.yolo26.YOLO")
    def test_filters_below_confidence(self, mock_yolo_cls: MagicMock) -> None:
        box = MagicMock()
        box.xyxy = MagicMock()
        box.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 30, 40]])
        box.conf = MagicMock()
        box.conf.cpu.return_value.numpy.return_value = np.array([0.3])
        box.cls = MagicMock()
        box.cls.cpu.return_value.numpy.return_value = np.array([0])

        result = MagicMock()
        result.boxes = [box]
        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        mock_model.return_value = [result]
        mock_yolo_cls.return_value = mock_model

        detector = YOLO26Detector(weights="yolo26n.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.predict(frame)
        assert len(detections) == 0

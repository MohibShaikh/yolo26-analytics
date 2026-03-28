from __future__ import annotations
import numpy as np
from yolo26_analytics.detection.sahi import SAHIDetector
from yolo26_analytics.models import Detection

class FakeDetector:
    def predict(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        return [Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_name="person")]

class TestSAHI:
    def test_tiles_and_offsets(self) -> None:
        det = FakeDetector()
        sahi = SAHIDetector(det, slice_size=640, overlap=0.25)
        frame = np.zeros((1280, 1920, 3), dtype=np.uint8)
        results = sahi.predict(frame)
        # Should have detections from multiple tiles, merged by IoU
        assert len(results) >= 1
        # All should be person class
        assert all(d.class_name == "person" for d in results)

    def test_merge_duplicates(self) -> None:
        dets = [
            Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_name="person"),
            Detection(bbox=(12, 12, 52, 52), confidence=0.8, class_name="person"),
        ]
        merged = SAHIDetector._merge_detections(dets, iou_threshold=0.5)
        assert len(merged) == 1
        assert merged[0].confidence == 0.9

    def test_no_merge_different_classes(self) -> None:
        dets = [
            Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_name="person"),
            Detection(bbox=(12, 12, 52, 52), confidence=0.8, class_name="car"),
        ]
        merged = SAHIDetector._merge_detections(dets, iou_threshold=0.5)
        assert len(merged) == 2

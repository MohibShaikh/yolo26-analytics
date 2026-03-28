"""SAHI (Slicing Aided Hyper Inference) wrapper for any detector."""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from yolo26_analytics.models import Detection


class SAHIDetector:
    """Wraps any Detector with slice-based inference for small object detection."""

    def __init__(
        self,
        detector: object,
        slice_size: int = 640,
        overlap: float = 0.25,
    ) -> None:
        self._detector = detector
        self._slice_size = slice_size
        self._overlap = overlap

    def predict(self, frame: npt.NDArray[Any]) -> list[Detection]:
        h, w = frame.shape[:2]
        stride = int(self._slice_size * (1 - self._overlap))
        all_detections: list[Detection] = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + self._slice_size, w)
                y2 = min(y + self._slice_size, h)
                tile = frame[y:y2, x:x2]
                tile_dets = self._detector.predict(tile)  # type: ignore[attr-defined]
                for det in tile_dets:
                    bx1, by1, bx2, by2 = det.bbox
                    all_detections.append(
                        Detection(
                            bbox=(bx1 + x, by1 + y, bx2 + x, by2 + y),
                            confidence=det.confidence,
                            class_name=det.class_name,
                        )
                    )
        return self._merge_detections(all_detections)

    @staticmethod
    def _merge_detections(
        detections: list[Detection], iou_threshold: float = 0.5
    ) -> list[Detection]:
        if not detections:
            return []
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for det in sorted_dets:
            is_dup = False
            for kept_det in kept:
                if det.class_name != kept_det.class_name:
                    continue
                iou = SAHIDetector._compute_iou(det.bbox, kept_det.bbox)
                if iou > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(det)
        return kept

    @staticmethod
    def _compute_iou(
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

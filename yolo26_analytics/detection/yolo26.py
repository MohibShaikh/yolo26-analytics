"""YOLO26 detector adapter using Ultralytics."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from yolo26_analytics.models import Detection


class YOLO26Detector:
    """Wraps Ultralytics YOLO for the Detector protocol."""

    def __init__(self, weights: str = "yolo26n.pt", confidence: float = 0.5) -> None:
        self._model = YOLO(weights)
        self._confidence = confidence
        self._names: dict[int, str] = self._model.names

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf.cpu().numpy()[0])
                if conf < self._confidence:
                    continue
                xyxy = box.xyxy.cpu().numpy()[0]
                cls_id = int(box.cls.cpu().numpy()[0])
                detections.append(
                    Detection(
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=conf,
                        class_name=self._names.get(cls_id, f"class_{cls_id}"),
                    )
                )
        return detections

"""RF-DETR detector adapter using the rfdetr package."""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from yolo26_analytics.models import Detection


class RFDETRDetector:
    """Wraps RF-DETR for the Detector protocol."""

    def __init__(
        self,
        model_size: str = "base",
        confidence: float = 0.5,
    ) -> None:
        from rfdetr import RFDETRBase, RFDETRLarge
        from rfdetr.util.coco_classes import COCO_CLASSES

        self._confidence = confidence
        self._names: list[str] = COCO_CLASSES
        if model_size == "large":
            self._model = RFDETRLarge()
        else:
            self._model = RFDETRBase()

    def predict(self, frame: npt.NDArray[Any]) -> list[Detection]:
        result = self._model.predict(frame, threshold=self._confidence)
        detections: list[Detection] = []
        for class_id, conf, xyxy in zip(
            result.class_id, result.confidence, result.xyxy, strict=False
        ):
            detections.append(
                Detection(
                    bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    confidence=float(conf),
                    class_name=self._names[int(class_id)],
                )
            )
        return detections

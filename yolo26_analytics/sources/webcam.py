"""Webcam source."""
from __future__ import annotations
from collections.abc import AsyncIterator
from datetime import datetime, timezone
import cv2
from yolo26_analytics.models import FrameMeta
from yolo26_analytics.sources.base import FrameItem


class WebcamSource:
    """Reads frames from a webcam device."""
    def __init__(self, device: int = 0, source_id: str = "webcam") -> None:
        self._device = device
        self._source_id = source_id
        self._cap: cv2.VideoCapture | None = None

    async def __aiter__(self) -> AsyncIterator[FrameItem]:
        self._cap = cv2.VideoCapture(self._device)
        frame_index = 0
        try:
            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    break
                meta = FrameMeta(
                    timestamp=datetime.now(tz=timezone.utc),
                    frame_index=frame_index,
                    source_id=self._source_id,
                )
                yield frame, meta
                frame_index += 1
        finally:
            self._cap.release()

    async def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

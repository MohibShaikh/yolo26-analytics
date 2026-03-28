"""RTSP stream source."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import cv2

from yolo26_analytics.models import FrameMeta
from yolo26_analytics.sources.base import FrameItem


class RTSPSource:
    """Reads frames from an RTSP stream."""

    def __init__(self, url: str, source_id: str = "rtsp") -> None:
        self._url = url
        self._source_id = source_id
        self._cap: cv2.VideoCapture | None = None

    async def __aiter__(self) -> AsyncIterator[FrameItem]:
        self._cap = cv2.VideoCapture(self._url)
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

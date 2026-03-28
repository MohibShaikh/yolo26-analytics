"""Video file source."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import cv2

from yolo26_analytics.models import FrameMeta
from yolo26_analytics.sources.base import FrameItem


class VideoFileSource:
    """Reads frames from a video file."""

    def __init__(self, path: str, source_id: str) -> None:
        self._path = path
        self._source_id = source_id
        self._cap: cv2.VideoCapture | None = None

    async def __aiter__(self) -> AsyncIterator[FrameItem]:
        self._cap = cv2.VideoCapture(self._path)
        frame_index = 0
        try:
            while True:
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

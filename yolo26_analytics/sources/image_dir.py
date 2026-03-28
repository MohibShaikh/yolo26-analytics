"""Image directory source."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import cv2

from yolo26_analytics.models import FrameMeta
from yolo26_analytics.sources.base import FrameItem

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


class ImageDirSource:
    """Reads frames from a directory of images, sorted by filename."""

    def __init__(self, path: str, source_id: str = "image_dir") -> None:
        self._path = Path(path)
        self._source_id = source_id

    async def __aiter__(self) -> AsyncIterator[FrameItem]:
        files = sorted(f for f in self._path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        for i, filepath in enumerate(files):
            frame = cv2.imread(str(filepath))
            if frame is None:
                continue
            meta = FrameMeta(
                timestamp=datetime.now(tz=timezone.utc),
                frame_index=i,
                source_id=self._source_id,
            )
            yield frame, meta

    async def close(self) -> None:
        pass

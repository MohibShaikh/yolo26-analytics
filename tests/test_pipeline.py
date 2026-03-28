from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from yolo26_analytics.core.pipeline import Pipeline
from yolo26_analytics.models import Detection, FrameMeta, Track


class FakeSource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    async def __aiter__(self):
        for i, frame in enumerate(self._frames):
            meta = FrameMeta(
                timestamp=datetime.now(tz=timezone.utc),
                frame_index=i,
                source_id="test",
            )
            yield frame, meta

    async def close(self) -> None:
        pass


class FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def predict(self, frame: np.ndarray) -> list[Detection]:
        return self._detections


class FakeTracker:
    def __init__(self, tracks: list[Track]) -> None:
        self._tracks = tracks

    def update(self, detections: list[Detection]) -> list[Track]:
        return self._tracks

    def reset(self) -> None:
        pass


class TestPipeline:
    @pytest.mark.asyncio
    async def test_runs_pipeline_loop(self) -> None:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        det = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person")
        track = Track(track_id=1, detection=det)
        source = FakeSource(frames)
        detector = FakeDetector([det])
        tracker = FakeTracker([track])
        store = AsyncMock()
        store.write_tracks = AsyncMock()
        store.log_events = AsyncMock()
        alert_manager = AsyncMock()
        alert_manager.dispatch = AsyncMock()
        pipeline = Pipeline(
            source=source,
            detector=detector,
            tracker=tracker,
            store=store,
            zone_analyzer=None,
            alert_manager=alert_manager,
        )
        await pipeline.run_async()
        assert store.write_tracks.call_count == 3
        alert_manager.dispatch.assert_not_called()

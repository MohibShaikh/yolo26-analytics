from __future__ import annotations

from datetime import datetime, timezone

import pytest

from yolo26_analytics.models import Detection, Event, FrameMeta, Track
from yolo26_analytics.store.sqlite import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = SQLiteStore(path=db_path)
    await s.initialize()
    return s


class TestSQLiteTrackStore:
    @pytest.mark.asyncio
    async def test_write_and_query_tracks(self, store: SQLiteStore) -> None:
        tracks = [
            Track(
                track_id=1,
                detection=Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person"),
            ),
            Track(
                track_id=2,
                detection=Detection(
                    bbox=(300, 300, 400, 400), confidence=0.8, class_name="forklift"
                ),
            ),
        ]
        meta = FrameMeta(timestamp=datetime.now(tz=timezone.utc), frame_index=0, source_id="cam1")
        await store.write_tracks(tracks, meta)
        results = await store.query_tracks(source_id="cam1")
        assert len(results) == 2
        assert results[0]["track_id"] == 1
        assert results[0]["object_class"] == "person"
        assert results[1]["track_id"] == 2

    @pytest.mark.asyncio
    async def test_query_tracks_by_zone(self, store: SQLiteStore) -> None:
        tracks = [
            Track(
                track_id=1,
                detection=Detection(bbox=(150, 150, 250, 250), confidence=0.9, class_name="person"),
            )
        ]
        meta = FrameMeta(timestamp=datetime.now(tz=timezone.utc), frame_index=0, source_id="cam1")
        await store.write_tracks(tracks, meta, zone_name="Loading Dock")
        results = await store.query_tracks(zone_name="Loading Dock")
        assert len(results) == 1
        assert results[0]["zone_name"] == "Loading Dock"
        results = await store.query_tracks(zone_name="Entrance")
        assert len(results) == 0


class TestSQLiteEventStore:
    @pytest.mark.asyncio
    async def test_log_and_query_events(self, store: SQLiteStore) -> None:
        events = [
            Event(
                timestamp=datetime.now(tz=timezone.utc),
                event_type="entry",
                zone_name="Loading Dock",
                track_id=1,
                object_class="person",
                metadata={"direction": "inward"},
                confidence=0.9,
                frame_snapshot=b"",
                bbox=(100, 100, 200, 200),
            )
        ]
        await store.log_events(events)
        results = await store.query_events(zone_name="Loading Dock")
        assert len(results) == 1
        assert results[0]["event_type"] == "entry"
        assert results[0]["metadata"]["direction"] == "inward"

    @pytest.mark.asyncio
    async def test_query_events_by_type(self, store: SQLiteStore) -> None:
        events = [
            Event(
                timestamp=datetime.now(tz=timezone.utc),
                event_type="entry",
                zone_name="Dock",
                track_id=1,
                object_class="person",
                metadata={},
                confidence=0.9,
                frame_snapshot=b"",
                bbox=(100, 100, 200, 200),
            ),
            Event(
                timestamp=datetime.now(tz=timezone.utc),
                event_type="dwell_exceeded",
                zone_name="Dock",
                track_id=1,
                object_class="person",
                metadata={"dwell_seconds": 400},
                confidence=0.9,
                frame_snapshot=b"",
                bbox=(100, 100, 200, 200),
            ),
        ]
        await store.log_events(events)
        results = await store.query_events(event_type="dwell_exceeded")
        assert len(results) == 1
        assert results[0]["event_type"] == "dwell_exceeded"

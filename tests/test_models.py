from datetime import datetime, timezone

from yolo26_analytics.models import Detection, Event, FrameMeta, Track


class TestDetection:
    def test_create_detection(self) -> None:
        det = Detection(
            bbox=(100, 200, 300, 400),
            confidence=0.85,
            class_name="person",
        )
        assert det.bbox == (100, 200, 300, 400)
        assert det.confidence == 0.85
        assert det.class_name == "person"

    def test_centroid(self) -> None:
        det = Detection(bbox=(100, 200, 300, 400), confidence=0.9, class_name="person")
        assert det.centroid == (200, 300)


class TestTrack:
    def test_create_track(self) -> None:
        det = Detection(bbox=(100, 200, 300, 400), confidence=0.85, class_name="person")
        track = Track(track_id=1, detection=det)
        assert track.track_id == 1
        assert track.detection.class_name == "person"
        assert track.centroid == (200, 300)


class TestEvent:
    def test_create_event(self) -> None:
        now = datetime.now(tz=timezone.utc)
        event = Event(
            timestamp=now,
            event_type="entry",
            zone_name="Loading Dock",
            track_id=1,
            object_class="person",
            metadata={"direction": "inward"},
            confidence=0.85,
            frame_snapshot=b"",
            bbox=(100, 200, 300, 400),
        )
        assert event.event_type == "entry"
        assert event.metadata["direction"] == "inward"


class TestFrameMeta:
    def test_create_frame_meta(self) -> None:
        now = datetime.now(tz=timezone.utc)
        meta = FrameMeta(
            timestamp=now,
            frame_index=42,
            source_id="cam1",
        )
        assert meta.frame_index == 42
        assert meta.source_id == "cam1"

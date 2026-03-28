from datetime import datetime, timedelta, timezone
from yolo26_analytics.models import Detection, Track
from yolo26_analytics.zones.throughput import ThroughputTracker
from yolo26_analytics.zones.polygon import Zone

class TestThroughputTracker:
    def _zone(self) -> Zone:
        return Zone(name="test", polygon=[(0, 0), (200, 0), (200, 200), (0, 200)], track_classes=["person"])

    def test_counts_unique_tracks_per_interval(self) -> None:
        tp = ThroughputTracker(interval=60)
        zone = self._zone()
        now = datetime.now(tz=timezone.utc)
        tracks = [Track(track_id=1, detection=Detection(bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"))]
        tp.update(zone, tracks, now)
        tp.update(zone, tracks, now + timedelta(seconds=10))
        tracks2 = [Track(track_id=2, detection=Detection(bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"))]
        tp.update(zone, tracks2, now + timedelta(seconds=20))
        stats = tp.get_stats(zone.name)
        assert stats["unique_tracks"] == 2

    def test_resets_after_interval(self) -> None:
        tp = ThroughputTracker(interval=60)
        zone = self._zone()
        now = datetime.now(tz=timezone.utc)
        tracks = [Track(track_id=1, detection=Detection(bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"))]
        tp.update(zone, tracks, now)
        tp.update(zone, tracks, now + timedelta(seconds=70))
        stats = tp.get_stats(zone.name)
        assert stats["unique_tracks"] == 1

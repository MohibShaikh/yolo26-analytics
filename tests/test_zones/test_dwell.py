from datetime import datetime, timedelta, timezone
from yolo26_analytics.models import Detection, Track
from yolo26_analytics.zones.dwell import DwellTracker
from yolo26_analytics.zones.polygon import Zone

class TestDwellTracker:
    def _zone(self) -> Zone:
        return Zone(name="test", polygon=[(0, 0), (200, 0), (200, 200), (0, 200)], track_classes=["person"])

    def test_tracks_dwell_time(self) -> None:
        dwell = DwellTracker(alert_threshold=10)
        zone = self._zone()
        tracks = [Track(track_id=1, detection=Detection(bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"))]
        now = datetime.now(tz=timezone.utc)
        events = dwell.update(zone, tracks, now)
        assert len(events) == 0
        events = dwell.update(zone, tracks, now + timedelta(seconds=15))
        assert len(events) == 1
        assert events[0]["type"] == "dwell_exceeded"

    def test_no_alert_below_threshold(self) -> None:
        dwell = DwellTracker(alert_threshold=60)
        zone = self._zone()
        tracks = [Track(track_id=1, detection=Detection(bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"))]
        now = datetime.now(tz=timezone.utc)
        dwell.update(zone, tracks, now)
        events = dwell.update(zone, tracks, now + timedelta(seconds=5))
        assert len(events) == 0

    def test_clears_when_track_leaves(self) -> None:
        dwell = DwellTracker(alert_threshold=10)
        zone = self._zone()
        inside = [Track(track_id=1, detection=Detection(bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"))]
        outside = [Track(track_id=1, detection=Detection(bbox=(500, 500, 600, 600), confidence=0.9, class_name="person"))]
        now = datetime.now(tz=timezone.utc)
        dwell.update(zone, inside, now)
        dwell.update(zone, outside, now + timedelta(seconds=5))
        events = dwell.update(zone, inside, now + timedelta(seconds=6))
        assert len(events) == 0

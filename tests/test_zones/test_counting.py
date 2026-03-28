from yolo26_analytics.models import Detection, Track
from yolo26_analytics.zones.counting import ZoneCounting
from yolo26_analytics.zones.polygon import Zone

class TestZoneCounting:
    def _zone(self) -> Zone:
        return Zone(name="test", polygon=[(0, 0), (200, 0), (200, 200), (0, 200)], track_classes=["person"])

    def test_counts_tracks_in_zone(self) -> None:
        counter = ZoneCounting()
        zone = self._zone()
        tracks = [
            Track(track_id=1, detection=Detection(bbox=(50, 50, 150, 150), confidence=0.9, class_name="person")),
            Track(track_id=2, detection=Detection(bbox=(300, 300, 400, 400), confidence=0.8, class_name="person")),
        ]
        count = counter.update(zone, tracks)
        assert count == {"person": 1}

    def test_ignores_untracked_classes(self) -> None:
        counter = ZoneCounting()
        zone = self._zone()
        tracks = [Track(track_id=1, detection=Detection(bbox=(50, 50, 150, 150), confidence=0.9, class_name="car"))]
        count = counter.update(zone, tracks)
        assert count == {}

from yolo26_analytics.models import Detection, Track
from yolo26_analytics.zones.entry_exit import EntryExitDetector
from yolo26_analytics.zones.polygon import Zone

class TestEntryExitDetector:
    def _zone(self) -> Zone:
        return Zone(name="test", polygon=[(100, 100), (300, 100), (300, 300), (100, 300)], track_classes=["person"])

    def test_detects_entry(self) -> None:
        detector = EntryExitDetector()
        zone = self._zone()
        tracks1 = [Track(track_id=1, detection=Detection(bbox=(10, 10, 60, 60), confidence=0.9, class_name="person"))]
        events = detector.update(zone, tracks1)
        assert len(events) == 0
        tracks2 = [Track(track_id=1, detection=Detection(bbox=(150, 150, 250, 250), confidence=0.9, class_name="person"))]
        events = detector.update(zone, tracks2)
        assert len(events) == 1
        assert events[0]["type"] == "entry"

    def test_detects_exit(self) -> None:
        detector = EntryExitDetector()
        zone = self._zone()
        tracks1 = [Track(track_id=1, detection=Detection(bbox=(150, 150, 250, 250), confidence=0.9, class_name="person"))]
        detector.update(zone, tracks1)
        tracks2 = [Track(track_id=1, detection=Detection(bbox=(10, 10, 60, 60), confidence=0.9, class_name="person"))]
        events = detector.update(zone, tracks2)
        assert len(events) == 1
        assert events[0]["type"] == "exit"

    def test_no_event_if_stays_inside(self) -> None:
        detector = EntryExitDetector()
        zone = self._zone()
        tracks = [Track(track_id=1, detection=Detection(bbox=(150, 150, 250, 250), confidence=0.9, class_name="person"))]
        detector.update(zone, tracks)
        events = detector.update(zone, tracks)
        assert len(events) == 0

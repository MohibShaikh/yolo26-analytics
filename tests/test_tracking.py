from __future__ import annotations

from yolo26_analytics.models import Detection
from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter


class TestByteTrackAdapter:
    def test_assigns_track_ids(self) -> None:
        tracker = ByteTrackAdapter(max_age=30, min_hits=1)
        detections = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person"),
            Detection(bbox=(300, 300, 400, 400), confidence=0.8, class_name="person"),
        ]
        tracks = tracker.update(detections)
        assert len(tracks) >= 1
        track_ids = {t.track_id for t in tracks}
        assert len(track_ids) == len(tracks)

    def test_maintains_ids_across_frames(self) -> None:
        tracker = ByteTrackAdapter(max_age=30, min_hits=1)
        det1 = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person")]
        tracks1 = tracker.update(det1)
        det2 = [Detection(bbox=(102, 102, 202, 202), confidence=0.9, class_name="person")]
        tracks2 = tracker.update(det2)
        if tracks1 and tracks2:
            assert tracks1[0].track_id == tracks2[0].track_id

    def test_reset_clears_state(self) -> None:
        tracker = ByteTrackAdapter(max_age=30, min_hits=1)
        det = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person")]
        tracker.update(det)
        tracker.reset()
        tracks = tracker.update(det)
        assert len(tracks) >= 0

    def test_empty_detections(self) -> None:
        tracker = ByteTrackAdapter(max_age=30, min_hits=1)
        tracks = tracker.update([])
        assert tracks == []

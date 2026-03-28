"""ByteTrack adapter using Ultralytics' built-in tracker."""

from __future__ import annotations

import numpy as np

from yolo26_analytics.models import Detection, Track


class _DetectionResults:
    """Wraps a numpy detection array to match the interface expected by BYTETracker.update."""

    def __init__(self, data: np.ndarray) -> None:
        # data shape: (N, 6) — [x1, y1, x2, y2, confidence, class_id]
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        return _DetectionResults(self._data[idx])

    @property
    def conf(self) -> np.ndarray:
        return self._data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self._data[:, 5]

    @property
    def xywh(self) -> np.ndarray:
        """Convert x1y1x2y2 to cx, cy, w, h."""
        x1, y1, x2, y2 = (
            self._data[:, 0],
            self._data[:, 1],
            self._data[:, 2],
            self._data[:, 3],
        )
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return np.stack([cx, cy, w, h], axis=1)

    @property
    def xyxy(self) -> np.ndarray:
        return self._data[:, :4]


def _make_args(max_age: int = 30) -> object:
    class _Args:
        track_high_thresh: float = 0.25
        track_low_thresh: float = 0.1
        new_track_thresh: float = 0.25
        track_buffer: int = max_age
        match_thresh: float = 0.8
        fuse_score: bool = False

    return _Args()


class ByteTrackAdapter:
    """Wraps Ultralytics ByteTrack for the Tracker protocol."""

    def __init__(self, max_age: int = 30, min_hits: int = 3) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._max_age = max_age
        self._min_hits = min_hits
        self._tracker = BYTETracker(args=_make_args(max_age), frame_rate=30)

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            return []

        det_array = np.array(
            [[*d.bbox, d.confidence, 0] for d in detections],
            dtype=np.float32,
        )

        results = _DetectionResults(det_array)
        online_targets = self._tracker.update(results)

        tracks: list[Track] = []
        for row in online_targets:
            # result format: [x1, y1, x2, y2, track_id, score, cls, idx]
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            track_id = int(row[4])
            score = float(row[5])
            bbox = (x1, y1, x2, y2)
            best_det = self._match_detection(bbox, detections)
            tracks.append(
                Track(
                    track_id=track_id,
                    detection=Detection(
                        bbox=bbox,
                        confidence=best_det.confidence if best_det else score,
                        class_name=best_det.class_name if best_det else "unknown",
                    ),
                )
            )
        return tracks

    @staticmethod
    def _match_detection(
        bbox: tuple[int, int, int, int], detections: list[Detection]
    ) -> Detection | None:
        best: Detection | None = None
        best_iou = 0.0
        bx1, by1, bx2, by2 = bbox
        for det in detections:
            dx1, dy1, dx2, dy2 = det.bbox
            ix1 = max(bx1, dx1)
            iy1 = max(by1, dy1)
            ix2 = min(bx2, dx2)
            iy2 = min(by2, dy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_b = (bx2 - bx1) * (by2 - by1)
            area_d = (dx2 - dx1) * (dy2 - dy1)
            union = area_b + area_d - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best = det
        return best

    def reset(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(args=_make_args(self._max_age), frame_rate=30)

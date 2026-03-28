from __future__ import annotations

from datetime import datetime

from yolo26_analytics.models import Track
from yolo26_analytics.zones.polygon import Zone


class DwellTracker:
    def __init__(self, alert_threshold: int = 300) -> None:
        self._threshold = alert_threshold
        self._enter_times: dict[str, dict[int, datetime]] = {}
        self._alerted: dict[str, set[int]] = {}

    def update(self, zone: Zone, tracks: list[Track], now: datetime) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        zone_enters = self._enter_times.setdefault(zone.name, {})
        zone_alerted = self._alerted.setdefault(zone.name, set())
        current_inside: set[int] = set()
        for track in tracks:
            if not zone.should_track(track.class_name):
                continue
            cx, cy = track.centroid
            if not zone.contains_point(cx, cy):
                zone_enters.pop(track.track_id, None)
                zone_alerted.discard(track.track_id)
                continue
            current_inside.add(track.track_id)
            if track.track_id not in zone_enters:
                zone_enters[track.track_id] = now
                continue
            dwell_secs = (now - zone_enters[track.track_id]).total_seconds()
            if dwell_secs >= self._threshold and track.track_id not in zone_alerted:
                zone_alerted.add(track.track_id)
                events.append(
                    {
                        "type": "dwell_exceeded",
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "dwell_seconds": dwell_secs,
                    }
                )
        for tid in list(zone_enters.keys()):
            if tid not in current_inside:
                zone_enters.pop(tid, None)
                zone_alerted.discard(tid)
        return events

from __future__ import annotations
from yolo26_analytics.models import Track
from yolo26_analytics.zones.polygon import Zone

class EntryExitDetector:
    def __init__(self) -> None:
        self._prev_state: dict[str, dict[int, bool]] = {}

    def update(self, zone: Zone, tracks: list[Track]) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        zone_state = self._prev_state.setdefault(zone.name, {})
        current_ids: set[int] = set()
        for track in tracks:
            if not zone.should_track(track.class_name):
                continue
            current_ids.add(track.track_id)
            cx, cy = track.centroid
            is_inside = zone.contains_point(cx, cy)
            was_inside = zone_state.get(track.track_id)
            if was_inside is not None:
                if not was_inside and is_inside:
                    events.append({"type": "entry", "track_id": track.track_id, "class_name": track.class_name})
                elif was_inside and not is_inside:
                    events.append({"type": "exit", "track_id": track.track_id, "class_name": track.class_name})
            zone_state[track.track_id] = is_inside
        for tid in list(zone_state.keys()):
            if tid not in current_ids:
                del zone_state[tid]
        return events

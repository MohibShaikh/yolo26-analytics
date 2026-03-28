from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from yolo26_analytics.models import Track
from yolo26_analytics.zones.polygon import Zone


@dataclass
class _WindowState:
    window_start: datetime
    unique_ids: set[int] = field(default_factory=set)


class ThroughputTracker:
    def __init__(self, interval: int = 3600) -> None:
        self._interval = timedelta(seconds=interval)
        self._windows: dict[str, _WindowState] = {}

    def update(self, zone: Zone, tracks: list[Track], now: datetime) -> None:
        state = self._windows.get(zone.name)
        if state is None or (now - state.window_start) >= self._interval:
            self._windows[zone.name] = _WindowState(window_start=now)
            state = self._windows[zone.name]
        for track in tracks:
            if not zone.should_track(track.class_name):
                continue
            cx, cy = track.centroid
            if zone.contains_point(cx, cy):
                state.unique_ids.add(track.track_id)

    def get_stats(self, zone_name: str) -> dict[str, object]:
        state = self._windows.get(zone_name)
        if state is None:
            return {"unique_tracks": 0, "window_start": None}
        return {"unique_tracks": len(state.unique_ids), "window_start": state.window_start}

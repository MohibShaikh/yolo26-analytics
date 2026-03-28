from __future__ import annotations

from collections import defaultdict

from yolo26_analytics.models import Track
from yolo26_analytics.zones.polygon import Zone


class ZoneCounting:
    def update(self, zone: Zone, tracks: list[Track]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for track in tracks:
            if not zone.should_track(track.class_name):
                continue
            cx, cy = track.centroid
            if zone.contains_point(cx, cy):
                counts[track.class_name] += 1
        return dict(counts)

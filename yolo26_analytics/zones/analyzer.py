from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from yolo26_analytics.config.schema import ZoneConfig
from yolo26_analytics.models import Event, Track
from yolo26_analytics.zones.counting import ZoneCounting
from yolo26_analytics.zones.dwell import DwellTracker
from yolo26_analytics.zones.entry_exit import EntryExitDetector
from yolo26_analytics.zones.polygon import Zone
from yolo26_analytics.zones.throughput import ThroughputTracker


class ZoneAnalyzer:
    def __init__(self, zone_configs: list[ZoneConfig]) -> None:
        self._zones: list[Zone] = []
        self._counters: dict[str, ZoneCounting] = {}
        self._entry_exit: dict[str, EntryExitDetector] = {}
        self._dwell: dict[str, DwellTracker] = {}
        self._throughput: dict[str, ThroughputTracker] = {}
        self._cooldowns: dict[tuple[int, str, str], datetime] = {}
        for zc in zone_configs:
            polygon_tuples = [(p[0], p[1]) for p in zc.polygon]
            zone = Zone(
                name=zc.name,
                polygon=polygon_tuples,
                track_classes=zc.track_classes,
                cooldown=zc.cooldown,
            )
            self._zones.append(zone)
            for rule in zc.analytics:
                match rule.type:
                    case "count":
                        self._counters[zc.name] = ZoneCounting()
                    case "entry_exit":
                        self._entry_exit[zc.name] = EntryExitDetector()
                    case "dwell":
                        self._dwell[zc.name] = DwellTracker(
                            alert_threshold=rule.alert_threshold or 300
                        )
                    case "throughput":
                        self._throughput[zc.name] = ThroughputTracker(
                            interval=rule.interval or 3600
                        )

    def check(self, tracks: list[Track]) -> list[Event]:
        now = datetime.now(tz=timezone.utc)
        events: list[Event] = []
        for zone in self._zones:
            if zone.name in self._counters:
                self._counters[zone.name].update(zone, tracks)
            if zone.name in self._entry_exit:
                for ee in self._entry_exit[zone.name].update(zone, tracks):
                    event = self._make_event(
                        event_type=str(ee["type"]),
                        zone_name=zone.name,
                        track_id=int(ee["track_id"]),
                        object_class=str(ee["class_name"]),
                        metadata=ee,
                        now=now,
                        cooldown=zone.cooldown,
                    )
                    if event:
                        events.append(event)
            if zone.name in self._dwell:
                for de in self._dwell[zone.name].update(zone, tracks, now):
                    event = self._make_event(
                        event_type="dwell_exceeded",
                        zone_name=zone.name,
                        track_id=int(de["track_id"]),
                        object_class=str(de["class_name"]),
                        metadata=de,
                        now=now,
                        cooldown=zone.cooldown,
                    )
                    if event:
                        events.append(event)
            if zone.name in self._throughput:
                self._throughput[zone.name].update(zone, tracks, now)
        return events

    def get_zone_counts(self) -> dict[str, dict[str, int]]:
        return {z.name: {} for z in self._zones if z.name in self._counters}

    def get_throughput_stats(self) -> dict[str, dict[str, object]]:
        return {name: tp.get_stats(name) for name, tp in self._throughput.items()}

    @property
    def zones(self) -> list[Zone]:
        return self._zones

    def _make_event(
        self,
        event_type: str,
        zone_name: str,
        track_id: int,
        object_class: str,
        metadata: dict[str, Any],
        now: datetime,
        cooldown: int,
    ) -> Event | None:
        key = (track_id, zone_name, event_type)
        last = self._cooldowns.get(key)
        if last is not None and (now - last).total_seconds() < cooldown:
            return None
        self._cooldowns[key] = now
        return Event(
            timestamp=now,
            event_type=event_type,
            zone_name=zone_name,
            track_id=track_id,
            object_class=object_class,
            metadata=metadata,
            confidence=0.0,
            frame_snapshot=b"",
            bbox=(0, 0, 0, 0),
        )

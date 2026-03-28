"""Zone statistics aggregation."""

from __future__ import annotations


class ZoneStatsAggregator:
    """Aggregates zone statistics for the dashboard and API."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._entries: dict[str, int] = {}
        self._exits: dict[str, int] = {}

    def update_counts(self, zone_name: str, counts: dict[str, int]) -> None:
        self._counts[zone_name] = counts

    def record_entry(self, zone_name: str) -> None:
        self._entries[zone_name] = self._entries.get(zone_name, 0) + 1

    def record_exit(self, zone_name: str) -> None:
        self._exits[zone_name] = self._exits.get(zone_name, 0) + 1

    def get_stats(self, zone_name: str) -> dict[str, object]:
        return {
            "counts": self._counts.get(zone_name, {}),
            "total_entries": self._entries.get(zone_name, 0),
            "total_exits": self._exits.get(zone_name, 0),
        }

    def get_all_stats(self) -> dict[str, dict[str, object]]:
        all_zones = set(self._counts.keys()) | set(self._entries.keys()) | set(self._exits.keys())
        return {zone: self.get_stats(zone) for zone in all_zones}

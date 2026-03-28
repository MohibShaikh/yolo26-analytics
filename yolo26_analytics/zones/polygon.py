from __future__ import annotations

from shapely.geometry import Point, Polygon


class Zone:
    def __init__(
        self,
        name: str,
        polygon: list[tuple[int, int]],
        track_classes: list[str],
        cooldown: int = 30,
    ) -> None:
        self.name = name
        self.track_classes = track_classes
        self.cooldown = cooldown
        self._polygon = Polygon(polygon)

    def contains_point(self, x: int, y: int) -> bool:
        return self._polygon.contains(Point(x, y))

    def should_track(self, class_name: str) -> bool:
        return class_name in self.track_classes

    @property
    def polygon(self) -> Polygon:
        return self._polygon

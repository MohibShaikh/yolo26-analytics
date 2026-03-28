from yolo26_analytics.zones.polygon import Zone


class TestZone:
    def test_point_inside(self) -> None:
        zone = Zone(
            name="test", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], track_classes=["person"]
        )
        assert zone.contains_point(50, 50)

    def test_point_outside(self) -> None:
        zone = Zone(
            name="test", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], track_classes=["person"]
        )
        assert not zone.contains_point(150, 50)

    def test_filters_by_class(self) -> None:
        zone = Zone(
            name="test", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], track_classes=["person"]
        )
        assert zone.should_track("person")
        assert not zone.should_track("car")

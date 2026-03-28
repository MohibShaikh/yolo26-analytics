from __future__ import annotations
from datetime import datetime, timezone
import pytest
from yolo26_analytics.alerts.manager import AlertManager
from yolo26_analytics.config.schema import AlertFilterConfig
from yolo26_analytics.models import Event

def _make_event(zone: str = "Dock", event_type: str = "entry") -> Event:
    return Event(timestamp=datetime.now(tz=timezone.utc), event_type=event_type,
        zone_name=zone, track_id=1, object_class="person", metadata={},
        confidence=0.9, frame_snapshot=b"", bbox=(0, 0, 0, 0))

class FakeBackend:
    def __init__(self) -> None:
        self.received: list[Event] = []
    async def send(self, event: Event) -> None:
        self.received.append(event)

class TestAlertManager:
    @pytest.mark.asyncio
    async def test_dispatches_to_all_backends(self) -> None:
        b1 = FakeBackend()
        b2 = FakeBackend()
        manager = AlertManager(backends=[(b1, None), (b2, None)])
        await manager.dispatch([_make_event()])
        assert len(b1.received) == 1
        assert len(b2.received) == 1

    @pytest.mark.asyncio
    async def test_filters_by_zone(self) -> None:
        b = FakeBackend()
        filt = AlertFilterConfig(zones=["Dock"], event_types=[])
        manager = AlertManager(backends=[(b, filt)])
        await manager.dispatch([_make_event(zone="Dock")])
        await manager.dispatch([_make_event(zone="Other")])
        assert len(b.received) == 1

    @pytest.mark.asyncio
    async def test_filters_by_event_type(self) -> None:
        b = FakeBackend()
        filt = AlertFilterConfig(zones=[], event_types=["dwell_exceeded"])
        manager = AlertManager(backends=[(b, filt)])
        await manager.dispatch([_make_event(event_type="entry")])
        await manager.dispatch([_make_event(event_type="dwell_exceeded")])
        assert len(b.received) == 1
        assert b.received[0].event_type == "dwell_exceeded"

    @pytest.mark.asyncio
    async def test_no_filter_receives_all(self) -> None:
        b = FakeBackend()
        manager = AlertManager(backends=[(b, None)])
        await manager.dispatch([_make_event(zone="A"), _make_event(zone="B")])
        assert len(b.received) == 2

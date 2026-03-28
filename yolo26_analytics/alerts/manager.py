from __future__ import annotations
from yolo26_analytics.config.schema import AlertFilterConfig
from yolo26_analytics.models import Event
from yolo26_analytics.protocols import AlertBackend

class AlertManager:
    def __init__(self, backends: list[tuple[AlertBackend, AlertFilterConfig | None]]) -> None:
        self._backends = backends

    async def dispatch(self, events: list[Event]) -> None:
        for event in events:
            for backend, filt in self._backends:
                if self._matches_filter(event, filt):
                    await backend.send(event)

    @staticmethod
    def _matches_filter(event: Event, filt: AlertFilterConfig | None) -> bool:
        if filt is None:
            return True
        if filt.zones and event.zone_name not in filt.zones:
            return False
        if filt.event_types and event.event_type not in filt.event_types:
            return False
        return True

from __future__ import annotations
from yolo26_analytics.models import Event

class ConsoleAlert:
    async def send(self, event: Event) -> None:
        print(f"[ALERT] {event.timestamp.isoformat()} | {event.event_type} | zone={event.zone_name} | track={event.track_id} | class={event.object_class} | meta={event.metadata}")

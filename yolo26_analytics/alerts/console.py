from __future__ import annotations

from yolo26_analytics.models import Event


class ConsoleAlert:
    async def send(self, event: Event) -> None:
        ts = event.timestamp.isoformat()
        print(
            f"[ALERT] {ts} | {event.event_type} | zone={event.zone_name}"
            f" | track={event.track_id} | class={event.object_class}"
            f" | meta={event.metadata}"
        )

"""Discord webhook alert backend."""

from __future__ import annotations

import httpx

from yolo26_analytics.models import Event


class DiscordAlert:
    """Sends alerts to a Discord channel via webhook."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url
        self._client = httpx.AsyncClient(timeout=10.0)

    async def send(self, event: Event) -> None:
        fields: list[dict[str, object]] = [
            {"name": "Zone", "value": event.zone_name, "inline": True},
            {"name": "Class", "value": event.object_class, "inline": True},
            {"name": "Track ID", "value": str(event.track_id), "inline": True},
            {
                "name": "Time",
                "value": event.timestamp.strftime("%H:%M:%S"),
                "inline": True,
            },
        ]
        if event.metadata:
            for k, v in event.metadata.items():
                fields.append({"name": k, "value": str(v), "inline": True})
        embed: dict[str, object] = {
            "title": event.event_type.replace("_", " ").upper(),
            "color": 0xFF4444,
            "fields": fields,
        }
        payload: dict[str, object] = {"embeds": [embed]}
        await self._client.post(self._url, json=payload)

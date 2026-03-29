"""Slack webhook alert backend."""

from __future__ import annotations

import httpx

from yolo26_analytics.models import Event


class SlackAlert:
    """Sends alerts to a Slack channel via incoming webhook."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url
        self._client = httpx.AsyncClient(timeout=10.0)

    async def send(self, event: Event) -> None:
        text = (
            f":rotating_light: *{event.event_type.replace('_', ' ').upper()}*\n"
            f">*Zone:* {event.zone_name}\n"
            f">*Class:* {event.object_class}\n"
            f">*Track ID:* {event.track_id}\n"
            f">*Time:* {event.timestamp.strftime('%H:%M:%S')}"
        )
        if event.metadata:
            meta_lines = "\n".join(
                f">*{k}:* {v}" for k, v in event.metadata.items()
            )
            text += f"\n{meta_lines}"
        payload = {"text": text}
        await self._client.post(self._url, json=payload)

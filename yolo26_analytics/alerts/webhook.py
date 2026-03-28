from __future__ import annotations

import base64

import httpx

from yolo26_analytics.models import Event


class WebhookAlert:
    def __init__(self, url: str) -> None:
        self._url = url
        self._client = httpx.AsyncClient(timeout=10.0)

    async def send(self, event: Event) -> None:
        payload = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "zone_name": event.zone_name,
            "track_id": event.track_id,
            "object_class": event.object_class,
            "metadata": event.metadata,
            "confidence": event.confidence,
            "bbox": list(event.bbox),
        }
        if event.frame_snapshot:
            payload["snapshot_base64"] = base64.b64encode(event.frame_snapshot).decode()
        await self._client.post(self._url, json=payload)

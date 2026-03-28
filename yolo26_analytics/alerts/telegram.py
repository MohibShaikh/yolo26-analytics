from __future__ import annotations

import httpx

from yolo26_analytics.models import Event


class TelegramAlert:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._client = httpx.AsyncClient(timeout=10.0)
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

    async def send(self, event: Event) -> None:
        text = (
            f"ALERT: {event.event_type.replace('_', ' ').upper()}\n"
            f"Zone: {event.zone_name}\nClass: {event.object_class}\n"
            f"Track: {event.track_id}\nTime: {event.timestamp.strftime('%H:%M:%S')}\n"
        )
        if event.metadata:
            for k, v in event.metadata.items():
                text += f"{k}: {v}\n"
        if event.frame_snapshot:
            await self._client.post(
                f"{self._base_url}/sendPhoto",
                data={"chat_id": self._chat_id, "caption": text},
                files={"photo": ("snapshot.jpg", event.frame_snapshot, "image/jpeg")},
            )
        else:
            await self._client.post(
                f"{self._base_url}/sendMessage", json={"chat_id": self._chat_id, "text": text}
            )

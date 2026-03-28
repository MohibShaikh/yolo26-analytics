from __future__ import annotations
import json
import aiomqtt
from yolo26_analytics.models import Event

class MQTTAlert:
    def __init__(self, broker: str, topic: str = "yolo26/events") -> None:
        broker = broker.replace("mqtt://", "")
        parts = broker.split(":")
        self._host = parts[0]
        self._port = int(parts[1]) if len(parts) > 1 else 1883
        self._topic = topic

    async def send(self, event: Event) -> None:
        payload = json.dumps({"timestamp": event.timestamp.isoformat(), "event_type": event.event_type,
            "zone_name": event.zone_name, "track_id": event.track_id,
            "object_class": event.object_class, "metadata": event.metadata, "confidence": event.confidence})
        async with aiomqtt.Client(self._host, self._port) as client:
            await client.publish(self._topic, payload.encode())

"""Protocol definitions for all pipeline stages."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

import numpy.typing as npt

from yolo26_analytics.models import Detection, Event, FrameMeta, Track


@runtime_checkable
class VideoSource(Protocol):
    """Yields video frames."""

    async def __aiter__(self) -> AsyncIterator[tuple[npt.NDArray[Any], FrameMeta]]: ...

    async def close(self) -> None: ...


@runtime_checkable
class Detector(Protocol):
    """Runs object detection on a frame."""

    def predict(self, frame: npt.NDArray[Any]) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Assigns persistent track IDs to detections across frames."""

    def update(self, detections: list[Detection]) -> list[Track]: ...

    def reset(self) -> None: ...


@runtime_checkable
class TrackStore(Protocol):
    """Persists track updates."""

    async def write_tracks(self, tracks: list[Track], meta: FrameMeta) -> None: ...

    async def query_tracks(
        self,
        source_id: str | None = None,
        zone_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, object]]: ...


@runtime_checkable
class EventStore(Protocol):
    """Persists zone-generated events."""

    async def log_events(self, events: list[Event]) -> None: ...

    async def query_events(
        self,
        zone_name: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, object]]: ...


@runtime_checkable
class AlertBackend(Protocol):
    """Sends an alert for a zone event."""

    async def send(self, event: Event) -> None: ...

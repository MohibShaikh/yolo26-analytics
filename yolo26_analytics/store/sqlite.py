"""SQLite store — dev fallback. Implements both TrackStore and EventStore."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from yolo26_analytics.models import Event, FrameMeta, Track
from yolo26_analytics.store.models import Base, EventRow, TrackRow


class SQLiteStore:
    """Combined TrackStore + EventStore backed by SQLite."""

    def __init__(self, path: str = "./data/yolo26_analytics.db") -> None:
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def write_tracks(
        self, tracks: list[Track], meta: FrameMeta, zone_name: str | None = None
    ) -> None:
        async with self._session_factory() as session:
            for track in tracks:
                cx, cy = track.centroid
                x1, y1, x2, y2 = track.bbox
                row = TrackRow(
                    timestamp=meta.timestamp,
                    track_id=track.track_id,
                    object_class=track.class_name,
                    confidence=track.confidence,
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    centroid_x=cx,
                    centroid_y=cy,
                    zone_name=zone_name,
                    source_id=meta.source_id,
                )
                session.add(row)
            await session.commit()

    async def query_tracks(
        self,
        source_id: str | None = None,
        zone_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        async with self._session_factory() as session:
            stmt = select(TrackRow).order_by(TrackRow.timestamp).limit(limit)
            if source_id is not None:
                stmt = stmt.where(TrackRow.source_id == source_id)
            if zone_name is not None:
                stmt = stmt.where(TrackRow.zone_name == zone_name)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "track_id": r.track_id,
                    "object_class": r.object_class,
                    "confidence": r.confidence,
                    "bbox": (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2),
                    "centroid": (r.centroid_x, r.centroid_y),
                    "zone_name": r.zone_name,
                    "source_id": r.source_id,
                }
                for r in rows
            ]

    async def log_events(self, events: list[Event]) -> None:
        async with self._session_factory() as session:
            for event in events:
                row = EventRow(
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    zone_name=event.zone_name,
                    track_id=event.track_id,
                    object_class=event.object_class,
                    metadata_json=json.dumps(event.metadata),
                    confidence=event.confidence,
                    snapshot_path=None,
                )
                session.add(row)
            await session.commit()

    async def query_events(
        self,
        zone_name: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        async with self._session_factory() as session:
            stmt = select(EventRow).order_by(EventRow.timestamp).limit(limit)
            if zone_name is not None:
                stmt = stmt.where(EventRow.zone_name == zone_name)
            if event_type is not None:
                stmt = stmt.where(EventRow.event_type == event_type)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "event_type": r.event_type,
                    "zone_name": r.zone_name,
                    "track_id": r.track_id,
                    "object_class": r.object_class,
                    "metadata": json.loads(r.metadata_json) if r.metadata_json else {},
                    "confidence": r.confidence,
                    "snapshot_path": r.snapshot_path,
                }
                for r in rows
            ]

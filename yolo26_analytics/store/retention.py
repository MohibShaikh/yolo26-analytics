"""Periodic retention cleanup for old tracks, events, and snapshots."""
from __future__ import annotations
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker
from yolo26_analytics.store.models import EventRow, TrackRow


def parse_duration(s: str) -> timedelta:
    match = re.match(r"^(\d+)([dhms])$", s)
    if not match:
        raise ValueError(f"Invalid duration format: {s}")
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "d": return timedelta(days=value)
    if unit == "h": return timedelta(hours=value)
    if unit == "m": return timedelta(minutes=value)
    return timedelta(seconds=value)


async def run_retention_cleanup(session_factory: async_sessionmaker,
    tracks_retention: str = "7d", events_retention: str = "90d",
    snapshots_retention: str = "30d", snapshots_dir: str = "data/snapshots") -> None:
    now = datetime.now(tz=timezone.utc)
    async with session_factory() as session:
        cutoff = now - parse_duration(tracks_retention)
        await session.execute(delete(TrackRow).where(TrackRow.timestamp < cutoff))
        cutoff = now - parse_duration(events_retention)
        await session.execute(delete(EventRow).where(EventRow.timestamp < cutoff))
        await session.commit()
    snap_cutoff = now - parse_duration(snapshots_retention)
    snap_dir = Path(snapshots_dir)
    if snap_dir.exists():
        for date_dir in snap_dir.iterdir():
            if date_dir.is_dir():
                for f in date_dir.iterdir():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                    if mtime < snap_cutoff:
                        f.unlink()
                if not any(date_dir.iterdir()):
                    date_dir.rmdir()

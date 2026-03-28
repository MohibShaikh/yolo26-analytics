"""SQLAlchemy ORM models for tracks, events, and zone_stats."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TrackRow(Base):
    __tablename__ = "tracks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    object_class: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[int] = mapped_column(Integer)
    bbox_y1: Mapped[int] = mapped_column(Integer)
    bbox_x2: Mapped[int] = mapped_column(Integer)
    bbox_y2: Mapped[int] = mapped_column(Integer)
    centroid_x: Mapped[int] = mapped_column(Integer)
    centroid_y: Mapped[int] = mapped_column(Integer)
    zone_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    source_id: Mapped[str] = mapped_column(String(100), nullable=False)


class EventRow(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    zone_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    object_class: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(255), nullable=True)


class ZoneStatRow(Base):
    __tablename__ = "zone_stats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    zone_name: Mapped[str] = mapped_column(String(100), nullable=False)
    object_class: Mapped[str] = mapped_column(String(50), nullable=False)
    current_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entries_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exits_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_dwell_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

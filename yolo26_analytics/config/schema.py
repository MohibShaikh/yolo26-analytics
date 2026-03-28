"""Configuration schema and YAML loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class SourceConfig(BaseModel):
    type: str = "video_file"
    path: str | None = None
    url: str | None = None
    device: int | None = None


class SAHIConfig(BaseModel):
    enabled: bool = False
    slice_size: int = 640
    overlap: float = 0.25


class ModelConfig(BaseModel):
    type: str = "yolo26"
    weights: str = "yolo26n.pt"
    confidence: float = 0.5
    sahi: SAHIConfig | None = None


class TrackingConfig(BaseModel):
    engine: str = "bytetrack"
    max_age: int = 30
    min_hits: int = 3


class RetentionConfig(BaseModel):
    tracks: str = "7d"
    events: str = "90d"
    snapshots: str = "30d"


class StoreConfig(BaseModel):
    type: str = "sqlite"
    url: str | None = None
    path: str = "./data/yolo26_analytics.db"
    retention: RetentionConfig = RetentionConfig()


class ZoneAnalyticsRule(BaseModel):
    type: str  # "count", "dwell", "entry_exit", "throughput"
    alert_threshold: int | None = None
    direction: str | None = None
    interval: int | None = None


class ZoneConfig(BaseModel):
    name: str
    polygon: list[list[int]]
    track_classes: list[str] = ["person"]
    analytics: list[ZoneAnalyticsRule] = []
    cooldown: int = 30

    @field_validator("polygon")
    @classmethod
    def polygon_must_have_3_points(cls, v: list[list[int]]) -> list[list[int]]:
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 points")
        return v


class AlertFilterConfig(BaseModel):
    zones: list[str] = []
    event_types: list[str] = []


class AlertConfig(BaseModel):
    type: str  # "console", "webhook", "mqtt", "telegram"
    url: str | None = None
    bot_token: str | None = None
    chat_id: str | None = None
    broker: str | None = None
    topic: str | None = None
    filter: AlertFilterConfig | None = None


class AppConfig(BaseModel):
    source: SourceConfig
    model: ModelConfig = ModelConfig()
    tracking: TrackingConfig = TrackingConfig()
    store: StoreConfig = StoreConfig()
    zones: list[ZoneConfig] = []
    alerts: list[AlertConfig] = []
    dashboard: bool = False


def load_config(path: str) -> AppConfig:
    """Load and validate configuration from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(raw)

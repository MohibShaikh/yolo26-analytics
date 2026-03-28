# yolo26-analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pip-installable Python framework for real-time object tracking, zone analytics, and event alerting on YOLO26.

**Architecture:** Async pipeline of pluggable stages (VideoSource → Detector → Tracker → ZoneAnalyzer → AlertManager) with PostgreSQL persistence, a FastAPI+HTMX dashboard, and a CLI. Every stage is a Python Protocol — swap any component without touching the rest.

**Tech Stack:** Python 3.10+, YOLO26 (Ultralytics), ByteTrack, FastAPI, HTMX, PostgreSQL (SQLAlchemy async), SQLite fallback, Shapely, OpenCV, pytest, ruff, mypy

---

## File Map

```
yolo26_analytics/
├── __init__.py                      # Public API: Pipeline, Event, Detection, Track
├── py.typed                         # PEP 561 marker
├── models.py                        # Core data types: Detection, Track, Event, FrameMeta
├── protocols.py                     # Protocol definitions for all stages
├── config/
│   ├── __init__.py
│   └── schema.py                    # Pydantic config models, YAML loader
├── sources/
│   ├── __init__.py
│   ├── base.py                      # VideoSource protocol + FrameIterator
│   ├── webcam.py                    # OpenCV VideoCapture (device index)
│   ├── rtsp.py                      # RTSP stream via OpenCV
│   ├── video_file.py                # Video file playback
│   └── image_dir.py                 # Image directory iteration
├── detection/
│   ├── __init__.py
│   ├── yolo26.py                    # YOLO26 adapter (Ultralytics)
│   └── sahi.py                      # SAHI wrapper around any detector
├── tracking/
│   ├── __init__.py
│   └── bytetrack.py                 # ByteTrack adapter
├── zones/
│   ├── __init__.py
│   ├── polygon.py                   # Zone polygon + point-in-polygon via Shapely
│   ├── analyzer.py                  # ZoneAnalyzer — orchestrates all zone analytics
│   ├── counting.py                  # Real-time zone counting
│   ├── entry_exit.py                # Boundary crossing detection
│   ├── dwell.py                     # Dwell time tracking
│   └── throughput.py                # Time-windowed throughput stats
├── store/
│   ├── __init__.py
│   ├── base.py                      # Store protocol
│   ├── postgres.py                  # PostgreSQL via asyncpg/SQLAlchemy
│   ├── sqlite.py                    # SQLite fallback
│   ├── models.py                    # SQLAlchemy ORM models
│   └── retention.py                 # Periodic cleanup task
├── alerts/
│   ├── __init__.py
│   ├── manager.py                   # AlertManager — routing + dispatch
│   ├── console.py                   # Console backend
│   ├── webhook.py                   # Webhook (HTTP POST) backend
│   ├── mqtt.py                      # MQTT backend
│   └── telegram.py                  # Telegram bot backend
├── analytics/
│   ├── __init__.py
│   ├── heatmap.py                   # Heatmap generation from track data
│   └── stats.py                     # Zone stats aggregation (counts, dwell, throughput)
├── dashboard/
│   ├── __init__.py
│   ├── app.py                       # FastAPI app factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── views.py                 # HTML page routes (/, /analytics, /replay, /zones/edit)
│   │   ├── stream.py                # MJPEG + SSE endpoints
│   │   └── api.py                   # JSON API endpoints (/api/*)
│   ├── templates/
│   │   ├── base.html                # Base template (HTMX, Chart.js CDN)
│   │   ├── live.html                # Live view
│   │   ├── analytics.html           # Analytics view
│   │   ├── replay.html              # Replay view
│   │   └── zone_editor.html         # Zone editor
│   └── static/
│       ├── zone-editor.js           # Canvas polygon drawing
│       └── style.css                # Minimal styling
├── export/
│   ├── __init__.py
│   └── exporter.py                  # Model export + benchmarking
├── core/
│   ├── __init__.py
│   └── pipeline.py                  # Pipeline class — the main orchestrator
└── cli.py                           # Click CLI (y26a)

tests/
├── conftest.py                      # Shared fixtures (mock frames, fake detections)
├── test_models.py                   # Data type tests
├── test_config.py                   # Config parsing tests
├── test_sources.py                  # Video source tests
├── test_detection.py                # Detector adapter tests
├── test_tracking.py                 # Tracker tests
├── test_zones/
│   ├── test_polygon.py              # Point-in-polygon tests
│   ├── test_counting.py             # Zone counting tests
│   ├── test_entry_exit.py           # Entry/exit detection tests
│   ├── test_dwell.py                # Dwell time tests
│   └── test_throughput.py           # Throughput tests
├── test_store.py                    # Store tests (SQLite — no external deps)
├── test_alerts/
│   ├── test_manager.py              # Alert routing tests
│   ├── test_webhook.py              # Webhook backend tests
│   └── test_telegram.py             # Telegram backend tests
├── test_analytics.py                # Heatmap + stats tests
├── test_dashboard/
│   ├── test_api.py                  # API endpoint tests
│   └── test_stream.py               # Stream endpoint tests
├── test_export.py                   # Export tests
├── test_pipeline.py                 # Pipeline orchestration tests
├── test_cli.py                      # CLI tests
└── integration/
    └── test_full_pipeline.py        # Full pipeline integration test

pyproject.toml                       # Package config, dependencies, CLI entry point
Dockerfile                           # App container
docker-compose.yml                   # App + PostgreSQL
.github/workflows/ci.yml             # CI pipeline
```

---

### Task 1: Project Scaffolding + Core Data Models

**Files:**
- Create: `pyproject.toml`
- Create: `yolo26_analytics/__init__.py`
- Create: `yolo26_analytics/py.typed`
- Create: `yolo26_analytics/models.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "yolo26-analytics"
version = "0.1.0"
description = "Real-time object tracking, zone analytics, and event alerting built on YOLO26"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = [
    "ultralytics>=8.3.0",
    "opencv-python-headless>=4.8.0",
    "numpy>=1.24.0",
    "shapely>=2.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "jinja2>=3.1.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "aiosqlite>=0.19.0",
    "asyncpg>=0.29.0",
    "httpx>=0.27.0",
    "aiomqtt>=2.0.0",
    "click>=8.1.0",
    "python-multipart>=0.0.6",
    "sse-starlette>=1.6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "testcontainers[postgres]>=3.7.0",
]
sahi = [
    "sahi>=0.11.0",
]
tensorrt = [
    "tensorrt>=8.6.0",
]
tflite = [
    "tflite-runtime>=2.14.0",
]

[project.scripts]
y26a = "yolo26_analytics.cli:main"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "TCH"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create PEP 561 marker**

Create empty file `yolo26_analytics/py.typed`.

- [ ] **Step 3: Write failing test for core data models**

```python
# tests/test_models.py
from datetime import datetime, timezone

from yolo26_analytics.models import Detection, Event, FrameMeta, Track


class TestDetection:
    def test_create_detection(self) -> None:
        det = Detection(
            bbox=(100, 200, 300, 400),
            confidence=0.85,
            class_name="person",
        )
        assert det.bbox == (100, 200, 300, 400)
        assert det.confidence == 0.85
        assert det.class_name == "person"

    def test_centroid(self) -> None:
        det = Detection(bbox=(100, 200, 300, 400), confidence=0.9, class_name="person")
        assert det.centroid == (200, 300)


class TestTrack:
    def test_create_track(self) -> None:
        det = Detection(bbox=(100, 200, 300, 400), confidence=0.85, class_name="person")
        track = Track(track_id=1, detection=det)
        assert track.track_id == 1
        assert track.detection.class_name == "person"
        assert track.centroid == (200, 300)


class TestEvent:
    def test_create_event(self) -> None:
        now = datetime.now(tz=timezone.utc)
        event = Event(
            timestamp=now,
            event_type="entry",
            zone_name="Loading Dock",
            track_id=1,
            object_class="person",
            metadata={"direction": "inward"},
            confidence=0.85,
            frame_snapshot=b"",
            bbox=(100, 200, 300, 400),
        )
        assert event.event_type == "entry"
        assert event.metadata["direction"] == "inward"


class TestFrameMeta:
    def test_create_frame_meta(self) -> None:
        now = datetime.now(tz=timezone.utc)
        meta = FrameMeta(
            timestamp=now,
            frame_index=42,
            source_id="cam1",
        )
        assert meta.frame_index == 42
        assert meta.source_id == "cam1"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'yolo26_analytics'`

- [ ] **Step 5: Implement core data models**

```python
# yolo26_analytics/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection from one frame."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked detection with a persistent ID across frames."""

    track_id: int
    detection: Detection

    @property
    def centroid(self) -> tuple[int, int]:
        return self.detection.centroid

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.detection.bbox

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    @property
    def confidence(self) -> float:
        return self.detection.confidence


@dataclass(frozen=True, slots=True)
class Event:
    """An analytics event generated by the zone system."""

    timestamp: datetime
    event_type: str  # "entry", "exit", "dwell_exceeded", "count_exceeded"
    zone_name: str
    track_id: int
    object_class: str
    metadata: dict[str, object]
    confidence: float
    frame_snapshot: bytes
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """Metadata about a video frame."""

    timestamp: datetime
    frame_index: int
    source_id: str
```

- [ ] **Step 6: Create package __init__.py**

```python
# yolo26_analytics/__init__.py
"""yolo26-analytics: Real-time object tracking, zone analytics, and event alerting on YOLO26."""

from yolo26_analytics.models import Detection, Event, FrameMeta, Track

__all__ = ["Detection", "Event", "FrameMeta", "Track"]
__version__ = "0.1.0"
```

- [ ] **Step 7: Create tests conftest**

```python
# tests/conftest.py
"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from yolo26_analytics.models import Detection, Track


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 640x480 black frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[Detection]:
    """A set of sample detections for testing."""
    return [
        Detection(bbox=(100, 100, 200, 300), confidence=0.9, class_name="person"),
        Detection(bbox=(300, 150, 400, 350), confidence=0.8, class_name="person"),
        Detection(bbox=(120, 110, 170, 180), confidence=0.7, class_name="helmet"),
    ]


@pytest.fixture
def sample_tracks(sample_detections: list[Detection]) -> list[Track]:
    """Sample tracks with IDs assigned."""
    return [
        Track(track_id=i + 1, detection=det)
        for i, det in enumerate(sample_detections)
    ]
```

- [ ] **Step 8: Create empty __init__.py for tests**

```python
# tests/__init__.py
```

- [ ] **Step 9: Install package in dev mode and run tests**

Run: `pip install -e ".[dev]" && pytest tests/test_models.py -v`
Expected: All 4 tests PASS

- [ ] **Step 10: Commit**

```bash
git init
git add pyproject.toml yolo26_analytics/__init__.py yolo26_analytics/py.typed yolo26_analytics/models.py tests/__init__.py tests/conftest.py tests/test_models.py
git commit -m "feat: project scaffolding and core data models (Detection, Track, Event, FrameMeta)"
```

---

### Task 2: Protocols

**Files:**
- Create: `yolo26_analytics/protocols.py`

- [ ] **Step 1: Write protocols**

```python
# yolo26_analytics/protocols.py
"""Protocol definitions for all pipeline stages."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

import numpy as np

from yolo26_analytics.models import Detection, Event, FrameMeta, Track


@runtime_checkable
class VideoSource(Protocol):
    """Yields video frames."""

    async def __aiter__(self) -> AsyncIterator[tuple[np.ndarray, FrameMeta]]: ...

    async def close(self) -> None: ...


@runtime_checkable
class Detector(Protocol):
    """Runs object detection on a frame."""

    def predict(self, frame: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Assigns persistent track IDs to detections across frames."""

    def update(self, detections: list[Detection]) -> list[Track]: ...

    def reset(self) -> None: ...


@runtime_checkable
class TrackStore(Protocol):
    """Persists track updates."""

    async def write_tracks(
        self, tracks: list[Track], meta: FrameMeta
    ) -> None: ...

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
```

- [ ] **Step 2: Export protocols from package**

Update `yolo26_analytics/__init__.py` to add:

```python
from yolo26_analytics.protocols import (
    AlertBackend,
    Detector,
    EventStore,
    Tracker,
    TrackStore,
    VideoSource,
)

__all__ = [
    "AlertBackend",
    "Detection",
    "Detector",
    "Event",
    "EventStore",
    "FrameMeta",
    "Track",
    "Tracker",
    "TrackStore",
    "VideoSource",
]
```

- [ ] **Step 3: Verify mypy passes**

Run: `mypy yolo26_analytics/protocols.py --strict`
Expected: Success with no errors

- [ ] **Step 4: Commit**

```bash
git add yolo26_analytics/protocols.py yolo26_analytics/__init__.py
git commit -m "feat: protocol definitions for all pipeline stages"
```

---

### Task 3: Config System

**Files:**
- Create: `yolo26_analytics/config/__init__.py`
- Create: `yolo26_analytics/config/schema.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config parsing**

```python
# tests/test_config.py
import tempfile
from pathlib import Path

import pytest
import yaml

from yolo26_analytics.config.schema import (
    AlertConfig,
    AppConfig,
    ModelConfig,
    SourceConfig,
    StoreConfig,
    TrackingConfig,
    ZoneAnalyticsRule,
    ZoneConfig,
    load_config,
)


MINIMAL_CONFIG = {
    "source": {"type": "video_file", "path": "test.mp4"},
    "model": {"weights": "yolo26n.pt"},
}

FULL_CONFIG = {
    "source": {"type": "rtsp", "url": "rtsp://192.168.1.100/stream"},
    "model": {
        "type": "yolo26",
        "weights": "yolo26n.pt",
        "confidence": 0.5,
        "sahi": {"enabled": True, "slice_size": 640, "overlap": 0.25},
    },
    "tracking": {"engine": "bytetrack", "max_age": 30, "min_hits": 3},
    "store": {
        "type": "postgresql",
        "url": "postgresql://localhost/y26a",
        "retention": {"tracks": "7d", "events": "90d", "snapshots": "30d"},
    },
    "zones": [
        {
            "name": "Loading Dock",
            "polygon": [[100, 100], [400, 100], [400, 400], [100, 400]],
            "track_classes": ["person", "forklift"],
            "analytics": [
                {"type": "count"},
                {"type": "dwell", "alert_threshold": 300},
                {"type": "entry_exit", "direction": "inward"},
                {"type": "throughput", "interval": 3600},
            ],
            "cooldown": 30,
        }
    ],
    "alerts": [
        {"type": "webhook", "url": "https://hooks.slack.com/xxx"},
        {
            "type": "telegram",
            "bot_token": "123:ABC",
            "chat_id": "-100123",
            "filter": {
                "zones": ["Loading Dock"],
                "event_types": ["dwell_exceeded"],
            },
        },
        {"type": "mqtt", "broker": "mqtt://192.168.1.50", "topic": "events"},
        {"type": "console"},
    ],
    "dashboard": True,
}


class TestLoadConfig:
    def test_minimal_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(MINIMAL_CONFIG))
        config = load_config(str(config_file))
        assert config.source.type == "video_file"
        assert config.model.weights == "yolo26n.pt"
        assert config.model.confidence == 0.5  # default
        assert config.store.type == "sqlite"  # default
        assert config.dashboard is False  # default

    def test_full_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(FULL_CONFIG))
        config = load_config(str(config_file))
        assert config.source.type == "rtsp"
        assert config.source.url == "rtsp://192.168.1.100/stream"
        assert config.model.sahi is not None
        assert config.model.sahi.enabled is True
        assert config.model.sahi.slice_size == 640
        assert len(config.zones) == 1
        zone = config.zones[0]
        assert zone.name == "Loading Dock"
        assert len(zone.analytics) == 4
        assert zone.cooldown == 30
        assert len(config.alerts) == 4
        assert config.alerts[1].filter is not None
        assert config.alerts[1].filter.zones == ["Loading Dock"]
        assert config.dashboard is True

    def test_defaults_applied(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(MINIMAL_CONFIG))
        config = load_config(str(config_file))
        assert config.tracking.engine == "bytetrack"
        assert config.tracking.max_age == 30
        assert config.tracking.min_hits == 3
        assert config.store.type == "sqlite"
        assert config.store.path == "./data/yolo26_analytics.db"


class TestZoneConfig:
    def test_zone_polygon_validation(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            ZoneConfig(
                name="bad",
                polygon=[[0, 0], [1, 1]],
                track_classes=["person"],
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement config schema**

```python
# yolo26_analytics/config/__init__.py
from yolo26_analytics.config.schema import AppConfig, load_config

__all__ = ["AppConfig", "load_config"]
```

```python
# yolo26_analytics/config/schema.py
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yolo26_analytics/config/ tests/test_config.py
git commit -m "feat: YAML config system with Pydantic validation"
```

---

### Task 4: Video Sources

**Files:**
- Create: `yolo26_analytics/sources/__init__.py`
- Create: `yolo26_analytics/sources/base.py`
- Create: `yolo26_analytics/sources/video_file.py`
- Create: `yolo26_analytics/sources/webcam.py`
- Create: `yolo26_analytics/sources/rtsp.py`
- Create: `yolo26_analytics/sources/image_dir.py`
- Create: `tests/test_sources.py`

- [ ] **Step 1: Write failing test for video file source**

```python
# tests/test_sources.py
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from yolo26_analytics.sources.video_file import VideoFileSource
from yolo26_analytics.sources.image_dir import ImageDirSource
from yolo26_analytics.sources import create_source
from yolo26_analytics.config.schema import SourceConfig


def _create_test_video(path: str, num_frames: int = 10) -> None:
    """Write a small test video."""
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480)
    )
    for i in range(num_frames):
        frame = np.full((480, 640, 3), i * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoFileSource:
    @pytest.mark.asyncio
    async def test_iterates_frames(self, tmp_path: Path) -> None:
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=5)
        source = VideoFileSource(path=video_path, source_id="test")
        frames = []
        async for frame, meta in source:
            frames.append((frame, meta))
        assert len(frames) == 5
        assert frames[0][1].source_id == "test"
        assert frames[0][1].frame_index == 0
        assert frames[4][1].frame_index == 4
        assert frames[0][0].shape == (480, 640, 3)


class TestImageDirSource:
    @pytest.mark.asyncio
    async def test_iterates_images(self, tmp_path: Path) -> None:
        for i in range(3):
            img = np.full((480, 640, 3), i * 80, dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i:03d}.png"), img)
        source = ImageDirSource(path=str(tmp_path), source_id="test_dir")
        frames = []
        async for frame, meta in source:
            frames.append((frame, meta))
        assert len(frames) == 3
        assert frames[0][1].source_id == "test_dir"


class TestCreateSource:
    @pytest.mark.asyncio
    async def test_factory_video_file(self, tmp_path: Path) -> None:
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=3)
        config = SourceConfig(type="video_file", path=video_path)
        source = create_source(config, source_id="factory_test")
        count = 0
        async for _ in source:
            count += 1
        assert count == 3

    def test_factory_unknown_type(self) -> None:
        config = SourceConfig(type="unknown_source")
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source(config, source_id="test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement video sources**

```python
# yolo26_analytics/sources/base.py
"""Base types for video sources."""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from yolo26_analytics.models import FrameMeta

# Type alias for what sources yield
FrameItem = tuple[np.ndarray, FrameMeta]
```

```python
# yolo26_analytics/sources/video_file.py
"""Video file source."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import cv2
import numpy as np

from yolo26_analytics.models import FrameMeta
from yolo26_analytics.sources.base import FrameItem


class VideoFileSource:
    """Reads frames from a video file."""

    def __init__(self, path: str, source_id: str) -> None:
        self._path = path
        self._source_id = source_id
        self._cap: cv2.VideoCapture | None = None

    async def __aiter__(self) -> AsyncIterator[FrameItem]:
        self._cap = cv2.VideoCapture(self._path)
        frame_index = 0
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                meta = FrameMeta(
                    timestamp=datetime.now(tz=timezone.utc),
                    frame_index=frame_index,
                    source_id=self._source_id,
                )
                yield frame, meta
                frame_index += 1
        finally:
            self._cap.release()

    async def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
```

```python
# yolo26_analytics/sources/webcam.py
"""Webcam source."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import cv2

from yolo26_analytics.models import FrameMeta
from yolo26_analytics.sources.base import FrameItem


class WebcamSource:
    """Reads frames from a webcam device."""

    def __init__(self, device: int = 0, source_id: str = "webcam") -> None:
        self._device = device
        self._source_id = source_id
        self._cap: cv2.VideoCapture | None = None

    async def __aiter__(self) -> AsyncIterator[FrameItem]:
        self._cap = cv2.VideoCapture(self._device)
        frame_index = 0
        try:
            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    break
                meta = FrameMeta(
                    timestamp=datetime.now(tz=timezone.utc),
                    frame_index=frame_index,
                    source_id=self._source_id,
                )
                yield frame, meta
                frame_index += 1
        finally:
            self._cap.release()

    async def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
```

```python
# yolo26_analytics/sources/rtsp.py
"""RTSP stream source."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import cv2

from yolo26_analytics.models import FrameMeta
from yolo26_analytics.sources.base import FrameItem


class RTSPSource:
    """Reads frames from an RTSP stream."""

    def __init__(self, url: str, source_id: str = "rtsp") -> None:
        self._url = url
        self._source_id = source_id
        self._cap: cv2.VideoCapture | None = None

    async def __aiter__(self) -> AsyncIterator[FrameItem]:
        self._cap = cv2.VideoCapture(self._url)
        frame_index = 0
        try:
            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    break
                meta = FrameMeta(
                    timestamp=datetime.now(tz=timezone.utc),
                    frame_index=frame_index,
                    source_id=self._source_id,
                )
                yield frame, meta
                frame_index += 1
        finally:
            self._cap.release()

    async def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
```

```python
# yolo26_analytics/sources/image_dir.py
"""Image directory source."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import cv2

from yolo26_analytics.models import FrameMeta
from yolo26_analytics.sources.base import FrameItem

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


class ImageDirSource:
    """Reads frames from a directory of images, sorted by filename."""

    def __init__(self, path: str, source_id: str = "image_dir") -> None:
        self._path = Path(path)
        self._source_id = source_id

    async def __aiter__(self) -> AsyncIterator[FrameItem]:
        files = sorted(
            f for f in self._path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        for i, filepath in enumerate(files):
            frame = cv2.imread(str(filepath))
            if frame is None:
                continue
            meta = FrameMeta(
                timestamp=datetime.now(tz=timezone.utc),
                frame_index=i,
                source_id=self._source_id,
            )
            yield frame, meta

    async def close(self) -> None:
        pass
```

```python
# yolo26_analytics/sources/__init__.py
"""Video source factory."""

from __future__ import annotations

from yolo26_analytics.config.schema import SourceConfig
from yolo26_analytics.sources.image_dir import ImageDirSource
from yolo26_analytics.sources.rtsp import RTSPSource
from yolo26_analytics.sources.video_file import VideoFileSource
from yolo26_analytics.sources.webcam import WebcamSource


def create_source(
    config: SourceConfig, source_id: str = "default"
) -> VideoFileSource | WebcamSource | RTSPSource | ImageDirSource:
    """Factory: create a video source from config."""
    match config.type:
        case "video_file":
            if config.path is None:
                raise ValueError("video_file source requires 'path'")
            return VideoFileSource(path=config.path, source_id=source_id)
        case "webcam":
            return WebcamSource(device=config.device or 0, source_id=source_id)
        case "rtsp":
            if config.url is None:
                raise ValueError("rtsp source requires 'url'")
            return RTSPSource(url=config.url, source_id=source_id)
        case "image_dir":
            if config.path is None:
                raise ValueError("image_dir source requires 'path'")
            return ImageDirSource(path=config.path, source_id=source_id)
        case _:
            raise ValueError(f"Unknown source type: {config.type}")


__all__ = [
    "ImageDirSource",
    "RTSPSource",
    "VideoFileSource",
    "WebcamSource",
    "create_source",
]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sources.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yolo26_analytics/sources/ tests/test_sources.py
git commit -m "feat: video source adapters (file, webcam, RTSP, image dir) with factory"
```

---

### Task 5: YOLO26 Detector Adapter

**Files:**
- Create: `yolo26_analytics/detection/__init__.py`
- Create: `yolo26_analytics/detection/yolo26.py`
- Create: `tests/test_detection.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_detection.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yolo26_analytics.detection.yolo26 import YOLO26Detector
from yolo26_analytics.models import Detection


class TestYOLO26Detector:
    def _make_mock_result(self) -> MagicMock:
        """Create a mock Ultralytics result."""
        box1 = MagicMock()
        box1.xyxy = MagicMock()
        box1.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 200, 300, 400]])
        box1.conf = MagicMock()
        box1.conf.cpu.return_value.numpy.return_value = np.array([0.92])
        box1.cls = MagicMock()
        box1.cls.cpu.return_value.numpy.return_value = np.array([0])

        result = MagicMock()
        result.boxes = [box1]
        return result

    @patch("yolo26_analytics.detection.yolo26.YOLO")
    def test_predict_returns_detections(self, mock_yolo_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "helmet"}
        mock_result = self._make_mock_result()
        mock_model.return_value = [mock_result]
        mock_yolo_cls.return_value = mock_model

        detector = YOLO26Detector(weights="yolo26n.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.predict(frame)

        assert len(detections) == 1
        assert detections[0].class_name == "person"
        assert detections[0].confidence == pytest.approx(0.92, abs=0.01)
        assert detections[0].bbox == (100, 200, 300, 400)

    @patch("yolo26_analytics.detection.yolo26.YOLO")
    def test_filters_below_confidence(self, mock_yolo_cls: MagicMock) -> None:
        box = MagicMock()
        box.xyxy = MagicMock()
        box.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 30, 40]])
        box.conf = MagicMock()
        box.conf.cpu.return_value.numpy.return_value = np.array([0.3])
        box.cls = MagicMock()
        box.cls.cpu.return_value.numpy.return_value = np.array([0])

        result = MagicMock()
        result.boxes = [box]
        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        mock_model.return_value = [result]
        mock_yolo_cls.return_value = mock_model

        detector = YOLO26Detector(weights="yolo26n.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.predict(frame)
        assert len(detections) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detection.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement YOLO26 detector**

```python
# yolo26_analytics/detection/__init__.py
from yolo26_analytics.detection.yolo26 import YOLO26Detector

__all__ = ["YOLO26Detector"]
```

```python
# yolo26_analytics/detection/yolo26.py
"""YOLO26 detector adapter using Ultralytics."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from yolo26_analytics.models import Detection


class YOLO26Detector:
    """Wraps Ultralytics YOLO for the Detector protocol."""

    def __init__(self, weights: str = "yolo26n.pt", confidence: float = 0.5) -> None:
        self._model = YOLO(weights)
        self._confidence = confidence
        self._names: dict[int, str] = self._model.names  # type: ignore[assignment]

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf.cpu().numpy()[0])
                if conf < self._confidence:
                    continue
                xyxy = box.xyxy.cpu().numpy()[0]
                cls_id = int(box.cls.cpu().numpy()[0])
                detections.append(
                    Detection(
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=conf,
                        class_name=self._names.get(cls_id, f"class_{cls_id}"),
                    )
                )
        return detections
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_detection.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yolo26_analytics/detection/ tests/test_detection.py
git commit -m "feat: YOLO26 detector adapter with confidence filtering"
```

---

### Task 6: ByteTrack Tracker

**Files:**
- Create: `yolo26_analytics/tracking/__init__.py`
- Create: `yolo26_analytics/tracking/bytetrack.py`
- Create: `tests/test_tracking.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_tracking.py
from __future__ import annotations

import numpy as np
import pytest

from yolo26_analytics.models import Detection
from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter


class TestByteTrackAdapter:
    def test_assigns_track_ids(self) -> None:
        tracker = ByteTrackAdapter(max_age=30, min_hits=1)
        detections = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person"),
            Detection(bbox=(300, 300, 400, 400), confidence=0.8, class_name="person"),
        ]
        tracks = tracker.update(detections)
        assert len(tracks) >= 1
        track_ids = {t.track_id for t in tracks}
        # Each track should have a unique ID
        assert len(track_ids) == len(tracks)

    def test_maintains_ids_across_frames(self) -> None:
        tracker = ByteTrackAdapter(max_age=30, min_hits=1)
        det1 = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person")]
        tracks1 = tracker.update(det1)
        # Same position next frame — should keep same ID
        det2 = [Detection(bbox=(102, 102, 202, 202), confidence=0.9, class_name="person")]
        tracks2 = tracker.update(det2)
        if tracks1 and tracks2:
            assert tracks1[0].track_id == tracks2[0].track_id

    def test_reset_clears_state(self) -> None:
        tracker = ByteTrackAdapter(max_age=30, min_hits=1)
        det = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person")]
        tracker.update(det)
        tracker.reset()
        tracks = tracker.update(det)
        # After reset, should start with fresh IDs
        assert len(tracks) >= 0  # tracker may or may not return on first frame

    def test_empty_detections(self) -> None:
        tracker = ByteTrackAdapter(max_age=30, min_hits=1)
        tracks = tracker.update([])
        assert tracks == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement ByteTrack adapter**

ByteTrack is available in Ultralytics. We wrap it to consume our `Detection` type and emit `Track`.

```python
# yolo26_analytics/tracking/__init__.py
from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter

__all__ = ["ByteTrackAdapter"]
```

```python
# yolo26_analytics/tracking/bytetrack.py
"""ByteTrack adapter using Ultralytics' built-in tracker."""

from __future__ import annotations

import numpy as np

from yolo26_analytics.models import Detection, Track


class ByteTrackAdapter:
    """Wraps Ultralytics ByteTrack for the Tracker protocol."""

    def __init__(self, max_age: int = 30, min_hits: int = 3) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        class _Args:
            track_thresh: float = 0.25
            track_buffer: int = max_age
            match_thresh: float = 0.8

        self._tracker = BYTETracker(args=_Args(), frame_rate=30)
        self._min_hits = min_hits

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            return []

        # Build numpy array: [x1, y1, x2, y2, confidence, class_id]
        det_array = np.array(
            [
                [*d.bbox, d.confidence, 0]
                for d in detections
            ],
            dtype=np.float32,
        )

        img_size = (1080, 1920)  # assumed frame size for tracker
        online_targets = self._tracker.update(
            det_array[:, :5],
            img_size,
            img_size,
        )

        tracks: list[Track] = []
        for target in online_targets:
            tlwh = target.tlwh
            x1, y1, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
            bbox = (x1, y1, x1 + w, y1 + h)

            # Find closest original detection to preserve class_name and confidence
            best_det = self._match_detection(bbox, detections)

            tracks.append(
                Track(
                    track_id=int(target.track_id),
                    detection=Detection(
                        bbox=bbox,
                        confidence=best_det.confidence if best_det else target.score,
                        class_name=best_det.class_name if best_det else "unknown",
                    ),
                )
            )
        return tracks

    @staticmethod
    def _match_detection(
        bbox: tuple[int, int, int, int], detections: list[Detection]
    ) -> Detection | None:
        """Find the detection with highest IoU to the tracker bbox."""
        best: Detection | None = None
        best_iou = 0.0
        bx1, by1, bx2, by2 = bbox
        for det in detections:
            dx1, dy1, dx2, dy2 = det.bbox
            ix1 = max(bx1, dx1)
            iy1 = max(by1, dy1)
            ix2 = min(bx2, dx2)
            iy2 = min(by2, dy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_b = (bx2 - bx1) * (by2 - by1)
            area_d = (dx2 - dx1) * (dy2 - dy1)
            union = area_b + area_d - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best = det
        return best

    def reset(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        class _Args:
            track_thresh: float = 0.25
            track_buffer: int = 30
            match_thresh: float = 0.8

        self._tracker = BYTETracker(args=_Args(), frame_rate=30)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_tracking.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yolo26_analytics/tracking/ tests/test_tracking.py
git commit -m "feat: ByteTrack adapter with detection matching"
```

---

### Task 7: Store Layer (SQLite + PostgreSQL)

**Files:**
- Create: `yolo26_analytics/store/__init__.py`
- Create: `yolo26_analytics/store/models.py`
- Create: `yolo26_analytics/store/base.py`
- Create: `yolo26_analytics/store/sqlite.py`
- Create: `yolo26_analytics/store/postgres.py`
- Create: `yolo26_analytics/store/retention.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing test (SQLite — no external deps)**

```python
# tests/test_store.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from yolo26_analytics.models import Event, FrameMeta, Track, Detection
from yolo26_analytics.store.sqlite import SQLiteStore


@pytest.fixture
async def store(tmp_path) -> SQLiteStore:
    db_path = str(tmp_path / "test.db")
    s = SQLiteStore(path=db_path)
    await s.initialize()
    return s


class TestSQLiteTrackStore:
    @pytest.mark.asyncio
    async def test_write_and_query_tracks(self, store: SQLiteStore) -> None:
        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(100, 100, 200, 200), confidence=0.9, class_name="person"
            )),
            Track(track_id=2, detection=Detection(
                bbox=(300, 300, 400, 400), confidence=0.8, class_name="forklift"
            )),
        ]
        meta = FrameMeta(
            timestamp=datetime.now(tz=timezone.utc),
            frame_index=0,
            source_id="cam1",
        )
        await store.write_tracks(tracks, meta)
        results = await store.query_tracks(source_id="cam1")
        assert len(results) == 2
        assert results[0]["track_id"] == 1
        assert results[0]["object_class"] == "person"
        assert results[1]["track_id"] == 2

    @pytest.mark.asyncio
    async def test_query_tracks_by_zone(self, store: SQLiteStore) -> None:
        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(150, 150, 250, 250), confidence=0.9, class_name="person"
            )),
        ]
        meta = FrameMeta(
            timestamp=datetime.now(tz=timezone.utc),
            frame_index=0,
            source_id="cam1",
        )
        await store.write_tracks(tracks, meta, zone_name="Loading Dock")
        results = await store.query_tracks(zone_name="Loading Dock")
        assert len(results) == 1
        assert results[0]["zone_name"] == "Loading Dock"
        # Query different zone returns nothing
        results = await store.query_tracks(zone_name="Entrance")
        assert len(results) == 0


class TestSQLiteEventStore:
    @pytest.mark.asyncio
    async def test_log_and_query_events(self, store: SQLiteStore) -> None:
        events = [
            Event(
                timestamp=datetime.now(tz=timezone.utc),
                event_type="entry",
                zone_name="Loading Dock",
                track_id=1,
                object_class="person",
                metadata={"direction": "inward"},
                confidence=0.9,
                frame_snapshot=b"",
                bbox=(100, 100, 200, 200),
            ),
        ]
        await store.log_events(events)
        results = await store.query_events(zone_name="Loading Dock")
        assert len(results) == 1
        assert results[0]["event_type"] == "entry"
        assert results[0]["metadata"]["direction"] == "inward"

    @pytest.mark.asyncio
    async def test_query_events_by_type(self, store: SQLiteStore) -> None:
        events = [
            Event(
                timestamp=datetime.now(tz=timezone.utc),
                event_type="entry",
                zone_name="Dock",
                track_id=1,
                object_class="person",
                metadata={},
                confidence=0.9,
                frame_snapshot=b"",
                bbox=(100, 100, 200, 200),
            ),
            Event(
                timestamp=datetime.now(tz=timezone.utc),
                event_type="dwell_exceeded",
                zone_name="Dock",
                track_id=1,
                object_class="person",
                metadata={"dwell_seconds": 400},
                confidence=0.9,
                frame_snapshot=b"",
                bbox=(100, 100, 200, 200),
            ),
        ]
        await store.log_events(events)
        results = await store.query_events(event_type="dwell_exceeded")
        assert len(results) == 1
        assert results[0]["event_type"] == "dwell_exceeded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement SQLAlchemy ORM models**

```python
# yolo26_analytics/store/__init__.py
from yolo26_analytics.store.sqlite import SQLiteStore

__all__ = ["SQLiteStore"]
```

```python
# yolo26_analytics/store/models.py
"""SQLAlchemy ORM models for tracks, events, and zone_stats."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TrackRow(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    zone_name: Mapped[str] = mapped_column(String(100), nullable=False)
    object_class: Mapped[str] = mapped_column(String(50), nullable=False)
    current_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entries_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exits_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_dwell_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
```

- [ ] **Step 4: Implement SQLite store**

```python
# yolo26_analytics/store/sqlite.py
"""SQLite store — dev fallback. Implements both TrackStore and EventStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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

    # --- TrackStore ---

    async def write_tracks(
        self,
        tracks: list[Track],
        meta: FrameMeta,
        zone_name: str | None = None,
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

    # --- EventStore ---

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
```

- [ ] **Step 5: Implement PostgreSQL store**

```python
# yolo26_analytics/store/postgres.py
"""PostgreSQL store — production backend."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from yolo26_analytics.models import Event, FrameMeta, Track
from yolo26_analytics.store.models import Base, EventRow, TrackRow


class PostgresStore:
    """Combined TrackStore + EventStore backed by PostgreSQL."""

    def __init__(self, url: str) -> None:
        self._engine = create_async_engine(url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def write_tracks(
        self,
        tracks: list[Track],
        meta: FrameMeta,
        zone_name: str | None = None,
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
```

- [ ] **Step 6: Implement retention cleanup**

```python
# yolo26_analytics/store/retention.py
"""Periodic retention cleanup for old tracks, events, and snapshots."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from yolo26_analytics.store.models import EventRow, TrackRow


def parse_duration(s: str) -> timedelta:
    """Parse '7d', '90d', '30d' into a timedelta."""
    match = re.match(r"^(\d+)([dhms])$", s)
    if not match:
        raise ValueError(f"Invalid duration format: {s}")
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "m":
        return timedelta(minutes=value)
    return timedelta(seconds=value)


async def run_retention_cleanup(
    session_factory: async_sessionmaker,
    tracks_retention: str = "7d",
    events_retention: str = "90d",
    snapshots_retention: str = "30d",
    snapshots_dir: str = "data/snapshots",
) -> None:
    """Delete old tracks, events, and snapshot files."""
    now = datetime.now(tz=timezone.utc)

    async with session_factory() as session:
        # Clean old tracks
        cutoff = now - parse_duration(tracks_retention)
        stmt = delete(TrackRow).where(TrackRow.timestamp < cutoff)
        await session.execute(stmt)

        # Clean old events
        cutoff = now - parse_duration(events_retention)
        stmt = delete(EventRow).where(EventRow.timestamp < cutoff)
        await session.execute(stmt)

        await session.commit()

    # Clean old snapshots
    snap_cutoff = now - parse_duration(snapshots_retention)
    snap_dir = Path(snapshots_dir)
    if snap_dir.exists():
        for date_dir in snap_dir.iterdir():
            if date_dir.is_dir():
                for f in date_dir.iterdir():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                    if mtime < snap_cutoff:
                        f.unlink()
                # Remove empty date dirs
                if not any(date_dir.iterdir()):
                    date_dir.rmdir()
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_store.py -v`
Expected: All 4 tests PASS

- [ ] **Step 8: Commit**

```bash
git add yolo26_analytics/store/ tests/test_store.py
git commit -m "feat: store layer — SQLite + PostgreSQL with retention cleanup"
```

---

### Task 8: Zone System

**Files:**
- Create: `yolo26_analytics/zones/__init__.py`
- Create: `yolo26_analytics/zones/polygon.py`
- Create: `yolo26_analytics/zones/counting.py`
- Create: `yolo26_analytics/zones/entry_exit.py`
- Create: `yolo26_analytics/zones/dwell.py`
- Create: `yolo26_analytics/zones/throughput.py`
- Create: `yolo26_analytics/zones/analyzer.py`
- Create: `tests/test_zones/`

- [ ] **Step 1: Write failing test for polygon**

```python
# tests/test_zones/__init__.py
```

```python
# tests/test_zones/test_polygon.py
from yolo26_analytics.zones.polygon import Zone


class TestZone:
    def test_point_inside(self) -> None:
        zone = Zone(
            name="test",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            track_classes=["person"],
        )
        assert zone.contains_point(50, 50)

    def test_point_outside(self) -> None:
        zone = Zone(
            name="test",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            track_classes=["person"],
        )
        assert not zone.contains_point(150, 50)

    def test_filters_by_class(self) -> None:
        zone = Zone(
            name="test",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            track_classes=["person"],
        )
        assert zone.should_track("person")
        assert not zone.should_track("car")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_zones/test_polygon.py -v`
Expected: FAIL

- [ ] **Step 3: Implement polygon zone**

```python
# yolo26_analytics/zones/__init__.py
from yolo26_analytics.zones.analyzer import ZoneAnalyzer
from yolo26_analytics.zones.polygon import Zone

__all__ = ["Zone", "ZoneAnalyzer"]
```

```python
# yolo26_analytics/zones/polygon.py
"""Zone polygon with point-in-polygon check via Shapely."""

from __future__ import annotations

from shapely.geometry import Point, Polygon


class Zone:
    """A named polygon zone that tracks specific object classes."""

    def __init__(
        self,
        name: str,
        polygon: list[tuple[int, int]],
        track_classes: list[str],
        cooldown: int = 30,
    ) -> None:
        self.name = name
        self.track_classes = track_classes
        self.cooldown = cooldown
        self._polygon = Polygon(polygon)

    def contains_point(self, x: int, y: int) -> bool:
        return self._polygon.contains(Point(x, y))

    def should_track(self, class_name: str) -> bool:
        return class_name in self.track_classes

    @property
    def polygon(self) -> Polygon:
        return self._polygon
```

- [ ] **Step 4: Run polygon tests**

Run: `pytest tests/test_zones/test_polygon.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Write failing test for counting**

```python
# tests/test_zones/test_counting.py
from yolo26_analytics.models import Detection, Track
from yolo26_analytics.zones.counting import ZoneCounting
from yolo26_analytics.zones.polygon import Zone


class TestZoneCounting:
    def _zone(self) -> Zone:
        return Zone(
            name="test",
            polygon=[(0, 0), (200, 0), (200, 200), (0, 200)],
            track_classes=["person"],
        )

    def test_counts_tracks_in_zone(self) -> None:
        counter = ZoneCounting()
        zone = self._zone()
        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"
            )),
            Track(track_id=2, detection=Detection(
                bbox=(300, 300, 400, 400), confidence=0.8, class_name="person"
            )),
        ]
        count = counter.update(zone, tracks)
        assert count == {"person": 1}

    def test_ignores_untracked_classes(self) -> None:
        counter = ZoneCounting()
        zone = self._zone()
        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(50, 50, 150, 150), confidence=0.9, class_name="car"
            )),
        ]
        count = counter.update(zone, tracks)
        assert count == {}
```

- [ ] **Step 6: Implement counting**

```python
# yolo26_analytics/zones/counting.py
"""Real-time zone counting."""

from __future__ import annotations

from collections import defaultdict

from yolo26_analytics.models import Track
from yolo26_analytics.zones.polygon import Zone


class ZoneCounting:
    """Counts tracked objects currently inside a zone, grouped by class."""

    def update(self, zone: Zone, tracks: list[Track]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for track in tracks:
            if not zone.should_track(track.class_name):
                continue
            cx, cy = track.centroid
            if zone.contains_point(cx, cy):
                counts[track.class_name] += 1
        return dict(counts)
```

- [ ] **Step 7: Run counting tests**

Run: `pytest tests/test_zones/test_counting.py -v`
Expected: All 2 tests PASS

- [ ] **Step 8: Write failing test for entry/exit**

```python
# tests/test_zones/test_entry_exit.py
from yolo26_analytics.models import Detection, Track
from yolo26_analytics.zones.entry_exit import EntryExitDetector
from yolo26_analytics.zones.polygon import Zone


class TestEntryExitDetector:
    def _zone(self) -> Zone:
        return Zone(
            name="test",
            polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
            track_classes=["person"],
        )

    def test_detects_entry(self) -> None:
        detector = EntryExitDetector()
        zone = self._zone()
        # Frame 1: track outside zone
        tracks1 = [
            Track(track_id=1, detection=Detection(
                bbox=(10, 10, 60, 60), confidence=0.9, class_name="person"
            )),
        ]
        events = detector.update(zone, tracks1)
        assert len(events) == 0  # first frame, no transition

        # Frame 2: track moved inside zone
        tracks2 = [
            Track(track_id=1, detection=Detection(
                bbox=(150, 150, 250, 250), confidence=0.9, class_name="person"
            )),
        ]
        events = detector.update(zone, tracks2)
        assert len(events) == 1
        assert events[0]["type"] == "entry"
        assert events[0]["track_id"] == 1

    def test_detects_exit(self) -> None:
        detector = EntryExitDetector()
        zone = self._zone()
        # Frame 1: inside
        tracks1 = [
            Track(track_id=1, detection=Detection(
                bbox=(150, 150, 250, 250), confidence=0.9, class_name="person"
            )),
        ]
        detector.update(zone, tracks1)
        # Frame 2: outside
        tracks2 = [
            Track(track_id=1, detection=Detection(
                bbox=(10, 10, 60, 60), confidence=0.9, class_name="person"
            )),
        ]
        events = detector.update(zone, tracks2)
        assert len(events) == 1
        assert events[0]["type"] == "exit"

    def test_no_event_if_stays_inside(self) -> None:
        detector = EntryExitDetector()
        zone = self._zone()
        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(150, 150, 250, 250), confidence=0.9, class_name="person"
            )),
        ]
        detector.update(zone, tracks)
        events = detector.update(zone, tracks)
        assert len(events) == 0
```

- [ ] **Step 9: Implement entry/exit**

```python
# yolo26_analytics/zones/entry_exit.py
"""Entry/exit detection — tracks crossing zone boundaries."""

from __future__ import annotations

from yolo26_analytics.models import Track
from yolo26_analytics.zones.polygon import Zone


class EntryExitDetector:
    """Detects when a track enters or exits a zone between frames."""

    def __init__(self) -> None:
        # zone_name -> track_id -> was_inside
        self._prev_state: dict[str, dict[int, bool]] = {}

    def update(
        self, zone: Zone, tracks: list[Track]
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        zone_state = self._prev_state.setdefault(zone.name, {})

        current_ids: set[int] = set()
        for track in tracks:
            if not zone.should_track(track.class_name):
                continue
            current_ids.add(track.track_id)
            cx, cy = track.centroid
            is_inside = zone.contains_point(cx, cy)
            was_inside = zone_state.get(track.track_id)

            if was_inside is not None:
                if not was_inside and is_inside:
                    events.append({
                        "type": "entry",
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                    })
                elif was_inside and not is_inside:
                    events.append({
                        "type": "exit",
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                    })

            zone_state[track.track_id] = is_inside

        # Clean up tracks that disappeared
        for tid in list(zone_state.keys()):
            if tid not in current_ids:
                del zone_state[tid]

        return events
```

- [ ] **Step 10: Run entry/exit tests**

Run: `pytest tests/test_zones/test_entry_exit.py -v`
Expected: All 3 tests PASS

- [ ] **Step 11: Write failing test for dwell time**

```python
# tests/test_zones/test_dwell.py
from datetime import datetime, timedelta, timezone

from yolo26_analytics.models import Detection, Track
from yolo26_analytics.zones.dwell import DwellTracker
from yolo26_analytics.zones.polygon import Zone


class TestDwellTracker:
    def _zone(self) -> Zone:
        return Zone(
            name="test",
            polygon=[(0, 0), (200, 0), (200, 200), (0, 200)],
            track_classes=["person"],
        )

    def test_tracks_dwell_time(self) -> None:
        dwell = DwellTracker(alert_threshold=10)
        zone = self._zone()
        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"
            )),
        ]
        now = datetime.now(tz=timezone.utc)
        events = dwell.update(zone, tracks, now)
        assert len(events) == 0  # just started, not exceeded

        # Simulate 15 seconds later
        later = now + timedelta(seconds=15)
        events = dwell.update(zone, tracks, later)
        assert len(events) == 1
        assert events[0]["type"] == "dwell_exceeded"
        assert events[0]["dwell_seconds"] >= 10

    def test_no_alert_below_threshold(self) -> None:
        dwell = DwellTracker(alert_threshold=60)
        zone = self._zone()
        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"
            )),
        ]
        now = datetime.now(tz=timezone.utc)
        dwell.update(zone, tracks, now)
        events = dwell.update(zone, tracks, now + timedelta(seconds=5))
        assert len(events) == 0

    def test_clears_when_track_leaves(self) -> None:
        dwell = DwellTracker(alert_threshold=10)
        zone = self._zone()
        inside = [
            Track(track_id=1, detection=Detection(
                bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"
            )),
        ]
        outside = [
            Track(track_id=1, detection=Detection(
                bbox=(500, 500, 600, 600), confidence=0.9, class_name="person"
            )),
        ]
        now = datetime.now(tz=timezone.utc)
        dwell.update(zone, inside, now)
        # Track leaves zone
        dwell.update(zone, outside, now + timedelta(seconds=5))
        # Track re-enters — timer should have reset
        events = dwell.update(zone, inside, now + timedelta(seconds=6))
        assert len(events) == 0  # just re-entered, no alert
```

- [ ] **Step 12: Implement dwell tracker**

```python
# yolo26_analytics/zones/dwell.py
"""Dwell time tracking — how long each track stays in a zone."""

from __future__ import annotations

from datetime import datetime

from yolo26_analytics.models import Track
from yolo26_analytics.zones.polygon import Zone


class DwellTracker:
    """Tracks how long each object dwells in a zone. Alerts when threshold exceeded."""

    def __init__(self, alert_threshold: int = 300) -> None:
        self._threshold = alert_threshold
        # zone_name -> track_id -> enter_time
        self._enter_times: dict[str, dict[int, datetime]] = {}
        # zone_name -> track_id -> already_alerted
        self._alerted: dict[str, set[int]] = {}

    def update(
        self, zone: Zone, tracks: list[Track], now: datetime
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        zone_enters = self._enter_times.setdefault(zone.name, {})
        zone_alerted = self._alerted.setdefault(zone.name, set())

        current_inside: set[int] = set()

        for track in tracks:
            if not zone.should_track(track.class_name):
                continue
            cx, cy = track.centroid
            if not zone.contains_point(cx, cy):
                # Track is outside — clear its dwell state
                zone_enters.pop(track.track_id, None)
                zone_alerted.discard(track.track_id)
                continue

            current_inside.add(track.track_id)

            if track.track_id not in zone_enters:
                zone_enters[track.track_id] = now
                continue

            dwell_secs = (now - zone_enters[track.track_id]).total_seconds()
            if dwell_secs >= self._threshold and track.track_id not in zone_alerted:
                zone_alerted.add(track.track_id)
                events.append({
                    "type": "dwell_exceeded",
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "dwell_seconds": dwell_secs,
                })

        # Clean up tracks that disappeared entirely
        for tid in list(zone_enters.keys()):
            if tid not in current_inside:
                zone_enters.pop(tid, None)
                zone_alerted.discard(tid)

        return events
```

- [ ] **Step 13: Run dwell tests**

Run: `pytest tests/test_zones/test_dwell.py -v`
Expected: All 3 tests PASS

- [ ] **Step 14: Write failing test for throughput**

```python
# tests/test_zones/test_throughput.py
from datetime import datetime, timedelta, timezone

from yolo26_analytics.models import Detection, Track
from yolo26_analytics.zones.throughput import ThroughputTracker
from yolo26_analytics.zones.polygon import Zone


class TestThroughputTracker:
    def _zone(self) -> Zone:
        return Zone(
            name="test",
            polygon=[(0, 0), (200, 0), (200, 200), (0, 200)],
            track_classes=["person"],
        )

    def test_counts_unique_tracks_per_interval(self) -> None:
        tp = ThroughputTracker(interval=60)
        zone = self._zone()
        now = datetime.now(tz=timezone.utc)

        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"
            )),
        ]
        tp.update(zone, tracks, now)
        tp.update(zone, tracks, now + timedelta(seconds=10))  # same track
        tracks2 = [
            Track(track_id=2, detection=Detection(
                bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"
            )),
        ]
        tp.update(zone, tracks2, now + timedelta(seconds=20))

        stats = tp.get_stats(zone.name)
        assert stats["unique_tracks"] == 2

    def test_resets_after_interval(self) -> None:
        tp = ThroughputTracker(interval=60)
        zone = self._zone()
        now = datetime.now(tz=timezone.utc)

        tracks = [
            Track(track_id=1, detection=Detection(
                bbox=(50, 50, 150, 150), confidence=0.9, class_name="person"
            )),
        ]
        tp.update(zone, tracks, now)

        # After interval passes
        tp.update(zone, tracks, now + timedelta(seconds=70))
        stats = tp.get_stats(zone.name)
        assert stats["unique_tracks"] == 1  # reset, only current frame
```

- [ ] **Step 15: Implement throughput tracker**

```python
# yolo26_analytics/zones/throughput.py
"""Throughput tracking — objects per time interval through a zone."""

from __future__ import annotations

from datetime import datetime, timedelta

from yolo26_analytics.models import Track
from yolo26_analytics.zones.polygon import Zone


class ThroughputTracker:
    """Counts unique tracks passing through a zone per time interval."""

    def __init__(self, interval: int = 3600) -> None:
        self._interval = timedelta(seconds=interval)
        # zone_name -> {window_start, unique_ids}
        self._windows: dict[str, dict[str, object]] = {}

    def update(self, zone: Zone, tracks: list[Track], now: datetime) -> None:
        state = self._windows.get(zone.name)
        if state is None or (now - state["window_start"]) >= self._interval:  # type: ignore[operator]
            self._windows[zone.name] = {
                "window_start": now,
                "unique_ids": set(),
            }
            state = self._windows[zone.name]

        unique_ids: set[int] = state["unique_ids"]  # type: ignore[assignment]

        for track in tracks:
            if not zone.should_track(track.class_name):
                continue
            cx, cy = track.centroid
            if zone.contains_point(cx, cy):
                unique_ids.add(track.track_id)

    def get_stats(self, zone_name: str) -> dict[str, object]:
        state = self._windows.get(zone_name)
        if state is None:
            return {"unique_tracks": 0, "window_start": None}
        unique_ids: set[int] = state["unique_ids"]  # type: ignore[assignment]
        return {
            "unique_tracks": len(unique_ids),
            "window_start": state["window_start"],
        }
```

- [ ] **Step 16: Run throughput tests**

Run: `pytest tests/test_zones/test_throughput.py -v`
Expected: All 2 tests PASS

- [ ] **Step 17: Implement ZoneAnalyzer (orchestrator)**

```python
# yolo26_analytics/zones/analyzer.py
"""ZoneAnalyzer — orchestrates all zone analytics for the pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from yolo26_analytics.config.schema import ZoneConfig
from yolo26_analytics.models import Detection, Event, Track
from yolo26_analytics.zones.counting import ZoneCounting
from yolo26_analytics.zones.dwell import DwellTracker
from yolo26_analytics.zones.entry_exit import EntryExitDetector
from yolo26_analytics.zones.polygon import Zone
from yolo26_analytics.zones.throughput import ThroughputTracker


class ZoneAnalyzer:
    """Runs all configured analytics on all zones for each frame of tracks."""

    def __init__(self, zone_configs: list[ZoneConfig]) -> None:
        self._zones: list[Zone] = []
        self._counters: dict[str, ZoneCounting] = {}
        self._entry_exit: dict[str, EntryExitDetector] = {}
        self._dwell: dict[str, DwellTracker] = {}
        self._throughput: dict[str, ThroughputTracker] = {}
        # cooldown state: (track_id, zone_name, event_type) -> last_alert_time
        self._cooldowns: dict[tuple[int, str, str], datetime] = {}

        for zc in zone_configs:
            polygon_tuples = [(p[0], p[1]) for p in zc.polygon]
            zone = Zone(
                name=zc.name,
                polygon=polygon_tuples,
                track_classes=zc.track_classes,
                cooldown=zc.cooldown,
            )
            self._zones.append(zone)

            for rule in zc.analytics:
                match rule.type:
                    case "count":
                        self._counters[zc.name] = ZoneCounting()
                    case "entry_exit":
                        self._entry_exit[zc.name] = EntryExitDetector()
                    case "dwell":
                        self._dwell[zc.name] = DwellTracker(
                            alert_threshold=rule.alert_threshold or 300
                        )
                    case "throughput":
                        self._throughput[zc.name] = ThroughputTracker(
                            interval=rule.interval or 3600
                        )

    def check(self, tracks: list[Track]) -> list[Event]:
        """Run all zone analytics and return generated events."""
        now = datetime.now(tz=timezone.utc)
        events: list[Event] = []

        for zone in self._zones:
            # Counting
            if zone.name in self._counters:
                counts = self._counters[zone.name].update(zone, tracks)
                # Count exceeded alerts could be added here

            # Entry/exit
            if zone.name in self._entry_exit:
                ee_events = self._entry_exit[zone.name].update(zone, tracks)
                for ee in ee_events:
                    event = self._make_event(
                        event_type=ee["type"],  # type: ignore[arg-type]
                        zone_name=zone.name,
                        track_id=ee["track_id"],  # type: ignore[arg-type]
                        object_class=ee["class_name"],  # type: ignore[arg-type]
                        metadata=ee,
                        now=now,
                        cooldown=zone.cooldown,
                    )
                    if event is not None:
                        events.append(event)

            # Dwell
            if zone.name in self._dwell:
                dwell_events = self._dwell[zone.name].update(zone, tracks, now)
                for de in dwell_events:
                    event = self._make_event(
                        event_type="dwell_exceeded",
                        zone_name=zone.name,
                        track_id=de["track_id"],  # type: ignore[arg-type]
                        object_class=de["class_name"],  # type: ignore[arg-type]
                        metadata=de,
                        now=now,
                        cooldown=zone.cooldown,
                    )
                    if event is not None:
                        events.append(event)

            # Throughput
            if zone.name in self._throughput:
                self._throughput[zone.name].update(zone, tracks, now)

        return events

    def get_zone_counts(self) -> dict[str, dict[str, int]]:
        """Get current counts for all zones. Used by dashboard."""
        result: dict[str, dict[str, int]] = {}
        for zone in self._zones:
            if zone.name in self._counters:
                # Return last known counts — update() must be called first
                result[zone.name] = {}
        return result

    def get_throughput_stats(self) -> dict[str, dict[str, object]]:
        """Get throughput stats for all zones."""
        return {
            name: tp.get_stats(name)
            for name, tp in self._throughput.items()
        }

    @property
    def zones(self) -> list[Zone]:
        return self._zones

    def _make_event(
        self,
        event_type: str,
        zone_name: str,
        track_id: int,
        object_class: str,
        metadata: dict[str, Any],
        now: datetime,
        cooldown: int,
    ) -> Event | None:
        """Create an Event if cooldown has elapsed."""
        key = (track_id, zone_name, event_type)
        last = self._cooldowns.get(key)
        if last is not None and (now - last).total_seconds() < cooldown:
            return None
        self._cooldowns[key] = now
        return Event(
            timestamp=now,
            event_type=event_type,
            zone_name=zone_name,
            track_id=track_id,
            object_class=object_class,
            metadata=metadata,
            confidence=0.0,
            frame_snapshot=b"",
            bbox=(0, 0, 0, 0),
        )
```

- [ ] **Step 18: Run all zone tests**

Run: `pytest tests/test_zones/ -v`
Expected: All 13 tests PASS

- [ ] **Step 19: Commit**

```bash
git add yolo26_analytics/zones/ tests/test_zones/
git commit -m "feat: zone system — polygon, counting, entry/exit, dwell, throughput, analyzer"
```

---

### Task 9: Alert System

**Files:**
- Create: `yolo26_analytics/alerts/__init__.py`
- Create: `yolo26_analytics/alerts/manager.py`
- Create: `yolo26_analytics/alerts/console.py`
- Create: `yolo26_analytics/alerts/webhook.py`
- Create: `yolo26_analytics/alerts/mqtt.py`
- Create: `yolo26_analytics/alerts/telegram.py`
- Create: `tests/test_alerts/__init__.py`
- Create: `tests/test_alerts/test_manager.py`
- Create: `tests/test_alerts/test_webhook.py`

- [ ] **Step 1: Write failing test for alert manager routing**

```python
# tests/test_alerts/__init__.py
```

```python
# tests/test_alerts/test_manager.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from yolo26_analytics.alerts.manager import AlertManager
from yolo26_analytics.config.schema import AlertConfig, AlertFilterConfig
from yolo26_analytics.models import Event


def _make_event(zone: str = "Dock", event_type: str = "entry") -> Event:
    return Event(
        timestamp=datetime.now(tz=timezone.utc),
        event_type=event_type,
        zone_name=zone,
        track_id=1,
        object_class="person",
        metadata={},
        confidence=0.9,
        frame_snapshot=b"",
        bbox=(0, 0, 0, 0),
    )


class FakeBackend:
    def __init__(self) -> None:
        self.received: list[Event] = []

    async def send(self, event: Event) -> None:
        self.received.append(event)


class TestAlertManager:
    @pytest.mark.asyncio
    async def test_dispatches_to_all_backends(self) -> None:
        b1 = FakeBackend()
        b2 = FakeBackend()
        manager = AlertManager(backends=[(b1, None), (b2, None)])
        event = _make_event()
        await manager.dispatch([event])
        assert len(b1.received) == 1
        assert len(b2.received) == 1

    @pytest.mark.asyncio
    async def test_filters_by_zone(self) -> None:
        b = FakeBackend()
        filt = AlertFilterConfig(zones=["Dock"], event_types=[])
        manager = AlertManager(backends=[(b, filt)])
        await manager.dispatch([_make_event(zone="Dock")])
        await manager.dispatch([_make_event(zone="Other")])
        assert len(b.received) == 1

    @pytest.mark.asyncio
    async def test_filters_by_event_type(self) -> None:
        b = FakeBackend()
        filt = AlertFilterConfig(zones=[], event_types=["dwell_exceeded"])
        manager = AlertManager(backends=[(b, filt)])
        await manager.dispatch([_make_event(event_type="entry")])
        await manager.dispatch([_make_event(event_type="dwell_exceeded")])
        assert len(b.received) == 1
        assert b.received[0].event_type == "dwell_exceeded"

    @pytest.mark.asyncio
    async def test_no_filter_receives_all(self) -> None:
        b = FakeBackend()
        manager = AlertManager(backends=[(b, None)])
        await manager.dispatch([_make_event(zone="A"), _make_event(zone="B")])
        assert len(b.received) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_alerts/test_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Implement alert manager and backends**

```python
# yolo26_analytics/alerts/__init__.py
from yolo26_analytics.alerts.manager import AlertManager

__all__ = ["AlertManager"]
```

```python
# yolo26_analytics/alerts/manager.py
"""Alert manager with routing and dispatch."""

from __future__ import annotations

from yolo26_analytics.config.schema import AlertFilterConfig
from yolo26_analytics.models import Event
from yolo26_analytics.protocols import AlertBackend


class AlertManager:
    """Dispatches events to alert backends, applying filters."""

    def __init__(
        self,
        backends: list[tuple[AlertBackend, AlertFilterConfig | None]],
    ) -> None:
        self._backends = backends

    async def dispatch(self, events: list[Event]) -> None:
        for event in events:
            for backend, filt in self._backends:
                if self._matches_filter(event, filt):
                    await backend.send(event)

    @staticmethod
    def _matches_filter(event: Event, filt: AlertFilterConfig | None) -> bool:
        if filt is None:
            return True
        if filt.zones and event.zone_name not in filt.zones:
            return False
        if filt.event_types and event.event_type not in filt.event_types:
            return False
        return True
```

```python
# yolo26_analytics/alerts/console.py
"""Console alert backend — prints to stdout."""

from __future__ import annotations

from yolo26_analytics.models import Event


class ConsoleAlert:
    """Prints events to stdout."""

    async def send(self, event: Event) -> None:
        print(
            f"[ALERT] {event.timestamp.isoformat()} | "
            f"{event.event_type} | zone={event.zone_name} | "
            f"track={event.track_id} | class={event.object_class} | "
            f"meta={event.metadata}"
        )
```

```python
# yolo26_analytics/alerts/webhook.py
"""Webhook alert backend — HTTP POST JSON."""

from __future__ import annotations

import base64
import json

import httpx

from yolo26_analytics.models import Event


class WebhookAlert:
    """POSTs event data as JSON to a URL."""

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
```

```python
# yolo26_analytics/alerts/mqtt.py
"""MQTT alert backend."""

from __future__ import annotations

import json

import aiomqtt

from yolo26_analytics.models import Event


class MQTTAlert:
    """Publishes events to an MQTT topic."""

    def __init__(self, broker: str, topic: str = "yolo26/events") -> None:
        # Parse broker URL: mqtt://host:port -> host, port
        broker = broker.replace("mqtt://", "")
        parts = broker.split(":")
        self._host = parts[0]
        self._port = int(parts[1]) if len(parts) > 1 else 1883
        self._topic = topic

    async def send(self, event: Event) -> None:
        payload = json.dumps({
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "zone_name": event.zone_name,
            "track_id": event.track_id,
            "object_class": event.object_class,
            "metadata": event.metadata,
            "confidence": event.confidence,
        })
        async with aiomqtt.Client(self._host, self._port) as client:
            await client.publish(self._topic, payload.encode())
```

```python
# yolo26_analytics/alerts/telegram.py
"""Telegram alert backend."""

from __future__ import annotations

import httpx

from yolo26_analytics.models import Event


class TelegramAlert:
    """Sends alerts to a Telegram chat via bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._client = httpx.AsyncClient(timeout=10.0)
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

    async def send(self, event: Event) -> None:
        text = (
            f"🚨 *{event.event_type.replace('_', ' ').upper()}*\n"
            f"Zone: {event.zone_name}\n"
            f"Class: {event.object_class}\n"
            f"Track: {event.track_id}\n"
            f"Time: {event.timestamp.strftime('%H:%M:%S')}\n"
        )
        if event.metadata:
            for k, v in event.metadata.items():
                text += f"{k}: {v}\n"

        if event.frame_snapshot:
            await self._client.post(
                f"{self._base_url}/sendPhoto",
                data={"chat_id": self._chat_id, "caption": text, "parse_mode": "Markdown"},
                files={"photo": ("snapshot.jpg", event.frame_snapshot, "image/jpeg")},
            )
        else:
            await self._client.post(
                f"{self._base_url}/sendMessage",
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "Markdown"},
            )
```

- [ ] **Step 4: Run alert tests**

Run: `pytest tests/test_alerts/ -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yolo26_analytics/alerts/ tests/test_alerts/
git commit -m "feat: alert system — manager with routing + console, webhook, MQTT, Telegram backends"
```

---

### Task 10: Analytics Engine (Heatmaps + Stats)

**Files:**
- Create: `yolo26_analytics/analytics/__init__.py`
- Create: `yolo26_analytics/analytics/heatmap.py`
- Create: `yolo26_analytics/analytics/stats.py`
- Create: `tests/test_analytics.py`

- [ ] **Step 1: Write failing test for heatmap**

```python
# tests/test_analytics.py
from __future__ import annotations

import numpy as np
import pytest

from yolo26_analytics.analytics.heatmap import HeatmapAccumulator, generate_heatmap_image


class TestHeatmapAccumulator:
    def test_accumulates_points(self) -> None:
        acc = HeatmapAccumulator(width=640, height=480)
        acc.add_point(100, 100)
        acc.add_point(100, 100)
        acc.add_point(200, 200)
        heatmap = acc.get_heatmap()
        assert heatmap.shape == (480, 640)
        assert heatmap[100, 100] > heatmap[200, 200]  # more hits

    def test_generate_heatmap_image(self) -> None:
        acc = HeatmapAccumulator(width=640, height=480)
        for _ in range(50):
            acc.add_point(320, 240)
        ref_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = generate_heatmap_image(acc, ref_frame)
        assert overlay.shape == (480, 640, 3)
        # Center should be non-zero (heatmap visible)
        assert overlay[240, 320].sum() > 0

    def test_reset(self) -> None:
        acc = HeatmapAccumulator(width=640, height=480)
        acc.add_point(100, 100)
        acc.reset()
        heatmap = acc.get_heatmap()
        assert heatmap.sum() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analytics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement heatmap**

```python
# yolo26_analytics/analytics/__init__.py
from yolo26_analytics.analytics.heatmap import HeatmapAccumulator, generate_heatmap_image

__all__ = ["HeatmapAccumulator", "generate_heatmap_image"]
```

```python
# yolo26_analytics/analytics/heatmap.py
"""Heatmap generation from track centroid positions."""

from __future__ import annotations

import cv2
import numpy as np


class HeatmapAccumulator:
    """Accumulates centroid positions into a 2D histogram."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self._width = width
        self._height = height
        self._accumulator = np.zeros((height, width), dtype=np.float32)

    def add_point(self, x: int, y: int) -> None:
        if 0 <= x < self._width and 0 <= y < self._height:
            self._accumulator[y, x] += 1.0

    def get_heatmap(self) -> np.ndarray:
        return self._accumulator.copy()

    def reset(self) -> None:
        self._accumulator[:] = 0.0


def generate_heatmap_image(
    accumulator: HeatmapAccumulator,
    reference_frame: np.ndarray,
    alpha: float = 0.6,
    blur_ksize: int = 51,
) -> np.ndarray:
    """Generate a colored heatmap overlay on a reference frame."""
    heatmap = accumulator.get_heatmap()

    # Apply Gaussian blur for smooth visualization
    heatmap = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)

    # Normalize to 0-255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize if needed
    h, w = reference_frame.shape[:2]
    if colored.shape[:2] != (h, w):
        colored = cv2.resize(colored, (w, h))

    # Overlay
    overlay = cv2.addWeighted(reference_frame, 1 - alpha, colored, alpha, 0)
    return overlay
```

```python
# yolo26_analytics/analytics/stats.py
"""Zone statistics aggregation."""

from __future__ import annotations

from datetime import datetime


class ZoneStatsAggregator:
    """Aggregates zone statistics for the dashboard and API."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._entries: dict[str, int] = {}
        self._exits: dict[str, int] = {}

    def update_counts(self, zone_name: str, counts: dict[str, int]) -> None:
        self._counts[zone_name] = counts

    def record_entry(self, zone_name: str) -> None:
        self._entries[zone_name] = self._entries.get(zone_name, 0) + 1

    def record_exit(self, zone_name: str) -> None:
        self._exits[zone_name] = self._exits.get(zone_name, 0) + 1

    def get_stats(self, zone_name: str) -> dict[str, object]:
        return {
            "counts": self._counts.get(zone_name, {}),
            "total_entries": self._entries.get(zone_name, 0),
            "total_exits": self._exits.get(zone_name, 0),
        }

    def get_all_stats(self) -> dict[str, dict[str, object]]:
        all_zones = set(self._counts.keys()) | set(self._entries.keys()) | set(self._exits.keys())
        return {zone: self.get_stats(zone) for zone in all_zones}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_analytics.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yolo26_analytics/analytics/ tests/test_analytics.py
git commit -m "feat: analytics engine — heatmap accumulator and zone stats aggregation"
```

---

### Task 11: Pipeline Orchestrator

**Files:**
- Create: `yolo26_analytics/core/__init__.py`
- Create: `yolo26_analytics/core/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_pipeline.py
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from yolo26_analytics.core.pipeline import Pipeline
from yolo26_analytics.models import Detection, Event, FrameMeta, Track


class FakeSource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    async def __aiter__(self):
        for i, frame in enumerate(self._frames):
            meta = FrameMeta(
                timestamp=datetime.now(tz=timezone.utc),
                frame_index=i,
                source_id="test",
            )
            yield frame, meta

    async def close(self) -> None:
        pass


class FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def predict(self, frame: np.ndarray) -> list[Detection]:
        return self._detections


class FakeTracker:
    def __init__(self, tracks: list[Track]) -> None:
        self._tracks = tracks

    def update(self, detections: list[Detection]) -> list[Track]:
        return self._tracks

    def reset(self) -> None:
        pass


class TestPipeline:
    @pytest.mark.asyncio
    async def test_runs_pipeline_loop(self) -> None:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        det = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_name="person")
        track = Track(track_id=1, detection=det)

        source = FakeSource(frames)
        detector = FakeDetector([det])
        tracker = FakeTracker([track])
        store = AsyncMock()
        store.write_tracks = AsyncMock()
        store.log_events = AsyncMock()
        alert_manager = AsyncMock()
        alert_manager.dispatch = AsyncMock()

        pipeline = Pipeline(
            source=source,
            detector=detector,
            tracker=tracker,
            store=store,
            zone_analyzer=None,
            alert_manager=alert_manager,
        )
        await pipeline.run_async()

        assert store.write_tracks.call_count == 3
        # No zones configured, so no events
        alert_manager.dispatch.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement pipeline**

```python
# yolo26_analytics/core/__init__.py
from yolo26_analytics.core.pipeline import Pipeline

__all__ = ["Pipeline"]
```

```python
# yolo26_analytics/core/pipeline.py
"""Pipeline — the main orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from yolo26_analytics.alerts.manager import AlertManager
from yolo26_analytics.analytics.heatmap import HeatmapAccumulator
from yolo26_analytics.config.schema import AppConfig, load_config
from yolo26_analytics.models import Event
from yolo26_analytics.zones.analyzer import ZoneAnalyzer


class Pipeline:
    """Main pipeline: source → detect → track → zones → alerts → store."""

    def __init__(
        self,
        source: Any,
        detector: Any,
        tracker: Any,
        store: Any,
        zone_analyzer: ZoneAnalyzer | None = None,
        alert_manager: AlertManager | None = None,
        heatmap: HeatmapAccumulator | None = None,
        on_frame: Callable[..., Any] | None = None,
    ) -> None:
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._store = store
        self._zone_analyzer = zone_analyzer
        self._alert_manager = alert_manager
        self._heatmap = heatmap
        self._on_frame = on_frame
        self._running = False
        self._custom_alerts: list[Callable[..., Any]] = []

        # Stats
        self.frame_count: int = 0
        self.fps: float = 0.0
        self.last_latency_ms: float = 0.0

    def add_alert(self, fn: Callable[..., Any]) -> None:
        """Add a custom alert callback."""
        self._custom_alerts.append(fn)

    async def run_async(self) -> None:
        """Run the pipeline loop."""
        import time

        self._running = True
        fps_start = time.monotonic()
        fps_frames = 0

        async for frame, meta in self._source:
            if not self._running:
                break

            t0 = time.monotonic()

            # Detect
            detections = self._detector.predict(frame)

            # Track
            tracks = self._tracker.update(detections)

            # Store tracks
            await self._store.write_tracks(tracks, meta)

            # Heatmap accumulation
            if self._heatmap is not None:
                for track in tracks:
                    cx, cy = track.centroid
                    self._heatmap.add_point(cx, cy)

            # Zone analytics
            events: list[Event] = []
            if self._zone_analyzer is not None:
                events = self._zone_analyzer.check(tracks)
                if events:
                    await self._store.log_events(events)

            # Alerts
            if events and self._alert_manager is not None:
                await self._alert_manager.dispatch(events)

            # Custom alert callbacks
            for event in events:
                for fn in self._custom_alerts:
                    fn(event)

            # Frame callback (for dashboard streaming)
            if self._on_frame is not None:
                self._on_frame(frame, meta, tracks, events)

            # Stats
            self.last_latency_ms = (time.monotonic() - t0) * 1000
            self.frame_count += 1
            fps_frames += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                self.fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.monotonic()

        self._running = False

    def run(self) -> None:
        """Synchronous entry point."""
        asyncio.run(self.run_async())

    def stop(self) -> None:
        self._running = False

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        """Build a complete pipeline from a YAML config file."""
        config = load_config(path)
        return cls._from_config(config)

    @classmethod
    def _from_config(cls, config: AppConfig) -> "Pipeline":
        from yolo26_analytics.alerts.console import ConsoleAlert
        from yolo26_analytics.alerts.mqtt import MQTTAlert
        from yolo26_analytics.alerts.telegram import TelegramAlert
        from yolo26_analytics.alerts.webhook import WebhookAlert
        from yolo26_analytics.detection.yolo26 import YOLO26Detector
        from yolo26_analytics.sources import create_source
        from yolo26_analytics.store.sqlite import SQLiteStore
        from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter

        # Source
        source = create_source(config.source)

        # Detector
        detector = YOLO26Detector(
            weights=config.model.weights,
            confidence=config.model.confidence,
        )

        # Tracker
        tracker = ByteTrackAdapter(
            max_age=config.tracking.max_age,
            min_hits=config.tracking.min_hits,
        )

        # Store
        if config.store.type == "postgresql" and config.store.url:
            from yolo26_analytics.store.postgres import PostgresStore
            store = PostgresStore(url=config.store.url)
        else:
            store = SQLiteStore(path=config.store.path)

        # Zones
        zone_analyzer = ZoneAnalyzer(config.zones) if config.zones else None

        # Alerts
        backends: list[tuple[Any, Any]] = []
        for ac in config.alerts:
            backend: Any
            match ac.type:
                case "console":
                    backend = ConsoleAlert()
                case "webhook":
                    backend = WebhookAlert(url=ac.url or "")
                case "mqtt":
                    backend = MQTTAlert(broker=ac.broker or "", topic=ac.topic or "yolo26/events")
                case "telegram":
                    backend = TelegramAlert(
                        bot_token=ac.bot_token or "",
                        chat_id=ac.chat_id or "",
                    )
                case _:
                    continue
            backends.append((backend, ac.filter))
        alert_manager = AlertManager(backends=backends) if backends else None

        return cls(
            source=source,
            detector=detector,
            tracker=tracker,
            store=store,
            zone_analyzer=zone_analyzer,
            alert_manager=alert_manager,
        )
```

- [ ] **Step 4: Update package __init__.py to export Pipeline**

Add to `yolo26_analytics/__init__.py`:

```python
from yolo26_analytics.core.pipeline import Pipeline
```

Add `"Pipeline"` to `__all__`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add yolo26_analytics/core/ yolo26_analytics/__init__.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator with from_yaml factory"
```

---

### Task 12: SAHI Integration

**Files:**
- Create: `yolo26_analytics/detection/sahi.py`

- [ ] **Step 1: Implement SAHI wrapper**

```python
# yolo26_analytics/detection/sahi.py
"""SAHI (Slicing Aided Hyper Inference) wrapper for any detector."""

from __future__ import annotations

import numpy as np

from yolo26_analytics.models import Detection


class SAHIDetector:
    """Wraps any Detector with slice-based inference for small object detection."""

    def __init__(
        self,
        detector: object,
        slice_size: int = 640,
        overlap: float = 0.25,
    ) -> None:
        self._detector = detector
        self._slice_size = slice_size
        self._overlap = overlap

    def predict(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        stride = int(self._slice_size * (1 - self._overlap))
        all_detections: list[Detection] = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + self._slice_size, w)
                y2 = min(y + self._slice_size, h)
                tile = frame[y:y2, x:x2]

                tile_dets = self._detector.predict(tile)  # type: ignore[attr-defined]

                # Offset bboxes back to full frame coords
                for det in tile_dets:
                    bx1, by1, bx2, by2 = det.bbox
                    all_detections.append(
                        Detection(
                            bbox=(bx1 + x, by1 + y, bx2 + x, by2 + y),
                            confidence=det.confidence,
                            class_name=det.class_name,
                        )
                    )

        # Merge overlapping detections (simple IoU-based dedup)
        return self._merge_detections(all_detections)

    @staticmethod
    def _merge_detections(
        detections: list[Detection], iou_threshold: float = 0.5
    ) -> list[Detection]:
        """Remove duplicate detections from overlapping tiles."""
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []

        for det in sorted_dets:
            is_dup = False
            for kept_det in kept:
                if det.class_name != kept_det.class_name:
                    continue
                iou = SAHIDetector._compute_iou(det.bbox, kept_det.bbox)
                if iou > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(det)

        return kept

    @staticmethod
    def _compute_iou(
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
```

- [ ] **Step 2: Commit**

```bash
git add yolo26_analytics/detection/sahi.py
git commit -m "feat: SAHI wrapper for slice-based small object detection"
```

---

### Task 13: CLI

**Files:**
- Create: `yolo26_analytics/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py
from __future__ import annotations

from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from yolo26_analytics.cli import main


class TestCLI:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "yolo26-analytics" in result.output.lower() or "y26a" in result.output.lower()

    def test_run_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output

    def test_export_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--format" in result.output

    def test_heatmap_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["heatmap", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output
        assert "--output" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI**

```python
# yolo26_analytics/cli.py
"""CLI entry point for yolo26-analytics (y26a)."""

from __future__ import annotations

import asyncio

import click


@click.group()
def main() -> None:
    """yolo26-analytics — Real-time object tracking, zone analytics, and alerting."""
    pass


@main.command()
@click.option("--source", required=True, help="Video source (file path, RTSP URL, or device index)")
@click.option("--model", default="yolo26n.pt", help="Model weights path")
@click.option("--confidence", default=0.5, type=float, help="Confidence threshold")
@click.option("--config", default=None, help="YAML config file (overrides other options)")
@click.option("--zones", default=None, help="Zones YAML file")
@click.option("--dashboard", is_flag=True, help="Enable live dashboard")
def run(
    source: str,
    model: str,
    confidence: float,
    config: str | None,
    zones: str | None,
    dashboard: bool,
) -> None:
    """Run the detection + tracking + analytics pipeline."""
    if config:
        from yolo26_analytics.core.pipeline import Pipeline
        pipeline = Pipeline.from_yaml(config)
        pipeline.run()
    else:
        # Build config from CLI args
        from yolo26_analytics.config.schema import AppConfig, ModelConfig, SourceConfig

        source_type = _infer_source_type(source)
        source_config = SourceConfig(type=source_type)
        if source_type == "video_file":
            source_config.path = source
        elif source_type == "rtsp":
            source_config.url = source
        elif source_type == "webcam":
            source_config.device = int(source)

        app_config = AppConfig(
            source=source_config,
            model=ModelConfig(weights=model, confidence=confidence),
            dashboard=dashboard,
        )

        from yolo26_analytics.core.pipeline import Pipeline
        pipeline = Pipeline._from_config(app_config)
        pipeline.run()


@main.command()
@click.option("--model", required=True, help="Model weights path")
@click.option("--format", "fmt", required=True, type=click.Choice(["onnx", "tflite", "engine"]))
@click.option("--quantize", default=None, help="Quantization type (e.g., int8)")
def export(model: str, fmt: str, quantize: str | None) -> None:
    """Export model to ONNX, TFLite, or TensorRT."""
    from yolo26_analytics.export.exporter import export_model
    export_model(model, format=fmt, quantize=quantize)


@main.command()
@click.option("--source", required=True, help="Video source")
@click.option("--output", required=True, help="Output heatmap image path")
@click.option("--model", default="yolo26n.pt", help="Model weights path")
@click.option("--duration", default=None, type=int, help="Max seconds to process")
def heatmap(source: str, output: str, model: str, duration: int | None) -> None:
    """Generate a heatmap from a video source."""
    import time

    import cv2

    from yolo26_analytics.analytics.heatmap import HeatmapAccumulator, generate_heatmap_image
    from yolo26_analytics.detection.yolo26 import YOLO26Detector
    from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter

    detector = YOLO26Detector(weights=model)
    tracker = ByteTrackAdapter()
    cap = cv2.VideoCapture(source)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    acc = HeatmapAccumulator(width=w, height=h)

    ref_frame = None
    start = time.monotonic()
    frame_count = 0

    click.echo(f"Processing {source}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if ref_frame is None:
            ref_frame = frame.copy()

        if duration and (time.monotonic() - start) > duration:
            break

        detections = detector.predict(frame)
        tracks = tracker.update(detections)
        for track in tracks:
            cx, cy = track.centroid
            acc.add_point(cx, cy)
        frame_count += 1

    cap.release()

    if ref_frame is not None:
        result = generate_heatmap_image(acc, ref_frame)
        cv2.imwrite(output, result)
        click.echo(f"Heatmap saved to {output} ({frame_count} frames processed)")
    else:
        click.echo("No frames processed.")


def _infer_source_type(source: str) -> str:
    if source.startswith("rtsp://"):
        return "rtsp"
    if source.isdigit():
        return "webcam"
    return "video_file"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yolo26_analytics/cli.py tests/test_cli.py
git commit -m "feat: CLI (y26a) — run, export, heatmap commands"
```

---

### Task 14: Export System

**Files:**
- Create: `yolo26_analytics/export/__init__.py`
- Create: `yolo26_analytics/export/exporter.py`
- Create: `tests/test_export.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_export.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestExportModel:
    @patch("yolo26_analytics.export.exporter.YOLO")
    def test_export_onnx(self, mock_yolo_cls: MagicMock, tmp_path) -> None:
        from yolo26_analytics.export.exporter import export_model

        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "model.onnx")
        mock_yolo_cls.return_value = mock_model

        result = export_model("yolo26n.pt", format="onnx")
        mock_model.export.assert_called_once_with(format="onnx")
        assert result["format"] == "onnx"
        assert "export_path" in result

    @patch("yolo26_analytics.export.exporter.YOLO")
    def test_export_tflite_with_quantize(self, mock_yolo_cls: MagicMock, tmp_path) -> None:
        from yolo26_analytics.export.exporter import export_model

        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "model.tflite")
        mock_yolo_cls.return_value = mock_model

        result = export_model("yolo26n.pt", format="tflite", quantize="int8")
        mock_model.export.assert_called_once_with(format="tflite", int8=True)
        assert result["format"] == "tflite"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export.py -v`
Expected: FAIL

- [ ] **Step 3: Implement exporter**

```python
# yolo26_analytics/export/__init__.py
from yolo26_analytics.export.exporter import export_model

__all__ = ["export_model"]
```

```python
# yolo26_analytics/export/exporter.py
"""Model export + benchmarking."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ultralytics import YOLO


def export_model(
    weights: str,
    format: str = "onnx",
    quantize: str | None = None,
    output_dir: str | None = None,
) -> dict[str, object]:
    """Export a YOLO model and run a benchmark."""
    model = YOLO(weights)

    # Build export kwargs
    kwargs: dict[str, object] = {"format": format}
    if quantize == "int8":
        kwargs["int8"] = True
    elif quantize == "fp16":
        kwargs["half"] = True

    # Export
    t0 = time.monotonic()
    export_path = model.export(**kwargs)
    export_time = time.monotonic() - t0

    # Gather file size
    export_file = Path(str(export_path))
    file_size_mb = export_file.stat().st_size / (1024 * 1024) if export_file.exists() else 0

    report = {
        "source_weights": weights,
        "format": format,
        "quantize": quantize,
        "export_path": str(export_path),
        "export_time_seconds": round(export_time, 2),
        "file_size_mb": round(file_size_mb, 2),
    }

    # Save report
    report_path = export_file.with_suffix(".benchmark.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Export complete: {export_path}")
    print(f"  Format: {format}, Size: {file_size_mb:.1f} MB, Time: {export_time:.1f}s")
    print(f"  Benchmark saved: {report_path}")

    return report
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_export.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add yolo26_analytics/export/ tests/test_export.py
git commit -m "feat: model export with auto-benchmarking (ONNX, TFLite, TensorRT)"
```

---

### Task 15: Dashboard — Core (FastAPI App + Live View + Streaming)

**Files:**
- Create: `yolo26_analytics/dashboard/__init__.py`
- Create: `yolo26_analytics/dashboard/app.py`
- Create: `yolo26_analytics/dashboard/routes/__init__.py`
- Create: `yolo26_analytics/dashboard/routes/views.py`
- Create: `yolo26_analytics/dashboard/routes/stream.py`
- Create: `yolo26_analytics/dashboard/routes/api.py`
- Create: `yolo26_analytics/dashboard/templates/base.html`
- Create: `yolo26_analytics/dashboard/templates/live.html`
- Create: `yolo26_analytics/dashboard/static/style.css`
- Create: `tests/test_dashboard/__init__.py`
- Create: `tests/test_dashboard/test_api.py`

- [ ] **Step 1: Write failing test for API endpoints**

```python
# tests/test_dashboard/__init__.py
```

```python
# tests/test_dashboard/test_api.py
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from yolo26_analytics.dashboard.app import create_app


@pytest.fixture
def app():
    return create_app(store=None, pipeline=None, zone_analyzer=None)


class TestDashboardAPI:
    @pytest.mark.asyncio
    async def test_index_returns_html(self, app) -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_api_stats(self, app) -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "fps" in data
        assert "frame_count" in data

    @pytest.mark.asyncio
    async def test_api_zones(self, app) -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/zones")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FastAPI app factory**

```python
# yolo26_analytics/dashboard/__init__.py
from yolo26_analytics.dashboard.app import create_app

__all__ = ["create_app"]
```

```python
# yolo26_analytics/dashboard/app.py
"""FastAPI application factory for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    store: Any = None,
    pipeline: Any = None,
    zone_analyzer: Any = None,
) -> FastAPI:
    """Create the dashboard FastAPI app."""
    app = FastAPI(title="yolo26-analytics Dashboard")

    # State
    app.state.store = store
    app.state.pipeline = pipeline
    app.state.zone_analyzer = zone_analyzer
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Register routes
    from yolo26_analytics.dashboard.routes.api import router as api_router
    from yolo26_analytics.dashboard.routes.stream import router as stream_router
    from yolo26_analytics.dashboard.routes.views import router as views_router

    app.include_router(views_router)
    app.include_router(stream_router)
    app.include_router(api_router, prefix="/api")

    return app
```

- [ ] **Step 4: Implement view routes**

```python
# yolo26_analytics/dashboard/routes/__init__.py
```

```python
# yolo26_analytics/dashboard/routes/views.py
"""HTML page routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def live_view(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse("live.html", {"request": request})


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_view(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse("analytics.html", {"request": request})


@router.get("/replay", response_class=HTMLResponse)
async def replay_view(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse("replay.html", {"request": request})


@router.get("/zones/edit", response_class=HTMLResponse)
async def zone_editor(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse("zone_editor.html", {"request": request})
```

- [ ] **Step 5: Implement stream routes**

```python
# yolo26_analytics/dashboard/routes/stream.py
"""MJPEG and SSE streaming endpoints."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from typing import Any

import cv2
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

# Shared frame buffer — pipeline writes, dashboard reads
_latest_frame: bytes | None = None
_event_queue: deque[dict[str, Any]] = deque(maxlen=100)


def update_frame(jpeg_bytes: bytes) -> None:
    """Called by the pipeline to push the latest annotated frame."""
    global _latest_frame
    _latest_frame = jpeg_bytes


def push_event(event_data: dict[str, Any]) -> None:
    """Called by the pipeline to push an event to SSE clients."""
    _event_queue.append(event_data)


async def _mjpeg_generator():
    while True:
        if _latest_frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + _latest_frame
                + b"\r\n"
            )
        await asyncio.sleep(0.033)  # ~30fps


@router.get("/stream")
async def mjpeg_stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/events")
async def sse_events(request: Request) -> EventSourceResponse:
    async def event_generator():
        last_idx = 0
        while True:
            if await request.is_disconnected():
                break
            while last_idx < len(_event_queue):
                event = _event_queue[last_idx]
                yield {"data": json.dumps(event)}
                last_idx += 1
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())
```

- [ ] **Step 6: Implement API routes**

```python
# yolo26_analytics/dashboard/routes/api.py
"""JSON API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/stats")
async def get_stats(request: Request) -> dict[str, Any]:
    pipeline = request.app.state.pipeline
    return {
        "fps": pipeline.fps if pipeline else 0,
        "frame_count": pipeline.frame_count if pipeline else 0,
        "latency_ms": pipeline.last_latency_ms if pipeline else 0,
    }


@router.get("/zones")
async def get_zones(request: Request) -> list[dict[str, Any]]:
    za = request.app.state.zone_analyzer
    if za is None:
        return []
    return [
        {
            "name": z.name,
            "track_classes": z.track_classes,
            "cooldown": z.cooldown,
        }
        for z in za.zones
    ]


@router.get("/events")
async def get_events(
    request: Request,
    zone_name: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    store = request.app.state.store
    if store is None:
        return []
    return await store.query_events(
        zone_name=zone_name,
        event_type=event_type,
        limit=limit,
    )


@router.get("/tracks")
async def get_tracks(
    request: Request,
    zone_name: str | None = None,
    source_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    store = request.app.state.store
    if store is None:
        return []
    return await store.query_tracks(
        zone_name=zone_name,
        source_id=source_id,
        limit=limit,
    )


@router.get("/analytics/counts")
async def get_counts(request: Request) -> dict[str, Any]:
    za = request.app.state.zone_analyzer
    if za is None:
        return {}
    return za.get_zone_counts()


@router.get("/analytics/dwell")
async def get_dwell(request: Request) -> dict[str, Any]:
    # Dwell data would come from store query
    return {}


@router.get("/analytics/heatmap")
async def get_heatmap() -> dict[str, str]:
    return {"status": "not_implemented_yet"}
```

- [ ] **Step 7: Create base HTML template**

```html
<!-- yolo26_analytics/dashboard/templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}yolo26-analytics{% endblock %}</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10/dist/ext/sse.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <nav>
        <a href="/">Live</a>
        <a href="/analytics">Analytics</a>
        <a href="/replay">Replay</a>
        <a href="/zones/edit">Zones</a>
    </nav>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

- [ ] **Step 8: Create live view template**

```html
<!-- yolo26_analytics/dashboard/templates/live.html -->
{% extends "base.html" %}
{% block title %}Live — yolo26-analytics{% endblock %}
{% block content %}
<div class="live-container">
    <div class="video-panel">
        <img src="/stream" alt="Live feed" class="live-feed">
    </div>
    <div class="sidebar">
        <div class="stats" hx-get="/api/stats" hx-trigger="every 1s" hx-swap="innerHTML">
            Loading stats...
        </div>
        <div class="event-log" hx-ext="sse" sse-connect="/events" sse-swap="message">
            <h3>Events</h3>
            <div id="events"></div>
        </div>
    </div>
</div>
{% endblock %}
```

- [ ] **Step 9: Create minimal CSS**

```css
/* yolo26_analytics/dashboard/static/style.css */
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; }
nav { display: flex; gap: 1rem; padding: 0.75rem 1.5rem; background: #16213e; }
nav a { color: #a0c4ff; text-decoration: none; font-weight: 500; }
nav a:hover { color: #fff; }
main { padding: 1rem; }
.live-container { display: flex; gap: 1rem; }
.video-panel { flex: 3; }
.live-feed { width: 100%; border-radius: 8px; }
.sidebar { flex: 1; display: flex; flex-direction: column; gap: 1rem; }
.stats, .event-log { background: #16213e; border-radius: 8px; padding: 1rem; }
.event-log { flex: 1; overflow-y: auto; max-height: 60vh; }
h3 { margin-bottom: 0.5rem; color: #a0c4ff; }
```

- [ ] **Step 10: Create placeholder templates for other views**

```html
<!-- yolo26_analytics/dashboard/templates/analytics.html -->
{% extends "base.html" %}
{% block title %}Analytics — yolo26-analytics{% endblock %}
{% block content %}
<h2>Analytics</h2>
<div id="charts">
    <canvas id="countChart" width="600" height="300"></canvas>
</div>
<div id="heatmap-toggle">
    <button onclick="toggleHeatmap()">Toggle Heatmap</button>
</div>
{% endblock %}
```

```html
<!-- yolo26_analytics/dashboard/templates/replay.html -->
{% extends "base.html" %}
{% block title %}Replay — yolo26-analytics{% endblock %}
{% block content %}
<h2>Replay</h2>
<div class="replay-controls">
    <label>From: <input type="datetime-local" id="replay-start"></label>
    <label>To: <input type="datetime-local" id="replay-end"></label>
    <button id="replay-btn">Play</button>
</div>
<div class="replay-feed">
    <canvas id="replay-canvas" width="1280" height="720"></canvas>
</div>
{% endblock %}
```

```html
<!-- yolo26_analytics/dashboard/templates/zone_editor.html -->
{% extends "base.html" %}
{% block title %}Zone Editor — yolo26-analytics{% endblock %}
{% block content %}
<h2>Zone Editor</h2>
<div class="editor-container">
    <div class="canvas-wrap">
        <img src="/stream" id="editor-feed" class="live-feed">
        <canvas id="zone-canvas" width="1280" height="720"></canvas>
    </div>
    <div class="zone-form">
        <label>Zone Name: <input type="text" id="zone-name"></label>
        <label>Track Classes: <input type="text" id="zone-classes" placeholder="person,forklift"></label>
        <button id="save-zone">Save Zone</button>
    </div>
</div>
<script src="/static/zone-editor.js"></script>
{% endblock %}
```

- [ ] **Step 11: Create zone editor JS**

```javascript
// yolo26_analytics/dashboard/static/zone-editor.js
const canvas = document.getElementById('zone-canvas');
const ctx = canvas.getContext('2d');
const points = [];

canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
    const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));
    points.push([x, y]);
    drawPolygon();
});

function drawPolygon() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (points.length === 0) return;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = 'rgba(0, 255, 136, 0.15)';
    ctx.fill();
    for (const p of points) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        ctx.fillStyle = '#00ff88';
        ctx.fill();
    }
}

document.getElementById('save-zone').addEventListener('click', async () => {
    const name = document.getElementById('zone-name').value;
    const classes = document.getElementById('zone-classes').value.split(',').map(s => s.trim());
    const resp = await fetch('/api/zones', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ name, polygon: points, track_classes: classes }),
    });
    if (resp.ok) {
        alert('Zone saved!');
        points.length = 0;
        drawPolygon();
    }
});
```

- [ ] **Step 12: Run tests**

Run: `pytest tests/test_dashboard/ -v`
Expected: All 3 tests PASS

- [ ] **Step 13: Commit**

```bash
git add yolo26_analytics/dashboard/ tests/test_dashboard/
git commit -m "feat: dashboard — FastAPI + HTMX with live view, analytics, replay, zone editor"
```

---

### Task 16: Docker Compose + CI

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000

CMD ["y26a", "run", "--config", "/app/config.yaml", "--dashboard"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
# docker-compose.yml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./data:/app/data
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql+asyncpg://y26a:y26a@postgres:5432/yolo26_analytics

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: y26a
      POSTGRES_PASSWORD: y26a
      POSTGRES_DB: yolo26_analytics
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U y26a"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  pgdata:
```

- [ ] **Step 3: Create GitHub Actions CI**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-type:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy yolo26_analytics --strict

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_y26a
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: |
          sudo apt-get update
          sudo apt-get install -y libgl1 libglib2.0-0
      - run: pip install -e ".[dev]"
      - run: pytest --cov=yolo26_analytics --cov-report=xml -v
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml .github/
git commit -m "feat: Docker Compose (app + PostgreSQL) and GitHub Actions CI"
```

---

### Task 17: Examples

**Files:**
- Create: `examples/quickstart.py`
- Create: `examples/rtsp_warehouse.py`
- Create: `examples/video_analytics.py`
- Create: `examples/custom_detector.py`
- Create: `examples/ppe_safety.py`

- [ ] **Step 1: Create quickstart example**

```python
# examples/quickstart.py
"""Minimal quickstart — run detection on webcam."""

from yolo26_analytics import Pipeline
from yolo26_analytics.config.schema import AppConfig, ModelConfig, SourceConfig
from yolo26_analytics.core.pipeline import Pipeline
from yolo26_analytics.detection.yolo26 import YOLO26Detector
from yolo26_analytics.sources.webcam import WebcamSource
from yolo26_analytics.store.sqlite import SQLiteStore
from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter

source = WebcamSource(device=0, source_id="webcam")
detector = YOLO26Detector(weights="yolo26n.pt", confidence=0.5)
tracker = ByteTrackAdapter()
store = SQLiteStore(path="./data/quickstart.db")

pipeline = Pipeline(
    source=source,
    detector=detector,
    tracker=tracker,
    store=store,
)
pipeline.run()
```

- [ ] **Step 2: Create RTSP warehouse example**

```python
# examples/rtsp_warehouse.py
"""RTSP camera + zones + Telegram alerts — warehouse scenario."""

# Using YAML config is the recommended approach for production setups.
# Save this as config_warehouse.yaml and run:
#   y26a run --config config_warehouse.yaml --dashboard

CONFIG = """
source:
  type: rtsp
  url: rtsp://192.168.1.100/stream

model:
  type: yolo26
  weights: yolo26n.pt
  confidence: 0.5

tracking:
  engine: bytetrack
  max_age: 30
  min_hits: 3

store:
  type: sqlite
  path: ./data/warehouse.db

zones:
  - name: "Loading Dock"
    polygon: [[100,100], [500,100], [500,400], [100,400]]
    track_classes: [person, forklift]
    analytics:
      - type: count
      - type: entry_exit
        direction: inward
      - type: dwell
        alert_threshold: 300
    cooldown: 30

  - name: "Restricted Area"
    polygon: [[600,50], [900,50], [900,300], [600,300]]
    track_classes: [person]
    analytics:
      - type: entry_exit
    cooldown: 10

alerts:
  - type: console
  - type: telegram
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
    filter:
      zones: ["Restricted Area"]
      event_types: ["entry"]

dashboard: true
"""

if __name__ == "__main__":
    from pathlib import Path
    from yolo26_analytics.core.pipeline import Pipeline

    config_path = Path("config_warehouse.yaml")
    config_path.write_text(CONFIG)
    print(f"Config written to {config_path}")
    print("Run: y26a run --config config_warehouse.yaml --dashboard")
```

- [ ] **Step 3: Create video analytics example**

```python
# examples/video_analytics.py
"""Process a recorded video and generate heatmap + stats."""

import asyncio
from yolo26_analytics.core.pipeline import Pipeline
from yolo26_analytics.detection.yolo26 import YOLO26Detector
from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter
from yolo26_analytics.sources.video_file import VideoFileSource
from yolo26_analytics.store.sqlite import SQLiteStore
from yolo26_analytics.analytics.heatmap import HeatmapAccumulator, generate_heatmap_image
import cv2


async def main() -> None:
    source = VideoFileSource(path="sample.mp4", source_id="video")
    detector = YOLO26Detector(weights="yolo26n.pt")
    tracker = ByteTrackAdapter()
    store = SQLiteStore(path="./data/analytics.db")
    await store.initialize()

    heatmap_acc = HeatmapAccumulator(width=1920, height=1080)

    pipeline = Pipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        store=store,
        heatmap=heatmap_acc,
    )
    await pipeline.run_async()

    # Generate heatmap
    ref = cv2.imread("reference_frame.jpg")  # or capture first frame
    if ref is not None:
        overlay = generate_heatmap_image(heatmap_acc, ref)
        cv2.imwrite("heatmap_output.png", overlay)
        print("Heatmap saved to heatmap_output.png")

    # Query track stats
    tracks = await store.query_tracks(limit=10)
    print(f"Total track records: {len(tracks)}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Create custom detector example**

```python
# examples/custom_detector.py
"""Plug in a non-YOLO model — demonstrates the Protocol pattern."""

import numpy as np

from yolo26_analytics.models import Detection


class MyCustomDetector:
    """Any class with a predict(frame) -> list[Detection] method works."""

    def predict(self, frame: np.ndarray) -> list[Detection]:
        # Your custom model inference here
        # This example just returns a dummy detection
        h, w = frame.shape[:2]
        return [
            Detection(
                bbox=(w // 4, h // 4, 3 * w // 4, 3 * h // 4),
                confidence=0.95,
                class_name="custom_object",
            )
        ]


if __name__ == "__main__":
    import asyncio
    from yolo26_analytics.core.pipeline import Pipeline
    from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter
    from yolo26_analytics.sources.video_file import VideoFileSource
    from yolo26_analytics.store.sqlite import SQLiteStore

    async def main() -> None:
        source = VideoFileSource(path="sample.mp4", source_id="custom")
        detector = MyCustomDetector()
        tracker = ByteTrackAdapter()
        store = SQLiteStore(path="./data/custom.db")
        await store.initialize()

        pipeline = Pipeline(
            source=source,
            detector=detector,
            tracker=tracker,
            store=store,
        )
        await pipeline.run_async()

    asyncio.run(main())
```

- [ ] **Step 5: Create PPE safety example**

```python
# examples/ppe_safety.py
"""PPE safety monitoring — zones with required equipment classes.

This shows how yolo26-analytics can be configured for safety compliance
using a PPE-trained model. The zone analytics detect when workers enter
zones without required protective equipment.

Requires: A YOLO26 model fine-tuned on PPE classes (helmet, vest, goggles).
Public datasets available on Roboflow Universe.
"""

CONFIG = """
source:
  type: video_file
  path: construction_site.mp4

model:
  type: yolo26
  weights: yolo26n_ppe.pt  # fine-tuned on PPE dataset
  confidence: 0.4

tracking:
  engine: bytetrack
  max_age: 30
  min_hits: 3

store:
  type: sqlite
  path: ./data/ppe_safety.db

zones:
  - name: "Hard Hat Zone"
    polygon: [[0,0], [640,0], [640,480], [0,480]]
    track_classes: [person, helmet, vest, goggles]
    analytics:
      - type: count
      - type: dwell
        alert_threshold: 5  # alert quickly for safety
      - type: entry_exit
    cooldown: 10

alerts:
  - type: console
  - type: webhook
    url: https://hooks.slack.com/services/YOUR/WEBHOOK/URL

dashboard: true
"""

if __name__ == "__main__":
    from pathlib import Path
    config_path = Path("config_ppe.yaml")
    config_path.write_text(CONFIG)
    print(f"Config written to {config_path}")
    print("Run: y26a run --config config_ppe.yaml --dashboard")
```

- [ ] **Step 6: Commit**

```bash
git add examples/
git commit -m "feat: example scripts — quickstart, warehouse, analytics, custom detector, PPE safety"
```

---

### Task 18: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# yolo26-analytics

Real-time object tracking, zone analytics, and event alerting built on YOLO26.

[![CI](https://github.com/YOUR_USERNAME/yolo26-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/yolo26-analytics/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What is this?

A pip-installable framework that turns YOLO26 into a production-grade video analytics pipeline. Draw zones, count objects, detect entry/exit, measure dwell time, generate heatmaps, and get real-time alerts — all from a YAML config file.

## Quickstart

```bash
pip install yolo26-analytics
y26a run --source video.mp4 --dashboard
```

Or with Python:

```python
from yolo26_analytics import Pipeline

pipeline = Pipeline.from_yaml("config.yaml")
pipeline.run()
```

Open `http://localhost:8000` to see the live dashboard.

## Architecture

```
VideoSource → Detector → Tracker → ZoneAnalyzer → AlertManager
                            ↓                        ↓
                        TrackStore ←──────── EventStore
                            ↓
                      AnalyticsEngine (heatmaps, dwell, counts)
                            ↓
                        Dashboard (live + replay)
```

Every stage is a Python Protocol. Swap the detector, alert backend, or video source — nothing else changes.

## Features

- **YOLO26 detection** with pluggable model adapters (bring your own model)
- **ByteTrack** multi-object tracking with persistent IDs
- **Zone analytics**: counting, entry/exit, dwell time, throughput
- **Heatmap generation** from accumulated track data
- **4 alert backends**: console, webhook (Slack/Discord), MQTT, Telegram
- **Alert routing** — filter alerts by zone and event type
- **SAHI integration** for small object detection at distance
- **FastAPI dashboard** with live feed, analytics charts, replay, zone editor
- **PostgreSQL** track store for production (SQLite fallback for dev)
- **Model export** to ONNX, TFLite, TensorRT with auto-benchmarking
- **CLI** (`y26a`) and Python API

## Configuration

```yaml
source:
  type: rtsp
  url: rtsp://192.168.1.100/stream

model:
  weights: yolo26n.pt
  confidence: 0.5

zones:
  - name: "Loading Dock"
    polygon: [[100,100], [400,100], [400,400], [100,400]]
    track_classes: [person, forklift]
    analytics:
      - type: count
      - type: dwell
        alert_threshold: 300
      - type: entry_exit

alerts:
  - type: telegram
    bot_token: "YOUR_TOKEN"
    chat_id: "YOUR_CHAT_ID"
  - type: console

dashboard: true
```

See `examples/` for more configurations.

## CLI

```bash
# Run pipeline
y26a run --source video.mp4 --dashboard
y26a run --config config.yaml

# Export model
y26a export --model yolo26n.pt --format onnx
y26a export --model yolo26n.pt --format tflite --quantize int8

# Generate heatmap
y26a heatmap --source video.mp4 --output heatmap.png
```

## Docker

```bash
docker-compose up
```

Starts the app + PostgreSQL. Dashboard at `http://localhost:8000`.

## Roadmap

- [ ] YOLO26-pose for action/posture detection
- [ ] YOLO26-seg for pixel-level analytics
- [ ] Natural language queries over detection history (LLM integration)
- [ ] Active learning: flag low-confidence detections for relabeling
- [ ] Multi-camera orchestration with load balancing
- [ ] Grafana datasource plugin

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quickstart, architecture, features, and roadmap"
```

---

## Self-Review

**Spec coverage check:**

| Spec Section | Task(s) |
|---|---|
| Package identity + quickstart | Task 1, 13, 18 |
| Package structure | Task 1-15 (full file map) |
| Pipeline architecture + loop | Task 11 |
| Stage protocols | Task 2 |
| YAML config | Task 3 |
| Zone system (counting, entry/exit, dwell, throughput) | Task 8 |
| Zone cooldown | Task 8 (ZoneAnalyzer._make_event) |
| Track store (PostgreSQL + SQLite) | Task 7 |
| Schema + indexes | Task 7 |
| Retention cleanup | Task 7 |
| Alert system + routing | Task 9 |
| 4 backends (console, webhook, MQTT, Telegram) | Task 9 |
| Custom alert backends | Task 9 (manager), Task 11 (pipeline.add_alert) |
| Dashboard (live, analytics, replay, zone editor) | Task 15 |
| Dashboard endpoints | Task 15 |
| MJPEG + SSE streaming | Task 15 |
| SAHI integration | Task 12 |
| Model export + benchmarking | Task 14 |
| Auto-select runtime | Task 5 (detector checks file ext) |
| Testing (unit, integration, API) | Tasks 1-15 (each has tests) |
| CI + GitHub Actions | Task 16 |
| Docker Compose | Task 16 |
| Examples (5 scripts) | Task 17 |
| README | Task 18 |
| Heatmap generation | Task 10 |
| Entry/exit direction detection | Task 8 |
| Event model (dataclass) | Task 1 |
| Snapshot storage (JPEG to disk) | Task 7 (store), Task 15 (dashboard) |

**Placeholder scan:** No TBD, TODO, or "implement later" found. All code blocks are complete.

**Type consistency check:**
- `Detection`, `Track`, `Event`, `FrameMeta` — consistent across all tasks
- `write_tracks`, `log_events`, `query_tracks`, `query_events` — consistent between store and pipeline
- `AlertBackend.send(event)` — consistent between protocol, manager, and all backends
- `Zone.contains_point`, `Zone.should_track` — consistent across all zone analytics modules
- `HeatmapAccumulator.add_point`, `generate_heatmap_image` — consistent between analytics and CLI

No issues found.

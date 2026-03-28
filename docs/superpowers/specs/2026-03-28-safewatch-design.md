# yolo26-analytics — Design Spec

## Overview

**yolo26-analytics** is a pip-installable Python framework for real-time object tracking, zone analytics, and event alerting built on YOLO26. General-purpose — works for warehouse monitoring, retail foot traffic, traffic analysis, safety compliance, parking management. Not locked to one vertical.

**Target:** Open-source tool good enough to get cloned, starred, integrated into ecosystems like Roboflow, or lead to hiring. Not a portfolio demo.

**Why now:** YOLO26 dropped Jan 2026. NMS-free, no DFL, MuSGD optimizer, up to 43% faster CPU inference than YOLO11. Almost nobody has shipped production projects with it yet. The gap: no production-ready tracking/analytics framework exists for YOLO26.

---

## Package Identity

- **Name:** `yolo26-analytics`
- **Install:** `pip install yolo26-analytics`
- **Python:** 3.10+
- **License:** MIT (or Apache 2.0 — TBD at publish time)

### Quickstart — Python API

```python
from yolo26_analytics import Pipeline

pipeline = Pipeline.from_yaml("config.yaml")
pipeline.run()
```

### Quickstart — CLI

```bash
y26a run --source rtsp://cam1 --model yolo26n.pt --zones zones.yaml --dashboard
y26a run --source video.mp4 --heatmap output.png
y26a export --model yolo26n.pt --format onnx
```

---

## Package Structure

```
yolo26_analytics/
  core/           # Pipeline orchestration
  detection/      # Model adapters (YOLO26 default, pluggable)
  tracking/       # ByteTrack with persistent ID management
  zones/          # Zone definition, counting, entry/exit, dwell time
  analytics/      # Heatmaps, statistics, time-series aggregation
  alerts/         # Alert backends (webhook, MQTT, Telegram, console)
  dashboard/      # FastAPI + HTMX + vanilla JS live dashboard with replay
  export/         # Model export helpers (ONNX, TFLite, TensorRT)
  store/          # PostgreSQL event/track storage + query layer
  config/         # YAML config parsing
```

---

## Core Pipeline Architecture

### Data Flow

```
VideoSource → Detector → Tracker → ZoneAnalyzer → AlertManager
                            ↓                        ↓
                        TrackStore ←──────── EventStore
                            ↓
                      AnalyticsEngine (heatmaps, dwell, counts)
                            ↓
                        Dashboard (live + replay)
```

### Stage Protocols

Each stage implements a Python `Protocol`:

- **VideoSource** — yields frames. Built-in: webcam, RTSP, video file, image directory.
- **Detector** — takes a frame, returns detections (bbox, class, confidence). Default: YOLO26. Anyone can write an adapter for any model.
- **Tracker** — takes detections across frames, assigns persistent IDs. Default: ByteTrack.
- **ZoneAnalyzer** — takes tracked detections + zone config, emits events (entry, exit, dwell exceeded, count exceeded).
- **AlertManager** — takes events, dispatches to configured backends with routing/filtering.
- **TrackStore** — persists every track update to PostgreSQL. The backbone table everything else queries.
- **EventStore** — persists zone-generated events to PostgreSQL.
- **AnalyticsEngine** — queries TrackStore for heatmaps, counts, dwell stats, throughput.

### Pipeline Loop

```python
async for frame in source:
    detections = detector.predict(frame)
    tracks = tracker.update(detections)
    track_store.write(tracks, frame_meta)
    violations = zone_analyzer.check(tracks)
    if violations:
        event_store.log(violations)
        await alert_manager.dispatch(violations)
    analytics.accumulate(tracks)
```

**Why async:** Dashboard streams frames via SSE. Alert backends (webhook, MQTT, Telegram) are I/O-bound. Async lets the pipeline push frames to the dashboard and fire alerts without blocking inference.

### Configuration — YAML

```yaml
source:
  type: rtsp
  url: rtsp://192.168.1.100/stream

model:
  type: yolo26
  weights: yolo26n.pt
  confidence: 0.5
  sahi:
    enabled: true
    slice_size: 640
    overlap: 0.25

tracking:
  engine: bytetrack
  max_age: 30
  min_hits: 3

store:
  type: postgresql
  url: postgresql://localhost/yolo26_analytics
  retention:
    tracks: 7d
    events: 90d
    snapshots: 30d

zones:
  - name: "Loading Dock"
    polygon: [[100,100], [400,100], [400,400], [100,400]]
    track_classes: [person, forklift]
    analytics:
      - type: count
      - type: dwell
        alert_threshold: 300
      - type: entry_exit
        direction: inward
      - type: throughput
        interval: 3600
    cooldown: 30

alerts:
  - type: webhook
    url: https://hooks.slack.com/...
  - type: telegram
    bot_token: "123456:ABC-DEF"
    chat_id: "-1001234567890"
    filter:
      zones: ["Loading Dock"]
      event_types: ["dwell_exceeded", "count_exceeded"]
  - type: mqtt
    broker: mqtt://192.168.1.50
    topic: analytics/events
  - type: console

dashboard: true
```

---

## Zone System & Analytics

### Zone Capabilities

Each zone supports multiple analytics modes:

- **Counting** — how many objects of each class are currently in the zone. Real-time + historical.
- **Entry/Exit** — triggered when a track crosses the zone boundary. Directional — you know if they entered or left.
- **Dwell Time** — how long each track stays in the zone. Alerts when threshold exceeded.
- **Throughput** — objects per hour/day passing through the zone. Time-series data.

### Zone Config

```yaml
zones:
  - name: "Checkout Area"
    polygon: [[100,100], [500,100], [500,400], [100,400]]
    track_classes: [person]
    analytics:
      - type: count
      - type: dwell
        alert_threshold: 180
      - type: throughput
        interval: 3600

  - name: "Entrance"
    polygon: [[0,200], [100,200], [100,400], [0,400]]
    track_classes: [person]
    analytics:
      - type: entry_exit
        direction: inward
```

### Entry/Exit Detection Logic

Uses track centroid crossing the polygon boundary between consecutive frames. Direction determined by the cross product of the movement vector and the boundary edge normal. Standard approach used by retail foot traffic counters.

### Heatmap Generation

```bash
y26a heatmap --source video.mp4 --output heatmap.png --duration 3600
```

```python
from yolo26_analytics.analytics import generate_heatmap
heatmap = generate_heatmap(track_store, time_range=("14:00", "16:00"))
```

Accumulates track centroid positions into a 2D histogram, applies Gaussian blur, overlays on a reference frame. Output as PNG or as a live overlay on the dashboard.

### Event Model

```python
@dataclass
class Event:
    timestamp: datetime
    event_type: str          # "entry", "exit", "dwell_exceeded", "count_exceeded"
    zone_name: str
    track_id: int
    object_class: str
    metadata: dict           # flexible — dwell_seconds, count, direction, etc.
    confidence: float
    frame_snapshot: bytes
    bbox: tuple[int, int, int, int]
```

### Cooldown

Keyed on `(track_id, zone_name, event_type)`. Configurable per zone. Default 30 seconds.

---

## Track Store (PostgreSQL)

### Schema

```sql
CREATE TABLE tracks (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    track_id INTEGER NOT NULL,
    object_class VARCHAR(50) NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 INTEGER, bbox_y1 INTEGER,
    bbox_x2 INTEGER, bbox_y2 INTEGER,
    centroid_x INTEGER, centroid_y INTEGER,
    zone_name VARCHAR(100),
    source_id VARCHAR(100) NOT NULL
);

CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    zone_name VARCHAR(100) NOT NULL,
    track_id INTEGER NOT NULL,
    object_class VARCHAR(50) NOT NULL,
    metadata JSONB,
    confidence REAL,
    snapshot_path VARCHAR(255)
);

CREATE TABLE zone_stats (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    zone_name VARCHAR(100) NOT NULL,
    object_class VARCHAR(50) NOT NULL,
    current_count INTEGER,
    entries_total INTEGER,
    exits_total INTEGER,
    avg_dwell_seconds REAL
);

CREATE INDEX idx_tracks_time ON tracks (timestamp);
CREATE INDEX idx_tracks_zone ON tracks (zone_name, timestamp);
CREATE INDEX idx_tracks_track_id ON tracks (track_id, timestamp);
CREATE INDEX idx_events_time ON events (timestamp);
CREATE INDEX idx_events_zone ON events (zone_name, timestamp);
```

### Why PostgreSQL

- **Concurrent reads:** Dashboard, analytics engine, and API all query simultaneously. SQLite locks on write.
- **JSONB:** Event metadata is flexible per event type. Queryable (`metadata->>'dwell_seconds' > 300`).
- **Time-series queries:** Zone stats over time windows, hourly throughput, daily trends.
- **Option C ready:** When the LLM layer arrives, PostgreSQL + pgvector for embeddings is a natural extension.

### Snapshot Storage

Frame snapshots saved as JPEGs to disk (`data/snapshots/{date}/{event_id}.jpg`), path stored in `events.snapshot_path`. Not blobs in the database.

### Retention

Configurable in YAML. Cleanup runs as a periodic async task in the pipeline.

```yaml
store:
  retention:
    tracks: 7d
    events: 90d
    snapshots: 30d
```

### Dev Fallback

For quick local testing without PostgreSQL:

```yaml
store:
  type: sqlite
  path: ./data/yolo26_analytics.db
```

Same query interface, same schema (minus JSONB — uses TEXT with JSON). SQLite adapter is intentionally minimal — production uses PostgreSQL, SQLite is for `git clone && run`.

---

## Alert System

### Alert Backend Protocol

```python
class AlertBackend(Protocol):
    async def send(self, event: Event) -> None: ...
```

### Built-in Backends

- **Console** — prints to stdout. Dev/testing.
- **Webhook** — POST JSON to any URL (Slack, Discord, custom). Includes event metadata + snapshot URL.
- **MQTT** — publishes to a topic. Industrial IoT standard.
- **Telegram** — sends message + snapshot image to a chat/group via bot API.

### Alert Routing

Alerts can be filtered by event type and zone:

```yaml
alerts:
  - type: telegram
    bot_token: "..."
    chat_id: "..."
    filter:
      zones: ["Loading Dock", "Entrance"]
      event_types: ["dwell_exceeded", "count_exceeded"]

  - type: webhook
    url: https://hooks.slack.com/xxx
    # no filter — receives everything
```

### Custom Backends

```python
pipeline = Pipeline.from_yaml("config.yaml")
pipeline.add_alert(lambda e: print(f"Custom: {e.zone_name} {e.event_type}"))
```

---

## Dashboard

### Tech

FastAPI + Jinja2 templates + HTMX + vanilla JS canvas. No React, no Node, no build step. Chart.js via CDN for time-series charts.

### Views

**1. Live View (default)**
- MJPEG stream with bounding boxes, track IDs, zone overlays rendered server-side via OpenCV
- Stats bar: active tracks per zone, FPS, model latency
- Event log: live-updating via SSE, filterable by zone and event type

**2. Analytics View**
- Zone count time-series charts (Chart.js)
- Dwell time distribution per zone
- Throughput graph (objects/hour over selected time range)
- Heatmap overlay toggle on a reference frame

**3. Replay View**
- Time range selector
- Plays back track data overlaid on reference frames (not full video storage)
- Scrub bar to jump to specific events

**4. Zone Editor**
- Click-to-draw polygons on canvas overlay of live feed
- Assign name, tracked classes, analytics rules via form
- Live preview — zones appear on feed immediately
- Saves to YAML config

### Endpoints

```
GET  /                        — live view
GET  /analytics               — analytics view
GET  /replay                  — replay view
GET  /zones/edit              — zone editor

GET  /stream                  — MJPEG video feed
GET  /events                  — SSE stream of events
GET  /api/events              — query historical events (JSON)
GET  /api/tracks              — query track history (JSON)
GET  /api/zones               — list zones + current stats
POST /api/zones               — create/update zone
GET  /api/stats               — current pipeline stats
GET  /api/analytics/counts    — zone count time-series
GET  /api/analytics/dwell     — dwell time data
GET  /api/analytics/heatmap   — heatmap image (PNG)
```

### Design Principle

Dashboard is a consumer, not part of the pipeline. Pipeline runs headless. Dashboard is optional — enable with `--dashboard` flag or `dashboard: true` in YAML.

---

## Model Export & Edge Readiness

### Supported Formats

- **ONNX** — universal, runs anywhere via `onnxruntime`.
- **TFLite** — Raspberry Pi, Coral TPU, mobile.
- **TensorRT** — Jetson, NVIDIA GPUs.

### CLI

```bash
y26a export --model yolo26n.pt --format onnx
y26a export --model yolo26n.pt --format tflite --quantize int8
```

### Python API

```python
from yolo26_analytics.export import export_model
export_model("yolo26n.pt", format="tflite", quantize="int8")
```

### What Export Adds Over Raw Ultralytics

- Automatic benchmarking after export — logs inference speed, model size, mAP delta.
- Benchmark results saved to a JSON report for README/blog use.
- Validates the exported model actually loads and runs before declaring success.

### Auto-Select Runtime

```yaml
model:
  weights: yolo26n.tflite  # automatically uses TFLite runtime
```

Detector adapter checks file extension and picks the right inference backend (Ultralytics, onnxruntime, tflite-runtime). User doesn't configure this.

### SAHI Integration

For small object detection at distance (high-mounted cameras covering large areas):

```yaml
model:
  sahi:
    enabled: true
    slice_size: 640
    overlap: 0.25
```

Tiles the frame into overlapping slices, runs inference on each, merges detections. YOLO26's NMS-free architecture makes tile merging cleaner — no double-NMS problem.

---

## Testing & Quality

### Test Structure

- **Unit tests** — each stage in isolation. Mock frames through detector, verify tracker assigns IDs, verify zone counting/entry-exit/dwell logic, verify alert dispatch and filtering. `pytest`.
- **Integration tests** — full pipeline with test video + PostgreSQL via `testcontainers`. Verifies: detections → tracks → store → zone events → alerts.
- **Analytics tests** — verify heatmap generation, count aggregation, dwell calculation against known track data.
- **API tests** — `httpx` async client against FastAPI. Verify all endpoints return correct data.
- **Benchmark tests** — not pass/fail, but logged. FPS, latency, mAP. Results written to `benchmarks/` as JSON.

### CI — GitHub Actions

- On every PR: `ruff`, `mypy --strict`, unit tests, integration tests (Postgres via `testcontainers`).
- On merge to main: benchmark suite, results committed to `benchmarks/`.
- Badges: tests, coverage, latest benchmark.

### Code Quality

- Full `mypy --strict` compliance.
- Protocol classes typed.
- Docstrings on all public APIs.

### Examples

```
examples/
  quickstart.py           # minimal pipeline from webcam
  rtsp_warehouse.py       # RTSP + zones + Telegram alerts
  video_analytics.py      # process recorded video, generate heatmap + stats
  custom_detector.py      # plug in a non-YOLO model
  ppe_safety.py           # PPE safety monitoring (zones with required classes)
```

---

## Deployment Targets

Desktop-first (no edge hardware on hand currently). Clean export abstractions so that when Raspberry Pi or Jetson hardware is available, it's a config change — not a rewrite.

Docker Compose included for the full stack (app + PostgreSQL).

---

## Scope — V1

### Included

- YOLO26n/s detection, pluggable model adapter
- ByteTrack with persistent track IDs
- Zone analytics: counting, entry/exit, dwell time, throughput
- Heatmap generation (CLI + dashboard overlay)
- 4 alert backends: console, webhook, MQTT, Telegram
- Alert routing/filtering by zone and event type
- PostgreSQL track store with retention policies (SQLite fallback for dev)
- FastAPI dashboard: live view, analytics view, replay view, zone editor
- SAHI integration for small objects
- ONNX/TFLite/TensorRT export with auto-benchmarking
- CLI (`y26a`) + Python API
- Full test suite, CI, `mypy --strict`
- pip-installable, Docker Compose included
- `examples/` directory with 5 use-case scripts
- Clean README with architecture diagram, benchmarks table, quickstart

### Excluded from V1

- Pose estimation (YOLO26-pose)
- Segmentation (YOLO26-seg)
- Multi-camera orchestration (V1: one source per pipeline instance)
- User auth / multi-tenant
- Cloud storage / S3 uploads
- Mobile app
- Retraining / active learning loop

### V2: LLM Query Layer (Option C — ~1 extra week)

- Structured event embeddings via pgvector
- Natural language query interface: "How many forklifts entered zone B between 2pm and 4pm?"
- Automated scene summaries via Claude API
- Plain-English alert rules: "Notify me when the loading dock is empty for more than 10 minutes"
- Chat UI on the dashboard

### V2+: Future Roadmap (README mentions only)

- YOLO26-pose for action/posture detection
- YOLO26-seg for pixel-level analytics
- Active learning: flag low-confidence detections for relabeling
- Multi-camera orchestration with load balancing
- Grafana datasource plugin

---

## Content & Distribution

- **Blog post:** "Building a Real-Time Video Analytics Framework on YOLO26" — Medium + dev.to. Technical deep-dive, not a tutorial.
- **GitHub repo:** clean README with architecture diagram, benchmarks table, GIF/video of dashboard, 3-command quickstart.
- **Demo video:** 2-minute screen recording — dashboard live view with zones → analytics with heatmap → replay scrubbing → Telegram alert on phone.
- **Upwork positioning:** "I built an open-source video analytics framework on YOLO26 with X stars. Here's the demo. I can configure it for your use case in [timeframe]."

### Upwork Target Categories

- Object detection for manufacturing/warehouse/construction — $500-2000
- Deploy ML model on edge device — $300-800
- Real-time video analytics — $1000-3000
- Computer vision for safety/compliance — $500-1500

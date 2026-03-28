# yolo26-analytics

Real-time object tracking, zone analytics, and event alerting built on YOLO26.

[![CI](https://github.com/MohibShaikh/yolo26-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/MohibShaikh/yolo26-analytics/actions)
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

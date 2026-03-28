from pathlib import Path

import pytest
import yaml

from yolo26_analytics.config.schema import (
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

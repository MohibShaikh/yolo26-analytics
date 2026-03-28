"""Pipeline — the main orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yolo26_analytics.store.postgres import PostgresStore

from yolo26_analytics.alerts.manager import AlertManager
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
        heatmap: Any | None = None,
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
        self.frame_count: int = 0
        self.fps: float = 0.0
        self.last_latency_ms: float = 0.0

    def add_alert(self, fn: Callable[..., Any]) -> None:
        self._custom_alerts.append(fn)

    async def run_async(self) -> None:
        import time

        self._running = True
        fps_start = time.monotonic()
        fps_frames = 0
        async for frame, meta in self._source:
            if not self._running:
                break
            t0 = time.monotonic()
            detections = self._detector.predict(frame)
            tracks = self._tracker.update(detections)
            await self._store.write_tracks(tracks, meta)
            if self._heatmap is not None:
                for track in tracks:
                    cx, cy = track.centroid
                    self._heatmap.add_point(cx, cy)
            events: list[Event] = []
            if self._zone_analyzer is not None:
                events = self._zone_analyzer.check(tracks)
                if events:
                    await self._store.log_events(events)
            if events and self._alert_manager is not None:
                await self._alert_manager.dispatch(events)
            for event in events:
                for fn in self._custom_alerts:
                    fn(event)
            if self._on_frame is not None:
                self._on_frame(frame, meta, tracks, events)
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
        asyncio.run(self.run_async())

    def stop(self) -> None:
        self._running = False

    @classmethod
    def from_yaml(cls, path: str) -> Pipeline:
        config = load_config(path)
        return cls._from_config(config)

    @classmethod
    def _from_config(cls, config: AppConfig) -> Pipeline:
        from yolo26_analytics.alerts.console import ConsoleAlert
        from yolo26_analytics.alerts.mqtt import MQTTAlert
        from yolo26_analytics.alerts.telegram import TelegramAlert
        from yolo26_analytics.alerts.webhook import WebhookAlert
        from yolo26_analytics.detection.yolo26 import YOLO26Detector
        from yolo26_analytics.sources import create_source
        from yolo26_analytics.store.sqlite import SQLiteStore
        from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter

        source = create_source(config.source)
        detector = YOLO26Detector(weights=config.model.weights, confidence=config.model.confidence)
        tracker = ByteTrackAdapter(
            max_age=config.tracking.max_age, min_hits=config.tracking.min_hits
        )
        store: SQLiteStore | PostgresStore
        if config.store.type == "postgresql" and config.store.url:
            from yolo26_analytics.store.postgres import PostgresStore

            store = PostgresStore(url=config.store.url)
        else:
            store = SQLiteStore(path=config.store.path)
        zone_analyzer = ZoneAnalyzer(config.zones) if config.zones else None
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
                    backend = TelegramAlert(bot_token=ac.bot_token or "", chat_id=ac.chat_id or "")
                case _:
                    continue
            backends.append((backend, ac.filter))
        alert_manager = AlertManager(backends=backends) if backends else None

        heatmap = None
        try:
            from yolo26_analytics.analytics.heatmap import HeatmapAccumulator  # noqa: F401

            heatmap = HeatmapAccumulator()
        except (ImportError, AttributeError):
            pass

        return cls(
            source=source,
            detector=detector,
            tracker=tracker,
            store=store,
            zone_analyzer=zone_analyzer,
            alert_manager=alert_manager,
            heatmap=heatmap,
        )

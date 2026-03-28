"""yolo26-analytics: Real-time object tracking, zone analytics, and event alerting on YOLO26."""

from yolo26_analytics.models import Detection, Event, FrameMeta, Track
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
__version__ = "0.1.0"

"""Minimal quickstart — run detection on webcam."""

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

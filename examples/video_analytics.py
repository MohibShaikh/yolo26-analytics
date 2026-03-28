"""Process a recorded video and generate heatmap + stats."""

import asyncio

import cv2

from yolo26_analytics.analytics.heatmap import HeatmapAccumulator, generate_heatmap_image
from yolo26_analytics.core.pipeline import Pipeline
from yolo26_analytics.detection.yolo26 import YOLO26Detector
from yolo26_analytics.sources.video_file import VideoFileSource
from yolo26_analytics.store.sqlite import SQLiteStore
from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter


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
    ref = cv2.imread("reference_frame.jpg")
    if ref is not None:
        overlay = generate_heatmap_image(heatmap_acc, ref)
        cv2.imwrite("heatmap_output.png", overlay)
        print("Heatmap saved to heatmap_output.png")

    # Query track stats
    tracks = await store.query_tracks(limit=10)
    print(f"Total track records: {len(tracks)}")


if __name__ == "__main__":
    asyncio.run(main())

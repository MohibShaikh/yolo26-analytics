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
    from yolo26_analytics.sources.video_file import VideoFileSource
    from yolo26_analytics.store.sqlite import SQLiteStore
    from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter

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

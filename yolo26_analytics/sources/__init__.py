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

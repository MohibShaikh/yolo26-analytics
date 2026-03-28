from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from yolo26_analytics.config.schema import SourceConfig
from yolo26_analytics.sources import create_source
from yolo26_analytics.sources.image_dir import ImageDirSource
from yolo26_analytics.sources.video_file import VideoFileSource


def _create_test_video(path: str, num_frames: int = 10) -> None:
    """Write a small test video."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
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

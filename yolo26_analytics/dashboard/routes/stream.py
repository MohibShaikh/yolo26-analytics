"""MJPEG and SSE streaming endpoints."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

router = APIRouter()
_latest_frame: bytes | None = None
_event_queue: deque[dict[str, Any]] = deque(maxlen=100)


def update_frame(jpeg_bytes: bytes) -> None:
    global _latest_frame
    _latest_frame = jpeg_bytes


def push_event(event_data: dict[str, Any]) -> None:
    _event_queue.append(event_data)


async def _mjpeg_generator():
    while True:
        if _latest_frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _latest_frame + b"\r\n")
        await asyncio.sleep(0.033)


@router.get("/stream")
async def mjpeg_stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/events")
async def sse_events(request: Request) -> EventSourceResponse:
    async def event_generator():
        last_idx = 0
        while True:
            if await request.is_disconnected():
                break
            while last_idx < len(_event_queue):
                event = _event_queue[last_idx]
                yield {"data": json.dumps(event)}
                last_idx += 1
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())

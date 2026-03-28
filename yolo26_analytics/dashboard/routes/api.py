"""JSON API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/stats")
async def get_stats(request: Request) -> dict[str, Any]:
    pipeline = request.app.state.pipeline
    result: dict[str, Any] = {
        "fps": pipeline.fps if pipeline else 0,
        "frame_count": pipeline.frame_count if pipeline else 0,
        "latency_ms": pipeline.last_latency_ms if pipeline else 0,
    }
    return result


@router.get("/zones")
async def get_zones(request: Request) -> list[dict[str, Any]]:
    za = request.app.state.zone_analyzer
    if za is None:
        return []
    result: list[dict[str, Any]] = [
        {"name": z.name, "track_classes": z.track_classes, "cooldown": z.cooldown} for z in za.zones
    ]
    return result


@router.get("/events")
async def get_events(
    request: Request, zone_name: str | None = None, event_type: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    store = request.app.state.store
    if store is None:
        return []
    result: list[dict[str, Any]] = await store.query_events(
        zone_name=zone_name, event_type=event_type, limit=limit
    )
    return result


@router.get("/tracks")
async def get_tracks(
    request: Request,
    zone_name: str | None = None,
    source_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    store = request.app.state.store
    if store is None:
        return []
    result: list[dict[str, Any]] = await store.query_tracks(
        zone_name=zone_name, source_id=source_id, limit=limit
    )
    return result


@router.get("/analytics/counts")
async def get_counts(request: Request) -> dict[str, Any]:
    za = request.app.state.zone_analyzer
    if za is None:
        return {}
    result: dict[str, Any] = za.get_zone_counts()
    return result


@router.get("/analytics/dwell")
async def get_dwell(request: Request) -> dict[str, Any]:
    return {}


@router.get("/analytics/heatmap")
async def get_heatmap() -> dict[str, str]:
    return {"status": "not_implemented_yet"}

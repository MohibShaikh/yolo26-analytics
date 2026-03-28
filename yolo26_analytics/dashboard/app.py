"""FastAPI application factory for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(store: Any = None, pipeline: Any = None, zone_analyzer: Any = None) -> FastAPI:
    """Create the dashboard FastAPI app."""
    app = FastAPI(title="yolo26-analytics Dashboard")
    app.state.store = store
    app.state.pipeline = pipeline
    app.state.zone_analyzer = zone_analyzer
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    from yolo26_analytics.dashboard.routes.api import router as api_router
    from yolo26_analytics.dashboard.routes.stream import router as stream_router
    from yolo26_analytics.dashboard.routes.views import router as views_router

    app.include_router(views_router)
    app.include_router(stream_router)
    app.include_router(api_router, prefix="/api")
    return app

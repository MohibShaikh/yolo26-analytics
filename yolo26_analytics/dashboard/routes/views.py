"""HTML page routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def live_view(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    response: HTMLResponse = templates.TemplateResponse(request=request, name="live.html")
    return response

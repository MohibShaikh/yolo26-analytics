from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from yolo26_analytics.dashboard.app import create_app


@pytest.fixture
def app():
    return create_app(store=None, pipeline=None, zone_analyzer=None)


class TestDashboardAPI:
    @pytest.mark.asyncio
    async def test_index_returns_html(self, app) -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_api_stats(self, app) -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "fps" in data
        assert "frame_count" in data

    @pytest.mark.asyncio
    async def test_api_zones(self, app) -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/zones")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

from __future__ import annotations
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from yolo26_analytics.cli import main


class TestCLI:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "yolo26-analytics" in result.output.lower() or "y26a" in result.output.lower()

    def test_run_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output

    def test_export_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--format" in result.output

    def test_heatmap_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["heatmap", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output
        assert "--output" in result.output

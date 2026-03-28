"""Model export + benchmarking."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ultralytics import YOLO  # type: ignore[attr-defined]


def export_model(
    weights: str,
    format: str = "onnx",
    quantize: str | None = None,
    output_dir: str | None = None,
) -> dict[str, object]:
    """Export a YOLO model and run a benchmark."""
    model = YOLO(weights)
    kwargs: dict[str, object] = {"format": format}
    if quantize == "int8":
        kwargs["int8"] = True
    elif quantize == "fp16":
        kwargs["half"] = True
    t0 = time.monotonic()
    export_path = model.export(**kwargs)
    export_time = time.monotonic() - t0
    export_file = Path(str(export_path))
    file_size_mb = export_file.stat().st_size / (1024 * 1024) if export_file.exists() else 0
    report: dict[str, object] = {
        "source_weights": weights,
        "format": format,
        "quantize": quantize,
        "export_path": str(export_path),
        "export_time_seconds": round(export_time, 2),
        "file_size_mb": round(file_size_mb, 2),
    }
    report_path = export_file.with_suffix(".benchmark.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Export complete: {export_path}")
    print(f"  Format: {format}, Size: {file_size_mb:.1f} MB, Time: {export_time:.1f}s")
    print(f"  Benchmark saved: {report_path}")
    return report

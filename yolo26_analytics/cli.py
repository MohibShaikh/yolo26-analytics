"""CLI entry point for yolo26-analytics (y26a)."""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """yolo26-analytics — Real-time object tracking, zone analytics, and alerting."""
    pass


@main.command()
@click.option("--source", required=True, help="Video source (file path, RTSP URL, or device index)")
@click.option("--model", default="yolo26n.pt", help="Model weights path")
@click.option("--confidence", default=0.5, type=float, help="Confidence threshold")
@click.option("--config", default=None, help="YAML config file (overrides other options)")
@click.option("--zones", default=None, help="Zones YAML file")
@click.option("--dashboard", is_flag=True, help="Enable live dashboard")
def run(
    source: str,
    model: str,
    confidence: float,
    config: str | None,
    zones: str | None,
    dashboard: bool,
) -> None:
    """Run the detection + tracking + analytics pipeline."""
    if config:
        from yolo26_analytics.core.pipeline import Pipeline

        pipeline = Pipeline.from_yaml(config)
        pipeline.run()
    else:
        from yolo26_analytics.config.schema import AppConfig, ModelConfig, SourceConfig

        source_type = _infer_source_type(source)
        source_config = SourceConfig(type=source_type)
        if source_type == "video_file":
            source_config.path = source
        elif source_type == "rtsp":
            source_config.url = source
        elif source_type == "webcam":
            source_config.device = int(source)
        app_config = AppConfig(
            source=source_config,
            model=ModelConfig(weights=model, confidence=confidence),
            dashboard=dashboard,
        )
        from yolo26_analytics.core.pipeline import Pipeline

        pipeline = Pipeline._from_config(app_config)
        pipeline.run()


@main.command()
@click.option("--model", required=True, help="Model weights path")
@click.option("--format", "fmt", required=True, type=click.Choice(["onnx", "tflite", "engine"]))
@click.option("--quantize", default=None, help="Quantization type (e.g., int8)")
def export(model: str, fmt: str, quantize: str | None) -> None:
    """Export model to ONNX, TFLite, or TensorRT."""
    from yolo26_analytics.export.exporter import export_model

    export_model(model, format=fmt, quantize=quantize)


@main.command()
@click.option("--source", required=True, help="Video source")
@click.option("--output", required=True, help="Output heatmap image path")
@click.option("--model", default="yolo26n.pt", help="Model weights path")
@click.option("--duration", default=None, type=int, help="Max seconds to process")
def heatmap(source: str, output: str, model: str, duration: int | None) -> None:
    """Generate a heatmap from a video source."""
    import time

    import cv2

    from yolo26_analytics.analytics.heatmap import HeatmapAccumulator, generate_heatmap_image
    from yolo26_analytics.detection.yolo26 import YOLO26Detector
    from yolo26_analytics.tracking.bytetrack import ByteTrackAdapter

    detector = YOLO26Detector(weights=model)
    tracker = ByteTrackAdapter()
    cap = cv2.VideoCapture(source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    acc = HeatmapAccumulator(width=w, height=h)
    ref_frame = None
    start = time.monotonic()
    frame_count = 0
    click.echo(f"Processing {source}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if ref_frame is None:
            ref_frame = frame.copy()
        if duration and (time.monotonic() - start) > duration:
            break
        detections = detector.predict(frame)
        tracks = tracker.update(detections)
        for track in tracks:
            cx, cy = track.centroid
            acc.add_point(cx, cy)
        frame_count += 1
    cap.release()
    if ref_frame is not None:
        result = generate_heatmap_image(acc, ref_frame)
        cv2.imwrite(output, result)
        click.echo(f"Heatmap saved to {output} ({frame_count} frames processed)")
    else:
        click.echo("No frames processed.")


def _infer_source_type(source: str) -> str:
    if source.startswith("rtsp://"):
        return "rtsp"
    if source.isdigit():
        return "webcam"
    return "video_file"

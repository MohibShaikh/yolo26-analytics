from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestExportModel:
    @patch("yolo26_analytics.export.exporter.YOLO")
    def test_export_onnx(self, mock_yolo_cls: MagicMock, tmp_path) -> None:
        from yolo26_analytics.export.exporter import export_model

        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "model.onnx")
        mock_yolo_cls.return_value = mock_model
        result = export_model("yolo26n.pt", format="onnx")
        mock_model.export.assert_called_once_with(format="onnx")
        assert result["format"] == "onnx"
        assert "export_path" in result

    @patch("yolo26_analytics.export.exporter.YOLO")
    def test_export_tflite_with_quantize(self, mock_yolo_cls: MagicMock, tmp_path) -> None:
        from yolo26_analytics.export.exporter import export_model

        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "model.tflite")
        mock_yolo_cls.return_value = mock_model
        result = export_model("yolo26n.pt", format="tflite", quantize="int8")
        mock_model.export.assert_called_once_with(format="tflite", int8=True)
        assert result["format"] == "tflite"

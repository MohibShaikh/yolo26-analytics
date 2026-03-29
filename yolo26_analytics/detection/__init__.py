from yolo26_analytics.detection.yolo26 import YOLO26Detector

__all__ = ["YOLO26Detector"]

try:
    from yolo26_analytics.detection.rfdetr import RFDETRDetector

    __all__ += ["RFDETRDetector"]
except ImportError:
    pass

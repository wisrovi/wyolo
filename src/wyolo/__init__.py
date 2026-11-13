"""Wyolo - Professional YOLO Training Library with MLOps integration and TensorRT Inference."""

from .inference import (
    LITSERVE_AVAILABLE,
    TRT_AVAILABLE,
    TensorRTYOLO,
    YOLOInference,
    YOLOTensorRTAPI,
    ensure_tensorrt_engine,
    export_to_engine,
    serve_yolo,
)
from .trainer.trainer_wrapper import create_trainer, train

__version__ = "2.2.17"
__author__ = "William Steve Rodriguez Villamizar"
__email__ = "wisrovi.rodriguez@gmail.com"

__all__ = [
    "train",
    "create_trainer",
    "ensure_tensorrt_engine",
    "export_to_engine",
    "TensorRTYOLO",
    "YOLOInference",
    "YOLOTensorRTAPI",
    "serve_yolo",
    "TRT_AVAILABLE",
    "LITSERVE_AVAILABLE",
]

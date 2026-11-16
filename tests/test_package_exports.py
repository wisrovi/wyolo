"""Unit tests for wyolo package exports and versioning."""

import pytest
import wyolo
from wyolo import __version__, create_trainer, train


def test_wyolo_exports() -> None:
    """Validates that wyolo exposes expected public API."""
    assert callable(train)
    assert callable(create_trainer)
    assert __version__ == "2.3.1"
    assert "train" in wyolo.__all__
    assert "create_trainer" in wyolo.__all__
    assert "ensure_tensorrt_engine" in wyolo.__all__
    assert "export_to_engine" in wyolo.__all__
    assert "YOLOInference" in wyolo.__all__
    assert "TensorRTYOLO" in wyolo.__all__
    assert "serve_yolo" in wyolo.__all__
    assert "WYoloInferenceStep" in wyolo.__all__

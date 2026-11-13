"""Unit tests for SafeStep decorator and exception containment in wyolo."""

from typing import Any, Dict
import pytest
from wyolo.trainer.safe_step import SafeStep


def test_safe_step_success_passthrough() -> None:
    """Validates that SafeStep passes successful function outputs cleanly."""

    def dummy_step(data: Dict[str, Any]) -> Dict[str, Any]:
        data["executed"] = True
        return data

    wrapped = SafeStep(dummy_step)
    result = wrapped({"initial": 1})
    assert result.get("initial") == 1
    assert result.get("executed") is True


def test_safe_step_exception_handling() -> None:
    """Validates that SafeStep intercepts exceptions without crashing the pipeline."""

    def failing_step(data: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("Simulated transient step failure")

    wrapped = SafeStep(failing_step)
    context = {"status": "ok", "trial": 1}
    result = wrapped(context)
    assert result == context
    assert result["status"] == "ok"


def test_safe_step_metadata_preservation() -> None:
    """Validates that SafeStep preserves wpipe metadata attributes."""

    def mock_wpipe_step(data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    mock_wpipe_step.NAME = "mock_wpipe_step"
    mock_wpipe_step.VERSION = "v1.0"
    mock_wpipe_step._wpipe_metadata = {"tag": "test"}

    wrapped = SafeStep(mock_wpipe_step)
    assert getattr(wrapped, "NAME", None) == "mock_wpipe_step"
    assert getattr(wrapped, "VERSION", None) == "v1.0"
    assert getattr(wrapped, "_wpipe_metadata", None) == {"tag": "test"}

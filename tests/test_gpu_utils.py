"""Unit tests for GPU utilities and diagnostic report generation in wyolo."""

import json
from typing import Any, Dict
import pytest
from wyolo.trainer.gpu_utils import obtener_info_gpu_json


def test_obtener_info_gpu_json() -> None:
    """Validates that GPU query function returns list of dictionaries containing GPU info or error."""
    info_raw = obtener_info_gpu_json()
    assert isinstance(info_raw, list)
    assert len(info_raw) > 0
    assert isinstance(info_raw[0], dict)

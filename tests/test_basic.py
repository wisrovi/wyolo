import pytest
from wyolo.core.utils import get_datetime

def test_get_datetime():
    dt = get_datetime()
    assert len(dt) == 15  # YYYYMMDD_HHMMSS
    assert "_" in dt

def test_placeholder():
    # Placeholder for future tests
    assert True

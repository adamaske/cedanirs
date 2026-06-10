import logging
import pytest
from pathlib import Path

# Ensure logs/ directory exists before pytest tries to write the log file
Path("logs").mkdir(exist_ok=True)
@pytest.fixture
def log(request):
    """Per-test logger named after the test function."""
    return logging.getLogger(request.node.name)

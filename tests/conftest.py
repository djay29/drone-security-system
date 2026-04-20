"""
conftest.py
-----------
Shared pytest fixtures for the drone security system test suite.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.memory.chroma_store import ChromaStore
from src.memory.sqlite_store import SQLiteStore
from src.perception.frame_preprocessor import FramePreprocessor
from src.perception.video_ingestor import FramePacket
from src.perception.yolo_detector import DetectedObject


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_db():
    """In-memory SQLiteStore — isolated per test."""
    return SQLiteStore(":memory:")


@pytest.fixture
def chroma_store():
    """Ephemeral (RAM-only) ChromaStore — isolated per test."""
    return ChromaStore(persist_dir=None)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def ts_day():
    """14:00 UTC — daytime, within allowed hours."""
    return datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def ts_night():
    """23:00 UTC — after hours (between 22:00 and 06:00)."""
    return datetime(2026, 4, 20, 23, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def ts_early_morning():
    """03:00 UTC — early morning, still after hours."""
    return datetime(2026, 4, 20, 3, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Image / frame helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_frame():
    """640×480 black BGR uint8 image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_packet(blank_frame):
    """FramePacket with a real image."""
    return FramePacket(
        frame_id=str(uuid.uuid4()),
        ts=datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc),
        video_id="test_video",
        frame_index=0,
        image=blank_frame,
    )


@pytest.fixture
def preprocessor():
    """FramePreprocessor with default settings."""
    return FramePreprocessor(yolo_size=640, vlm_every=5)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def make_detection(
    class_name: str = "person",
    confidence: float = 0.9,
    track_id: int | None = 1,
    bbox: tuple = (100, 100, 200, 200),
) -> DetectedObject:
    return DetectedObject(
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        track_id=track_id,
    )


# ---------------------------------------------------------------------------
# Rule YAML helpers
# ---------------------------------------------------------------------------

def write_rules_yaml(tmp_path: Path, rules_yaml: str) -> Path:
    """Write a rules.yaml string to a temp directory and return its path."""
    p = tmp_path / "rules.yaml"
    p.write_text(rules_yaml)
    return p

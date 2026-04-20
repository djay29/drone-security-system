"""
test_hybrid_retriever.py
------------------------
Unit tests for HybridRetriever — temporal_search, events_summary, and the
get_frames_by_ids batch helper.  Uses in-memory SQLite and ephemeral Chroma.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

import pytest

from src.memory.hybrid_retriever import HybridRetriever
from src.memory.sqlite_store import SQLiteStore
from src.memory.chroma_store import ChromaStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    return SQLiteStore(":memory:")


@pytest.fixture
def chroma():
    return ChromaStore(persist_dir=None)


@pytest.fixture
def retriever(db, chroma):
    return HybridRetriever(db, chroma)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(hour: int) -> datetime:
    return datetime(2026, 4, 20, hour, 0, 0, tzinfo=timezone.utc)


def _seed_frame(db: SQLiteStore, frame_id: str, ts: datetime, zone: str = "gate",
                class_names: list[str] | None = None) -> None:
    db.store_frame(
        frame_id=frame_id, ts=ts, video_id="vid1",
        frame_index=0, frame_path=None, caption="test", zone=zone,
    )
    if class_names:
        db.store_objects(frame_id, [
            {"class_name": c, "confidence": 0.9, "bbox": [0, 0, 10, 10], "track_id": None}
            for c in class_names
        ])


def _seed_event(db: SQLiteStore, event_type: str = "loitering",
                severity: str = "medium", zone: str = "gate",
                ts: datetime = None) -> str:
    eid = str(uuid.uuid4())
    db.store_event(
        event_id=eid, start_ts=ts or _ts(12),
        event_type=event_type, severity=severity,
        description=f"{event_type}", frame_ids=[], zone=zone,
    )
    return eid


# ---------------------------------------------------------------------------
# temporal_search
# ---------------------------------------------------------------------------

class TestTemporalSearch:

    def test_returns_frames_in_time_window(self, retriever, db):
        _seed_frame(db, "f1", _ts(10))
        _seed_frame(db, "f2", _ts(14))
        _seed_frame(db, "f3", _ts(20))

        results = retriever.temporal_search(start=_ts(12), end=_ts(18))
        ids = [r.frame_id for r in results]
        assert "f2" in ids
        assert "f1" not in ids
        assert "f3" not in ids

    def test_returns_empty_for_no_matches(self, retriever, db):
        _seed_frame(db, "f1", _ts(10))
        results = retriever.temporal_search(start=_ts(20), end=_ts(23))
        assert results == []

    def test_class_filter_returns_only_matching_frames(self, retriever, db):
        _seed_frame(db, "f1", _ts(10), class_names=["person"])
        _seed_frame(db, "f2", _ts(11), class_names=["car"])
        _seed_frame(db, "f3", _ts(12), class_names=["person", "car"])

        results = retriever.temporal_search(
            start=_ts(9), end=_ts(23), class_filter="person"
        )
        ids = [r.frame_id for r in results]
        assert "f1" in ids
        assert "f3" in ids
        assert "f2" not in ids

    def test_class_filter_with_no_matches_returns_empty(self, retriever, db):
        _seed_frame(db, "f1", _ts(10), class_names=["car"])
        results = retriever.temporal_search(
            start=_ts(9), end=_ts(23), class_filter="person"
        )
        assert results == []

    def test_zone_filter_narrows_results(self, retriever, db):
        _seed_frame(db, "f1", _ts(10), zone="gate")
        _seed_frame(db, "f2", _ts(11), zone="parking")

        results = retriever.temporal_search(start=_ts(9), end=_ts(23), zone="gate")
        ids = [r.frame_id for r in results]
        assert "f1" in ids
        assert "f2" not in ids

    def test_results_are_frame_result_objects(self, retriever, db):
        from src.memory.hybrid_retriever import FrameResult
        _seed_frame(db, "f1", _ts(10))
        results = retriever.temporal_search(start=_ts(9), end=_ts(23))
        assert all(isinstance(r, FrameResult) for r in results)

    def test_class_names_populated_in_results(self, retriever, db):
        _seed_frame(db, "f1", _ts(10), class_names=["person", "car"])
        results = retriever.temporal_search(start=_ts(9), end=_ts(23))
        r = next(r for r in results if r.frame_id == "f1")
        assert "person" in r.class_names
        assert "car" in r.class_names

    def test_top_k_limits_results(self, retriever, db):
        for i in range(10):
            _seed_frame(db, f"f{i}", _ts(10) + timedelta(minutes=i))

        results = retriever.temporal_search(start=_ts(9), end=_ts(23), top_k=5)
        assert len(results) <= 5

    def test_no_start_end_returns_all_frames(self, retriever, db):
        for i in range(3):
            _seed_frame(db, f"f{i}", _ts(10 + i))
        # Neither start nor end passed — all frames in DB should be returned
        results = retriever.temporal_search()
        assert len(results) == 3


# ---------------------------------------------------------------------------
# events_summary
# ---------------------------------------------------------------------------

class TestEventsSummary:

    def test_returns_all_events_unfiltered(self, retriever, db):
        _seed_event(db, "loitering")
        _seed_event(db, "after_hours_person")
        events = retriever.events_summary()
        assert len(events) >= 2

    def test_filters_by_event_type(self, retriever, db):
        _seed_event(db, "loitering")
        _seed_event(db, "after_hours_person")
        events = retriever.events_summary(event_type="loitering")
        assert all(e["type"] == "loitering" for e in events)

    def test_filters_by_severity(self, retriever, db):
        _seed_event(db, severity="high")
        _seed_event(db, severity="low")
        events = retriever.events_summary(severity="high")
        assert all(e["severity"] == "high" for e in events)

    def test_filters_by_time_window(self, retriever, db):
        _seed_event(db, ts=_ts(10))
        _seed_event(db, ts=_ts(20))
        events = retriever.events_summary(start=_ts(15), end=_ts(23))
        assert len(events) == 1

    def test_filters_by_zone(self, retriever, db):
        _seed_event(db, zone="gate")
        _seed_event(db, zone="parking")
        events = retriever.events_summary(zone="gate")
        assert all(e["zone"] == "gate" for e in events)


# ---------------------------------------------------------------------------
# frames_in_window (structured query)
# ---------------------------------------------------------------------------

class TestFramesInWindow:

    def test_returns_frames_with_detections(self, retriever, db):
        fid = str(uuid.uuid4())
        _seed_frame(db, fid, _ts(12), class_names=["person"])
        frames = retriever.frames_in_window(_ts(11), _ts(13))
        assert any(f["id"] == fid for f in frames)
        match = next(f for f in frames if f["id"] == fid)
        assert len(match["detections"]) == 1

    def test_returns_empty_for_empty_window(self, retriever, db):
        _seed_frame(db, "f1", _ts(10))
        frames = retriever.frames_in_window(_ts(20), _ts(23))
        assert frames == []

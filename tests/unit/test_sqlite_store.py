"""
test_sqlite_store.py
--------------------
Unit tests for SQLiteStore — CRUD operations, query helpers, filters.

All tests use an in-memory `:memory:` database so they are fully isolated
and leave no files on disk.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from src.memory.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(hour: int = 12) -> datetime:
    return datetime(2026, 4, 20, hour, 0, 0, tzinfo=timezone.utc)


def _store_sample_frame(db: SQLiteStore, frame_id: str = None, zone: str = "gate",
                         ts: datetime = None, video_id: str = "vid1") -> str:
    fid = frame_id or str(uuid.uuid4())
    db.store_frame(
        frame_id=fid,
        ts=ts or _ts(),
        video_id=video_id,
        frame_index=0,
        frame_path=None,
        caption="A person near the gate",
        zone=zone,
    )
    return fid


def _store_sample_event(db: SQLiteStore, event_type: str = "loitering",
                         severity: str = "medium", zone: str = "gate",
                         ts: datetime = None, frame_id: str = None) -> str:
    eid = str(uuid.uuid4())
    fid = frame_id or str(uuid.uuid4())
    db.store_event(
        event_id=eid,
        start_ts=ts or _ts(),
        event_type=event_type,
        severity=severity,
        description=f"{event_type} event",
        frame_ids=[fid],
        zone=zone,
    )
    return eid


# ---------------------------------------------------------------------------
# Tests: frames
# ---------------------------------------------------------------------------

class TestFrames:

    def test_store_and_retrieve_frame(self, sqlite_db):
        fid = _store_sample_frame(sqlite_db)
        rows = sqlite_db.get_frames_by_video("vid1")
        assert len(rows) == 1
        assert rows[0]["id"] == fid
        assert rows[0]["caption"] == "A person near the gate"
        assert rows[0]["zone"] == "gate"

    def test_get_frames_in_range_returns_correct_window(self, sqlite_db):
        _store_sample_frame(sqlite_db, frame_id="f1", ts=_ts(10))
        _store_sample_frame(sqlite_db, frame_id="f2", ts=_ts(14))
        _store_sample_frame(sqlite_db, frame_id="f3", ts=_ts(20))

        rows = sqlite_db.get_frames_in_range(_ts(13), _ts(21))
        ids = [r["id"] for r in rows]
        assert "f2" in ids
        assert "f3" in ids
        assert "f1" not in ids

    def test_get_frames_in_range_zone_filter(self, sqlite_db):
        _store_sample_frame(sqlite_db, frame_id="f1", zone="gate")
        _store_sample_frame(sqlite_db, frame_id="f2", zone="parking")

        rows = sqlite_db.get_frames_in_range(_ts(0), _ts(23), zone="gate")
        ids = [r["id"] for r in rows]
        assert "f1" in ids
        assert "f2" not in ids

    def test_get_frames_by_ids_returns_correct_rows(self, sqlite_db):
        f1 = _store_sample_frame(sqlite_db, frame_id="aaa")
        f2 = _store_sample_frame(sqlite_db, frame_id="bbb")
        _store_sample_frame(sqlite_db, frame_id="ccc")

        rows = sqlite_db.get_frames_by_ids(["aaa", "bbb"])
        ids = [r["id"] for r in rows]
        assert "aaa" in ids
        assert "bbb" in ids
        assert "ccc" not in ids

    def test_get_frames_by_ids_preserves_order(self, sqlite_db):
        _store_sample_frame(sqlite_db, frame_id="aaa")
        _store_sample_frame(sqlite_db, frame_id="bbb")
        _store_sample_frame(sqlite_db, frame_id="ccc")

        rows = sqlite_db.get_frames_by_ids(["ccc", "aaa", "bbb"])
        assert [r["id"] for r in rows] == ["ccc", "aaa", "bbb"]

    def test_get_frames_by_ids_empty_list(self, sqlite_db):
        assert sqlite_db.get_frames_by_ids([]) == []

    def test_get_frames_by_ids_ignores_missing_ids(self, sqlite_db):
        _store_sample_frame(sqlite_db, frame_id="exists")
        rows = sqlite_db.get_frames_by_ids(["exists", "nonexistent"])
        assert len(rows) == 1
        assert rows[0]["id"] == "exists"


# ---------------------------------------------------------------------------
# Tests: objects
# ---------------------------------------------------------------------------

class TestObjects:

    def test_store_objects_and_retrieve(self, sqlite_db):
        fid = _store_sample_frame(sqlite_db)
        sqlite_db.store_objects(fid, [
            {"class_name": "person", "confidence": 0.91, "bbox": [10, 20, 50, 80], "track_id": 1},
            {"class_name": "car",    "confidence": 0.78, "bbox": [100, 100, 300, 200], "track_id": 2},
        ])
        objs = sqlite_db.get_objects_for_frame(fid)
        assert len(objs) == 2
        classes = {o["class"] for o in objs}
        assert classes == {"person", "car"}

    def test_get_objects_by_class_returns_matching_class(self, sqlite_db):
        fid = _store_sample_frame(sqlite_db)
        sqlite_db.store_objects(fid, [
            {"class_name": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10], "track_id": 1},
            {"class_name": "truck",  "confidence": 0.8, "bbox": [0, 0, 10, 10], "track_id": 2},
        ])
        objs = sqlite_db.get_objects_by_class("person")
        assert all(o["class"] == "person" for o in objs)
        assert len(objs) == 1

    def test_get_objects_by_class_time_filter(self, sqlite_db):
        fid1 = _store_sample_frame(sqlite_db, frame_id="f1", ts=_ts(10))
        fid2 = _store_sample_frame(sqlite_db, frame_id="f2", ts=_ts(20))
        sqlite_db.store_objects(fid1, [{"class_name": "person", "confidence": 0.9, "bbox": [0,0,1,1], "track_id": 1}])
        sqlite_db.store_objects(fid2, [{"class_name": "person", "confidence": 0.9, "bbox": [0,0,1,1], "track_id": 2}])

        objs = sqlite_db.get_objects_by_class("person", start=_ts(15), end=_ts(23))
        assert len(objs) == 1
        assert objs[0]["frame_id"] == "f2"

    def test_get_vehicle_counts_today(self, sqlite_db):
        fid1 = _store_sample_frame(sqlite_db, frame_id="f1", ts=_ts(10))
        fid2 = _store_sample_frame(sqlite_db, frame_id="f2", ts=_ts(11))
        sqlite_db.store_objects(fid1, [{"class_name": "car", "confidence": 0.9, "bbox": [0,0,1,1], "track_id": 1}])
        sqlite_db.store_objects(fid2, [{"class_name": "car", "confidence": 0.9, "bbox": [0,0,1,1], "track_id": 2}])

        today_start = _ts(0)
        rows = sqlite_db.get_vehicle_counts_today(start=today_start, end=_ts(23))
        car_row = next((r for r in rows if r["class"] == "car"), None)
        assert car_row is not None
        assert car_row["distinct_tracks"] == 2


# ---------------------------------------------------------------------------
# Tests: events
# ---------------------------------------------------------------------------

class TestEvents:

    def test_store_and_retrieve_event(self, sqlite_db):
        eid = _store_sample_event(sqlite_db)
        events = sqlite_db.get_events()
        assert any(e["id"] == eid for e in events)

    def test_get_events_by_type_filter(self, sqlite_db):
        _store_sample_event(sqlite_db, event_type="loitering")
        _store_sample_event(sqlite_db, event_type="after_hours_person")

        events = sqlite_db.get_events(event_type="loitering")
        assert all(e["type"] == "loitering" for e in events)

    def test_get_events_by_severity_filter(self, sqlite_db):
        _store_sample_event(sqlite_db, severity="high")
        _store_sample_event(sqlite_db, severity="low")

        events = sqlite_db.get_events(severity="high")
        assert all(e["severity"] == "high" for e in events)

    def test_get_events_by_time_window(self, sqlite_db):
        _store_sample_event(sqlite_db, ts=_ts(10))
        _store_sample_event(sqlite_db, ts=_ts(20))

        events = sqlite_db.get_events(start=_ts(15), end=_ts(23))
        assert len(events) == 1
        assert events[0]["start_ts"].startswith("2026-04-20T20")

    def test_get_events_by_zone_filter(self, sqlite_db):
        _store_sample_event(sqlite_db, zone="gate")
        _store_sample_event(sqlite_db, zone="parking")

        events = sqlite_db.get_events(zone="gate")
        assert all(e["zone"] == "gate" for e in events)

    def test_frame_ids_stored_as_json(self, sqlite_db):
        eid = str(uuid.uuid4())
        fid = str(uuid.uuid4())
        sqlite_db.store_event(
            event_id=eid, start_ts=_ts(),
            event_type="test", severity="low",
            description="test", frame_ids=[fid], zone="gate",
        )
        events = sqlite_db.get_events()
        match = next(e for e in events if e["id"] == eid)
        stored_ids = json.loads(match["frame_ids"])
        assert fid in stored_ids


# ---------------------------------------------------------------------------
# Tests: alerts
# ---------------------------------------------------------------------------

class TestAlerts:

    def test_store_and_retrieve_alert(self, sqlite_db):
        eid = _store_sample_event(sqlite_db)
        aid = str(uuid.uuid4())
        sqlite_db.store_alert(
            alert_id=aid, event_id=eid, ts=_ts(),
            channel="dashboard", message="Test alert",
        )
        alerts = sqlite_db.get_alerts_for_event(eid)
        assert len(alerts) == 1
        assert alerts[0]["id"] == aid
        assert alerts[0]["message"] == "Test alert"
        assert alerts[0]["acked"] == 0

    def test_ack_alert_updates_state(self, sqlite_db):
        eid = _store_sample_event(sqlite_db)
        aid = str(uuid.uuid4())
        sqlite_db.store_alert(alert_id=aid, event_id=eid, ts=_ts(),
                               channel="dashboard", message="Alert")

        sqlite_db.ack_alert(aid)
        alerts = sqlite_db.get_alerts_for_event(eid)
        assert alerts[0]["acked"] == 1

    def test_get_unacked_alerts_excludes_acked(self, sqlite_db):
        eid = _store_sample_event(sqlite_db)
        aid_unacked = str(uuid.uuid4())
        aid_acked   = str(uuid.uuid4())

        sqlite_db.store_alert(alert_id=aid_unacked, event_id=eid, ts=_ts(),
                               channel="dashboard", message="Unacked")
        sqlite_db.store_alert(alert_id=aid_acked, event_id=eid, ts=_ts(),
                               channel="dashboard", message="Acked")
        sqlite_db.ack_alert(aid_acked)

        unacked = sqlite_db.get_unacked_alerts()
        ids = [a["id"] for a in unacked]
        assert aid_unacked in ids
        assert aid_acked not in ids


# ---------------------------------------------------------------------------
# Tests: stats
# ---------------------------------------------------------------------------

class TestStats:

    def test_get_stats_counts_correctly(self, sqlite_db):
        fid = _store_sample_frame(sqlite_db)
        sqlite_db.store_objects(fid, [
            {"class_name": "person", "confidence": 0.9, "bbox": [0,0,1,1], "track_id": 1},
        ])
        eid = _store_sample_event(sqlite_db, frame_id=fid)
        sqlite_db.store_alert(alert_id=str(uuid.uuid4()), event_id=eid,
                               ts=_ts(), channel="dashboard", message="Alert")

        stats = sqlite_db.get_stats()
        assert stats["frames"] == 1
        assert stats["objects"] == 1
        assert stats["events"] == 1
        assert stats["alerts"] == 1
        assert stats["unacked_alerts"] == 1

    def test_get_stats_empty_db(self, sqlite_db):
        stats = sqlite_db.get_stats()
        assert stats["frames"] == 0
        assert stats["objects"] == 0
        assert stats["events"] == 0
        assert stats["alerts"] == 0

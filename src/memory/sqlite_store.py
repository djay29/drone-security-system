"""
sqlite_store.py
---------------
Persistent structured store for all drone security data.

Tables (matches context sheet section 8 exactly):
  frames   — one row per emitted frame (caption, telemetry, path)
  objects  — detected objects per frame (YOLO output)
  events   — derived security events (loitering, vehicle_entry, etc.)
  alerts   — fired alerts with ack state

Ingestion helpers:
  store_frame_analysis()  — writes a FrameAnalysis in one call
  store_event()           — writes an Event
  store_alert()           — writes an Alert

Query helpers:
  get_frames_in_range()   — time-window filter
  get_objects_by_class()  — "show all truck events"
  get_events_by_type()    — filter by event type + severity
  get_unacked_alerts()    — dashboard alert feed
  search_frames_text()    — LIKE search on captions (fallback when CLIP unavailable)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id              TEXT PRIMARY KEY,
    ts              TIMESTAMP NOT NULL,
    video_id        TEXT NOT NULL,
    frame_index     INTEGER,
    frame_path      TEXT,
    caption         TEXT,
    telemetry_json  TEXT,
    zone            TEXT
);

CREATE TABLE IF NOT EXISTS objects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    TEXT NOT NULL REFERENCES frames(id),
    class       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    bbox        TEXT NOT NULL,
    track_id    INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    start_ts    TIMESTAMP NOT NULL,
    end_ts      TIMESTAMP,
    type        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    description TEXT NOT NULL,
    frame_ids   TEXT NOT NULL,
    zone        TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL REFERENCES events(id),
    ts          TIMESTAMP NOT NULL,
    channel     TEXT NOT NULL,
    message     TEXT NOT NULL,
    acked       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_frames_ts       ON frames(ts);
CREATE INDEX IF NOT EXISTS idx_frames_video    ON frames(video_id);
CREATE INDEX IF NOT EXISTS idx_frames_zone     ON frames(zone);
CREATE INDEX IF NOT EXISTS idx_objects_class   ON objects(class);
CREATE INDEX IF NOT EXISTS idx_objects_frame   ON objects(frame_id);
CREATE INDEX IF NOT EXISTS idx_objects_track   ON objects(track_id);
CREATE INDEX IF NOT EXISTS idx_events_type     ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
CREATE INDEX IF NOT EXISTS idx_events_start    ON events(start_ts);
CREATE INDEX IF NOT EXISTS idx_alerts_event    ON alerts(event_id);
CREATE INDEX IF NOT EXISTS idx_alerts_acked    ON alerts(acked);
"""


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------

class SQLiteStore:
    """
    Parameters
    ----------
    db_path : Path to the SQLite file. Use ':memory:' for tests.
    """

    def __init__(self, db_path: str | Path = "data/sql_db/drone_security.db") -> None:
        self.db_path = str(db_path)
        # Keep a single persistent connection for :memory: databases — each
        # sqlite3.connect(":memory:") would otherwise create a fresh empty DB.
        self._mem_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys=ON")
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True) if self.db_path != ":memory:" else None
        with self._get_conn() as conn:
            conn.executescript(SCHEMA)
        logger.info("[SQLiteStore] Initialised at %s", self.db_path)

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Thread-safe connection.

        For :memory: databases a single shared connection is reused so that
        the schema (and all data) survive across multiple _get_conn() calls.
        For file-backed databases a new connection is created per call.
        """
        if self._mem_conn is not None:
            # In-memory: yield the shared connection; manage transactions manually.
            try:
                yield self._mem_conn
                self._mem_conn.commit()
            except Exception:
                self._mem_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read perf
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Write: frames + objects (from pipeline output)
    # ------------------------------------------------------------------

    def store_frame(
        self,
        frame_id: str,
        ts: datetime,
        video_id: str,
        frame_index: int,
        frame_path: Optional[str] = None,
        caption: Optional[str] = None,
        telemetry: Optional[dict] = None,
        zone: Optional[str] = None,
    ) -> None:
        """Insert or replace a frame record."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO frames
                    (id, ts, video_id, frame_index, frame_path, caption, telemetry_json, zone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    frame_id,
                    ts.isoformat(),
                    video_id,
                    frame_index,
                    frame_path,
                    caption,
                    json.dumps(telemetry) if telemetry else None,
                    zone,
                ),
            )

    def store_objects(self, frame_id: str, detections: list[dict]) -> None:
        """
        Insert detected objects for a frame.

        Parameters
        ----------
        frame_id   : Parent frame id
        detections : List of dicts with keys: class_name, confidence, bbox, track_id
        """
        rows = [
            (
                frame_id,
                d["class_name"],
                d["confidence"],
                json.dumps(d["bbox"]),
                d.get("track_id"),
            )
            for d in detections
        ]
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT INTO objects (frame_id, class, confidence, bbox, track_id) VALUES (?,?,?,?,?)",
                rows,
            )

    def store_frame_from_caption_record(self, record: dict) -> None:
        """
        Convenience: ingest one record from captions.jsonl directly.

        Expected keys: frame_id, frame_index, ts, zone, caption, detections
        """
        self.store_frame(
            frame_id    = record["frame_id"],
            ts          = datetime.fromisoformat(record["ts"]),
            video_id    = record.get("video_id", "unknown"),
            frame_index = record["frame_index"],
            caption     = record.get("caption"),
            zone        = record.get("zone"),
        )
        if record.get("detections"):
            self.store_objects(record["frame_id"], record["detections"])

    # ------------------------------------------------------------------
    # Write: events + alerts
    # ------------------------------------------------------------------

    def store_event(
        self,
        event_id: str,
        start_ts: datetime,
        event_type: str,
        severity: str,
        description: str,
        frame_ids: list[str],
        end_ts: Optional[datetime] = None,
        zone: Optional[str] = None,
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO events
                    (id, start_ts, end_ts, type, severity, description, frame_ids, zone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    start_ts.isoformat(),
                    end_ts.isoformat() if end_ts else None,
                    event_type,
                    severity,
                    description,
                    json.dumps(frame_ids),
                    zone,
                ),
            )

    def store_alert(
        self,
        alert_id: str,
        event_id: str,
        ts: datetime,
        channel: str,
        message: str,
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alerts (id, event_id, ts, channel, message, acked)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (alert_id, event_id, ts.isoformat(), channel, message),
            )

    def ack_alert(self, alert_id: str) -> None:
        with self._get_conn() as conn:
            conn.execute("UPDATE alerts SET acked=1 WHERE id=?", (alert_id,))

    # ------------------------------------------------------------------
    # Query: frames
    # ------------------------------------------------------------------

    def get_frames_in_range(
        self,
        start: datetime,
        end: datetime,
        zone: Optional[str] = None,
    ) -> list[dict]:
        """Return frames within a time window, optionally filtered by zone."""
        sql  = "SELECT * FROM frames WHERE ts BETWEEN ? AND ?"
        args: list[Any] = [start.isoformat(), end.isoformat()]
        if zone:
            sql  += " AND zone = ?"
            args.append(zone)
        sql += " ORDER BY ts ASC"
        return self._fetchall(sql, args)

    def get_frames_by_video(self, video_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM frames WHERE video_id=? ORDER BY frame_index ASC",
            [video_id],
        )

    def get_frames_by_ids(self, frame_ids: list[str]) -> list[dict]:
        """Return frame rows for a list of frame IDs (preserves order)."""
        if not frame_ids:
            return []
        placeholders = ",".join("?" * len(frame_ids))
        rows = self._fetchall(
            f"SELECT * FROM frames WHERE id IN ({placeholders})",
            frame_ids,
        )
        # Preserve the original order
        order = {fid: i for i, fid in enumerate(frame_ids)}
        return sorted(rows, key=lambda r: order.get(r["id"], 999))

    def search_frames_caption(self, query: str) -> list[dict]:
        """
        Naive LIKE search on captions.
        Used as fallback when CLIP semantic search is unavailable.
        """
        return self._fetchall(
            "SELECT * FROM frames WHERE caption LIKE ? ORDER BY ts DESC",
            [f"%{query}%"],
        )

    # ------------------------------------------------------------------
    # Query: objects
    # ------------------------------------------------------------------

    def get_vehicle_counts_today(
        self,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Return distinct track counts per vehicle class in one query.
        Replaces 4 separate get_objects_by_class() calls in the contextualize node.
        """
        vehicle_classes = ("car", "truck", "bus", "motorcycle")
        placeholders = ",".join("?" * len(vehicle_classes))
        sql = f"""
            SELECT o.class, COUNT(DISTINCT o.track_id) as distinct_tracks
            FROM objects o
            JOIN frames f ON o.frame_id = f.id
            WHERE o.class IN ({placeholders})
              AND o.track_id IS NOT NULL
              AND f.ts >= ?
              AND f.ts <= ?
            GROUP BY o.class
        """
        return self._fetchall(sql, list(vehicle_classes) + [start.isoformat(), end.isoformat()])

    def get_objects_for_frame(self, frame_id: str) -> list[dict]:
        """Return all detected objects for a single frame."""
        return self._fetchall(
            "SELECT * FROM objects WHERE frame_id=? ORDER BY confidence DESC",
            [frame_id],
        )

    def get_objects_by_class(
        self,
        class_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Fetch all object detections of a given class.
        Joins with frames to get timestamp + zone.
        """
        sql = """
            SELECT o.*, f.ts, f.zone, f.caption, f.frame_path
            FROM objects o
            JOIN frames f ON o.frame_id = f.id
            WHERE o.class = ?
        """
        args: list[Any] = [class_name]
        if start:
            sql  += " AND f.ts >= ?"
            args.append(start.isoformat())
        if end:
            sql  += " AND f.ts <= ?"
            args.append(end.isoformat())
        sql += " ORDER BY f.ts ASC"
        return self._fetchall(sql, args)

    def get_track_history(self, track_id: int) -> list[dict]:
        """Return all detections for a specific ByteTrack track_id."""
        return self._fetchall(
            """
            SELECT o.*, f.ts, f.zone, f.frame_path
            FROM objects o JOIN frames f ON o.frame_id = f.id
            WHERE o.track_id = ?
            ORDER BY f.ts ASC
            """,
            [track_id],
        )

    def get_object_class_counts(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """Summary: how many detections per class in a time window."""
        sql  = "SELECT class, COUNT(*) as count FROM objects o"
        args: list[Any] = []
        if start or end:
            sql += " JOIN frames f ON o.frame_id = f.id WHERE 1=1"
            if start:
                sql  += " AND f.ts >= ?"
                args.append(start.isoformat())
            if end:
                sql  += " AND f.ts <= ?"
                args.append(end.isoformat())
        sql += " GROUP BY class ORDER BY count DESC"
        return self._fetchall(sql, args)

    # ------------------------------------------------------------------
    # Query: events + alerts
    # ------------------------------------------------------------------

    def get_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        zone: Optional[str] = None,
    ) -> list[dict]:
        sql  = "SELECT * FROM events WHERE 1=1"
        args: list[Any] = []
        if event_type:
            sql  += " AND type = ?"
            args.append(event_type)
        if severity:
            sql  += " AND severity = ?"
            args.append(severity)
        if start:
            sql  += " AND start_ts >= ?"
            args.append(start.isoformat())
        if end:
            sql  += " AND start_ts <= ?"
            args.append(end.isoformat())
        if zone:
            sql  += " AND zone = ?"
            args.append(zone)
        sql += " ORDER BY start_ts DESC"
        return self._fetchall(sql, args)

    def get_unacked_alerts(self) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM alerts WHERE acked=0 ORDER BY ts DESC", []
        )

    def get_alerts_for_event(self, event_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM alerts WHERE event_id=? ORDER BY ts ASC", [event_id]
        )

    # ------------------------------------------------------------------
    # Stats (for dashboard summary cards)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        with self._get_conn() as conn:
            frames   = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            objects  = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
            events   = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            alerts   = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            unacked  = conn.execute("SELECT COUNT(*) FROM alerts WHERE acked=0").fetchone()[0]
        return {
            "frames":         frames,
            "objects":        objects,
            "events":         events,
            "alerts":         alerts,
            "unacked_alerts": unacked,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetchall(self, sql: str, args: list) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None
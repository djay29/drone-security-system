"""
hybrid_retriever.py
-------------------
Combines ChromaDB semantic search + SQLite structured queries into a
single interface that powers the "Ask" tab in the Streamlit dashboard
and the agent's context-retrieval step.

Query flow
----------
  1. CLIP text embedding of the query                  (ChromaDB)
  2. Top-K semantically similar frames                 (ChromaDB)
  3. Fetch full frame + event metadata for those IDs   (SQLite)
  4. Merge and rank by combined score                  (hybrid)

Also exposes direct structured queries for the agent rule engine:
  - frames_in_window()    — "what happened between 12:00 and 13:00"
  - events_summary()      — "show all high-severity events today"
  - object_history()      — "show all truck events"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from .sqlite_store import SQLiteStore
from .chroma_store import ChromaStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class FrameResult:
    """Unified result from a hybrid search."""

    def __init__(
        self,
        frame_id: str,
        ts: str,
        zone: str,
        caption: str,
        class_names: list[str],
        semantic_score: float,
        frame_index: int = 0,
        frame_path: Optional[str] = None,
        telemetry: Optional[dict] = None,
    ) -> None:
        self.frame_id       = frame_id
        self.ts             = ts
        self.zone           = zone
        self.caption        = caption
        self.class_names    = class_names
        self.semantic_score = semantic_score
        self.frame_index    = frame_index
        self.frame_path     = frame_path
        self.telemetry      = telemetry

    def to_dict(self) -> dict:
        d = {
            "frame_id":       self.frame_id,
            "ts":             self.ts,
            "zone":           self.zone,
            "caption":        self.caption,
            "class_names":    self.class_names,
            "semantic_score": self.semantic_score,
            "frame_index":    self.frame_index,
            "frame_path":     self.frame_path,
        }
        if self.telemetry:
            d["telemetry"] = self.telemetry
        return d

    def __repr__(self) -> str:
        return (
            f"FrameResult(ts={self.ts}, zone={self.zone!r}, "
            f"score={self.semantic_score:.3f}, caption={self.caption!r})"
        )


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Parameters
    ----------
    sqlite : SQLiteStore instance
    chroma : ChromaStore instance (must have embedder set for text queries)
    top_k  : Default number of results to return
    """

    def __init__(
        self,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        top_k: int = 10,
    ) -> None:
        self.sqlite = sqlite
        self.chroma = chroma
        self.top_k  = top_k

    # ------------------------------------------------------------------
    # Primary: semantic + structured hybrid search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        zone: Optional[str] = None,
        class_filter: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[FrameResult]:
        """
        Main entry point for natural-language queries.

        Combines:
          - CLIP semantic similarity from ChromaDB
          - Optional structured filters (zone, class, time window) from SQLite

        Parameters
        ----------
        query        : Natural language e.g. "blue truck near main gate"
        top_k        : Max results (defaults to self.top_k)
        zone         : e.g. "main_gate"
        class_filter : e.g. "truck"
        start / end  : Narrow to a time window

        Returns
        -------
        List of FrameResult sorted by semantic similarity (descending).
        """
        k = top_k or self.top_k

        # 1. Semantic search via ChromaDB
        try:
            chroma_hits = self.chroma.query_by_text(
                query,
                top_k=k * 2,   # over-fetch, then filter
                zone=zone,
                class_filter=class_filter,
            )
        except Exception as exc:
            logger.warning("[HybridRetriever] ChromaDB query failed: %s. Falling back to SQL.", exc)
            return self._sql_fallback(query, top_k=k, zone=zone, start=start, end=end)

        if not chroma_hits:
            return []

        # 2. Enrich from SQLite — get full frame rows for these IDs
        frame_ids     = [h["frame_id"] for h in chroma_hits]
        score_by_id   = {h["frame_id"]: h["score"] for h in chroma_hits}
        sql_rows      = self._fetch_frames_by_ids(frame_ids)

        # 3. Time-window filter (applied post-fetch for simplicity)
        if start or end:
            sql_rows = [
                r for r in sql_rows
                if _in_window(r.get("ts", ""), start, end)
            ]

        # 4. Merge and build results — batch-fetch class names in one query
        sql_frame_ids = [row["id"] for row in sql_rows]
        classes_by_id = _get_classes_batch(self.sqlite, sql_frame_ids)

        results = []
        for row in sql_rows:
            fid = row["id"]
            results.append(FrameResult(
                frame_id       = fid,
                ts             = row.get("ts", ""),
                zone           = row.get("zone", ""),
                caption        = row.get("caption", ""),
                class_names    = classes_by_id.get(fid, []),
                semantic_score = score_by_id.get(fid, 0.0),
                frame_index    = row.get("frame_index", 0),
                frame_path     = row.get("frame_path"),
                telemetry      = _parse_telemetry(row.get("telemetry_json")),
            ))

        results.sort(key=lambda r: r.semantic_score, reverse=True)
        return results[:k]

    # ------------------------------------------------------------------
    # Structured queries (for agent rule engine + dashboard)
    # ------------------------------------------------------------------

    def frames_in_window(
        self,
        start: datetime,
        end: datetime,
        zone: Optional[str] = None,
    ) -> list[dict]:
        """
        Return all frames in a time window with their detected objects.
        Used by: "what happened between 12:00 and 13:00"
        """
        frames = self.sqlite.get_frames_in_range(start, end, zone=zone)
        for frame in frames:
            frame["detections"] = self.sqlite.get_objects_for_frame(frame["id"])
        return frames

    def object_history(
        self,
        class_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """
        All detections of a given class across all frames.
        Used by: "show all truck events"
        """
        return self.sqlite.get_objects_by_class(class_name, start=start, end=end)

    def events_summary(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        zone: Optional[str] = None,
    ) -> list[dict]:
        """
        Return filtered events for the agent or dashboard.
        """
        return self.sqlite.get_events(
            event_type=event_type,
            severity=severity,
            start=start,
            end=end,
            zone=zone,
        )

    def temporal_search(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        class_filter: Optional[str] = None,
        zone: Optional[str] = None,
        top_k: int = 100,
    ) -> list["FrameResult"]:
        """
        SQL-first retrieval for time- and/or class-based queries.

        Unlike search(), this does NOT go through ChromaDB at all — it queries
        SQLite directly so that "show me people from 2 days ago" reliably returns
        every matching frame regardless of semantic similarity ranking.

        Priority:
          1. If class_filter given: query objects table (has ts via JOIN)
          2. Otherwise: query frames table directly with time window
        """
        if class_filter:
            rows = self.sqlite.get_objects_by_class(
                class_filter, start=start, end=end
            )
            # Deduplicate by frame_id while preserving time order
            seen: set[str] = set()
            frame_ids: list[str] = []
            for row in rows:
                fid = row["frame_id"]
                if fid not in seen:
                    seen.add(fid)
                    frame_ids.append(fid)
            frame_rows = self._fetch_frames_by_ids(frame_ids[:top_k])
        else:
            # No class filter — query frames table directly
            _start = start or datetime(2000, 1, 1, tzinfo=timezone.utc)
            _end   = end   or datetime(2100, 1, 1, tzinfo=timezone.utc)
            frame_rows = self.sqlite.get_frames_in_range(_start, _end, zone=zone)[:top_k]

        if zone:
            frame_rows = [r for r in frame_rows if r.get("zone") == zone]

        ids = [r["id"] for r in frame_rows]
        classes_by_id = _get_classes_batch(self.sqlite, ids)

        results = []
        for row in frame_rows:
            fid = row["id"]
            results.append(FrameResult(
                frame_id       = fid,
                ts             = row.get("ts", ""),
                zone           = row.get("zone", ""),
                caption        = row.get("caption", ""),
                class_names    = classes_by_id.get(fid, []),
                semantic_score = 1.0,
                frame_index    = row.get("frame_index", 0),
                frame_path     = row.get("frame_path"),
                telemetry      = _parse_telemetry(row.get("telemetry_json")),
            ))
        return results

    def unacked_alerts(self) -> list[dict]:
        return self.sqlite.get_unacked_alerts()

    def stats(self) -> dict:
        return {
            "sqlite": self.sqlite.get_stats(),
            "chroma": {"indexed_frames": self.chroma.count()},
        }

    # ------------------------------------------------------------------
    # Context builder (for LLM agent)
    # ------------------------------------------------------------------

    def build_context_for_agent(
        self,
        query: str,
        top_k: int = 5,
        zone: Optional[str] = None,
    ) -> str:
        """
        Returns a formatted string of the most relevant frames + events
        to inject into the agent's LLM prompt as retrieved context.
        """
        results = self.search(query, top_k=top_k, zone=zone)

        if not results:
            return "No relevant footage found for this query."

        lines = [f"Retrieved {len(results)} relevant frames:\n"]
        for i, r in enumerate(results, 1):
            telem_str = ""
            if r.telemetry:
                t = r.telemetry
                telem_str = (
                    f"\n   Telemetry: bat={t.get('battery_pct','?')}% "
                    f"alt={t.get('alt_m','?')}m "
                    f"spd={t.get('speed_ms','?')}m/s "
                    f"mode={t.get('flight_mode','?')} "
                    f"sig={t.get('signal_strength','?')}%"
                )
            lines.append(
                f"{i}. [{r.ts}] zone={r.zone or 'unknown'}  "
                f"score={r.semantic_score:.2f}\n"
                f"   Objects: {', '.join(r.class_names) or 'none'}\n"
                f"   Caption: {r.caption or '(no caption)'}"
                f"{telem_str}\n"
            )

        # Also pull recent events for context
        recent_events = self.sqlite.get_events()[:5]
        if recent_events:
            lines.append("\nRecent security events:")
            for ev in recent_events:
                lines.append(
                    f"  [{ev['start_ts']}] {ev['severity'].upper()} — "
                    f"{ev['type']}: {ev['description']}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_frames_by_ids(self, frame_ids: list[str]) -> list[dict]:
        """Bulk-fetch frame rows from SQLite by ID list."""
        if not frame_ids:
            return []
        placeholders = ",".join("?" * len(frame_ids))
        return self.sqlite._fetchall(
            f"SELECT * FROM frames WHERE id IN ({placeholders})", frame_ids
        )

    def _sql_fallback(
        self,
        query: str,
        top_k: int,
        zone: Optional[str],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> list[FrameResult]:
        """Fallback to SQLite LIKE caption search when ChromaDB is unavailable."""
        rows = self.sqlite.search_frames_caption(query)
        if zone:
            rows = [r for r in rows if r.get("zone") == zone]
        if start or end:
            rows = [r for r in rows if _in_window(r.get("ts", ""), start, end)]

        top_rows = rows[:top_k]
        fallback_ids = [row["id"] for row in top_rows]
        classes_by_id = _get_classes_batch(self.sqlite, fallback_ids)

        results = []
        for row in top_rows:
            fid = row["id"]
            results.append(FrameResult(
                frame_id       = fid,
                ts             = row.get("ts", ""),
                zone           = row.get("zone", ""),
                caption        = row.get("caption", ""),
                class_names    = classes_by_id.get(fid, []),
                semantic_score = 0.0,
                frame_index    = row.get("frame_index", 0),
                frame_path     = row.get("frame_path"),
                telemetry      = _parse_telemetry(row.get("telemetry_json")),
            ))
        return results


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _parse_telemetry(telemetry_json: str | None) -> dict | None:
    """Deserialise the telemetry_json column into a dict (or None)."""
    if not telemetry_json:
        return None
    try:
        return json.loads(telemetry_json)
    except (json.JSONDecodeError, TypeError):
        return None


def _in_window(ts_str: str, start: Optional[datetime], end: Optional[datetime]) -> bool:
    if not ts_str:
        return True
    try:
        ts = datetime.fromisoformat(ts_str)
        if start and ts < start:
            return False
        if end and ts > end:
            return False
    except ValueError:
        return True
    return True


def _get_classes_batch(sqlite: SQLiteStore, frame_ids: list[str]) -> dict[str, list[str]]:
    """Fetch distinct class names for multiple frames in a single query."""
    if not frame_ids:
        return {}
    placeholders = ",".join("?" * len(frame_ids))
    rows = sqlite._fetchall(
        f"SELECT DISTINCT frame_id, class FROM objects WHERE frame_id IN ({placeholders})",
        frame_ids,
    )
    result: dict[str, list[str]] = {fid: [] for fid in frame_ids}
    for row in rows:
        result[row["frame_id"]].append(row["class"])
    return result
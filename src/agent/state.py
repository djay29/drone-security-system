"""
state.py
--------
Defines the AgentState TypedDict that flows through every node in the
LangGraph pipeline.

LangGraph requires a single state object passed between nodes. Every node
reads from it and returns a partial dict to merge back in.

Flow:
  perceive → contextualize → rule_check → llm_judge → alert → log
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, TypedDict

from src.perception.yolo_detector import DetectedObject
from src.perception.frame_preprocessor import PreprocessedFrame


class AgentState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.
    All fields are optional (total=False) — nodes only populate
    what they produce.
    """

    # ── Input (set by perceive node) ──────────────────────────────────
    frame_id:      str
    frame_index:   int
    ts:            datetime
    video_id:      str
    zone:          str

    preprocessed:  PreprocessedFrame        # raw preprocessed frame
    detections:    list[DetectedObject]     # YOLO output
    caption:       str | None              # VLM caption (may be None on stride)
    embedding:     list[float] | None      # CLIP vector

    # ── Contextualize node output ─────────────────────────────────────
    track_durations:   dict[int, float]    # track_id → seconds in zone
    recent_events:     list[dict]          # last N events from SQLite
    context_summary:   str                 # short text summary for LLM

    # ── Rule check node output ────────────────────────────────────────
    rule_hits:     list[dict]              # list of triggered rule dicts
    needs_llm:     bool                    # True → route to llm_judge node

    # ── LLM judge node output ─────────────────────────────────────────
    llm_verdict:   str | None              # LLM's assessment
    llm_alert_msg: str | None             # Alert message drafted by LLM

    # ── Telemetry (set by perceive, forwarded through all nodes) ─────────
    telemetry:     dict | None            # drone telemetry snapshot for this frame

    # ── Alert node output ─────────────────────────────────────────────
    alerts_fired:  list[dict]             # alerts actually dispatched

    # ── Log node output ───────────────────────────────────────────────
    logged:        bool
"""
test_agent_pipeline.py
----------------------
Integration tests for the full LangGraph agent graph.

All heavy models (YOLO, VLM, CLIP) are replaced with lightweight stubs
so the graph can execute in milliseconds without GPU / API credentials.
The SQLite and ChromaDB stores are real (in-memory / ephemeral).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent.graph import build_agent
from src.agent.rule_engine import RuleEngine
from src.memory.chroma_store import ChromaStore
from src.memory.sqlite_store import SQLiteStore
from src.perception.clip_embedder import CLIPEmbedder
from src.perception.frame_preprocessor import FramePreprocessor
from src.perception.video_ingestor import FramePacket
from src.perception.yolo_detector import DetectedObject


# ---------------------------------------------------------------------------
# Stubs — replace real models with deterministic fakes
# ---------------------------------------------------------------------------

class StubDetector:
    """Returns a fixed list of detections."""

    def __init__(self, detections: list[DetectedObject] | None = None):
        self._detections = detections or []

    def detect_from_preprocessed(self, preprocessed):
        return self._detections


class StubCaptioner:
    """Returns a fixed caption string."""

    def __init__(self, caption: str = ""):
        self._caption = caption

    def caption_from_preprocessed(self, preprocessed, detections=None, zone=""):
        return self._caption

    def caption(self, *args, **kwargs):
        return self._caption


class StubEmbedder:
    """Returns a zero 512-dim vector."""

    def embed_preprocessed(self, preprocessed):
        return [0.0] * 512

    def embed_text(self, query: str):
        return [0.0] * 512


def _make_detection(class_name="person", confidence=0.9, track_id=1):
    return DetectedObject(class_name=class_name, confidence=confidence,
                          bbox=(100, 100, 200, 200), track_id=track_id)


def _make_preprocessed(frame_id: str, ts: datetime, zone: str = "gate"):
    packet = FramePacket(
        frame_id=frame_id,
        ts=ts,
        video_id="test_video",
        frame_index=0,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    proc = FramePreprocessor(yolo_size=640, vlm_every=1)
    return proc.process(packet)


# ---------------------------------------------------------------------------
# Minimal rules YAML fixtures
# ---------------------------------------------------------------------------

AFTER_HOURS_YAML = """
rules:
  - name: after_hours_person
    condition:
      object_class: person
      time_range: { start: "22:00", end: "06:00" }
    severity: high
    message_template: "Person at {zone} at {time}"
    needs_llm: false
"""

LOITERING_YAML = """
rules:
  - name: loitering
    condition:
      same_track_id_in_zone_seconds: 30
    severity: medium
    message_template: "Loitering track #{track_id} for {duration}s"
    needs_llm: true
"""


@pytest.fixture
def db():
    return SQLiteStore(":memory:")


@pytest.fixture
def chroma():
    return ChromaStore(persist_dir=None)


def _build(detector, captioner, embedder, db, chroma, rules_yaml: str, tmp_path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(rules_yaml)
    rule_engine = RuleEngine(rules_path)
    return build_agent(detector, captioner, embedder, db, chroma, rule_engine,
                       async_vlm=False, vlm_workers=1)


# ---------------------------------------------------------------------------
# Tests: no-alert path (no rule hits)
# ---------------------------------------------------------------------------

class TestNoAlertPath:

    def test_log_node_runs_when_no_rule_hits(self, tmp_path, db, chroma):
        """Daytime frame with no rules → no alerts, but frame still logged."""
        ts = datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)
        agent = _build(
            StubDetector([_make_detection("person")]),
            StubCaptioner("A person walking"),
            StubEmbedder(), db, chroma, AFTER_HOURS_YAML, tmp_path,
        )
        fid = str(uuid.uuid4())
        result = agent.invoke({
            "preprocessed": _make_preprocessed(fid, ts),
            "zone": "main_gate",
        })
        # When no rules fire, graph skips the alert node → alerts_fired not set.
        assert result.get("alerts_fired", []) == []
        assert result.get("logged") is True

    def test_frame_stored_in_sqlite_after_no_alert(self, tmp_path, db, chroma):
        ts = datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)
        agent = _build(
            StubDetector(), StubCaptioner(), StubEmbedder(),
            db, chroma, AFTER_HOURS_YAML, tmp_path,
        )
        fid = str(uuid.uuid4())
        agent.invoke({"preprocessed": _make_preprocessed(fid, ts), "zone": "gate"})

        frames = db.get_frames_by_video("test_video")
        assert any(f["id"] == fid for f in frames)


# ---------------------------------------------------------------------------
# Tests: alert path (rule hits, no LLM)
# ---------------------------------------------------------------------------

class TestAlertPath:

    def test_alert_fired_for_rule_hit(self, tmp_path, db, chroma):
        ts = datetime(2026, 4, 20, 23, 0, 0, tzinfo=timezone.utc)  # night
        agent = _build(
            StubDetector([_make_detection("person")]),
            StubCaptioner(), StubEmbedder(),
            db, chroma, AFTER_HOURS_YAML, tmp_path,
        )
        fid = str(uuid.uuid4())
        result = agent.invoke({"preprocessed": _make_preprocessed(fid, ts), "zone": "gate"})

        assert len(result["alerts_fired"]) == 1
        assert result["alerts_fired"][0]["rule_name"] == "after_hours_person"
        assert result["alerts_fired"][0]["severity"] == "high"

    def test_alert_stored_in_sqlite(self, tmp_path, db, chroma):
        ts = datetime(2026, 4, 20, 23, 0, 0, tzinfo=timezone.utc)
        agent = _build(
            StubDetector([_make_detection("person")]),
            StubCaptioner(), StubEmbedder(),
            db, chroma, AFTER_HOURS_YAML, tmp_path,
        )
        agent.invoke({
            "preprocessed": _make_preprocessed(str(uuid.uuid4()), ts),
            "zone": "gate",
        })

        events = db.get_events(event_type="after_hours_person")
        assert len(events) >= 1

    def test_alert_deduplication_cooldown(self, tmp_path, db, chroma):
        """Second identical alert within cooldown window should be suppressed."""
        ts = datetime(2026, 4, 20, 23, 0, 0, tzinfo=timezone.utc)
        agent = _build(
            StubDetector([_make_detection("person")]),
            StubCaptioner(), StubEmbedder(),
            db, chroma, AFTER_HOURS_YAML, tmp_path,
        )

        r1 = agent.invoke({"preprocessed": _make_preprocessed(str(uuid.uuid4()), ts), "zone": "gate"})
        r2 = agent.invoke({"preprocessed": _make_preprocessed(str(uuid.uuid4()), ts), "zone": "gate"})

        assert len(r1["alerts_fired"]) == 1
        assert len(r2["alerts_fired"]) == 0   # suppressed by cooldown


# ---------------------------------------------------------------------------
# Tests: LLM judge path (needs_llm=True)
# ---------------------------------------------------------------------------

class TestLLMJudgePath:

    def _build_with_llm_mock(self, tmp_path, db, chroma, verdict: str, msg: str):
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text(LOITERING_YAML)
        rule_engine = RuleEngine(rules_path)

        # Patch ChatBedrock BEFORE build_agent so the llm_judge node uses the mock.
        # ChatBedrock is instantiated inside make_llm_judge_node() at graph-build time.
        fake_response = MagicMock()
        fake_response.content = f"VERDICT: {verdict}\nALERT: {msg}\nREASON: test"
        with patch("src.agent.nodes.ChatBedrock") as MockLLM:
            MockLLM.return_value.invoke.return_value = fake_response
            agent = build_agent(
                StubDetector([_make_detection("person", track_id=5)]),
                StubCaptioner("A person standing still"),
                StubEmbedder(), db, chroma, rule_engine,
                async_vlm=False, vlm_workers=1,
            )
            ts_start = datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)
            ts_later  = datetime(2026, 4, 20, 14, 1, 5, tzinfo=timezone.utc)  # +65s

            # First invoke seeds the track (0s elapsed → no loitering)
            agent.invoke({
                "preprocessed": _make_preprocessed(str(uuid.uuid4()), ts_start),
                "zone": "gate",
            })
            # Second invoke: 65s have elapsed → loitering fires (threshold=30s)
            result = agent.invoke({
                "preprocessed": _make_preprocessed(str(uuid.uuid4()), ts_later),
                "zone": "gate",
            })
        return result

    def test_genuine_verdict_fires_alert(self, tmp_path, db, chroma):
        result = self._build_with_llm_mock(tmp_path, db, chroma,
                                            verdict="genuine", msg="Suspicious loitering")
        assert len(result["alerts_fired"]) == 1
        assert result["alerts_fired"][0]["message"] == "Suspicious loitering"

    def test_false_positive_stored_with_review_tag(self, tmp_path, db, chroma):
        result = self._build_with_llm_mock(tmp_path, db, chroma,
                                            verdict="false_positive", msg="N/A")
        # False positives are still stored but tagged for review
        assert len(result["alerts_fired"]) == 1
        assert "[LLM: REVIEW" in result["alerts_fired"][0]["message"]


# ---------------------------------------------------------------------------
# Tests: state field propagation
# ---------------------------------------------------------------------------

class TestStateFields:

    def test_perceive_node_sets_frame_id(self, tmp_path, db, chroma):
        ts = datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)
        agent = _build(StubDetector(), StubCaptioner(), StubEmbedder(),
                       db, chroma, AFTER_HOURS_YAML, tmp_path)
        fid = str(uuid.uuid4())
        result = agent.invoke({"preprocessed": _make_preprocessed(fid, ts), "zone": "gate"})
        assert result["frame_id"] == fid

    def test_perceive_node_sets_zone(self, tmp_path, db, chroma):
        ts = datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)
        agent = _build(StubDetector(), StubCaptioner("cap"), StubEmbedder(),
                       db, chroma, AFTER_HOURS_YAML, tmp_path)
        result = agent.invoke({
            "preprocessed": _make_preprocessed(str(uuid.uuid4()), ts),
            "zone": "server_room",
        })
        assert result["zone"] == "server_room"

    def test_caption_from_captioner_in_state(self, tmp_path, db, chroma):
        ts = datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)
        agent = _build(StubDetector(), StubCaptioner("Two people near the fence"),
                       StubEmbedder(), db, chroma, AFTER_HOURS_YAML, tmp_path)
        result = agent.invoke({
            "preprocessed": _make_preprocessed(str(uuid.uuid4()), ts),
            "zone": "gate",
        })
        assert result.get("caption") == "Two people near the fence"

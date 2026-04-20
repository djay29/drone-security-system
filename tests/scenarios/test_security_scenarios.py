"""
test_security_scenarios.py
--------------------------
End-to-end scenario tests — simulate real security events passing through
the full rule engine and verify the correct alert is raised with the right
severity, rule name, and LLM routing decision.

These tests use RuleEngine directly (no graph overhead) so they run fast
and stay readable as living documentation of the system's security posture.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from src.agent.rule_engine import RuleEngine
from src.perception.yolo_detector import DetectedObject


def make_detection(class_name="person", confidence=0.9, track_id=1,
                   bbox=(100, 100, 200, 200)):
    return DetectedObject(class_name=class_name, confidence=confidence,
                          bbox=bbox, track_id=track_id)


# ---------------------------------------------------------------------------
# Load production rules for scenario tests
# ---------------------------------------------------------------------------

PROD_RULES = Path("configs/rules.yaml")


@pytest.fixture(scope="module")
def engine():
    """Load the production rules.yaml once for all scenario tests."""
    return RuleEngine(PROD_RULES)


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 4, 20, hour, minute, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Scenario 1: After-hours intruder
# ---------------------------------------------------------------------------

class TestAfterHoursIntruder:

    def test_person_at_2am_triggers_high_alert(self, engine):
        hits = engine.evaluate(
            ts=_ts(2), zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "after_hours_person" in names
        match = next(h for h in hits if h["rule_name"] == "after_hours_person")
        assert match["severity"] == "high"
        assert match["needs_llm"] is False

    def test_vehicle_at_midnight_triggers_high_alert(self, engine):
        hits = engine.evaluate(
            ts=_ts(0), zone="parking",
            detections=[make_detection("truck")],
            track_durations={}, vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert any("vehicle" in n for n in names)

    def test_no_alert_for_person_at_midday(self, engine):
        hits = engine.evaluate(
            ts=_ts(12), zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "after_hours_person" not in names


# ---------------------------------------------------------------------------
# Scenario 2: Loitering suspect
# ---------------------------------------------------------------------------

class TestLoiteringSuspect:

    def test_loitering_fires_after_threshold(self, engine):
        """Person in zone for 90 seconds — loitering threshold (60s) exceeded."""
        hits = engine.evaluate(
            ts=_ts(14), zone="parking",
            detections=[make_detection("person", track_id=7)],
            track_durations={7: 90.0},
            vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "loitering" in names
        match = next(h for h in hits if h["rule_name"] == "loitering")
        assert match["severity"] == "medium"
        assert match["needs_llm"] is True   # loitering requires LLM confirmation

    def test_loitering_below_threshold_no_alert(self, engine):
        hits = engine.evaluate(
            ts=_ts(14), zone="parking",
            detections=[make_detection("person", track_id=7)],
            track_durations={7: 30.0},   # only 30s
            vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "loitering" not in names


# ---------------------------------------------------------------------------
# Scenario 3: Crowd gathering / tailgating
# ---------------------------------------------------------------------------

class TestCrowdAndTailgating:

    def test_crowd_fires_for_3_or_more_people(self, engine):
        detections = [make_detection("person", track_id=i) for i in range(3)]
        hits = engine.evaluate(
            ts=_ts(14), zone="courtyard",
            detections=detections,
            track_durations={}, vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "crowd_gathering" in names

    def test_tailgating_fires_at_gate_with_2_people(self, engine):
        """Two people simultaneously at main_gate → tailgating risk."""
        detections = [
            make_detection("person", track_id=1),
            make_detection("person", track_id=2),
        ]
        hits = engine.evaluate(
            ts=_ts(9), zone="main_gate",
            detections=detections,
            track_durations={}, vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "tailgating" in names

    def test_tailgating_does_not_fire_in_wrong_zone(self, engine):
        detections = [
            make_detection("person", track_id=1),
            make_detection("person", track_id=2),
        ]
        hits = engine.evaluate(
            ts=_ts(9), zone="parking_lot",   # not main_gate
            detections=detections,
            track_durations={}, vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "tailgating" not in names


# ---------------------------------------------------------------------------
# Scenario 4: Gate tampering (caption-based)
# ---------------------------------------------------------------------------

class TestGateTampering:

    def test_gate_tampering_fires_for_tools_at_gate(self, engine):
        """
        Scenario from the context: three people near a gate, one using tools.
        The VLM caption should trigger the gate_tampering_attempt rule.
        """
        hits = engine.evaluate(
            ts=_ts(14), zone="main_gate",
            detections=[make_detection("person", track_id=i) for i in range(3)],
            track_durations={},
            vehicle_counts={},
            caption="Three people near the gate, one bending down with tools near the lock",
        )
        names = [h["rule_name"] for h in hits]
        assert "gate_tampering_attempt" in names
        match = next(h for h in hits if h["rule_name"] == "gate_tampering_attempt")
        assert match["needs_llm"] is True

    def test_gate_tampering_no_fire_on_unrelated_caption(self, engine):
        hits = engine.evaluate(
            ts=_ts(14), zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
            caption="A person walking past the building",
        )
        names = [h["rule_name"] for h in hits]
        assert "gate_tampering_attempt" not in names


# ---------------------------------------------------------------------------
# Scenario 5: Coordinated breach attempt
# ---------------------------------------------------------------------------

class TestCoordinatedBreachAttempt:

    def test_coordinated_breach_fires_for_guard_standing_watch(self, engine):
        """
        Scenario: one person bending at gate (tools), another standing guard.
        Should trigger coordinated_breach_attempt or lookout_behavior.
        """
        hits = engine.evaluate(
            ts=_ts(14), zone="main_gate",
            detections=[make_detection("person", track_id=i) for i in range(2)],
            track_durations={}, vehicle_counts={},
            caption=(
                "Two people near the gated entrance — one bending down inspecting "
                "the lock with tools while another stands guard watching the area"
            ),
        )
        names = [h["rule_name"] for h in hits]
        triggered = any(n in names for n in (
            "coordinated_breach_attempt", "lookout_behavior",
            "gate_tampering_attempt", "tool_use_near_infrastructure",
        ))
        assert triggered, f"Expected a breach-related rule to fire. Got: {names}"


# ---------------------------------------------------------------------------
# Scenario 6: Multiple simultaneous alerts
# ---------------------------------------------------------------------------

class TestMultipleSimultaneousAlerts:

    def test_after_hours_plus_loitering_both_fire(self, engine):
        """
        Night-time + long loitering = two independent rules should both fire.
        """
        hits = engine.evaluate(
            ts=_ts(23), zone="parking",
            detections=[make_detection("person", track_id=3)],
            track_durations={3: 120.0},   # 2 min
            vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "after_hours_person" in names
        assert "loitering" in names
        # High-severity rule present
        severities = {h["rule_name"]: h["severity"] for h in hits}
        assert severities["after_hours_person"] == "high"

    def test_all_hits_have_required_fields(self, engine):
        """Every hit must have rule_name, severity, message, needs_llm."""
        hits = engine.evaluate(
            ts=_ts(23), zone="main_gate",
            detections=[make_detection("person", track_id=1)],
            track_durations={1: 90.0},
            vehicle_counts={},
            caption="Person using tools near the gate lock",
        )
        for hit in hits:
            assert "rule_name" in hit
            assert "severity"  in hit
            assert "message"   in hit
            assert "needs_llm" in hit
            assert hit["severity"] in ("low", "medium", "high")
            assert isinstance(hit["needs_llm"], bool)
            assert isinstance(hit["message"], str) and len(hit["message"]) > 0


# ---------------------------------------------------------------------------
# Scenario 7: Empty / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_no_detections_no_alerts(self, engine):
        hits = engine.evaluate(
            ts=_ts(23), zone="main_gate",
            detections=[], track_durations={}, vehicle_counts={},
        )
        # Only caption-based or telemetry rules could fire with no detections
        detection_rules = {"after_hours_person", "loitering", "crowd_gathering",
                           "tailgating", "after_hours_vehicle", "forbidden_zone"}
        names = {h["rule_name"] for h in hits}
        assert names.isdisjoint(detection_rules)

    def test_empty_caption_does_not_crash(self, engine):
        hits = engine.evaluate(
            ts=_ts(14), zone="gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
            caption="",
        )
        assert isinstance(hits, list)

    def test_none_caption_does_not_crash(self, engine):
        hits = engine.evaluate(
            ts=_ts(14), zone="gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
            caption=None,
        )
        assert isinstance(hits, list)

"""
test_rule_engine.py
-------------------
Unit tests for RuleEngine — condition evaluation, routing, message rendering.

Strategy: each test writes a minimal rules.yaml via tmp_path so tests are
fully isolated from production configs/rules.yaml content.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agent.rule_engine import RuleEngine
from src.perception.yolo_detector import DetectedObject


def make_detection(class_name="person", confidence=0.9, track_id=1,
                   bbox=(100, 100, 200, 200)):
    return DetectedObject(class_name=class_name, confidence=confidence,
                          bbox=bbox, track_id=track_id)


def write_rules_yaml(tmp_path, rules_yaml: str):
    p = tmp_path / "rules.yaml"
    p.write_text(rules_yaml)
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ts_day():
    return datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc)   # 14:00


@pytest.fixture
def ts_night():
    return datetime(2026, 4, 20, 23, 0, 0, tzinfo=timezone.utc)   # 23:00


@pytest.fixture
def ts_early():
    return datetime(2026, 4, 20, 3, 0, 0, tzinfo=timezone.utc)    # 03:00 (past midnight)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AFTER_HOURS_RULE = """
rules:
  - name: after_hours_person
    condition:
      object_class: person
      time_range: { start: "22:00", end: "06:00" }
    severity: high
    message_template: "Person at {zone} at {time}"
    needs_llm: false
"""

LOITERING_RULE = """
rules:
  - name: loitering
    condition:
      same_track_id_in_zone_seconds: 30
    severity: medium
    message_template: "Loitering: track #{track_id} for {duration}s"
    needs_llm: true
"""

CROWD_RULE = """
rules:
  - name: crowd_gathering
    condition:
      object_class: person
      min_object_count: 3
    severity: medium
    message_template: "Crowd: {count} people at {zone}"
    needs_llm: true
"""

ZONE_RULE = """
rules:
  - name: forbidden_zone
    condition:
      zone_in: [restricted_area, server_room]
    severity: high
    message_template: "Object detected in forbidden zone {zone}"
    needs_llm: false
"""

CAPTION_RULE = """
rules:
  - name: gate_tampering
    condition:
      caption_keywords: [tools, gate, lock]
      caption_min_match: 2
    severity: high
    message_template: "Gate tampering detected at {zone}"
    needs_llm: true
"""

VEHICLE_RULE = """
rules:
  - name: after_hours_vehicle
    condition:
      object_class_in: [car, truck, bus]
      time_range: { start: "22:00", end: "06:00" }
    severity: high
    message_template: "Vehicle at {zone} after hours"
    needs_llm: false
"""

COMBINED_RULES = """
rules:
  - name: after_hours_person
    condition:
      object_class: person
      time_range: { start: "22:00", end: "06:00" }
    severity: high
    message_template: "Person at {zone}"
    needs_llm: false
  - name: loitering
    condition:
      same_track_id_in_zone_seconds: 30
    severity: medium
    message_template: "Loitering track #{track_id}"
    needs_llm: true
"""


# ===========================================================================
# After-hours person
# ===========================================================================

class TestAfterHoursPerson:

    def test_triggers_at_night(self, tmp_path, ts_night):
        engine = RuleEngine(write_rules_yaml(tmp_path, AFTER_HOURS_RULE))
        hits = engine.evaluate(
            ts=ts_night,
            zone="main_gate",
            detections=[make_detection("person")],
            track_durations={},
            vehicle_counts={},
        )
        assert len(hits) == 1
        assert hits[0]["rule_name"] == "after_hours_person"
        assert hits[0]["severity"] == "high"

    def test_no_trigger_during_day(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, AFTER_HOURS_RULE))
        hits = engine.evaluate(
            ts=ts_day,
            zone="main_gate",
            detections=[make_detection("person")],
            track_durations={},
            vehicle_counts={},
        )
        assert hits == []

    def test_triggers_past_midnight(self, tmp_path, ts_early):
        """03:00 is inside the 22:00–06:00 window."""
        engine = RuleEngine(write_rules_yaml(tmp_path, AFTER_HOURS_RULE))
        hits = engine.evaluate(
            ts=ts_early,
            zone="main_gate",
            detections=[make_detection("person")],
            track_durations={},
            vehicle_counts={},
        )
        assert len(hits) == 1

    def test_no_trigger_without_person_detection(self, tmp_path, ts_night):
        engine = RuleEngine(write_rules_yaml(tmp_path, AFTER_HOURS_RULE))
        hits = engine.evaluate(
            ts=ts_night,
            zone="main_gate",
            detections=[make_detection("car")],   # car, not person
            track_durations={},
            vehicle_counts={},
        )
        assert hits == []

    def test_message_contains_zone(self, tmp_path, ts_night):
        engine = RuleEngine(write_rules_yaml(tmp_path, AFTER_HOURS_RULE))
        hits = engine.evaluate(
            ts=ts_night,
            zone="main_gate",
            detections=[make_detection("person")],
            track_durations={},
            vehicle_counts={},
        )
        assert "main_gate" in hits[0]["message"]


# ===========================================================================
# Loitering
# ===========================================================================

class TestLoitering:

    def test_triggers_when_duration_exceeded(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, LOITERING_RULE))
        hits = engine.evaluate(
            ts=ts_day,
            zone="parking",
            detections=[make_detection("person", track_id=42)],
            track_durations={42: 60.0},   # 60s > 30s threshold
            vehicle_counts={},
        )
        assert len(hits) == 1
        assert hits[0]["rule_name"] == "loitering"
        assert hits[0]["needs_llm"] is True

    def test_no_trigger_below_threshold(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, LOITERING_RULE))
        hits = engine.evaluate(
            ts=ts_day,
            zone="parking",
            detections=[make_detection("person", track_id=42)],
            track_durations={42: 10.0},   # 10s < 30s threshold
            vehicle_counts={},
        )
        assert hits == []

    def test_no_trigger_without_track_durations(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, LOITERING_RULE))
        hits = engine.evaluate(
            ts=ts_day,
            zone="parking",
            detections=[make_detection("person", track_id=None)],
            track_durations={},
            vehicle_counts={},
        )
        assert hits == []

    def test_triggers_on_exact_threshold(self, tmp_path, ts_day):
        """Duration equal to threshold should trigger."""
        engine = RuleEngine(write_rules_yaml(tmp_path, LOITERING_RULE))
        hits = engine.evaluate(
            ts=ts_day,
            zone="parking",
            detections=[make_detection("person", track_id=7)],
            track_durations={7: 30.0},
            vehicle_counts={},
        )
        assert len(hits) == 1


# ===========================================================================
# Crowd gathering
# ===========================================================================

class TestCrowdGathering:

    def test_triggers_at_min_count(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, CROWD_RULE))
        detections = [make_detection("person", track_id=i) for i in range(3)]
        hits = engine.evaluate(
            ts=ts_day, zone="courtyard",
            detections=detections,
            track_durations={}, vehicle_counts={},
        )
        assert len(hits) == 1
        assert hits[0]["rule_name"] == "crowd_gathering"

    def test_no_trigger_below_min_count(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, CROWD_RULE))
        detections = [make_detection("person", track_id=i) for i in range(2)]
        hits = engine.evaluate(
            ts=ts_day, zone="courtyard",
            detections=detections,
            track_durations={}, vehicle_counts={},
        )
        assert hits == []

    def test_non_person_detections_not_counted(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, CROWD_RULE))
        # 2 people + 1 car = does not reach min_object_count=3 for person
        detections = [
            make_detection("person", track_id=1),
            make_detection("person", track_id=2),
            make_detection("car", track_id=3),
        ]
        hits = engine.evaluate(
            ts=ts_day, zone="courtyard",
            detections=detections,
            track_durations={}, vehicle_counts={},
        )
        assert hits == []


# ===========================================================================
# Zone rule
# ===========================================================================

class TestForbiddenZone:

    def test_triggers_in_forbidden_zone(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, ZONE_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="restricted_area",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
        )
        assert len(hits) == 1
        assert hits[0]["rule_name"] == "forbidden_zone"

    def test_no_trigger_in_allowed_zone(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, ZONE_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
        )
        assert hits == []

    def test_triggers_for_second_zone_in_list(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, ZONE_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="server_room",
            detections=[make_detection("car")],
            track_durations={}, vehicle_counts={},
        )
        assert len(hits) == 1


# ===========================================================================
# Caption-based rules
# ===========================================================================

class TestCaptionKeywords:

    def test_triggers_when_min_keywords_matched(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, CAPTION_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
            caption="A person crouching near the gate holding tools",
        )
        assert len(hits) == 1
        assert hits[0]["rule_name"] == "gate_tampering"

    def test_no_trigger_below_min_match(self, tmp_path, ts_day):
        """Only 1 keyword present (tools), caption_min_match=2."""
        engine = RuleEngine(write_rules_yaml(tmp_path, CAPTION_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
            caption="A person carrying tools near the building",
        )
        assert hits == []

    def test_no_trigger_empty_caption(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, CAPTION_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
            caption="",
        )
        assert hits == []

    def test_case_insensitive_keyword_match(self, tmp_path, ts_day):
        """Keywords should match regardless of caption casing."""
        engine = RuleEngine(write_rules_yaml(tmp_path, CAPTION_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
            caption="TOOLS found near the GATE and LOCK",
        )
        assert len(hits) == 1


# ===========================================================================
# Vehicle rules
# ===========================================================================

class TestVehicleAfterHours:

    def test_triggers_for_car_at_night(self, tmp_path, ts_night):
        engine = RuleEngine(write_rules_yaml(tmp_path, VEHICLE_RULE))
        hits = engine.evaluate(
            ts=ts_night, zone="parking",
            detections=[make_detection("car")],
            track_durations={}, vehicle_counts={},
        )
        assert len(hits) == 1

    def test_triggers_for_truck_at_night(self, tmp_path, ts_night):
        engine = RuleEngine(write_rules_yaml(tmp_path, VEHICLE_RULE))
        hits = engine.evaluate(
            ts=ts_night, zone="parking",
            detections=[make_detection("truck")],
            track_durations={}, vehicle_counts={},
        )
        assert len(hits) == 1

    def test_no_trigger_for_car_during_day(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, VEHICLE_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="parking",
            detections=[make_detection("car")],
            track_durations={}, vehicle_counts={},
        )
        assert hits == []


# ===========================================================================
# Multiple rules / edge cases
# ===========================================================================

class TestMultipleRules:

    def test_multiple_rules_fire_simultaneously(self, tmp_path, ts_night):
        """Both after-hours and loitering should fire for the same frame."""
        engine = RuleEngine(write_rules_yaml(tmp_path, COMBINED_RULES))
        hits = engine.evaluate(
            ts=ts_night, zone="main_gate",
            detections=[make_detection("person", track_id=5)],
            track_durations={5: 120.0},
            vehicle_counts={},
        )
        rule_names = [h["rule_name"] for h in hits]
        assert "after_hours_person" in rule_names
        assert "loitering" in rule_names

    def test_no_detections_produces_no_hits(self, tmp_path, ts_night):
        engine = RuleEngine(write_rules_yaml(tmp_path, COMBINED_RULES))
        hits = engine.evaluate(
            ts=ts_night, zone="main_gate",
            detections=[],
            track_durations={}, vehicle_counts={},
        )
        assert hits == []

    def test_needs_llm_false_propagated(self, tmp_path, ts_night):
        engine = RuleEngine(write_rules_yaml(tmp_path, AFTER_HOURS_RULE))
        hits = engine.evaluate(
            ts=ts_night, zone="x",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
        )
        assert hits[0]["needs_llm"] is False

    def test_needs_llm_true_propagated(self, tmp_path, ts_day):
        engine = RuleEngine(write_rules_yaml(tmp_path, LOITERING_RULE))
        hits = engine.evaluate(
            ts=ts_day, zone="x",
            detections=[make_detection("person", track_id=1)],
            track_durations={1: 60.0},
            vehicle_counts={},
        )
        assert hits[0]["needs_llm"] is True

    def test_reload_picks_up_updated_rules(self, tmp_path, ts_night):
        """Hot-reload should swap in new rules without restart."""
        path = write_rules_yaml(tmp_path, AFTER_HOURS_RULE)
        engine = RuleEngine(path)

        # Replace with loitering-only rules
        path.write_text(LOITERING_RULE)
        engine.reload()

        # After-hours rule should no longer fire
        hits = engine.evaluate(
            ts=ts_night, zone="main_gate",
            detections=[make_detection("person")],
            track_durations={}, vehicle_counts={},
        )
        names = [h["rule_name"] for h in hits]
        assert "after_hours_person" not in names

    def test_missing_yaml_uses_defaults(self):
        """RuleEngine falls back to built-in defaults when file not found."""
        engine = RuleEngine("nonexistent_path/rules.yaml")
        assert len(engine.rules) > 0

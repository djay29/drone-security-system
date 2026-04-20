"""
rule_engine.py
--------------
Loads rules from configs/rules.yaml and evaluates them against the
current AgentState. Returns a list of triggered rule hits.

Rules are purely declarative YAML — no code changes needed to add
new alert conditions. The engine handles:

  after_hours_person     — person detected outside allowed hours
  loitering              — same track_id in zone > N seconds
  vehicle_repeat_entry   — same vehicle class enters > N times in window
  forbidden_zone         — any object in a restricted zone
  low_battery            — telemetry battery below threshold (bonus)

Each rule hit dict:
  {
    "rule_name": str,
    "severity":  "low" | "medium" | "high",
    "message":   str,              # formatted from template
    "needs_llm": bool,             # True → escalate to LLM judge
  }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, time
from pathlib import Path
from typing import Any

import yaml

from src.perception.yolo_detector import DetectedObject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------

class RuleEngine:
    """
    Parameters
    ----------
    rules_path : Path to rules.yaml (configs/rules.yaml)
    """

    def __init__(self, rules_path: str | Path = "configs/rules.yaml") -> None:
        self.rules_path = Path(rules_path)
        self.rules: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.rules_path.exists():
            logger.warning(
                "[RuleEngine] rules.yaml not found at %s — using defaults.", self.rules_path
            )
            self.rules = _default_rules()
            return

        with self.rules_path.open() as fh:
            data = yaml.safe_load(fh)
        self.rules = data.get("rules", [])
        logger.info("[RuleEngine] Loaded %d rules from %s", len(self.rules), self.rules_path)

    def reload(self) -> None:
        """Hot-reload rules without restarting the agent."""
        self._load()

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        ts: datetime,
        zone: str,
        detections: list[DetectedObject],
        track_durations: dict[int, float],   # track_id → seconds in zone
        vehicle_counts: dict[str, int],      # class → count today
        telemetry: dict | None = None,
        caption: str | None = None,
    ) -> list[dict]:
        """
        Evaluate all rules against the current frame state.

        Returns
        -------
        List of triggered rule hit dicts (may be empty).
        """
        hits = []
        for rule in self.rules:
            name      = rule.get("name", "unknown")
            condition = rule.get("condition", {})
            severity  = rule.get("severity", "low")
            template  = rule.get("message_template", f"Rule {name} triggered")
            needs_llm = rule.get("needs_llm", False)

            try:
                triggered, ctx = self._check(
                    name, condition, ts, zone, detections,
                    track_durations, vehicle_counts, telemetry or {},
                    caption or "",
                )
            except Exception as exc:
                logger.error("[RuleEngine] Error evaluating rule '%s': %s", name, exc)
                continue

            if triggered:
                message = _render_template(template, ctx)
                hits.append({
                    "rule_name": name,
                    "severity":  severity,
                    "message":   message,
                    "needs_llm": needs_llm,
                    "context":   ctx,
                })
                logger.info("[RuleEngine] HIT: %s — %s", name, message)

        return hits

    # ------------------------------------------------------------------
    # Individual rule checkers
    # ------------------------------------------------------------------

    def _check(
        self,
        name: str,
        condition: dict,
        ts: datetime,
        zone: str,
        detections: list[DetectedObject],
        track_durations: dict[int, float],
        vehicle_counts: dict[str, int],
        telemetry: dict,
        caption: str = "",
    ) -> tuple[bool, dict]:
        """
        Compound AND-based condition evaluator.

        Every condition key present in the rule is evaluated independently.
        ALL must pass for the rule to trigger (logical AND).
        This allows arbitrary combinations like:
          object_class + time_range + zone_in  (after-hours zone entry)
          object_class + min_object_count      (crowd gathering)
          object_class + min_object_count + zone_in  (tailgating at gate)

        Returns (triggered: bool, context: dict for template rendering).
        """
        ctx: dict[str, Any] = {
            "zone": zone or "unknown",
            "time": ts.strftime("%H:%M:%S"),
            "ts":   ts.isoformat(),
        }

        results: list[bool] = []
        known_keys = {
            "time_range", "zone_in",
            "object_class", "object_class_in", "object_class_filter",
            "min_object_count", "total_object_count_above",
            "same_track_id_in_zone_seconds",
            "same_vehicle_enters_times", "within_hours",
            "vehicle_without_person", "min_distinct_vehicle_classes",
            "battery_below", "altitude_above", "altitude_below",
            "caption_keywords", "caption_min_match",
        }

        unknown = set(condition.keys()) - known_keys
        if unknown:
            logger.warning("[RuleEngine] Rule '%s' has unrecognised condition keys: %s", name, unknown)

        # ── Time range ────────────────────────────────────────────────
        if "time_range" in condition:
            t_start = _parse_time(condition["time_range"]["start"])
            t_end   = _parse_time(condition["time_range"]["end"])
            current = ts.time().replace(second=0, microsecond=0)
            results.append(_time_in_range(current, t_start, t_end))

        # ── Zone filter ───────────────────────────────────────────────
        if "zone_in" in condition:
            results.append(zone in condition["zone_in"])

        # ── Single object class presence (no count requirement) ───────
        # Skipped when min_object_count is also set (handled there instead)
        if "object_class" in condition and "min_object_count" not in condition:
            target = condition["object_class"]
            matched = [d for d in detections if d.class_name == target]
            ctx["object_class"] = target
            results.append(len(matched) > 0)

        # ── Object class in list (no count requirement) ───────────────
        if "object_class_in" in condition and "min_object_count" not in condition:
            allowed  = condition["object_class_in"]
            matched  = [d for d in detections if d.class_name in allowed]
            if matched:
                ctx["object_class"] = matched[0].class_name
            results.append(len(matched) > 0)

        # ── Min object count (crowd / tailgating) ─────────────────────
        if "min_object_count" in condition:
            threshold    = condition["min_object_count"]
            target_class = condition.get("object_class") or condition.get("object_class_in")
            if isinstance(target_class, list):
                matched = [d for d in detections if d.class_name in target_class]
            elif target_class:
                matched = [d for d in detections if d.class_name == target_class]
            else:
                matched = list(detections)
            count = len(matched)
            ctx["count"]        = count
            ctx["object_class"] = target_class if isinstance(target_class, str) else (matched[0].class_name if matched else "")
            results.append(count >= threshold)

        # ── Total object count above threshold ────────────────────────
        if "total_object_count_above" in condition:
            threshold = condition["total_object_count_above"]
            count     = len(detections)
            ctx["count"] = count
            results.append(count > threshold)

        # ── Track duration (loitering / stopped vehicle) ──────────────
        if "same_track_id_in_zone_seconds" in condition:
            threshold     = condition["same_track_id_in_zone_seconds"]
            class_filter  = condition.get("object_class_filter")   # list or None
            hit_track = False
            for track_id, duration in track_durations.items():
                if duration < threshold:
                    continue
                if class_filter:
                    # Only trigger if the long-duration track belongs to one of the filtered classes
                    track_classes = {d.class_name for d in detections if d.track_id == track_id}
                    if not track_classes.intersection(set(class_filter)):
                        continue
                ctx["track_id"] = track_id
                ctx["duration"] = int(duration)
                hit_track = True
                break
            results.append(hit_track)

        # ── Vehicle repeat entry ──────────────────────────────────────
        if "same_vehicle_enters_times" in condition:
            threshold = condition["same_vehicle_enters_times"]
            hit_vehicle = False
            for cls, count in vehicle_counts.items():
                if count >= threshold:
                    ctx["description"] = cls
                    ctx["count"]       = count
                    hit_vehicle = True
                    break
            results.append(hit_vehicle)

        # ── Vehicle present without any person visible ─────────────────
        if "vehicle_without_person" in condition:
            vehicle_classes = {"car", "truck", "bus", "motorcycle", "boat"}
            has_vehicle = any(d.class_name in vehicle_classes for d in detections)
            has_person  = any(d.class_name == "person" for d in detections)
            results.append(has_vehicle and not has_person)

        # ── Convoy: multiple distinct vehicle types in same frame ──────
        if "min_distinct_vehicle_classes" in condition:
            vehicle_classes   = {"car", "truck", "bus", "motorcycle"}
            threshold         = condition["min_distinct_vehicle_classes"]
            present_v_classes = {d.class_name for d in detections if d.class_name in vehicle_classes}
            ctx["vehicle_types"] = ", ".join(sorted(present_v_classes))
            results.append(len(present_v_classes) >= threshold)

        # ── Battery threshold ─────────────────────────────────────────
        if "battery_below" in condition:
            threshold      = condition["battery_below"]
            battery        = telemetry.get("battery", 100)
            ctx["battery"] = battery
            results.append(battery < threshold)

        # ── Altitude thresholds ───────────────────────────────────────
        if "altitude_above" in condition:
            threshold        = condition["altitude_above"]
            altitude         = telemetry.get("alt_m", 0)
            ctx["altitude"]  = round(altitude, 1)
            results.append(altitude > threshold)

        if "altitude_below" in condition:
            threshold        = condition["altitude_below"]
            altitude         = telemetry.get("alt_m", 0)
            ctx["altitude"]  = round(altitude, 1)
            results.append(altitude < threshold)

        # ── Caption keyword matching ───────────────────────────────────
        # Matches against the VLM-generated scene description.
        # caption_keywords  : list of words/phrases (OR candidates)
        # caption_min_match : how many must appear (default 1 = any match)
        if "caption_keywords" in condition:
            keywords   = [str(kw).lower() for kw in condition["caption_keywords"]]
            min_match  = int(condition.get("caption_min_match", 1))
            cap_lower  = caption.lower()
            matched    = [kw for kw in keywords if kw in cap_lower]
            ctx["matched_keywords"]  = ", ".join(matched) if matched else "none"
            ctx["caption_snippet"]   = caption[:120]
            results.append(len(matched) >= min_match)

        # ── No recognised conditions found ────────────────────────────
        if not results:
            logger.warning("[RuleEngine] No evaluable conditions for rule '%s' — skipping.", name)
            return False, ctx

        return all(results), ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time(t_str: str) -> time:
    h, m = t_str.split(":")
    return time(int(h), int(m))


def _time_in_range(current: time, start: time, end: time) -> bool:
    """Handle ranges that wrap midnight (e.g. 22:00 → 06:00)."""
    if start <= end:
        return start <= current <= end
    # wraps midnight
    return current >= start or current <= end


def _render_template(template: str, ctx: dict) -> str:
    try:
        return template.format(**ctx)
    except KeyError:
        return template


def _default_rules() -> list[dict]:
    """Fallback rules used when rules.yaml is missing."""
    return [
        {
            "name": "after_hours_person",
            "condition": {"object_class": "person", "time_range": {"start": "22:00", "end": "06:00"}},
            "severity": "high",
            "message_template": "Person detected at {zone} at {time}",
            "needs_llm": False,
        },
        {
            "name": "after_hours_zone_entry",
            "condition": {"object_class": "person", "zone_in": ["garage", "main_gate"], "time_range": {"start": "20:00", "end": "07:00"}},
            "severity": "high",
            "message_template": "Person detected in {zone} during restricted hours at {time}",
            "needs_llm": False,
        },
        {
            "name": "loitering",
            "condition": {"same_track_id_in_zone_seconds": 60},
            "severity": "medium",
            "message_template": "Loitering detected: track #{track_id} in {zone} for {duration}s",
            "needs_llm": True,
        },
        {
            "name": "crowd_gathering",
            "condition": {"object_class": "person", "min_object_count": 3},
            "severity": "medium",
            "message_template": "Crowd gathering: {count} people detected at {zone} at {time}",
            "needs_llm": True,
        },
        {
            "name": "tailgating",
            "condition": {"object_class": "person", "min_object_count": 2, "zone_in": ["main_gate"]},
            "severity": "high",
            "message_template": "Multiple persons ({count}) at {zone} simultaneously — possible tailgating",
            "needs_llm": True,
        },
        {
            "name": "after_hours_vehicle",
            "condition": {"object_class_in": ["car", "truck", "bus", "motorcycle"], "time_range": {"start": "22:00", "end": "06:00"}},
            "severity": "high",
            "message_template": "Vehicle ({object_class}) detected at {zone} outside allowed hours at {time}",
            "needs_llm": False,
        },
        {
            "name": "vehicle_repeat_entry",
            "condition": {"same_vehicle_enters_times": 2, "within_hours": 24},
            "severity": "low",
            "message_template": "Vehicle '{description}' entered {count} times today",
            "needs_llm": True,
        },
        {
            "name": "stopped_vehicle",
            "condition": {"same_track_id_in_zone_seconds": 120, "object_class_filter": ["car", "truck", "bus"]},
            "severity": "medium",
            "message_template": "Stopped vehicle: track #{track_id} stationary at {zone} for {duration}s",
            "needs_llm": True,
        },
        {
            "name": "unattended_vehicle",
            "condition": {"vehicle_without_person": True},
            "severity": "low",
            "message_template": "Unattended vehicle detected at {zone} — no personnel visible",
            "needs_llm": False,
        },
        {
            "name": "convoy_detection",
            "condition": {"min_distinct_vehicle_classes": 2},
            "severity": "medium",
            "message_template": "Multiple vehicle types at {zone}: {vehicle_types}",
            "needs_llm": True,
        },
        {
            "name": "forbidden_zone",
            "condition": {"object_class_in": ["person", "car", "truck"], "zone_in": ["restricted_area"]},
            "severity": "high",
            "message_template": "{object_class} detected in restricted zone: {zone}",
            "needs_llm": False,
        },
        {
            "name": "perimeter_breach",
            "condition": {"object_class_in": ["person", "car", "truck", "motorcycle"], "zone_in": ["perimeter"]},
            "severity": "low",
            "message_template": "{object_class} detected at perimeter: {zone} at {time}",
            "needs_llm": False,
        },
        {
            "name": "gate_tampering_attempt",
            "condition": {
                "object_class": "person", "min_object_count": 1,
                "caption_keywords": ["open", "opening", "tool", "tools", "lock", "picking", "tampering", "bends down", "bent down", "handle", "forced", "prying"],
                "caption_min_match": 2,
            },
            "severity": "high",
            "message_template": "Gate tampering attempt at {zone} — matched: {matched_keywords}",
            "needs_llm": True,
        },
        {
            "name": "coordinated_breach_attempt",
            "condition": {
                "object_class": "person", "min_object_count": 2,
                "caption_keywords": ["guard", "lookout", "tool", "open", "gate", "fence", "break", "force", "attempt", "while another", "one bends", "one holds"],
                "caption_min_match": 2,
            },
            "severity": "high",
            "message_template": "Coordinated breach attempt at {zone}: {count} people — {matched_keywords}",
            "needs_llm": True,
        },
        {
            "name": "lookout_behavior",
            "condition": {
                "object_class": "person", "min_object_count": 2,
                "caption_keywords": ["guard", "lookout", "keeping watch", "standing guard", "watching", "stands nearby", "third stands", "one watches"],
                "caption_min_match": 1,
            },
            "severity": "medium",
            "message_template": "Lookout behavior detected at {zone}: {count} people, one possibly standing guard",
            "needs_llm": True,
        },
        {
            "name": "tool_use_near_infrastructure",
            "condition": {
                "object_class": "person",
                "caption_keywords": ["tool", "tools", "wrench", "cutters", "wire cutters", "bolt cutters", "screwdriver", "holding tools"],
                "caption_min_match": 1,
            },
            "severity": "high",
            "message_template": "Person with tools detected near {zone} at {time} — {matched_keywords}",
            "needs_llm": True,
        },
        {
            "name": "suspicious_crouching",
            "condition": {
                "object_class": "person",
                "caption_keywords": ["crouch", "crouching", "bending", "bends", "kneel", "kneeling", "inspecting", "examining", "possibly inspecting", "hunched"],
                "caption_min_match": 1,
            },
            "severity": "medium",
            "message_template": "Suspicious crouching/inspecting behavior at {zone} at {time}",
            "needs_llm": True,
        },
        {
            "name": "perimeter_climbing_attempt",
            "condition": {
                "object_class": "person",
                "caption_keywords": ["climb", "climbing", "scale", "scaling", "jump", "over the fence", "over the wall", "near the fence"],
                "caption_min_match": 1,
            },
            "severity": "high",
            "message_template": "Possible perimeter climbing detected at {zone}",
            "needs_llm": True,
        },
        {
            "name": "forced_entry_attempt",
            "condition": {
                "object_class": "person",
                "caption_keywords": ["forcing", "forced entry", "break in", "breaking in", "jimmying", "prying open", "cutting", "breaking", "push open"],
                "caption_min_match": 1,
            },
            "severity": "high",
            "message_template": "Forced entry attempt detected at {zone} at {time}",
            "needs_llm": True,
        },
        {
            "name": "suspicious_group_activity",
            "condition": {
                "object_class": "person", "min_object_count": 2,
                "caption_keywords": ["attempting", "attempt", "secured", "unauthorized", "suspicious", "unknown individuals", "unidentified", "breaching"],
                "caption_min_match": 1,
            },
            "severity": "medium",
            "message_template": "Suspicious group activity at {zone}: {count} people — {matched_keywords}",
            "needs_llm": True,
        },
        {
            "name": "high_object_density",
            "condition": {"total_object_count_above": 5},
            "severity": "low",
            "message_template": "High activity: {count} objects detected at {zone} at {time}",
            "needs_llm": False,
        },
        {
            "name": "battery_critical",
            "condition": {"battery_below": 10},
            "severity": "high",
            "message_template": "CRITICAL: Drone battery at {battery}% — immediate return required",
            "needs_llm": False,
        },
        {
            "name": "low_battery",
            "condition": {"battery_below": 20},
            "severity": "medium",
            "message_template": "Drone battery low: {battery}% at {zone}",
            "needs_llm": False,
        },
        {
            "name": "altitude_anomaly_high",
            "condition": {"altitude_above": 120},
            "severity": "medium",
            "message_template": "Drone altitude too high: {altitude}m at {zone}",
            "needs_llm": False,
        },
        {
            "name": "altitude_anomaly_low",
            "condition": {"altitude_below": 5},
            "severity": "medium",
            "message_template": "Drone altitude dangerously low: {altitude}m at {zone}",
            "needs_llm": False,
        },
    ]
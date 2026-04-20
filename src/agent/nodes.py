"""
nodes.py
--------
LangGraph node functions for the drone security agent pipeline.

Each function:
  - Takes AgentState as input
  - Returns a partial dict to merge back into state
  - Has no side effects beyond its stated purpose

Node order:
  perceive → contextualize → rule_check → (llm_judge?) → alert → log
                                               ↑
                                     conditional edge:
                                     needs_llm=True → llm_judge
                                     needs_llm=False → alert
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

from dotenv import load_dotenv

from src.agent.state import AgentState
from src.agent.rule_engine import RuleEngine
from src.memory.sqlite_store import SQLiteStore
from src.memory.chroma_store import ChromaStore
from src.perception.yolo_detector import YOLODetector
from src.perception.vlm_captioner import VLMCaptioner
from src.perception.clip_embedder import CLIPEmbedder

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node: perceive
# ---------------------------------------------------------------------------

def make_perceive_node(
    detector: YOLODetector,
    captioner: VLMCaptioner,
    embedder: CLIPEmbedder,
    vlm_executor=None,   # concurrent.futures.ThreadPoolExecutor | None
):
    """
    Factory that closes over the perception components.
    Returns the node function.

    Sync mode (vlm_executor=None):
        YOLO → VLM (stride-gated, blocks ~500ms) → CLIP
        Use for file processing where latency is acceptable.

    Async mode (vlm_executor provided):
        YOLO → CLIP run every frame (~60ms total).
        VLM is submitted to the thread pool when run_vlm=True and a
        Future is not already pending. The last completed caption is
        returned immediately — no blocking on the Bedrock/Moondream call.
        Use for RTSP / high-FPS sources.
    """
    from concurrent.futures import Future as _Future

    # Async VLM state (lives in this closure; worker is single-threaded so no lock needed)
    _vlm: dict = {"future": None, "last_caption": None}

    def perceive(state: AgentState) -> dict:
        preprocessed = state["preprocessed"]
        packet       = preprocessed.packet
        zone         = state.get("zone", packet.metadata.get("zone", ""))

        # ── YOLO (always, ~20-50ms) ──────────────────────────────────
        detections = detector.detect_from_preprocessed(preprocessed)

        # ── CLIP (always, ~20-50ms) ──────────────────────────────────
        embedding = embedder.embed_preprocessed(preprocessed)

        # ── VLM caption ──────────────────────────────────────────────
        if vlm_executor is None:
            # Synchronous — blocks until caption is ready
            caption = captioner.caption_from_preprocessed(
                preprocessed, detections=detections, zone=zone
            )
        else:
            # Async — collect completed result, submit new job, return last caption
            if _vlm["future"] is not None and _vlm["future"].done():
                try:
                    result = _vlm["future"].result(timeout=0)
                    if result:
                        _vlm["last_caption"] = result
                except Exception as exc:
                    logger.warning("[Perceive] Async VLM future failed: %s", exc)
                finally:
                    _vlm["future"] = None

            # Submit new VLM job when stride says it's time and no job is pending
            if (
                preprocessed.run_vlm
                and preprocessed.vlm_input is not None
                and _vlm["future"] is None
            ):
                _vlm["future"] = vlm_executor.submit(
                    captioner.caption,
                    packet.frame_id,
                    preprocessed.vlm_input,
                    detections,
                    zone,
                )
                logger.debug("[Perceive] VLM job submitted for frame %s", packet.frame_id[:8])

            # Return last completed caption (may be None for first few frames)
            caption = _vlm["last_caption"]

        # Extract telemetry attached by VideoIngestor (may be None)
        telemetry = packet.metadata.get("telemetry")

        return {
            "frame_id":    packet.frame_id,
            "frame_index": packet.frame_index,
            "ts":          packet.ts,
            "video_id":    packet.video_id,
            "zone":        zone,
            "detections":  detections,
            "caption":     caption,
            "embedding":   embedding,
            "telemetry":   telemetry,
        }

    return perceive


# ---------------------------------------------------------------------------
# Node: contextualize
# ---------------------------------------------------------------------------

def make_contextualize_node(sqlite: SQLiteStore):
    """
    Builds context from accumulated history:
      - How long has each track been in this zone?
      - How many vehicle entries happened today?
      - What events fired recently?
    """

    # In-process track duration accumulator
    # {(zone, track_id) -> first_seen_ts}
    _track_first_seen: dict[tuple, datetime] = {}

    def contextualize(state: AgentState) -> dict:
        ts          = state["ts"]
        zone        = state.get("zone", "")
        detections  = state.get("detections", [])

        # ── Track duration ───────────────────────────────────────────
        track_durations: dict[int, float] = {}
        for det in detections:
            if det.track_id is None:
                continue
            key = (zone, det.track_id)
            if key not in _track_first_seen:
                _track_first_seen[key] = ts
            elapsed = (ts - _track_first_seen[key]).total_seconds()
            track_durations[det.track_id] = elapsed

        # Clean up tracks no longer visible (not in current detections)
        active_keys = {(zone, d.track_id) for d in detections if d.track_id is not None}
        stale = [k for k in _track_first_seen if k not in active_keys]
        for k in stale:
            del _track_first_seen[k]

        # ── Vehicle entry counts (today) ─────────────────────────────
        today_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        vehicle_counts: dict[str, int] = {}
        vehicle_rows = sqlite.get_vehicle_counts_today(start=today_start, end=ts)
        for row in vehicle_rows:
            cls = row["class"]
            distinct = row["distinct_tracks"]
            if distinct > 0:
                vehicle_counts[cls] = distinct

        # ── Recent events (last 10 mins) ─────────────────────────────
        recent_events = sqlite.get_events(
            start=ts - timedelta(minutes=10),
            end=ts,
        )

        # ── Short context summary for LLM ────────────────────────────
        det_summary = _summarise_detections(detections)
        track_summary = ", ".join(
            f"track#{tid} {int(dur)}s" for tid, dur in track_durations.items()
        ) or "none"
        event_summary = (
            "; ".join(f"{e['type']}({e['severity']})" for e in recent_events[:3])
            or "none"
        )

        # ── Telemetry summary ────────────────────────────────────────
        telemetry = state.get("telemetry")
        telemetry_summary = "none"
        if telemetry:
            telemetry_summary = (
                f"bat={telemetry.get('battery_pct', '?')}% "
                f"alt={telemetry.get('alt_m', '?')}m "
                f"spd={telemetry.get('speed_ms', '?')}m/s "
                f"mode={telemetry.get('flight_mode', '?')} "
                f"sig={telemetry.get('signal_strength', '?')}%"
            )

        context_summary = (
            f"Zone: {zone or 'unknown'} | "
            f"Time: {ts.strftime('%H:%M:%S')} | "
            f"Detections: {det_summary} | "
            f"Track durations: {track_summary} | "
            f"Telemetry: {telemetry_summary} | "
            f"Recent events: {event_summary}"
        )

        return {
            "track_durations":  track_durations,
            "recent_events":    recent_events,
            "context_summary":  context_summary,
            "_vehicle_counts":  vehicle_counts,
        }

    return contextualize


# ---------------------------------------------------------------------------
# Node: rule_check
# ---------------------------------------------------------------------------

def make_rule_check_node(rule_engine: RuleEngine):
    """
    Evaluates all YAML rules against current state.
    Sets needs_llm=True if any hit requires LLM escalation.
    """

    def rule_check(state: AgentState) -> dict:
        hits = rule_engine.evaluate(
            ts              = state["ts"],
            zone            = state.get("zone", ""),
            detections      = state.get("detections", []),
            track_durations = state.get("track_durations", {}),
            vehicle_counts  = state.get("_vehicle_counts", {}),
            telemetry       = state.get("telemetry"),
            caption         = state.get("caption") or "",
        )

        needs_llm = any(h["needs_llm"] for h in hits)

        return {
            "rule_hits": hits,
            "needs_llm": needs_llm,
        }

    return rule_check


# ---------------------------------------------------------------------------
# Node: llm_judge
# ---------------------------------------------------------------------------

def make_llm_judge_node(region: str = ""):
    """
    Called only when needs_llm=True.
    Uses Claude Haiku via Amazon Bedrock (langchain_aws) to assess
    ambiguous situations and draft alert messages.

    Auth: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env vars.
    Region and model_id can be configured via AWS_REGION and BEDROCK_MODEL_ID env vars.
    """

    region = region or os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("VLM_CAPTIONER_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

    llm = ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={"max_tokens": 200},
    )

    def llm_judge(state: AgentState) -> dict:
        rule_hits       = state.get("rule_hits", [])
        context_summary = state.get("context_summary", "")
        caption         = state.get("caption", "")

        hits_text = "\n".join(
            f"- [{h['severity'].upper()}] {h['rule_name']}: {h['message']}"
            for h in rule_hits
        )
        
        logger.debug("[LLMJudge] context=%s | caption=%s", context_summary, caption)

        prompt = f"""You are a drone security analyst AI.

Context: {context_summary}
Visual description: {caption or '(no caption available)'}

Triggered security rules:
{hits_text}

Task:
1. Assess whether these rule hits represent a genuine security concern or a false positive.
2. If genuine, write a concise alert message (1-2 sentences, factual, no speculation).
3. If false positive, explain briefly why.

Respond in this exact format:
VERDICT: <genuine|false_positive>
ALERT: <alert message or 'N/A'>
REASON: <one sentence explanation>"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            verdict   = _extract_field(raw, "VERDICT")
            alert_msg = _extract_field(raw, "ALERT")
            reason    = _extract_field(raw, "REASON")

            logger.info("[LLMJudge] verdict=%s | %s", verdict, reason)

            return {
                "llm_verdict":   verdict,
                "llm_alert_msg": alert_msg if verdict == "genuine" else None,
            }

        except Exception as exc:
            logger.error("[LLMJudge] Bedrock call failed: %s — defaulting to rule hits.", exc)
            return {
                "llm_verdict":   "genuine",
                "llm_alert_msg": rule_hits[0]["message"] if rule_hits else None,
            }

    return llm_judge


# ---------------------------------------------------------------------------
# Node: alert
# ---------------------------------------------------------------------------

def make_alert_node(sqlite: SQLiteStore):
    """
    Fires alerts for genuine rule hits.
    Deduplicates: won't re-fire the same rule in the same zone
    within a cooldown window (default 5 min).
    """

    # {(rule_name, zone) -> last_fired_ts}
    _last_fired: dict[tuple, datetime] = {}
    COOLDOWN_SECONDS = 300   # 5 minutes

    def alert(state: AgentState) -> dict:
        rule_hits  = state.get("rule_hits", [])
        llm_verdict = state.get("llm_verdict")
        llm_msg    = state.get("llm_alert_msg")
        ts         = state["ts"]
        zone       = state.get("zone", "")
        frame_id   = state["frame_id"]
        alerts_fired = []
        # logging.info(f"State is: {state}")
        for hit in rule_hits:
            rule_name = hit["rule_name"]
            severity  = hit["severity"]
            is_llm_hit = hit.get("needs_llm", False)
            is_false_positive = is_llm_hit and llm_verdict == "false_positive"

            # Deduplication cooldown
            key = (rule_name, zone)
            last = _last_fired.get(key)
            if last and (ts - last).total_seconds() < COOLDOWN_SECONDS:
                logger.debug("[Alert] Cooldown active for %s in %s — skipping.", rule_name, zone)
                continue

            # Draft message — prefer LLM message for escalated genuine hits;
            # annotate with review tag if LLM flagged as false positive (still store for audit trail)
            if is_false_positive:
                message = f"[LLM: REVIEW — possible false positive] {hit['message']}"
                logger.info("[Alert] LLM flagged false positive for %s — storing with review tag.", rule_name)
            elif is_llm_hit and llm_msg:
                message = llm_msg
            else:
                message = hit["message"]

            # Store event in SQLite
            event_id = str(uuid.uuid4())
            sqlite.store_event(
                event_id    = event_id,
                start_ts    = ts,
                event_type  = rule_name,
                severity    = severity,
                description = message,
                frame_ids   = [frame_id],
                zone        = zone,
            )

            # Store alert
            alert_id = str(uuid.uuid4())
            sqlite.store_alert(
                alert_id = alert_id,
                event_id = event_id,
                ts       = ts,
                channel  = "dashboard",
                message  = message,
            )

            _last_fired[key] = ts
            alerts_fired.append({
                "alert_id":  alert_id,
                "event_id":  event_id,
                "rule_name": rule_name,
                "severity":  severity,
                "message":   message,
                "ts":        ts.isoformat(),
            })

            logger.info("[Alert] FIRED [%s] %s: %s", severity.upper(), rule_name, message)

        return {"alerts_fired": alerts_fired}

    return alert


# ---------------------------------------------------------------------------
# Node: log
# ---------------------------------------------------------------------------

def make_log_node(sqlite: SQLiteStore, chroma: ChromaStore):
    """
    Persists the frame + detections + embedding into both stores.
    Always runs — even if no alerts fired.
    """

    def log(state: AgentState) -> dict:
        frame_id    = state["frame_id"]
        ts          = state["ts"]
        video_id    = state["video_id"]
        zone        = state.get("zone", "")
        caption     = state.get("caption") or ""
        embedding   = state.get("embedding")
        detections  = state.get("detections", [])
        preprocessed = state.get("preprocessed")

        frame_path = None
        if preprocessed and preprocessed.packet.frame_path:
            frame_path = str(preprocessed.packet.frame_path)

        # SQLite: frame (with telemetry)
        sqlite.store_frame(
            frame_id    = frame_id,
            ts          = ts,
            video_id    = video_id,
            frame_index = state.get("frame_index", 0),
            frame_path  = frame_path,
            caption     = caption or None,
            telemetry   = state.get("telemetry"),
            zone        = zone or None,
        )

        # SQLite: objects
        if detections:
            sqlite.store_objects(frame_id, [
                {
                    "class_name":  d.class_name,
                    "confidence":  d.confidence,
                    "bbox":        list(d.bbox),
                    "track_id":    d.track_id,
                }
                for d in detections
            ])

        # ChromaDB: embedding
        if embedding:
            chroma.add_frame(
                frame_id    = frame_id,
                embedding   = embedding,
                frame_index = state.get("frame_index", 0),
                ts          = ts.isoformat(),
                video_id    = video_id,
                zone        = zone,
                caption     = caption,
                class_names = [d.class_name for d in detections],
            )

        return {"logged": True}

    return log


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _summarise_detections(detections) -> str:
    if not detections:
        return "none"
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    return ", ".join(f"{v}x{k}" for k, v in counts.items())


def _extract_field(text: str, field: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"{field}:"):
            return line[len(field) + 1:].strip()
    return ""
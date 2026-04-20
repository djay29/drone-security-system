"""
streamlit_app.py
----------------
Drone Security Analyst — Streamlit Dashboard

Three tabs:
  🔴 Live      — Process video file frame-by-frame, show detections + alerts
  📋 Timeline  — Filterable event + alert log from SQLite
  💬 Ask       — RAG chat: LLM answer + source frames from Chroma + SQLite

Run:
    streamlit run src/dashboard/streamlit_app.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import time
import json
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from queue import Queue, Empty

from dotenv import load_dotenv

import cv2
import numpy as np
import streamlit as st

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.perception.video_ingestor     import VideoIngestor
from src.perception.frame_preprocessor import FramePreprocessor
from src.perception.yolo_detector      import YOLODetector
from src.perception.vlm_captioner      import VLMCaptioner, CaptionBackend
from src.perception.clip_embedder      import CLIPEmbedder
from src.memory.sqlite_store           import SQLiteStore
from src.memory.chroma_store           import ChromaStore
from src.memory.hybrid_retriever       import HybridRetriever
from src.agent.rule_engine             import RuleEngine
from src.agent.graph                   import build_agent
from src.pipeline.stream_processor     import StreamProcessor, recommended_sample_every
from src.telemetry.simulator           import TelemetrySimulator

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",   # saves logs to file
    filemode="a"
)

logging.info("App started")
# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Drone Security Analyst",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark industrial aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0c0f;
    color: #c8d0d8;
}
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; letter-spacing: 0.05em; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1318;
    border-right: 1px solid #1e2530;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1318;
    border-bottom: 1px solid #1e2530;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    padding: 0.6rem 1.4rem;
    color: #5a6a7a;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
    background: transparent !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f1318;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 0.8rem;
}

/* Alert boxes */
.alert-high   { background:#1a0a0a; border-left:3px solid #ff3b3b; padding:8px 12px; margin:4px 0; border-radius:2px; font-size:0.85rem; }
.alert-medium { background:#1a130a; border-left:3px solid #ff9500; padding:8px 12px; margin:4px 0; border-radius:2px; font-size:0.85rem; }
.alert-low    { background:#0a1218; border-left:3px solid #00aaff; padding:8px 12px; margin:4px 0; border-radius:2px; font-size:0.85rem; }

/* Caption strip */
.caption-box {
    background:#0f1318; border:1px solid #1e2530;
    padding:8px 12px; border-radius:2px;
    font-size:0.82rem; color:#8a9aaa;
    font-family:'Share Tech Mono', monospace;
}

/* Chat bubbles */
.chat-user     { background:#0f1a2a; border-left:3px solid #0088ff; padding:10px 14px; margin:6px 0; border-radius:2px; }
.chat-assistant{ background:#0f1318; border-left:3px solid #00e5ff; padding:10px 14px; margin:6px 0; border-radius:2px; }
.source-frame  { background:#080b0e; border:1px solid #1e2530; padding:8px; margin:3px 0; border-radius:2px; font-size:0.78rem; font-family:'Share Tech Mono',monospace; }

/* Buttons */
.stButton > button {
    background: #0f1318;
    border: 1px solid #00e5ff44;
    color: #00e5ff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    border-radius: 2px;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #00e5ff11;
    border-color: #00e5ff;
}

/* Status dot */
.status-live { color:#ff3b3b; animation: blink 1s step-start infinite; }
@keyframes blink { 50% { opacity:0; } }

div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resource initialisation
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising pipeline…")
def init_pipeline():
    yolo_model = os.getenv("YOLO_MODEL_NAME", "yolov8s.pt")
    yolo_confidence = float(os.getenv("YOLO_CONFIDENCE", "0.4"))
    detector    = YOLODetector(model_name=yolo_model, confidence=yolo_confidence, use_tracking=True)
    captioner   = VLMCaptioner(backend=CaptionBackend.AUTO)
    embedder    = CLIPEmbedder()
    db          = SQLiteStore("data/sql_data/drone_security.db")
    chroma      = ChromaStore("data/chroma", embedder=embedder)
    retriever   = HybridRetriever(db, chroma)
    rule_engine = RuleEngine("configs/rules.yaml")
    # async_vlm=True: VLM runs in a background thread pool so YOLO + rules
    # are never blocked by the ~500ms Bedrock/Moondream caption call.
    agent       = build_agent(detector, captioner, embedder, db, chroma, rule_engine,
                              async_vlm=True, vlm_workers=2)
    return {
        "detector":   detector,
        "captioner":  captioner,
        "embedder":   embedder,
        "db":         db,
        "chroma":     chroma,
        "retriever":  retriever,
        "agent":      agent,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🛸 DRONE SECURITY")
    st.markdown("---")

    source_type = st.radio("Input source", ["📁 Video File", "📡 RTSP Stream"], horizontal=True)

    if source_type == "📁 Video File":
        video_path  = st.text_input("Video file path", value="data/videos/drone_patrol.mp4")
        rtsp_url    = None
    else:
        rtsp_url    = st.text_input("RTSP URL", value="rtsp://username:password@192.168.1.1:554/stream")
        video_path  = None
    sample_every = st.slider("Sample every N frames", 1, 15, 5)

    if source_type == "📁 Video File":
        limit_frames = st.checkbox("Limit frames processed", value=False)
        if limit_frames:
            max_frames: int | None = st.slider("Max frames", 10, 2000, 300)
        else:
            max_frames = None
            st.caption("Processing all frames — press STOP to end early.")
    else:
        # RTSP: no hard cap — runs until STOP is pressed
        max_frames = None
        st.caption("RTSP stream runs until **STOP** is pressed.")
        rec_24  = recommended_sample_every(24)
        rec_30  = recommended_sample_every(30)
        rec_60  = recommended_sample_every(60)
        st.caption(f"💡 Recommended sample_every: 24fps→{rec_24}, 30fps→{rec_30}, 60fps→{rec_60} (targets ~6fps processed)")
    vlm_every    = st.slider("VLM caption every N frames", 1, 20, 5)
    # aws_region   = st.text_input("AWS Region", value="us-east-1")

    st.markdown("---")
    st.markdown("**Frame storage quality**")

    # (save_size, save_quality) — save_size=None means native resolution
    _STORAGE_PRESETS = {
        "Balanced — 640×360 / q75 (~35 KB/frame)":   ((640, 360), 75),
        "High — 854×480 / q82 (~65 KB/frame)":        ((854, 480), 82),
        "Compact — 426×240 / q65 (~15 KB/frame)":     ((426, 240), 65),
        "Minimal — 320×180 / q55 (~8 KB/frame)":      ((320, 180), 55),
        "Native resolution / q90 (full source size)":  (None,       90),
    }
    preset_label = st.selectbox(
        "Save preset",
        list(_STORAGE_PRESETS.keys()),
        index=0,
        help=(
            "Controls only the JPEG thumbnail written to disk for the dashboard. "
            "YOLO (640 px), VLM (378 px), and CLIP (224 px) always use their own "
            "optimal resolutions — this setting does not affect detection quality."
        ),
    )
    _save_size, _save_quality = _STORAGE_PRESETS[preset_label]

    # Rough estimate: 35 KB baseline at Balanced, scaled by quality ratio
    if _save_size:
        _est_kb = int(35 * (_save_quality / 75) * (_save_size[0] * _save_size[1]) / (640 * 360))
        st.caption(f"≈ {_est_kb} KB/frame · {round(_est_kb * 100 / 1024, 1)} MB per 100 frames")

    st.markdown("---")
    st.markdown("**Telemetry simulation**")
    telem_scenario = st.selectbox(
        "Scenario",
        ["None (no telemetry)", "perimeter_patrol", "night_watch", "emergency_response", "battery_low_test"],
        index=1,
        help="Attach simulated drone telemetry to every processed frame.",
    )
    if telem_scenario != "None (no telemetry)":
        telem_start_bat = st.slider("Start battery %", 10, 100, 95)
    else:
        telem_start_bat = 95

    st.markdown("---")
    stats_placeholder = st.empty()


# ---------------------------------------------------------------------------
# RAG answer helper (Bedrock) — defined before tabs so it's in scope
# ---------------------------------------------------------------------------

def _bedrock_rag_answer(
    query: str,
    context: str,
    region: str,
    chat_history: list | None = None,
) -> str:
    """
    Call Claude Haiku on Bedrock to answer a RAG query.

    chat_history: list of {"role": "user"|"assistant", "content": str} dicts.
    When provided the full conversation is sent as multi-turn messages so the
    model can resolve follow-up references ("those vehicles", "the same zone", etc.).
    """
    try:
        from langchain_aws import ChatBedrock
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        region = region or os.getenv("AWS_REGION", "us-east-1")

        llm = ChatBedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs={"max_tokens": 400},
        )

        system = (
            "You are a drone security analyst. Answer questions about surveillance "
            "footage based only on the retrieved context below. Be concise and factual. "
            "If the context doesn't contain enough information, say so.\n\n"
            f"Retrieved surveillance context:\n{context}"
        )

        messages: list = [SystemMessage(content=system)]

        # Inject prior turns (up to 4 exchanges = 8 messages) so the model
        # can resolve pronoun references in follow-up questions.
        if chat_history:
            for msg in chat_history[:-1][-8:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=query))

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as exc:
        lines = [l for l in context.splitlines() if l.strip()]
        return f"(LLM unavailable: {exc})\n\n" + "\n".join(lines[:8])


def _reformulate_query(query: str, chat_history: list, region: str) -> str:
    """
    Rewrite a follow-up question as a self-contained search query.

    Example: "What about the vehicles?" + prior context about people loitering
    → "vehicles detected in surveillance footage"
    Falls back to the original query on any error or when there's no history.
    """
    if len(chat_history) < 2:
        return query

    history_text = ""
    for msg in chat_history[-6:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    try:
        from langchain_aws import ChatBedrock
        from langchain_core.messages import HumanMessage

        model_id = os.getenv("VLM_CAPTIONER_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        region = region or os.getenv("AWS_REGION", "us-east-1")

        llm = ChatBedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs={"max_tokens": 80},
        )

        prompt = (
            f"Conversation about surveillance footage:\n{history_text}\n"
            f"Follow-up question: {query}\n\n"
            "Rewrite the follow-up as a standalone search query for a surveillance "
            "database. Return ONLY the rewritten query, nothing else."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        reformulated = response.content.strip()
        return reformulated if reformulated else query

    except Exception:
        return query


def _suggest_followups(query: str, answer: str, region: str) -> list[str]:
    """
    Generate up to 3 suggested follow-up questions a security analyst might ask
    after receiving `answer` to `query`.  Returns [] on any failure.
    """
    try:
        from langchain_aws import ChatBedrock
        from langchain_core.messages import HumanMessage

        model_id = os.getenv("VLM_CAPTIONER_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        region = region or os.getenv("AWS_REGION", "us-east-1")

        llm = ChatBedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs={"max_tokens": 120},
        )

        prompt = (
            f"Surveillance Q&A:\nQ: {query}\nA: {answer}\n\n"
            "Suggest 3 concise follow-up questions a security analyst might ask next. "
            "Return ONLY the 3 questions, one per line, no numbering or bullets."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
        return lines[:3]

    except Exception:
        return []


def _parse_query_intent(query: str, aws_region: str) -> dict:
    """
    Use the LLM to extract structured filters from a natural-language query.

    Returns a dict:
      {
        "has_time":       bool,
        "start":          datetime | None,   # UTC
        "end":            datetime | None,   # UTC
        "class_filter":   str | None,        # YOLO class name
        "zone":           str | None,
        "semantic_query": str,               # cleaned query for CLIP
      }

    Falls back to {"has_time": False, ...} on any error so the caller can
    always fall back to the existing semantic search path.

    Examples
    --------
    "show me all people entered 2 days ago"
      → has_time=True, start=<2 days ago 00:00>, end=<2 days ago 23:59>, class_filter="person"

    "show me the truck entered on 13th April"
      → has_time=True, start=2026-04-13T00:00Z, end=2026-04-13T23:59Z, class_filter="truck"

    "blue car near main gate"
      → has_time=False, class_filter="car", zone="main_gate", semantic_query="blue car main gate"
    """
    _fallback = {
        "has_time":            False,
        "start":               None,
        "end":                 None,
        "class_filter":        None,
        "zone":                None,
        "semantic_query":      query,
        "has_telemetry":       False,
        "battery_below":       None,
        "altitude_above":      None,
        "altitude_below":      None,
        "flight_mode_filter":  None,
    }

    try:
        import json as _json
        from langchain_aws import ChatBedrock
        from langchain_core.messages import HumanMessage

        model_id = os.getenv("VLM_CAPTIONER_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        region   = aws_region or os.getenv("AWS_REGION", "us-east-1")

        llm = ChatBedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs={"max_tokens": 200},
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        prompt = f"""Today's date (UTC): {today}

Parse this surveillance query and return a JSON object ONLY — no extra text.

Query: "{query}"

Rules:
- "has_time": true if the query references a specific date, day, or time range.
- "start" / "end": ISO-8601 UTC strings covering the requested period.
  - "2 days ago" → that full calendar day (00:00:00 to 23:59:59 UTC).
  - "yesterday"  → previous full calendar day.
  - "13th April" → 2026-04-13 full day (assume current year if unambiguous).
  - "last hour"  → now minus 1 hour to now.
  - No time mentioned → null.
- "class_filter": YOLO class name if a specific object type is requested.
  Map: people/person/man/woman/pedestrian → "person";
       truck/lorry/van → "truck"; car/vehicle/automobile → "car";
       motorcycle/motorbike/bike → "motorcycle"; bus → "bus".
  null if not specified.
- "zone": exact zone name if mentioned, else null.
- "semantic_query": a clean keyword phrase for CLIP visual search (remove date/time/telemetry words).
- "has_telemetry": true if the query asks about drone telemetry (battery, altitude, speed, flight mode, signal).
- "battery_below": numeric threshold if query asks about low battery (e.g. "battery below 20" → 20), else null.
- "altitude_above": numeric threshold if query asks about high altitude (e.g. "above 50 meters" → 50), else null.
- "altitude_below": numeric threshold if query asks about low altitude, else null.
- "flight_mode_filter": flight mode string if mentioned — one of TAKEOFF, PATROL, HOVER, RETURN, LANDING — else null.

Return JSON only:
{{"has_time": true/false, "start": "...|null", "end": "...|null", "class_filter": "...|null", "zone": "...|null", "semantic_query": "...", "has_telemetry": true/false, "battery_below": null/number, "altitude_above": null/number, "altitude_below": null/number, "flight_mode_filter": "...|null"}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = _json.loads(raw)

        def _parse_dt(s):
            if not s or s == "null":
                return None
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None

        def _parse_num(val):
            try:
                return float(val) if val is not None and str(val) != "null" else None
            except (TypeError, ValueError):
                return None

        return {
            "has_time":           bool(parsed.get("has_time", False)),
            "start":              _parse_dt(parsed.get("start")),
            "end":                _parse_dt(parsed.get("end")),
            "class_filter":       parsed.get("class_filter") or None,
            "zone":               parsed.get("zone") or None,
            "semantic_query":     parsed.get("semantic_query") or query,
            "has_telemetry":      bool(parsed.get("has_telemetry", False)),
            "battery_below":      _parse_num(parsed.get("battery_below")),
            "altitude_above":     _parse_num(parsed.get("altitude_above")),
            "altitude_below":     _parse_num(parsed.get("altitude_below")),
            "flight_mode_filter": parsed.get("flight_mode_filter") or None,
        }

    except Exception as exc:
        logger.warning("[QueryParser] Failed to parse intent: %s", exc)
        return _fallback


def _build_smart_context(
    retriever,
    intent: dict,
    original_query: str,
    top_k: int = 100,
) -> tuple[list, str]:
    """
    Route to the right retrieval strategy based on parsed intent.

    Returns (results, context_string) where results is a list of FrameResult.

    Routing:
      - has_time=True  → temporal_search (SQL-first: every match, ordered by time)
      - has_time=False → semantic search (CLIP-first: ranked by visual similarity)
      In both cases class_filter and zone are forwarded.
    """
    from src.memory.hybrid_retriever import HybridRetriever  # already imported above

    if intent["has_time"]:
        results = retriever.temporal_search(
            start        = intent["start"],
            end          = intent["end"],
            class_filter = intent["class_filter"],
            zone         = intent["zone"],
            top_k        = top_k,
        )
        retrieval_note = (
            f"SQL time-window search"
            + (f" · class={intent['class_filter']}" if intent["class_filter"] else "")
            + (f" · {intent['start'].strftime('%Y-%m-%d') if intent['start'] else '?'}"
               f" → {intent['end'].strftime('%Y-%m-%d') if intent['end'] else '?'}")
        )
    else:
        results = retriever.search(
            intent["semantic_query"],
            top_k        = top_k,
            class_filter = intent["class_filter"],
            zone         = intent["zone"],
        )
        retrieval_note = "CLIP semantic search"

    if not results:
        # Fallback: try broader semantic search with original query
        results = retriever.search(original_query, top_k=top_k)
        retrieval_note += " (broadened)"

    # Build context string for LLM
    if not results:
        context = "No relevant footage found for this query."
    else:
        lines = [
            f"Retrieval method: {retrieval_note}",
            f"Found {len(results)} matching frames:\n",
        ]
        for i, r in enumerate(results[:50], 1):   # cap at 50 in prompt
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
                f"{i}. [{r.ts}] zone={r.zone or 'unknown'}\n"
                f"   Objects: {', '.join(r.class_names) or 'none'}\n"
                f"   Caption: {r.caption or '(no caption)'}"
                f"{telem_str}\n"
            )

        # Append any security events in the same time window
        if intent["has_time"] and (intent["start"] or intent["end"]):
            events = retriever.events_summary(
                start=intent["start"],
                end=intent["end"],
                zone=intent["zone"],
            )
            if events:
                lines.append(f"\nSecurity events in this window ({len(events)}):")
                for ev in events[:10]:
                    lines.append(
                        f"  [{ev['start_ts']}] {ev['severity'].upper()} — "
                        f"{ev['type']}: {ev['description']}"
                    )

        context = "\n".join(lines)

    return results, context


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_live, tab_timeline, tab_ask = st.tabs([
    "🔴  LIVE FEED",
    "📋  TIMELINE",
    "💬  ASK",
])


# ===========================================================================
# TAB 1 — LIVE
# ===========================================================================

with tab_live:
    col_feed, col_alerts = st.columns([3, 2], gap="medium")

    with col_feed:
        st.markdown("##### LIVE FEED")
        frame_placeholder   = st.empty()
        caption_placeholder = st.empty()
        progress_bar        = st.empty()

    with col_alerts:
        st.markdown("##### ALERT FEED")
        metrics_row         = st.empty()
        telem_placeholder   = st.empty()
        alerts_placeholder  = st.empty()

    run_col, stop_col, _ = st.columns([1, 1, 4])
    with run_col:
        start_btn = st.button("▶  START", use_container_width=True)
    with stop_col:
        stop_btn  = st.button("■  STOP",  use_container_width=True)

    if "running"      not in st.session_state: st.session_state.running      = False
    if "alerts_list"  not in st.session_state: st.session_state.alerts_list  = []
    if "frame_count"  not in st.session_state: st.session_state.frame_count  = 0
    if "detect_count" not in st.session_state: st.session_state.detect_count = 0
    if "alert_count"  not in st.session_state: st.session_state.alert_count  = 0
    if "processor"    not in st.session_state: st.session_state.processor    = None

    if stop_btn:
        st.session_state.running = False
        if st.session_state.processor is not None:
            st.session_state.processor.stop()
            st.session_state.processor = None

    if start_btn:
        # Validate source
        if source_type == "📁 Video File":
            if not video_path or not Path(video_path).exists():
                st.error(f"File not found: `{video_path}`")
                st.stop()
            source_label = Path(video_path).name
        else:
            if not rtsp_url or not rtsp_url.startswith("rtsp://"):
                st.error("Enter a valid RTSP URL starting with `rtsp://`")
                st.stop()
            source_label = rtsp_url

        st.session_state.running      = True
        st.session_state.alerts_list  = []
        st.session_state.frame_count  = 0
        st.session_state.detect_count = 0
        st.session_state.alert_count  = 0

        pipeline     = init_pipeline()
        agent        = pipeline["agent"]
        preprocessor = FramePreprocessor(yolo_size=640, vlm_every=vlm_every)

        # Build telemetry simulator (if a scenario is selected)
        _telem_sim: TelemetrySimulator | None = None
        if telem_scenario != "None (no telemetry)":
            _telem_sim = TelemetrySimulator(start_ts=datetime.now(timezone.utc))
            _telem_sim.load_scenario(telem_scenario)
            _telem_sim.start_battery = telem_start_bat

        # Build ingestor
        if source_type == "📁 Video File":
            ingestor = VideoIngestor.from_file(
                path=video_path,
                sample_every=sample_every,
                max_frames=max_frames,
                start_ts=datetime.now(timezone.utc),
                save_frames=True,
                save_size=_save_size,
                save_quality=_save_quality,
                telemetry_simulator=_telem_sim,
            )
        else:
            # For RTSP: compute a recommended sample_every based on typical FPS
            # We can't know the stream FPS without connecting, so use the slider value.
            # StreamProcessor's bounded queue will drop excess frames automatically.
            ingestor = VideoIngestor.from_rtsp(
                url=rtsp_url,
                sample_every=sample_every,
                max_frames=max_frames,
                save_frames=True,
                save_size=_save_size,
                save_quality=_save_quality,
                telemetry_simulator=_telem_sim,
            )

        col_feed.caption(f"{'📁' if source_type == '📁 Video File' else '📡'} Source: `{source_label}`")

        BBOX_COLORS = {
            "person": (0,230,0), "car": (255,140,0), "truck": (0,140,255),
            "motorcycle": (240,0,240), "bicycle": (0,240,240),
            "bus": (240,240,0), "boat": (140,240,140),
        }

        # ── StreamProcessor: decoupled reader + worker threads ──────────
        is_rtsp = (source_type != "📁 Video File")
        if is_rtsp:
            # RTSP/live: small queue, drop oldest frames to stay real-time.
            # No frame cap — runs until STOP is pressed.
            processor = StreamProcessor(
                agent, preprocessor,
                queue_size=4, result_size=200, drop_on_full=True,
            )
        else:
            # File: blocking reader (drop_on_full=False) guarantees every
            # sampled frame reaches the worker even if LLM judge is slow.
            # Fixed queue of 32 frames — enough to pipeline reader+worker
            # without buffering the entire video in memory.
            processor = StreamProcessor(
                agent, preprocessor,
                queue_size=32, result_size=200, drop_on_full=False,
            )
        processor.start(ingestor, zone="")
        st.session_state.processor = processor

        last_caption = ""
        start_time   = time.time()
        frames_shown = 0

        while st.session_state.running:
            result = processor.get_result(timeout=0.1)

            # Stop once the processor has finished (source exhausted or stopped)
            if result is None:
                if not processor.running:
                    break
                continue

            detections   = result.get("detections", [])
            caption      = result.get("caption")
            alerts       = result.get("alerts_fired", [])
            preprocessed = result.get("preprocessed")   # PreprocessedFrame

            if caption:
                last_caption = caption

            st.session_state.frame_count  += 1
            st.session_state.detect_count += len(detections)
            st.session_state.alert_count  += len(alerts)
            frames_shown += 1

            # ── Annotate and display frame ───────────────────────────────
            frame_img = preprocessed.packet.image if preprocessed else None
            if frame_img is not None:
                img = frame_img.copy()
                for det in detections:
                    x1,y1,x2,y2 = det.bbox
                    color = BBOX_COLORS.get(det.class_name, (200,200,200))
                    cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
                    tid   = f"#{det.track_id} " if det.track_id else ""
                    label = f"{tid}{det.class_name} {det.confidence:.2f}"
                    (tw,th),bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img,(x1,y1-th-bl-4),(x1+tw+4,y1),color,-1)
                    cv2.putText(img,label,(x1+2,y1-bl-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
                frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

            # ── Caption strip ────────────────────────────────────────────
            if last_caption:
                caption_placeholder.markdown(
                    f'<div class="caption-box">🔍 {last_caption}</div>',
                    unsafe_allow_html=True,
                )

            # ── Progress bar ─────────────────────────────────────────────
            if max_frames:
                pct = int((st.session_state.frame_count / max_frames) * 100)
                progress_bar.progress(
                    min(pct, 100),
                    text=f"Frame {st.session_state.frame_count} / {max_frames}",
                )
            else:
                # Unlimited — pulse the bar and show a live counter
                progress_bar.progress(
                    min((st.session_state.frame_count % 100), 100),
                    text=f"▶ Frame {st.session_state.frame_count} processed  (press STOP to end)",
                )

            # ── Metrics + throughput ─────────────────────────────────────
            p_stats  = processor.stats
            elapsed  = time.time() - start_time
            fps_live = p_stats["processed"] / elapsed if elapsed > 0 else 0.0
            metrics_row.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:8px">
              <div style="flex:1;background:#0f1318;border:1px solid #1e2530;padding:8px;text-align:center;border-radius:2px">
                <div style="font-size:1.4rem;font-family:'Share Tech Mono',monospace;color:#00e5ff">{st.session_state.frame_count}</div>
                <div style="font-size:0.7rem;color:#5a6a7a">FRAMES</div>
              </div>
              <div style="flex:1;background:#0f1318;border:1px solid #1e2530;padding:8px;text-align:center;border-radius:2px">
                <div style="font-size:1.4rem;font-family:'Share Tech Mono',monospace;color:#ff9500">{st.session_state.detect_count}</div>
                <div style="font-size:0.7rem;color:#5a6a7a">DETECTIONS</div>
              </div>
              <div style="flex:1;background:#0f1318;border:1px solid #1e2530;padding:8px;text-align:center;border-radius:2px">
                <div style="font-size:1.4rem;font-family:'Share Tech Mono',monospace;color:#ff3b3b">{st.session_state.alert_count}</div>
                <div style="font-size:0.7rem;color:#5a6a7a">ALERTS</div>
              </div>
              <div style="flex:1;background:#0f1318;border:1px solid #1e2530;padding:8px;text-align:center;border-radius:2px">
                <div style="font-size:1.1rem;font-family:'Share Tech Mono',monospace;color:#44cc88">{fps_live:.1f} fps</div>
                <div style="font-size:0.65rem;color:#5a6a7a">PROC | {p_stats['drop_pct']}% DROP</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Telemetry strip ──────────────────────────────────────────
            telemetry = result.get("telemetry")
            if telemetry:
                bat   = telemetry.get("battery_pct", "?")
                alt   = telemetry.get("alt_m", "?")
                spd   = telemetry.get("speed_ms", "?")
                mode  = telemetry.get("flight_mode", "?")
                sig   = telemetry.get("signal_strength", "?")
                lat   = telemetry.get("lat", "?")
                lon   = telemetry.get("lon", "?")
                bat_color  = "#ff3b3b" if isinstance(bat, (int, float)) and bat < 20 else (
                             "#ff9500" if isinstance(bat, (int, float)) and bat < 40 else "#44cc88")
                sig_color  = "#ff3b3b" if isinstance(sig, (int, float)) and sig < 50 else "#44cc88"
                telem_placeholder.markdown(f"""
                <div style="background:#0a0f15;border:1px solid #1e2530;border-radius:3px;padding:8px 10px;margin-bottom:6px;font-family:'Share Tech Mono',monospace;font-size:0.75rem">
                  <div style="color:#5a6a7a;font-size:0.65rem;margin-bottom:4px">TELEMETRY · {mode}</div>
                  <div style="display:flex;gap:12px;flex-wrap:wrap">
                    <span>🔋 <span style="color:{bat_color}">{bat}%</span></span>
                    <span>↕ <span style="color:#00e5ff">{alt}m</span></span>
                    <span>💨 <span style="color:#c8d0d8">{spd}m/s</span></span>
                    <span>📡 <span style="color:{sig_color}">{sig}%</span></span>
                    <span style="color:#3a4a5a">GPS {lat},{lon}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Alerts feed (cap at 100 to avoid unbounded memory) ───────
            for a in alerts:
                st.session_state.alerts_list.insert(0, a)
            if len(st.session_state.alerts_list) > 100:
                st.session_state.alerts_list = st.session_state.alerts_list[:100]

            if st.session_state.alerts_list:
                html = ""
                for a in st.session_state.alerts_list[:15]:
                    sev   = a.get("severity","low")
                    ts_s  = a.get("ts","")[:19]
                    msg   = a.get("message","")
                    rule  = a.get("rule_name","")
                    html += f'<div class="alert-{sev}"><b>[{sev.upper()}]</b> {ts_s}<br>{msg}<span style="color:#3a4a5a;font-size:0.75rem;margin-left:8px">#{rule}</span></div>'
                alerts_placeholder.markdown(html, unsafe_allow_html=True)

        processor.stop()
        st.session_state.processor = None
        st.session_state.running   = False
        done_text = f"Stopped — {st.session_state.frame_count} frames processed"
        progress_bar.progress(100 if max_frames else 0, text=done_text)


# ===========================================================================
# TAB 2 — TIMELINE
# ===========================================================================

with tab_timeline:
    pipeline = init_pipeline()
    db       = pipeline["db"]

    st.markdown("##### EVENT TIMELINE")

    # Filters
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
    with fc1:
        evt_type = st.selectbox(
            "Event type",
            [
                "All",
                # person
                "after_hours_person", "after_hours_zone_entry",
                "loitering", "crowd_gathering", "tailgating",
                # vehicle
                "after_hours_vehicle", "vehicle_repeat_entry",
                "stopped_vehicle", "unattended_vehicle", "convoy_detection",
                # zone
                "forbidden_zone", "perimeter_breach",
                # caption-based behavioural
                "gate_tampering_attempt", "coordinated_breach_attempt",
                "lookout_behavior", "tool_use_near_infrastructure",
                "suspicious_crouching", "perimeter_climbing_attempt",
                "forced_entry_attempt", "suspicious_group_activity",
                # activity
                "high_object_density",
                # telemetry
                "battery_critical", "low_battery",
                "altitude_anomaly_high", "altitude_anomaly_low",
            ],
        )
    with fc2:
        severity = st.selectbox("Severity", ["All", "high", "medium", "low"])
    with fc3:
        zone_filter = st.text_input("Zone (leave blank for all)", "")
    with fc4:
        hours_back = st.number_input("Hours back", min_value=1, max_value=168, value=24)

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours_back)

    events = db.get_events(
        event_type = None if evt_type == "All" else evt_type,
        severity   = None if severity == "All" else severity,
        start      = start_dt,
        end        = end_dt,
        zone       = zone_filter or None,
    )

    # Stats row
    sc1, sc2, sc3, sc4 = st.columns(4)
    stats = db.get_stats()
    sc1.metric("Total Frames",    stats["frames"])
    sc2.metric("Total Objects",   stats["objects"])
    sc3.metric("Total Events",    stats["events"])
    sc4.metric("Unacked Alerts",  stats["unacked_alerts"])

    st.markdown("---")

    if not events:
        st.info("No events found for the selected filters.")
    else:
        for ev in events:
            sev         = ev.get("severity", "low")
            description = ev.get("description", "")
            is_review   = description.startswith("[LLM: REVIEW")
            color       = {"high": "#ff3b3b", "medium": "#ff9500", "low": "#00aaff"}.get(sev, "#5a6a7a")
            review_tag  = " ⚠ REVIEW" if is_review else ""
            with st.expander(
                f"[{sev.upper()}]{review_tag}  {ev['type']}  —  {ev['start_ts'][:19]}  |  {ev.get('zone','') or 'unknown zone'}",
                expanded=False,
            ):
                if is_review:
                    st.warning("LLM flagged this event as a possible false positive. Human review recommended.")
                st.markdown(f"**Description:** {description}")
                st.markdown(f"**Zone:** `{ev.get('zone') or '—'}`")
                st.markdown(f"**Severity:** :{color}[{sev}]")
                frame_ids = json.loads(ev.get("frame_ids", "[]"))
                st.markdown(f"**Frame IDs:** {', '.join(frame_ids[:3])}{'…' if len(frame_ids) > 3 else ''}")

                # ── Source frame images ──────────────────────────────────
                if frame_ids:
                    frames_meta = db.get_frames_by_ids(frame_ids[:4])
                    frames_with_image = [
                        f for f in frames_meta
                        if f.get("frame_path") and Path(f["frame_path"]).exists()
                    ]
                    if frames_with_image:
                        st.markdown("**Evidence frames:**")
                        img_cols = st.columns(min(len(frames_with_image), 4))
                        for col, fr in zip(img_cols, frames_with_image):
                            img_bgr = cv2.imread(fr["frame_path"])
                            if img_bgr is not None:
                                with col:
                                    st.image(
                                        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                        use_container_width=True,
                                        caption=f"{(fr.get('ts') or '')[:19]}",
                                    )
                                    cap_text = fr.get("caption") or ""
                                    if cap_text:
                                        st.caption(cap_text[:120])

                # Show alerts for this event
                alerts = db.get_alerts_for_event(ev["id"])
                if alerts:
                    st.markdown("**Alerts fired:**")
                    for al in alerts:
                        acked = "✅" if al["acked"] else "🔴"
                        st.markdown(f"  {acked} `{al['ts'][:19]}` — {al['message']}")
                        if not al["acked"]:
                            if st.button(f"Ack {al['id'][:8]}", key=f"ack_{al['id']}"):
                                db.ack_alert(al["id"])
                                st.rerun()


# ===========================================================================
# TAB 3 — ASK (RAG chat)
# ===========================================================================

with tab_ask:
    pipeline  = init_pipeline()
    retriever = pipeline["retriever"]
    embedder  = pipeline["embedder"]

    st.markdown("##### ASK THE SECURITY ANALYST")
    st.caption("Query the indexed footage using natural language. Follow-up questions are supported.")

    if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
    if "pending_query"  not in st.session_state: st.session_state.pending_query  = None

    # ── Clear chat button ────────────────────────────────────────────────
    if st.button("🗑 Clear chat", key="clear_chat"):
        st.session_state.chat_history  = []
        st.session_state.pending_query = None
        st.rerun()

    # ── Render chat history ──────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🛸 {msg["content"]}</div>', unsafe_allow_html=True)

            # Source frames expander
            if msg.get("sources"):
                with st.expander("Source frames", expanded=False):
                    for src in msg["sources"][:5]:
                        meta       = src["metadata"]
                        ts         = (meta.get("ts") or "")[:19]
                        zn         = meta.get("zone") or "unknown"
                        cap        = meta.get("caption") or ""
                        sc         = src.get("score", 0)
                        frame_path = meta.get("frame_path")

                        img_col, info_col = st.columns([1, 2], gap="small")

                        with img_col:
                            if frame_path and Path(frame_path).exists():
                                img_bgr = cv2.imread(frame_path)
                                if img_bgr is not None:
                                    st.image(
                                        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                        use_container_width=True,
                                    )
                                else:
                                    st.markdown("*(image unreadable)*")
                            else:
                                st.markdown("*(frame not on disk)*")

                        with info_col:
                            telem_meta = meta.get("telemetry")
                            telem_line = ""
                            if telem_meta:
                                telem_line = (
                                    f"<br><span style='color:#5a6a7a;font-size:0.75rem'>"
                                    f"🔋{telem_meta.get('battery_pct','?')}% "
                                    f"↕{telem_meta.get('alt_m','?')}m "
                                    f"💨{telem_meta.get('speed_ms','?')}m/s "
                                    f"{telem_meta.get('flight_mode','?')}</span>"
                                )
                            st.markdown(
                                f'<div class="source-frame">'
                                f'📍 {ts}<br>'
                                f'zone={zn}&nbsp;&nbsp;score={sc:.3f}<br>'
                                f'{cap}'
                                f'{telem_line}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("<hr style='border-color:#1e2530;margin:6px 0'>", unsafe_allow_html=True)

            # Suggested follow-up buttons (only on the last assistant message)
            is_last = (i == len(st.session_state.chat_history) - 1)
            suggestions = msg.get("suggestions", [])
            if is_last and suggestions:
                st.markdown(
                    '<div style="font-size:0.75rem;color:#5a6a7a;margin-top:8px;margin-bottom:4px">'
                    '💡 Suggested follow-ups:</div>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(len(suggestions))
                for j, suggestion in enumerate(suggestions):
                    with cols[j]:
                        if st.button(suggestion, key=f"followup_{i}_{j}", use_container_width=True):
                            st.session_state.pending_query = suggestion
                            st.rerun()

    # ── Input — typed query or injected follow-up ────────────────────────
    typed_query = st.chat_input("Ask anything about the footage…")

    # Follow-up button takes priority if no new typed query
    query = typed_query
    if query is None and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

    if query:
        aws_region = os.environ.get("AWS_REGION", "us-east-1")
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("Searching footage…"):
            history = st.session_state.chat_history

            # 1. Reformulate follow-up questions as standalone queries
            #    ("what about the vehicles?" → "vehicles detected in footage")
            standalone = _reformulate_query(query, history, aws_region)

            # 2. Parse structured intent — extracts time windows, class filters, zones
            #    so "show people 2 days ago" becomes {start, end, class_filter="person"}
            intent = _parse_query_intent(standalone, aws_region)

            # Show the resolved time window (helps user understand what was searched)
            if intent["has_time"] and intent["start"]:
                _tw = (
                    f"{intent['start'].strftime('%Y-%m-%d %H:%M')} → "
                    f"{intent['end'].strftime('%Y-%m-%d %H:%M') if intent['end'] else 'now'}"
                )
                st.info(f"🕐 Time window detected: **{_tw}**"
                        + (f" · class: `{intent['class_filter']}`" if intent["class_filter"] else ""))

            # Show detected telemetry filters
            if intent.get("has_telemetry"):
                _tfilters = []
                if intent.get("battery_below") is not None:
                    _tfilters.append(f"battery < {intent['battery_below']}%")
                if intent.get("altitude_above") is not None:
                    _tfilters.append(f"altitude > {intent['altitude_above']}m")
                if intent.get("altitude_below") is not None:
                    _tfilters.append(f"altitude < {intent['altitude_below']}m")
                if intent.get("flight_mode_filter"):
                    _tfilters.append(f"mode={intent['flight_mode_filter']}")
                if _tfilters:
                    st.info(f"📡 Telemetry filter detected: **{' · '.join(_tfilters)}**")

            # 3. Route to the correct retrieval strategy:
            #    - has_time=True  → temporal_search (SQL-first, every match in window)
            #    - has_time=False → semantic search  (CLIP-first, ranked by similarity)
            results, context = _build_smart_context(retriever, intent, standalone, top_k=100)

            # 4. Multi-turn LLM answer with full conversation history
            answer = _bedrock_rag_answer(query, context, aws_region, chat_history=history)

            # 5. Source frames for display
            sources = [
                {"metadata": r.to_dict(), "score": r.semantic_score}
                for r in results
            ]

            # 6. Suggested follow-ups
            suggestions = _suggest_followups(query, answer, aws_region)

        st.session_state.chat_history.append({
            "role":        "assistant",
            "content":     answer,
            "sources":     sources,
            "suggestions": suggestions,
        })
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar stats (live refresh)
# ---------------------------------------------------------------------------

pipeline = init_pipeline()
stats    = pipeline["db"].get_stats()
stats_placeholder.markdown(f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#5a6a7a">
DB STATUS<br>
────────────────<br>
frames   : {stats['frames']}<br>
objects  : {stats['objects']}<br>
events   : {stats['events']}<br>
alerts   : {stats['alerts']}<br>
unacked  : {stats['unacked_alerts']}<br>
chroma   : {pipeline['chroma'].count()} indexed<br>
────────────────
</div>
""", unsafe_allow_html=True)
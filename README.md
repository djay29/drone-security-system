# Drone Security System

An AI-powered drone surveillance system that performs real-time multi-modal threat detection, alert escalation via LLM reasoning, and natural-language search over historical footage.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Setup & Installation](#setup--installation)
4. [Running the System](#running-the-system)
5. [Configuration](#configuration)
6. [Design Decisions](#design-decisions)
7. [AI Tools Integrated](#ai-tools-integrated)
8. [Project Structure](#project-structure)

---

## System Overview

The system ingests live drone video (RTSP) or recorded footage, runs a multi-model perception pipeline on every frame, applies a declarative YAML rule engine to detect security events, and escalates ambiguous cases to a cloud LLM for final verdict. All detections, captions, embeddings, and alerts are persisted and made searchable through a Streamlit dashboard.

**Core capabilities:**

| Capability | Technology |
|---|---|
| Object detection + tracking | YOLOv8s + ByteTrack |
| Scene understanding | Claude Haiku via Amazon Bedrock (primary) / Moondream2 (fallback) |
| Semantic frame search | OpenCLIP ViT-B/32 embeddings + ChromaDB |
| Rule-based threat detection | Declarative YAML rules engine |
| LLM escalation / false-positive filtering | Claude Haiku via LangChain + LangGraph |
| Structured event storage | SQLite (WAL mode) |
| Real-time pipeline orchestration | Multi-threaded StreamProcessor |
| Dashboard + RAG chat | Streamlit |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           VIDEO INPUT LAYER                                  │
│  VideoIngestor  ──  from_file() / from_rtsp() / from_fallback()             │
│  FramePacket: {frame_id, ts, video_id, frame_index, image}                  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STREAM PROCESSOR                                     │
│  Reader thread → bounded queue → Worker thread → result queue               │
│  drop_on_full=True (RTSP: evict oldest)  /  False (file: block reader)      │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH AGENT PIPELINE                                │
│                                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐                          │
│  │ perceive │──▶│contextualize │──▶│ rule_check │                          │
│  │          │   │              │   │            │                          │
│  │ YOLO det │   │ track dur.   │   │ YAML rules │                          │
│  │ CLIP emb │   │ recent evts  │   │ → hits[]   │                          │
│  │ VLM cap  │   │ context str  │   │ needs_llm? │                          │
│  └──────────┘   └──────────────┘   └─────┬──────┘                          │
│                                          │                                  │
│                        ┌─────────────────┴─────────────────┐               │
│                        │                                   │               │
│                  needs_llm=True                     needs_llm=False         │
│                        │                              rule_hits?            │
│                        ▼                                   │               │
│                 ┌────────────┐                    YES ──── ▼               │
│                 │ llm_judge  │                    ┌──────────────┐          │
│                 │ Claude H.  │──────────────────▶│    alert     │          │
│                 │ verdict +  │                    │ dedup/cooldown│          │
│                 │ alert msg  │                    │ SQLite store  │          │
│                 └────────────┘                    └──────┬───────┘          │
│                                                         │                  │
│                                              NO ────────┘                  │
│                                                         ▼                  │
│                                                  ┌────────────┐            │
│                                                  │    log     │            │
│                                                  │ SQLite frm │            │
│                                                  │ ChromaDB   │            │
│                                                  └────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
┌─────────────────────┐               ┌──────────────────────┐
│   SQLiteStore       │               │    ChromaStore       │
│  frames, objects    │               │  512-dim CLIP vecs   │
│  events, alerts     │               │  + frame metadata    │
└────────┬────────────┘               └──────────┬───────────┘
         │                                        │
         └──────────────────┬─────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │    HybridRetriever      │
              │  temporal_search()      │
              │  + semantic search()    │
              └────────────┬────────────┘
                           ▼
            ┌──────────────────────────────┐
            │      Streamlit Dashboard     │
            │  Live  |  Timeline  |  Ask   │
            └──────────────────────────────┘
```

### Data flow per frame

1. `VideoIngestor` emits a `FramePacket` (raw BGR image + metadata)
2. `FramePreprocessor` normalises it into three views: YOLO (640×640 letterboxed), VLM (RGB 378×378), CLIP (224×224 float32)
3. `perceive` node runs YOLO → detections; CLIP → 512-dim embedding; VLM caption (async thread pool or sync)
4. `contextualize` node builds per-track loitering durations + pulls recent events from SQLite
5. `rule_check` node evaluates all YAML rules → list of `rule_hits`
6. Conditional routing: if any hit has `needs_llm=True` → `llm_judge`; else if hits exist → `alert`; else → `log`
7. `llm_judge` calls Claude Haiku with context + caption + rule hits → `genuine | false_positive` verdict
8. `alert` node deduplicates (5-minute cooldown per rule+zone), stores event + alert in SQLite
9. `log` node persists frame, detections, and embedding regardless of alert outcome

---

## Setup & Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | >= 3.10 |
| uv (package manager) | latest |
| AWS account | Bedrock access (us-east-1) |
| GPU (optional) | CUDA-capable for faster YOLO/CLIP |

### 1. Clone the repository

```bash
git clone <repo-url>
cd drone-security-system
```

### 2. Install dependencies

```bash
# Install uv if not already installed
pip install uv

# Create virtual environment and install all dependencies
uv sync
```

### 3. Configure environment variables

```bash
cp .env
```

Edit `.env` with your credentials:

```dotenv
# AWS Bedrock (required for LLM captioning + LLM judge)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION_NAME=us-east-1

# VLM model (Claude Haiku via Bedrock)
VLM_CAPTIONER_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0

# Storage
SQLITE_DB_PATH=data/security_memory.db

# Optional: Hugging Face token for Moondream2 local fallback
HF_TOKEN=your_hf_token

# Dashboard
DASHBOARD_PORT=8501
LOG_LEVEL=INFO
```

### 4. Verify installation

```bash
uv run python -c "from src.agent.graph import build_agent; print('OK')"
```

---

## Running the System

### Streamlit dashboard (recommended)

```bash
uv run streamlit run src/dashboard/streamlit_app.py
```

Open http://localhost:8501 in your browser.

**Live tab workflow:**
1. Select source: `Video File` (upload MP4/AVI, or use the bundled test clips in `data/videos/`) or `RTSP Stream` (enter URL)
2. Choose security zone from the sidebar
3. Click **START** — frames process in real time
4. Alerts appear in the feed as they fire; bounding boxes overlay the frame thumbnail
5. Click **STOP** when done

> **Test videos:** The `data/videos/` directory contains pre-packaged sample clips covering common security scenarios (after-hours pedestrian, loitering, gate approach). These can be used immediately without any external video source — simply select `Video File` in the dashboard and pick a clip from that folder.

**Timeline tab:**
- Filter events by date range, severity, zone, or event type
- Each event row shows timestamp, rule triggered, severity badge, and frame thumbnail

**Ask tab:**
- Type natural language queries: *"show me all people near the gate after 10pm"*
- The system parses temporal / class intent, queries SQLite + ChromaDB, and generates a RAG answer
- Follow-up questions are supported (multi-turn conversation history)

---

## Configuration

### Security rules (`configs/rules.yaml`)

Rules are hot-reloadable (call `engine.reload()`). No code change required to add new threat patterns.

```yaml
rules:
  - name: after_hours_person
    condition:
      object_class: person
      time_range: { start: "22:00", end: "06:00" }
    severity: high
    message_template: "Person detected at {zone} at {time}"
    needs_llm: false

  - name: loitering
    condition:
      same_track_id_in_zone_seconds: 60
    severity: medium
    message_template: "Loitering: track #{track_id} in {zone} for {duration}s"
    needs_llm: true

  - name: gate_tampering_attempt
    condition:
      caption_keywords: [tools, gate, lock, wrench, crowbar]
      caption_min_match: 2
    severity: high
    message_template: "Gate tampering at {zone} — keywords: {matched_keywords}"
    needs_llm: true
```

**Supported condition types:**

| Key | Description |
|---|---|
| `object_class` | Single YOLO class must be present |
| `object_class_in` | Any of these YOLO classes must be present |
| `time_range` | Fires only within a HH:MM window (handles midnight wrap) |
| `zone_in` | Fires only when current zone matches |
| `min_object_count` | Requires N or more detections of the specified class |
| `same_track_id_in_zone_seconds` | Track must persist in zone for >= N seconds |
| `caption_keywords` + `caption_min_match` | VLM caption must contain >= M keywords |
| `battery_below` | Drone telemetry battery threshold |

### Storage presets (dashboard sidebar)

| Preset | Resolution | JPEG quality | Approx size |
|---|---|---|---|
| Balanced | 640x360 | 75 | ~35 KB/frame |
| High | 854x480 | 82 | ~65 KB/frame |
| Compact | 426x240 | 65 | ~15 KB/frame |
| Minimal | 320x180 | 55 | ~8 KB/frame |
| Native | source res | 90 | varies |

Model inputs (YOLO / CLIP / VLM) always use the original full-resolution image. Only the saved JPEG thumbnail is downscaled.

---

## Design Decisions

### 1. LangGraph for pipeline orchestration

A sequential function pipeline would be simpler, but LangGraph was chosen because:

- **Conditional routing is first-class** — the `needs_llm` flag forks the pipeline without `if/else` scattered across unrelated functions
- **State is explicit and typed** — `AgentState` TypedDict documents exactly what each node reads and writes; bugs surface as missing-key errors rather than silent None propagation
- **Nodes are individually testable** — each node function can be unit-tested in isolation from the graph
- **Extensibility** — adding a parallel CLIP anomaly branch or a human-in-the-loop pause requires no refactoring of existing nodes

### 2. Two-tier VLM strategy

VLM captioning is the most expensive step (~500–2000 ms per frame):

- **Amazon Nova 2 Lite via Bedrock** (primary): 25-word captions, sub-second latency, no local GPU requirement
- **Moondream2 local** (fallback): ~2 GB model, runs on CPU at ~1–2 fps, fully offline

The tiered approach avoids vendor lock-in and supports air-gapped deployments.

### 3. Async VLM via thread pool

YOLO + rules evaluation takes ~30–80 ms per frame. Synchronous VLM (~500 ms) would cap throughput at 2 fps. The async design submits caption jobs to a `ThreadPoolExecutor` and collects completed results on the next frame — the fast path (YOLO + rules) is never blocked.

The tradeoff: caption-based rules fire one frame later than detection-based rules. Acceptable for behavioral analysis which requires multi-second evidence anyway.

### 4. Declarative YAML rule engine

Rules are expressed in YAML rather than Python because:

- **Zero-deployment updates** — operators tune rules without a code change or redeploy
- **Hot reload** — `engine.reload()` swaps rules mid-run
- **Auditable** — non-developers (security managers) can read and verify trigger conditions
- **LLM escalation flag** — any rule can set `needs_llm: true` to gate alerts on Claude's contextual judgment

### 5. Hybrid retrieval (ChromaDB + SQLite)

Two query strategies serve different user intents:

- **Semantic search (ChromaDB)**: answers *"show me suspicious behaviour near the gate"* via CLIP embedding similarity
- **Temporal/structured search (SQLite)**: answers *"show me all trucks on 13 April between 2 pm and 4 pm"* via SQL

`HybridRetriever` uses a lightweight LLM call to parse the user's intent (`has_time`, `class_filter`, `zone`) and routes to the appropriate strategy.

### 6. Frame storage vs. model quality

Source frames can be 1080p+ at 30+ fps. The system separates concerns:

- **Model inputs** use the full-resolution source frame
- **Saved thumbnails** are compressed JPEG at a configurable preset (default 640×360 / q75 ≈ 35 KB)

Storage scales down without any reduction in detection accuracy.

### 7. File vs. RTSP streaming modes

`StreamProcessor` uses a `drop_on_full` flag:

- **RTSP (live)** `drop_on_full=True`: evict oldest queued frame when the worker falls behind. Keeps latency low at the cost of missing some frames.
- **File (recorded)** `drop_on_full=False`: reader blocks until the worker has queue space. Every frame is processed.

This prevents the regression where only 3–4 frames of a recorded file were processed because the reader filled and drained the queue before the worker could consume any frames.

---

## AI Tools Integrated

### Amazon Bedrock — Claude Haiku

**VLM captioning:** Each frame is sent with a security-focused prompt — *"In 25 words or fewer, describe what is happening from a security perspective."* The 25-word cap keeps costs low and forces the model to prioritise the most security-relevant observation.

**LLM judge:** When a rule fires with `needs_llm: true`, Claude receives context summary + caption + rule hits. It responds with a structured verdict:
```
VERDICT: genuine
ALERT: Person with tools at gate lock — possible forced entry.
REASON: Caption explicitly describes tool use at lock location.
```
This filters false positives from intentionally broad rules (e.g., loitering during a lunch break).

**Impact:** Eliminated the need for complex hand-tuned per-zone thresholds. A single `needs_llm: true` flag delegates ambiguous cases to the model.

### YOLOv8s + ByteTrack (Ultralytics)

The `s` (small) variant balances throughput (30+ fps on modest hardware) against detection quality. ByteTrack assigns persistent `track_id` values across frames — this is the foundation for all duration-based rules (loitering, stopped vehicles).

**Impact:** Single `model.track()` call delivers detection + tracking with no hand-written motion logic.

### OpenCLIP (ViT-B/32)

512-dimensional embeddings bridge the semantic gap between text queries and raw video. The same pre-trained weight space is used for both frame indexing and query encoding.

**Impact:** The Ask tab's semantic search required zero training data — CLIP generalises well to security camera scenes out of the box.

### LangGraph

Turned the processing pipeline from nested function calls into an inspectable state machine. The `needs_llm` conditional edge is a routing function; each node is a pure function with typed inputs and outputs.

**Impact:** Clean separation of concerns; each of the six nodes (perceive, contextualize, rule_check, llm_judge, alert, log) is independently unit-testable.

### ChromaDB

HNSW index provides sub-millisecond approximate nearest-neighbour search over millions of frame embeddings. Metadata fields (zone, ts, class_names) enable pre-filtering before vector similarity comparison.

### Moondream2 (Hugging Face Transformers)

2 GB local VLM fallback for offline or no-credentials environments. The `VLMCaptioner` tries Bedrock first and falls back to Moondream transparently, with no change to downstream code.

---

## Project Structure

```
drone-security-system/
├── src/
│   ├── agent/
│   │   ├── graph.py              # LangGraph pipeline builder
│   │   ├── nodes.py              # All 6 node implementations
│   │   ├── rule_engine.py        # YAML rule evaluation engine
│   │   └── state.py              # AgentState TypedDict
│   ├── alerts/
│   │   └── dispatcher.py         # Multi-channel alert dispatch
│   ├── dashboard/
│   │   └── streamlit_app.py      # Live / Timeline / Ask tabs
│   ├── memory/
│   │   ├── chroma_store.py       # CLIP vector store (ChromaDB)
│   │   ├── hybrid_retriever.py   # Unified semantic + structured search
│   │   └── sqlite_store.py       # Relational store (4 tables)
│   ├── perception/
│   │   ├── clip_embedder.py      # OpenCLIP ViT-B/32
│   │   ├── frame_preprocessor.py # Letterbox + normalise
│   │   ├── video_ingestor.py     # File / RTSP / fallback source
│   │   ├── vlm_captioner.py      # Bedrock (primary) + Moondream2 (fallback)
│   │   └── yolo_detector.py      # YOLOv8s + ByteTrack
│   ├── pipeline/
│   │   └── stream_processor.py   # Multi-threaded reader/worker queues
│   └── telemetry/
│       └── simulator.py          # Drone telemetry stub
├── configs/
│   ├── rules.yaml                # Security rules (hot-reloadable)
│   ├── settings.yaml             # App-wide configuration
│   └── zones.yaml                # Named zone polygons
├── data/
│   ├── videos/                   # Bundled test videos for local testing (MP4/AVI)
│   │                             # Covers scenarios: after-hours pedestrian,
│   │                             # loitering, gate approach, vehicle detection
│   └── security_memory.db        # SQLite database (auto-created on first run)
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── unit/                     # Per-module isolated tests
│   ├── integration/              # Full pipeline with stubbed models
│   └── scenarios/                # End-to-end security event simulations
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TESTING.md
│   └── AI_TOOLS.md
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```
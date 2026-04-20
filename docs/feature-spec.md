# Feature Specification — Drone Security Agent

---

## Part 1: Value Proposition for Property Owners

### Problem

Traditional security cameras record footage passively. A guard must watch every feed in real time, or footage is only reviewed after an incident has already occurred. For a property owner with multiple zones (main gate, parking, perimeter, server room), the cost of 24/7 human monitoring is high, and the risk of a missed event is constant.

### What the agent delivers

**The drone security agent enhances property security by replacing passive recording with continuous, automated analysis — detecting threats in real time, filtering out false alarms using AI judgment, and surfacing only the alerts that require human action.**

| Owner pain point | How the agent addresses it |
|---|---|
| Guards miss events on overnight shifts | After-hours rules fire immediately on the first detected person or vehicle, regardless of time |
| Too many false alerts cause alarm fatigue | LLM judge reviews ambiguous rule hits before they page anyone |
| Reviewing footage after an incident is slow | Natural language search: *"show me everyone near the gate after midnight"* returns relevant frames in seconds |
| Security requirements change frequently | YAML rules updated in seconds with no code change or redeploy |
| Storage costs for 24/7 footage are high | Configurable thumbnail compression (8–65 KB/frame) with no impact on detection accuracy |

---

### Key Requirements

#### KR-1 — Real-time threat detection with < 5-second alert latency

The system must detect a security event (person after hours, vehicle at restricted zone, gate tampering attempt) and fire an alert within 5 seconds of the triggering frame being captured. This window covers frame ingestion, YOLO detection, rule evaluation, optional LLM escalation, and alert dispatch.

**Acceptance criteria:**
- Alert appears in the dashboard feed within 5 seconds of the event frame timestamp
- `after_hours_person` and zone-based rules (no LLM escalation needed) must fire within 2 seconds
- Rules with `needs_llm: true` must fire within 5 seconds, including the Bedrock round-trip

#### KR-2 — False-positive rate below 10% on ambiguous rules

Rules that cover behavioural patterns (loitering, crowd gathering, gate tampering) are intentionally broad — they will match benign scenes (a delivery driver waiting, employees gathering for a break). The LLM judge must reduce the false-positive rate on these rules to below 10%, meaning at least 9 out of 10 genuinely benign scenes that trigger these rules are correctly classified as false positives and suppressed.

**Acceptance criteria:**
- Rules with `needs_llm: true` route all hits through the LLM judge before dispatching
- Correctly suppressed hits are tagged `[LLM: REVIEW]` and stored for audit, not discarded
- 5-minute cooldown prevents the same event from generating repeated alerts

#### KR-3 — Natural language retrieval of any historical event within 3 seconds

A property owner or security officer must be able to query the full recorded archive in plain English and receive relevant frames with context. The system must correctly handle both time-anchored queries (*"show me trucks on 13 April"*) and concept queries (*"show me anyone near the gate at night"*).

**Acceptance criteria:**
- Temporal queries (explicit date/time/class) use SQL-first retrieval — 100% recall within the specified window
- Semantic queries use CLIP vector search — top result is visually relevant in ≥ 90% of test queries
- End-to-end query response time (intent parse + retrieval + LLM answer) under 3 seconds for archives up to 30 days

---

## Part 2: Architecture for Telemetry and Video Processing

### Overview

The architecture is divided into four layers: **Ingestion**, **Processing**, **Storage**, and **Delivery**. Each layer has a single well-defined responsibility and communicates with adjacent layers through typed interfaces.

```
+-------------------------------+
|       INGESTION LAYER         |
|  Video + Telemetry Sources    |
+-------------------------------+
               |
               v
+-------------------------------+
|       PROCESSING LAYER        |
|  Multi-modal AI Pipeline      |
+-------------------------------+
               |
               v
+-------------------------------+
|        STORAGE LAYER          |
|  Relational + Vector Stores   |
+-------------------------------+
               |
               v
+-------------------------------+
|        DELIVERY LAYER         |
|  Alerts + Dashboard + Search  |
+-------------------------------+
```

---

### Layer 1: Ingestion

Responsible for accepting raw video and telemetry from the drone and producing a normalised stream of frame packets.

```
Drone hardware
  |
  +-- RTSP video stream  (H.264, up to 4K)
  |
  +-- Telemetry feed     (MAVLink / custom UDP)
       battery, altitude, GPS, heading, speed

          |                        |
          v                        v
   VideoIngestor            TelemetryReceiver
   - frame sampling         - parse MAVLink
     (every Nth frame)      - normalise units
   - optional JPEG save     - attach to FramePacket
   - FramePacket output

          |                        |
          +----------+-------------+
                     |
                     v
              FramePacket
              {
                frame_id:    UUID
                ts:          datetime (UTC)
                video_id:    str
                frame_index: int
                image:       BGR ndarray
                metadata: {
                  battery_pct:  float
                  altitude_m:   float
                  gps_lat/lon:  float
                  heading_deg:  float
                }
              }
```

**Design note:** Telemetry is attached to the frame packet at ingestion time so that all downstream processing sees a single unified object. Telemetry-based rules (e.g., `battery_below: 10`) can fire in the same rule engine as visual rules without any special handling.

---

### Layer 2: Processing (the AI pipeline)

The processing layer is a multi-threaded pipeline where a reader thread feeds frames into a bounded queue, and a worker thread runs the LangGraph agent on each frame.

```
FramePacket stream
       |
       v
+------------------+      +-------------------+
| Reader Thread    |      | Worker Thread     |
|                  |      |                   |
| preprocess frame | ====>| LangGraph Agent   |
| letterbox/resize |queue | (6-node pipeline) |
| put(queue)       |      |                   |
|                  |      | perceive          |
| RTSP: evict old  |      |   YOLO detection  |
|   if queue full  |      |   CLIP embedding  |
|                  |      |   VLM caption     |
| File: block      |      |     [async pool]  |
|   until space    |      |                   |
+------------------+      | contextualize     |
                          |   track durations |
                          |   recent events   |
                          |                   |
                          | rule_check        |
                          |   YAML rules eval |
                          |   -> hits[]       |
                          |                   |
                          | llm_judge (maybe) |
                          |   Bedrock Haiku   |
                          |   genuine/fp?     |
                          |                   |
                          | alert             |
                          |   dedup/cooldown  |
                          |   store event     |
                          |                   |
                          | log               |
                          |   SQLite frame    |
                          |   ChromaDB embed  |
                          +-------------------+
                                    |
                                    v
                             result dict
                             {alerts_fired,
                              logged, ...}
```

**Data pipeline within the processing layer:**

```
Raw image (source res)
       |
       +---> YOLO letterbox (640x640)  ---> YOLODetector   --> DetectedObject[]
       |
       +---> VLM resize (378x378 RGB)  ---> VLMCaptioner   --> caption string
       |         [async thread pool]
       |
       +---> CLIP resize (224x224 f32) ---> CLIPEmbedder   --> [float x 512]
       |
       +---> JPEG thumbnail (640x360)  ---> disk save       --> frame_path

       All four happen in parallel. YOLO + CLIP are synchronous (~50ms total).
       VLM is async (~500ms, result collected on next iteration).
       Thumbnail save is background I/O.
```

---

### Layer 3: Storage

Two complementary stores serve different query patterns. A unified retriever combines them.

```
+----------------------------------+     +----------------------------------+
|         SQLiteStore              |     |         ChromaStore              |
|  (structured, relational)        |     |  (vector, semantic)              |
|                                  |     |                                  |
|  frames                          |     |  collection: security-events     |
|    id, ts, zone, caption,        |     |                                  |
|    frame_path, telemetry         |     |  per document:                   |
|                                  |     |    id = frame_id                 |
|  objects                         |     |    embedding = [float x 512]     |
|    frame_id, class, conf,        |     |    metadata:                     |
|    bbox, track_id                |     |      ts, zone, caption           |
|                                  |     |      class_names, video_id       |
|  events                          |     |                                  |
|    type, severity, zone,         |     |  hnsw:space = cosine             |
|    start_ts, frame_ids           |     |  pre-filter on metadata          |
|                                  |     |  before vector search            |
|  alerts                          |     +----------------------------------+
|    event_id, channel,            |
|    message, acked                |
+----------------------------------+

              Both accessed via HybridRetriever:

              User query
                  |
                  v
          parse_query_intent()
                  |
            has_time=True?
             /          \
           YES            NO
            |              |
     temporal_search()   search()
     SQL WHERE           CLIP embed query
     ts BETWEEN          cosine similarity
     class=?             top-K frames
     zone=?
            |              |
            +---------+----+
                      |
               FrameResult[]
               {frame_id, ts, zone,
                caption, class_names,
                semantic_score}
```

**Retention policy:** Frames older than `retention_days` (default: 30) are automatically purged from both stores. Events and alerts are retained indefinitely for audit purposes.

---

### Layer 4: Delivery

The delivery layer surfaces processed results to humans through three channels:

```
Processing layer results
         |
         +------+----------+----------+
         |               |           |
         v               v           v
+----------------+  +-----------+  +-------------------+
|  Alert         |  | Dashboard |  | Search / RAG      |
|  Dispatcher    |  | (Live tab)|  | (Ask tab)         |
|                |  |           |  |                   |
| console        |  | frame +   |  | natural language  |
| email (SMTP)   |  | bboxes    |  | query             |
| webhook (HTTP) |  |           |  |    |              |
| slack (future) |  | alert feed|  |    v              |
|                |  |           |  | HybridRetriever   |
| 5-min cooldown |  | FPS / obj |  |    |              |
| per rule+zone  |  | metrics   |  |    v              |
|                |  |           |  | Bedrock RAG       |
| severity badge |  | progress  |  | answer + frames   |
| high = red     |  | bar       |  |                   |
| medium = orange|  +-----------+  | multi-turn        |
| low = blue     |                 | chat history      |
+----------------+                 +-------------------+
         |
         v
  SQLite: store_alert()
  (permanent audit log)
```

**Alert flow for a single security event:**

```
Rule fires (e.g., after_hours_person)
    |
    | needs_llm=false -> skip judge
    v
Cooldown check: same rule+zone fired < 5 min ago?
    |
    YES -> suppress, no action
    NO
    |
    v
store_event(type=after_hours_person, severity=high, zone=main_gate)
store_alert(event_id, channel=console, message="Person at main_gate at 23:14")
    |
    v
AlertDispatcher.dispatch()
  -> console: print RED [HIGH] Person at main_gate at 23:14
  -> email:   if enabled, send to configured recipients
  -> webhook: if enabled, POST {alert_id, severity, message, ts, zone}
    |
    v
Dashboard alert feed updated (live polling)
```

---

### Architecture Summary

| Layer | Component | Technology | Responsibility |
|---|---|---|---|
| Ingestion | VideoIngestor | OpenCV + Python | Frame extraction, JPEG save |
| Ingestion | TelemetryReceiver | MAVLink / UDP | Drone state normalisation |
| Processing | FramePreprocessor | NumPy + PIL | Three parallel image views |
| Processing | YOLODetector | Ultralytics YOLOv8s | Object detection + tracking |
| Processing | VLMCaptioner | Bedrock Claude Haiku | Scene description |
| Processing | CLIPEmbedder | OpenCLIP ViT-B/32 | Semantic frame vectors |
| Processing | RuleEngine | PyYAML + Python | Declarative threat evaluation |
| Processing | LangGraph Agent | LangGraph | Pipeline orchestration |
| Processing | LLM Judge | Bedrock Claude Haiku | False-positive filtering |
| Storage | SQLiteStore | SQLite (WAL) | Structured events + metadata |
| Storage | ChromaStore | ChromaDB HNSW | Vector frame index |
| Storage | HybridRetriever | Python | Unified semantic + SQL search |
| Delivery | AlertDispatcher | Python | Console / email / webhook |
| Delivery | Streamlit Dashboard | Streamlit | Live view + timeline + chat |

# System Flow Diagrams

## 1. End-to-End Frame Processing Flow

```
+------------------+
|   Video Source   |
|  MP4 / RTSP /    |
|  Description     |
+--------+---------+
         |
         | FramePacket
         | {frame_id, ts, video_id, image}
         v
+------------------+      +------------------+
|  FramePreprocessor|      |  File mode:      |
|                  |      |  reader BLOCKS   |
|  yolo  640x640   |      |  until worker    |
|  vlm   378x378   |      |  has space       |
|  clip  224x224   |      |                  |
+--------+---------+      |  RTSP mode:      |
         |                |  reader EVICTS   |
         v                |  oldest frame    |
+------------------+      +------------------+
|  StreamProcessor |
|  Reader thread   |
|  Worker thread   |
|  Result queue    |
+--------+---------+
         |
         | state = {preprocessed, zone}
         v
+=============================+
|   LangGraph Agent Pipeline  |
+=============================+

 Node 1: perceive
 +--------------------------------+
 | YOLODetector.detect()          |
 |   -> list[DetectedObject]      |
 |      {class, conf, bbox,       |
 |       track_id}                |
 |                                |
 | CLIPEmbedder.embed()           |
 |   -> [float x 512]             |
 |                                |
 | VLMCaptioner.caption()  [async]|
 |   -> "Two people at gate..."   |
 +----------------+---------------+
                  |
                  v

 Node 2: contextualize
 +--------------------------------+
 | Track duration accumulator     |
 |   _track_first_seen dict       |
 |   elapsed = now - first_seen   |
 |   -> {track_id: seconds}       |
 |                                |
 | SQLite: get_vehicle_counts()   |
 | SQLite: get_events(last 10min) |
 |                                |
 | context_summary string         |
 |   "Zone: gate | 23:00 | ..."   |
 +----------------+---------------+
                  |
                  v

 Node 3: rule_check
 +--------------------------------+
 | For each rule in rules.yaml:   |
 |                                |
 |  after_hours_person?           |
 |    person detected             |
 |    AND 22:00 <= now <= 06:00   |
 |    -> HIT (severity=high)      |
 |                                |
 |  loitering?                    |
 |    track_duration >= 60s       |
 |    -> HIT (needs_llm=true)     |
 |                                |
 |  gate_tampering_attempt?       |
 |    caption has "tools"+"gate"  |
 |    -> HIT (needs_llm=true)     |
 |                                |
 | -> rule_hits[], needs_llm bool |
 +----------------+---------------+
                  |
         +--------+---------+
         |                  |
  rule_hits=[]     rule_hits not empty
         |                  |
         |        +---------+--------+
         |        |                  |
         |  needs_llm=False    needs_llm=True
         |        |                  |
         |        v                  v

 (skip)      Node 4: alert     Node 4: llm_judge
             +------------+    +------------------+
             |Dedup check:|    | Claude Haiku via |
             |5-min cool- |    | Amazon Bedrock   |
             |down per    |    |                  |
             |(rule,zone) |    | context_summary  |
             |            |    | + caption        |
             |store_event |    | + rule_hits      |
             |store_alert |    |                  |
             |            |    | VERDICT: genuine |
             +-----+------+    | ALERT: message   |
                   |           | REASON: ...      |
                   |           +--------+---------+
                   |                    |
                   |                    v
                   |             Node 4b: alert
                   |             (same as left)
                   |
                   +-------+----+
                           |
                           v

 Node 5: log (always runs)
 +--------------------------------+
 | SQLite: store_frame()          |
 | SQLite: store_objects()        |
 | ChromaDB: add_frame(embedding) |
 | -> {logged: True}              |
 +--------------------------------+
```

---

## 2. Alert Decision Flow

```
rule_hits = RuleEngine.evaluate(ts, zone, detections, track_durations, caption)

     |
     | empty?
     +--------YES-------> log only, no alert
     |
     NO
     |
     | any needs_llm=True?
     +--------YES-------> llm_judge
     |                      |
     |              VERDICT: genuine?
     |                 YES |    NO
     |                     |     +-> store [LLM: REVIEW] alert
     |                     v
     +--------NO-------> alert node
                            |
                    cooldown check:
                    same (rule, zone) fired < 5 min ago?
                         YES |    NO
                             |     +-> store_event()
                             |         store_alert()
                             |         alerts_fired.append()
                             |
                    suppress (no re-alert)
```

---

## 3. Query Routing in the Ask Tab

```
User types: "show me vehicles near the gate on 13 April"
                         |
                         v
            _parse_query_intent() [Bedrock call]
                         |
                  Returns JSON:
            {
              "has_time": true,
              "start": "2026-04-13T00:00:00Z",
              "end":   "2026-04-13T23:59:59Z",
              "class_filter": "car",
              "zone": "gate",
              "semantic_query": "vehicles near gate"
            }
                         |
               has_time == True?
                    /          \
                 YES             NO
                  |               |
                  v               v
         temporal_search()    chroma.query()
         SQLite:               [embed query text]
         JOIN objects          [cosine similarity]
         WHERE class=car       [top-K frames]
         AND ts BETWEEN
         AND zone=gate
                  |               |
                  +-------+-------+
                          |
                  list[FrameResult]
                          |
                  _build_smart_context()
                    caption snippets
                    + event log
                    + frame timestamps
                          |
                          v
               _bedrock_rag_answer()
               [Claude Haiku]
               [chat_history injected]
                          |
                  Answer + source frames
                  displayed in Ask tab
```

---

## 4. Storage Architecture

```
+-----------------------------------+
|         SQLiteStore               |
|  (data/security_memory.db)        |
|                                   |
|  frames ----+                     |
|  |id|ts|    |                     |
|  |zone|cap  |                     |
|  |path|...  |                     |
|             |                     |
|  objects <--+                     |
|  |frame_id| |                     |
|  |class|    |                     |
|  |conf|bbox |                     |
|  |track_id  |                     |
|             v                     |
|  events                           |
|  |id|type|  |                     |
|  |severity  |                     |
|  |zone|...  |                     |
|             |                     |
|  alerts <---+                     |
|  |event_id| |                     |
|  |channel|  |                     |
|  |acked|... |                     |
+-----------------------------------+
         ^              |
         | writes       | reads
         |              v
+--------+--------------+----------+
|           Agent Pipeline         |
|  log node, alert node,           |
|  contextualize node              |
+----------------------------------+
         ^
         |  ChromaDB (vector store)
         |
+-----------------------------------+
|   collection: security-events    |
|                                   |
|  doc_id: frame_id                 |
|  embedding: [float x 512]         |
|  metadata:                        |
|    ts, zone, caption              |
|    class_names, video_id          |
+-----------------------------------+
         ^
         | writes (log node)
         |
         | queries (HybridRetriever)
         v
+-----------------------------------+
|        HybridRetriever           |
|  temporal_search() -> SQL path   |
|  search()          -> vector path|
+-----------------------------------+
```

---

## 5. Multi-threaded StreamProcessor

```
Main thread (Streamlit)
      |
      | processor.start(ingestor, zone)
      v
+-------------------------------------+
|         StreamProcessor             |
|                                     |
| +----------------+  +-------------+ |
| | Reader Thread  |  | Worker Thread| |
| |                |  |             | |
| | for pkt in     |  | while True: | |
| | ingestor:      |  |   pkt=q.get | |
| |                |  |   pre=prepr | |
| |  pre=preprocess|  |   state={   | |
| |                |  |    preproc, | |
| |  RTSP mode:    |  |    zone}    | |
| |  if q.full():  |  |             | |
| |   q.get_nowait |  |  result=    | |
| |   dropped++    |  |  agent.invoke| |
| |  q.put(pre)    |  |             | |
| |                |  |  rq.put(res)| |
| |  File mode:    |  |  stats++    | |
| |  q.put(pre)    |  |             | |
| |  [blocks if    |  +-------------+ |
| |   full]        |                  |
| +----------------+                  |
|                                     |
| running = _running                  |
|        OR not q.empty()             |
|        OR not rq.empty()            |
+-------------------------------------+
      |
      | processor.get_result(timeout=0.1)
      v
 Streamlit UI updates frame + alerts
```

---

## 6. VLM Async Processing

```
Frame N arrives:
+------------------------------------------+
| perceive node                            |
|                                          |
| 1. Poll VLM futures (non-blocking)       |
|    future[N-K].done()? collect caption   |
|                                          |
| 2. Submit VLM job for frame N            |
|    executor.submit(captioner.caption, N) |
|                                          |
| 3. Run YOLO on frame N       [~30ms]     |
| 4. Run CLIP on frame N       [~20ms]     |
| 5. Return state (caption=None if pending)|
+------------------------------------------+
                  ... (500ms later) ...
Frame N+K arrives:
+------------------------------------------+
| 1. Poll VLM futures                      |
|    future[N].done() -> caption = "..."   |
|    Now caption-based rules can fire      |
|    for context of frame N+K              |
+------------------------------------------+
```

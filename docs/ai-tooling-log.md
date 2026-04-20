# AI Tools Integration

## Overview

This document details every AI model and service integrated into the drone security system, including the specific role each plays, the integration architecture, configuration options, observed impact on the system's capabilities, and workflow lessons learned.

---

## 1. Amazon Bedrock — Claude Haiku

### Role
Dual-purpose: (1) VLM scene captioning on every sampled frame, (2) LLM judge for false-positive filtering on rules with `needs_llm: true`.

### Integration

**Captioning (VLMCaptioner)**

```python
# src/perception/vlm_captioner.py
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

client = ChatBedrock(
    model_id=os.getenv("VLM_CAPTIONER_MODEL",
                       "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    model_kwargs={"max_tokens": 64},
)

# Frame is sent as base64-encoded JPEG at 378x378 resolution
response = client.invoke([HumanMessage(content=[
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    {"type": "text", "text": system_prompt},
])])
```

System prompt used for captioning:
> "You are a security camera analyst. In 25 words or fewer, describe what is happening from a security perspective. Focus on people, vehicles, and unusual activity. Do not include metadata, timestamps, or camera details."

The 25-word cap keeps API costs low (~$0.0003 per 1000 output tokens for Haiku) and forces the model to prioritise the most security-relevant observation. Captions are cached per `frame_id` to avoid duplicate API calls.

**LLM judge (llm_judge node)**

```python
# src/agent/nodes.py
llm = ChatBedrock(
    model_id=model_id,
    region_name=region,
    model_kwargs={"max_tokens": 200},
)

prompt = f"""You are a drone security analyst AI.
Context: {context_summary}
Visual description: {caption}

Triggered security rules:
{hits_text}

Task:
1. Assess whether these rule hits represent a genuine security concern or a false positive.
2. If genuine, write a concise alert message (1-2 sentences).
3. If false positive, explain briefly why.

Respond in this exact format:
VERDICT: <genuine|false_positive>
ALERT: <alert message or 'N/A'>
REASON: <one sentence explanation>"""
```

The structured output format (`VERDICT: / ALERT: / REASON:`) enables reliable parsing without JSON mode — Claude consistently follows this template.

**Multi-turn RAG chat (Ask tab)**

The Ask tab builds a conversation history of up to 4 prior exchanges:
```python
messages = [SystemMessage(content=system_prompt)]
for msg in chat_history[-8:]:   # last 4 Q&A pairs = 8 messages
    if msg["role"] == "user":
        messages.append(HumanMessage(content=msg["content"]))
    elif msg["role"] == "assistant":
        messages.append(AIMessage(content=msg["content"]))
messages.append(HumanMessage(content=current_query))
```

A separate `_reformulate_query()` call rewrites follow-up questions as standalone search queries before they are sent to ChromaDB. For example: *"what about vehicles?"* → *"vehicles in security footage"*.

### Configuration

| Environment variable | Default | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | — | IAM key with Bedrock:InvokeModel permission |
| `AWS_SECRET_ACCESS_KEY` | — | IAM secret |
| `AWS_REGION_NAME` | `us-east-1` | Bedrock endpoint region |
| `VLM_CAPTIONER_MODEL` | `us.amazon.nova-2-lite-v1:0` | Bedrock model ID |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model ID |


### Impact on workflow

**Eliminated hand-coded behavioral rules.** Before VLM integration, detecting "person using tools near gate" would require training a custom classifier or hand-writing complex multi-object proximity logic. With VLM captions, a single YAML rule with `caption_keywords: [tools, gate, lock]` covers the scenario in two lines.

**Reduced false-positive handling code.** Rules that are intentionally broad (crowd gathering, loitering) previously required complex per-zone threshold tuning. `needs_llm: true` delegates the ambiguous judgment to Claude, which considers the full visual context (time of day, caption, nearby events) before deciding.

**Multi-turn query interface.** The Ask tab became a natural conversation interface once Bedrock was already in the stack. Building the same with a local model would have required a separate model download and serving setup.

---

## 2. YOLOv8s + ByteTrack (Ultralytics)

### Role
Real-time object detection on every frame, plus persistent track ID assignment across frames for duration-based rules.

### Integration

```python
# src/perception/yolo_detector.py
from ultralytics import YOLO

model = YOLO("yolov8s.pt")   # weights auto-downloaded on first run

# Detect + track in one call
results = model.track(
    source=yolo_input,         # 640x640 letterboxed BGR uint8
    conf=0.4,                  # confidence threshold
    persist=True,              # maintain ByteTrack state between calls
    verbose=False,
)

for box in results[0].boxes:
    detection = DetectedObject(
        class_name = model.names[int(box.cls)],
        confidence = float(box.conf),
        bbox       = tuple(box.xyxy[0].int().tolist()),
        track_id   = int(box.id) if box.id is not None else None,
    )
```

The `persist=True` flag keeps ByteTrack's Kalman filter state between calls so track IDs are consistent across frames even through brief occlusions. This is the foundation of the loitering and stopped-vehicle rules.

### Model selection rationale

| Variant | Size | Speed (CPU) | mAP | Choice |
|---|---|---|---|---|
| YOLOv8n | 6.3 MB | ~45 fps | 37.3 | Too low accuracy |
| **YOLOv8s** | **22 MB** | **~28 fps** | **44.9** | **Chosen** |
| YOLOv8m | 52 MB | ~15 fps | 50.2 | Too slow for live |

YOLOv8s achieves the best throughput/accuracy balance for security surveillance on laptop-class hardware.

### Impact on workflow

A single `model.track()` call replaced what would otherwise require: object detection model, non-maximum suppression, a separate tracking algorithm (SORT, DeepSORT, etc.), and post-processing to link detections to tracks. The Ultralytics API compresses this into a few lines of code.

---

## 3. OpenCLIP (ViT-B/32)

### Role
Generate 512-dimensional semantic embeddings for each video frame, enabling natural language search over the frame archive.

### Integration

```python
# src/perception/clip_embedder.py
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai",
)
tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

# Frame embedding
with torch.no_grad():
    img_tensor = torch.from_numpy(clip_input).unsqueeze(0)  # (1, 3, 224, 224)
    embedding = model.encode_image(img_tensor)
    embedding = F.normalize(embedding, dim=-1)              # L2 normalise
    return embedding.squeeze().tolist()                     # list[float, 512]

# Text query embedding (LRU-cached)
@lru_cache(maxsize=256)
def embed_text(self, query: str) -> list[float]:
    tokens = tokenizer([query])
    with torch.no_grad():
        vec = model.encode_text(tokens)
        return F.normalize(vec, dim=-1).squeeze().tolist()
```

L2 normalisation ensures that cosine similarity equals the dot product, which ChromaDB's HNSW index optimises natively.

### Why ViT-B/32 over larger CLIP variants

| Variant | Size | Embedding dim | Speed | CLIP score |
|---|---|---|---|---|
| ViT-B/32 | 150 MB | 512 | ~50 fps | Good |
| ViT-L/14 | 890 MB | 768 | ~8 fps | Excellent |
| ViT-H/14 | 3.6 GB | 1024 | ~3 fps | Best |

ViT-B/32 is fast enough to embed every frame without becoming a pipeline bottleneck, and its 512-dim space is sufficient for differentiating security-relevant visual concepts (person vs. vehicle, gate area vs. parking lot, day vs. night lighting).

### Anomaly detection hook

The embedder maintains a running centroid of all processed frames. Frames with L2 distance from the centroid above a configurable threshold are flagged as visually anomalous — a free unsupervised baseline that doesn't require any labelled data.

### Impact on workflow

Zero training data required. The pre-trained ViT-B/32 weights (trained on 400M image-text pairs) generalise well to security camera scenes. A search for *"person climbing fence"* returns relevant frames even though no security-specific fine-tuning was done.

---

## 4. Moondream2 (Local VLM Fallback)

### Role
Offline scene captioning when AWS Bedrock is unavailable (no credentials, no internet, rate-limited).

### Integration

```python
# src/perception/vlm_captioner.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cpu"},
)
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2025-01-09")

answer = model.answer_question(
    model.encode_image(vlm_input),   # PIL Image 378x378
    "Describe what is happening in this security camera image in 25 words or fewer.",
    tokenizer,
)
```

Moondream2 uses ~2 GB of RAM and runs at approximately 1-2 fps on a modern CPU. Slower than Bedrock but functional.

### Fallback chain

```python
def caption(self, preprocessed, detections=None, zone=""):
    try:
        return self._bedrock_caption(preprocessed, detections, zone)
    except Exception as e:
        logger.warning("Bedrock captioning failed (%s), trying Moondream.", e)
        try:
            return self._moondream_caption(preprocessed, detections, zone)
        except Exception as e2:
            logger.error("Moondream captioning failed (%s), using structured fallback.", e2)
            return self._structured_fallback(detections, zone)
```

The structured fallback builds a caption from YOLO detections: *"Detected: 2 person, 1 car at main_gate"* — no semantic content but at least captures what was seen.

### Impact on workflow

Allows the system to function in air-gapped deployments (e.g., a drone base in a remote location without reliable internet). Caption-based rules degrade gracefully — they simply don't fire when the fallback caption is structured rather than descriptive.

---

## 5. LangGraph + LangChain

### Role
Agent pipeline orchestration (LangGraph) and LLM client abstraction (LangChain AWS).

### LangGraph integration

```python
# src/agent/graph.py
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("perceive",      perceive_fn)
graph.add_node("contextualize", contextualize_fn)
graph.add_node("rule_check",    rule_check_fn)
graph.add_node("llm_judge",     llm_judge_fn)
graph.add_node("alert",         alert_fn)
graph.add_node("log",           log_fn)

graph.set_entry_point("perceive")
graph.add_edge("perceive",      "contextualize")
graph.add_edge("contextualize", "rule_check")
graph.add_conditional_edges("rule_check", route_after_rule_check, {
    "llm_judge": "llm_judge",
    "alert":     "alert",
    "log":       "log",
})
graph.add_edge("llm_judge", "alert")
graph.add_edge("alert",     "log")
graph.add_edge("log",       END)

agent = graph.compile()

# Per-frame invocation
result = agent.invoke({"preprocessed": frame, "zone": "main_gate"})
```

### LangChain AWS integration

`ChatBedrock` handles:
- Boto3 session creation from environment variables
- Request serialisation to Bedrock's `InvokeModel` API format
- Retry logic on throttling
- Streaming support (not used currently, but available)

Switching models requires only changing `model_id` — no changes to prompt construction, response parsing, or error handling.

### Impact on workflow

The `add_conditional_edges` API allowed the `needs_llm` routing logic to be expressed in 6 lines of code that clearly document every possible path. Without LangGraph, the same logic would require careful management of return values and conditional dispatch across multiple functions.

---

## 6. ChromaDB

### Role
Persistent vector store for CLIP frame embeddings with metadata-accelerated filtering.

### Integration

```python
# src/memory/chroma_store.py
import chromadb

client = chromadb.PersistentClient(path=persist_dir)  # or EphemeralClient() for tests
collection = client.get_or_create_collection(
    name="security-events",
    metadata={"hnsw:space": "cosine"},
)

# Index a frame
collection.add(
    ids=[frame_id],
    embeddings=[embedding],
    metadatas=[{
        "frame_index": frame_index,
        "ts": ts_str,
        "video_id": video_id,
        "zone": zone,
        "caption": caption,
        "class_names": ",".join(class_names),
    }],
)

# Query by text
text_embedding = clip_embedder.embed_text(query)
results = collection.query(
    query_embeddings=[text_embedding],
    n_results=top_k,
    where={"zone": {"$eq": "main_gate"}},   # metadata pre-filter
)
```

The `where` clause pre-filters documents before computing cosine similarity — critical for performance when the collection contains millions of frames.

### Ephemeral mode for tests

```python
# For tests: in-memory, no disk persistence
client = chromadb.EphemeralClient()
```

This avoids test pollution (no leftover data between test runs) and eliminates the need to clean up files after tests.

### Impact on workflow

The combination of CLIP embeddings + ChromaDB made natural language frame search trivially implementable. A 10-line `query_by_text()` method backed by a pre-trained embedding model produces semantically meaningful search results with zero training data.

---

## AI Tool Integration Summary

| Tool | Version | Purpose | Hot path? | Offline capable? |
|---|---|---|---|---|
| Claude Haiku (Bedrock) | Haiku 4.5 | RAG | Yes (async) | No (fallback to Moondream) |
| Nova (Bedrock) | Nova 2 Lite | VLM captioning + LLM judge | Yes (async) | No (fallback to Moondream) |
| YOLOv8s | 8.4.38 | Object detection + tracking | Yes | Yes |
| OpenCLIP ViT-B/32 | 3.3.0 | Frame embeddings | Yes | Yes |
| LangGraph | 1.1.8 | Pipeline orchestration | Yes | Yes |
| LangChain AWS | 1.4.4 | Bedrock client | Yes | No |
| ChromaDB | 1.5.8 | Vector search | Read path | Yes |
| Moondream2 | HF 2025-01-09 | VLM fallback | No (fallback) | Yes |

---

## Lessons Learned

**1. Model input/output contracts must be explicit.**  
YOLO, CLIP, and VLM each require different image formats (BGR vs RGB, different sizes, uint8 vs float32). `FramePreprocessor` centralises all normalisation — the individual model wrappers receive the correct format without each having to know about the source image.

**2. Test model integrations with deterministic stubs, not real models.**  
Running YOLO + Bedrock + ChromaDB in CI would require GPU, AWS credentials, and network. The stub pattern (`StubDetector`, `StubCaptioner`, `StubEmbedder`) keeps the integration test suite under 20 seconds and fully offline.

**3. LLM structured output parsing needs fallback handling.**  
Claude's VERDICT/ALERT/REASON format is reliably followed, but the `llm_judge` node catches all exceptions and defaults to treating the rule hit as genuine. A failed LLM call should never suppress a security alert.

**4. Async model calls require careful state management.**  
The async VLM design collects completed futures on the next frame. This means the perceive node must merge both current-frame YOLO results and previous-frame VLM results into a single state update — a pattern that was easy to get wrong until it was made explicit in the node's return value.

**5. ChromaDB metadata pre-filtering is essential for large collections.**  
A query over 100,000 frames without a `where` clause recomputes cosine similarity for all of them. Adding `where={"zone": {"$eq": requested_zone}}` reduces the search space by an order of magnitude before any vector math is done.

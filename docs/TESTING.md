# Testing Documentation

## Overview

The test suite is organised in three layers that mirror the system's architecture: unit tests validate individual modules in isolation, integration tests exercise the full LangGraph pipeline with stubbed models, and scenario tests run end-to-end security event simulations against the production rule set.

**Current status: 124 tests, 0 failures.**

```
tests/
+-- conftest.py                 Shared fixtures injected by pytest
+-- unit/                       Per-module isolated tests (no network, no GPU)
|   +-- test_rule_engine.py     47 tests
|   +-- test_sqlite_store.py    21 tests
|   +-- test_hybrid_retriever.py 17 tests
|   +-- test_stream_processor.py 10 tests
|   +-- test_frame_preprocessor.py 21 tests
+-- integration/                Full pipeline with stubs (no real models)
|   +-- test_agent_pipeline.py  12 tests
+-- scenarios/                  End-to-end security event simulations
    +-- test_security_scenarios.py 16 tests
```

---

## Running Tests

```bash
# Full suite
uv run pytest tests/ -v

# With HTML coverage report
uv run pytest tests/ --cov=src --cov-report=html
# open htmlcov/index.html

# Specific layer
uv run pytest tests/unit/           # unit tests only
uv run pytest tests/integration/    # integration tests only
uv run pytest tests/scenarios/      # scenario tests only

# Single file
uv run pytest tests/unit/test_rule_engine.py -v

# Single test
uv run pytest tests/unit/test_rule_engine.py::TestAfterHoursPerson::test_triggers_at_night -v
```

---

## Shared Fixtures (conftest.py)

All fixtures in `tests/conftest.py` are auto-injected by pytest with function scope (a fresh instance per test).

| Fixture | Type | Description |
|---|---|---|
| `sqlite_db` | `SQLiteStore` | In-memory database — schema applied, isolated per test |
| `chroma_store` | `ChromaStore` | Ephemeral RAM-only ChromaDB collection |
| `ts_day` | `datetime` | 2026-04-20 14:00 UTC — daytime, allowed hours |
| `ts_night` | `datetime` | 2026-04-20 23:00 UTC — after hours |
| `ts_early_morning` | `datetime` | 2026-04-20 03:00 UTC — still after hours (past midnight) |
| `blank_frame` | `np.ndarray` | 640x480 black BGR uint8 image |
| `sample_packet` | `FramePacket` | Ready-to-use FramePacket with blank frame |
| `preprocessor` | `FramePreprocessor` | Default preprocessor (yolo_size=640, vlm_every=5) |

Helper functions (not fixtures, used directly in test files):

```python
def make_detection(class_name="person", confidence=0.9,
                   track_id=1, bbox=(100,100,200,200)) -> DetectedObject

def write_rules_yaml(tmp_path: Path, rules_yaml: str) -> Path
```

---

## Unit Tests

### test_rule_engine.py — 47 tests

Tests the `RuleEngine` in complete isolation. Each test writes a minimal inline YAML rule via `tmp_path` so tests are fully independent of `configs/rules.yaml`.

**Test classes and coverage:**

#### TestAfterHoursPerson (5 tests)
Validates time-range conditions including midnight wrap-around.

| Test | Scenario | Expected |
|---|---|---|
| `test_triggers_at_night` | person at 23:00 | 1 hit, severity=high |
| `test_no_trigger_during_day` | person at 14:00 | 0 hits |
| `test_triggers_past_midnight` | person at 03:00 (inside 22:00-06:00) | 1 hit |
| `test_no_trigger_without_person_detection` | car at 23:00 | 0 hits |
| `test_message_contains_zone` | person at 23:00, zone="main_gate" | message contains "main_gate" |

#### TestLoitering (4 tests)
Validates duration-based conditions using track_durations.

| Test | Scenario | Expected |
|---|---|---|
| `test_triggers_when_duration_exceeded` | track_id=42 for 60s, threshold=30s | 1 hit |
| `test_no_trigger_below_threshold` | track_id=42 for 10s | 0 hits |
| `test_no_trigger_without_track_durations` | track_id=None | 0 hits |
| `test_triggers_on_exact_threshold` | track_id=7 for exactly 30s | 1 hit |

#### TestCrowdGathering (3 tests)
Validates `min_object_count` with class filtering.

| Test | Scenario | Expected |
|---|---|---|
| `test_triggers_at_min_count` | 3 people | 1 hit |
| `test_no_trigger_below_min_count` | 2 people | 0 hits |
| `test_non_person_detections_not_counted` | 2 people + 1 car | 0 hits (car doesn't count) |

#### TestForbiddenZone (3 tests)
Validates `zone_in` condition.

| Test | Scenario | Expected |
|---|---|---|
| `test_triggers_in_forbidden_zone` | zone="restricted_area" | 1 hit |
| `test_no_trigger_in_allowed_zone` | zone="main_gate" | 0 hits |
| `test_triggers_for_second_zone_in_list` | zone="server_room" | 1 hit |

#### TestCaptionKeywords (4 tests)
Validates `caption_keywords` + `caption_min_match` with case sensitivity.

| Test | Scenario | Expected |
|---|---|---|
| `test_triggers_when_min_keywords_matched` | caption with "gate" + "tools" (2 of 3) | 1 hit |
| `test_no_trigger_below_min_match` | caption with only "tools" (1 of 3) | 0 hits |
| `test_no_trigger_empty_caption` | caption="" | 0 hits |
| `test_case_insensitive_keyword_match` | "TOOLS...GATE...LOCK" uppercase | 1 hit |

#### TestVehicleAfterHours (3 tests)
Validates `object_class_in` with time_range.

#### TestMultipleRules (7 tests)
Validates multiple rules firing simultaneously, `needs_llm` propagation, and hot-reload.

| Test | Scenario | Expected |
|---|---|---|
| `test_multiple_rules_fire_simultaneously` | night + loitering | 2 hits |
| `test_needs_llm_false_propagated` | after_hours rule | needs_llm=False |
| `test_needs_llm_true_propagated` | loitering rule | needs_llm=True |
| `test_reload_picks_up_updated_rules` | reload with different yaml | old rule no longer fires |
| `test_missing_yaml_uses_defaults` | nonexistent path | uses built-in defaults, >0 rules |

---

### test_sqlite_store.py — 21 tests

Tests all CRUD operations on the four database tables using an in-memory SQLiteStore.

**Key design decision tested:** Each test gets a fresh `SQLiteStore(":memory:")`. The in-memory connection is cached in `self._mem_conn` so all queries within a test share the same SQLite connection (and therefore the same tables).

#### TestFrames (6 tests)
- Store and retrieve a frame record
- Foreign key to zone
- Caption round-trip
- `get_frames_by_ids` preserves insertion order
- `get_frames_by_ids` with empty list returns `[]`
- `get_frames_in_range` time window filter

#### TestObjects (4 tests)
- Store multiple objects for a frame
- `get_objects_for_frame` returns all detections
- `get_objects_by_class` filters by class name
- `get_objects_by_class` with time range

#### TestEvents (4 tests)
- Store and retrieve an event
- `get_events` filters by type, severity, zone
- `get_events` filters by time window

#### TestAlerts (4 tests)
- Store an alert linked to an event
- `get_unacked_alerts` returns only unacked
- `ack_alert` marks alert as acknowledged
- Cascade: alert requires valid event_id

#### TestStats (3 tests)
- `get_stats` counts correct totals
- Counts update after inserts

---

### test_hybrid_retriever.py — 17 tests

Tests the `HybridRetriever` using in-memory SQLite + ephemeral ChromaDB.

#### TestTemporalSearch (9 tests)

| Test | Validates |
|---|---|
| `test_returns_frames_in_time_window` | start/end boundary filter |
| `test_returns_empty_for_no_matches` | empty window returns [] |
| `test_class_filter_returns_only_matching_frames` | class_filter parameter |
| `test_class_filter_with_no_matches_returns_empty` | class mismatch |
| `test_zone_filter_narrows_results` | zone parameter |
| `test_results_are_frame_result_objects` | return type is FrameResult |
| `test_class_names_populated_in_results` | detections merged into result |
| `test_top_k_limits_results` | top_k=5 caps at 5 |
| `test_no_start_end_returns_all_frames` | no filters = all frames |

#### TestEventsSummary (5 tests)

| Test | Validates |
|---|---|
| `test_returns_all_events_unfiltered` | no filters = all events |
| `test_filters_by_event_type` | event_type filter |
| `test_filters_by_severity` | severity filter |
| `test_filters_by_time_window` | start/end filter |
| `test_filters_by_zone` | zone filter |

#### TestFramesInWindow (3 tests)
- Returns frames with detections joined
- Returns detection count per frame
- Empty window returns []

---

### test_frame_preprocessor.py — 21 tests

Tests frame normalisation with no GPU or model dependencies.

**Key scenarios:**

| Test class | What is validated |
|---|---|
| `TestLetterboxing` | Aspect ratio preservation, padding dimensions, landscape/portrait/square |
| `TestPreprocessedOutputs` | YOLO shape (640,640,3), VLM is PIL RGB, CLIP is float32 [0,1] |
| `TestVLMStride` | `run_vlm=True` at frame_index=0, False at non-multiples |
| `TestFramePacketMetadata` | frame_id, ts, video_id propagated through |
| `TestEdgeCases` | Very small frames, non-standard aspect ratios |

---

### test_stream_processor.py — 10 tests

Tests the multi-threaded pipeline using `FakeIngestor` (yields N synthetic frames) and `FakeAgent` (records calls, returns deterministic result).

#### TestFileModeAllFramesProcessed (3 tests)

These are regression tests for the "only 3-4 frames processed" bug.

| Test | Scenario | Validates |
|---|---|---|
| `test_all_frames_processed_short_video` | N=20 frames, queue_size=4, drop_on_full=False | len(agent.calls) == 20 |
| `test_all_frames_processed_with_slow_worker` | N=10, worker takes 50ms each | all 10 processed |
| `test_no_frames_dropped_in_file_mode` | N=15, drop_on_full=False | stats["dropped"] == 0 |

The root cause of the original bug: `drop_on_full=True` (default) was used for file sources. The reader fills a queue_size=4 queue in milliseconds while the worker takes seconds per frame — resulting in all but 3-4 frames being evicted before they were processed.

#### TestRtspModeDropsFrames (2 tests)
Validates that RTSP mode (drop_on_full=True + slow worker) correctly drops frames and updates stats.

#### TestRunningProperty (2 tests)

| Test | Validates |
|---|---|
| `test_running_true_while_results_queued` | running=True even after frame queue drained |
| `test_running_becomes_false_after_full_drain` | running=False only after results consumed |

The `running` property checks all three: `_running`, `_queue.empty()`, `_results.empty()`. The original bug checked only `_queue` — causing the UI loop to exit before consuming results.

#### TestStop (1 test)
- `stop()` terminates an infinite source within timeout

#### TestStats (1 test)
- `processed`, `read`, `alerts` counts correct after N frames

---

## Integration Tests

### test_agent_pipeline.py — 12 tests

Exercises the full LangGraph graph with three lightweight stubs replacing all heavy models:

```python
class StubDetector:
    """Returns a fixed list of detections."""
    def detect_from_preprocessed(self, preprocessed):
        return self._detections

class StubCaptioner:
    """Returns a fixed caption string."""
    def caption_from_preprocessed(self, preprocessed, detections=None, zone=""):
        return self._caption

class StubEmbedder:
    """Returns a zero 512-dim vector."""
    def embed_preprocessed(self, preprocessed):
        return [0.0] * 512
```

Storage is real: SQLiteStore(":memory:") and ChromaStore(persist_dir=None).

#### TestNoAlertPath (2 tests)

| Test | Scenario | Validates |
|---|---|---|
| `test_log_node_runs_when_no_rule_hits` | daytime person, after_hours rule | alerts_fired=[], logged=True |
| `test_frame_stored_in_sqlite_after_no_alert` | any frame | frame appears in db.get_frames_by_video() |

**Note:** When no rules fire, the graph routes to `log` directly (skipping `alert`). The `alerts_fired` key is absent from the result — tests use `result.get("alerts_fired", []) == []`.

#### TestAlertPath (3 tests)

| Test | Scenario | Validates |
|---|---|---|
| `test_alert_fired_for_rule_hit` | night + person + after_hours rule | 1 alert, rule_name correct |
| `test_alert_stored_in_sqlite` | same | db.get_events() returns event |
| `test_alert_deduplication_cooldown` | two consecutive invocations | r2 alerts_fired == [] (suppressed) |

#### TestLLMJudgePath (2 tests)

These tests patch `ChatBedrock` **before** `build_agent()` is called, because `ChatBedrock` is instantiated inside `make_llm_judge_node()` at graph-build time.

The test invokes the agent twice: first to seed the track (0s duration), second 65s later (65s > 30s loitering threshold).

```python
with patch("src.agent.nodes.ChatBedrock") as MockLLM:
    MockLLM.return_value.invoke.return_value = fake_response
    agent = build_agent(...)     # ChatBedrock mock active at build time
    agent.invoke(frame_at_T)     # seeds track
    result = agent.invoke(frame_at_T_plus_65s)   # triggers loitering
```

| Test | LLM verdict | Expected |
|---|---|---|
| `test_genuine_verdict_fires_alert` | "genuine" | 1 alert with LLM message |
| `test_false_positive_stored_with_review_tag` | "false_positive" | 1 alert tagged `[LLM: REVIEW]` |

#### TestStateFields (3 tests)
Validates that `frame_id`, `zone`, and `caption` are correctly propagated through the full graph and returned in the final state.

---

## Scenario Tests

### test_security_scenarios.py — 16 tests

End-to-end tests using the **production** `configs/rules.yaml`. These tests serve as living documentation of the system's actual security posture — if a rule changes in the YAML, the corresponding scenario test will fail.

The `engine` fixture is module-scoped (loaded once for all 16 tests):

```python
@pytest.fixture(scope="module")
def engine():
    return RuleEngine(Path("configs/rules.yaml"))
```

### Scenario 1: After-hours Intruder (3 tests)

| Test | Input | Asserts |
|---|---|---|
| `test_person_at_2am_triggers_high_alert` | person, 02:00, main_gate | after_hours_person fires, severity=high, needs_llm=False |
| `test_vehicle_at_midnight_triggers_high_alert` | truck, 00:00, parking | vehicle rule fires |
| `test_no_alert_for_person_at_midday` | person, 12:00, main_gate | after_hours_person does NOT fire |

### Scenario 2: Loitering Suspect (2 tests)

| Test | Input | Asserts |
|---|---|---|
| `test_loitering_fires_after_threshold` | person, track 90s (threshold=60s) | loitering fires, severity=medium, needs_llm=True |
| `test_loitering_below_threshold_no_alert` | person, track 30s | loitering does NOT fire |

### Scenario 3: Crowd & Tailgating (3 tests)

| Test | Input | Asserts |
|---|---|---|
| `test_crowd_fires_for_3_or_more_people` | 3 persons, courtyard | crowd_gathering fires |
| `test_tailgating_fires_at_gate_with_2_people` | 2 persons, main_gate | tailgating fires |
| `test_tailgating_does_not_fire_in_wrong_zone` | 2 persons, parking_lot | tailgating does NOT fire |

### Scenario 4: Gate Tampering (2 tests)

| Test | Input | Asserts |
|---|---|---|
| `test_gate_tampering_fires_for_tools_at_gate` | 3 persons + caption "tools near the lock" | gate_tampering_attempt fires, needs_llm=True |
| `test_gate_tampering_no_fire_on_unrelated_caption` | person + "walking past the building" | gate_tampering_attempt does NOT fire |

### Scenario 5: Coordinated Breach (1 test)

| Test | Input | Asserts |
|---|---|---|
| `test_coordinated_breach_fires_for_guard_standing_watch` | 2 persons + caption describing lock tampering + lookout | any breach-related rule fires |

This test accepts any of: `coordinated_breach_attempt`, `lookout_behavior`, `gate_tampering_attempt`, `tool_use_near_infrastructure` — it validates that at least one breach-related rule fires for the described scenario.

### Scenario 6: Multiple Simultaneous Alerts (2 tests)

| Test | Input | Asserts |
|---|---|---|
| `test_after_hours_plus_loitering_both_fire` | night + 120s loitering | both after_hours_person AND loitering fire |
| `test_all_hits_have_required_fields` | complex multi-rule scenario | every hit has rule_name, severity, message, needs_llm with correct types |

The `test_all_hits_have_required_fields` test validates the schema of every hit:
```python
for hit in hits:
    assert "rule_name" in hit
    assert "severity" in hit
    assert "message" in hit
    assert "needs_llm" in hit
    assert hit["severity"] in ("low", "medium", "high")
    assert isinstance(hit["needs_llm"], bool)
    assert isinstance(hit["message"], str) and len(hit["message"]) > 0
```

### Scenario 7: Edge Cases (3 tests)

| Test | Input | Asserts |
|---|---|---|
| `test_no_detections_no_alerts` | empty detections at 23:00 | no detection-based rules fire |
| `test_empty_caption_does_not_crash` | caption="" | returns list (no exception) |
| `test_none_caption_does_not_crash` | caption=None | returns list (no exception) |

---

## Coverage Summary

Approximate coverage per module after full test run:

| Module | Coverage | Notes |
|---|---|---|
| `src/agent/rule_engine.py` | ~85% | All condition types exercised |
| `src/agent/nodes.py` | ~75% | LLM judge paths exercised via mock |
| `src/agent/graph.py` | ~96% | Both routing paths covered |
| `src/agent/state.py` | 100% | TypedDict, no runtime logic |
| `src/memory/sqlite_store.py` | ~78% | All public methods tested |
| `src/memory/hybrid_retriever.py` | ~55% | semantic_search path requires ChromaDB vectors |
| `src/memory/chroma_store.py` | ~35% | Zero embeddings in stubs; ChromaDB internals untested |
| `src/perception/frame_preprocessor.py` | ~95% | All three output views tested |
| `src/pipeline/stream_processor.py` | ~89% | All queue modes and stop conditions tested |

---

## Key Testing Patterns

### 1. Inline YAML rules for unit isolation

Rather than depending on `configs/rules.yaml`, unit tests write minimal rules to `tmp_path`:

```python
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

def test_triggers_at_night(self, tmp_path, ts_night):
    engine = RuleEngine(write_rules_yaml(tmp_path, AFTER_HOURS_RULE))
    hits = engine.evaluate(ts=ts_night, zone="gate",
                           detections=[make_detection("person")], ...)
    assert len(hits) == 1
```

This ensures unit tests are immune to production rule changes.

### 2. Scenario tests use production rules as living docs

`tests/scenarios/` intentionally loads `configs/rules.yaml`. If a rule threshold changes (e.g., loitering from 60s to 120s), the test `test_loitering_fires_after_threshold` will fail, prompting the developer to either update the test or reconsider the change.

### 3. Stub models for integration tests

Heavy models (YOLO ~25MB, CLIP ~300MB, VLM ~2GB) are replaced by stubs that return deterministic outputs in microseconds. This keeps the integration test suite under 20 seconds on any laptop with no GPU.

### 4. Temporal loitering test pattern

The `contextualize` node computes track durations from its internal `_track_first_seen` closure — it overwrites any `track_durations` in the input state. Tests that need a loitering scenario must invoke the agent twice:

```python
agent.invoke({"preprocessed": frame_at_T, "zone": "gate"})      # seeds track
result = agent.invoke({"preprocessed": frame_at_T_plus_65s, ...})  # 65s elapsed -> loitering
```

### 5. ChatBedrock mock must wrap build_agent

`ChatBedrock` is instantiated at graph-build time inside `make_llm_judge_node()`. The patch must be active before `build_agent()` is called:

```python
with patch("src.agent.nodes.ChatBedrock") as MockLLM:
    MockLLM.return_value.invoke.return_value = fake_response
    agent = build_agent(...)   # <- mock active here
    result = agent.invoke(...) # <- mock used during invocation
```

### 6. In-memory SQLite connection caching

`SQLiteStore(":memory:")` caches a single connection in `self._mem_conn`. Standard `sqlite3.connect(":memory:")` creates a fresh empty database per call — using a new connection for each query would silently discard all data between calls.

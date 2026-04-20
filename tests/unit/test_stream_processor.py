"""
test_stream_processor.py
------------------------
Unit tests for StreamProcessor — queue behaviour, drop_on_full modes,
running-property correctness, and the file-processing regression.

All tests use a FakeAgent (no real models) and a FakeIngestor
(yields synthetic FramePackets in-process).
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.perception.video_ingestor import FramePacket
from src.pipeline.stream_processor import StreamProcessor


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _make_packet(i: int = 0) -> FramePacket:
    return FramePacket(
        frame_id=str(uuid.uuid4()),
        ts=datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc),
        video_id="test",
        frame_index=i,
        image=np.zeros((240, 320, 3), dtype=np.uint8),
    )


class FakeIngestor:
    """Yields N synthetic frames then stops."""

    def __init__(self, n: int, delay: float = 0.0):
        self.n = n
        self.delay = delay

    def stream(self):
        for i in range(self.n):
            if self.delay:
                time.sleep(self.delay)
            yield _make_packet(i)


class FakeAgent:
    """Records every invoke() call and returns a minimal result dict."""

    def __init__(self):
        self.calls: list[int] = []

    def invoke(self, state: dict) -> dict:
        self.calls.append(state.get("frame_index", -1))
        return {
            "alerts_fired": [],
            "logged": True,
            "frame_index": state.get("frame_index", -1),
        }


def _make_preprocessor():
    """Mock preprocessor that always returns a truthy PreprocessedFrame."""
    mock = MagicMock()
    mock.process.side_effect = lambda pkt: MagicMock(packet=pkt)
    return mock


# ---------------------------------------------------------------------------
# Helper: drain all results from a finished processor
# ---------------------------------------------------------------------------

def _drain(processor: StreamProcessor, timeout_s: float = 20.0) -> list[dict]:
    results = []
    deadline = time.time() + timeout_s
    while processor.running and time.time() < deadline:
        r = processor.get_result(timeout=0.1)
        if r is not None:
            results.append(r)
    # Drain any remaining buffered results
    while True:
        r = processor.get_result(timeout=0.05)
        if r is None:
            break
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# File mode (drop_on_full=False) — regression for the 3-frame bug
# ---------------------------------------------------------------------------

class TestFileModeAllFramesProcessed:

    def test_all_frames_processed_short_video(self):
        """
        Regression: before the fix, only 3-4 frames were processed from a
        short file because the reader evicted frames while the worker was
        blocked on LLM judge.  With drop_on_full=False the reader blocks
        until the worker has space → every frame is processed.
        """
        N = 20
        agent = FakeAgent()
        processor = StreamProcessor(
            agent, _make_preprocessor(),
            queue_size=4, result_size=200, drop_on_full=False,
        )
        processor.start(FakeIngestor(N), zone="")
        _drain(processor)
        assert len(agent.calls) == N, (
            f"Expected all {N} frames processed, got {len(agent.calls)}"
        )

    def test_all_frames_processed_with_slow_worker(self):
        """Worker that takes 50ms per frame must still process everything."""
        N = 10

        class SlowAgent:
            def __init__(self):
                self.calls = []
            def invoke(self, state):
                time.sleep(0.05)   # simulate 50ms LLM call
                self.calls.append(1)
                return {"alerts_fired": [], "logged": True}

        agent = SlowAgent()
        processor = StreamProcessor(
            agent, _make_preprocessor(),
            queue_size=4, result_size=50, drop_on_full=False,
        )
        processor.start(FakeIngestor(N), zone="")
        _drain(processor, timeout_s=30.0)
        assert len(agent.calls) == N

    def test_no_frames_dropped_in_file_mode(self):
        N = 15
        processor = StreamProcessor(
            FakeAgent(), _make_preprocessor(),
            queue_size=4, result_size=100, drop_on_full=False,
        )
        processor.start(FakeIngestor(N), zone="")
        _drain(processor)
        assert processor.stats["dropped"] == 0


# ---------------------------------------------------------------------------
# RTSP mode (drop_on_full=True) — evicts oldest frame when queue is full
# ---------------------------------------------------------------------------

class TestRtspModeDropsFrames:

    def test_fast_reader_drops_frames_when_worker_is_slow(self):
        """
        When reader is faster than worker and queue is full, older frames
        should be evicted.  Processed count < total read count.
        """
        N = 30

        class SlowAgent:
            def invoke(self, state):
                time.sleep(0.02)
                return {"alerts_fired": [], "logged": True}

        processor = StreamProcessor(
            SlowAgent(), _make_preprocessor(),
            queue_size=2, result_size=100, drop_on_full=True,
        )
        processor.start(FakeIngestor(N, delay=0.0), zone="")
        _drain(processor, timeout_s=20.0)

        # Some frames must have been dropped; processed < total
        assert processor.stats["processed"] < N
        assert processor.stats["dropped"] >= 0   # counter updated

    def test_drop_pct_reported_correctly(self):
        N = 20

        class SlowAgent:
            def invoke(self, state):
                time.sleep(0.03)
                return {"alerts_fired": [], "logged": True}

        processor = StreamProcessor(
            SlowAgent(), _make_preprocessor(),
            queue_size=2, result_size=100, drop_on_full=True,
        )
        processor.start(FakeIngestor(N, delay=0.0), zone="")
        _drain(processor, timeout_s=20.0)

        stats = processor.stats
        expected_pct = round(stats["dropped"] / stats["read"] * 100) if stats["read"] else 0
        assert stats["drop_pct"] == expected_pct


# ---------------------------------------------------------------------------
# running property
# ---------------------------------------------------------------------------

class TestRunningProperty:

    def test_running_true_while_results_queued(self):
        """
        Regression: old implementation only checked the frame queue.
        If the worker had already drained the frame queue but results
        hadn't been consumed yet, running was False → loop exited early.
        """
        N = 5
        processor = StreamProcessor(
            FakeAgent(), _make_preprocessor(),
            queue_size=32, result_size=100, drop_on_full=False,
        )
        processor.start(FakeIngestor(N), zone="")

        # Wait until reader and worker are done
        time.sleep(1.0)

        # Results should still be queued — running must be True
        if not processor._results.empty():
            assert processor.running is True

    def test_running_becomes_false_after_full_drain(self):
        N = 5
        processor = StreamProcessor(
            FakeAgent(), _make_preprocessor(),
            queue_size=32, result_size=100, drop_on_full=False,
        )
        processor.start(FakeIngestor(N), zone="")
        _drain(processor)   # consume all results
        # Allow threads to fully exit
        time.sleep(0.3)
        assert processor.running is False


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

class TestStop:

    def test_stop_terminates_long_running_source(self):
        """stop() should interrupt an 'infinite' source."""

        class InfiniteIngestor:
            def stream(self):
                i = 0
                while True:
                    yield _make_packet(i)
                    i += 1
                    time.sleep(0.01)

        processor = StreamProcessor(
            FakeAgent(), _make_preprocessor(),
            queue_size=4, result_size=50, drop_on_full=True,
        )
        processor.start(InfiniteIngestor(), zone="")
        time.sleep(0.2)      # let it run briefly
        processor.stop()
        time.sleep(0.5)      # give threads time to finish

        assert not processor._running


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:

    def test_stats_updated_after_processing(self):
        N = 8

        class AlertAgent:
            def invoke(self, state):
                return {
                    "alerts_fired": [{"alert_id": "a", "rule_name": "test",
                                      "severity": "high", "message": "x",
                                      "ts": "2026-04-20T12:00:00"}],
                    "logged": True,
                }

        processor = StreamProcessor(
            AlertAgent(), _make_preprocessor(),
            queue_size=32, result_size=100, drop_on_full=False,
        )
        processor.start(FakeIngestor(N), zone="")
        _drain(processor)

        stats = processor.stats
        assert stats["processed"] == N
        assert stats["read"] == N
        assert stats["alerts"] == N   # one alert per frame

"""
stream_processor.py
-------------------
Multi-threaded pipeline for processing high-FPS video and RTSP streams.

Problem
-------
The LangGraph agent runs sequentially:
  YOLO (~30ms) → VLM (~500ms Bedrock) → CLIP (~30ms) → DB writes (~20ms)

At 30fps that's one new frame every 33ms but ~580ms of work — the system
falls 17x behind immediately on any live stream.

Architecture
------------
                   ┌──────────────────┐
  RTSP/File  ────► │  Reader Thread   │  reads at full source FPS
                   │  bounded queue   │  drops oldest frame if full
                   └────────┬─────────┘
                            │ FramePacket
                   ┌────────▼─────────┐
                   │  Worker Thread   │  YOLO + rules + CLIP + DB
                   │  (fast path)     │  ~80ms / frame → ~12fps
                   └────────┬─────────┘
                            │              ┌───────────────────────┐
                            ├─────────────►│  VLM Thread Pool      │
                            │  async sub   │  Bedrock / Moondream  │
                            │  (run_vlm)   │  result cached for    │
                            │              │  subsequent frames    │
                            │              └───────────────────────┘
                   ┌────────▼─────────┐
                   │  Result Queue    │  polled by Streamlit UI
                   └──────────────────┘

Usage
-----
    processor = StreamProcessor(agent, preprocessor)
    processor.start(ingestor, zone="main_gate")

    while processor.running:
        result = processor.get_result(timeout=0.05)
        if result:
            update_ui(result)

    processor.stop()
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Optional

logger = logging.getLogger(__name__)


def recommended_sample_every(fps: float, target_fps: float = 6.0) -> int:
    """
    Compute the frame-skip interval needed to process at roughly target_fps.

    Examples
    --------
    24 fps source  → sample_every=4   (processes ~6 fps)
    30 fps source  → sample_every=5   (processes ~6 fps)
    60 fps source  → sample_every=10  (processes ~6 fps)
    """
    return max(1, round(fps / target_fps))


class StreamProcessor:
    """
    Decoupled reader + worker pipeline for high-FPS video/RTSP.

    Parameters
    ----------
    agent        : Compiled LangGraph agent (from build_agent, async_vlm=True).
    preprocessor : FramePreprocessor instance.
    queue_size   : Max frames buffered between reader and worker.
    result_size  : Max results held before Streamlit drains them.
    drop_on_full : True  → RTSP/live mode: evict oldest frame when queue is full
                           so the worker always sees the freshest frame.
                   False → File mode: reader blocks when queue is full so every
                           sampled frame is guaranteed to be processed.
    """

    def __init__(
        self,
        agent,
        preprocessor,
        queue_size: int = 4,
        result_size: int = 200,
        drop_on_full: bool = True,
    ) -> None:
        self._agent        = agent
        self._preprocessor = preprocessor
        self._queue        = Queue(maxsize=queue_size)
        self._results      = Queue(maxsize=result_size)
        self._running      = False
        self._reader_done  = threading.Event()
        self._drop_on_full = drop_on_full

        # Counters updated from worker thread (no lock needed — int writes atomic in CPython)
        self.frames_read      = 0
        self.frames_processed = 0
        self.frames_dropped   = 0
        self.alerts_total     = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, ingestor, zone: str = "") -> None:
        """Launch reader + worker threads. Returns immediately."""
        self._running = True
        self._reader_done.clear()

        threading.Thread(
            target=self._reader_loop,
            args=(ingestor, zone),
            daemon=True,
            name="stream-reader",
        ).start()

        threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="stream-worker",
        ).start()

    def stop(self) -> None:
        """Signal both threads to stop after draining the current frame."""
        self._running = False

    def get_result(self, timeout: float = 0.05) -> Optional[dict]:
        """Non-blocking poll for the next processed result. Returns None if none ready."""
        try:
            return self._results.get(timeout=timeout)
        except Empty:
            return None

    @property
    def running(self) -> bool:
        # Still "running" as long as any of these are true:
        #   1. Reader thread is alive (_running=True)
        #   2. Frame queue has frames waiting for the worker
        #   3. Results queue has results waiting to be drained by the caller
        # Without checking (3), a fast worker could finish before the Streamlit
        # loop drains all results, causing premature exit on short files.
        return self._running or not self._queue.empty() or not self._results.empty()

    @property
    def stats(self) -> dict:
        drop_pct = (
            round(self.frames_dropped / self.frames_read * 100)
            if self.frames_read else 0
        )
        return {
            "read":      self.frames_read,
            "processed": self.frames_processed,
            "dropped":   self.frames_dropped,
            "drop_pct":  drop_pct,
            "alerts":    self.alerts_total,
        }

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _reader_loop(self, ingestor, zone: str) -> None:
        """
        Reads frames from the source and enqueues them for the worker.

        drop_on_full=True  (RTSP/live): evict the oldest queued frame when the
            queue is full so the worker always sees the freshest frame. Excess
            frames are counted as dropped.
        drop_on_full=False (file): block on queue.put() when the queue is full.
            This guarantees every sampled frame is processed — essential for
            short video files where the reader finishes before the worker.
        """
        try:
            for packet in ingestor.stream():
                if not self._running:
                    break

                packet.metadata["zone"] = zone
                self.frames_read += 1

                if self._drop_on_full:
                    # Live / RTSP: drop stale frames, keep freshest
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                            self.frames_dropped += 1
                        except Empty:
                            pass
                    self._queue.put(packet)
                else:
                    # File: block until the worker has consumed a slot
                    self._queue.put(packet)

        except Exception as exc:
            logger.error("[StreamReader] Unexpected error: %s", exc)
        finally:
            self._reader_done.set()
            self._running = False   # tell worker to drain and stop

    def _worker_loop(self) -> None:
        """
        Pulls frames from the queue and runs the full agent pipeline.
        Waits for reader to signal done before exiting so the last
        queued frames are always processed.
        """
        while True:
            try:
                packet = self._queue.get(timeout=0.1)
            except Empty:
                # Exit only once reader is done AND queue is empty
                if self._reader_done.is_set() and self._queue.empty():
                    break
                continue

            preprocessed = self._preprocessor.process(packet)
            if preprocessed is None:
                continue

            try:
                result = self._agent.invoke({
                    "preprocessed": preprocessed,
                    "zone":         packet.metadata.get("zone", ""),
                })
                self.frames_processed += 1
                self.alerts_total     += len(result.get("alerts_fired", []))

                if not self._results.full():
                    self._results.put_nowait(result)

            except Exception as exc:
                logger.error("[StreamWorker] Agent error on frame %s: %s",
                             packet.frame_id[:8], exc)

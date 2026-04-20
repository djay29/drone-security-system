"""
video_ingestor.py
-----------------
Ingests drone video from three sources:
  1. Local video file (MP4, AVI, MOV, etc.)
  2. RTSP stream (live drone feed)
  3. Cached description mode — reads frames.jsonl for CPU-only / no-GPU reviewers

Yields FramePacket objects consumed by the perception pipeline.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterator

import cv2
import numpy as np

from src.telemetry.simulator import TelemetrySimulator

# ---------------------------------------------------------------------------
# Internal data contract between ingestor and perception layer
# ---------------------------------------------------------------------------

@dataclass
class FramePacket:
    """Minimal unit passed downstream to the perception layer."""

    frame_id: str
    """Unique ID for this frame (UUID4)."""

    ts: datetime
    """Wall-clock timestamp when the frame was captured / read."""

    video_id: str
    """Identifier for the source (filename stem or stream URL hash)."""

    frame_index: int
    """0-based sequential index within the source."""

    image: np.ndarray | None
    """BGR uint8 ndarray (H×W×3). None when running in text-fallback mode."""

    frame_path: Path | None = None
    """Path to the saved frame on disk (set after optional persistence)."""

    description: str | None = None
    """Pre-baked text description — populated in fallback mode instead of image."""

    metadata: dict = field(default_factory=dict)
    """Catch-all for extra fields (e.g., original fps, source type)."""


# ---------------------------------------------------------------------------
# Source enum + config
# ---------------------------------------------------------------------------

class SourceType:
    FILE = "file"
    RTSP = "rtsp"
    FALLBACK = "fallback"   # text descriptions from .jsonl


# ---------------------------------------------------------------------------
# VideoIngestor
# ---------------------------------------------------------------------------

class VideoIngestor:
    """
    Thin wrapper around OpenCV VideoCapture with frame-skipping, optional
    frame persistence to disk, and a text-fallback mode.

    Usage
    -----
    ingestor = VideoIngestor.from_file("data/videos/patrol.mp4", sample_every=5)
    for packet in ingestor.stream():
        ...

    ingestor = VideoIngestor.from_fallback("data/frames/frames.jsonl")
    for packet in ingestor.stream():
        ...
    """

    def __init__(
        self,
        source: str | Path,
        source_type: str,
        video_id: str | None = None,
        sample_every: int = 5,
        max_frames: int | None = None,
        save_frames: bool = False,
        save_dir: Path | None = None,
        frame_size: tuple[int, int] | None = None,
        start_ts: datetime | None = None,
        realtime_sleep: bool = False,
        save_size: tuple[int, int] | None = (640, 360),
        save_quality: int = 75,
        telemetry_simulator: "TelemetrySimulator | None" = None,
    ) -> None:
        """
        Parameters
        ----------
        source        : Path to file / RTSP URL / path to .jsonl
        source_type   : SourceType constant
        video_id      : Human-readable ID; defaults to filename stem
        sample_every  : Emit every Nth frame (1 = every frame)
        max_frames    : Hard stop after N emitted frames (None = unlimited)
        save_frames   : Write each emitted frame as JPEG to save_dir
        save_dir      : Destination for saved frames
        frame_size    : (width, height) to resize the *emitted* frame (affects all
                        downstream models). None = native resolution.
        start_ts      : Treat this as the timestamp of frame 0
        realtime_sleep: For file sources, sleep between frames to mimic real fps
        save_size     : (width, height) to resize before writing the JPEG thumbnail.
                        Applied only at save time — model inputs are unaffected.
                        Default (640, 360) ≈ 360p; None = save at emitted resolution.
        save_quality  : JPEG quality for saved thumbnails (0-100). Default 75 gives
                        good visual quality at ~40% smaller files than quality 90.
        """
        self.source = Path(source) if source_type != SourceType.RTSP else source
        self.source_type = source_type
        self.video_id = video_id or _derive_video_id(source)
        self.sample_every = max(1, sample_every)
        self.max_frames = max_frames
        self.save_frames = save_frames
        self.save_dir = save_dir or Path("data/frames") / self.video_id
        self.frame_size = frame_size
        self.start_ts = start_ts or datetime.now(timezone.utc)
        self.realtime_sleep = realtime_sleep
        self.save_size = save_size
        self.save_quality = max(1, min(100, save_quality))
        self.telemetry_simulator = telemetry_simulator

        # Populated on first use
        self._cap: cv2.VideoCapture | None = None
        self._native_fps: float = 30.0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        sample_every: int = 5,
        max_frames: int | None = None,
        save_frames: bool = False,
        save_dir: Path | None = None,
        frame_size: tuple[int, int] | None = None,
        start_ts: datetime | None = None,
        save_size: tuple[int, int] | None = (640, 360),
        save_quality: int = 75,
        telemetry_simulator: "TelemetrySimulator | None" = None,
    ) -> "VideoIngestor":
        return cls(
            source=path,
            source_type=SourceType.FILE,
            sample_every=sample_every,
            max_frames=max_frames,
            save_frames=save_frames,
            save_dir=save_dir,
            frame_size=frame_size,
            start_ts=start_ts,
            save_size=save_size,
            save_quality=save_quality,
            telemetry_simulator=telemetry_simulator,
        )

    @classmethod
    def from_rtsp(
        cls,
        url: str,
        sample_every: int = 5,
        max_frames: int | None = None,
        save_frames: bool = False,
        save_dir: Path | None = None,
        frame_size: tuple[int, int] | None = None,
        save_size: tuple[int, int] | None = (640, 360),
        save_quality: int = 75,
        telemetry_simulator: "TelemetrySimulator | None" = None,
    ) -> "VideoIngestor":
        return cls(
            source=url,
            source_type=SourceType.RTSP,
            sample_every=sample_every,
            max_frames=max_frames,
            save_frames=save_frames,
            save_dir=save_dir,
            frame_size=frame_size,
            realtime_sleep=False,
            save_size=save_size,
            save_quality=save_quality,
            telemetry_simulator=telemetry_simulator,
        )

    @classmethod
    def from_fallback(
        cls,
        jsonl_path: str | Path,
        start_ts: datetime | None = None,
    ) -> "VideoIngestor":
        """Text-description fallback for CPU-only / no-GPU environments."""
        return cls(
            source=jsonl_path,
            source_type=SourceType.FALLBACK,
            start_ts=start_ts,
        )

    # ------------------------------------------------------------------
    # Main streaming generator
    # ------------------------------------------------------------------

    def stream(self) -> Generator[FramePacket, None, None]:
        """Yield FramePackets until source is exhausted or max_frames hit."""
        if self.source_type == SourceType.FALLBACK:
            yield from self._stream_fallback()
        else:
            yield from self._stream_video()

    def stream_latest(self) -> Generator[FramePacket, None, None]:
        """
        Always yield the most recent available frame.

        A background thread reads from the source at full speed and
        overwrites a single shared slot. The generator yields whatever
        is in that slot when polled — intermediate frames are silently
        discarded so the caller never falls behind a fast source.

        Use this instead of stream() for RTSP or any high-FPS source.
        """
        import threading as _th

        _latest: list = [None]
        _stop         = _th.Event()
        _has_frame    = _th.Event()

        def _reader() -> None:
            try:
                for packet in self._stream_video():
                    if _stop.is_set():
                        break
                    _latest[0] = packet
                    _has_frame.set()
            finally:
                _stop.set()
                _has_frame.set()   # wake the generator so it can exit cleanly

        t = _th.Thread(target=_reader, daemon=True, name="ingestor-reader")
        t.start()

        try:
            while not (_stop.is_set() and _latest[0] is None):
                _has_frame.wait(timeout=1.0)
                _has_frame.clear()
                frame = _latest[0]
                if frame is None:
                    continue
                _latest[0] = None
                yield frame
        finally:
            _stop.set()
            t.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Video stream (file + RTSP)
    # ------------------------------------------------------------------

    def _stream_video(self) -> Generator[FramePacket, None, None]:
        cap = self._open_capture()
        self._native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration_s = 1.0 / self._native_fps

        raw_index = 0       # every frame read from source
        emitted_count = 0   # frames actually yielded

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # EOF or stream dropped

                # Frame sampling — skip non-sampled frames cheaply
                if raw_index % self.sample_every != 0:
                    raw_index += 1
                    continue

                # Optional resize
                if self.frame_size:
                    frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)

                # Compute timestamp
                ts = _frame_ts(self.start_ts, raw_index, self._native_fps)

                telemetry = (
                    self.telemetry_simulator.get_telemetry(ts)
                    if self.telemetry_simulator is not None
                    else None
                )
                packet = FramePacket(
                    frame_id=str(uuid.uuid4()),
                    ts=ts,
                    video_id=self.video_id,
                    frame_index=emitted_count,
                    image=frame,
                    metadata={
                        "source_type":    self.source_type,
                        "raw_frame_index": raw_index,
                        "native_fps":     self._native_fps,
                        "telemetry":      telemetry,
                    },
                )

                # Optional disk persistence
                if self.save_frames:
                    packet.frame_path = self._save_frame(packet)

                yield packet
                emitted_count += 1

                if self.max_frames and emitted_count >= self.max_frames:
                    break

                if self.realtime_sleep:
                    time.sleep(frame_duration_s * self.sample_every)

                raw_index += 1

        finally:
            cap.release()

    def _open_capture(self) -> cv2.VideoCapture:
        source = str(self.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"[VideoIngestor] Cannot open source: {source!r}. "
                "Check path / RTSP URL and codec support."
            )
        return cap

    # ------------------------------------------------------------------
    # Text-fallback stream
    # ------------------------------------------------------------------

    def _stream_fallback(self) -> Generator[FramePacket, None, None]:
        """
        Reads a .jsonl file where each line is a JSON object with at least:
          { "description": "...", "ts": "ISO8601 or omit", "zone": "..." }
        Optional fields passed through as metadata.
        """
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"[VideoIngestor] Fallback file not found: {path}")

        with path.open() as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue

                record: dict = json.loads(line)
                ts_raw = record.get("ts")
                ts = (
                    datetime.fromisoformat(ts_raw)
                    if ts_raw
                    else _frame_ts(self.start_ts, idx, fps=1.0)
                )

                yield FramePacket(
                    frame_id=record.get("frame_id", str(uuid.uuid4())),
                    ts=ts,
                    video_id=self.video_id,
                    frame_index=idx,
                    image=None,
                    description=record.get("description", ""),
                    metadata={k: v for k, v in record.items()
                               if k not in ("frame_id", "ts", "description")},
                )

                if self.max_frames and (idx + 1) >= self.max_frames:
                    break

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _save_frame(self, packet: FramePacket) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{packet.frame_index:06d}_{packet.frame_id[:8]}.jpg"
        out_path = self.save_dir / fname

        img = packet.image
        if self.save_size is not None:
            # Resize only the saved thumbnail — does NOT affect model inputs
            img = cv2.resize(img, self.save_size, interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, self.save_quality])
        return out_path

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int | None:
        """Total frame count of the source (None for streams / fallback)."""
        if self.source_type != SourceType.FILE:
            return None
        cap = cv2.VideoCapture(str(self.source))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None

    @property
    def native_fps(self) -> float | None:
        if self.source_type != SourceType.FILE:
            return None
        cap = cv2.VideoCapture(str(self.source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps or None

    def __repr__(self) -> str:
        return (
            f"VideoIngestor(source={self.source!r}, "
            f"type={self.source_type}, "
            f"sample_every={self.sample_every})"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _derive_video_id(source: str | Path) -> str:
    p = Path(str(source))
    # For RTSP URLs, hash the URL into a short ID
    if str(source).startswith("rtsp://"):
        import hashlib
        return "rtsp_" + hashlib.md5(str(source).encode()).hexdigest()[:8]
    return p.stem


def _frame_ts(start_ts: datetime, frame_index: int, fps: float) -> datetime:
    """Compute wall-clock timestamp for a frame given its index and fps."""
    from datetime import timedelta
    offset_s = frame_index / fps
    return start_ts + timedelta(seconds=offset_s)

"""
yolo_detector.py
----------------
Wraps yolov8s (Ultralytics) to detect and optionally track objects in
pre-processed drone frames.

Consumes:  PreprocessedFrame.yolo_input  (640×640 BGR uint8 ndarray)
Produces:  list[DetectedObject]          (Pydantic model from section 7)

Design notes
------------
- Model is loaded once on first call (lazy) and cached on the instance.
- Tracking (ByteTrack) is opt-in; disable for static analysis / tests.
- Only security-relevant COCO classes are forwarded by default; the
  filter list is configurable so new classes need no code changes.
- bbox coordinates are un-letterboxed back to original image space using
  the meta dict from FramePreprocessor.
- Fallback / text-only packets (yolo_input=None) return [] immediately.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

# Pydantic model (from context sheet section 7)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical data model (mirrors context sheet section 7)
# ---------------------------------------------------------------------------

class DetectedObject(BaseModel):
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in original image coords
    track_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Default security-relevant COCO class filter
# ---------------------------------------------------------------------------

DEFAULT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "boat",
    # Add more as needed — matched against YOLO's own label names
}


# ---------------------------------------------------------------------------
# YOLODetector
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    Parameters
    ----------
    model_name      : Ultralytics model identifier or path to custom weights.
                      'yolov8s.pt' downloads automatically on first run (~6 MB).
    confidence      : Minimum confidence threshold (0–1).
    iou_threshold   : NMS IoU threshold.
    use_tracking    : Enable ByteTrack cross-frame tracking.
    allowed_classes : Set of class names to forward; None = all classes.
    device          : 'cpu', 'cuda', 'mps', or None (auto-detect).
    """

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence: float = 0.4,
        iou_threshold: float = 0.45,
        use_tracking: bool = True,
        allowed_classes: Optional[set[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.use_tracking = use_tracking
        self.allowed_classes = allowed_classes if allowed_classes is not None else DEFAULT_CLASSES
        self.device = device or _auto_device()

        self._model = None   # lazy-loaded on first detect() call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        yolo_input: np.ndarray,
        letterbox_meta: Optional[dict] = None,
    ) -> list[DetectedObject]:
        """
        Run detection (+ optional tracking) on a pre-processed frame.

        Parameters
        ----------
        yolo_input      : Letterboxed BGR uint8 ndarray (H×W×3).
                          Typically 640×640 from FramePreprocessor.
        letterbox_meta  : Dict from FramePreprocessor with scale/padding info.
                          If None, bbox coords are returned in letterboxed space.

        Returns
        -------
        List of DetectedObject (may be empty).
        """
        if yolo_input is None:
            return []

        model = self._load_model()

        if self.use_tracking:
            results = model.track(
                source=yolo_input,
                conf=self.confidence,
                iou=self.iou_threshold,
                persist=True,          # keep tracker state across calls
                verbose=False,
                device=self.device,
            )
        else:
            results = model.predict(
                source=yolo_input,
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
            )

        return self._parse_results(results, letterbox_meta)

    def detect_from_preprocessed(self, preprocessed_frame) -> list[DetectedObject]:
        """
        Convenience wrapper that accepts a PreprocessedFrame directly.

        Parameters
        ----------
        preprocessed_frame : PreprocessedFrame from FramePreprocessor
        """
        if preprocessed_frame.yolo_input is None:
            return []
        return self.detect(
            yolo_input=preprocessed_frame.yolo_input,
            letterbox_meta=preprocessed_frame.letterbox_meta or None,
        )

    def warmup(self) -> None:
        """Force model load + one dummy inference. Call once at startup."""
        logger.info("[YOLODetector] Warming up model: %s", self.model_name)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("[YOLODetector] Warmup done.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO  # deferred import — keeps module importable without ultralytics
            logger.info("[YOLODetector] Loading model: %s on %s", self.model_name, self.device)
            self._model = YOLO(self.model_name)
        return self._model

    def _parse_results(self, results, letterbox_meta: Optional[dict]) -> list[DetectedObject]:
        detections: list[DetectedObject] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            names = result.names  # {0: 'person', 1: 'bicycle', ...}

            for i in range(len(boxes)):
                cls_id     = int(boxes.cls[i].item())
                class_name = names.get(cls_id, str(cls_id))

                # Class filter
                if self.allowed_classes and class_name not in self.allowed_classes:
                    continue

                conf = float(boxes.conf[i].item())

                # BBox in letterboxed space (xyxy)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # Un-letterbox back to original image coordinates
                if letterbox_meta:
                    x1, y1, x2, y2 = _unletterbox(x1, y1, x2, y2, letterbox_meta)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Track ID (None when tracking disabled or not yet assigned)
                track_id = None
                if boxes.id is not None:
                    raw_id = boxes.id[i]
                    if raw_id is not None:
                        track_id = int(raw_id.item())

                detections.append(
                    DetectedObject(
                        class_name=class_name,
                        confidence=round(conf, 3),
                        bbox=(x1, y1, x2, y2),
                        track_id=track_id,
                    )
                )

        return detections


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    """Pick best available device without requiring torch at import time."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _unletterbox(
    x1: float, y1: float, x2: float, y2: float,
    meta: dict,
) -> tuple[int, int, int, int]:
    """Map bbox from letterboxed canvas back to original image coordinates."""
    scale    = meta["scale"]
    pad_top  = meta["pad_top"]
    pad_left = meta["pad_left"]
    orig_h   = meta["orig_h"]
    orig_w   = meta["orig_w"]

    ox1 = int(np.clip((x1 - pad_left) / scale, 0, orig_w))
    oy1 = int(np.clip((y1 - pad_top)  / scale, 0, orig_h))
    ox2 = int(np.clip((x2 - pad_left) / scale, 0, orig_w))
    oy2 = int(np.clip((y2 - pad_top)  / scale, 0, orig_h))

    return ox1, oy1, ox2, oy2
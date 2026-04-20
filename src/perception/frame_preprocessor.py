"""
frame_preprocessor.py
---------------------
Normalizes raw FramePackets coming from VideoIngestor into two
ready-to-use tensor/array representations:

  1. **YOLO input**   — letterboxed BGR ndarray, uint8, (640, 640, 3)
  2. **VLM input**    — RGB PIL Image, resized to model's expected resolution

Also validates packets and decides whether to skip VLM captioning based
on the configured stride (caption every Nth emitted frame).

Nothing in this module writes to disk or calls any model.
It is a pure transformation layer — easy to test, easy to swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .video_ingestor import FramePacket


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedFrame:
    """
    Downstream consumers (YOLODetector, VLMCaptioner, CLIPEmbedder)
    each pull their slice from this object.
    """

    packet: FramePacket
    """Original packet — full provenance preserved."""

    yolo_input: np.ndarray | None
    """
    Letterboxed BGR uint8 ndarray ready for YOLOv8.
    Shape: (yolo_size, yolo_size, 3).
    None in fallback / text-only mode.
    """

    vlm_input: Image.Image | None
    """
    RGB PIL Image resized for the VLM (Moondream2 default: 378×378).
    None if this frame is skipped for captioning (stride logic) or
    running in text-only mode.
    """

    clip_input: np.ndarray | None
    """
    RGB float32 ndarray normalized to [0, 1] for CLIP embedding.
    Shape: (224, 224, 3). None in text-only mode.
    """

    run_vlm: bool = True
    """
    False → skip VLM captioning this frame (stride / cost control).
    The downstream captioner checks this flag before calling the model.
    """

    letterbox_meta: dict = field(default_factory=dict)
    """
    Stores scale + padding offsets so detections can be un-letterboxed
    back to original image coordinates.
    Keys: scale, pad_top, pad_left, orig_h, orig_w
    """


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class FramePreprocessor:
    """
    Stateless-ish transformer.  Only stateful thing is the VLM-stride counter.

    Parameters
    ----------
    yolo_size       : Square side length for YOLO input (default 640).
    vlm_size        : (width, height) for VLM input. Moondream2 = (378, 378).
    clip_size       : Square side length for CLIP (always 224).
    vlm_every       : Run VLM on every Nth emitted frame (1 = every frame).
    skip_invalid    : Return None for packets with no image and no description
                      instead of raising.
    """

    YOLO_DEFAULT   = 640
    VLM_DEFAULT    = (378, 378)
    CLIP_DEFAULT   = 224

    def __init__(
        self,
        yolo_size: int = YOLO_DEFAULT,
        vlm_size: tuple[int, int] = VLM_DEFAULT,
        clip_size: int = CLIP_DEFAULT,
        vlm_every: int = 5,
        skip_invalid: bool = True,
    ) -> None:
        self.yolo_size = yolo_size
        self.vlm_size  = vlm_size
        self.clip_size = clip_size
        self.vlm_every = max(1, vlm_every)
        self.skip_invalid = skip_invalid
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, packet: FramePacket) -> Optional[PreprocessedFrame]:
        """
        Transform a single FramePacket into a PreprocessedFrame.
        Returns None if packet is invalid and skip_invalid=True.
        """
        # --- Fallback / text-only mode ---
        if packet.image is None:
            if not packet.description:
                if self.skip_invalid:
                    return None
                raise ValueError(
                    f"[Preprocessor] Packet {packet.frame_id} has neither "
                    "image nor description."
                )
            # Text-only — no pixel ops needed
            self._frame_counter += 1
            return PreprocessedFrame(
                packet=packet,
                yolo_input=None,
                vlm_input=None,
                clip_input=None,
                run_vlm=False,
            )

        # --- Pixel mode ---
        image_bgr: np.ndarray = packet.image

        # Validate shape
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            if self.skip_invalid:
                return None
            raise ValueError(
                f"[Preprocessor] Unexpected image shape: {image_bgr.shape}"
            )

        # 1. YOLO input
        yolo_input, lb_meta = _letterbox(image_bgr, self.yolo_size)

        # 2. CLIP input  (RGB float32 [0,1])
        clip_input = _resize_rgb_float(image_bgr, self.clip_size)

        # 3. VLM input (RGB PIL, only on stride)
        run_vlm = (self._frame_counter % self.vlm_every == 0)
        if run_vlm:
            vlm_input = _pil_rgb(image_bgr, self.vlm_size)
        else:
            vlm_input = None

        self._frame_counter += 1

        return PreprocessedFrame(
            packet=packet,
            yolo_input=yolo_input,
            vlm_input=vlm_input,
            clip_input=clip_input,
            run_vlm=run_vlm,
            letterbox_meta=lb_meta,
        )

    def reset_counter(self) -> None:
        """Reset the VLM-stride counter (e.g., between videos)."""
        self._frame_counter = 0


# ---------------------------------------------------------------------------
# Image transform helpers (pure functions — easy to unit test)
# ---------------------------------------------------------------------------

def _letterbox(
    image: np.ndarray,
    target_size: int,
    fill_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, dict]:
    """
    Resize image to (target_size × target_size) while preserving aspect
    ratio by padding with fill_color.

    Returns
    -------
    canvas   : uint8 BGR ndarray of shape (target_size, target_size, 3)
    meta     : dict with scale, pad_top, pad_left, orig_h, orig_w
               — needed to map detection boxes back to original coords
    """
    orig_h, orig_w = image.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    meta = {
        "scale":     scale,
        "pad_top":   pad_top,
        "pad_left":  pad_left,
        "orig_h":    orig_h,
        "orig_w":    orig_w,
    }
    return canvas, meta


def _resize_rgb_float(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Resize BGR image to (size × size), convert to RGB float32 in [0, 1].
    Shape returned: (size, size, 3).
    """
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (rgb / 255.0).astype(np.float32)


def _pil_rgb(image_bgr: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Convert BGR ndarray to RGB PIL Image at given (width, height)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return pil.resize(size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# Coordinate recovery utility (used by YOLO detector after detection)
# ---------------------------------------------------------------------------

def unletterbox_bbox(
    x1: float, y1: float, x2: float, y2: float,
    meta: dict,
) -> tuple[int, int, int, int]:
    """
    Map a bounding box from letterboxed space back to original image coords.

    Parameters
    ----------
    x1, y1, x2, y2 : Pixel coordinates in the (yolo_size × yolo_size) canvas
    meta            : letterbox_meta dict from PreprocessedFrame

    Returns
    -------
    (x1, y1, x2, y2) in original image pixel coordinates (clamped, ints)
    """
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

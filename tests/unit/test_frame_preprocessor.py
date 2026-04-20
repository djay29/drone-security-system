"""
test_frame_preprocessor.py
--------------------------
Unit tests for FramePreprocessor — image transforms, letterboxing, VLM stride,
and coordinate recovery.  No model loading required.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from src.perception.frame_preprocessor import (
    FramePreprocessor,
    _letterbox,
    _resize_rgb_float,
    _pil_rgb,
    unletterbox_bbox,
)
from src.perception.video_ingestor import FramePacket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_packet(h: int = 480, w: int = 640, image=None) -> FramePacket:
    img = image if image is not None else np.zeros((h, w, 3), dtype=np.uint8)
    return FramePacket(
        frame_id=str(uuid.uuid4()),
        ts=datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc),
        video_id="test",
        frame_index=0,
        image=img,
    )


def _make_text_packet() -> FramePacket:
    return FramePacket(
        frame_id=str(uuid.uuid4()),
        ts=datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc),
        video_id="test",
        frame_index=0,
        image=None,
        description="A person near the gate",
    )


# ---------------------------------------------------------------------------
# _letterbox
# ---------------------------------------------------------------------------

class TestLetterbox:

    def test_output_is_square(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas, _ = _letterbox(img, 640)
        assert canvas.shape == (640, 640, 3)

    def test_output_dtype_uint8(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas, _ = _letterbox(img, 640)
        assert canvas.dtype == np.uint8

    def test_meta_contains_required_keys(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, meta = _letterbox(img, 640)
        for key in ("scale", "pad_top", "pad_left", "orig_h", "orig_w"):
            assert key in meta

    def test_meta_orig_dimensions_correct(self):
        img = np.zeros((300, 500, 3), dtype=np.uint8)
        _, meta = _letterbox(img, 640)
        assert meta["orig_h"] == 300
        assert meta["orig_w"] == 500

    def test_scale_preserves_aspect_ratio(self):
        """scale = min(target/w, target/h) — the smaller dimension drives scale."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, meta = _letterbox(img, 640)
        # Width is the limiting dimension: scale = 640/640 = 1.0
        assert abs(meta["scale"] - 1.0) < 1e-6

    def test_padding_is_symmetric_for_portrait(self):
        """Portrait (tall) image → padding added left+right."""
        img = np.zeros((640, 320, 3), dtype=np.uint8)   # 2:1 portrait
        _, meta = _letterbox(img, 640)
        assert meta["pad_top"] == 0     # no vertical padding
        assert meta["pad_left"] > 0     # horizontal padding added

    def test_fill_color_applied(self):
        img = np.zeros((320, 640, 3), dtype=np.uint8)   # landscape
        canvas, meta = _letterbox(img, 640, fill_color=(114, 114, 114))
        # Top padded rows should be the fill color
        if meta["pad_top"] > 0:
            assert np.all(canvas[0, :, 0] == 114)

    def test_square_input_no_padding(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, meta = _letterbox(img, 640)
        assert meta["pad_top"] == 0
        assert meta["pad_left"] == 0


# ---------------------------------------------------------------------------
# unletterbox_bbox
# ---------------------------------------------------------------------------

class TestUnletterboxBbox:

    def test_recovers_original_coordinates(self):
        """bbox mapped through letterbox then back should round-trip."""
        orig_h, orig_w = 480, 640
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        canvas, meta = _letterbox(img, 640)

        # Put a box at the top-left of original image
        ox1, oy1, ox2, oy2 = 50, 50, 150, 150

        # Map to letterbox space manually
        scale = meta["scale"]
        lx1 = int(ox1 * scale) + meta["pad_left"]
        ly1 = int(oy1 * scale) + meta["pad_top"]
        lx2 = int(ox2 * scale) + meta["pad_left"]
        ly2 = int(oy2 * scale) + meta["pad_top"]

        rx1, ry1, rx2, ry2 = unletterbox_bbox(lx1, ly1, lx2, ly2, meta)

        assert abs(rx1 - ox1) <= 2   # allow 1px rounding
        assert abs(ry1 - oy1) <= 2
        assert abs(rx2 - ox2) <= 2
        assert abs(ry2 - oy2) <= 2

    def test_output_clamped_to_image_bounds(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, meta = _letterbox(img, 640)
        # Pass coords slightly outside canvas
        rx1, ry1, rx2, ry2 = unletterbox_bbox(-10, -10, 700, 700, meta)
        assert rx1 >= 0 and ry1 >= 0
        assert rx2 <= 640 and ry2 <= 480


# ---------------------------------------------------------------------------
# _resize_rgb_float
# ---------------------------------------------------------------------------

class TestResizeRgbFloat:

    def test_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = _resize_rgb_float(img, 224)
        assert out.shape == (224, 224, 3)

    def test_output_dtype_float32(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = _resize_rgb_float(img, 224)
        assert out.dtype == np.float32

    def test_values_in_unit_range(self):
        img = (np.ones((64, 64, 3)) * 128).astype(np.uint8)
        out = _resize_rgb_float(img, 64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# FramePreprocessor.process()
# ---------------------------------------------------------------------------

class TestFramePreprocessorProcess:

    def test_process_returns_correct_yolo_shape(self, preprocessor):
        packet = _make_packet()
        result = preprocessor.process(packet)
        assert result is not None
        assert result.yolo_input.shape == (640, 640, 3)

    def test_process_returns_correct_clip_shape(self, preprocessor):
        packet = _make_packet()
        result = preprocessor.process(packet)
        assert result.clip_input.shape == (224, 224, 3)

    def test_process_clip_input_normalised(self, preprocessor):
        packet = _make_packet(image=(np.ones((480, 640, 3)) * 200).astype(np.uint8))
        result = preprocessor.process(packet)
        assert result.clip_input.min() >= 0.0
        assert result.clip_input.max() <= 1.0

    def test_process_preserves_packet_reference(self, preprocessor):
        packet = _make_packet()
        result = preprocessor.process(packet)
        assert result.packet is packet

    def test_process_text_fallback_returns_no_image_outputs(self):
        proc = FramePreprocessor()
        packet = _make_text_packet()
        result = proc.process(packet)
        assert result is not None
        assert result.yolo_input is None
        assert result.clip_input is None
        assert result.run_vlm is False

    def test_process_invalid_packet_returns_none(self, preprocessor):
        bad_packet = FramePacket(
            frame_id="x", ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
            video_id="v", frame_index=0, image=None, description=None,
        )
        result = preprocessor.process(bad_packet)
        assert result is None

    def test_vlm_stride_first_frame_runs_vlm(self):
        proc = FramePreprocessor(vlm_every=5)
        proc.reset_counter()
        result = proc.process(_make_packet())
        assert result.run_vlm is True
        assert result.vlm_input is not None

    def test_vlm_stride_skips_intermediate_frames(self):
        proc = FramePreprocessor(vlm_every=5)
        proc.reset_counter()
        # Frame 0 runs VLM, frames 1-4 skip it
        proc.process(_make_packet())   # frame 0 — runs
        skipped = []
        for _ in range(4):
            r = proc.process(_make_packet())
            skipped.append(r.run_vlm)
        assert not any(skipped), "Frames 1-4 should have run_vlm=False"

    def test_vlm_stride_fires_again_at_N(self):
        proc = FramePreprocessor(vlm_every=3)
        proc.reset_counter()
        results = [proc.process(_make_packet()) for _ in range(6)]
        vlm_frames = [i for i, r in enumerate(results) if r.run_vlm]
        assert vlm_frames == [0, 3]

    def test_vlm_input_is_pil_image(self, preprocessor):
        from PIL import Image
        proc = FramePreprocessor(vlm_every=1)
        result = proc.process(_make_packet())
        assert isinstance(result.vlm_input, Image.Image)

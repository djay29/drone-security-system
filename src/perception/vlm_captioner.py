"""
vlm_captioner.py
----------------
Generates natural-language captions for drone video frames using a
two-tier strategy:

  Tier 1 (primary)  : Claude API  — claude-haiku-3-5 (fast + cheap)
  Tier 2 (fallback) : Moondream2  — local, CPU-friendly, ~2 GB weights

The captioner receives a PreprocessedFrame and returns a filled-in
caption string. YOLO detections are passed as context so the VLM
can focus on *describing* rather than *detecting* — richer output.

Design principles
-----------------
- One class, two backends, same interface: caption(frame, detections) -> str
- Claude backend encodes the PIL image to base64 and sends with a
  security-focused system prompt + YOLO hint in the user message.
- Moondream2 backend runs entirely offline; weights download once via HF.
- Caching: identical frame_ids are never captioned twice in a session.
- Graceful degradation: if Claude fails (rate limit / no key), falls
  back to Moondream2 automatically; if both fail, returns a structured
  fallback string built from YOLO detections.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import base64
import io
import logging
import os
import time
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from .yolo_detector import DetectedObject

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
import langsmith
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------

class CaptionBackend(str, Enum):
    CLAUDE    = "claude"
    MOONDREAM = "moondream"
    AUTO      = "auto"      # try Claude, fall back to Moondream2


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a security analyst reviewing drone surveillance footage.
Your task is to write a single, concise sentence (max 25 words) describing what you see.

Focus on:
- People: count, location, activity, any suspicious behaviour
- Vehicles: type, colour, location, movement direction
- Zone context if visible (gate, fence, parking lot, building entrance)

Do NOT include timestamps, frame numbers, or any metadata.
Do NOT say "I see" or "The image shows". Just describe directly.
Example: "Two people walking toward the main gate; a white pickup truck parked near the fence."
"""

def _build_user_prompt(detections: list[DetectedObject], zone: str = "") -> str:
    """Construct user message — YOLO hint tells Claude what to focus on."""
    if not detections:
        hint = "No objects were pre-detected. Describe the scene."
    else:
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]
        hint = f"Pre-detected objects: {', '.join(parts)}."

    zone_hint = f" Zone: {zone}." if zone else ""
    return f"{hint}{zone_hint} Describe the scene."


# ---------------------------------------------------------------------------
# VLMCaptioner
# ---------------------------------------------------------------------------

class VLMCaptioner:
    """
    Parameters
    ----------
    backend         : CaptionBackend.AUTO (default) tries Claude then Moondream2.
    claude_model    : Claude model slug. Haiku is fast + cheap for captioning.
    claude_api_key  : API key. Falls back to ANTHROPIC_API_KEY env var.
    moondream_revision : HuggingFace revision pin for reproducibility.
    max_new_tokens  : Max tokens for Moondream2 generation.
    cache_captions  : Skip re-captioning the same frame_id in a session.
    """

    CLAUDE_MODEL    = os.getenv("VLM_CAPTIONER_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")   # Bedrock model ID
    MOONDREAM_REPO  = "vikhyatk/moondream2"
    MOONDREAM_REV   = "2025-01-09"

    def __init__(
        self,
        backend: CaptionBackend = CaptionBackend.AUTO,
        claude_model: str = None,
        aws_region: str = None,
        moondream_revision: str = MOONDREAM_REV,
        max_new_tokens: int = 80,
        cache_captions: bool = True,
    ) -> None:
        self.backend            = backend
        self.claude_model       = claude_model or self.CLAUDE_MODEL
        self.claude_api_key     = aws_region or os.getenv("AWS_REGION", "us-east-1")      # field reused as region for Bedrock
        self.moondream_revision = moondream_revision
        self.max_new_tokens     = max_new_tokens
        self.cache_captions     = cache_captions

        self._anthropic_client    = None   # holds ChatBedrock instance
        self._moondream_model     = None
        self._moondream_tokenizer = None
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def caption(
        self,
        frame_id: str,
        vlm_input: Optional[Image.Image],
        detections: Optional[list[DetectedObject]] = None,
        zone: str = "",
    ) -> str:
        """
        Generate a caption for one frame.

        Parameters
        ----------
        frame_id    : Unique frame identifier (for caching).
        vlm_input   : RGB PIL Image (from PreprocessedFrame.vlm_input).
                      If None, falls back to detection-based description only.
        detections  : YOLO detections for this frame (used as context hint).
        zone        : Named zone from telemetry (e.g. "main_gate").

        Returns
        -------
        A single descriptive sentence (str).
        """
        detections = detections or []

        # Cache hit
        if self.cache_captions and frame_id in self._cache:
            return self._cache[frame_id]

        # No image — build caption from detections only
        if vlm_input is None:
            caption = _caption_from_detections(detections, zone)
            self._store(frame_id, caption)
            return caption

        # Route to backend
        caption = self._dispatch(vlm_input, detections, zone)
        self._store(frame_id, caption)
        return caption

    def caption_from_preprocessed(
        self,
        preprocessed_frame,
        detections: Optional[list[DetectedObject]] = None,
        zone: str = "",
    ) -> Optional[str]:
        """
        Convenience wrapper. Returns None if run_vlm=False on the frame
        (stride skipping — caller should reuse last caption or skip).
        """
        if not preprocessed_frame.run_vlm:
            return None

        return self.caption(
            frame_id=preprocessed_frame.packet.frame_id,
            vlm_input=preprocessed_frame.vlm_input,
            detections=detections,
            zone=zone,
        )

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        image: Image.Image,
        detections: list[DetectedObject],
        zone: str,
    ) -> str:
        if self.backend == CaptionBackend.CLAUDE:
            return self._claude_caption(image, detections, zone)

        if self.backend == CaptionBackend.MOONDREAM:
            return self._moondream_caption(image, detections, zone)

        # AUTO — try Claude (Bedrock), fall back to Moondream2
        try:
            return self._claude_caption(image, detections, zone)
        except Exception as exc:
            logger.warning(
                "[VLMCaptioner] Bedrock call failed (%s). Falling back to Moondream2.", exc
            )

        try:
            return self._moondream_caption(image, detections, zone)
        except Exception as exc:
            logger.error("[VLMCaptioner] Moondream2 also failed: %s", exc)
            return _caption_from_detections(detections, zone)

    # ------------------------------------------------------------------
    # Claude backend (via Amazon Bedrock)
    # ------------------------------------------------------------------

    def _claude_caption(
        self,
        image: Image.Image,
        detections: list[DetectedObject],
        zone: str,
    ) -> str:


        llm = self._get_bedrock_client()
        b64 = _pil_to_base64(image)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": "image/jpeg",
                        "data":       b64,
                    },
                },
                {
                    "type": "text",
                    "text": _build_user_prompt(detections, zone),
                },
            ]),
        ]

        with langsmith.tracing_context(enabled=False):
            response = llm.invoke(messages)
        return response.content.strip()

    def _get_bedrock_client(self):
        if self._anthropic_client is None:
            self._anthropic_client = ChatBedrock(
                model_id=self.claude_model,
                region_name=self.claude_api_key or "us-east-1",
                model_kwargs={"max_tokens": 120},
            )
        return self._anthropic_client

    # ------------------------------------------------------------------
    # Moondream2 backend
    # ------------------------------------------------------------------

    def _moondream_caption(
        self,
        image: Image.Image,
        detections: list[DetectedObject],
        zone: str,
    ) -> str:
        model, tokenizer = self._load_moondream()
        prompt = _moondream_prompt(detections, zone)

        # Moondream2 encode + generate
        enc    = model.encode_image(image)
        result = model.answer_question(enc, prompt, tokenizer)
        return result.strip()

    def _load_moondream(self):
        if self._moondream_model is None:
            logger.info("[VLMCaptioner] Loading Moondream2 (first run downloads ~2 GB)…")

            self._moondream_tokenizer = AutoTokenizer.from_pretrained(
                self.MOONDREAM_REPO,
                revision=self.moondream_revision,
            )
            self._moondream_model = AutoModelForCausalLM.from_pretrained(
                self.MOONDREAM_REPO,
                revision=self.moondream_revision,
                trust_remote_code=True,
                torch_dtype=torch.float32,   # float32 for CPU compatibility
            )
            self._moondream_model.eval()
            logger.info("[VLMCaptioner] Moondream2 loaded.")
        return self._moondream_model, self._moondream_tokenizer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _store(self, frame_id: str, caption: str) -> None:
        if self.cache_captions:
            self._cache[frame_id] = caption

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Pure helpers (no class state)
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image, quality: int = 85) -> str:
    """Encode PIL image as JPEG base64 string for Claude API."""
    buf = io.BytesIO()
    # Ensure RGB (no alpha channel)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _caption_from_detections(detections: list[DetectedObject], zone: str = "") -> str:
    """
    Build a structured caption string purely from YOLO detections.
    Used when VLM is unavailable or image is None (text-fallback mode).
    """
    if not detections:
        zone_str = f" in {zone}" if zone else ""
        return f"No objects detected{zone_str}."

    counts: dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1

    parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]
    objects_str = ", ".join(parts)
    zone_str    = f" at {zone}" if zone else ""
    return f"{objects_str} detected{zone_str}."


def _moondream_prompt(detections: list[DetectedObject], zone: str = "") -> str:
    """
    Focused question for Moondream2 — shorter than the Claude system prompt
    since Moondream works better with a direct question than a role description.
    """
    if not detections:
        base = "Describe this drone surveillance scene in one sentence."
    else:
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        items = ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items())
        base = (
            f"This drone footage contains {items}. "
            f"Describe their location and activity in one sentence."
        )

    zone_str = f" The camera is monitoring the {zone}." if zone else ""
    return base + zone_str
"""
clip_embedder.py
----------------
Generates CLIP embeddings for drone video frames using OpenCLIP ViT-B/32.

Consumes:  PreprocessedFrame.clip_input  (224x224 float32 RGB ndarray)
Produces:  list[float]  — 512-dim L2-normalised vector → goes into ChromaDB

Two embedding modes
-------------------
  image_embed(clip_input)   → embed a frame (for indexing)
  text_embed(query)         → embed a search query (for retrieval)

Both return the same 512-dim space so cosine similarity works across them —
this is the foundation for "show me blue truck" semantic search.

Design notes
------------
- Model loaded once, cached on instance (lazy).
- Embeddings are L2-normalised so dot product == cosine similarity.
- Batch support: embed_batch() processes a list of frames in one forward
  pass — much faster than calling embed() in a loop.
- Anomaly hook: distance_from_centroid() supports the unsupervised anomaly
  detection innovation (section 11 of context sheet).
- Text embeddings are LRU-cached so repeated queries (e.g. "person at gate")
  don't re-run the model.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# CLIP embedding dimension for ViT-B/32
EMBED_DIM = 512


# ---------------------------------------------------------------------------
# CLIPEmbedder
# ---------------------------------------------------------------------------

class CLIPEmbedder:
    """
    Parameters
    ----------
    model_name   : OpenCLIP model architecture.  'ViT-B-32' is the default —
                   fast, 512-dim, good semantic quality.
    pretrained   : OpenCLIP pretrained weights tag.
    device       : 'cpu', 'cuda', 'mps', or None (auto-detect).
    normalise    : L2-normalise all output vectors (keep True for cosine sim).
    """

    MODEL_NAME  = os.getenv("CLIP_MODEL_NAME", "ViT-B-32-quickgelu")
    PRETRAINED  = os.getenv("CLIP_PRETRAINED", "openai")          # openai weights ship with open_clip_torch

    def __init__(
        self,
        model_name: str = None,
        pretrained: str = None,
        device: Optional[str] = None,
        normalise: bool = True,
    ) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self.pretrained = pretrained or self.PRETRAINED
        self.device     = device or _auto_device()
        self.normalise  = normalise

        # Lazy-loaded
        self._model      = None
        self._preprocess = None   # OpenCLIP image transform (not used — we preprocess ourselves)
        self._tokenizer  = None
        self._text_cache: dict[str, list[float]] = {}  # query → embedding

    # ------------------------------------------------------------------
    # Public API — single frame
    # ------------------------------------------------------------------

    def embed_frame(self, clip_input: np.ndarray) -> list[float]:
        """
        Embed one pre-processed frame.

        Parameters
        ----------
        clip_input : float32 RGB ndarray (224, 224, 3) in [0, 1]
                     from FramePreprocessor.clip_input

        Returns
        -------
        512-dim list[float], L2-normalised.
        """
        if clip_input is None:
            raise ValueError("[CLIPEmbedder] clip_input is None — text-fallback frames cannot be embedded.")

        tensor = self._to_tensor(clip_input[np.newaxis])   # (1, 3, 224, 224)
        vec    = self._encode_image_tensor(tensor)         # (1, 512)
        return vec[0].tolist()

    def embed_text(self, query: str) -> list[float]:
        """
        Embed a natural-language query for semantic search.

        Parameters
        ----------
        query : e.g. "blue truck near the gate"

        Returns
        -------
        512-dim list[float], L2-normalised. Results are cached by query string.
        """
        if query not in self._text_cache:
            self._text_cache[query] = self._embed_text_impl(self, query)
        return self._text_cache[query]

    # ------------------------------------------------------------------
    # Public API — batch (faster for bulk indexing)
    # ------------------------------------------------------------------

    def embed_batch(self, clip_inputs: list[np.ndarray]) -> list[list[float]]:
        """
        Embed a list of clip_input arrays in a single forward pass.

        Parameters
        ----------
        clip_inputs : list of (224, 224, 3) float32 arrays

        Returns
        -------
        list of 512-dim vectors, one per input.
        """
        if not clip_inputs:
            return []

        stacked = np.stack(clip_inputs, axis=0)          # (N, 224, 224, 3)
        tensor  = self._to_tensor(stacked)               # (N, 3, 224, 224)
        vecs    = self._encode_image_tensor(tensor)      # (N, 512)
        return vecs.tolist()

    # ------------------------------------------------------------------
    # Convenience: embed directly from PreprocessedFrame
    # ------------------------------------------------------------------

    def embed_preprocessed(self, preprocessed_frame) -> Optional[list[float]]:
        """
        Embed from a PreprocessedFrame object.
        Returns None if clip_input is None (text-fallback mode).
        """
        if preprocessed_frame.clip_input is None:
            return None
        return self.embed_frame(preprocessed_frame.clip_input)

    # ------------------------------------------------------------------
    # Anomaly detection hook
    # ------------------------------------------------------------------

    def distance_from_centroid(
        self,
        embedding: list[float],
        centroid: list[float],
    ) -> float:
        """
        Cosine distance between a frame embedding and a "normal scene" centroid.

        High distance → frame is visually unusual → flag for review.

        Usage
        -----
        # Build centroid from first N frames of a "normal" period:
        normal_vecs = [embedder.embed_frame(f) for f in normal_frames]
        centroid = np.mean(normal_vecs, axis=0).tolist()

        # Score live frames:
        score = embedder.distance_from_centroid(live_vec, centroid)
        if score > 0.3:
            alert("Unusual scene detected")

        Returns
        -------
        Float in [0, 2]: 0 = identical, 2 = opposite. Typically < 0.5
        for same-scene variations.
        """
        a = np.array(embedding,  dtype=np.float32)
        b = np.array(centroid,   dtype=np.float32)
        cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        return 1.0 - cosine_sim   # distance: 0 = same, 1 = orthogonal

    def build_centroid(self, embeddings: list[list[float]]) -> list[float]:
        """
        Compute L2-normalised mean of a set of embeddings.
        Use this on the first N frames of a video to define "normal".
        """
        if not embeddings:
            raise ValueError("Cannot build centroid from empty list.")
        mat = np.array(embeddings, dtype=np.float32)
        mean = mat.mean(axis=0)
        return _l2_normalise(mean).tolist()

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Force model load + one dummy forward pass. Call once at startup."""
        logger.info("[CLIPEmbedder] Warming up %s/%s on %s…", self.model_name, self.pretrained, self.device)
        dummy = np.zeros((224, 224, 3), dtype=np.float32)
        self.embed_frame(dummy)
        self.embed_text("warmup")
        logger.info("[CLIPEmbedder] Warmup done.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            import open_clip
            logger.info(
                "[CLIPEmbedder] Loading %s (%s) on %s…",
                self.model_name, self.pretrained, self.device,
            )
            model, _, _ = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            model.eval()
            self._model     = model
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            logger.info("[CLIPEmbedder] Model loaded.")
        return self._model

    def _encode_image_tensor(self, tensor) -> np.ndarray:
        """Forward pass on a (N, 3, 224, 224) tensor. Returns (N, 512) numpy array."""
        import torch
        model = self._load_model()
        with torch.no_grad():
            t = torch.tensor(tensor, dtype=torch.float32, device=self.device)
            features = model.encode_image(t)
            features = features.cpu().numpy().astype(np.float32)

        if self.normalise:
            features = np.array([_l2_normalise(v) for v in features])
        return features

    @staticmethod
    def _embed_text_impl(self_ref: "CLIPEmbedder", query: str) -> list[float]:
        import torch
        model     = self_ref._load_model()
        tokenizer = self_ref._tokenizer

        tokens = tokenizer([query]).to(self_ref.device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            vec = features.cpu().numpy().astype(np.float32)[0]

        if self_ref.normalise:
            vec = _l2_normalise(vec)
        return vec.tolist()

    def _to_tensor(self, array: np.ndarray) -> np.ndarray:
        """
        Convert (N, H, W, 3) float32 [0,1] array to (N, 3, H, W) for CLIP.
        Applies ImageNet normalisation expected by OpenCLIP ViT-B/32.
        """
        # Transpose HWC → CHW
        t = array.transpose(0, 3, 1, 2)   # (N, 3, 224, 224)

        # ImageNet normalisation
        mean = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
        std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        t = (t - mean[:, None, None]) / std[:, None, None]

        return t.astype(np.float32)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
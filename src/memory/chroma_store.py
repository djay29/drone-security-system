"""
chroma_store.py
---------------
Vector store for CLIP frame embeddings using ChromaDB.

Stores 512-dim ViT-B/32 embeddings with rich metadata so queries can
combine semantic similarity with structured filters.

Key operations
--------------
  add_frame()          — index one frame's embedding + metadata
  add_batch()          — bulk index (faster for initial load)
  query_by_text()      — "blue truck near fence" → top-K frames
  query_by_image_vec() — similar frames to a given embedding
  load_from_jsonl()    — ingest embeddings.jsonl from test pipeline

ChromaDB collection metadata schema (per document)
---------------------------------------------------
  frame_id     : str
  frame_index  : int
  ts           : ISO8601 str
  video_id     : str
  zone         : str
  caption      : str  (filled in if available)
  class_names  : str  (comma-separated detected classes, for $contains filter)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import chromadb

logger = logging.getLogger(__name__)

COLLECTION_NAME = "drone_frames"


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------

class ChromaStore:
    """
    Parameters
    ----------
    persist_dir  : Directory where ChromaDB persists its data.
                   Use None for an in-memory ephemeral store (tests).
    collection   : ChromaDB collection name.
    embedder     : Optional CLIPEmbedder instance for text→vec encoding.
                   If provided, query_by_text() works end-to-end.
                   If None, pass pre-computed vectors to query_by_image_vec().
    """

    def __init__(
        self,
        persist_dir: Optional[str | Path] = "data/chroma",
        collection: str = COLLECTION_NAME,
        embedder=None,   # CLIPEmbedder | None — avoid hard import
    ) -> None:
        self.persist_dir = str(persist_dir) if persist_dir else None
        self.collection_name = collection
        self.embedder = embedder

        self._client     = None
        self._collection = None
        self._init()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init(self) -> None:
        if self.persist_dir:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "[ChromaStore] Collection '%s' ready (%d docs)",
            self.collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_frame(
        self,
        frame_id: str,
        embedding: list[float],
        frame_index: int = 0,
        ts: str = "",
        video_id: str = "",
        zone: str = "",
        caption: str = "",
        class_names: Optional[list[str]] = None,
    ) -> None:
        """
        Index a single frame embedding.

        Parameters
        ----------
        embedding   : 512-d float list from CLIPEmbedder.embed_frame()
        class_names : List of detected class names for metadata filtering
                      e.g. ["person", "truck"]
        """
        self._collection.upsert(
            ids        = [frame_id],
            embeddings = [embedding],
            metadatas  = [{
                "frame_id":    frame_id,
                "frame_index": frame_index,
                "ts":          ts,
                "video_id":    video_id,
                "zone":        zone,
                "caption":     caption,
                "class_names": ",".join(class_names) if class_names else "",
            }],
        )

    def add_batch(self, records: list[dict]) -> None:
        """
        Bulk-index a list of embedding records.

        Each dict must have:
          frame_id, embedding  (required)
          frame_index, ts, video_id, zone, caption, class_names  (optional)
        """
        if not records:
            return

        ids        = [r["frame_id"] for r in records]
        embeddings = [r["embedding"] for r in records]
        metadatas  = [
            {
                "frame_id":    r["frame_id"],
                "frame_index": r.get("frame_index", 0),
                "ts":          r.get("ts", ""),
                "video_id":    r.get("video_id", ""),
                "zone":        r.get("zone", ""),
                "caption":     r.get("caption", ""),
                "class_names": ",".join(r["class_names"]) if r.get("class_names") else "",
            }
            for r in records
        ]

        self._collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas
        )
        logger.info("[ChromaStore] Indexed %d frames (total: %d)", len(records), self.count())

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_by_text(
        self,
        query: str,
        top_k: int = 100,
        zone: Optional[str] = None,
        class_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Natural language semantic search.
        Requires self.embedder to be set.

        Parameters
        ----------
        query        : e.g. "blue truck near the gate"
        top_k        : Number of results to return
        zone         : Filter to a specific zone (exact match)
        class_filter : Filter to frames containing this class
                       e.g. "truck" → only frames where truck was detected

        Returns
        -------
        List of result dicts with keys:
          frame_id, score, metadata (ts, zone, caption, class_names, ...)
        """
        if self.embedder is None:
            raise RuntimeError(
                "[ChromaStore] No embedder set. Pass a CLIPEmbedder instance "
                "to ChromaStore() to use query_by_text()."
            )
        vec = self.embedder.embed_text(query)
        return self.query_by_vector(vec, top_k=top_k, zone=zone, class_filter=class_filter)

    def query_by_vector(
        self,
        embedding: list[float],
        top_k: int = 100,
        zone: Optional[str] = None,
        class_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Query by a pre-computed embedding vector.
        Use this when you already have a CLIP vector (e.g. from a query frame).
        """
        where = _build_where(zone=zone, class_filter=class_filter)

        kwargs: dict = dict(
            query_embeddings=[embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
            include=["metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        for i, meta in enumerate(results["metadatas"][0]):
            output.append({
                "frame_id": meta["frame_id"],
                "score":    round(1 - results["distances"][0][i], 4),  # cosine sim
                "metadata": meta,
            })
        return output

    def get_by_frame_id(self, frame_id: str) -> Optional[dict]:
        """Fetch a single frame record by its ID."""
        result = self._collection.get(ids=[frame_id], include=["metadatas", "embeddings"])
        if not result["ids"]:
            return None
        return {
            "frame_id":  result["ids"][0],
            "metadata":  result["metadatas"][0],
            "embedding": result["embeddings"][0],
        }

    # ------------------------------------------------------------------
    # Bulk ingestion from pipeline output files
    # ------------------------------------------------------------------

    def load_from_jsonl(
        self,
        embeddings_path: str | Path,
        captions_path: Optional[str | Path] = None,
        video_id: str = "",
    ) -> int:
        """
        Ingest embeddings.jsonl (and optionally captions.jsonl) produced
        by test_video_pipeline.py into the collection.

        Parameters
        ----------
        embeddings_path : Path to embeddings.jsonl
        captions_path   : Optional path to captions.jsonl — used to enrich
                          metadata with captions + detected class names
        video_id        : Tag all records with this video_id

        Returns
        -------
        Number of frames indexed.
        """
        # Load captions into a lookup dict keyed by frame_id
        caption_lookup: dict[str, dict] = {}
        if captions_path and Path(captions_path).exists():
            with open(captions_path) as fh:
                for line in fh:
                    rec = json.loads(line.strip())
                    caption_lookup[rec["frame_id"]] = rec

        records = []
        with open(embeddings_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                emb_rec  = json.loads(line)
                fid      = emb_rec["frame_id"]
                cap_rec  = caption_lookup.get(fid, {})

                class_names = [
                    d["class"] for d in cap_rec.get("detections", [])
                ]

                records.append({
                    "frame_id":    fid,
                    "embedding":   emb_rec["embedding"],
                    "frame_index": emb_rec.get("frame_index", 0),
                    "ts":          emb_rec.get("ts", ""),
                    "video_id":    video_id or cap_rec.get("video_id", ""),
                    "zone":        cap_rec.get("zone", ""),
                    "caption":     cap_rec.get("caption", ""),
                    "class_names": class_names,
                })

        self.add_batch(records)
        return len(records)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection. Useful for tests / re-indexing."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("[ChromaStore] Collection reset.")


# ---------------------------------------------------------------------------
# ChromaDB where-filter builder
# ---------------------------------------------------------------------------

def _build_where(
    zone: Optional[str] = None,
    class_filter: Optional[str] = None,
) -> Optional[dict]:
    """
    Build a ChromaDB metadata filter dict.
    ChromaDB uses MongoDB-style operators: $eq, $contains.
    """
    conditions = []

    if zone:
        conditions.append({"zone": {"$eq": zone}})

    if class_filter:
        # class_names is stored as comma-separated string → use $contains
        conditions.append({"class_names": {"$contains": class_filter}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
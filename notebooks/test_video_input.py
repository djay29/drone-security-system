"""
test_video_pipeline.py
----------------------
End-to-end smoke test: VideoIngestor → FramePreprocessor → YOLOv8s → VLMCaptioner → CLIPEmbedder

HOW TO RUN
----------
    # Claude API (best quality — set your key first)
    export ANTHROPIC_API_KEY=sk-ant-...
    python test_video_pipeline.py path/to/your/video.mp4

    # Offline / Moondream2 only
    python test_video_pipeline.py path/to/your/video.mp4 --backend moondream

    # Auto (Claude → Moondream2 fallback, default)
    python test_video_pipeline.py path/to/your/video.mp4 --backend auto

OUTPUTS
-------
- Console: per-frame table (detections + captions)
- data/test_output/  — annotated JPEGs with bbox + caption burned in
- data/test_output/captions.jsonl — one JSON record per captioned frame
- data/test_output/embeddings.jsonl — one JSON record per embedded frame (frame_id + 512-d vector)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config — edit or override via CLI args
# ---------------------------------------------------------------------------
VIDEO_PATH    = r"E:\project\drone-security-system\data\videos\14145512_3840_2160_24fps.mp4"
SAMPLE_EVERY  = 5      # emit every Nth frame from source
MAX_FRAMES    = 30     # hard stop (None = whole video)
CONFIDENCE    = 0.4    # YOLO confidence threshold
VLM_EVERY     = 5      # run VLM on every Nth *emitted* frame
SAVE_OUTPUT   = True
OUTPUT_DIR    = Path("data/test_output")
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))

from src.perception.video_ingestor      import VideoIngestor
from src.perception.frame_preprocessor  import FramePreprocessor
from src.perception.yolo_detector       import YOLODetector, DetectedObject
from src.perception.vlm_captioner       import VLMCaptioner, CaptionBackend
from src.perception.clip_embedder       import CLIPEmbedder
from src.memory.sqlite_store            import SQLiteStore
from src.memory.chroma_store            import ChromaStore
from src.agent.rule_engine              import RuleEngine
from src.agent.graph                    import build_agent


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

BBOX_COLORS = {
    "person":     (0, 230, 0),
    "car":        (255, 140, 0),
    "truck":      (0, 140, 255),
    "motorcycle": (240, 0, 240),
    "bicycle":    (0, 240, 240),
    "bus":        (240, 240, 0),
    "boat":       (140, 240, 140),
}
DEFAULT_COLOR = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, detections: list[DetectedObject]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = BBOX_COLORS.get(det.class_name, DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tid   = f"#{det.track_id} " if det.track_id is not None else ""
        label = f"{tid}{det.class_name} {det.confidence:.2f}"
        (tw, th), bl = cv2.getTextSize(label, FONT, 0.52, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - bl - 2), FONT, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def overlay_caption(frame: np.ndarray, caption: str, max_width: int = 80) -> np.ndarray:
    """Burn caption text into the bottom of the frame with a dark background strip."""
    out = frame.copy()
    h, w = out.shape[:2]

    words = caption.split()
    lines, line = [], []
    for word in words:
        if sum(len(x) for x in line) + len(line) + len(word) <= max_width:
            line.append(word)
        else:
            if line:
                lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))

    font_scale, thickness, pad = 0.5, 1, 6
    line_h  = 18
    strip_h = len(lines) * line_h + pad * 2

    cv2.rectangle(out, (0, h - strip_h), (w, h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        y = h - strip_h + pad + (i + 1) * line_h - 2
        cv2.putText(out, text, (pad, y), FONT, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Drone security full-pipeline smoke test")
    p.add_argument("video", nargs="?", default=VIDEO_PATH, help="Path to input video")
    p.add_argument("--backend", choices=["auto", "claude", "moondream"], default="auto")
    p.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    p.add_argument("--max-frames",   type=int, default=MAX_FRAMES)
    p.add_argument("--confidence",   type=float, default=CONFIDENCE)
    p.add_argument("--vlm-every",    type=int, default=VLM_EVERY)
    p.add_argument("--no-save",      action="store_true")
    return p.parse_args()


def run(args) -> None:
    path = Path(args.video)
    if not path.exists():
        print(f"\n❌  File not found: {path.resolve()}")
        print("    Pass your video path as the first argument.\n")
        sys.exit(1)

    save = SAVE_OUTPUT and not args.no_save
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    backend = {
        "auto":      CaptionBackend.AUTO,
        "claude":    CaptionBackend.CLAUDE,
        "moondream": CaptionBackend.MOONDREAM,
    }[args.backend]

    print(f"\n{'='*65}")
    print(f"  Drone Security — Full Perception Smoke Test")
    print(f"{'='*65}")
    print(f"  Video      : {path.name}")
    print(f"  Sample     : every {args.sample_every} frame(s)")
    print(f"  Max frames : {args.max_frames or 'unlimited'}")
    print(f"  YOLO conf  : {args.confidence}  (model: yolov8s.pt)")
    print(f"  VLM every  : {args.vlm_every} emitted frame(s)")
    print(f"  VLM backend: {args.backend.upper()}")
    print(f"  CLIP       : ViT-B/32 (OpenCLIP) — every frame")
    print(f"  Output dir : {OUTPUT_DIR if save else 'disabled'}")
    print(f"{'='*65}\n")

    # ── Build pipeline ──────────────────────────────────────────────────
    ingestor = VideoIngestor.from_file(
        path=path,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        start_ts=datetime.now(timezone.utc),
    )
    preprocessor = FramePreprocessor(
        yolo_size=640,
        vlm_every=args.vlm_every,
    )
    detector = YOLODetector(
        model_name="yolov8s.pt",
        confidence=args.confidence,
        use_tracking=True,
    )
    captioner   = VLMCaptioner(backend=backend)
    embedder    = CLIPEmbedder()
    db          = SQLiteStore("data/drone_security.db")
    chroma      = ChromaStore("data/chroma", embedder=embedder)
    rule_engine = RuleEngine("configs/rules.yaml")
    agent       = build_agent(detector, captioner, embedder, db, chroma, rule_engine)

    # ── Stream ──────────────────────────────────────────────────────────
    total_frames   = 0
    total_detects  = 0
    total_captions = 0
    total_alerts   = 0
    caption_log    = []
    embedding_log  = []
    last_caption   = ""
    centroid       = None
    all_embeddings = []

    t0 = time.perf_counter()

    print(f"  {'Frm':>5}  {'Time':>10}  {'Detections':<35}  {'Anomaly':>7}  {'Alerts':<6}  Caption")
    print(f"  {'-'*5}  {'-'*10}  {'-'*35}  {'-'*7}  {'-'*6}  {'-'*40}")

    for packet in ingestor.stream():
        preprocessed = preprocessor.process(packet)
        if preprocessed is None:
            continue

        zone = packet.metadata.get("zone", "")

        # ── Invoke agent ──
        result = agent.invoke({"preprocessed": preprocessed, "zone": zone})

        detections = result.get("detections", [])
        caption    = result.get("caption")
        embedding  = result.get("embedding")
        alerts     = result.get("alerts_fired", [])

        total_detects += len(detections)
        total_alerts  += len(alerts)

        if caption:
            last_caption = caption
            total_captions += 1
            caption_log.append({
                "frame_id":    packet.frame_id,
                "frame_index": packet.frame_index,
                "ts":          packet.ts.isoformat(),
                "caption":     caption,
            })

        # ── CLIP anomaly scoring ──
        anomaly_flag = ""
        if embedding:
            all_embeddings.append(embedding)
            embedding_log.append({
                "frame_id":    packet.frame_id,
                "frame_index": packet.frame_index,
                "ts":          packet.ts.isoformat(),
                "embedding":   embedding,
            })
            if len(all_embeddings) == 10:
                centroid = embedder.build_centroid(all_embeddings)
            elif centroid is not None:
                score = embedder.distance_from_centroid(embedding, centroid)
                anomaly_flag = f"⚠ {score:.2f}" if score > 0.25 else f"  {score:.2f}"

        # ── Console row ──
        ts_str  = packet.ts.strftime("%H:%M:%S.%f")[:11]
        det_str = (
            "; ".join(f"{d.class_name}#{d.track_id or '?'}" for d in detections[:3])
            + (f" +{len(detections)-3}" if len(detections) > 3 else "")
        ) if detections else "(none)"
        alert_str  = f"🚨x{len(alerts)}" if alerts else ""
        vlm_marker = "◉" if caption else " "
        disp = (last_caption[:45] + "…") if len(last_caption) > 46 else last_caption
        print(f"  {packet.frame_index:>5}  {ts_str:>10}  {det_str:<35}  {anomaly_flag:>7}  {alert_str:<6}  {vlm_marker} {disp}")
        for a in alerts:
            print(f"  {'':>5}  {'':>10}  ⚡ [{a['severity'].upper()}] {a['message']}")

        # ── Save annotated frame ──
        if save and packet.image is not None:
            out_img = draw_detections(packet.image, detections)
            if last_caption:
                out_img = overlay_caption(out_img, last_caption)
            cv2.imwrite(
                str(OUTPUT_DIR / f"frame_{packet.frame_index:05d}.jpg"),
                out_img, [cv2.IMWRITE_JPEG_QUALITY, 88],
            )

        total_frames += 1

    elapsed = time.perf_counter() - t0

    # ── Save caption log ────────────────────────────────────────────────
    if save and caption_log:
        log_path = OUTPUT_DIR / "captions.jsonl"
        with log_path.open("w") as fh:
            for rec in caption_log:
                fh.write(json.dumps(rec) + "\n")
        print(f"\n  📄 Caption log saved    → {log_path}")

    if save and embedding_log:
        emb_path = OUTPUT_DIR / "embeddings.jsonl"
        with emb_path.open("w") as fh:
            for rec in embedding_log:
                fh.write(json.dumps(rec) + "\n")
        print(f"  📄 Embedding log saved  → {emb_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Frames processed : {total_frames}")
    print(f"  Total detections : {total_detects}")
    print(f"  VLM captions     : {total_captions}  (every {args.vlm_every} frames → {total_captions} API calls)")
    print(f"  CLIP embeddings  : {len(embedding_log)}  (512-d vectors, ViT-B/32)")
    print(f"  Alerts fired     : {total_alerts}")
    stats = db.get_stats()
    print(f"  SQLite frames    : {stats['frames']}  |  objects: {stats['objects']}")
    print(f"  ChromaDB indexed : {chroma.count()} frames")
    print(f"{'='*65}")

    # ── Live semantic search demo ──────────────────────────────────────
    if chroma.count() > 0:
        print("\n  🔍 Semantic search demo (query_by_text):")
        for demo_q in ["person near gate", "vehicle", "empty scene"]:
            hits = chroma.query_by_text(demo_q, top_k=2)
            print(f"    '{demo_q}':")
            for h in hits:
                cap = h['metadata'].get('caption', '')[:60]
                print(f"      score={h['score']:.3f}  ts={h['metadata'].get('ts','')[:19]}  {cap}")
    print(f"  Elapsed          : {elapsed:.1f}s  ({total_frames / max(elapsed, 0.01):.1f} fps pipeline)")
    print(f"  Avg per frame    : {elapsed / max(total_frames, 1) * 1000:.0f} ms")
    if save:
        print(f"  Annotated frames : {total_frames}  →  {OUTPUT_DIR}/")
    print(f"{'='*65}")

    if caption_log:
        print(f"\n  Sample captions (first {min(5, len(caption_log))}):")
        for rec in caption_log[:5]:
            print(f"    [{rec['frame_index']:>4}]  {rec['caption']}")
    print()


if __name__ == "__main__":
    run(parse_args())
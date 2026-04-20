"""
graph.py
--------
Builds and compiles the LangGraph state machine for the drone security agent.

Graph structure:
                          ┌─────────────┐
                          │   perceive  │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │contextualize│
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │ rule_check  │
                          └──────┬──────┘
                                 │
                   needs_llm? ───┤
                   True  ┌───────┘  False
                         │                │
                  ┌──────▼──────┐  ┌──────▼──────┐
                  │  llm_judge  │  │    alert    │
                  └──────┬──────┘  └──────┬──────┘
                         │                │
                         └───────┬────────┘
                                 │
                          ┌──────▼──────┐
                          │     log     │
                          └─────────────┘

Usage
-----
    from src.agent.graph import build_agent

    agent = build_agent(detector, captioner, embedder, sqlite, chroma, rule_engine)

    # Per-frame invocation:
    result = agent.invoke({
        "preprocessed": preprocessed_frame,
        "zone": "main_gate",
    })
    print(result["alerts_fired"])
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    make_perceive_node,
    make_contextualize_node,
    make_rule_check_node,
    make_llm_judge_node,
    make_alert_node,
    make_log_node,
)
from src.agent.rule_engine import RuleEngine
from src.memory.sqlite_store import SQLiteStore
from src.memory.chroma_store import ChromaStore
from src.perception.yolo_detector import YOLODetector
from src.perception.vlm_captioner import VLMCaptioner
from src.perception.clip_embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def route_after_rule_check(state: AgentState) -> str:
    """
    Conditional edge after rule_check:
      - If any rule hit needs LLM escalation → 'llm_judge'
      - If rule hits exist but none need LLM  → 'alert'
      - If no rule hits at all               → 'log' (skip alert)
    """
    rule_hits = state.get("rule_hits", [])
    if not rule_hits:
        return "log"
    if state.get("needs_llm", False):
        return "llm_judge"
    return "alert"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent(
    detector:     YOLODetector,
    captioner:    VLMCaptioner,
    embedder:     CLIPEmbedder,
    sqlite:       SQLiteStore,
    chroma:       ChromaStore,
    rule_engine:  RuleEngine | None = None,
    aws_region:   str = "us-east-1",
    async_vlm:    bool = False,
    vlm_workers:  int = 2,
) -> "CompiledGraph":
    """
    Assemble and compile the full agent graph.

    Parameters
    ----------
    detector    : YOLODetector instance
    captioner   : VLMCaptioner instance
    embedder    : CLIPEmbedder instance
    sqlite      : SQLiteStore instance
    chroma      : ChromaStore instance
    rule_engine : RuleEngine instance (loads rules.yaml). If None, defaults loaded.
    aws_region  : AWS region for Bedrock calls (default: us-east-1).
    async_vlm   : If True, VLM captioning runs in a thread pool so it never
                  blocks the YOLO/rules hot path. Recommended for RTSP streams.
    vlm_workers : Thread pool size for async VLM (default 2).

    Returns
    -------
    Compiled LangGraph (call .invoke(state) per frame).
    """
    if rule_engine is None:
        rule_engine = RuleEngine()

    vlm_executor = ThreadPoolExecutor(max_workers=vlm_workers, thread_name_prefix="vlm") if async_vlm else None

    perceive_fn      = make_perceive_node(detector, captioner, embedder, vlm_executor)
    contextualize_fn = make_contextualize_node(sqlite)
    rule_check_fn    = make_rule_check_node(rule_engine)
    llm_judge_fn     = make_llm_judge_node(region=aws_region)
    alert_fn         = make_alert_node(sqlite)
    log_fn           = make_log_node(sqlite, chroma)

    # Build graph
    graph = StateGraph(AgentState)

    graph.add_node("perceive",      perceive_fn)
    graph.add_node("contextualize", contextualize_fn)
    graph.add_node("rule_check",    rule_check_fn)
    graph.add_node("llm_judge",     llm_judge_fn)
    graph.add_node("alert",         alert_fn)
    graph.add_node("log",           log_fn)

    # Edges
    graph.set_entry_point("perceive")
    graph.add_edge("perceive",      "contextualize")
    graph.add_edge("contextualize", "rule_check")

    # Conditional branch after rule_check
    graph.add_conditional_edges(
        "rule_check",
        route_after_rule_check,
        {
            "llm_judge": "llm_judge",
            "alert":     "alert",
            "log":       "log",
        },
    )

    graph.add_edge("llm_judge", "alert")
    graph.add_edge("alert",     "log")
    graph.add_edge("log",       END)

    compiled = graph.compile()
    logger.info("[Agent] Graph compiled successfully.")
    return compiled
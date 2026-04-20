"""
generate_ai_tooling_doc.py
--------------------------
Generates the AI Tooling & Collaboration document as a clean white-background PDF.
Run: uv run python generate_ai_tooling_doc.py
Output: AI_Tooling_Document.pdf
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime


# ---------------------------------------------------------------------------
# Colour palette (white background theme)
# ---------------------------------------------------------------------------

BLACK      = (15, 15, 15)
DARK_GREY  = (50, 50, 60)
MID_GREY   = (110, 115, 130)
LIGHT_GREY = (235, 237, 240)
ACCENT     = (30, 100, 200)       # deep blue
ACCENT2    = (0, 140, 100)        # teal green
WHITE      = (255, 255, 255)
PAGE_BG    = (255, 255, 255)
RULE_CLR   = (210, 213, 220)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

INTRO = (
    "This document describes how AI-assisted tooling -- specifically Claude Code "
    "by Anthropic -- was used throughout the development of the Drone Security System. "
    "It explains the division of responsibility between human-directed design work and "
    "AI-assisted code generation, and reflects honestly on what each side contributed."
)

SECTIONS = [
    {
        "heading": "1.  Role Division: Human vs. AI",
        "accent": ACCENT,
        "items": [
            {
                "label": "My responsibilities (human-led)",
                "color": ACCENT,
                "bullets": [
                    "Defining the overall product goal and the problem to be solved.",
                    "Designing the system architecture: the four-layer pipeline (Ingestion -> Processing -> Storage -> Delivery), the choice of LangGraph for agent orchestration, and the dual-store memory strategy (SQLite + ChromaDB).",
                    "Designing the LangGraph workflow: identifying the six nodes (perceive, contextualize, rule_check, llm_judge, alert, log), deciding the conditional routing logic (needs_llm flag), and specifying what each node should read and write to AgentState.",
                    "Writing and iterating on prompts: the VLM captioning prompt, the LLM judge prompt (VERDICT / ALERT / REASON format), the RAG answer prompt, the query intent parser prompt, and the follow-up suggestion prompt.",
                    "Debugging logic and code: diagnosing the SQLite :memory: connection isolation bug, the ChatBedrock instantiation timing issue in tests, and the two-invoke pattern needed to trigger loitering rules in integration tests.",
                    "Validating outputs at every stage and deciding when generated code was correct, when it needed adjustment, and when the approach needed rethinking.",
                    "Writing all YAML rule definitions -- the condition schemas, severity levels, and needs_llm escalation flags for 20+ security rules.",
                    "Making all product decisions: telemetry simulator design, alert cooldown strategy, frame storage resolution separation, async VLM threading model.",
                ],
            },
            {
                "label": "Claude Code's contributions (AI-assisted)",
                "color": ACCENT2,
                "bullets": [
                    "Generating the implementation code for modules once the design was specified: SQLiteStore, ChromaStore, HybridRetriever, VideoIngestor, FramePreprocessor, TelemetrySimulator, StreamProcessor, RuleEngine, and all LangGraph node factories.",
                    "Writing the full Streamlit dashboard layout -- tab structure, CSS styling, the Live Feed result loop, Timeline event expanders, and the Ask tab RAG chat interface.",
                    "Writing the 124-test suite across three files (test_sqlite_store.py, test_hybrid_retriever.py, test_agent_pipeline.py) once test scenarios were described.",
                    "Generating documentation files: README.md, docs/architecture.md, docs/design-decisions.md, docs/TESTING.md, docs/feature-spec.md, and the system flow diagrams.",
                    "Implementing boilerplate and repetitive structures (TypedDict definitions, factory function patterns, context manager wrappers) that follow from the design.",
                ],
            },
        ],
    },
    {
        "heading": "2.  Where Human Judgment Was Most Critical",
        "accent": ACCENT,
        "paras": [
            {
                "subhead": "Workflow design",
                "body": (
                    "The decision to use LangGraph rather than a simple function chain was mine. "
                    "I identified that the pipeline needed conditional routing -- clean frames should "
                    "skip alerting entirely, ambiguous rule hits should escalate to the LLM, and "
                    "every frame regardless of outcome should be logged. A linear chain cannot express "
                    "this cleanly. I specified the node boundary contract (AgentState TypedDict) and "
                    "the routing function signature before any code was written."
                ),
            },
            {
                "subhead": "Prompt engineering",
                "body": (
                    "The LLM judge prompt required several iterations. The initial version returned "
                    "free-form text; I restructured it to enforce a rigid three-field format "
                    "(VERDICT: / ALERT: / REASON:) so the parsing logic could be deterministic. "
                    "The query intent parser prompt evolved significantly -- I added explicit mapping "
                    "rules for class name synonyms (people/person/pedestrian -> 'person'), the "
                    "instruction to return null rather than an empty string, and the telemetry filter "
                    "fields (battery_below, altitude_above, flight_mode_filter) as requirements grew. "
                    "Getting the LLM to return clean JSON without markdown fences required explicit "
                    "negative instructions in the prompt."
                ),
            },
            {
                "subhead": "Debugging",
                "body": (
                    "Three non-trivial bugs required diagnosis before Claude Code could fix them. "
                    "First, all in-memory SQLite tests were failing with 'no such table: frames' -- "
                    "I identified that sqlite3.connect(':memory:') creates a fresh empty database on "
                    "every call, so the schema applied in __init__ was invisible to subsequent queries. "
                    "The fix (caching self._mem_conn for in-memory databases) was my diagnosis. "
                    "Second, the LLM judge tests were failing because ChatBedrock was instantiated at "
                    "graph-build time, not at invoke time, so patching after build_agent() was called "
                    "had no effect. I identified the instantiation timing issue and directed the fix. "
                    "Third, loitering tests never triggered because the contextualize node's internal "
                    "_track_first_seen closure resets on every new graph invocation -- I designed the "
                    "two-invoke test pattern (seed the track, then invoke 65 seconds later) to "
                    "correctly exercise the accumulation logic."
                ),
            },
            {
                "subhead": "Architecture decisions",
                "body": (
                    "Every major architectural choice was mine: the stationary-drone assumption and "
                    "its documented implications for future GPS geofencing, the hybrid SQL + CLIP "
                    "retrieval routing strategy, the async VLM thread pool design, the "
                    "drop_on_full=True vs. False distinction between RTSP and file sources, and the "
                    "alert deduplication cooldown mechanism. Claude Code implemented these decisions "
                    "once they were specified, but did not originate them."
                ),
            },
        ],
    },
    {
        "heading": "3.  Specific Claude Code Interactions",
        "accent": ACCENT,
        "examples": [
            {
                "prompt_summary": "Design the LangGraph pipeline with six nodes and conditional routing based on a needs_llm flag.",
                "what_i_specified": "Node names, AgentState fields each node reads/writes, routing rules (needs_llm=True -> llm_judge, False -> alert, always -> log), and the factory function pattern (make_perceive_node returns perceive).",
                "what_ai_generated": "Full implementation of nodes.py including all six factory functions, the _summarise_detections and _extract_field helpers, and the async VLM closure pattern.",
                "what_i_changed": "Adjusted the perceive node to extract telemetry from packet.metadata and return it in state. Added the telemetry_summary block to the contextualize node's context string. Reviewed and corrected the cooldown logic in the alert node.",
            },
            {
                "prompt_summary": "Implement TelemetrySimulator with flight phase timeline, realistic battery/altitude/GPS/signal models, and named scenarios.",
                "what_i_specified": "The field names, the phase sequence (TAKEOFF -> PATROL -> HOVER -> RETURN -> LANDING), the discharge model (% per minute), circular GPS orbit, get_telemetry(ts) API, and the four named scenarios with their parameter sets.",
                "what_ai_generated": "Full implementation of simulator.py including all phase interpolation methods, ease-in/ease-out curves, GPS orbit calculation, battery temperature model, and signal degradation with 2% interference probability.",
                "what_i_changed": "Added the telemetry_summary string key to the output dict so it could be injected directly into the chatbot context. Verified the phase timeline boundary conditions.",
            },
            {
                "prompt_summary": "Write 124 tests covering SQLiteStore, HybridRetriever, and the full LangGraph pipeline.",
                "what_i_specified": "The test scenarios (loitering two-invoke pattern, LLM judge false positive path, SQL fallback, cooldown deduplication), the stub model classes, and the diagnosis of the three bugs described above.",
                "what_ai_generated": "All 124 test functions, fixture setup, the StubDetector / StubCaptioner / StubEmbedder classes, and the _make_preprocessed helper.",
                "what_i_changed": "Directed the fix for the :memory: connection bug. Moved build_agent() inside the ChatBedrock patch context. Rewrote the loitering test to use the two-invoke pattern after diagnosing why it never triggered.",
            },
            {
                "prompt_summary": "Build the Streamlit dashboard with Live Feed, Timeline, and Ask tabs.",
                "what_i_specified": "The three-tab layout, the Live tab's dual-column structure (video + alerts), the telemetry strip widget design, the Ask tab's intent-parse -> retrieve -> answer flow, and all prompt text for the LLM calls.",
                "what_ai_generated": "The full streamlit_app.py including CSS styling, the StreamProcessor integration loop, the _parse_query_intent / _build_smart_context / _bedrock_rag_answer functions, and the source frames expander.",
                "what_i_changed": "Added the telemetry scenario selector to the sidebar. Designed the telemetry metrics strip HTML. Extended _parse_query_intent with telemetry filter fields after deciding the chatbot should handle battery/altitude queries.",
            },
        ],
    },
    {
        "heading": "4.  Honest Assessment",
        "accent": ACCENT,
        "body": (
            "Claude Code generated the majority of the implementation code in this project. "
            "That is an honest statement and it reflects how modern AI-assisted development works "
            "in practice.\n\n"
            "What I brought to this project was the engineering judgment that made the code worth "
            "generating: the system design, the workflow logic, the identification of edge cases and "
            "failure modes, the prompt engineering that made the LLM components reliable, and the "
            "debugging that fixed the non-obvious issues that no amount of code generation would "
            "have resolved on its own.\n\n"
            "AI code generation is most useful -- and most dangerous -- when the human directing "
            "it understands what correct looks like. In this project, the design decisions (LangGraph "
            "routing, hybrid retrieval, async VLM, stationary-drone assumption) required domain "
            "reasoning that the AI did not originate. The implementation of those decisions, once "
            "fully specified, was generated efficiently by Claude Code.\n\n"
            "The skill being demonstrated here is not the ability to write every line by hand -- "
            "it is the ability to design a non-trivial system, decompose it into well-specified "
            "components, direct an AI tool to implement them correctly, and identify and fix the "
            "cases where the generated output is wrong."
        ),
    },
]


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class WhitePDF(FPDF):

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*WHITE)
        self.rect(0, 0, 210, 14, "F")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MID_GREY)
        self.set_xy(15, 4)
        self.cell(0, 6, "AI Tooling & Human Collaboration  |  Drone Security System", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RULE_CLR)
        self.line(15, 13, 195, 13)
        self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(*RULE_CLR)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MID_GREY)
        self.cell(0, 8, f"Page {self.page_no()}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ------------------------------------------------------------------

    def cover(self):
        self.add_page()
        self.set_fill_color(*WHITE)
        self.rect(0, 0, 210, 297, "F")

        # Top accent bar
        self.set_fill_color(*ACCENT)
        self.rect(0, 0, 210, 5, "F")

        # Document label
        self.set_y(38)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*MID_GREY)
        self.cell(0, 6, "PROJECT DOCUMENTATION", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

        # Main title
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*BLACK)
        self.cell(0, 12, "AI Tooling & Human Collaboration", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Subtitle
        self.set_font("Helvetica", "", 13)
        self.set_text_color(*DARK_GREY)
        self.cell(0, 9, "How Claude Code Assisted the Development of the", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 9, "Drone Security System", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Divider
        self.ln(6)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(10)

        # Intro paragraph in a light box
        self.set_fill_color(*LIGHT_GREY)
        self.set_draw_color(*RULE_CLR)
        self.set_line_width(0.3)
        box_y = self.get_y()
        box_h = 38
        self.rect(20, box_y, 170, box_h, "FD")
        self.set_xy(26, box_y + 5)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK_GREY)
        self.set_right_margin(26)
        self.multi_cell(158, 5.8, INTRO, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_right_margin(15)

        # Key highlights row
        self.set_y(box_y + box_h + 14)
        highlights = [
            ("Human-led", "Architecture design\nWorkflow & logic\nPrompt engineering\nDebugging"),
            ("AI-assisted", "Code generation\nTest scaffolding\nDocumentation\nBoilerplate"),
            ("Tool used", "Claude Code\nby Anthropic"),
        ]
        col_w = 54
        col_gap = 4
        start_x = (210 - (col_w * 3 + col_gap * 2)) / 2
        for i, (title, body) in enumerate(highlights):
            x = start_x + i * (col_w + col_gap)
            y = self.get_y()
            self.set_fill_color(*LIGHT_GREY)
            self.set_draw_color(*RULE_CLR)
            self.rect(x, y, col_w, 38, "FD")
            # Coloured top strip
            if i == 0:
                self.set_fill_color(*ACCENT)
            elif i == 1:
                self.set_fill_color(*ACCENT2)
            else:
                self.set_fill_color(*DARK_GREY)
            self.rect(x, y, col_w, 5, "F")
            self.set_font("Helvetica", "B", 8.5)
            self.set_text_color(*WHITE)
            self.set_xy(x, y + 0.5)
            self.cell(col_w, 5, title, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*DARK_GREY)
            self.set_xy(x + 4, y + 9)
            self.set_right_margin(210 - x - col_w + 4)
            self.multi_cell(col_w - 8, 5.5, body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_right_margin(15)

        # Date
        self.set_y(270)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*MID_GREY)
        self.cell(0, 6, f"Drone Security System  |  {datetime.now().strftime('%d %B %Y')}", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Bottom accent bar
        self.set_fill_color(*ACCENT)
        self.rect(0, 292, 210, 5, "F")

    # ------------------------------------------------------------------

    def section_heading(self, text, accent_color=ACCENT):
        self.ln(4)
        # Left accent bar
        bar_y = self.get_y()
        self.set_fill_color(*accent_color)
        self.rect(15, bar_y, 3, 8, "F")
        self.set_xy(21, bar_y)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*BLACK)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RULE_CLR)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)

    def subheading(self, text, color=ACCENT):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.set_x(15)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK_GREY)

    def body_text(self, text, indent=15):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK_GREY)
        self.set_x(indent)
        self.set_right_margin(15)
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            self.set_x(indent)
            self.multi_cell(195 - indent, 5.8, para, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)

    def bullet(self, text, indent=20, bullet_char="-"):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK_GREY)
        # Bullet symbol
        self.set_x(indent)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*ACCENT)
        self.cell(5, 5.8, bullet_char, new_x=XPos.END, new_y=YPos.LAST)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK_GREY)
        self.set_right_margin(15)
        self.multi_cell(170 - indent, 5.8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def label_badge(self, text, color):
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        w = self.get_string_width(text) + 8
        self.set_x(15)
        self.rect(15, self.get_y(), w, 6.5, "F")
        self.set_xy(15, self.get_y() + 0.5)
        self.cell(w, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def example_card(self, example):
        card_x  = 15
        card_y  = self.get_y()
        card_w  = 180
        # Estimate height — rough
        self.set_fill_color(248, 249, 252)
        self.set_draw_color(*RULE_CLR)
        self.set_line_width(0.3)
        self.rect(card_x, card_y, card_w, 4, "F")   # placeholder, redrawn after

        # Prompt summary header
        self.set_fill_color(230, 238, 255)
        self.rect(card_x, card_y, card_w, 8, "F")
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*ACCENT)
        self.set_xy(card_x + 3, card_y + 1)
        self.cell(card_w - 6, 6, "Prompt / task given to Claude Code:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*DARK_GREY)
        self.set_x(card_x + 4)
        self.set_right_margin(card_x + 4)
        start_y = self.get_y()
        self.multi_cell(card_w - 8, 5.5, example["prompt_summary"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

        rows = [
            ("What I specified",    example["what_i_specified"],    ACCENT),
            ("What AI generated",   example["what_ai_generated"],   ACCENT2),
            ("What I adjusted",     example["what_i_changed"],      DARK_GREY),
        ]
        for label, content, col in rows:
            self.set_x(card_x + 4)
            self.set_font("Helvetica", "B", 8.5)
            self.set_text_color(*col)
            self.cell(40, 5.5, label + ":", new_x=XPos.END, new_y=YPos.LAST)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*DARK_GREY)
            self.set_right_margin(card_x + 4)
            self.multi_cell(card_w - 48, 5.5, content, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(1)

        self.set_right_margin(15)
        # Draw card border retroactively
        card_h = self.get_y() - card_y + 3
        self.set_draw_color(*RULE_CLR)
        self.rect(card_x, card_y, card_w, card_h, "D")
        self.ln(5)

    # ------------------------------------------------------------------

    def build(self, output_path):
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 18, 15)

        # -- Cover --
        self.cover()

        # -- Content pages --
        self.add_page()

        for sec in SECTIONS:
            self.section_heading(sec["heading"], sec.get("accent", ACCENT))

            # Section type: role division (two labelled bullet lists)
            if "items" in sec:
                for item in sec["items"]:
                    self.label_badge(item["label"], item["color"])
                    for b in item["bullets"]:
                        self.bullet(b)
                    self.ln(3)

            # Section type: subheaded paragraphs
            elif "paras" in sec:
                for para in sec["paras"]:
                    self.subheading(para["subhead"])
                    self.body_text(para["body"], indent=18)
                    self.ln(1)

            # Section type: example cards
            elif "examples" in sec:
                for ex in sec["examples"]:
                    self.example_card(ex)

            # Section type: plain body text
            elif "body" in sec:
                self.body_text(sec["body"], indent=18)

        self.output(output_path)
        print(f"Written: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pdf = WhitePDF(orientation="P", unit="mm", format="A4")
    pdf.build("AI_Tooling_Document.pdf")

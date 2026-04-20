"""
generate_future_work.py
-----------------------
Generates the Future Work & Enhancements document as a clean white-background PDF.
Run: uv run python generate_future_work.py
Output: Future_Work.pdf
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

BLACK       = (15, 15, 15)
DARK_GREY   = (50, 50, 60)
MID_GREY    = (110, 115, 130)
LIGHT_GREY  = (235, 237, 240)
WHITE       = (255, 255, 255)
RULE_CLR    = (210, 213, 220)
BG_CARD     = (248, 249, 252)

# Per-section accent colours
C_ALERT     = (190, 50,  50)   # red    -- alert channels
C_AGENT     = (30,  110, 190)  # blue   -- self-correction agent
C_UI        = (100, 60,  180)  # purple -- UI
C_DOCS      = (0,   140, 100)  # teal   -- docs
C_DEMO      = (180, 110, 0)    # amber  -- demo video
C_HEADER    = (25,  35,  55)   # dark navy -- cover


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

INTRO = (
    "The Drone Security System was built under tight time constraints. This document "
    "captures the enhancements that would have been prioritised given more time -- "
    "not as wish-list items, but as concrete, well-reasoned extensions that follow "
    "directly from the current architecture and address real gaps in the system's "
    "operational readiness."
)

SECTIONS = [
    # -----------------------------------------------------------------------
    {
        "title": "1. Multi-Channel Alert Delivery",
        "icon": "[ALERTS]",
        "accent": C_ALERT,
        "summary": (
            "The current system stores alerts in SQLite and surfaces them in the "
            "Streamlit dashboard. In a production deployment, a security operator "
            "cannot be expected to watch a browser tab continuously. Alerts need to "
            "reach the operator wherever they are, through redundant channels."
        ),
        "items": [
            {
                "heading": "Email (SMTP / SendGrid)",
                "body": (
                    "Each HIGH severity alert would trigger an email to a configurable "
                    "recipient list. The email would include the alert message, timestamp, "
                    "zone, severity, and an attached JPEG of the evidence frame. "
                    "Implementation: a thin AlertDispatcher class wrapping smtplib or the "
                    "SendGrid Python SDK, called from the alert node after sqlite.store_alert(). "
                    "Rate-limiting would mirror the existing 5-minute cooldown to prevent "
                    "inbox flooding."
                ),
            },
            {
                "heading": "SMS (Twilio)",
                "body": (
                    "A one-line SMS summary for critical alerts (battery critical, perimeter "
                    "breach) delivered via the Twilio REST API. SMS is the highest-reliability "
                    "fallback channel when internet connectivity is degraded. The message body "
                    "would be: '[HIGH] perimeter_breach at main_gate 14:23 -- person detected "
                    "near fence. Review dashboard.' Configurable phone numbers stored in .env."
                ),
            },
            {
                "heading": "Slack / Microsoft Teams webhook",
                "body": (
                    "A formatted Slack Block Kit message posted to a security channel. "
                    "This is the most natural channel for a team of operators -- alerts appear "
                    "in chronological order alongside normal team communications, can be "
                    "threaded for discussion, and Slack's mobile app provides push notifications "
                    "reliably. The evidence frame thumbnail would be attached as an image block. "
                    "Teams support would use the Incoming Webhook connector with an Adaptive Card."
                ),
            },
            {
                "heading": "Push notifications (Firebase Cloud Messaging)",
                "body": (
                    "For a mobile companion app scenario, FCM would deliver a push notification "
                    "to the operator's phone with a tap-to-open deep link into the Timeline tab "
                    "filtered to the relevant event. This requires a small React Native or Flutter "
                    "shell app, but the backend integration is straightforward: one FCM API call "
                    "per alert in the dispatcher."
                ),
            },
            {
                "heading": "Dispatcher architecture",
                "body": (
                    "All channels would be managed by a single AlertDispatcher class configured "
                    "via rules.yaml -- each rule could specify which channels to use "
                    "(e.g., battery_critical: [email, sms, slack]; loitering: [slack]). "
                    "Channel failures would be caught and logged without blocking the pipeline. "
                    "A delivery receipt table in SQLite would track which channels confirmed "
                    "delivery, enabling retry logic for failed sends."
                ),
            },
        ],
    },

    # -----------------------------------------------------------------------
    {
        "title": "2. Self-Correcting Rule Agent",
        "icon": "[AGENT]",
        "accent": C_AGENT,
        "summary": (
            "The current rule engine is static -- rules are written once by the operator "
            "and remain fixed until manually edited. In practice, every deployment site has "
            "different lighting conditions, traffic patterns, and threat profiles that cause "
            "rules to generate false positives or miss real events. A self-correcting agent "
            "would observe the system's own performance and propose rule adjustments."
        ),
        "items": [
            {
                "heading": "False positive feedback loop",
                "body": (
                    "When an operator marks an alert as a false positive in the Timeline tab "
                    "(currently only 'ack' is implemented), that feedback would be recorded with "
                    "the associated rule name, zone, time-of-day, and detected object classes. "
                    "After accumulating N false positives for a rule, a correction agent would "
                    "wake up, retrieve the flagged frames via HybridRetriever, and use an LLM "
                    "to analyse the pattern: 'This loitering rule fires on maintenance staff "
                    "entering the west gate every morning between 07:00 and 08:00. Suggest "
                    "adding an after_hours condition or a zone exclusion.'"
                ),
            },
            {
                "heading": "Missed event detection",
                "body": (
                    "Operators who manually acknowledge a situation in the real world but find "
                    "no corresponding alert in the system would flag it via a 'missed event' "
                    "button. The agent would retrieve frames from that time window, analyse "
                    "what was visible, and propose a new rule or a threshold adjustment to "
                    "catch it in future. This closes the feedback loop in both directions."
                ),
            },
            {
                "heading": "Automated rule proposal",
                "body": (
                    "The agent would output proposed YAML rule diffs -- not automatically "
                    "applied, but presented to the operator in a 'Suggested Rule Changes' panel "
                    "for review and one-click approval. This keeps a human in the loop for all "
                    "policy changes while dramatically reducing the manual effort of rule tuning. "
                    "The RuleEngine.reload() hot-reload mechanism is already in place to apply "
                    "approved changes without a restart."
                ),
            },
            {
                "heading": "Threshold auto-calibration",
                "body": (
                    "Numeric thresholds (loitering duration, crowd size, vehicle count) could "
                    "be calibrated per zone based on baseline activity patterns. A zone with "
                    "naturally high footfall (a public entrance) would get a higher crowd "
                    "threshold than a restricted server room. The calibration agent would run "
                    "weekly, analyse the prior week's detection distribution, and propose "
                    "threshold updates. SQLite already stores all the data needed for this -- "
                    "no new storage infrastructure required."
                ),
            },
        ],
    },

    # -----------------------------------------------------------------------
    {
        "title": "3. Enhanced User Interface",
        "icon": "[UI]",
        "accent": C_UI,
        "summary": (
            "The current Streamlit dashboard is functional but uses Streamlit's page model, "
            "which re-runs the full script on every interaction and has limited real-time "
            "capability. A production security interface would need a more responsive, "
            "purpose-built frontend."
        ),
        "items": [
            {
                "heading": "React / Next.js frontend with WebSocket live feed",
                "body": (
                    "A dedicated React frontend would connect to a FastAPI backend via "
                    "WebSockets. The backend would push frame results, telemetry updates, "
                    "and alerts as JSON events. The frontend would render the annotated "
                    "video using a canvas element, updating at the full processed FPS without "
                    "the full-page rerun overhead of Streamlit. This would make the live feed "
                    "feel genuinely real-time rather than polling-based."
                ),
            },
            {
                "heading": "Multi-camera grid view",
                "body": (
                    "A 2x2 or 3x3 camera grid showing simultaneous feeds from multiple drone "
                    "sources or fixed cameras. Each tile would show a thumbnail, the active "
                    "detections count, and a red border if an alert is active. Clicking a tile "
                    "would expand it to the full single-camera view. The StreamProcessor "
                    "architecture already supports multiple independent instances -- the "
                    "frontend grid would be the main missing piece."
                ),
            },
            {
                "heading": "Interactive zone map",
                "body": (
                    "A top-down site map (SVG or Leaflet.js) overlaid with the defined security "
                    "zones as coloured polygons. Live detection counts and recent alert counts "
                    "per zone would be shown as badges on the map. Clicking a zone would filter "
                    "the Timeline and Ask tabs to that zone automatically. GPS coordinates from "
                    "the telemetry stream would show the drone's current position as a moving icon."
                ),
            },
            {
                "heading": "Alert acknowledgement workflow",
                "body": (
                    "The current Ack button is a single click with no follow-up. A production "
                    "workflow would require the operator to select an outcome (false positive, "
                    "investigated, escalated, resolved) and optionally add a note. This audit "
                    "trail feeds directly into the self-correcting rule agent described above "
                    "and provides a compliance record for security audits."
                ),
            },
            {
                "heading": "Mobile-responsive operator view",
                "body": (
                    "A simplified mobile layout showing only the active alert feed, a one-tap "
                    "acknowledge button, and the most recent annotated frame. Designed to be "
                    "used on a phone when the operator is on-site and away from a desktop. "
                    "Progressive Web App packaging would allow it to be installed and receive "
                    "push notifications via the FCM channel described above."
                ),
            },
        ],
    },

    # -----------------------------------------------------------------------
    {
        "title": "4. Documentation & Developer Experience",
        "icon": "[DOCS]",
        "accent": C_DOCS,
        "summary": (
            "The current documentation covers architecture, design decisions, testing, and "
            "AI tooling. With more time, the focus would shift to operational documentation "
            "that enables another developer or operator to set up, configure, and extend "
            "the system without needing to read the source code."
        ),
        "items": [
            {
                "heading": "Interactive API reference (Swagger / MkDocs)",
                "body": (
                    "A FastAPI backend would expose all pipeline operations as REST endpoints "
                    "(ingest video, query footage, acknowledge alert, reload rules) with "
                    "auto-generated Swagger documentation. MkDocs Material would host the "
                    "full developer reference including auto-generated docstring pages for "
                    "every public class and function, cross-linked with the architecture diagrams."
                ),
            },
            {
                "heading": "Operator runbook",
                "body": (
                    "A step-by-step guide written for a non-technical security operator: "
                    "how to start the system, what each alert severity means, how to "
                    "acknowledge and escalate events, how to ask questions in the Ask tab, "
                    "and what to do when the drone battery warning fires. Written in plain "
                    "language with annotated screenshots, not code snippets."
                ),
            },
            {
                "heading": "Rule authoring guide",
                "body": (
                    "A dedicated guide for security managers who need to write or modify rules "
                    "in rules.yaml. Would cover every supported condition type with examples "
                    "(time_range, class_count, duration_above, battery_below, caption_contains), "
                    "the needs_llm escalation flag, severity levels, message template variables, "
                    "and common patterns (after-hours rules, zone-specific rules, telemetry rules). "
                    "Would include a rule validation script that checks the YAML for schema "
                    "errors before deployment."
                ),
            },
            {
                "heading": "Docker Compose one-command setup",
                "body": (
                    "A docker-compose.yml that brings up all services -- the pipeline worker, "
                    "the Streamlit dashboard, the alert dispatcher, and a ChromaDB container -- "
                    "with a single 'docker compose up'. Currently requires manual environment "
                    "setup and assumes a local Python environment. Containerisation would make "
                    "deployment reproducible and portable, and would enable cloud deployment "
                    "on AWS ECS or Azure Container Apps with minimal additional configuration."
                ),
            },
            {
                "heading": "End-to-end scenario tests with synthetic video",
                "body": (
                    "Programmatically generated synthetic video clips (a moving bounding box "
                    "simulating a person loitering, a sequence of vehicle entries) would enable "
                    "fully deterministic end-to-end tests that verify the complete pipeline from "
                    "raw video input to stored alert output. These tests would be included in CI "
                    "and would catch regressions in the rule engine, LangGraph routing, and "
                    "storage layer in a single automated run."
                ),
            },
        ],
    },

    # -----------------------------------------------------------------------
    {
        "title": "5. High-Quality Demo Video",
        "icon": "[DEMO]",
        "accent": C_DEMO,
        "summary": (
            "The current demo video is a screen recording with voiceover covering the core "
            "functionality. A polished submission-quality demo would communicate the system's "
            "value more effectively and be appropriate for a broader audience including "
            "non-technical stakeholders."
        ),
        "items": [
            {
                "heading": "Edited multi-scene structure",
                "body": (
                    "Rather than a single continuous screen recording, the video would be "
                    "edited into clearly labelled scenes with title cards: Problem Statement, "
                    "Architecture Overview, Live Detection Demo, Alert Escalation, Chatbot Query, "
                    "Telemetry Monitoring, and Summary. Scene transitions would be clean cuts "
                    "with a brief title overlay. Total runtime would be kept under eight minutes "
                    "with tighter pacing than an unedited recording allows."
                ),
            },
            {
                "heading": "Real drone footage",
                "body": (
                    "The demo would use real aerial footage captured by a consumer drone over "
                    "a property, rather than a stock video file. This would demonstrate that "
                    "the YOLOv8 detection pipeline performs correctly on real drone perspective "
                    "footage (top-down angle, motion blur, varying altitude) and would make "
                    "the security use case immediately intuitive to viewers."
                ),
            },
            {
                "heading": "Annotated screen captures",
                "body": (
                    "Key moments in the video would be paused and annotated with callout arrows "
                    "pointing to the relevant UI elements being discussed -- for example, an arrow "
                    "pointing to the track ID badge on a bounding box while explaining loitering "
                    "detection, or a callout highlighting the telemetry battery percentage "
                    "turning red as it drops below the threshold."
                ),
            },
            {
                "heading": "Scripted alert scenario",
                "body": (
                    "Rather than waiting for a rule to fire organically during the demo, a "
                    "scripted scenario video would be prepared: a short clip where a person "
                    "enters a forbidden zone, the YOLO tracker assigns a track ID, the loitering "
                    "timer accumulates, the rule fires, the LLM judge confirms it as genuine, "
                    "and the alert appears in the dashboard and pings a Slack channel -- all "
                    "within 90 seconds. This gives a clean, repeatable narrative arc."
                ),
            },
            {
                "heading": "Professional audio",
                "body": (
                    "The voiceover would be re-recorded in a quiet environment using a dedicated "
                    "microphone with a pop filter, rather than a laptop microphone. Light "
                    "post-processing (noise reduction, normalisation) would be applied. "
                    "A short background music bed at low volume during non-narrated transitions "
                    "would improve perceived production quality significantly."
                ),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class FuturePDF(FPDF):

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*WHITE)
        self.rect(0, 0, 210, 14, "F")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MID_GREY)
        self.set_xy(15, 4)
        self.cell(0, 6, "Future Work & Enhancements  |  Drone Security System",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RULE_CLR)
        self.line(15, 13, 195, 13)
        self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(*RULE_CLR)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MID_GREY)
        self.cell(0, 8, f"Page {self.page_no()}", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ------------------------------------------------------------------
    # Cover page
    # ------------------------------------------------------------------

    def cover(self):
        self.add_page()

        # Solid dark navy top band
        self.set_fill_color(*C_HEADER)
        self.rect(0, 0, 210, 72, "F")

        # Thin accent line under band
        self.set_fill_color(*C_ALERT)
        self.rect(0, 72, 70, 2, "F")
        self.set_fill_color(*C_AGENT)
        self.rect(70, 72, 70, 2, "F")
        self.set_fill_color(*C_UI)
        self.rect(140, 72, 70, 2, "F")

        # Title
        self.set_y(18)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(160, 170, 190)
        self.cell(0, 6, "DRONE SECURITY SYSTEM  |  PLANNING DOCUMENT", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

        self.set_font("Helvetica", "B", 23)
        self.set_text_color(*WHITE)
        self.cell(0, 13, "Future Work & Enhancements", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Helvetica", "", 12)
        self.set_text_color(180, 195, 220)
        self.cell(0, 8, "What we would build given more time", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Five topic pills on dark band
        self.ln(6)
        topics = [
            ("Alert Channels",   C_ALERT),
            ("Self-Correction",  C_AGENT),
            ("Better UI",        C_UI),
            ("Documentation",    C_DOCS),
            ("Demo Video",       C_DEMO),
        ]
        total_w = sum(self.get_string_width(t) + 14 for t, _ in topics) + 8 * (len(topics)-1)
        x = (210 - total_w) / 2
        y = self.get_y()
        self.set_font("Helvetica", "B", 8)
        for label, color in topics:
            w = self.get_string_width(label) + 14
            self.set_fill_color(*color)
            self.rect(x, y, w, 8, "F")
            self.set_text_color(*WHITE)
            self.set_xy(x, y + 0.8)
            self.cell(w, 6.5, label, align="C")
            x += w + 8

        # Intro box
        self.set_y(86)
        bx, bw, bh = 18, 174, 32
        self.set_fill_color(*LIGHT_GREY)
        self.set_draw_color(*RULE_CLR)
        self.set_line_width(0.3)
        self.rect(bx, self.get_y(), bw, bh, "FD")
        self.set_xy(bx + 5, self.get_y() + 5)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK_GREY)
        self.set_right_margin(bx + 5)
        self.multi_cell(bw - 10, 5.8, INTRO,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_right_margin(15)

        # Summary cards
        self.set_y(130)
        cards = [
            ("1", "Alert\nChannels",  C_ALERT,
             "Email, SMS, Slack,\nTeams, push notifications"),
            ("2", "Self-Correct\nAgent",   C_AGENT,
             "Feedback loop, auto\nrule proposals, calibration"),
            ("3", "Enhanced\nUI",     C_UI,
             "React frontend, zone map,\nmulti-camera grid"),
            ("4", "Docs &\nDX",       C_DOCS,
             "API reference, runbook,\nDocker, CI tests"),
            ("5", "Demo\nVideo",      C_DEMO,
             "Real footage, edited\nscenes, scripted scenario"),
        ]
        cw = 32
        gap = 3.5
        sx = (210 - (cw * 5 + gap * 4)) / 2
        for i, (num, title, col, desc) in enumerate(cards):
            cx = sx + i * (cw + gap)
            cy = self.get_y()
            ch = 52
            self.set_fill_color(*WHITE)
            self.set_draw_color(*RULE_CLR)
            self.rect(cx, cy, cw, ch, "FD")
            # colour top strip
            self.set_fill_color(*col)
            self.rect(cx, cy, cw, 6, "F")
            # number badge
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*WHITE)
            self.set_xy(cx, cy + 0.5)
            self.cell(cw, 5, num, align="C")
            # title
            self.set_font("Helvetica", "B", 7.5)
            self.set_text_color(*col)
            self.set_xy(cx + 1, cy + 9)
            self.set_right_margin(210 - cx - cw + 1)
            self.multi_cell(cw - 2, 5, title,
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            # desc
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*DARK_GREY)
            self.set_xy(cx + 2, cy + 23)
            self.set_right_margin(210 - cx - cw + 2)
            self.multi_cell(cw - 4, 4.5, desc,
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.set_right_margin(15)

        # Date
        self.set_y(268)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*MID_GREY)
        self.cell(0, 6,
                  f"Drone Security System  |  {datetime.now().strftime('%d %B %Y')}",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ------------------------------------------------------------------
    # Section builder
    # ------------------------------------------------------------------

    def section_header(self, icon, title, color, summary):
        self.add_page()
        # Coloured band at top of section page
        self.set_fill_color(*color)
        self.rect(0, 16, 210, 18, "F")
        self.set_y(18)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*WHITE)
        self.set_x(15)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(200, 215, 235)
        self.set_x(15)
        self.cell(0, 5, icon + "  If we had no time constraints...",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(6)

        # Summary paragraph
        self.set_font("Helvetica", "I", 9.5)
        self.set_text_color(*DARK_GREY)
        self.set_fill_color(248, 249, 252)
        self.set_draw_color(*RULE_CLR)
        sy = self.get_y()
        self.rect(15, sy, 180, 1, "F")   # placeholder
        self.set_xy(20, sy + 4)
        self.set_right_margin(20)
        self.multi_cell(170, 5.8, summary,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        sh = self.get_y() - sy + 5
        self.set_fill_color(248, 249, 252)
        self.set_draw_color(*RULE_CLR)
        self.rect(15, sy, 180, sh, "FD")
        self.set_xy(20, sy + 4)
        self.set_right_margin(20)
        self.multi_cell(170, 5.8, summary,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_right_margin(15)
        self.ln(5)

    def item_block(self, heading, body, color):
        # Left colour bar + heading
        bar_y = self.get_y()
        self.set_fill_color(*color)
        self.rect(15, bar_y, 2.5, 7, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.set_xy(20, bar_y)
        self.cell(0, 7, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Body
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK_GREY)
        self.set_x(20)
        self.set_right_margin(15)
        self.multi_cell(175, 5.8, body,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_right_margin(15)
        self.ln(4)

    # ------------------------------------------------------------------

    def build(self, output_path):
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 18, 15)

        self.cover()

        for sec in SECTIONS:
            self.section_header(
                sec["icon"],
                sec["title"],
                sec["accent"],
                sec["summary"],
            )
            for item in sec["items"]:
                self.item_block(item["heading"], item["body"], sec["accent"])

        self.output(output_path)
        print(f"Written: {output_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pdf = FuturePDF(orientation="P", unit="mm", format="A4")
    pdf.build("Future_Work.pdf")

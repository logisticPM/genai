# FinChartAudit — Product Design Document v6

> CS 6180 Generative AI Final Project
> Last updated: 2026-03-20

---

## 1. Project Overview

FinChartAudit is an agentic **cross-modal consistency verification system** for SEC financial filings. It goes beyond single-chart visual inspection to detect inconsistencies **between** charts, text, tables, and regulatory requirements — using VLM agents that autonomously invoke OCR and rule-checking tools.

The system is organized around a **four-tier detection framework** that progresses from simple numerical checks to complex multi-modal reasoning:

```
T1: Numerical Consistency     → Do the numbers match across modalities?
T2: Visual Encoding Integrity → Does the chart faithfully encode the data?
T3: Pairing Completeness      → Are required companion elements present?  ← Case study
T4: Cross-Section Consistency  → Are claims consistent across document sections?
```

The system operates at two temporal scopes:

- **Single-Filing Audit**: T1–T4 checks within a single financial document.
- **Cross-Filing Temporal Audit**: T1–T4 patterns tracked across multiple years to detect progressive deterioration.

### 1.1 Why Cross-Modal?

Existing chart audit approaches treat charts in isolation. But SEC enforcement actions reveal that the most consequential violations occur **between modalities**:

- A chart shows a rosy trend, but the text tells a different story (T1)
- A chart truncates axes to exaggerate growth during a downturn (T2)
- A Non-GAAP chart exists without a required paired GAAP chart (T3)
- A chart's time window conveniently starts after a performance trough, while MD&A discusses that trough (T4)

No single modality — vision or text — can catch these. The system must reason across both.

### 1.2 Supported Models, Tools & Datasets

| Component | Type | Role |
|-----------|------|------|
| Claude Sonnet | Commercial VLM API | Primary reasoning agent |
| Qwen2.5-VL-7B | Open-source VLM, local inference | Baseline / fallback agent |
| PaddleOCR (PP-StructureV3) | Traditional OCR | Precise text + bbox extraction, table/chart structure recognition |
| PyMuPDF | PDF library | Embedded text extraction from non-scanned PDFs (zero error) |
| Misviz-synth | Benchmark (81,814 charts) | Primary evaluation — has data tables (for vision+text) + axis metadata (for OCR ground truth) |
| Misviz (real) | Benchmark (2,604 charts) | Generalization test — vision-only (no data tables) |
| SEC filings (EDGAR) | Real-world application data | T3 case study + T4 demonstration |

---

## 2. Four-Tier Detection Framework

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Detection Tiers                               │
│                                                                  │
│  T1: Numerical Consistency ──────────── Easiest, baseline task   │
│  │   Chart values vs. text values                                │
│  │   Chart values vs. table values                               │
│  │   Current year vs. prior year (cross-filing)                  │
│  │                                                               │
│  T2: Visual Encoding Integrity ──────── VLM + OCR verification   │
│  │   12 Misviz misleader types                                   │
│  │   Axis truncation, 3D distortion, area misrepresentation      │
│  │   Footnote completeness                                       │
│  │                                                               │
│  T3: Pairing Completeness ───────────── Case study        │
│  │   Non-GAAP chart ↔ GAAP chart pairing                        │
│  │   Non-GAAP metric ↔ Reconciliation table pairing             │
│  │   Visual prominence balance (Non-GAAP vs GAAP)                │
│  │   Requires VLM (chart classification) + LLM (rule reasoning)  │
│  │                                                               │
│  T4: Cross-Section Consistency ──────── Frontier, hardest        │
│      Chart time window vs. business cycle                        │
│      Chart metric definition vs. MD&A definition                 │
│      Non-GAAP exclusion items vs. disclosure text                │
│      Cross-filing presentation drift                             │
│      Cross-filing definition drift                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 T1: Numerical Consistency

**Core question**: Do the numbers match across charts, text, and tables?

| Check | Input | Method | SEC Basis |
|-------|-------|--------|-----------|
| Chart value vs. text claim | Chart image + narrative text | VLM reads chart trend + OCR extracts data labels → `extract_pdf_text` gets text claims → `rule_check("value_mismatch")` | SEC sample comment letters cite inconsistent unit disclosures across pages |
| Chart value vs. table value | Chart image + financial table | OCR extracts both → programmatic comparison | Same metric appearing in chart and table should match exactly |
| Prior-year reference vs. actual | Current filing table + prior filing table | `query_memory(scope="historical")` → `rule_check("value_mismatch")` | Undisclosed restatements |

**Why cross-modal matters**: VLMs fail to notice text-chart contradictions in ~99% of cases when given both simultaneously. The system must decompose: extract from each modality separately, then compare programmatically.

### 2.2 T2: Visual Encoding Integrity

**Core question**: Does the chart faithfully represent the underlying data?

| Check | Input | Method | SEC Basis |
|-------|-------|--------|-----------|
| Truncated axis | Chart image | `traditional_ocr(region="y_axis")` → `rule_check("truncated_axis")` | CPA Journal: 86% of S&P 500 companies have at least one broken chart |
| Inverted axis | Chart image | VLM semantic check + OCR axis value ordering verification | 100% of VLMs affected by axis inversion |
| 3D distortion | Chart image | VLM qualitative judgment (OCR cannot help here) | Visual area exaggeration from 3D effects |
| Area/volume misrepresentation | Chart image | VLM qualitative judgment | Bubble/pie charts where area doesn't match values |
| Dual axis abuse | Chart image | `traditional_ocr` both axes → `rule_check` scale comparison | Misleading correlation implications |
| Cherry-picked time window | Chart image + context | `traditional_ocr(region="x_axis")` → cross-reference with business cycle (T4) | Selective omission of unfavorable periods |
| Inappropriate chart type | Chart image | VLM semantic judgment | e.g., pie chart for time-series data |
| Misleading annotations | Chart image | VLM semantic judgment | Arrows, callouts that emphasize favorable data |
| Broken scale | Chart image | `traditional_ocr` axis values → `rule_check` even spacing | Non-linear scales without disclosure |
| Missing baseline | Chart image | VLM + `traditional_ocr` | Growth charts without clear reference point |
| Color/contrast manipulation | Chart image | VLM color recognition (near perfect) | Muting unfavorable data series |
| Footnote completeness | Chart image | `traditional_ocr(region="bottom")` → keyword check for source, date range, units, methodology | Missing required disclosures |

**Tool dependency**: OCR provides the ground truth (axis values, labels) that VLMs cannot reliably extract. Rule engine makes definitive determinations. VLM handles qualitative judgments where no numerical ground truth exists.

### 2.3 T3: Pairing Completeness (Case Study)

**Core question**: When a Non-GAAP element exists, does its required GAAP counterpart also exist with comparable prominence?

This tier is the system's primary academic contribution. It requires **joint VLM + LLM reasoning** that neither can perform alone:

- **VLM alone** can identify a chart as "Adjusted EBITDA" but cannot determine whether a paired GAAP chart exists elsewhere in the document
- **LLM alone** can understand pairing rules but cannot see whether charts exist or assess their visual prominence
- **Together**: VLM classifies all charts → LLM applies pairing rules → system checks completeness across pages

| Check | Input | Method | SEC Basis |
|-------|-------|--------|-----------|
| Non-GAAP chart ↔ GAAP chart | All chart images in filing | VLM classifies each chart as GAAP/Non-GAAP → build pairing matrix → check each Non-GAAP has a GAAP counterpart | SEC: "do not present a non-GAAP measure... without presenting the most directly comparable GAAP measure with equal or greater prominence" |
| Non-GAAP metric ↔ Reconciliation table | Chart images + all pages | VLM identifies Non-GAAP charts → `query_memory` for reconciliation tables → if missing, `extract_pdf_text` searches all pages | SEC Regulation G requires reconciliation |
| Prominence balance | Page screenshots with paired elements | VLM identifies metrics → `traditional_ocr(mode="bbox")` for font sizes and positions → `rule_check("font_size_comparison")` | Non-GAAP must not have "equal or greater prominence" than GAAP |
| Pairing comparability | Paired chart images | VLM compares: same time window? same scale type? same chart type? | Paired charts must be genuinely comparable |

**Detection pipeline for T3**:

```
Step 1: Chart Census
  VLM scans each page → identifies and classifies all charts
  Output: [{chart_id, page, metric_name, is_gaap, chart_type, time_window}, ...]
  Stored in: FilingMemory.chart_registry

Step 2: Pairing Matrix Construction
  For each Non-GAAP chart, find candidate GAAP counterparts:
  - Semantic matching: "Adjusted EBITDA" → "Net Income" or "EBITDA"
  - LLM applies SEC pairing rules to determine the "most directly comparable" GAAP metric
  Output: PairingMatrix with status per Non-GAAP chart

Step 3: Pairing Verification
  For each pair:
  - Existence: Does the GAAP counterpart chart actually exist?
  - Prominence: Is the GAAP chart presented with >= prominence? (OCR bbox for font sizes)
  - Comparability: Same time window? Same chart type? Same page or adjacent?
  - Reconciliation: Does a reconciliation table exist? (text search)

Step 4: Findings
  Missing pair → HIGH risk
  Pair exists but lower prominence → MEDIUM risk
  Pair exists but different time window → MEDIUM risk
  Pair exists and comparable → PASS
```

### 2.4 T4: Cross-Section Consistency (Powered by DSPM)

**Core question**: Are claims, definitions, and presentations consistent across different sections of the document and across years?

**Key insight**: A 10-K filing is structurally analogous to a multi-session conversation — each section (MD&A, Financial Statements, Risk Factors, Non-GAAP Disclosure) discusses the same entities (metrics, events, risks) from different perspectives. We adapt **DSPM (Domain-Structured Personal Memory)**, our four-layer memory architecture, to model this cross-section structure. DSPM's L1 overwrite semantics become a **detection signal**: when the same metric's definition is overwritten by a conflicting value from a later section, the changelog entry itself constitutes an audit finding.

#### DSPM Adaptation for Filing Analysis

| DSPM Layer | Original Purpose | T4 Adaptation |
|-----------|-----------------|---------------|
| **L1: Domain Profile** | Current state KV (overwrite) | **Filing Profile**: each metric's current definition, value, presentation. **Overwrites produce changelog entries → definition inconsistency detection** |
| **L2: Advisory Log** | Per-session key facts (append) | **Section Log**: what each section says about each metric (append-only, preserves all versions) |
| **L3: Event Stream** | Life events with timestamps | **Business Event Timeline**: performance events extracted from MD&A (e.g., "2020 supply chain disruption", "2023 recovery") |
| **L4: Understanding** | Cross-domain narrative (periodic rewrite) | **Cross-Section Synthesis**: auto-generated narrative identifying inconsistencies across sections |
| **Domain Schema** | Keyword-based topic routing | **Section Schema**: routes queries to relevant filing sections (MD&A, Financial Statements, Non-GAAP Disclosure, Risk Factors, Charts) |

#### Filing Domain Schema

```python
FILING_SCHEMA = {
    "mda": DomainDef(
        name="mda",
        description="Management Discussion & Analysis — narrative, trends, outlook",
        keywords=["revenue growth", "operating margin", "outlook", "headwinds",
                  "tailwinds", "year-over-year", "compared to prior", "momentum"],
        fields=["revenue_trend", "margin_trend", "key_risks", "outlook_tone",
                "mentioned_metrics", "time_references", "growth_claims"],
    ),
    "financial_statements": DomainDef(
        name="financial_statements",
        description="Income statement, balance sheet, cash flow — exact numbers",
        keywords=["total revenue", "net income", "operating income",
                  "cash flow", "assets", "liabilities", "earnings per share"],
        fields=["revenue", "net_income", "operating_income", "ebitda",
                "free_cash_flow", "total_assets"],
    ),
    "nongaap_disclosure": DomainDef(
        name="nongaap_disclosure",
        description="Non-GAAP metric definitions, reconciliation, exclusions",
        keywords=["adjusted", "non-gaap", "excluding", "reconciliation",
                  "stock-based compensation", "restructuring", "one-time"],
        fields=["metric_definitions", "excluded_items", "reconciliation_location"],
    ),
    "risk_factors": DomainDef(
        name="risk_factors",
        description="Risk disclosures, uncertainties, forward-looking caveats",
        keywords=["risk", "uncertainty", "may", "could", "forward-looking",
                  "no assurance", "adverse", "decline"],
        fields=["key_risks", "new_risks_this_year", "risk_tone"],
    ),
    "charts_and_visuals": DomainDef(
        name="charts_and_visuals",
        description="Chart properties, visual presentation choices",
        keywords=["chart", "graph", "figure", "illustration", "trend line"],
        fields=["chart_types_used", "time_windows", "axis_treatments",
                "prominence_scores", "nongaap_chart_list", "gaap_chart_list"],
    ),
}
```

#### T4 Detection Sub-Tasks

**T4-A: Time Window vs. Business Cycle**

```
Step 1: T4 Agent queries DSPM:
        filing_dspm.get_memory_context_for_question(
            "What time periods do charts show and what business events happened?"
        )
        → DSPM routes to charts_and_visuals (time windows) + mda (business events)
          + risk_factors (disclosed risks)
        → Returns cross-section context with zero LLM calls

Step 2: From L3 Event Stream, extract business event timeline:
        → [{"year": 2020, "event": "supply chain disruption", "impact": "negative"},
           {"year": 2021, "event": "recovery began", "impact": "positive"}]

Step 3: From chart_registry (T2), get each chart's time window:
        → Revenue chart: 2022-2024 (3 years)

Step 4: rule_check("time_window_vs_events"):
        → Window starts at 2022, excluding 2020-2021 negative events
        → Cross-reference with historical windows (query_memory → TemporalMemory)
        → Prior filings showed 5-year windows → shortened this year

Step 5: VLM synthesizes: shortened window + excluded trough + no disclosed reason
        → Finding: Selective time window (HIGH)
```

**T4-B: Metric Definition Consistency (via DSPM L1 Changelog)**

```
Step 1: DSPM processes each section as a "session":
        filing_dspm.process_conversation(chart_footnote_text, session_id="chart_footnote")
        filing_dspm.process_conversation(mda_text, session_id="mda")
        filing_dspm.process_conversation(reconciliation_text, session_id="reconciliation")

Step 2: L1 overwrites produce changelog entries:
        field: "adjusted_ebitda_exclusions"
        changelog: [
          {"session": "chart_footnote", "value": "SBC, restructuring"},
          {"session": "mda", "value": "SBC, restructuring, acquisition costs"},
          {"session": "reconciliation", "value": "SBC, restructuring, acquisition costs, litigation"}
        ]
        → 3 different definitions across 3 sections!

Step 3: rule_check("definition_consistency"):
        → Inconsistency detected: chart footnote excludes 2 items,
          MD&A excludes 3 items, reconciliation excludes 4 items
        → Finding: Non-GAAP definition inconsistency (HIGH)

        This detection requires ZERO LLM calls — pure L1 changelog comparison.
```

**T4-C: Narrative vs. Data Consistency**

```
Step 1: filing_dspm.get_memory_context_for_question(
            "What does management say about growth trajectory?"
        )
        → Routes to mda domain
        → Returns: "management describes 'strong momentum' and 'accelerating growth'"

Step 2: Query financial_statements domain for actual data:
        → Revenue growth: 15% (2022) → 12% (2023) → 8% (2024)
        → Trend: decelerating, not accelerating

Step 3: VLM compares narrative tone vs. data trend
        → Finding: Narrative contradicts data (MEDIUM)
```

**T4-D: Cross-Filing Drift (via TemporalMemory)**

| Check | Method |
|-------|--------|
| Presentation drift | `query_memory(scope="historical")` for prior snapshots → `rule_check("presentation_drift")` |
| Definition drift | Compare DSPM L1 changelogs across years → `rule_check("nongaap_definition_change")` |
| Time window drift | Compare chart time windows across years → `rule_check("time_window_change")` |

#### Why DSPM Works for T4

| DSPM Property | T4 Benefit |
|--------------|------------|
| **L1 overwrite with changelog** | Definition inconsistencies detected automatically — the changelog IS the finding |
| **L2 append-only** | Preserves every section's exact wording for evidence |
| **L3 event stream** | Business cycle events extracted and timestamped for time window analysis |
| **L4 cross-domain narrative** | Auto-generated summary highlighting cross-section tensions |
| **Zero LLM read path** | T4-B (definition consistency) needs zero LLM calls — pure changelog comparison |
| **Domain keyword routing** | Automatically retrieves relevant cross-section context for VLM judgment |
| **"Compute in code, understand in LLM"** | DSPM's core principle directly applies: structured extraction → programmatic comparison → LLM only for semantic judgment |

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Streamlit Frontend                          │
│                                                                      │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────────┐  │
│  │ Single Chart  │  │ Filing Scanner │  │ Company Timeline         │  │
│  │ Audit Page    │  │ Page           │  │ (Cross-Filing Audit)     │  │
│  └──────┬───────┘  └───────┬────────┘  └────────────┬─────────────┘  │
│         │                  │                        │                │
│  Sidebar: API Key | Model Config | Threshold | Tier Selection        │
└─────────┼──────────────────┼────────────────────────┼────────────────┘
          │                  │                        │
┌─────────▼──────────────────▼────────────────────────▼────────────────┐
│                          Core Engine                                  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                       Memory Layer                               │ │
│  │                                                                  │ │
│  │  ┌─────────────────────┐  ┌──────────────────┐  ┌────────────┐  │ │
│  │  │  FilingMemory        │  │  DSPM (Filing     │  │ Temporal   │  │ │
│  │  │  (in-memory)         │  │  Edition)         │  │ Memory    │  │ │
│  │  │                      │  │                   │  │ (SQLite)  │  │ │
│  │  │  • Document Map      │  │  L1: Filing       │  │           │  │ │
│  │  │  • Chart Registry    │  │  Profile +        │  │ • Company │  │ │
│  │  │  • Pairing Matrix    │  │  Changelog        │  │   Profile │  │ │
│  │  │  • Financial Claims  │  │  L2: Section Log  │  │ • History │  │ │
│  │  │  • GAAP/Non-GAAP     │  │  L3: Business     │  │ • Metrics │  │ │
│  │  │  • OCR Cache         │  │  Event Timeline   │  │ • Deltas  │  │ │
│  │  │  • Findings          │  │  L4: Cross-Section│  │           │  │ │
│  │  │  • Audit Trace       │  │  Synthesis        │  │           │  │ │
│  │  └─────────┬───────────┘  └────────┬──────────┘  └─────┬─────┘  │ │
│  │            │    T1/T2/T3 use       │  T4 uses          │ Trend  │ │
│  │            └───────────────────────┼───────────────────┘ Detect │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│  ┌──────────────────────────▼───────────────────────────────────────┐ │
│  │                  VLM Agent Layer (Tool-Use)                      │ │
│  │                                                                  │ │
│  │   Each agent is a VLM with function-calling capability.         │ │
│  │   Agents autonomously decide when to invoke tools.              │ │
│  │                                                                  │ │
│  │   Orchestrator                                                   │ │
│  │     │                                                            │ │
│  │     ├── T1 Agent: Numerical Consistency Auditor                  │ │
│  │     ├── T2 Agent: Visual Encoding Auditor                        │ │
│  │     ├── T3 Agent: Pairing Completeness Auditor   ← Case study         │ │
│  │     ├── T4 Agent: Cross-Section Auditor                          │ │
│  │     │                                                            │ │
│  │     └── Cross-Validator Agent                                    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│                    Agents call tools ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                       Tool Registry                              │ │
│  │                                                                  │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │ │
│  │  │ extract_       │  │ traditional_   │  │ doc_ocr          │   │ │
│  │  │ pdf_text       │  │ ocr            │  │ (PaddleOCR-VL /  │   │ │
│  │  │ (PyMuPDF)      │  │ (PaddleOCR)    │  │  GLM-OCR)        │   │ │
│  │  └────────────────┘  └────────────────┘  └──────────────────┘   │ │
│  │                                                                  │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │ │
│  │  │ query_memory   │  │ rule_check     │  │ sec_api          │   │ │
│  │  └────────────────┘  └────────────────┘  └──────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                     Report Generator                             │ │
│  │        PDF / HTML with Audit Trace + Risk Narrative              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tool-Use Architecture

### 4.1 Design Philosophy

VLM agents are strong at semantic reasoning but weak at precise data extraction. Traditional OCR is the opposite. Rather than hardcoding a fixed pipeline, we let VLM agents **autonomously decide when they need precise data** and call tools on demand — like a human auditor reaching for a calculator.

This design is grounded in empirical VLM capability research:

| VLM Capability | Reliability | Implication |
|---------------|-------------|-------------|
| Detecting text-chart contradictions (zero-shot) | **~1%** detection rate | Must decompose: extract separately → compare programmatically (T1) |
| Detecting truncated axes | **~30%** success rate | Must use OCR for axis values + rule engine (T2) |
| Classifying chart as GAAP vs Non-GAAP | **Strong** | VLM's semantic strength — basis for T3 pairing |
| Comparing font sizes / visual weight | **Near random** | Must use OCR bbox heights (T3 prominence) |
| Reading numbers from charts | **~5% error** | Use OCR for precision; VLM only for trends (T1) |
| Color recognition | **Near perfect** | Reliable for prominence assessment (T3) |
| Semantic understanding / classification | **Strong** | Core strength for chart classification, metric matching, narrative generation |

### 4.2 Tool Definitions

```python
tools = [
    {
        "name": "extract_pdf_text",
        "description": (
            "Extract embedded text directly from a PDF page. "
            "Zero OCR error. Returns None if scanned. Use FIRST."
        ),
        "parameters": {
            "page": "int — Page number (1-indexed)",
            "extract_tables": "bool — Also extract tables as structured data (default: True)"
        }
    },
    {
        "name": "traditional_ocr",
        "description": (
            "Extract text with precise bounding boxes from an image using PaddleOCR. "
            "Returns text + bbox + estimated font size. "
            "ONLY tool that provides bounding boxes. "
            "Use for: axis labels, data labels, font size comparison, spatial positions."
        ),
        "parameters": {
            "image_id": "str — Reference to the image",
            "region": "str — 'full' | 'y_axis' | 'x_axis' | 'legend' | 'title' | 'bottom' | bbox coordinates",
            "mode": "str — 'text' | 'bbox' | 'table' | 'chart_to_table'"
        }
    },
    {
        "name": "doc_ocr",
        "description": (
            "LLM-based OCR for complex/degraded documents. "
            "More accurate on messy layouts but slower, no bounding boxes. "
            "Use as fallback when traditional_ocr fails."
        ),
        "parameters": {
            "image_id": "str — Reference to the image",
            "mode": "str — 'text' | 'table' | 'markdown'",
            "focus": "str | None — Specific content to focus on"
        }
    },
    {
        "name": "query_memory",
        "description": (
            "Query memory for previously extracted data, findings, or OCR cache. "
            "Check BEFORE calling OCR to avoid redundant extraction."
        ),
        "parameters": {
            "query": "str — What to look for",
            "scope": "str — 'current_filing' | 'historical' | 'all'"
        }
    },
    {
        "name": "rule_check",
        "description": (
            "Run deterministic rule-based validation on extracted data. "
            "More reliable than VLM for quantitative checks. No hallucination."
        ),
        "parameters": {
            "check_type": "str — 'truncated_axis' | 'broken_scale' | 'font_size_comparison' | 'value_mismatch' | 'time_window_change' | 'nongaap_definition_change' | 'pairing_completeness' | 'prominence_score'",
            "data": "object — Extracted data to validate"
        }
    },
    {
        "name": "query_dspm",
        "description": (
            "Query the DSPM Filing Edition for cross-section information. "
            "Zero LLM calls — uses keyword-based domain routing. "
            "Use for T4 checks: definition consistency, narrative comparison, "
            "business event timeline. Also exposes L1 changelog for "
            "automatic definition inconsistency detection."
        ),
        "parameters": {
            "question": "str — Natural language question (routed to relevant filing sections)",
            "mode": "str — 'context' (assembled cross-section context) | 'changelog' (L1 overwrite log) | 'events' (L3 business timeline) | 'synthesis' (L4 cross-section narrative)"
        }
    },
    {
        "name": "sec_api",
        "description": (
            "Query SEC EDGAR for filings, comment letters, or full-text search."
        ),
        "parameters": {
            "action": "str — 'search_filings' | 'get_comment_letters' | 'full_text_search'",
            "query": "str — Search query or CIK"
        }
    }
]
```

### 4.3 Tool Selection Priority

```
1. query_memory     → Already extracted? Use cached data.
2. query_dspm       → Cross-section question? Zero-LLM domain routing. (T4)
3. extract_pdf_text → Embedded text available? Zero error.
4. traditional_ocr  → Need bbox/font size/precise values? PaddleOCR.
5. doc_ocr          → Complex/degraded document? LLM-OCR fallback.
6. rule_check       → Have numbers? Deterministic validation.
```

---

## 5. Agents

### 5.1 Orchestrator

Manages the full audit lifecycle:

1. **Document Parsing**: Extract page images, identify content regions.
2. **T1–T4 Dispatch**: Run tier agents in order (T1/T2 can run in parallel; T3 depends on chart census from T2; T4 depends on T3 findings).
3. **Reflection**: After each agent, decide if follow-up tasks are needed.
4. **Cross-Validation**: Invoke Cross-Validator to correlate findings across tiers.
5. **Reporting**: Generate audit report.

### 5.2 T1 Agent: Numerical Consistency Auditor

- **Tier**: T1 — Numerical Consistency
- **Available tools**: `extract_pdf_text`, `traditional_ocr`, `query_memory`, `rule_check`
- **Capabilities**:
  - Extract numerical claims from narrative text
  - Extract data values from charts (via OCR chart_to_table or data labels)
  - Extract values from financial tables
  - Compare across modalities using `rule_check("value_mismatch")`
- **Key behavior**: Never compares values using VLM judgment. Always extracts from each source independently, then uses `rule_check` for programmatic comparison.

### 5.3 T2 Agent: Visual Encoding Auditor

- **Tier**: T2 — Visual Encoding Integrity
- **Available tools**: `traditional_ocr`, `doc_ocr`, `query_memory`, `rule_check`
- **Capabilities**:
  - Detect 12 Misviz misleader types
  - Check footnote completeness
  - Produce region-level highlight annotations (from OCR bbox)
  - Generate textual correction suggestions
- **Key behavior**: For quantitative checks (truncated axis, broken scale), always uses OCR + rule engine. For qualitative checks (inappropriate chart type, misleading annotations, 3D distortion), uses VLM semantic judgment.
- **Side effect**: Builds `chart_registry` in FilingMemory — every chart is classified by metric, type, and properties. This registry is consumed by T3.

### 5.4 T3 Agent: Pairing Completeness Auditor

- **Tier**: T3 — Pairing Completeness
- **Available tools**: `traditional_ocr`, `extract_pdf_text`, `query_memory`, `rule_check`
- **Capabilities**:
  - Read chart registry from memory (built by T2)
  - Classify each chart as GAAP or Non-GAAP (VLM semantic classification)
  - Apply SEC pairing rules (LLM reasoning)
  - Build and verify pairing matrix
  - Assess prominence balance using OCR bbox data
  - Check reconciliation table existence
- **Key behavior**: This agent uniquely requires both VLM (visual chart classification) and LLM (regulatory rule application) reasoning, plus tool calls for precise measurement. This joint requirement is what makes T3 the system's differentiating contribution.

**Pairing Matrix Example**:

```
┌──────────────────────┬──────────────────┬─────────┬────────────┬──────────────┐
│ Non-GAAP Chart       │ Required GAAP    │ Found?  │ Prominence │ Comparable?  │
│                      │ Counterpart      │         │ Balance    │              │
├──────────────────────┼──────────────────┼─────────┼────────────┼──────────────┤
│ Adjusted EBITDA (p3) │ Net Income       │ Yes(p4) │ 1.75x ⚠   │ Yes          │
│ Free Cash Flow (p7)  │ Operating CF     │ No  ⚠⚠  │ N/A        │ N/A          │
│ Organic Revenue (p5) │ Total Revenue    │ Yes(p5) │ 1.0x ✓    │ No (diff     │
│                      │                  │         │            │ time window)⚠│
└──────────────────────┴──────────────────┴─────────┴────────────┴──────────────┘
```

### 5.5 T4 Agent: Cross-Section Consistency Auditor (DSPM-Powered)

- **Tier**: T4 — Cross-Section Consistency
- **Available tools**: `extract_pdf_text`, `traditional_ocr`, `query_memory`, `query_dspm`, `rule_check`
- **Underlying infrastructure**: DSPM (Filing Edition) — each filing section is processed as a DSPM "session"
- **Capabilities**:
  - **T4-A: Time window vs. business cycle** — queries DSPM L3 (event timeline) + chart_registry, uses `rule_check` to detect selective time windows
  - **T4-B: Definition consistency** — reads DSPM L1 changelog directly, inconsistencies detected with zero LLM calls
  - **T4-C: Narrative vs. data** — queries DSPM across mda + financial_statements domains, VLM compares tone vs. numbers
  - **T4-D: Cross-filing drift** — queries TemporalMemory for historical DSPM states
- **Key behavior**: Unlike other agents that call OCR tools for data, T4 Agent primarily reads from DSPM's structured memory. DSPM's zero-LLM read path means most T4 checks are computationally cheap. The L1 overwrite changelog is the most novel mechanism — definition inconsistencies are detected as a side effect of DSPM's normal write operations, not through explicit comparison logic.
- **DSPM initialization**: Before T4 runs, the Orchestrator feeds each section's extracted text through `filing_dspm.process_conversation()`. This uses 1 LLM call per section (~5-8 sections per filing = 5-8 LLM calls total for DSPM population).

### 5.6 Cross-Validator Agent

- **Input**: All findings from T1–T4 agents
- **Available tools**: `query_memory`, `rule_check`
- **Capabilities**:
  - Correlate findings across tiers (e.g., T2 truncated axis + T1 text contradiction + T3 missing GAAP pair = CRITICAL)
  - De-duplicate overlapping findings
  - Assign final risk levels (LOW / MEDIUM / HIGH / CRITICAL)
  - Identify systemic patterns (e.g., "all Non-GAAP charts are more prominent than GAAP counterparts")

### 5.7 Agent Lifecycle

```python
class BaseAgent:
    def plan(self, context: FilingMemory) -> list[AuditTask]
    def execute(self, task: AuditTask, context: FilingMemory) -> AuditResult
    def reflect(self, result: AuditResult) -> list[AuditTask]  # may produce follow-up tasks
```

All agents execute via VLM function calling. During `execute`, the VLM receives the task, available tool schemas, and image/text inputs. It generates tool calls interleaved with reasoning, ultimately producing an `AuditResult`.

### 5.8 Agentic Behavior Examples

**Scenario 1: T1 — Numerical inconsistency via decomposed extraction**

```
[1] T1 Agent calls extract_pdf_text(page=5)
    → Extracts: "Revenue remained relatively stable at $92M"
[2] T1 Agent calls traditional_ocr(image="p5_chart1", mode="chart_to_table")
    → Extracts: Revenue values [89, 91, 94, 92] from bar chart
[3] T1 Agent calls rule_check("value_mismatch",
      data={"text_claim": "stable at $92M", "chart_values": [89, 91, 94, 92]})
    → Result: Partial match — $92M matches Q4 but "stable" is misleading
       given the 89→94→92 fluctuation
[4] Finding: Text-chart inconsistency (MEDIUM)
```

**Scenario 2: T2 — OCR-verified axis truncation**

```
[1] T2 Agent looks at chart image
    → Thought: "Bar chart, Y-axis labels are small. Let me check axis values."
[2] Calls traditional_ocr(region="y_axis", mode="bbox")
    → Returns: [{"text": "80", ...}, {"text": "85", ...}, ...]
[3] Calls rule_check("truncated_axis",
      data={"axis_values": [80,85,90,95,100], "chart_type": "bar"})
    → Returns: {"is_truncated": true, "exaggeration_factor": 5.0}
[4] Side effect: Registers chart in chart_registry
    → {chart_id: "p5_c1", metric: "Revenue", is_gaap: true, chart_type: "bar",
       axis_origin: 80, time_window: "Q1-Q4 2024"}
[5] Finding: Truncated Y-axis (HIGH, confidence: 0.95)
```

**Scenario 3: T3 — Missing GAAP chart pairing**

```
[1] T3 Agent calls query_memory("chart_registry")
    → Returns all classified charts from T2:
      - p3: "Adjusted EBITDA" (Non-GAAP, bar chart, 2020-2024)
      - p5: "Revenue" (GAAP, bar chart, Q1-Q4 2024)
      - p7: "Free Cash Flow" (Non-GAAP, line chart, 2022-2024)
[2] VLM applies SEC pairing rules:
    → "Adjusted EBITDA" needs "Net Income" or "EBITDA" chart → searching...
    → "Free Cash Flow" needs "Operating Cash Flow" chart → searching...
[3] Calls query_memory("chart_registry where is_gaap=true")
    → Only GAAP chart: Revenue (p5)
    → No Net Income chart, no Operating CF chart
[4] Calls traditional_ocr(mode="bbox") on p3 for prominence data
    → "Adjusted EBITDA" title: font_size=28px, position=top
[5] Calls extract_pdf_text to search for reconciliation tables
    → Found reconciliation on p18 for Adjusted EBITDA
    → No reconciliation found for Free Cash Flow
[6] Calls rule_check("pairing_completeness", data={...})
[7] Findings:
    - Free Cash Flow: missing GAAP pair + missing reconciliation → CRITICAL
    - Adjusted EBITDA: GAAP pair missing but reconciliation exists → HIGH
```

**Scenario 4: T4-A — Time window vs. business cycle (via DSPM)**

```
[1] Orchestrator has already populated DSPM with each section's text:
    filing_dspm.process_conversation(mda_text, session_id="mda")         → 1 LLM call
    filing_dspm.process_conversation(risk_text, session_id="risk_factors") → 1 LLM call
    ... (DSPM now contains L1 profile, L2 section log, L3 event stream)

[2] T4 Agent calls query_dspm(
        "What time periods do charts show and what negative business events occurred?"
    )
    → DSPM routes to charts_and_visuals + mda + risk_factors (0 LLM calls)
    → Returns:
      L1: {revenue_chart_window: "2022-2024"}
      L3: [{"year": 2020, "event": "supply chain disruption", "impact": "negative"},
           {"year": 2021, "event": "demand contraction", "impact": "negative"}]

[3] Calls rule_check("time_window_vs_events",
      data={"window": [2022,2024], "negative_events": [2020, 2021]})
    → Window excludes all negative event years

[4] Calls query_memory(scope="historical", query="Revenue chart time window")
    → 2022 filing: 5-year window
    → 2023 filing: 5-year window
    → 2024 filing: 3-year window ← shortened

[5] Finding: Selective time window (HIGH)
    Evidence: Window shortened from 5yr to 3yr this year,
    excluding 2020-2021 trough documented in MD&A.
    Detection cost: 0 additional LLM calls (DSPM read path is free)
```

**Scenario 4b: T4-B — Definition inconsistency (via DSPM L1 changelog)**

```
[1] DSPM L1 changelog reveals (no agent action needed — produced during ingestion):
    field: "adjusted_ebitda_exclusions"
    changelog: [
      {"session": "chart_footnote",   "value": "SBC, restructuring"},
      {"session": "mda",              "value": "SBC, restructuring, acquisition costs"},
      {"session": "reconciliation",   "value": "SBC, restructuring, acquisition costs, litigation"}
    ]

[2] T4 Agent reads changelog → 3 different definitions across 3 sections
    Calls rule_check("definition_consistency",
      data={"field": "adjusted_ebitda_exclusions", "changelog": [...]})
    → Inconsistency confirmed: 2 items → 3 items → 4 items

[3] Finding: Non-GAAP definition inconsistency (HIGH)
    Evidence: Adjusted EBITDA excludes 2 items in chart footnote,
    3 items in MD&A, 4 items in reconciliation table.
    Detection cost: ZERO LLM calls — pure changelog comparison
```

**Scenario 5: Cross-filing comparison (stretch goal)**

```
Run the system on 2022, 2023, 2024 filings for the same company separately.
Compare JSON outputs manually or via a simple diff script.
If time permits, build the Company Timeline page (Page 3) for visual comparison.
```

---

## 6. Memory Module

The system uses a **two-component memory architecture**:

| Component | Scope | Used By | Technology |
|-----------|-------|---------|------------|
| **FilingMemory** | Single filing, single session | T1, T2, T3 agents | In-memory (Python objects) |
| **DSPM (Filing Edition)** | Single filing, cross-section | T4 agent | Adapted from DSPM personal memory system |

Cross-filing comparison (stretch goal) is handled by running the system on each filing independently and comparing JSON outputs — no persistent database needed at current project scale.

### 6.1 FilingMemory (In-Memory, Single-Session)

```python
class FilingMemory:

    # ── Document Structure ──
    document_map: DocumentMap
    # Per-page content type index

    # ── Section Index ──
    section_index: SectionIndex
    # Filing chapter structure: MD&A (p5-p20), Financial Statements (p21-p45), etc.

    # ── Chart Registry (built by T2, consumed by T3/T4) ──
    chart_registry: list[ChartRecord]
    # Every chart classified: metric, GAAP/Non-GAAP, type, time window, axis properties

    # ── Pairing Matrix (built by T3) ──
    pairing_matrix: list[PairingEntry]
    # Non-GAAP ↔ GAAP pairing status

    # ── Factual Memory ──
    financial_claims: list[Claim]
    chart_data: dict[int, ChartInfo]
    gaap_metrics: dict[str, Metric]
    nongaap_metrics: dict[str, Metric]
    reconciliations: dict[str, int]

    # ── OCR Cache ──
    ocr_cache: dict[str, OCRResult]

    # ── Audit Findings ──
    findings: list[AuditFinding]
    suspicions: list[Suspicion]

    # ── Decision Memory ──
    model_performance: dict[str, dict]
    audit_trace: list[TraceEntry]
```

### 6.2 DSPM Filing Edition (Cross-Section Memory for T4)

Adapted from the DSPM (Domain-Structured Personal Memory) system. Each filing section is processed as a DSPM "session", populating a four-layer structured memory that enables zero-LLM-call cross-section queries.

```python
class FilingDSPM:
    """
    DSPM adapted for filing analysis.
    Write: 1 LLM call per section (~5-8 sections per filing)
    Read: 0 LLM calls (keyword-based domain routing)
    """

    # ── L1: Filing Profile (overwrite semantics) ──
    # Current state of each metric/definition — last writer wins
    # CRITICAL: changelog tracks overwrites → definition inconsistencies
    domain_profile: dict[str, dict[str, str]]
    # e.g., {"nongaap_disclosure": {"adjusted_ebitda_exclusions": "SBC, restructuring, acquisition"}}

    changelog: list[ChangelogEntry]
    # e.g., [{"field": "adjusted_ebitda_exclusions",
    #          "session": "chart_footnote", "old": None, "new": "SBC, restructuring"},
    #         {"field": "adjusted_ebitda_exclusions",
    #          "session": "mda", "old": "SBC, restructuring", "new": "SBC, restructuring, acquisition"}]
    # ^^^ This changelog IS the T4-B detection signal

    # ── L2: Section Log (append-only) ──
    # Per-section key facts — preserves every section's exact claims
    section_log: list[SectionLogEntry]
    # e.g., [{"session": "mda", "key_facts": ["revenue grew 8%", "margin compressed"],
    #          "verbatim_anchors": ["strong momentum despite headwinds"]}]

    # ── L3: Business Event Timeline (append + link) ──
    # Performance events with timestamps, extracted from MD&A and Risk Factors
    event_stream: list[BusinessEvent]
    # e.g., [{"year": 2020, "event": "supply chain disruption", "impact": "negative", "source": "mda"}]

    # ── L4: Cross-Section Synthesis (periodic rewrite) ──
    # Auto-generated narrative highlighting cross-section tensions
    understanding: str
    # e.g., "MD&A emphasizes 'accelerating growth' but Financial Statements show
    #        decelerating revenue growth (15%→12%→8%). Risk Factors disclose
    #        'material uncertainty' not reflected in MD&A's optimistic tone."

    # ── Domain Schema ──
    schema: dict[str, DomainDef]  # FILING_SCHEMA defined in Section 2.4

    # ── Methods ──
    def process_section(self, section_text: str, section_name: str) -> dict:
        """Process one filing section. 1 LLM call for extraction."""
        # Extracts key facts, domain profile updates, events
        # L1 overwrites produce changelog entries automatically

    def get_context_for_question(self, question: str) -> str:
        """Zero LLM calls. Keyword-based domain routing assembles context."""
        # Routes to relevant domains, assembles L1+L2+L3+L4 within token budget

    def get_definition_inconsistencies(self) -> list[DefinitionConsistencyCheck]:
        """Zero LLM calls. Scans L1 changelog for overwrite conflicts."""
        # Groups changelog by field → any field with >1 distinct value = inconsistency
```

**DSPM write path for a filing**:

```
Filing PDF
  │
  ├─ Parser extracts section boundaries (Table of Contents or heading detection)
  │
  ├─ Section 1: "MD&A" ────────────► filing_dspm.process_section()  → 1 LLM call
  ├─ Section 2: "Financial Stmt" ──► filing_dspm.process_section()  → 1 LLM call
  ├─ Section 3: "Non-GAAP Disc." ──► filing_dspm.process_section()  → 1 LLM call
  ├─ Section 4: "Risk Factors" ────► filing_dspm.process_section()  → 1 LLM call
  ├─ Section 5: "Chart Footnotes" ─► filing_dspm.process_section()  → 1 LLM call
  │
  └─ DSPM now contains:
     L1: Current profile + changelog (definition inconsistencies visible)
     L2: Per-section key facts (all claims preserved)
     L3: Business event timeline (for time window analysis)
     L4: Cross-section synthesis (auto-generated tension summary)
```

**DSPM read path (T4 queries)**:

```
T4 Agent asks: "How is Adjusted EBITDA defined across sections?"
  │
  ├─ DSPM keyword routing → nongaap_disclosure + mda + charts_and_visuals
  ├─ Assembles context from L1 (current definition) + L2 (per-section descriptions)
  ├─ Returns formatted context (~1500 tokens)
  └─ Cost: 0 LLM calls
```

### 6.3 New Data Models (T3 + T4 Specific)

```python
@dataclass
class ChartRecord:
    chart_id: str
    page: int
    metric_name: str
    is_gaap: bool
    chart_type: str            # bar, line, pie, 3d_bar, etc.
    axis_origin: float | None
    time_window_start: str
    time_window_end: str
    time_window_years: int
    visual_weight: float       # 0-1, from OCR bbox area
    font_size_title: float     # From OCR bbox height
    image: bytes

@dataclass
class PairingEntry:
    nongaap_chart: ChartRecord
    expected_gaap_metric: str          # "Net Income", "Operating CF", etc.
    gaap_chart: ChartRecord | None     # None = missing pair
    pairing_status: str                # "paired" | "missing" | "incomplete"
    prominence_ratio: float | None     # Non-GAAP size / GAAP size
    comparability_issues: list[str]    # ["different time window", "different chart type"]
    reconciliation_page: int | None    # Page where reconciliation found, or None
    risk_level: str                    # LOW | MEDIUM | HIGH | CRITICAL

@dataclass
class SectionIndex:
    """Filing chapter structure"""
    sections: list[Section]

@dataclass
class Section:
    name: str              # "mda", "financial_statements", "risk_factors", etc.
    title: str             # Original heading text
    start_page: int
    end_page: int
    subsections: list[Section]

@dataclass
class ChangelogEntry:
    """DSPM L1 overwrite record — definition inconsistencies are detected here"""
    field: str             # e.g., "adjusted_ebitda_exclusions"
    domain: str            # e.g., "nongaap_disclosure"
    session: str           # Section that wrote this value
    old_value: str | None  # Previous value (None if first write)
    new_value: str         # New value that overwrote
    timestamp: str

@dataclass
class SectionLogEntry:
    """DSPM L2 per-section key facts"""
    session: str           # Section name
    key_facts: list[str]   # Extracted facts from this section
    verbatim_anchors: list[str]  # Notable direct quotes
    domain_facts: dict[str, dict[str, str]]  # Domain-structured extractions

@dataclass
class BusinessEvent:
    """DSPM L3 business event from MD&A / Risk Factors"""
    year: int
    event: str
    impact: str            # "positive" | "negative" | "neutral"
    source_section: str
    source_page: int

@dataclass
class DefinitionConsistencyCheck:
    """T4-B output: metric definition consistency across sections"""
    metric_name: str
    field: str
    instances: list[ChangelogEntry]
    is_consistent: bool
    discrepancies: list[str]
    risk_level: str

@dataclass
class TimeWindowCheck:
    """T4-A output: time window vs business cycle"""
    chart_id: str
    metric_name: str
    current_window: tuple[int, int]
    historical_windows: list[tuple[int, int]]
    excluded_events: list[BusinessEvent]
    window_shortened: bool
    has_disclosed_reason: bool
    risk_level: str
```

### 6.4 Existing Data Models

```python
@dataclass
class Claim:
    text: str
    page: int
    metric: str
    value: float | None
    context: str

@dataclass
class Metric:
    name: str
    value: float | None
    page: int
    visual_weight: float
    is_gaap: bool

@dataclass
class ChartInfo:
    chart_id: int
    page: int
    chart_type: str
    image: bytes
    caption: str | None

@dataclass
class OCRResult:
    image_id: str
    region: str
    mode: str
    text_blocks: list[dict]
    tables: list[dict] | None
    timestamp: str

@dataclass
class AuditFinding:
    finding_id: str
    tier: str              # T1 | T2 | T3 | T4
    category: str          # numerical | encoding | pairing | cross_section | temporal
    subcategory: str
    page: int
    chart_id: str | None
    risk_level: str        # LOW | MEDIUM | HIGH | CRITICAL
    confidence: float
    description: str
    correction: str
    evidence: list[str]
    tool_calls: list[str]
    trace: list[TraceEntry]

@dataclass
class TraceEntry:
    timestamp: str
    agent: str
    action: str            # "vlm_reasoning" | "tool_call" | "tool_result" | "finding" | "decision"
    tool_name: str | None
    input_summary: str
    output_summary: str
    decision: str | None

@dataclass
class CompanyProfile:
    name: str
    cik: str | None
    ticker: str | None
    sector: str | None

@dataclass
class FilingRecord:
    year: str
    filing_type: str
    filing_date: str
    page_count: int
    total_charts: int
    total_issues: int
    overall_risk: str

@dataclass
class MetricDefinition:
    metric_name: str
    year: str
    is_gaap: bool
    calculation_description: str
    excluded_items: list[str]

@dataclass
class PresentationSnapshot:
    metric_name: str
    year: str
    chart_type: str
    axis_origin: float | None
    time_window_years: int
    time_window_start: str
    visual_weight: float
    page: int

@dataclass
class TemporalDelta:
    metric_name: str
    year_from: str
    year_to: str
    delta_type: str
    description: str
    severity: str
    evidence_from: str
    evidence_to: str

@dataclass
class Suspicion:
    source_agent: str
    target_agent: str | None
    description: str
    page: int
    priority: int
```

### 6.5 Memory Roles

| Role | Component | Description | Example |
|------|-----------|-------------|---------|
| **Chart Registry** | FilingMemory | T2 builds, T3/T4 consume | T2 classifies all charts → T3 reads registry to build pairing matrix |
| **Pairing Matrix** | FilingMemory | T3 builds, Cross-Validator/Report consume | T3 builds Non-GAAP↔GAAP pairings → report shows pairing table |
| **OCR Cache** | FilingMemory | Avoid redundant extraction across agents | T2 extracts axis labels → T1 reuses same data |
| **Cross-Section Memory** | DSPM | T4 reads cross-section context with 0 LLM calls | T4 queries "definition of Adjusted EBITDA" → DSPM returns all sections' descriptions |
| **Definition Inconsistency** | DSPM L1 Changelog | Overwrites = detection signals | chart_footnote says 2 exclusions, MD&A says 3 → changelog captures discrepancy |
| **Business Cycle** | DSPM L3 | Event timeline for time window analysis | "2020 supply chain disruption" → chart excluding 2020 = selective window |
| **Cross-Tier Correlation** | FilingMemory | Cross-Validator reads all tier findings | T2 truncated axis + T1 text contradiction + T3 missing pair = CRITICAL |

---

## 7. Frontend Design

### 7.1 Page 1: Single Chart Audit

**Purpose**: Quick T1–T2 analysis of an individual chart image.

```
┌─────────────────────────────────────────────────────────┐
│  Upload Area (drag & drop PNG/JPG)                      │
├─────────────────────────────────────────────────────────┤
│  [Optional] Ground-truth textual context input          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  e.g., "Revenue: Q1=120M, Q2=125M, Q3=118M"       │ │
│  └────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  Model: [Claude ▼]     Tiers: [✓T1] [✓T2]              │
│  [Run Audit]                                            │
├────────────────────────┬────────────────────────────────┤
│  Original Chart        │  Annotated Chart               │
│  (uploaded image)      │  (OCR bbox highlights)         │
├────────────────────────┴────────────────────────────────┤
│  Findings (grouped by tier)                             │
│                                                         │
│  ── T2: Visual Encoding ──                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ ⚠ HIGH — Truncated Y-axis                         │ │
│  │ Y-axis starts at 80, 5x exaggeration factor       │ │
│  │ Evidence: OCR axis values [80,85,90,95,100]        │ │
│  │ Rule: truncated_axis = true                        │ │
│  │ Suggestion: Set Y-axis origin to 0                 │ │
│  │ [▶ Show tool trace]                               │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  ── T1: Numerical Consistency ──                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │ ⚠ MEDIUM — Text-chart inconsistency               │ │
│  │ Text: "stable at $92M" vs Chart: [89,91,94,92]    │ │
│  │ [▶ Show tool trace]                               │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Page 2: Filing Scanner

**Purpose**: Full T1–T4 audit with cross-tier correlation.

```
┌─────────────────────────────────────────────────────────┐
│  Upload PDF (Annual Report / Investor Presentation)     │
│  Model: [Claude ▼]     Tiers: [✓T1] [✓T2] [✓T3] [✓T4] │
│  [Run Full Audit]                                       │
├─────────────────────────────────────────────────────────┤
│  Summary Panel                                          │
│  ┌─────────┬─────────┬─────────┬─────────┬───────────┐ │
│  │ Charts  │ T1      │ T2      │ T3      │ T4        │ │
│  │ Found:24│ Issues:2│ Issues:3│ Issues:2│ Issues:1  │ │
│  │         │         │         │ ← Case study  │           │ │
│  └─────────┴─────────┴─────────┴─────────┴───────────┘ │
│  Overall Risk: HIGH                                     │
├─────────────────────────────────────────────────────────┤
│  Tabs: [Pairing Matrix] [All Findings] [Audit Trace]    │
│        [Export]                                          │
│                                                         │
│  ── Pairing Matrix Tab (T3) ──                     │
│  ┌────────────────┬──────────┬───────┬───────┬───────┐ │
│  │ Non-GAAP       │ Required │ Found │ Prom. │ Comp. │ │
│  │ Chart          │ GAAP     │       │ Ratio │       │ │
│  ├────────────────┼──────────┼───────┼───────┼───────┤ │
│  │ Adj EBITDA(p3) │ Net Inc  │ ✗ ⚠⚠ │ N/A   │ N/A   │ │
│  │ Free CF (p7)   │ Op CF    │ ✗ ⚠⚠ │ N/A   │ N/A   │ │
│  │ Org Rev (p5)   │ Revenue  │ ✓(p5) │ 1.0 ✓│ ✗ ⚠  │ │
│  └────────────────┴──────────┴───────┴───────┴───────┘ │
│                                                         │
│  ── All Findings Tab ──                                 │
│  Grouped by tier, sorted by risk level                  │
│  Each card shows: tier badge, chart thumbnail,          │
│  description, evidence, tool calls, correction          │
│                                                         │
│  ── Audit Trace Tab ──                                  │
│  Full agent reasoning chain with tool call details      │
│                                                         │
│  ── Export Tab ──                                       │
│  [PDF Report] [HTML Report] [JSON]                      │
└─────────────────────────────────────────────────────────┘
```

### 7.3 Page 3: Company Timeline (Stretch Goal)

**Purpose**: Multi-year comparison. **Simplified from v5** — only built if time permits.

If implemented, allows uploading 2-3 years of filings for the same company and displays a side-by-side comparison of T1-T4 findings per year. Core functionality (T1-T4 detection) works without this page — each filing can be scanned individually via Page 2.

### 7.4 Sidebar

```
┌─────────────────────────┐
│  FinChartAudit           │
│                         │
│  ── Configuration ──    │
│  Claude API Key: [****] │
│  Qwen Endpoint: [    ]  │
│                         │
│  ── Detection Tiers ──  │
│  [✓] T1: Numerical      │
│  [✓] T2: Visual         │
│  [✓] T3: Pairing        │
│  [✓] T4: Cross-Section  │
│                         │
│  ── Threshold ──        │
│  Confidence:            │
│  [───●────] 0.6         │
│                         │
│  ── SEC Mapping ──      │
│  [ ] Enable mapping     │
│                         │
│  ── About ──            │
│  CS 6180 Final Project  │
└─────────────────────────┘
```

---

## 8. CLI Design

```bash
# Single chart audit (T1 + T2)
finchartaudit audit chart.png --model claude --context data.json

# Single chart, specific tiers
finchartaudit audit chart.png --tiers t1,t2

# Full filing scan (T1-T4)
finchartaudit scan filing.pdf --model claude --output report.pdf

# Filing scan, specific tiers
finchartaudit scan filing.pdf --tiers t1,t2,t3

# Output formats
finchartaudit scan filing.pdf --format json|html|pdf
```

---

## 9. Project Structure

```
finchartaudit/
├── app.py                          # Streamlit entry point
├── cli.py                          # CLI entry point (typer)
├── config.py                       # Configuration management
│
├── parser/
│   ├── __init__.py
│   ├── pdf_parser.py               # PDF → page images (PyMuPDF)
│   ├── chart_extractor.py          # Page → chart region extraction
│   └── text_extractor.py           # Page → narrative text extraction (PyMuPDF)
│
├── tools/
│   ├── __init__.py
│   ├── registry.py                 # Tool registry — all tool definitions
│   ├── extract_pdf_text.py         # PyMuPDF embedded text extraction
│   ├── traditional_ocr.py          # PaddleOCR wrapper (text + bbox + table + chart_to_table)
│   ├── doc_ocr.py                  # LLM-based OCR wrapper (PaddleOCR-VL / GLM-OCR)
│   ├── rule_check.py               # Deterministic rule engine
│   ├── query_memory.py             # Memory query tool
│   ├── query_dspm.py               # DSPM cross-section query tool (0 LLM calls)
│   └── sec_api.py                  # SEC EDGAR API wrapper
│
├── memory/
│   ├── __init__.py
│   ├── filing_memory.py            # FilingMemory (in-memory, single-session)
│   ├── models.py                   # All data models (Claim, Metric, ChartRecord, etc.)
│   └── trace.py                    # TraceEntry, audit trail logging
│
├── dspm/                           # DSPM Filing Edition (adapted from memory project)
│   ├── __init__.py
│   ├── filing_dspm.py              # FilingDSPM — main orchestrator
│   ├── domain_profile.py           # L1: Filing Profile (overwrite + changelog)
│   ├── section_log.py              # L2: Section Log (append-only)
│   ├── event_stream.py             # L3: Business Event Timeline
│   ├── understanding.py            # L4: Cross-Section Synthesis
│   ├── filing_schema.py            # Filing domain schema (mda, financial_statements, etc.)
│   ├── extractor.py                # LLM extraction prompts (1 call/section)
│   └── retriever.py                # Zero-LLM keyword-based retrieval
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py             # Central orchestrator
│   ├── base.py                     # Agent base class (plan/execute/reflect + tool-use)
│   ├── t1_numerical.py             # T1: Numerical Consistency Auditor
│   ├── t2_visual.py                # T2: Visual Encoding Auditor
│   ├── t3_pairing.py               # T3: Pairing Completeness Auditor ← Case study
│   ├── t4_cross_section.py         # T4: Cross-Section Consistency Auditor
│   └── cross_validator.py          # Cross-tier correlation
│
├── vlm/
│   ├── __init__.py
│   ├── base.py                     # VLM unified interface (with tool-use support)
│   ├── claude_client.py            # Claude Sonnet wrapper
│   └── qwen_client.py              # Qwen2.5-VL wrapper
│
├── report/
│   ├── __init__.py
│   ├── aggregator.py               # Result aggregation + risk scoring
│   ├── generator.py                # PDF/HTML report generation
│   └── templates/
│       ├── report.html.j2          # Single-filing report template
│       ├── pairing_matrix.html.j2  # T3 pairing matrix template
│       └── report_styles.css       # Report styling
│
├── prompts/
│   ├── t1_numerical.py             # T1 prompt templates
│   ├── t2_visual.py                # T2 prompt templates (12 misleader types)
│   ├── t3_pairing.py               # T3 prompt templates (pairing rules, classification)
│   └── t4_cross_section.py         # T4 prompt templates
│
└── tests/
    ├── test_tools.py
    ├── test_parser.py
    ├── test_memory.py
    ├── test_dspm.py                # DSPM filing edition tests
    ├── test_dspm_changelog.py      # L1 changelog → T4-B detection
    ├── test_t1_numerical.py
    ├── test_t2_visual.py
    ├── test_t3_pairing.py
    ├── test_t4_cross_section.py
    ├── test_cross_validator.py
    ├── test_vlm.py
    └── test_report.py
```

---

## 10. Evaluation Strategy

### 10.1 Data Reality

| Tier | Dataset | Volume | Ground Truth Available | Sufficiency |
|------|---------|--------|----------------------|-------------|
| **T2** | **Misviz-synth** | **81,814 charts** | **Misleader labels + data tables + axis metadata + Matplotlib code** | **Excellent — primary 2x2 experiment** |
| **T2** | **Misviz (real)** | **2,604 charts** | **Misleader labels only (no data tables)** | **Good — generalization test (vision-only)** |
| T1 | Any filing (auto-generated) | Unlimited | Manual verification on T3 case study filings | Sufficient |
| T3 | SEC comment letter cases (MYE, HURN, FXLV) | 3-5 companies | SEC comment letters as ground truth | Case study only |
| T4 | Any filing via DSPM (auto-generated) | Unlimited | Manual annotation on 2-3 filings | Sufficient for method demo |

### 10.2 Misviz Dataset Structure (Key for Experiment Design)

Misviz provides two datasets with different data availability:

**Misviz-synth (synthetic, 81,814 charts)**:
- Charts generated from real-world data tables (Our World in Data) via Matplotlib
- Each instance includes:
  - Chart image
  - **Underlying data table** (raw values the chart is based on) → serves as **textual ground truth for vision+text condition**
  - **Axis metadata** (4 columns: Seq, Axis, Label, Relative Position) → serves as **OCR extraction ground truth**
  - Matplotlib code used to generate the chart
  - Misleader type labels (12 types)

**Misviz (real-world, 2,604 charts)**:
- Charts collected from the web (news, reports, social media)
- Each instance includes:
  - Chart image (downloaded via URL)
  - Misleader type labels
  - **No underlying data tables or axis metadata**

**Axis metadata structure** (from Misviz-synth):
```
Seq | Axis | Label | Relative Position
1   | y    | 0     | 0.0
2   | y    | 20    | 1.0
3   | y    | 40    | 2.0
4   | y    | 60    | 3.0
```
This directly enables evaluation of our OCR extraction accuracy — compare `traditional_ocr` output against Misviz-synth axis metadata as ground truth.

### 10.3 Evaluation Plan

```
Paper Evaluation Structure:

1. T2 Main Experiment: 2x2 Factorial (Primary)       ← Proposal RQ1 + RQ2
   Dataset: Misviz-synth (81,814 charts, all have data tables)
   Design: 2x2 factorial
     (Claude Sonnet vs Qwen2.5-VL) x (vision-only vs vision+text)
   Vision-only: Chart image + misleader type definitions
   Vision+text: Chart image + underlying data table as textual context
   Metrics: Per-misleader-type accuracy, F1, confusion matrix
   Key question: Does providing ground-truth data improve detection?

2. T2 Real-World Generalization Test                  ← Proposal RQ1
   Dataset: Misviz real (2,604 charts, vision-only since no data tables)
   Design: Model comparison only (Claude vs Qwen, both vision-only)
   Key question: Does performance hold on real-world charts?

3. T2 Tool-Use Ablation                              ← Architecture contribution
   Dataset: Misviz-synth subset (sample ~1,000 charts)
   Design: VLM-only vs VLM+OCR+rules on same charts
   Ground truth for OCR accuracy: Misviz-synth axis metadata
   Key question: Does tool-use improve detection over VLM alone?

4. T3 SEC Case Study (Qualitative)                   ← Proposal RQ3
   Dataset: 3-5 companies with confirmed SEC comment letters
   Method: Run full system (T1-T4) on their filings
   Validation: Compare system output to actual SEC findings
   Presentation: Per-company pairing matrix + finding narrative
   Side benefit: T1 outputs manually verified on these filings (~30 samples)

5. T4 DSPM Method Demonstration                      ← Method contribution
   Dataset: 2-3 filings processed through DSPM Filing Edition
   Method: Show L1 changelog captures definition inconsistencies
   Validation: Manual annotation of actual inconsistencies in the filing
   Presentation: Changelog examples + detection accuracy
```

### 10.4 What Each Evaluation Proves

| Evaluation | Proves | Aligns with Proposal | Data |
|-----------|--------|---------------------|------|
| T2 2x2 on Misviz-synth | VLM misleader detection; **textual grounding effect**; model comparison | RQ1 + RQ2 | 81,814 charts with data tables |
| T2 on Misviz real | Generalization to real-world charts | RQ1 | 2,604 charts |
| T2 tool-use ablation | VLM+OCR+rules > VLM-only; OCR extraction accuracy | Architecture | Misviz-synth axis metadata as ground truth |
| T3 case study | System detects real SEC violations | RQ3 | 3-5 SEC comment letter companies |
| T4 DSPM demo | L1 changelog detects cross-section inconsistencies | Method | 2-3 filings |

---

## 11. Development Phases

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **P0: Foundation** | Tool registry, `extract_pdf_text`, `traditional_ocr`, `rule_check`, VLM interface with tool-use, PDF parser, SectionIndex builder, data models, config | None |
| **P1: T2 — Visual Encoding** | T2 agent, 12 misleader detection with OCR+rules, chart_registry construction, Single Chart page | P0 |
| **P2: T2 Evaluation** | Run 2x2 experiment on Misviz benchmark, analyze results | P1 |
| **P3: T1 — Numerical Consistency** | T1 agent, text-chart-table cross-comparison | P0, P1 |
| **P4: T3 — Pairing Completeness** | T3 agent, pairing matrix, GAAP/Non-GAAP classification, reconciliation search | P0, P1 (needs chart_registry) |
| **P5: Orchestrator + Filing Scanner** | Orchestrator dispatch, cross-tier correlation, Filing Scanner page | P1, P3, P4 |
| **P6: DSPM + T4** | Adapt DSPM from memory project, T4 agent, `query_dspm` tool | P0, P5 (reuses `../memory/dspm/`) |
| **P7: Report & Polish** | PDF/HTML report, region highlights, CLI tool, T3 case study execution | P5 |
| **P8: Stretch Goals** | Company Timeline page, `doc_ocr` fallback, cross-filing comparison | P7, if time permits |

**Notes**:
- P2 (T2 Evaluation on Misviz) is the primary quantitative experiment — prioritize this.
- P6 (DSPM) can start in parallel with P3/P4 since it reuses existing code.
- P8 is optional — the system is complete and evaluable without it.

---

## 12. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **T3 evaluation data too thin (3-5 cases)** | Cannot do quantitative T3 benchmark | Reframe as case study. T2 on Misviz is the quantitative backbone. |
| VLM silent failure on contradictions (~1% detection) | T1 misses inconsistencies | Decomposed extraction → programmatic comparison |
| VLM axis detection failure (~70% miss rate) | T2 misses truncated axes | OCR + rule engine for deterministic verification |
| GAAP/Non-GAAP chart classification errors | T3 pairing matrix corrupted | Structured prompt + human verification option |
| PaddleOCR installation issues | OCR tools unavailable | RapidOCR (ONNX) as drop-in fallback |
| DSPM L1 field explosion | T4 context exceeds token budget | Canonical field names + value compression (per DSPM research) |
| DSPM extraction misses implicit info (~11.5%) | T4 misses inconsistencies | Regex entity scanning (already in DSPM) |
| Section boundary detection errors | DSPM gets wrong section text | Table of Contents parsing + heading OCR |
| Chart extraction from PDF misses charts | Incomplete chart registry | Heuristic extraction + VLM page-level scan |
| Agent reflection loop cost | API cost overrun | Max reflection depth = 2; budget cap per filing |

---

## 13. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Four-tier framework** as the system contribution | Provides clear progression of cross-modal complexity. Each tier has distinct data requirements, tool dependencies, and evaluation methods. The framework itself is the primary contribution, not any single tier. |
| **T2 as primary quantitative evaluation** | Misviz provides 2,604 labeled charts — the only tier with sufficient data for benchmark-quality evaluation. This also directly fulfills the proposal's RQ1 and RQ2. |
| **T3 as case study** (not benchmark) | Only 3-5 confirmed SEC cases with chart-level pairing issues exist. A qualitative case study on real SEC violations is more honest and more compelling than a forced quantitative evaluation on tiny data. |
| **Tool-use agents** over fixed pipeline | Agents decide when precision is needed. Reduces unnecessary OCR calls. Rich audit traces. Aligns with AI Agent course topic. The VLM-only vs VLM+tool-use comparison is itself an evaluable contribution. |
| **OCR for quantitative, VLM for semantic** | VLMs fail at precise extraction (~5% error, near-random font size). OCR provides ground truth. Rule engine makes deterministic judgments. VLMs handle classification, interpretation, narrative. |
| **Decomposed contradiction detection** | VLMs fail to notice contradictions in ~99% of cases. Extract from each modality independently → compare programmatically. |
| **DSPM for T4** | Reuses existing tested codebase. L1 overwrite changelog is a novel detection mechanism. Zero-LLM read path keeps T4 computationally cheap. |
| **L1 overwrite as detection signal** | Turning DSPM's known weakness (overwrite loses info) into T4's strength (overwrite conflict = definition inconsistency). Novel insight. |
| **Simplified temporal scope** | Cross-filing analysis downgraded to stretch goal. Single-filing T1-T4 is the complete, evaluable system. Multi-year comparison can be done by running the system on each year separately. |
| **No SQLite / persistent storage** | 3-5 companies, 2-3 filings each. JSON files are sufficient. SQLite adds complexity without benefit at this scale. |

---

## 14. Paper Contribution Mapping

| # | Contribution | Type | Evaluation | Data |
|---|-------------|------|-----------|------|
| 1 | Four-tier cross-modal detection framework (T1-T4) | System | System demo on SEC filings | Any filing |
| 2 | VLM misleading chart detection with textual grounding (2x2 experiment) | Empirical | **Misviz-synth 81,814 charts** — primary 2x2 factorial | Misviz-synth (with data tables + axis metadata) |
| 3 | Tool-use architecture: VLM autonomously invokes OCR + rule engine | Architecture | VLM-only vs VLM+tool-use ablation; OCR accuracy vs axis metadata ground truth | Misviz-synth axis metadata |
| 4 | Real-world generalization of VLM chart detection | Empirical | Model comparison on real-world charts | Misviz real (2,604 charts, vision-only) |
| 5 | T3 pairing completeness detection (VLM+LLM joint reasoning) | Method | **Case study on 3-5 SEC comment letter companies** | MYE, HURN, FXLV + A's supplements |
| 6 | T4 cross-section consistency via DSPM (L1 overwrite as detection signal) | Method | Demonstration on 2-3 filings | Any filing |
| 7 | Application to real SEC filings validated against comment letters | Application | T3 case study results vs. actual SEC findings | SEC EDGAR |

**Paper structure alignment**:
- Contributions 2+3+4 → Main experiment section (quantitative, 81,814 + 2,604 charts)
- Contribution 5 → Application section (qualitative, real-world validation)
- Contribution 6 → Method section (novel technique demonstration)
- Contributions 1+7 → Introduction + System overview

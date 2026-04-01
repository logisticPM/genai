# FinChartAudit — Daily Progress Log

## 2026-03-20 (Day 1)

---

### Summary

From zero to a working end-to-end system in one session. Completed product design (v6), data infrastructure, core system implementation, and a functional Streamlit demo with SEC chart detection capabilities.

---

### 1. Product Design (DESIGN.md v1 → v6)

| Version | Key Change |
|---------|-----------|
| v1 | Basic demo architecture: 2 pages + sidebar |
| v2 | Added agentic design, memory module, cross-filing temporal audit |
| v3 | Added tool-use architecture (VLM autonomously calls OCR/rules) |
| v4 | Restructured around four-tier detection framework (T1-T4) |
| v5 | Integrated DSPM for T4 cross-section consistency, L1 changelog as detection signal |
| v6 (final) | Evaluation-aware redesign: T2 as primary quantitative eval (Misviz), T3 as case study, simplified temporal scope |

Final document: 1,495 lines covering architecture, agents, memory, tools, evaluation strategy, development phases, and paper contribution mapping.

### 2. Data Infrastructure

#### Misviz Benchmark (fully downloaded and verified)
- Misviz-synth: **57,665** chart images + data tables + axis metadata + code
- Misviz real: **2,593** / 2,604 chart images (99.6%)
- Data pipeline: loader, text_context builder, evaluator — all tested

#### SEC Filing Tools
- Company registry (3 placeholder companies)
- SEC EDGAR downloader
- PDF extractor (charts, tables, text, sections)
- Annotation tool (T2/T3/T4 Streamlit UI)
- Dataset export (JSON/CSV)

#### Data Request for Member A
- Chinese Word document with search keywords, validation steps, empty table, timeline
- 4 confirmed cases identified: Myers (MYE), Huron (HURN), F45 (FXLV), Unisys (UIS)

### 3. Core System (`finchartaudit/`)

**25 source files, 2,253 lines of code**

```
finchartaudit/
├── __init__.py                    # Package init
├── config.py                     # Settings (OpenRouter API, model config)
├── app.py                        # Streamlit demo app
│
├── memory/
│   ├── models.py                 # ChartRecord, AuditFinding, TraceEntry, enums
│   ├── trace.py                  # AuditTracer (log/print/export)
│   └── filing_memory.py          # FilingMemory (charts, findings, OCR cache)
│
├── tools/
│   ├── registry.py               # 6 tool schemas for VLM function calling
│   ├── rule_check.py             # RuleEngine (6 deterministic checks)
│   ├── extract_pdf_text.py       # PyMuPDF wrapper
│   ├── traditional_ocr.py        # PaddleOCR with v3 API support + MKLDNN fix
│   └── query_memory.py           # Memory search tool
│
├── vlm/
│   ├── base.py                   # VLMClient abstract + dataclasses
│   ├── claude_client.py          # OpenRouter API client (vision + tool-use)
│   └── qwen_client.py            # Stub for Member B
│
├── agents/
│   ├── base.py                   # BaseAgent with tool-use execution loop
│   └── t2_visual.py              # T2 agent (12 misleaders + 11 completeness)
│
├── parser/
│   └── pdf_parser.py             # FilingParser (pages, charts, sections)
│
└── prompts/
    └── t2_visual.py              # T2 prompts + SEC rule mapping
```

### 4. End-to-End Tests Passed

#### Test 1: Misviz truncated axis bar chart
- **Input**: Real-world chart with Y-axis starting at 0.38
- **Result**: Correctly detected `truncated axis` (HIGH, 100%) + `inappropriate axis range` (MEDIUM, 100%)
- **Tool trace**: query_memory → OCR(y_axis) → OCR(x_axis) → OCR(title) → OCR(full) → rule_check(truncated_axis) → rule_check(broken_scale) → JSON output
- **Time**: ~46s

#### Test 2: SEC Unisys TSR chart (missing axis labels)
- **Input**: Unisys proxy statement TSR chart (SEC Case 8)
- **Result**: Correctly detected 5 completeness issues including `missing_x_axis_values` (HIGH) — the exact issue SEC flagged
- **Tool trace**: OCR(full) → rule_check(truncated_axis=false) → JSON output
- **Time**: ~72s

### 5. Technical Issues Resolved

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| PaddleOCR `show_log` error | PP-OCRv5 API changed | Try/except fallback |
| PaddleOCR oneDNN crash | PIR+MKLDNN incompatibility in PaddlePaddle 3.3 | `enable_mkldnn=False` |
| PaddleOCR v3 result format | Returns OCRResult dict-like object, not list | Added `_parse_paddle_v3_result()` |
| OCR tool receives chart_id not file path | VLM passes `image_id="current"` | Added `_resolve_image_path()` |
| rule_check receives wrong param format | VLM used `y_min` instead of `axis_values` | Updated prompt with exact param formats |
| SEC charts timeout (278s) | 10-round tool loop + large images | Reduced to 3 rounds + image resize to 1200px |
| VLM truncated without JSON output | Hit tool loop limit mid-reasoning | Added force-answer nudge on loop exhaustion |
| Streamlit `ModuleNotFoundError` | Project root not on sys.path | Added `sys.path.insert` in app.py |

### 6. Key Design Decisions Made Today

1. **OpenRouter API instead of direct Anthropic SDK** — reuses existing API key from DSPM project
2. **PaddleOCR with MKLDNN disabled** — avoids PaddlePaddle 3.3 PIR compatibility bug
3. **3-round tool-use limit** — balances detection quality vs latency (~45-70s per chart)
4. **Image auto-resize to 1200px** — prevents timeout on large SEC filing images
5. **23 check items** = 12 Misviz misleaders + 11 completeness checks, each with SEC rule reference
6. **T2 as primary quantitative eval** — Misviz-synth 57,665 charts with ground truth data tables
7. **T3 as case study** — 3-5 SEC comment letter companies (data-constrained reality)

### 7. Files Created Today

| Category | Count | Key Files |
|----------|-------|-----------|
| Design docs | 4 | DESIGN.md, IMPLEMENTATION_PLAN.md, PROGRESS.md, DATA_REQUEST_FOR_A_v3.docx |
| Data tools | 15 | data_tools/*.py, data_tools/misviz/*.py |
| Core system | 25 | finchartaudit/**/*.py |
| Config | 2 | .env, .gitignore |
| **Total** | **~46** | |

### 8. What's Next (Week 1 remaining)

| Priority | Task | Status |
|----------|------|--------|
| P0 | Send DATA_REQUEST to Member A | Ready to send |
| P0 | Confirm Misviz-synth data table format with B | Data downloaded, format verified |
| P1 | Build T2 Misviz evaluation script (batch runner) | Not started |
| P1 | Build T1 agent (numerical consistency) | Not started |
| P1 | Build T3 agent (pairing completeness) | Not started |
| P2 | Filing Scanner page (full PDF → T1-T4) | Not started |

### 9. Open Risks

| Risk | Status | Mitigation |
|------|--------|-----------|
| A's company list delayed | Waiting | 4 confirmed cases can start immediately |
| API cost (OpenRouter) | ~$0.02 per chart analysis | Budget ~$50 for full Misviz eval |
| PaddleOCR accuracy on complex charts | Tested OK on simple charts | VLM visual judgment as fallback |
| Tool-use loop latency (45-70s/chart) | Acceptable for demo, slow for batch eval | Parallelize batch runs |

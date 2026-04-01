# FinChartAudit -- Project Progress

> Last updated: 2026-03-20
> Owner: Member C (Engineering, Text Pipeline, Demo)

---

## Overall Status

```
Design & Planning   [====================] 100%
Data Infrastructure [=================== ] 95%
Core System         [                    ] 0%
Evaluation          [                    ] 0%
Demo & Report       [                    ] 0%
Paper               [                    ] 0%
```

---

## 1. Design Documents

| Deliverable | Status | File |
|-------------|--------|------|
| Product Design Document (v6) | **Done** | `DESIGN.md` (1,495 lines) |
| Implementation Plan | **Done** | `IMPLEMENTATION_PLAN.md` |
| Data Request for Member A (Chinese, Word) | **Done** | `DATA_REQUEST_FOR_A_v3.docx` |
| SEC Cases Reference | **Done** | `sec_cases.md` (10 cases, 4 confirmed suitable) |

### Design Document Evolution
- v1: Basic demo architecture (2 pages + sidebar)
- v2: Added agentic design + memory module + cross-filing temporal audit
- v3: Added tool-use architecture (VLM calls OCR/rules autonomously)
- v4: Restructured around four-tier detection framework (T1-T4), T3 as core
- v5: Integrated DSPM for T4, added cross-section consistency via L1 changelog
- v6 (current): Evaluation-aware redesign. T2 as primary quantitative eval (Misviz), T3 as case study, simplified temporal scope

---

## 2. Data Infrastructure

### 2.1 Misviz Benchmark Data

| Dataset | Files | Size | Status |
|---------|-------|------|--------|
| Misviz-synth JSON metadata | 57,665 instances | 37.5 MB | **Done** |
| Misviz-synth chart images (PNG) | 57,665 files | ~5 GB | **Done** |
| Misviz-synth data tables (CSV) | 57,665 files | in archive | **Done** |
| Misviz-synth axis metadata (JSON) | 57,665 files | in archive | **Done** |
| Misviz-synth code snippets (PY) | 57,665 files | in archive | **Done** |
| Misviz real-world JSON metadata | 2,604 instances | 1.4 MB | **Done** |
| Misviz real-world images | 2,593 / 2,604 | ~0.5 GB | **Done** (99.6%, 11 permanently failed) |

**Data pipeline verified**: loader -> text_context builder -> evaluator all tested end-to-end.

### 2.2 SEC Filing Data

| Data | Status | Depends On |
|------|--------|-----------|
| Company registry (3 placeholder companies) | **Done** | Need A's real company list |
| SEC EDGAR downloader tool | **Done** | Ready to run once CIKs received |
| PDF extractor (charts, tables, text, sections) | **Done** | Ready |
| T3 confirmed companies (MYE, HURN, FXLV, UIS) | **Identified** | Need to download filings |
| T3 supplement companies (3-4 more) | **Waiting on A** | P0 deadline: 3/22 |
| Comment letter excerpts | **Waiting on A** | P1 deadline: 3/26 |
| Clean control companies (2-3) | **Waiting on A** | P2 deadline: 4/2 |

### 2.3 Data Tools Built

| Tool | File | Function | Tested |
|------|------|----------|--------|
| `config.py` | `data_tools/config.py` | Path configuration | Yes |
| `company_registry.py` | `data_tools/company_registry.py` | Company list management | Yes |
| `sec_downloader.py` | `data_tools/sec_downloader.py` | SEC EDGAR filing download | Yes (structure) |
| `pdf_extractor.py` | `data_tools/pdf_extractor.py` | PDF -> charts/tables/text/sections | Yes |
| `annotation_models.py` | `data_tools/annotation_models.py` | T2/T3/T4 annotation data models | Yes |
| `annotator_app.py` | `data_tools/annotator_app.py` | Streamlit annotation UI (3 modes) | Needs streamlit install |
| `export_dataset.py` | `data_tools/export_dataset.py` | Export annotations to JSON/CSV | Yes |
| `quickstart.py` | `data_tools/quickstart.py` | Environment check + setup | Yes |
| `misviz/loader.py` | `data_tools/misviz/loader.py` | Misviz dataset loader | Yes |
| `misviz/text_context.py` | `data_tools/misviz/text_context.py` | Build vision+text context | Yes |
| `misviz/evaluator.py` | `data_tools/misviz/evaluator.py` | Evaluation metrics (F1, EM, PM, 2x2) | Yes (logic) |
| `misviz/setup_data.py` | `data_tools/misviz/setup_data.py` | Data download + status check | Yes |

---

## 3. Core System (`finchartaudit/`)

**Status: Not started**

| Component | Files Needed | Status | Phase |
|-----------|-------------|--------|-------|
| VLM unified interface | `vlm/base.py`, `vlm/claude_client.py`, `vlm/qwen_client.py` | Not started | P0 |
| Tool registry | `tools/registry.py` | Not started | P0 |
| `extract_pdf_text` tool | `tools/extract_pdf_text.py` | Not started | P0 |
| `traditional_ocr` tool | `tools/traditional_ocr.py` | Not started | P0 |
| `rule_check` tool | `tools/rule_check.py` | Not started | P0 |
| `query_memory` tool | `tools/query_memory.py` | Not started | P0 |
| `query_dspm` tool | `tools/query_dspm.py` | Not started | P6 |
| FilingMemory | `memory/filing_memory.py`, `memory/models.py` | Not started | P0 |
| Data models | `memory/models.py` | Not started | P0 |
| Audit trace | `memory/trace.py` | Not started | P0 |
| PDF parser | `parser/pdf_parser.py` | `data_tools/pdf_extractor.py` exists, needs refactor | P0 |
| T2 agent | `agents/t2_visual.py` | Not started | P1 |
| T1 agent | `agents/t1_numerical.py` | Not started | P3 |
| T3 agent | `agents/t3_pairing.py` | Not started | P4 |
| T4 agent | `agents/t4_cross_section.py` | Not started | P6 |
| Orchestrator | `agents/orchestrator.py` | Not started | P5 |
| Cross-validator | `agents/cross_validator.py` | Not started | P5 |
| DSPM Filing Edition | `dspm/` (8 files) | Source exists at `memory/dspm/`, needs adaptation | P6 |
| Prompts | `prompts/t1_numerical.py` ~ `t4_cross_section.py` | Not started | P1-P6 |
| Report generator | `report/generator.py`, `report/templates/` | Not started | P7 |
| Streamlit app | `app.py` | Not started | P1 (Page 1), P5 (Page 2) |
| CLI | `cli.py` | Not started | P7 |

---

## 4. Evaluation

| Experiment | Dataset | Status | Phase |
|-----------|---------|--------|-------|
| T2 2x2 factorial (Claude x Qwen x vision-only x vision+text) | Misviz-synth 57,665 | Not started | P2 |
| T2 real-world generalization | Misviz real 2,593 | Not started | P2 |
| T2 tool-use ablation (VLM-only vs VLM+OCR+rules) | Misviz-synth ~1,000 sample | Not started | Week 4 |
| T3 SEC case study | 3-5 company filings | Not started (waiting on data) | P5/Week 3 |
| T4 DSPM demonstration | 2-3 filings | Not started | P6/Week 3 |
| False positive test | 2-3 clean control filings | Not started (waiting on data) | Week 4 |

---

## 5. Existing Assets (Reusable)

| Asset | Location | Reuse Plan |
|-------|----------|-----------|
| DSPM core code (4-layer memory) | `memory/dspm/` (10 files) | Adapt for T4 Filing Edition |
| DSPM evaluation framework | `memory/evaluation/` | Reference for evaluation methodology |
| DSPM research report | `memory/RESEARCH_REPORT.md` | Cited in paper for L1 changelog insight |
| LLM client | `memory/config/llm_client.py` | Adapt for VLM interface |
| PDF extraction logic | `data_tools/pdf_extractor.py` | Refactor into `finchartaudit/parser/` |

---

## 6. Dependencies

### Installed
- PyMuPDF (fitz) -- PDF parsing
- Pillow (PIL) -- Image processing
- httpx -- HTTP client (SEC EDGAR API)
- pydantic -- Data models
- anthropic -- Claude API (via pip)

### Need to Install
- `streamlit` -- Demo UI + Annotation tool
- `paddleocr` or `rapidocr-onnxruntime` -- OCR
- `typer` -- CLI framework

### Need API Keys
- Anthropic API key (Claude Sonnet) -- for VLM agents
- Qwen2.5-VL -- local inference setup (B's responsibility)

---

## 7. Blockers

| Blocker | Impact | Owner | ETA |
|---------|--------|-------|-----|
| A's company list (CIK numbers) | Cannot download SEC filings for T3 case study | A | 3/22 |
| PaddleOCR installation | Cannot build `traditional_ocr` tool | C | Week 1 |
| Streamlit installation | Cannot run annotation tool or demo | C | Today |
| Qwen2.5-VL local setup | Cannot run Qwen condition of 2x2 experiment | B | Week 1 |

---

## 8. Next Actions (Week 1 Priority)

| # | Task | Owner | Target |
|---|------|-------|--------|
| 1 | Send DATA_REQUEST_FOR_A_v3.docx to A | C | Today |
| 2 | `pip install streamlit paddleocr typer` | C | Today |
| 3 | Build VLM unified interface (`vlm/base.py` + `vlm/claude_client.py`) | C | Mon-Tue |
| 4 | Build tool registry + `traditional_ocr` + `rule_check` | C | Mon-Tue |
| 5 | **Minimum viable loop**: upload chart -> OCR axis -> rule_check truncated_axis -> finding | C | Wed |
| 6 | Build PDF parser + chart extractor | C | Thu |
| 7 | Build FilingMemory + data models | C | Fri |
| 8 | Confirm Misviz-synth data table format with B | C+B | Mon |
| 9 | Design T2 detection prompts | B | Week 1 |
| 10 | Search for T3 supplement companies | A | By Fri |

---

## 9. File Tree (Current)

```
DD_v1/
├── DESIGN.md                          # Product design v6 (1,495 lines)
├── IMPLEMENTATION_PLAN.md             # 4-week implementation plan
├── PROGRESS.md                        # This file
├── DATA_REQUEST_FOR_A_v3.docx         # Data request for Member A
├── sec_cases.md                       # 10 SEC comment letter cases
│
├── data_tools/                        # Data collection & annotation toolkit
│   ├── __init__.py
│   ├── config.py                      # Paths and constants
│   ├── company_registry.py            # Company list management
│   ├── sec_downloader.py              # SEC EDGAR downloader
│   ├── pdf_extractor.py               # PDF -> charts/tables/text
│   ├── annotation_models.py           # T2/T3/T4 annotation schemas
│   ├── annotator_app.py              # Streamlit annotation UI
│   ├── export_dataset.py             # Export to JSON/CSV
│   ├── quickstart.py                  # Environment check
│   └── misviz/                        # Misviz benchmark tools
│       ├── __init__.py
│       ├── config.py                  # Misviz paths + 12 misleader types
│       ├── loader.py                  # Dataset loader (synth + real)
│       ├── text_context.py            # Vision+text context builder
│       ├── evaluator.py               # Metrics (F1, EM, PM, 2x2)
│       ├── setup_data.py              # Download + verify data
│       └── DOWNLOAD_GUIDE.md          # Step-by-step download instructions
│
├── data/                              # All data files
│   ├── companies.json                 # Company registry (3 placeholders)
│   ├── misviz_synth/                  # Misviz synthetic benchmark
│   │   ├── misviz_synth.json          # 57,665 instance metadata
│   │   ├── png/                       # 57,665 chart images
│   │   ├── data_tables/               # 57,665 CSV data tables
│   │   ├── axis_data/                 # 57,665 JSON axis metadata
│   │   └── code/                      # 57,665 Matplotlib scripts
│   ├── misviz/                        # Misviz real-world benchmark
│   │   ├── misviz.json                # 2,604 instance metadata
│   │   └── img/                       # 2,593 chart images (99.6%)
│   ├── filings/                       # SEC filings (empty, waiting on A)
│   ├── charts/                        # Extracted charts (empty)
│   └── annotations/                   # Ground truth annotations (empty)
│
├── memory/                            # DSPM project (reusable for T4)
│   ├── dspm/                          # 10 Python files (tested, evaluated)
│   ├── evaluation/                    # LoCoMo + PersonaMem benchmarks
│   ├── scripts/                       # Diagnostic scripts
│   ├── DSPM_README.md
│   ├── RESEARCH_REPORT.md
│   └── PROPOSAL_SUMMARY.md
│
└── finchartaudit/                     # Core system (NOT YET CREATED)
    ├── app.py
    ├── cli.py
    ├── vlm/
    ├── tools/
    ├── memory/
    ├── dspm/
    ├── agents/
    ├── parser/
    ├── prompts/
    └── report/
```

---

## 10. Risk Status

| Risk | Level | Status |
|------|-------|--------|
| T3 data too thin (3-5 cases) | Medium | **Mitigated** -- reframed as case study, T2 on Misviz is primary eval |
| Misviz data availability | Low | **Resolved** -- all data downloaded and verified |
| VLM tool-use integration complexity | Medium | **Open** -- Week 1 critical path |
| PaddleOCR Windows compatibility | Low | RapidOCR as fallback |
| A's data delayed | Medium | **Open** -- P0 deadline 3/22 |
| Misviz-synth text context quality | Low | **Verified** -- data tables provide clean ground truth |

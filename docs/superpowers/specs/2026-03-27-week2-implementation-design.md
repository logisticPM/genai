# FinChartAudit Week 2+ Implementation Design

> Date: 2026-03-27
> Owner: Member C (Engineering)
> Approach: Validation-first + Evaluation-driven
> Timeline: 3/27 - 3/30 (3-4 days)

---

## Context

- **Project**: FinChartAudit — cross-modal consistency verification for SEC filings
- **Current state**: Core system skeleton ~40% (25 files in `finchartaudit/`), data 100% ready (Misviz + SEC filings downloaded)
- **3/20 baseline**: End-to-end test passed (Misviz truncated axis + Unisys proxy chart)
- **Environment**: Needs re-verification (1 week since last run)
- **T4 DSPM**: Best-effort, degrade to future work if time-constrained

## Principles

1. **Validate before building** — confirm 3/20 code still works before extending
2. **Evaluation drives priority** — T2 (Misviz experiment) > T3 (SEC case study) > T1 > Demo > T4
3. **Incremental delivery** — each phase produces a testable, usable artifact
4. **AI-assisted velocity** — Claude writes code, human reviews and tests

---

## Phase 0: Environment Verification (0.5h)

**Goal**: Confirm end-to-end pipeline runs on current environment.

**Steps**:
1. Check dependencies: PaddleOCR, Streamlit, PyMuPDF, httpx, pydantic, anthropic
2. Run single chart test: Misviz truncated-axis image → VLM → OCR → rule_check → AuditFinding
3. If broken: fix environment issues before proceeding

**Exit criteria**: One successful end-to-end finding output.

---

## Phase 1: T2 Agent Complete (3-4h)

**Goal**: T2 agent covers all 12 Misviz misleader types + 11 completeness checks. This unblocks Member B's 2x2 experiment.

### 1.1 Prompts (`prompts/t2_visual.py`)
- Structured detection instructions for each of 12 misleader types:
  misrepresentation, 3d, truncated_axis, inappropriate_pie, inconsistent_binning,
  dual_axis, inconsistent_tick_intervals, discretized_continuous, inappropriate_line,
  inappropriate_item_order, inverted_axis, inappropriate_axis_range
- 11 completeness checks: missing_title, missing_axis_labels/values, missing_legend,
  missing_units, missing_source, missing_date_range, missing_footnote
- SEC Regulation S-K Item 10(e) mapping per DESIGN.md
- Standardized output schema: `{check_type, risk_level, evidence, rule}`

### 1.2 Rule Engine (`tools/rule_check.py`)
- Verify all 23 rules have working implementations (not just schema stubs)
- Fix any rules with incomplete logic
- Unit test: each rule with a positive and negative case

### 1.3 T2 Agent (`agents/t2_visual.py`)
- Tool-use loop: up to 3 rounds of OCR + rule_check
- Reflect step: self-check for missed findings
- Error handling: OCR failure → fallback to doc_ocr (VLM-based)

### 1.4 Batch Evaluation Interface
- `eval/run_t2_batch.py`: load Misviz samples → run T2 per chart → output JSON
- Compatible with `data_tools/misviz/evaluator.py` metrics
- Validate on 50-sample subset before handoff to B

**Exit criteria**: T2 agent produces correct findings on 10+ Misviz samples across different misleader types; batch pipeline outputs valid JSON.

---

## Phase 2: T3 Agent — GAAP/Non-GAAP Pairing (3-4h)

**Goal**: Detect Non-GAAP prominence violations in SEC filings. Produces case study data for paper.

### 2.1 HTML Filing Parser
- Most case-group filings are HTML (8-K, 10-K, DEF14A), not PDF
- Add HTML parsing: extract tables, chart references, text paragraphs
- Reuse `tools/extract_pdf_text.py` for HURN's PDF annual report

### 2.2 T3 Agent (`agents/t3_pairing.py`)
- Extract all financial metrics from filing (tables + text)
- Classify each as GAAP or Non-GAAP via VLM
- Build pairing matrix: each Non-GAAP metric → corresponding GAAP metric presence
- Prominence check: relative visual weight (font size, position, color)
- Output: pairing matrix + list of AuditFindings

### 2.3 T3 Prompts (`prompts/t3_pairing.py`)
- Based on SEC C&DI 102.10 rules from DESIGN.md
- Structure: classify → pair → assess prominence

### 2.4 Case Study Baseline
- Run T3 on MYE (Myers Industries) as first validation
- Compare output against `CORRESP_2023-05-04.htm`
- Record match/miss metrics

**Exit criteria**: T3 agent outputs pairing matrix + findings for MYE; at least 1 SEC comment matched.

---

## Phase 3: T1 Agent — Numerical Consistency (1-2h)

**Goal**: Detect text-chart numerical mismatches. Simpler than T2/T3.

### 3.1 T1 Agent (`agents/t1_numerical.py`)
- Extract numerical claims from text ("revenue grew 15%", "EBITDA was $2.3B")
- Extract corresponding values from chart OCR results
- Rule-check comparison: value match, percentage calculation correctness
- Store in FilingMemory `claims_registry`

### 3.2 T1 Prompts (`prompts/t1_numerical.py`)
- Extraction prompt: identify numerical claims in text paragraphs
- Verification prompt: compare chart values against claims

**Exit criteria**: T1 detects a planted numerical inconsistency in a test case.

---

## Phase 4: Integration + Streamlit Demo (2-3h)

**Goal**: T1-T3 integrated into a demonstrable system.

### 4.1 Orchestrator (`agents/orchestrator.py`)
- Dispatch: T2 (per chart) → T1 (numerical) → T3 (pairing)
- Shared FilingMemory across agents
- `agents/cross_validator.py`: escalate risk when multiple tiers flag same issue

### 4.2 Streamlit Page 1 — Single Chart Audit
- Upload chart image → select model → run T2 → display findings + tool trace
- Risk level color coding (HIGH=red, MEDIUM=orange, LOW=blue)

### 4.3 Streamlit Page 2 — Filing Scanner
- Upload PDF/HTML → parse → run T1-T3 → summary + findings list
- End-to-end demo on one case-group filing

**Exit criteria**: Both pages functional; Filing Scanner completes on MYE filing.

---

## Phase 5: Evaluation + Case Study (2-3h)

**Goal**: Produce quantitative data for paper.

### 5.1 T2 Tool-Use Ablation
- ~1,000 Misviz-synth samples, two conditions: VLM-only vs VLM+OCR+rules
- Use axis metadata as OCR ground truth
- Output: comparison table + OCR precision data

### 5.2 T3 Case Study Execution
- Run T3 on all 7 case-group companies + 3 clean-group companies
- Compare system findings vs SEC Comment Letter content
- Per-company precision/recall table

### 5.3 False Positive Test
- Run on CTAS, SHW, ROK (10-K filings)
- Verify findings count significantly lower than case group

**Exit criteria**: Ablation table, case study comparison table, false positive rate — all ready for paper.

---

## Phase 6: T4 + Polish (2-3h, best-effort)

**Goal**: DSPM cross-section consistency (if time permits) + final deliverables.

### 6.1 T4 DSPM Adaptation (best-effort)
- Copy DSPM core from `memory/` → `finchartaudit/dspm/`
- Adapt: L1→Filing Profile, L2→Section Log, L3→Event Timeline, L4→Cross-Section Synthesis
- Demo on 1-2 filings
- **If time-constrained**: skip entirely, note as future work in paper

### 6.2 CLI + Report + Demo
- `cli.py` via typer: `finchartaudit audit <image>` and `finchartaudit scan <filing>`
- Report generator: HTML output with findings summary
- README.md
- Demo video (~3 min)

**Exit criteria**: System fully deliverable.

---

## Timeline

| When | Phase | Duration | Deliverable |
|------|-------|----------|-------------|
| 3/27 Fri PM | Phase 0: Env verify | 0.5h | Baseline confirmed |
| 3/27 Fri PM | Phase 1: T2 complete | 3-4h | T2 agent + batch eval pipeline |
| 3/28 Sat AM | Phase 2: T3 agent | 3-4h | T3 agent + MYE case study baseline |
| 3/28 Sat PM | Phase 3: T1 agent | 1-2h | T1 agent |
| 3/29 Sun AM | Phase 4: Integration | 2-3h | Streamlit 2 pages |
| 3/29 Sun PM | Phase 5: Evaluation | 2-3h | Paper-ready eval data |
| 3/30 Mon | Phase 6: T4 + polish | 2-3h | CLI, report, demo (+ T4 if time) |

**Total: ~3.5 days, complete by 3/30 (Mon)**

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| PaddleOCR broken on current env | Blocks Phase 0 | Fallback to RapidOCR |
| OpenRouter API key expired/rate-limited | Blocks all VLM calls | Check in Phase 0, switch to direct Anthropic SDK if needed |
| HTML filing parsing harder than expected | Slows Phase 2 | Use VLM to extract from screenshots as fallback |
| T4 DSPM adaptation too complex | Phase 6 incomplete | Already scoped as best-effort / future work |
| Tool-use loop latency (45-70s/chart) | Slow batch eval | Parallelize; reduce sample size if needed |

---

## Dependencies on B and A

| What | Who | When needed | Status |
|------|-----|-------------|--------|
| T2 prompt feedback | B | Phase 1 | Async — C writes first draft, B reviews |
| Misviz 2x2 full run | B | After Phase 1 | B runs independently once batch pipeline ready |
| Comment letter excerpts | A | Phase 5 | Already in downloaded CORRESP files |
| Case study review | A | Phase 5 | A validates system output vs SEC findings |

# Daily Log — 2026-04-01

## Overview

Two major improvements: T3 filing reading overhaul (4/7 → 6/7 SEC cases) and T2 pipeline prompt restructuring (per-type FP -33% on 8-sample test).

---

## T3: Filing Reading System Overhaul

### Problem
T3 (Non-GAAP pairing detection) missed 3/7 SEC case companies:
- **MYE, FXLV**: 0 findings despite actual violations
- **CNM**: Couldn't access exhibits
- **SHW, ROK**: 400 API error (3MB+ HTML files)

### Root Causes Found
1. 8-K filings embed Exhibit 99.1 via SGML `<DOCUMENT>` tags — extractor only read the shell
2. Large 10-K files sent in full to VLM, exceeding context limit
3. VLM was used as both "finder" and "judge" via multi-turn tool-use — unreliable
4. JSON parse failures were silent (MYE had pairing_count=9 but findings=0)
5. GAAP regex matched XBRL namespace noise (`us-gaap:CommonStock...`) — 414 false GAAP mentions in CTAS
6. Prompt didn't distinguish margin (ratio) vs absolute metric pairing

### Changes Made

**`finchartaudit/tools/html_extract.py`**:
- Added `split_sgml_documents()` — splits SGML multi-document filings (8-K with embedded exhibits)
- Added `detect_filing_type()` — detects 8-K/10-K/DEF14A from HTML content
- Added `extract_filing_complete()` — main entry point: handles exhibits, sectioning, truncation
- Added `extract_sections()` — splits 10-K by SEC section headings (Item 7, 8, etc.)
- Added `_smart_truncate()` — keeps front+back of text (financial highlights + reconciliation)
- Added `MAX_TEXT_CHARS = 150,000` (~37K tokens) truncation limit
- XBRL namespace cleanup: strips `us-gaap:`, `dei:`, `xmlns:` before tag removal
- Expanded Non-GAAP regex: added `adj.`, `funds from operations`, `constant currency`, `comparable store sales`, `segment ebitda`
- Expanded GAAP regex: added `total revenue`, `net revenue/sales`

**`finchartaudit/agents/t3_pairing.py`**:
- Rewrote `execute()`: pre-extracts filing data deterministically, then single VLM call
- Removed `html_extract` from `available_tools` — no longer needs multi-turn tool-use
- Added JSON parse failure logging with raw VLM output
- Added retry logic: only triggers when JSON parsing actually fails (not when VLM finds no violations)
- Renamed `_parse_response()` → `_build_findings()` (separates parsing from finding construction)
- Expanded `NONGAAP_TO_GAAP` mapping: added organic sales/growth, adj. ebitda, FFO, comparable store sales, constant currency

**`finchartaudit/prompts/t3_pairing.py`**:
- Complete prompt rewrite: pre-extracted data injected (VLM is "validator" not "finder")
- Filing type-specific instructions (8-K, 10-K, DEF14A)
- Margin vs absolute distinction: "Adjusted EBITDA Margin requires GAAP Net Income Margin, not just Net Income amount"
- Standalone KPI vs narrative mention guidance (reduces false positives on descriptive text)

**`finchartaudit/vlm/claude_client.py`**:
- Added input size guard: truncates prompt to 600K chars if too large

### Results

| Company | Before | After | Note |
|---------|--------|-------|------|
| **MYE** | 0 | **6** | Exhibit 99 extraction + margin distinction |
| **FXLV** | 0 | **7** | 150K truncation for 2.5MB 10-K |
| ALV | 29 | **12** | More precise (less duplication) |
| AAP | 16 | **19** | Improved coverage |
| OC | 4 | **2** | Margin-specific findings |
| UIS | 4 | **6** | More thorough |
| CNM | 0 | **0** | Still failing (exhibit is external link, not embedded) |
| SHW 10-K | 400 error | **0** | Truncation fixed the error; clean company correctly 0 |
| ROK 10-K | 400 error | **0** | Same fix; clean company correctly 0 |
| CTAS 10-K | N/A | **0** | Was 4 FP, fixed by XBRL cleanup + KPI guidance |

**Case detection: 4/7 → 6/7 (86%)**
**400 errors: 2 → 0**
**Clean company FP: 0**

---

## T2: Pipeline Prompt Restructuring

### Problem
T2 (chart misleader detection) has 85% of false positives from VLM hallucination.
- Clean chart FP rate ~40%
- Rule veto simulation (Phase 1) showed only +0.5% F1 from pure post-processing
- Key FP sources: truncated axis (52 FP), inappropriate axis range (59 FP), misrepresentation (43 FP)

### Analysis: Misviz Benchmark vs SEC
- Misviz benchmark lessons do NOT directly transfer to SEC cases
- SEC cases are 10/11 T3 (filing-level), not T2 (chart-level)
- SEC chart F1 is 55% vs Misviz 83% — completely different error profile
- T2 routing improvements help Misviz but irrelevant to SEC

### Changes Made

**`finchartaudit/agents/t2_pipeline.py`**:

1. **Two-section prompt architecture**:
   - Section A (structural): OCR evidence + rule verdicts guide VLM
   - Section B (visual): VLM uses eyes only, no OCR evidence interference

2. **Tiered rule verdicts** (key insight):
   - `[CLEAN]`/`[FLAGGED]`: Only for reliable rules (truncated_axis, dual_axis)
   - `[INFO]`: For unreliable rules (inverted_axis, inappropriate_axis_range, inconsistent_tick_intervals) — gives data but lets VLM decide
   - Prevents rule errors from misleading VLM on uncertain types

3. **Post-processing rule veto** (Strategy F from simulation):
   - truncated_axis: requires rule confirmation to survive → F1 35.7% → 69.0% in simulation
   - dual_axis: veto if no right Y-axis in OCR
   - Other types: no veto (rules not reliable enough)

4. **System prompt calibration**:
   - Section A: "Trust [CLEAN] verdicts, don't override unless clear discrepancy"
   - Section B: "Look carefully, be thorough" (encourages visual detection)
   - Removed overly conservative "most charts are correct" (caused recall drop)

### 8-Sample Test Results

| Metric | Old Pipeline | v2 (too conservative) | **v3 (current)** |
|--------|-------------|----------------------|-------------------|
| Binary TP | 8 | 5 | **7** |
| Binary FN | 0 | 3 | **1** |
| Per-type wrong | 12 | 4 | **8 (-33%)** |

Specific improvements:
- id=1 (gt=truncated axis): FP 3→1 (removed inverted, tick interval hallucinations)
- id=6 (gt=3d+misrepresentation): Exact match maintained
- id=0 (gt=misrepresentation): Recall recovered (was lost in v2)
- id=3 (gt=inconsistent tick): FP 2→2 (same), TP maintained

Remaining issues:
- id=17 (gt=inverted axis): VLM can't detect axis inversion from image — VLM capability limit
- id=81 (gt=misrepresentation): VLM distracted by truncated axis rule verdict

### Not Yet Done
- Full 271-sample eval (~$5-10 API cost, ~30min)
- Integration into `run_llm_ocr_rules.py` eval script (currently uses its own prompt building)

---

## Routing Simulation (`simulate_routing.py`)

Created offline simulation script testing 5 strategies on existing 271-chart results:

| Strategy | F1 | Precision | Recall | FP |
|----------|-----|-----------|--------|-----|
| A: Baseline | 80.1% | 77.9% | 82.5% | 40 |
| B: Rule Veto (all structural) | 79.8% | 78.9% | 80.7% | 37 |
| C: Strict truncated+range | 78.4% | 83.0% | 74.3% | 26 |
| **F: Optimal per-type** | **80.6%** | **78.8%** | **82.5%** | **38** |

Key finding: Post-processing ceiling is low (+0.5% F1). Real improvement requires prompt/pipeline changes (which we started above).

---

## Files Modified

- `finchartaudit/tools/html_extract.py` — SGML, sectioning, truncation, XBRL cleanup, regex expansion
- `finchartaudit/agents/t3_pairing.py` — pre-extraction mode, JSON logging, retry, _build_findings
- `finchartaudit/prompts/t3_pairing.py` — complete rewrite with filing type + pre-extracted data
- `finchartaudit/vlm/claude_client.py` — input size guard
- `finchartaudit/agents/t2_pipeline.py` — two-section prompt, tiered rule verdicts, rule veto
- `tests/test_t3_pairing.py` — updated for _build_findings rename
- `simulate_routing.py` — new file, offline routing simulation

## Files Created

- `simulate_routing.py` — routing strategy simulation on Misviz results
- `data/eval_results/routing_simulation.json` — simulation results
- `data/eval_results/t3_casestudy/full_results.json` — new T3 eval results

## Next Steps

1. **T2**: Run full 271-sample eval with new pipeline to get definitive F1 numbers
2. **T2**: Integrate new prompt into `run_llm_ocr_rules.py` eval script
3. **T3**: Fix CNM (need to download external exhibit from EDGAR)
4. **Paper**: Update RQ1-RQ3 sections with new results

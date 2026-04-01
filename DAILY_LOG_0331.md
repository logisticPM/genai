# FinChartAudit — Progress Log (3/31)

## 2026-03-31 (Day 4): Full-Scale Experiments + Error Analysis + Integration

### Summary
Ran full-scale 271-chart experiments with multiple tool-augmented conditions, performed error attribution analysis, completed Sonnet comparison, SEC real-chart validation, statistical significance tests, and integrated code with B's repo via PR.

### Experiments Completed

**Experiment 1: VLM + Rules (271 Misviz real charts)**
- Condition: VLM extracts structured data → Rule Engine → VLM judgment
- Result: F1=72.6%, significantly worse than VLM-only (p<0.05)
- Finding: Interpretive extraction introduces VLM bias, degrades performance

**Experiment 2: LLM-OCR + Rules (271 Misviz real charts)**
- Condition: VLM-as-OCR (pure text extraction) → Rule Engine → VLM judgment
- Result: F1=80.1%, not significantly different from VLM-only (p>=0.05)
- Finding: Pure OCR extraction is more objective than interpretive extraction

**Experiment 3: Sonnet Comparison (50 charts)**
- Sonnet VLM-only: F1=86.6%
- Sonnet LLM-OCR+Rules: F1=84.9%
- Finding: Larger model also doesn't benefit from tools overall

**Experiment 4: SEC Chart Comparison (28 real SEC charts)**
- VLM-only: 1/28 flagged (FXLV g2)
- LLM-OCR+Rules: 2/28 flagged (FXLV g1 + g2)
- Finding: LLM-OCR+Rules caught one additional chart on confirmed-violation company

**Statistical Significance (McNemar Tests)**
- Haiku VLM-only vs LLM-OCR+Rules: not significant (chi2=0.583)
- Haiku VLM-only vs VLM+Rules: significant (chi2=5.628, p<0.05)
- Haiku vision_only vs vision_text: not significant (chi2=0.029)
- Claude vs Qwen: significant (chi2=9.797, p<0.05)

### Rule Engine Improvements

**New rules added (4)**:
- `inverted_axis`: detect Y-axis direction reversal
- `dual_axis`: detect dual Y-axes with different scales
- `inappropriate_axis_range`: detect narrow axis range exaggeration
- `inconsistent_binning`: detect unequal histogram bin widths

**Rule fixes based on error analysis**:
- `truncated_axis`: now chart-type-aware (only bar/area, not line/scatter), requires >2x exaggeration
- `broken_scale`: relaxed threshold from 15% to 30% (less OCR noise sensitivity)
- `inverted_axis`: fixed logic (normal Y-axis reads top-to-bottom as decreasing, not increasing), excludes negative values
- `inappropriate_axis_range`: now chart-type-aware, stricter threshold (<10% range ratio)

**Test coverage**: 28 unit tests, all passing

### Error Attribution Analysis

Key findings from 271-chart LLM-OCR+Rules run:
1. **85%+ of FPs are VLM hallucination** (not rule-caused)
   - truncated axis: 2 rule-caused, 50 VLM-hallucinated
   - inappropriate axis range: 6 rule-caused, 53 VLM-hallucinated
   - misrepresentation: 0 rule-caused, 43 VLM-hallucinated
2. **Only inverted_axis FPs are rule-dominated** (28/32)
3. **Rule evidence has minimal impact on FP rate**: 57% with rules vs 54% without
4. **Per-type impact pattern**:
   - Tools help: axis/structural issues (quantitative evidence available)
   - Tools hurt: visual-spatial issues (OCR distracts from visual reasoning)
5. **Complementary detection**: we catch 54 instances B misses (especially discretized variable), B catches 17 we miss (especially misrepresentation)

### Integration

- Uploaded codebase to `https://github.com/logisticPM/genai.git`
- Forked B's repo and created PR: `https://github.com/PCBZ/FinChartAudit/pull/4`
  - 16 files: 4 experiment scripts, rule engine, pipeline agent, tests, results
- Created `EXPERIMENT_SUMMARY.md` with 7 key findings (F1-F7)

### Technical Issues Resolved

- **PaddleOCR memory leak**: attempted batch subprocess approach, but too slow (~38s/chart)
- **Solution**: replaced PaddleOCR with LLM-as-OCR (VLM reads text), eliminated memory issues
- **API key exhaustion**: 204/271 charts failed mid-run, resumed after recharge
- **Sample alignment**: B's 271 samples used HuggingFace ordering (different from local JSON), resolved via content matching (gt_labels + chart_type)

### Current Status (as of 3/31 EOD)

```
Design & Planning   [====================] 100%
Data Infrastructure [====================] 100%
Core System         [=================   ] 85%
Evaluation          [==================  ] 90%
Demo & Polish       [================    ] 80%
Paper               [                    ] 0%
```

### What's Done (cumulative)
- T1/T2/T3 agents implemented and tested
- Orchestrator with cross-tier validation
- Streamlit demo (2 pages)
- Rule engine with 10 checks, 28 tests
- Full 271-chart experiments: VLM-only (B), VLM+text (B), LLM-OCR+Rules (C), VLM+Rules (C)
- Sonnet comparison (50 charts)
- SEC chart comparison (28 charts)
- T3 case study (7 companies, 57% detection, 0% FP)
- McNemar statistical significance tests
- Error attribution analysis
- B's full 2x2 Misviz + RQ3 SEC results integrated
- Code pushed to GitHub, PR submitted to B's repo

### What Remains
- Paper writing (RQ1-RQ3 sections, system overview, related work)
- T4 DSPM adaptation (may defer)
- CLI tool
- Report generator
- Demo video
- README

### Git Log (today)
```
faa3ae3 feat: add experiment results, rule engine improvements, and full experiment summary
```

### PR
```
https://github.com/PCBZ/FinChartAudit/pull/4
```

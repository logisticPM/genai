# FinChartAudit Week 2+ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete T1-T3 agents, Streamlit demo, batch evaluation pipeline, and SEC case study — delivering all of Member C's engineering work for the FinChartAudit project.

**Architecture:** Extend existing `finchartaudit/` codebase (25 files, ~2,253 LOC). The system uses VLM (via OpenRouter) + PaddleOCR + deterministic rule engine in a tool-use loop. Agents share a `FilingMemory` instance. New agents (T1, T3) follow the same `BaseAgent` pattern as T2.

**Tech Stack:** Python 3.13, OpenRouter API (Claude Sonnet 4 / Qwen 2.5-VL), PaddleOCR/RapidOCR, PyMuPDF, Streamlit, pydantic-settings, httpx

---

## File Map

### Existing files to modify
- `finchartaudit/tools/rule_check.py` — add `inverted_axis`, `inconsistent_ticks` checks
- `finchartaudit/agents/t2_visual.py` — add reflect step
- `finchartaudit/app.py` — add Filing Scanner page (Page 2)
- `finchartaudit/tools/registry.py` — add `html_extract` tool schema
- `finchartaudit/agents/base.py` — register html_extract tool executor

### New files to create
- `finchartaudit/agents/t1_numerical.py` — T1 numerical consistency agent
- `finchartaudit/agents/t3_pairing.py` — T3 GAAP/Non-GAAP pairing agent
- `finchartaudit/agents/orchestrator.py` — multi-tier orchestration
- `finchartaudit/prompts/t1_numerical.py` — T1 prompts
- `finchartaudit/prompts/t3_pairing.py` — T3 prompts
- `finchartaudit/tools/html_extract.py` — HTML filing text/table extraction
- `finchartaudit/eval/run_t2_batch.py` — T2 batch evaluation runner
- `finchartaudit/eval/run_t3_casestudy.py` — T3 SEC case study runner
- `finchartaudit/eval/__init__.py`
- `tests/test_rule_check.py` — rule engine tests
- `tests/test_t2_parse.py` — T2 response parsing tests
- `tests/test_html_extract.py` — HTML extractor tests
- `tests/test_t1_numerical.py` — T1 agent logic tests
- `tests/test_t3_pairing.py` — T3 agent logic tests

---

## Task 0: Environment Verification

**Files:**
- Check: `finchartaudit/config.py`, `.env`
- Run: `finchartaudit/app.py` (or inline script)

- [ ] **Step 0.1: Check Python dependencies are installed**

```bash
cd C:\Users\chntw\Documents\7180\DD_v1
python -c "
import fitz; print(f'PyMuPDF {fitz.version}')
import PIL; print(f'Pillow {PIL.__version__}')
import httpx; print(f'httpx {httpx.__version__}')
import pydantic; print(f'pydantic {pydantic.__version__}')
import pydantic_settings; print('pydantic-settings OK')
"
```

Expected: All imports succeed with version numbers.

- [ ] **Step 0.2: Check OCR backend**

```bash
python -c "
import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
ocr = TraditionalOCRTool()
print(f'Backend: {ocr._backend}')
print(f'Available: {ocr.is_available}')
"
```

Expected: `Backend: paddleocr` or `rapidocr`, `Available: True`.
If fails: `pip install rapidocr-onnxruntime` as fallback.

- [ ] **Step 0.3: Check OpenRouter API key**

```bash
python -c "
from finchartaudit.config import get_config
cfg = get_config()
print(f'API key set: {bool(cfg.openrouter_api_key)}')
print(f'Model: {cfg.vlm_model}')
"
```

Expected: `API key set: True`. If False, create `.env` file:
```
FCA_OPENROUTER_API_KEY=sk-or-v1-xxxxx
```

- [ ] **Step 0.4: End-to-end smoke test on one Misviz chart**

```bash
python -c "
import os, sys, json
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, '.')

from finchartaudit.config import get_config
from finchartaudit.vlm.claude_client import OpenRouterVLMClient
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.agents.t2_visual import T2VisualAgent
from data_tools.misviz.loader import MisvizLoader

config = get_config()
vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=config.vlm_model)
memory = FilingMemory()
ocr = TraditionalOCRTool()
agent = T2VisualAgent(vlm=vlm, memory=memory)
agent.set_ocr_tool(ocr)

# Load first misleading synth chart
loader = MisvizLoader()
synth = loader.load_synth()
# Find a truncated_axis example
for i, d in enumerate(synth):
    if 'truncated axis' in d.get('misleader', []):
        instance = loader.get_synth_instance(i)
        break

print(f'Testing: {instance.image_path}')
print(f'Ground truth: {instance.misleader}')

findings = agent.execute({'image_path': instance.image_path, 'page': 1, 'chart_id': 'test'})
print(f'Findings: {len(findings)}')
for f in findings:
    print(f'  [{f.risk_level}] {f.subcategory} ({f.confidence:.0%}): {f.description[:80]}')
"
```

Expected: At least 1 finding for `truncated axis`. If this passes, the baseline is confirmed.

---

## Task 1: Rule Engine Hardening

**Files:**
- Modify: `finchartaudit/tools/rule_check.py`
- Create: `tests/test_rule_check.py`

- [ ] **Step 1.1: Write rule engine tests**

```python
# tests/test_rule_check.py
"""Tests for the deterministic rule engine."""
import pytest
from finchartaudit.tools.rule_check import RuleEngine


@pytest.fixture
def engine():
    return RuleEngine()


class TestTruncatedAxis:
    def test_truncated_bar_chart(self, engine):
        result = engine.run_check("truncated_axis", {
            "axis_values": [80, 85, 90, 95, 100],
            "chart_type": "bar",
        })
        assert result["is_truncated"] is True
        assert result["origin"] == 80
        assert result["exaggeration_factor"] > 1

    def test_non_truncated_bar(self, engine):
        result = engine.run_check("truncated_axis", {
            "axis_values": [0, 25, 50, 75, 100],
            "chart_type": "bar",
        })
        assert result["is_truncated"] is False

    def test_line_chart_allowed(self, engine):
        """Line charts starting above 0 are acceptable."""
        result = engine.run_check("truncated_axis", {
            "axis_values": [80, 85, 90, 95, 100],
            "chart_type": "line",
        })
        assert result["is_truncated"] is False

    def test_empty_values(self, engine):
        result = engine.run_check("truncated_axis", {
            "axis_values": [],
            "chart_type": "bar",
        })
        assert result["is_truncated"] is False


class TestBrokenScale:
    def test_consistent_intervals(self, engine):
        result = engine.run_check("broken_scale", {
            "axis_values": [0, 10, 20, 30, 40],
        })
        assert result["is_broken"] is False

    def test_broken_intervals(self, engine):
        result = engine.run_check("broken_scale", {
            "axis_values": [0, 10, 20, 50, 60],
        })
        assert result["is_broken"] is True

    def test_too_few_values(self, engine):
        result = engine.run_check("broken_scale", {
            "axis_values": [0, 10],
        })
        assert result["is_broken"] is False


class TestValueMismatch:
    def test_matching_values(self, engine):
        result = engine.run_check("value_mismatch", {
            "text_value": 100,
            "chart_value": 102,
            "tolerance": 0.05,
        })
        assert result["is_mismatch"] is False

    def test_mismatched_values(self, engine):
        result = engine.run_check("value_mismatch", {
            "text_value": 100,
            "chart_value": 120,
            "tolerance": 0.05,
        })
        assert result["is_mismatch"] is True
        assert result["difference_pct"] == 20.0


class TestPairingCompleteness:
    def test_all_paired(self, engine):
        result = engine.run_check("pairing_completeness", {
            "nongaap_charts": [
                {"metric_name": "Adjusted EBITDA", "expected_gaap_metric": "net income"},
            ],
            "gaap_charts": [
                {"metric_name": "Net Income"},
            ],
        })
        assert result["all_paired"] is True

    def test_missing_pair(self, engine):
        result = engine.run_check("pairing_completeness", {
            "nongaap_charts": [
                {"metric_name": "Adjusted EBITDA", "expected_gaap_metric": "net income"},
            ],
            "gaap_charts": [],
        })
        assert result["all_paired"] is False
        assert result["missing_pairs"] == 1


class TestProminence:
    def test_balanced(self, engine):
        result = engine.run_check("prominence_score", {
            "nongaap": {"size": 14, "position": 0.5},
            "gaap": {"size": 14, "position": 0.5},
        })
        assert result["is_undue"] is False

    def test_undue_prominence(self, engine):
        result = engine.run_check("prominence_score", {
            "nongaap": {"size": 24, "position": 0.2},
            "gaap": {"size": 12, "position": 0.8},
        })
        assert result["is_undue"] is True


class TestUnknownCheck:
    def test_raises_on_unknown(self, engine):
        with pytest.raises(ValueError, match="Unknown check type"):
            engine.run_check("nonexistent_check", {})
```

- [ ] **Step 1.2: Run tests to verify they pass on existing engine**

```bash
cd C:\Users\chntw\Documents\7180\DD_v1
python -m pytest tests/test_rule_check.py -v
```

Expected: All tests pass. The existing `RuleEngine` already handles all these check types.

- [ ] **Step 1.3: Commit**

```bash
git add tests/test_rule_check.py
git commit -m "test: add rule engine tests for T1-T3 check types"
```

---

## Task 2: T2 Agent Reflect Step

**Files:**
- Modify: `finchartaudit/agents/t2_visual.py`
- Create: `tests/test_t2_parse.py`

- [ ] **Step 2.1: Write T2 parsing tests**

```python
# tests/test_t2_parse.py
"""Tests for T2 response parsing — no VLM calls needed."""
import json
import pytest
from unittest.mock import MagicMock
from finchartaudit.agents.t2_visual import T2VisualAgent
from finchartaudit.memory.filing_memory import FilingMemory


@pytest.fixture
def agent():
    vlm = MagicMock()
    memory = FilingMemory()
    a = T2VisualAgent(vlm=vlm, memory=memory)
    return a


class TestExtractJson:
    def test_direct_json(self, agent):
        text = '{"misleaders": {}, "completeness": {}}'
        result = agent._extract_json(text)
        assert result is not None
        assert "misleaders" in result

    def test_json_in_markdown(self, agent):
        text = 'Here is my analysis:\n```json\n{"misleaders": {}, "completeness": {}}\n```'
        result = agent._extract_json(text)
        assert result is not None

    def test_json_with_surrounding_text(self, agent):
        text = 'I found issues. {"misleaders": {"truncated axis": {"present": true, "confidence": 0.9, "evidence": "Y starts at 80"}}, "completeness": {}} That is all.'
        result = agent._extract_json(text)
        assert result is not None
        assert result["misleaders"]["truncated axis"]["present"] is True

    def test_no_json(self, agent):
        result = agent._extract_json("No JSON here at all.")
        assert result is None


class TestParseResponse:
    def test_misleader_finding(self, agent):
        text = json.dumps({
            "chart_type": "bar",
            "metric_name": "Revenue",
            "is_gaap": True,
            "misleaders": {
                "truncated axis": {"present": True, "confidence": 0.9, "evidence": "Y-axis starts at 80"},
                "3d": {"present": False, "confidence": 0.1, "evidence": ""},
            },
            "completeness": {},
        })
        findings = agent._parse_response(text, "c1", 1, [])
        assert len(findings) == 1
        assert findings[0].subcategory == "truncated axis"
        assert findings[0].confidence == 0.9

    def test_completeness_finding(self, agent):
        text = json.dumps({
            "misleaders": {},
            "completeness": {
                "missing_chart_title": {"present": True, "confidence": 0.8, "evidence": "No title"},
                "missing_legend": {"present": False, "confidence": 0.2, "evidence": ""},
            },
        })
        findings = agent._parse_response(text, "c1", 1, [])
        assert len(findings) == 1
        assert findings[0].category == "completeness"
        assert findings[0].subcategory == "missing_chart_title"

    def test_low_confidence_filtered(self, agent):
        text = json.dumps({
            "misleaders": {
                "truncated axis": {"present": True, "confidence": 0.2, "evidence": "Maybe"},
            },
            "completeness": {},
        })
        findings = agent._parse_response(text, "c1", 1, [])
        assert len(findings) == 0  # Below 0.3 threshold

    def test_risk_level_assignment(self, agent):
        text = json.dumps({
            "misleaders": {
                "truncated axis": {"present": True, "confidence": 0.95, "evidence": "Y starts at 80"},
            },
            "completeness": {},
        })
        findings = agent._parse_response(text, "c1", 1, [])
        assert findings[0].risk_level == "HIGH"
```

- [ ] **Step 2.2: Run tests**

```bash
python -m pytest tests/test_t2_parse.py -v
```

Expected: All pass.

- [ ] **Step 2.3: Add reflect method to T2VisualAgent**

Add to `finchartaudit/agents/t2_visual.py` after the `execute` method:

```python
    def reflect(self, findings: list[AuditFinding]) -> list[dict]:
        """Check if important checks were missed and suggest follow-ups."""
        follow_ups = []
        found_types = {f.subcategory for f in findings}

        # If OCR extracted numeric axis values but no truncated_axis check was done
        has_axis_check = "truncated axis" in found_types or any(
            f.subcategory == "truncated axis" for f in findings
        )
        if not has_axis_check:
            # Check if any tool results contained numeric axis values
            trace = self.memory.audit_trace.get_trace()
            for entry in trace:
                if entry.tool_name == "traditional_ocr" and entry.output_summary:
                    if any(c.isdigit() for c in entry.output_summary):
                        follow_ups.append({
                            "reason": "OCR found numeric values but truncated_axis not checked",
                            "action": "run_rule_check",
                            "check_type": "truncated_axis",
                        })
                        break

        return follow_ups
```

- [ ] **Step 2.4: Run all tests**

```bash
python -m pytest tests/test_t2_parse.py tests/test_rule_check.py -v
```

Expected: All pass.

- [ ] **Step 2.5: Commit**

```bash
git add finchartaudit/agents/t2_visual.py tests/test_t2_parse.py
git commit -m "feat: add T2 reflect step and parsing tests"
```

---

## Task 3: T2 Batch Evaluation Runner

**Files:**
- Create: `finchartaudit/eval/__init__.py`
- Create: `finchartaudit/eval/run_t2_batch.py`

- [ ] **Step 3.1: Create eval package**

```python
# finchartaudit/eval/__init__.py
```

- [ ] **Step 3.2: Write batch runner**

```python
# finchartaudit/eval/run_t2_batch.py
"""Batch evaluation runner for T2 agent on Misviz dataset.

Usage:
    python -m finchartaudit.eval.run_t2_batch --n 50 --dataset synth --output data/eval_results/t2_pilot
    python -m finchartaudit.eval.run_t2_batch --n 1000 --dataset synth --condition vision_only --model claude
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from finchartaudit.config import get_config
from finchartaudit.vlm.claude_client import OpenRouterVLMClient
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.agents.t2_visual import T2VisualAgent
from data_tools.misviz.loader import MisvizLoader
from data_tools.misviz.evaluator import MisvizEvaluator


# Map Misviz misleader names to our detection names
MISVIZ_TO_FCA = {
    "truncated axis": "truncated axis",
    "inverted axis": "inverted axis",
    "misrepresentation": "misrepresentation",
    "3d": "3d",
    "dual axis": "dual axis",
    "inappropriate use of pie chart": "inappropriate use of pie chart",
    "inappropriate use of line chart": "inappropriate use of line chart",
    "inconsistent binning size": "inconsistent binning size",
    "inconsistent tick intervals": "inconsistent tick intervals",
    "discretized continuous variable": "discretized continuous variable",
    "inappropriate item order": "inappropriate item order",
    "inappropriate axis range": "inappropriate axis range",
}


def run_batch(n: int, dataset: str, output_dir: str,
              condition: str = "vision_only", model_name: str = "claude"):
    config = get_config()
    model_id = config.vlm_model if model_name == "claude" else config.qwen_model

    vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=model_id)
    ocr = TraditionalOCRTool()
    loader = MisvizLoader()
    evaluator = MisvizEvaluator()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if dataset == "synth":
        raw_data = loader.load_synth()
    else:
        raw_data = loader.load_real()

    # Sample: take first n instances that have images
    count = 0
    errors = 0
    results = []

    for idx, raw in enumerate(raw_data):
        if count >= n:
            break

        if dataset == "synth":
            instance = loader.get_synth_instance(idx)
        else:
            instance = loader.get_real_instance(idx)

        if not Path(instance.image_path).exists():
            continue

        count += 1
        print(f"[{count}/{n}] id={instance.instance_id} gt={instance.misleader}")

        # Fresh memory per chart
        memory = FilingMemory()
        agent = T2VisualAgent(vlm=vlm, memory=memory)
        agent.set_ocr_tool(ocr)

        try:
            start = time.time()
            findings = agent.execute({
                "image_path": instance.image_path,
                "page": 1,
                "chart_id": f"eval_{instance.instance_id}",
            })
            elapsed = time.time() - start

            # Map findings to misleader names
            predicted = list({f.subcategory for f in findings if f.category == "misleader"})
            gt = [MISVIZ_TO_FCA.get(m, m) for m in instance.misleader]

            evaluator.add_prediction(
                instance_id=instance.instance_id,
                ground_truth=gt,
                predicted=predicted,
                confidences={f.subcategory: f.confidence for f in findings if f.category == "misleader"},
                condition=condition,
                model=model_name,
            )

            result_entry = {
                "instance_id": instance.instance_id,
                "ground_truth": gt,
                "predicted": predicted,
                "findings_count": len(findings),
                "elapsed_s": round(elapsed, 1),
                "all_findings": [f.to_dict() for f in findings],
            }
            results.append(result_entry)
            print(f"  -> predicted={predicted} ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"  -> ERROR: {e}")
            results.append({
                "instance_id": instance.instance_id,
                "error": str(e),
            })

        # Rate limiting
        time.sleep(0.5)

    # Save results
    results_file = out / "raw_results.json"
    results_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Save metrics
    experiment_name = f"t2_{dataset}_{model_name}_{condition}_n{n}"
    metrics = evaluator.save_results(experiment_name)
    evaluator.print_summary()

    print(f"\nDone: {count} charts, {errors} errors")
    print(f"Results: {results_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run T2 batch evaluation on Misviz")
    parser.add_argument("--n", type=int, default=50, help="Number of charts to evaluate")
    parser.add_argument("--dataset", choices=["synth", "real"], default="synth")
    parser.add_argument("--output", type=str, default="data/eval_results/t2_pilot")
    parser.add_argument("--condition", choices=["vision_only", "vision_text"], default="vision_only")
    parser.add_argument("--model", choices=["claude", "qwen"], default="claude")
    args = parser.parse_args()

    run_batch(args.n, args.dataset, args.output, args.condition, args.model)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.3: Test batch runner on 3 charts**

```bash
python -m finchartaudit.eval.run_t2_batch --n 3 --dataset synth --output data/eval_results/t2_smoke
```

Expected: 3 charts processed, `raw_results.json` and `metrics.json` created, summary printed.

- [ ] **Step 3.4: Commit**

```bash
git add finchartaudit/eval/
git commit -m "feat: add T2 batch evaluation runner for Misviz"
```

---

## Task 4: HTML Filing Extractor

**Files:**
- Create: `finchartaudit/tools/html_extract.py`
- Create: `tests/test_html_extract.py`
- Modify: `finchartaudit/tools/registry.py`
- Modify: `finchartaudit/agents/base.py`

- [ ] **Step 4.1: Write HTML extractor tests**

```python
# tests/test_html_extract.py
"""Tests for HTML filing extraction."""
import pytest
from finchartaudit.tools.html_extract import HtmlFilingExtractor


@pytest.fixture
def extractor():
    return HtmlFilingExtractor()


class TestHtmlExtraction:
    def test_extract_tables(self, extractor):
        html = """
        <html><body>
        <table><tr><th>Metric</th><th>2023</th></tr>
        <tr><td>Revenue</td><td>$1,234M</td></tr>
        <tr><td>EBITDA</td><td>$456M</td></tr>
        </table>
        </body></html>
        """
        result = extractor.extract_from_string(html)
        assert len(result["tables"]) >= 1
        assert any("Revenue" in str(t) for t in result["tables"])

    def test_extract_text(self, extractor):
        html = """
        <html><body>
        <p>Adjusted EBITDA margin improved to 25.3% in 2023.</p>
        </body></html>
        """
        result = extractor.extract_from_string(html)
        assert "Adjusted EBITDA" in result["text"]

    def test_detect_nongaap_terms(self, extractor):
        html = """
        <html><body>
        <p>Non-GAAP operating income was $500M. Adjusted EBITDA reached $600M.</p>
        </body></html>
        """
        result = extractor.extract_from_string(html)
        assert len(result["nongaap_mentions"]) >= 1

    def test_extract_from_file(self, extractor, tmp_path):
        html_file = tmp_path / "test.htm"
        html_file.write_text("<html><body><p>Test content</p></body></html>", encoding="utf-8")
        result = extractor.extract_from_file(str(html_file))
        assert "Test content" in result["text"]
```

- [ ] **Step 4.2: Run tests (should fail — html_extract.py not yet created)**

```bash
python -m pytest tests/test_html_extract.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4.3: Implement HTML extractor**

```python
# finchartaudit/tools/html_extract.py
"""HTML filing extractor — parses SEC HTML filings for tables, text, and Non-GAAP mentions."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


# Common Non-GAAP indicator terms
NONGAAP_PATTERNS = [
    r"(?i)\bnon[- ]?gaap\b",
    r"(?i)\badjusted\s+(?:ebitda|ebit|operating|net|gross|eps|income|revenue|margin)",
    r"(?i)\bcore\s+(?:earnings|income|operating)",
    r"(?i)\bfree\s+cash\s+flow\b",
    r"(?i)\borganic\s+(?:revenue|growth|sales)",
    r"(?i)\bpro\s*forma\b",
]
NONGAAP_RE = re.compile("|".join(NONGAAP_PATTERNS))

# Common GAAP metric terms
GAAP_PATTERNS = [
    r"(?i)\bgaap\b",
    r"(?i)\bnet\s+(?:income|loss|earnings)\b",
    r"(?i)\boperating\s+(?:income|loss)\b",
    r"(?i)\bgross\s+profit\b",
    r"(?i)\bearnings\s+per\s+share\b",
    r"(?i)\beps\b",
    r"(?i)\bnet\s+cash\s+(?:provided|used)\b",
]
GAAP_RE = re.compile("|".join(GAAP_PATTERNS))


class _TableParser(HTMLParser):
    """Simple HTML table parser."""

    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: str = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in ("td", "th") and self._in_row:
            self._in_cell = True
            self._current_cell = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._current_row:
                self._current_table.append(self._current_row)
        elif tag == "table" and self._in_table:
            self._in_table = False
            if self._current_table:
                self.tables.append(self._current_table)

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell += data


class HtmlFilingExtractor:
    """Extract structured data from SEC HTML filings."""

    def extract_from_file(self, file_path: str) -> dict:
        html = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return self.extract_from_string(html)

    def extract_from_string(self, html: str) -> dict:
        # Strip tags for plain text
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        # Parse tables
        parser = _TableParser()
        parser.feed(html)

        # Find Non-GAAP mentions with context
        nongaap_mentions = []
        for m in NONGAAP_RE.finditer(text):
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            nongaap_mentions.append({
                "term": m.group(),
                "position": m.start(),
                "context": text[start:end].strip(),
            })

        # Find GAAP mentions
        gaap_mentions = []
        for m in GAAP_RE.finditer(text):
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            gaap_mentions.append({
                "term": m.group(),
                "position": m.start(),
                "context": text[start:end].strip(),
            })

        return {
            "text": text,
            "tables": parser.tables,
            "nongaap_mentions": nongaap_mentions,
            "gaap_mentions": gaap_mentions,
            "table_count": len(parser.tables),
            "text_length": len(text),
        }

    def run(self, file_path: str) -> dict:
        """Tool interface for agent use."""
        return self.extract_from_file(file_path)
```

- [ ] **Step 4.4: Run tests**

```bash
python -m pytest tests/test_html_extract.py -v
```

Expected: All pass.

- [ ] **Step 4.5: Register tool in registry and base agent**

Add to `finchartaudit/tools/registry.py` TOOL_DEFINITIONS list:

```python
    {
        "name": "html_extract",
        "description": (
            "Extract text, tables, and Non-GAAP/GAAP mentions from an HTML SEC filing. "
            "Returns plain text, parsed tables, and term positions. "
            "Use for 8-K, 10-K, DEF 14A filings in HTML format."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to HTML filing"},
            },
            "required": ["file_path"],
        },
    },
```

Add to `finchartaudit/agents/base.py` in `_init_tool_executors`:

```python
        from finchartaudit.tools.html_extract import HtmlFilingExtractor
        self._html_extractor = HtmlFilingExtractor()
        self._tool_executors["html_extract"] = lambda args: self._html_extractor.run(args["file_path"])
```

- [ ] **Step 4.6: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All pass.

- [ ] **Step 4.7: Commit**

```bash
git add finchartaudit/tools/html_extract.py finchartaudit/tools/registry.py finchartaudit/agents/base.py tests/test_html_extract.py
git commit -m "feat: add HTML filing extractor for SEC 8-K/10-K parsing"
```

---

## Task 5: T1 Numerical Consistency Agent

**Files:**
- Create: `finchartaudit/agents/t1_numerical.py`
- Create: `finchartaudit/prompts/t1_numerical.py`
- Create: `tests/test_t1_numerical.py`

- [ ] **Step 5.1: Write T1 prompts**

```python
# finchartaudit/prompts/t1_numerical.py
"""Prompts for T1 Numerical Consistency Agent."""

T1_SYSTEM_PROMPT = """You are a financial document auditor checking numerical consistency.
Your job: verify that numbers in charts match numbers in the surrounding text.

You have access to tools:
- traditional_ocr: Extract numbers from chart images (axis values, data labels).
- html_extract: Extract text and tables from HTML filings.
- query_memory: Check previously extracted data.
- rule_check: Use "value_mismatch" to compare text vs chart values.

WORKFLOW:
1. Extract text claims (numbers, percentages, growth rates) from the filing text.
2. Extract corresponding values from the chart via OCR.
3. Use rule_check(check_type="value_mismatch") to compare pairs.
4. Report any mismatches."""

T1_EXTRACTION_PROMPT = """Analyze this chart and the surrounding text context.

TEXT CONTEXT:
{text_context}

TASK:
1. From the text, identify numerical claims (e.g., "revenue of $1.2B", "grew 15%", "margin improved to 25.3%").
2. From the chart image, use OCR to extract the corresponding data values.
3. Compare each text claim to its chart value using rule_check(check_type="value_mismatch").

Respond with JSON:
{{
  "claims": [
    {{
      "text_claim": "revenue of $1.2B",
      "text_value": 1200,
      "chart_value": 1180,
      "metric": "revenue",
      "match": true,
      "evidence": "Text says $1.2B, chart shows ~$1.18B, within 5% tolerance"
    }}
  ],
  "mismatches_found": 0
}}"""


def build_t1_prompt(text_context: str) -> str:
    return T1_EXTRACTION_PROMPT.format(text_context=text_context)
```

- [ ] **Step 5.2: Write T1 agent**

```python
# finchartaudit/agents/t1_numerical.py
"""T1 Numerical Consistency Agent — detects text-chart value mismatches."""
from __future__ import annotations

import json

from finchartaudit.agents.base import BaseAgent
from finchartaudit.memory.models import AuditFinding, Claim, RiskLevel, Tier
from finchartaudit.prompts.t1_numerical import T1_SYSTEM_PROMPT, build_t1_prompt


class T1NumericalAgent(BaseAgent):
    """Detects numerical inconsistencies between text claims and chart values."""

    agent_name = "t1_numerical"
    available_tools = ["traditional_ocr", "rule_check", "query_memory", "html_extract"]

    def execute(self, task: dict) -> list[AuditFinding]:
        """Analyze chart + text for numerical consistency.

        Args:
            task: {
                "image_path": str,       # Chart image
                "text_context": str,      # Surrounding text from filing
                "page": int,
                "chart_id": str,
            }
        """
        image_path = task["image_path"]
        text_context = task.get("text_context", "")
        page = task.get("page", 0)
        chart_id = task.get("chart_id", f"p{page}_c1")

        if not text_context:
            self.memory.audit_trace.log_reasoning(
                self.agent_name, f"No text context for {chart_id}, skipping T1")
            return []

        prompt = build_t1_prompt(text_context=text_context)

        final_text, tool_results = self.run_with_tools(
            image_path=image_path,
            prompt=prompt,
            system=T1_SYSTEM_PROMPT,
        )

        findings = self._parse_response(final_text, chart_id, page)
        return findings

    def _parse_response(self, text: str, chart_id: str, page: int) -> list[AuditFinding]:
        findings = []
        json_data = self._extract_json(text)
        if not json_data:
            return findings

        for claim_data in json_data.get("claims", []):
            if not claim_data.get("match", True) is False:
                continue

            # Register claim in memory
            claim = Claim(
                text=claim_data.get("text_claim", ""),
                page=page,
                metric=claim_data.get("metric", ""),
                value=claim_data.get("text_value"),
                context=claim_data.get("evidence", ""),
            )
            self.memory.add_claim(claim)

            text_val = claim_data.get("text_value", 0)
            chart_val = claim_data.get("chart_value", 0)
            if text_val and chart_val:
                diff_pct = abs(text_val - chart_val) / abs(text_val) * 100

                findings.append(AuditFinding(
                    tier=Tier.T1,
                    category="text_chart",
                    subcategory="value_mismatch",
                    page=page,
                    chart_id=chart_id,
                    risk_level=RiskLevel.HIGH if diff_pct > 10 else RiskLevel.MEDIUM,
                    confidence=min(0.9, diff_pct / 20),
                    description=(
                        f"Text claims {claim_data.get('metric', 'value')} = {text_val}, "
                        f"but chart shows {chart_val} ({diff_pct:.1f}% difference)"
                    ),
                    correction="Verify and correct the discrepancy between text and chart values.",
                    evidence=[claim_data.get("evidence", "")],
                ))

        for f in findings:
            self.memory.add_finding(f)

        return findings

    def _extract_json(self, text: str) -> dict | None:
        """Reuse T2's JSON extraction logic."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None
```

- [ ] **Step 5.3: Write T1 tests**

```python
# tests/test_t1_numerical.py
"""Tests for T1 Numerical Consistency Agent — parsing only."""
import json
import pytest
from unittest.mock import MagicMock
from finchartaudit.agents.t1_numerical import T1NumericalAgent
from finchartaudit.memory.filing_memory import FilingMemory


@pytest.fixture
def agent():
    vlm = MagicMock()
    memory = FilingMemory()
    return T1NumericalAgent(vlm=vlm, memory=memory)


class TestT1ParseResponse:
    def test_mismatch_detected(self, agent):
        text = json.dumps({
            "claims": [{
                "text_claim": "revenue of $1.2B",
                "text_value": 1200,
                "chart_value": 900,
                "metric": "revenue",
                "match": False,
                "evidence": "Text says $1.2B, chart shows $900M",
            }],
            "mismatches_found": 1,
        })
        findings = agent._parse_response(text, "c1", 1)
        assert len(findings) == 1
        assert findings[0].tier == "T1"
        assert findings[0].subcategory == "value_mismatch"
        assert findings[0].risk_level == "HIGH"

    def test_no_mismatch(self, agent):
        text = json.dumps({
            "claims": [{
                "text_claim": "revenue of $1.2B",
                "text_value": 1200,
                "chart_value": 1210,
                "metric": "revenue",
                "match": True,
                "evidence": "Values match within tolerance",
            }],
            "mismatches_found": 0,
        })
        findings = agent._parse_response(text, "c1", 1)
        assert len(findings) == 0

    def test_no_text_context_skips(self, agent):
        findings = agent.execute({
            "image_path": "fake.png",
            "text_context": "",
            "page": 1,
        })
        assert len(findings) == 0
```

- [ ] **Step 5.4: Run tests**

```bash
python -m pytest tests/test_t1_numerical.py -v
```

Expected: All pass.

- [ ] **Step 5.5: Commit**

```bash
git add finchartaudit/agents/t1_numerical.py finchartaudit/prompts/t1_numerical.py tests/test_t1_numerical.py
git commit -m "feat: add T1 numerical consistency agent"
```

---

## Task 6: T3 GAAP/Non-GAAP Pairing Agent

**Files:**
- Create: `finchartaudit/agents/t3_pairing.py`
- Create: `finchartaudit/prompts/t3_pairing.py`
- Create: `tests/test_t3_pairing.py`

- [ ] **Step 6.1: Write T3 prompts**

```python
# finchartaudit/prompts/t3_pairing.py
"""Prompts for T3 GAAP/Non-GAAP Pairing Agent."""

T3_SYSTEM_PROMPT = """You are an SEC compliance auditor specializing in Non-GAAP financial measures.

Your task: analyze a financial filing to check compliance with SEC Regulation S-K Item 10(e)
and C&DI 100.01-102.13 regarding Non-GAAP measures.

KEY RULES:
1. Every Non-GAAP metric must have a corresponding GAAP metric presented with EQUAL or GREATER prominence.
2. Non-GAAP metrics must be clearly labeled as "Non-GAAP" or "Adjusted".
3. A reconciliation to the most directly comparable GAAP measure must be provided.
4. Non-GAAP measures should not be presented more prominently than GAAP measures (font size, position, color).

You have access to tools:
- html_extract: Extract text, tables, and Non-GAAP mentions from HTML filings.
- rule_check: Use "pairing_completeness" to check if all Non-GAAP metrics have GAAP counterparts.
  Use "prominence_score" to check visual prominence balance.
- query_memory: Check charts and claims already registered.

WORKFLOW:
1. Use html_extract to get all tables and text from the filing.
2. Identify all Non-GAAP metrics and their corresponding GAAP metrics.
3. Build a pairing matrix: for each Non-GAAP metric, is the GAAP counterpart present?
4. Check prominence: is the Non-GAAP metric presented more prominently?
5. Report all violations."""

T3_ANALYSIS_PROMPT = """Analyze this SEC filing for Non-GAAP compliance.

FILING: {file_path}

TASK:
1. Extract all financial metrics from the filing using html_extract.
2. Classify each as GAAP or Non-GAAP.
3. For each Non-GAAP metric, identify the expected GAAP counterpart:
   - Adjusted EBITDA → Net Income
   - Adjusted Operating Income → Operating Income (GAAP)
   - Non-GAAP EPS → GAAP EPS
   - Adjusted Gross Margin → Gross Margin (GAAP)
   - Free Cash Flow → Net Cash from Operations
   - Organic Revenue → Total Revenue (GAAP)
4. Check if the GAAP counterpart is presented in the same section with equal prominence.
5. Use rule_check(check_type="pairing_completeness") with the identified pairs.

Respond with JSON:
{{
  "metrics": [
    {{
      "name": "Adjusted EBITDA",
      "type": "non_gaap",
      "page_or_section": "...",
      "expected_gaap": "Net Income",
      "gaap_found": true/false,
      "prominence_issue": true/false,
      "evidence": "..."
    }}
  ],
  "pairing_matrix": {{
    "total_nongaap": 3,
    "paired": 1,
    "missing": 2,
    "violations": ["Adjusted EBITDA shown without Net Income", "..."]
  }}
}}"""


def build_t3_prompt(file_path: str) -> str:
    return T3_ANALYSIS_PROMPT.format(file_path=file_path)
```

- [ ] **Step 6.2: Write T3 agent**

```python
# finchartaudit/agents/t3_pairing.py
"""T3 GAAP/Non-GAAP Pairing Agent — detects prominence and pairing violations."""
from __future__ import annotations

import json

from finchartaudit.agents.base import BaseAgent
from finchartaudit.memory.models import AuditFinding, PairingEntry, RiskLevel, Tier
from finchartaudit.prompts.t3_pairing import T3_SYSTEM_PROMPT, build_t3_prompt


# Known Non-GAAP → GAAP mappings
NONGAAP_TO_GAAP = {
    "adjusted ebitda": "net income",
    "adjusted ebitda margin": "net income margin",
    "adjusted operating income": "operating income",
    "adjusted operating margin": "operating income margin",
    "adjusted net income": "net income",
    "adjusted eps": "earnings per share",
    "non-gaap eps": "earnings per share",
    "non-gaap operating income": "operating income",
    "free cash flow": "net cash from operations",
    "organic revenue": "total revenue",
    "core operating income": "operating income",
    "adjusted gross margin": "gross margin",
    "adjusted ebit": "operating income",
    "adjusted ebit margin": "operating income margin",
}


class T3PairingAgent(BaseAgent):
    """Detects Non-GAAP prominence and pairing violations in SEC filings."""

    agent_name = "t3_pairing"
    available_tools = ["html_extract", "rule_check", "query_memory"]

    def execute(self, task: dict) -> list[AuditFinding]:
        """Analyze a filing for Non-GAAP compliance.

        Args:
            task: {
                "file_path": str,    # Path to HTML/PDF filing
                "ticker": str,       # Company ticker
                "filing_type": str,  # "8-K", "10-K", etc.
            }
        """
        file_path = task["file_path"]
        ticker = task.get("ticker", "")
        page = task.get("page", 0)

        prompt = build_t3_prompt(file_path=file_path)

        # T3 doesn't need an image — pass empty string
        final_text, tool_results = self.run_with_tools(
            image_path="",
            prompt=prompt,
            system=T3_SYSTEM_PROMPT,
        )

        findings = self._parse_response(final_text, ticker, page)
        return findings

    def _parse_response(self, text: str, ticker: str, page: int) -> list[AuditFinding]:
        findings = []
        json_data = self._extract_json(text)
        if not json_data:
            return findings

        # Process individual metrics
        for metric in json_data.get("metrics", []):
            if metric.get("type") != "non_gaap":
                continue

            name = metric.get("name", "")
            gaap_found = metric.get("gaap_found", True)
            prominence_issue = metric.get("prominence_issue", False)

            # Register pairing in memory
            pairing = PairingEntry(
                expected_gaap_metric=metric.get("expected_gaap", ""),
                pairing_status="paired" if gaap_found else "missing",
            )
            self.memory.pairing_matrix.append(pairing)

            if not gaap_found:
                findings.append(AuditFinding(
                    tier=Tier.T3,
                    category="pairing",
                    subcategory="missing_gaap_counterpart",
                    page=page,
                    risk_level=RiskLevel.HIGH,
                    confidence=0.8,
                    description=(
                        f"{name} presented without corresponding GAAP metric "
                        f"({metric.get('expected_gaap', 'unknown')})"
                    ),
                    correction=(
                        f"Present {metric.get('expected_gaap', 'GAAP counterpart')} "
                        f"with equal or greater prominence alongside {name}."
                    ),
                    evidence=[
                        metric.get("evidence", ""),
                        "SEC basis: Reg S-K Item 10(e)(1)(i)(A) — GAAP comparison required",
                    ],
                ))

            if prominence_issue:
                findings.append(AuditFinding(
                    tier=Tier.T3,
                    category="pairing",
                    subcategory="undue_prominence",
                    page=page,
                    risk_level=RiskLevel.HIGH,
                    confidence=0.7,
                    description=f"{name} has undue prominence over GAAP counterpart",
                    correction="Ensure Non-GAAP measures are not presented more prominently than GAAP.",
                    evidence=[
                        metric.get("evidence", ""),
                        "SEC basis: C&DI 102.10 — Non-GAAP must not have undue prominence",
                    ],
                ))

        # Process overall pairing violations
        pairing_data = json_data.get("pairing_matrix", {})
        for violation in pairing_data.get("violations", []):
            # Avoid duplicating findings already created from metrics
            if not any(violation in f.description for f in findings):
                findings.append(AuditFinding(
                    tier=Tier.T3,
                    category="pairing",
                    subcategory="pairing_violation",
                    page=page,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.6,
                    description=violation,
                    correction="Ensure all Non-GAAP metrics have corresponding GAAP measures.",
                    evidence=["SEC basis: Reg S-K Item 10(e)"],
                ))

        for f in findings:
            self.memory.add_finding(f)
            self.memory.audit_trace.log_finding(
                self.agent_name, f"{f.subcategory}: {f.description[:80]}")

        return findings

    def _extract_json(self, text: str) -> dict | None:
        """Reuse JSON extraction logic."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None
```

- [ ] **Step 6.3: Write T3 tests**

```python
# tests/test_t3_pairing.py
"""Tests for T3 Pairing Agent — parsing only."""
import json
import pytest
from unittest.mock import MagicMock
from finchartaudit.agents.t3_pairing import T3PairingAgent, NONGAAP_TO_GAAP
from finchartaudit.memory.filing_memory import FilingMemory


@pytest.fixture
def agent():
    vlm = MagicMock()
    memory = FilingMemory()
    return T3PairingAgent(vlm=vlm, memory=memory)


class TestT3ParseResponse:
    def test_missing_gaap_counterpart(self, agent):
        text = json.dumps({
            "metrics": [{
                "name": "Adjusted EBITDA Margin",
                "type": "non_gaap",
                "expected_gaap": "Net Income Margin",
                "gaap_found": False,
                "prominence_issue": False,
                "evidence": "Only Adjusted EBITDA Margin shown in Key Metrics section",
            }],
            "pairing_matrix": {
                "total_nongaap": 1,
                "paired": 0,
                "missing": 1,
                "violations": [],
            },
        })
        findings = agent._parse_response(text, "MYE", 1)
        assert len(findings) == 1
        assert findings[0].subcategory == "missing_gaap_counterpart"
        assert findings[0].risk_level == "HIGH"

    def test_undue_prominence(self, agent):
        text = json.dumps({
            "metrics": [{
                "name": "Adjusted EBITDA",
                "type": "non_gaap",
                "expected_gaap": "Net Income",
                "gaap_found": True,
                "prominence_issue": True,
                "evidence": "Adjusted EBITDA in 18pt bold, Net Income in 12pt regular",
            }],
            "pairing_matrix": {"total_nongaap": 1, "paired": 1, "missing": 0, "violations": []},
        })
        findings = agent._parse_response(text, "MYE", 1)
        assert len(findings) == 1
        assert findings[0].subcategory == "undue_prominence"

    def test_all_paired_no_issues(self, agent):
        text = json.dumps({
            "metrics": [{
                "name": "Adjusted Operating Income",
                "type": "non_gaap",
                "expected_gaap": "Operating Income",
                "gaap_found": True,
                "prominence_issue": False,
                "evidence": "Both shown in same table with equal formatting",
            }],
            "pairing_matrix": {"total_nongaap": 1, "paired": 1, "missing": 0, "violations": []},
        })
        findings = agent._parse_response(text, "CTAS", 1)
        assert len(findings) == 0


class TestNongaapMapping:
    def test_known_mappings(self):
        assert NONGAAP_TO_GAAP["adjusted ebitda"] == "net income"
        assert NONGAAP_TO_GAAP["free cash flow"] == "net cash from operations"
        assert NONGAAP_TO_GAAP["organic revenue"] == "total revenue"
```

- [ ] **Step 6.4: Run tests**

```bash
python -m pytest tests/test_t3_pairing.py -v
```

Expected: All pass.

- [ ] **Step 6.5: Commit**

```bash
git add finchartaudit/agents/t3_pairing.py finchartaudit/prompts/t3_pairing.py tests/test_t3_pairing.py
git commit -m "feat: add T3 GAAP/Non-GAAP pairing agent"
```

---

## Task 7: Orchestrator

**Files:**
- Create: `finchartaudit/agents/orchestrator.py`

- [ ] **Step 7.1: Write orchestrator**

```python
# finchartaudit/agents/orchestrator.py
"""Multi-tier orchestrator — dispatches T1, T2, T3 agents and cross-validates."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.memory.models import AuditFinding, RiskLevel
from finchartaudit.vlm.base import VLMClient


class Orchestrator:
    """Runs T1-T3 agents on a filing and cross-validates findings."""

    def __init__(self, vlm: VLMClient, memory: FilingMemory):
        self.vlm = vlm
        self.memory = memory
        self._ocr_tool = None
        self._pdf_tool = None

    def set_ocr_tool(self, ocr_tool):
        self._ocr_tool = ocr_tool

    def set_pdf_tool(self, pdf_tool):
        self._pdf_tool = pdf_tool

    def audit_chart(self, image_path: str, text_context: str = "",
                    page: int = 0, chart_id: str = "") -> list[AuditFinding]:
        """Run T2 (+ T1 if text context available) on a single chart."""
        from finchartaudit.agents.t2_visual import T2VisualAgent
        from finchartaudit.agents.t1_numerical import T1NumericalAgent

        all_findings = []

        # T2: Visual Encoding
        t2 = T2VisualAgent(vlm=self.vlm, memory=self.memory)
        if self._ocr_tool:
            t2.set_ocr_tool(self._ocr_tool)
        t2_findings = t2.execute({
            "image_path": image_path, "page": page,
            "chart_id": chart_id or f"p{page}_c1",
        })
        all_findings.extend(t2_findings)

        # T1: Numerical Consistency (only if text provided)
        if text_context:
            t1 = T1NumericalAgent(vlm=self.vlm, memory=self.memory)
            if self._ocr_tool:
                t1.set_ocr_tool(self._ocr_tool)
            t1_findings = t1.execute({
                "image_path": image_path, "text_context": text_context,
                "page": page, "chart_id": chart_id or f"p{page}_c1",
            })
            all_findings.extend(t1_findings)

        # Cross-validate: escalate if T1+T2 flag same chart
        all_findings = self._cross_validate(all_findings)

        return all_findings

    def audit_filing(self, file_path: str, ticker: str = "",
                     filing_type: str = "") -> list[AuditFinding]:
        """Run T3 on a full filing. T1/T2 should be run on individual charts first."""
        from finchartaudit.agents.t3_pairing import T3PairingAgent

        t3 = T3PairingAgent(vlm=self.vlm, memory=self.memory)
        t3_findings = t3.execute({
            "file_path": file_path,
            "ticker": ticker,
            "filing_type": filing_type,
        })

        # Cross-validate T3 with any existing T2 findings
        all_findings = list(self.memory.findings)
        all_findings = self._cross_validate(all_findings)

        return all_findings

    def _cross_validate(self, findings: list[AuditFinding]) -> list[AuditFinding]:
        """Escalate risk when multiple tiers flag the same chart or metric."""
        # Group findings by chart_id
        by_chart: dict[str, list[AuditFinding]] = {}
        for f in findings:
            if f.chart_id:
                by_chart.setdefault(f.chart_id, []).append(f)

        for chart_id, chart_findings in by_chart.items():
            tiers = {f.tier for f in chart_findings}
            if len(tiers) >= 2:
                # Multiple tiers flagged same chart — escalate medium → high
                for f in chart_findings:
                    if f.risk_level == RiskLevel.MEDIUM:
                        f.risk_level = RiskLevel.HIGH
                        f.evidence.append(
                            f"Cross-validated: flagged by {len(tiers)} tiers ({', '.join(sorted(tiers))})")

        return findings
```

- [ ] **Step 7.2: Commit**

```bash
git add finchartaudit/agents/orchestrator.py
git commit -m "feat: add multi-tier orchestrator with cross-validation"
```

---

## Task 8: Streamlit Filing Scanner (Page 2)

**Files:**
- Modify: `finchartaudit/app.py`

- [ ] **Step 8.1: Add page navigation and Filing Scanner to app.py**

Add page selector and Page 2 logic. Replace the `main()` function in `finchartaudit/app.py`:

```python
def main():
    st.set_page_config(
        page_title="FinChartAudit",
        page_icon="📊",
        layout="wide",
    )

    # Sidebar
    with st.sidebar:
        st.title("FinChartAudit")
        st.caption("Cross-modal consistency verification for SEC financial filings")
        st.divider()

        page = st.radio("Page", ["Single Chart Audit", "Filing Scanner"], index=0)

        st.subheader("Configuration")
        init_components()

        config = st.session_state.config
        st.text(f"Model: {config.vlm_model}")
        st.text(f"OCR: {st.session_state.ocr._backend} "
                f"({'available' if st.session_state.ocr.is_available else 'unavailable'})")

        st.divider()
        st.caption("CS 6180 Generative AI | Final Project")

    if page == "Single Chart Audit":
        page_single_chart()
    else:
        page_filing_scanner()


def page_single_chart():
    """Original Single Chart Audit page."""
    st.header("Single Chart Audit")
    st.markdown("Upload a chart image to detect misleading visualization techniques.")

    uploaded = st.file_uploader(
        "Upload chart image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Drag and drop a chart image or click to browse",
    )

    with st.expander("Optional: Ground-truth textual context"):
        text_context = st.text_area(
            "Provide actual data values for vision+text analysis",
            placeholder="e.g., Revenue: Q1=120M, Q2=125M, Q3=118M, Q4=122M",
            height=80,
        )

    if not uploaded:
        st.info("Upload a chart image to begin, or try one of the sample charts below.")
        show_sample_selector()
        return

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(uploaded.read())
        image_path = f.name

    run_analysis(image_path, text_context)


def page_filing_scanner():
    """Page 2: Full filing scanner with T1-T3."""
    st.header("Filing Scanner")
    st.markdown("Upload an SEC filing (HTML or PDF) to run full T1-T3 compliance audit.")

    # File selection
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload SEC filing",
            type=["htm", "html", "pdf"],
            help="Upload an 8-K, 10-K, or DEF 14A filing",
        )
    with col2:
        ticker = st.text_input("Ticker (optional)", placeholder="e.g., MYE")
        filing_type = st.selectbox("Filing Type", ["8-K", "10-K", "DEF 14A", "20-F", "6-K"])

    # Or pick from downloaded filings
    with st.expander("Or select from downloaded filings"):
        from pathlib import Path
        filings_dir = Path("data/filings")
        if filings_dir.exists():
            companies = sorted([d.name for d in filings_dir.iterdir() if d.is_dir()])
            selected_company = st.selectbox("Company", [""] + companies)
            if selected_company:
                company_dir = filings_dir / selected_company
                files = sorted([
                    f.name for f in (company_dir / "filing").glob("*")
                    if f.suffix in (".htm", ".html", ".pdf") and not f.name.endswith("_meta.json")
                ]) if (company_dir / "filing").exists() else []
                selected_file = st.selectbox("Filing", [""] + files)
                if selected_file and st.button("Load selected filing"):
                    st.session_state.scanner_file = str(company_dir / "filing" / selected_file)
                    st.session_state.scanner_ticker = selected_company
                    st.rerun()

    # Determine file path
    file_path = None
    if uploaded:
        import tempfile
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(uploaded.read())
            file_path = f.name
    elif "scanner_file" in st.session_state:
        file_path = st.session_state.scanner_file
        ticker = ticker or st.session_state.get("scanner_ticker", "")

    if not file_path:
        st.info("Upload a filing or select from downloaded filings to begin.")
        return

    # Run T3 analysis
    if st.button("Run Filing Audit", type="primary"):
        run_filing_analysis(file_path, ticker, filing_type)


def run_filing_analysis(file_path: str, ticker: str, filing_type: str):
    """Run T3 pairing analysis on a filing."""
    from finchartaudit.agents.orchestrator import Orchestrator

    reset_memory()
    memory = st.session_state.memory

    orchestrator = Orchestrator(vlm=st.session_state.vlm, memory=memory)
    orchestrator.set_ocr_tool(st.session_state.ocr)

    status = st.status("Running Filing Audit (T3 Pairing Analysis)...", expanded=True)
    import time
    start = time.time()

    try:
        with status:
            st.write(f"Analyzing {Path(file_path).name}...")
            findings = orchestrator.audit_filing(
                file_path=file_path,
                ticker=ticker,
                filing_type=filing_type,
            )
            elapsed = time.time() - start
            st.write(f"Completed in {elapsed:.1f}s")

        status.update(label=f"Filing audit complete ({elapsed:.1f}s)", state="complete")
    except Exception as e:
        status.update(label="Audit failed", state="error")
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    # Results
    st.success(f"Filing audit complete: {len(findings)} findings in {elapsed:.1f}s")

    summary = memory.get_summary()
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Findings", summary["total_findings"])
    mcol2.metric("T3 Pairing", len([f for f in findings if f.tier == "T3"]))

    risk_dist = summary.get("findings_by_risk", {})
    high_count = sum(v for k, v in risk_dist.items() if "HIGH" in str(k) or "CRITICAL" in str(k))
    mcol3.metric("High/Critical", high_count)
    mcol4.metric("Pairings Tracked", len(memory.pairing_matrix))

    st.divider()

    tab_findings, tab_trace, tab_json = st.tabs(["Findings", "Audit Trace", "Raw JSON"])

    with tab_findings:
        if not findings:
            st.success("No compliance violations detected.")
        else:
            for f in sorted(findings, key=lambda x: ["CRITICAL", "HIGH", "MEDIUM", "LOW"].index(
                    str(x.risk_level).replace("RiskLevel.", "")) if str(x.risk_level).replace("RiskLevel.", "") in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] else 3):
                risk_str = str(f.risk_level).replace("RiskLevel.", "")
                color = RISK_COLORS.get(risk_str, RISK_COLORS.get(str(f.risk_level), "#95a5a6"))
                tier_badge = f'<span style="background: #eee; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">{f.tier}</span>'

                sec_ref = ""
                for ev in f.evidence:
                    if isinstance(ev, str) and "SEC basis:" in ev:
                        sec_ref = ev
                        break
                sec_html = (f'<br/><span style="color: #8e44ad; font-size: 0.85em;">{sec_ref}</span>' if sec_ref else "")

                st.markdown(
                    f'<div style="border-left: 4px solid {color}; padding: 12px; '
                    f'margin: 8px 0; background: #f8f9fa; border-radius: 4px;">'
                    f'{tier_badge} '
                    f'<span style="color: {color}; font-weight: bold;">{risk_str}</span> '
                    f'&mdash; <strong>{f.subcategory}</strong> '
                    f'<span style="color: #666;">({f.confidence:.0%})</span><br/>'
                    f'{f.description}<br/>'
                    f'<span style="color: #27ae60; font-size: 0.9em;">Suggestion: {f.correction}</span>'
                    f'{sec_html}</div>',
                    unsafe_allow_html=True,
                )

    with tab_trace:
        trace = memory.audit_trace.get_trace()
        for i, entry in enumerate(trace, 1):
            icon = {"tool_call": "🔧", "tool_result": "📋", "vlm_reasoning": "🧠",
                    "finding": "🎯", "decision": "⚡"}.get(entry.action, "•")
            tool_info = f" **[{entry.tool_name}]**" if entry.tool_name else ""
            with st.expander(f"{icon} Step {i}: {entry.action}{tool_info}", expanded=False):
                if entry.input_summary:
                    st.markdown(f"**Input:** `{entry.input_summary[:300]}`")
                if entry.output_summary:
                    st.markdown(f"**Output:** `{entry.output_summary[:500]}`")

    with tab_json:
        export = memory.export_json()
        st.json(export)
        st.download_button("Download JSON", data=json.dumps(export, indent=2, default=str),
                          file_name=f"finchartaudit_{ticker or 'filing'}_results.json",
                          mime="application/json")
```

- [ ] **Step 8.2: Test Streamlit app launches**

```bash
cd C:\Users\chntw\Documents\7180\DD_v1
streamlit run finchartaudit/app.py --server.headless true
```

Expected: App launches on localhost, both pages navigate correctly.

- [ ] **Step 8.3: Commit**

```bash
git add finchartaudit/app.py
git commit -m "feat: add Filing Scanner page with T3 pairing analysis"
```

---

## Task 9: T3 Case Study Runner

**Files:**
- Create: `finchartaudit/eval/run_t3_casestudy.py`

- [ ] **Step 9.1: Write case study runner**

```python
# finchartaudit/eval/run_t3_casestudy.py
"""Run T3 pairing analysis on all SEC case study companies.

Usage:
    python -m finchartaudit.eval.run_t3_casestudy
    python -m finchartaudit.eval.run_t3_casestudy --ticker MYE
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from finchartaudit.config import get_config
from finchartaudit.vlm.claude_client import OpenRouterVLMClient
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.agents.orchestrator import Orchestrator


FILINGS_DIR = Path("data/filings")
EVAL_DIR = Path("data/eval_results/t3_casestudy")


def run_casestudy(ticker: str | None = None):
    config = get_config()
    vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=config.vlm_model)
    ocr = TraditionalOCRTool()

    results = {}

    # Find companies with filings
    companies = sorted([d.name for d in FILINGS_DIR.iterdir() if d.is_dir()])
    if ticker:
        companies = [c for c in companies if c == ticker]

    for company in companies:
        filing_dir = FILINGS_DIR / company / "filing"
        if not filing_dir.exists():
            continue

        # Find the main filing (not metadata)
        filings = [f for f in filing_dir.iterdir()
                   if f.suffix in (".htm", ".html", ".pdf") and "_meta" not in f.name]
        if not filings:
            continue

        for filing_path in filings:
            print(f"\n{'='*60}")
            print(f"[{company}] {filing_path.name}")
            print(f"{'='*60}")

            memory = FilingMemory()
            orchestrator = Orchestrator(vlm=vlm, memory=memory)
            orchestrator.set_ocr_tool(ocr)

            try:
                start = time.time()
                findings = orchestrator.audit_filing(
                    file_path=str(filing_path),
                    ticker=company,
                    filing_type=filing_path.stem.split("_")[-1] if "_" in filing_path.stem else "",
                )
                elapsed = time.time() - start

                print(f"  Findings: {len(findings)}")
                for f in findings:
                    print(f"    [{f.risk_level}] {f.subcategory}: {f.description[:80]}")

                results[f"{company}/{filing_path.name}"] = {
                    "ticker": company,
                    "file": filing_path.name,
                    "findings_count": len(findings),
                    "elapsed_s": round(elapsed, 1),
                    "findings": [f.to_dict() for f in findings],
                    "summary": memory.get_summary(),
                }

            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"{company}/{filing_path.name}"] = {
                    "ticker": company,
                    "file": filing_path.name,
                    "error": str(e),
                }

            time.sleep(1)  # Rate limiting between companies

    # Save results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output = EVAL_DIR / "casestudy_results.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {output}")

    # Summary table
    print(f"\n{'Ticker':<8} {'File':<30} {'Findings':>8} {'Time':>6}")
    print("-" * 56)
    for key, data in results.items():
        if "error" not in data:
            print(f"{data['ticker']:<8} {data['file']:<30} {data['findings_count']:>8} {data['elapsed_s']:>5.1f}s")
        else:
            print(f"{data['ticker']:<8} {data['file']:<30} {'ERROR':>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None, help="Run for specific company only")
    args = parser.parse_args()
    run_casestudy(args.ticker)


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Test on one company**

```bash
python -m finchartaudit.eval.run_t3_casestudy --ticker MYE
```

Expected: T3 analysis runs on MYE filing, findings printed, JSON saved.

- [ ] **Step 9.3: Commit**

```bash
git add finchartaudit/eval/run_t3_casestudy.py
git commit -m "feat: add T3 SEC case study evaluation runner"
```

---

## Task 10: Init Git Repository

**Files:**
- Create: `.gitignore`

- [ ] **Step 10.1: Initialize git and create .gitignore**

```bash
cd C:\Users\chntw\Documents\7180\DD_v1
git init
```

```
# .gitignore
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/
dist/
build/
.pytest_cache/

# Large data files
data/misviz_synth/png/
data/misviz_synth/data_tables/
data/misviz_synth/axis_data/
data/misviz_synth/code/
data/misviz/img/
data/arxiv2025-misviz/
data/misviz_download/

# Keep JSON metadata
!data/misviz_synth/misviz_synth.json
!data/misviz/misviz.json

# Temp files
~$*
*.tmp
```

- [ ] **Step 10.2: Initial commit**

```bash
git add .gitignore
git add finchartaudit/ data_tools/ tests/ docs/
git add DESIGN.md IMPLEMENTATION_PLAN.md PROGRESS.md
git add *.py *.md data/companies.json
git commit -m "feat: initial commit — FinChartAudit with T1-T3 agents, Streamlit demo, eval pipeline"
```

---

## Execution Dependency Graph

```
Task 0 (env verify)
  └─> Task 1 (rule tests)
  └─> Task 2 (T2 reflect + parse tests) ──> Task 3 (batch eval runner)
  └─> Task 4 (HTML extractor) ──> Task 5 (T1 agent)
                                └─> Task 6 (T3 agent) ──> Task 7 (orchestrator)
                                                        └─> Task 9 (case study runner)
                                    Task 7 ──> Task 8 (Streamlit Page 2)
  Task 10 (git init) — can run anytime
```

Tasks 1, 2, 4 can run in parallel after Task 0.
Tasks 5, 6 can run in parallel after Task 4.

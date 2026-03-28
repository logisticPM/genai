"""T3 GAAP/Non-GAAP Pairing Agent — detects prominence and pairing violations."""
from __future__ import annotations

import json

from finchartaudit.agents.base import BaseAgent
from finchartaudit.memory.models import AuditFinding, PairingEntry, RiskLevel, Tier
from finchartaudit.prompts.t3_pairing import T3_SYSTEM_PROMPT, build_t3_prompt


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
    agent_name = "t3_pairing"
    available_tools = ["html_extract", "rule_check", "query_memory"]

    def execute(self, task: dict) -> list[AuditFinding]:
        file_path = task["file_path"]
        ticker = task.get("ticker", "")
        page = task.get("page", 0)

        prompt = build_t3_prompt(file_path=file_path)
        final_text, tool_results = self.run_with_tools(
            image_path="", prompt=prompt, system=T3_SYSTEM_PROMPT)

        return self._parse_response(final_text, ticker, page)

    def _parse_response(self, text: str, ticker: str, page: int) -> list[AuditFinding]:
        findings = []
        json_data = self._extract_json(text)
        if not json_data:
            return findings

        for metric in json_data.get("metrics", []):
            if metric.get("type") != "non_gaap":
                continue

            name = metric.get("name", "")
            gaap_found = metric.get("gaap_found", True)
            prominence_issue = metric.get("prominence_issue", False)

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

        pairing_data = json_data.get("pairing_matrix", {})
        for violation in pairing_data.get("violations", []):
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

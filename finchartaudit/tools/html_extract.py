"""HTML filing extractor — parses SEC HTML filings for tables, text, and Non-GAAP mentions."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

NONGAAP_PATTERNS = [
    r"\bnon[- ]?gaap\b",
    r"\badjusted\s+(?:ebitda|ebit|operating|net|gross|eps|income|revenue|margin)",
    r"\bcore\s+(?:earnings|income|operating)",
    r"\bfree\s+cash\s+flow\b",
    r"\borganic\s+(?:revenue|growth|sales)",
    r"\bpro\s*forma\b",
]
NONGAAP_RE = re.compile("|".join(NONGAAP_PATTERNS), re.IGNORECASE)

GAAP_PATTERNS = [
    r"\bgaap\b",
    r"\bnet\s+(?:income|loss|earnings)\b",
    r"\boperating\s+(?:income|loss)\b",
    r"\bgross\s+profit\b",
    r"\bearnings\s+per\s+share\b",
    r"\beps\b",
    r"\bnet\s+cash\s+(?:provided|used)\b",
]
GAAP_RE = re.compile("|".join(GAAP_PATTERNS), re.IGNORECASE)


class _TableParser(HTMLParser):
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
    def extract_from_file(self, file_path: str) -> dict:
        html = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return self.extract_from_string(html)

    def extract_from_string(self, html: str) -> dict:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        parser = _TableParser()
        parser.feed(html)

        nongaap_mentions = []
        for m in NONGAAP_RE.finditer(text):
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            nongaap_mentions.append({
                "term": m.group(),
                "position": m.start(),
                "context": text[start:end].strip(),
            })

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
        return self.extract_from_file(file_path)

"""
Company registry — manages the list of target companies and their metadata.

Usage:
    from data_tools.company_registry import CompanyRegistry

    registry = CompanyRegistry()
    registry.add("AAPL", cik="0000320193", name="Apple Inc.", sector="Technology")
    registry.add("TSLA", cik="0001318605", name="Tesla Inc.", sector="Automotive")

    registry.list_all()
    company = registry.get("AAPL")
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import COMPANIES_FILE


@dataclass
class Company:
    ticker: str
    cik: str
    name: str
    sector: str = ""
    comment_letter_years: list[str] = field(default_factory=list)
    filings_downloaded: list[str] = field(default_factory=list)  # e.g., ["2022_10K", "2023_10K"]
    notes: str = ""


class CompanyRegistry:
    def __init__(self, path: Path = COMPANIES_FILE):
        self.path = path
        self.companies: dict[str, Company] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for ticker, info in data.items():
                self.companies[ticker] = Company(**info)

    def _save(self):
        data = {t: asdict(c) for t, c in self.companies.items()}
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def add(self, ticker: str, cik: str, name: str, sector: str = "",
            comment_letter_years: list[str] | None = None, notes: str = ""):
        self.companies[ticker] = Company(
            ticker=ticker, cik=cik, name=name, sector=sector,
            comment_letter_years=comment_letter_years or [], notes=notes,
        )
        self._save()
        print(f"Added {ticker} ({name}, CIK={cik})")

    def get(self, ticker: str) -> Company | None:
        return self.companies.get(ticker)

    def mark_filing_downloaded(self, ticker: str, filing_id: str):
        if ticker in self.companies:
            if filing_id not in self.companies[ticker].filings_downloaded:
                self.companies[ticker].filings_downloaded.append(filing_id)
                self._save()

    def list_all(self):
        if not self.companies:
            print("No companies registered.")
            return
        print(f"{'Ticker':<8} {'CIK':<12} {'Name':<30} {'Filings':<20} {'Comment Letters'}")
        print("-" * 90)
        for c in self.companies.values():
            filings = ", ".join(c.filings_downloaded) or "none"
            letters = ", ".join(c.comment_letter_years) or "none"
            print(f"{c.ticker:<8} {c.cik:<12} {c.name:<30} {filings:<20} {letters}")

    def list_tickers(self) -> list[str]:
        return list(self.companies.keys())


if __name__ == "__main__":
    registry = CompanyRegistry()
    registry.list_all()

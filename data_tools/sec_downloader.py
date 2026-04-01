"""
SEC EDGAR filing downloader.

Downloads 10-K filings and comment letters for registered companies.

Usage:
    python -m data_tools.sec_downloader --ticker AAPL --years 2022,2023,2024
    python -m data_tools.sec_downloader --all --years 2023,2024
    python -m data_tools.sec_downloader --ticker AAPL --comment-letters
"""
import argparse
import json
import time
from pathlib import Path

import httpx

from .config import FILINGS_DIR, SEC_USER_AGENT
from .company_registry import CompanyRegistry


HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


def search_filings(cik: str, form_type: str = "10-K", count: int = 10) -> list[dict]:
    """Search SEC EDGAR for filings by CIK and form type."""
    # Use the submissions API
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    with httpx.Client(headers=HEADERS, timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    results = []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == form_type and i < len(dates):
            results.append({
                "form": form,
                "date": dates[i],
                "accession": accessions[i],
                "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                "year": dates[i][:4],
            })
            if len(results) >= count:
                break

    return results


def download_filing_document(cik: str, accession: str, primary_doc: str,
                              output_path: Path) -> bool:
    """Download the primary document of a filing."""
    cik_clean = cik.lstrip("0")
    accession_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{primary_doc}"

    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        if resp.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
            print(f"  Downloaded: {output_path.name} ({len(resp.content) / 1024:.0f} KB)")
            return True
        else:
            print(f"  Failed to download: {url} (status {resp.status_code})")
            return False


def search_comment_letters(cik: str) -> list[dict]:
    """Search for SEC comment letters (UPLOAD type) for a company."""
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": f'"comment letter"',
        "dateRange": "custom",
        "startdt": "2022-01-01",
        "enddt": "2025-12-31",
        "forms": "UPLOAD",
        "entities": cik,
    }

    # Alternative: use full-text search API
    search_url = "https://efts.sec.gov/LATEST/search-index"
    params2 = {
        "q": "non-GAAP prominence",
        "forms": "UPLOAD,CORRESP",
    }

    # Simple approach: search submissions for CORRESP forms
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    with httpx.Client(headers=HEADERS, timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    results = []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form in ("CORRESP", "UPLOAD") and i < len(dates):
            results.append({
                "form": form,
                "date": dates[i],
                "accession": accessions[i],
                "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
            })

    return results


def download_filings_for_company(ticker: str, years: list[str],
                                  registry: CompanyRegistry):
    """Download 10-K filings for a company for specified years."""
    company = registry.get(ticker)
    if not company:
        print(f"Company {ticker} not found in registry. Add it first.")
        return

    print(f"\n{'='*60}")
    print(f"Downloading filings for {company.name} ({ticker}, CIK={company.cik})")
    print(f"{'='*60}")

    filings = search_filings(company.cik, form_type="10-K", count=20)
    if not filings:
        print("  No 10-K filings found.")
        return

    print(f"  Found {len(filings)} 10-K filings")

    for filing in filings:
        if filing["year"] not in years:
            continue

        filing_id = f"{filing['year']}_10K"
        output_dir = FILINGS_DIR / ticker / filing_id
        output_path = output_dir / filing["primary_doc"]

        if output_path.exists():
            print(f"  {filing_id} already downloaded, skipping.")
            continue

        print(f"  Downloading {filing_id} (filed {filing['date']})...")
        success = download_filing_document(
            company.cik, filing["accession"], filing["primary_doc"], output_path
        )

        if success:
            # Save filing metadata
            meta = output_dir / "metadata.json"
            meta.write_text(json.dumps(filing, indent=2), encoding="utf-8")
            registry.mark_filing_downloaded(ticker, filing_id)

        time.sleep(0.5)  # Rate limiting


def download_comment_letters_for_company(ticker: str, registry: CompanyRegistry):
    """Download comment letters for a company."""
    company = registry.get(ticker)
    if not company:
        print(f"Company {ticker} not found in registry.")
        return

    print(f"\nSearching comment letters for {company.name}...")
    letters = search_comment_letters(company.cik)

    if not letters:
        print("  No comment letters found.")
        return

    print(f"  Found {len(letters)} correspondence filings")
    output_dir = FILINGS_DIR / ticker / "comment_letters"

    for letter in letters:
        if not letter["primary_doc"]:
            continue

        output_path = output_dir / f"{letter['date']}_{letter['primary_doc']}"
        if output_path.exists():
            continue

        download_filing_document(
            company.cik, letter["accession"], letter["primary_doc"], output_path
        )
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Download SEC filings")
    parser.add_argument("--ticker", type=str, help="Company ticker")
    parser.add_argument("--all", action="store_true", help="Download for all registered companies")
    parser.add_argument("--years", type=str, default="2023,2024",
                        help="Comma-separated years (default: 2023,2024)")
    parser.add_argument("--comment-letters", action="store_true",
                        help="Download comment letters instead of 10-K")
    parser.add_argument("--list", action="store_true", help="List available filings for a company")
    args = parser.parse_args()

    registry = CompanyRegistry()
    years = [y.strip() for y in args.years.split(",")]

    if args.list and args.ticker:
        company = registry.get(args.ticker)
        if company:
            filings = search_filings(company.cik, count=20)
            for f in filings:
                print(f"  {f['form']} | {f['date']} | {f['accession']}")
        return

    tickers = registry.list_tickers() if args.all else ([args.ticker] if args.ticker else [])

    if not tickers:
        print("No companies specified. Use --ticker AAPL or --all")
        registry.list_all()
        return

    for ticker in tickers:
        if args.comment_letters:
            download_comment_letters_for_company(ticker, registry)
        else:
            download_filings_for_company(ticker, years, registry)


if __name__ == "__main__":
    main()

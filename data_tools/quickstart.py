"""
Quick start script — register sample companies and verify tool setup.

Usage:
    python -m data_tools.quickstart
"""
from .company_registry import CompanyRegistry
from .config import DATA_DIR, FILINGS_DIR, CHARTS_DIR, ANNOTATIONS_DIR


# Sample companies from SEC comment letters (Non-GAAP prominence issues, 2023-2024)
# Replace with actual CIKs from Member A's research
SAMPLE_COMPANIES = [
    # These are placeholder examples — update with actual companies from A's list
    ("AAPL", "0000320193", "Apple Inc.", "Technology"),
    ("MSFT", "0000789019", "Microsoft Corp.", "Technology"),
    ("TSLA", "0001318605", "Tesla Inc.", "Automotive"),
]


def main():
    print("=" * 60)
    print("FinChartAudit — Data Tools Quick Start")
    print("=" * 60)

    # Check directories
    print(f"\nData directory:       {DATA_DIR}")
    print(f"Filings directory:    {FILINGS_DIR}")
    print(f"Charts directory:     {CHARTS_DIR}")
    print(f"Annotations directory: {ANNOTATIONS_DIR}")

    # Register companies
    print(f"\n--- Company Registry ---")
    registry = CompanyRegistry()

    for ticker, cik, name, sector in SAMPLE_COMPANIES:
        if not registry.get(ticker):
            registry.add(ticker, cik=cik, name=name, sector=sector)

    print()
    registry.list_all()

    # Check dependencies
    print(f"\n--- Dependency Check ---")
    deps = {
        "PyMuPDF (fitz)": "fitz",
        "Pillow (PIL)": "PIL",
        "httpx": "httpx",
        "streamlit": "streamlit",
        "pydantic": "pydantic",
    }
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name} -- install with: pip install {module}")

    # Usage instructions
    print(f"\n--- Next Steps ---")
    print("""
1. Update SAMPLE_COMPANIES in quickstart.py with actual companies from Member A

2. Download filings:
   python -m data_tools.sec_downloader --ticker AAPL --years 2023,2024

3. Extract charts from PDF:
   python -m data_tools.pdf_extractor --input data/filings/AAPL/2024_10K/filing.pdf --extract-all

4. Start annotation tool:
   streamlit run data_tools/annotator_app.py

5. Export datasets:
   python -m data_tools.export_dataset --format json --output data/eval_datasets/
""")


if __name__ == "__main__":
    main()

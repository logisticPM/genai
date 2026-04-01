"""Configuration for data tools."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
CHARTS_DIR = DATA_DIR / "charts"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
COMPANIES_FILE = DATA_DIR / "companies.json"

# SEC EDGAR
SEC_BASE_URL = "https://efts.sec.gov/LATEST"
SEC_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_USER_AGENT = "FinChartAudit Research project@cornell.edu"

# Ensure directories exist
for d in [DATA_DIR, FILINGS_DIR, CHARTS_DIR, ANNOTATIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

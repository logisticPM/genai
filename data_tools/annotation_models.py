"""
Data models for annotation — shared between annotation tool and export.
"""
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .config import ANNOTATIONS_DIR


# ── T2: Chart-Level Annotation ──

@dataclass
class ChartAnnotation:
    """Annotation for a single chart (T2 ground truth)."""
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    chart_id: str = ""                  # e.g., "p5_c1"
    company: str = ""                   # Ticker
    filing_year: str = ""               # e.g., "2024"
    page_num: int = 0
    image_path: str = ""

    # Chart properties
    chart_type: str = ""                # bar, line, pie, 3d_bar, scatter, etc.
    metric_name: str = ""               # e.g., "Revenue", "Adjusted EBITDA"
    is_gaap: bool = True
    time_window_start: str = ""         # e.g., "2020"
    time_window_end: str = ""           # e.g., "2024"
    axis_origin: float | None = None    # Y-axis start value (None = not applicable)

    # Misleader labels (T2)
    misleaders: list[str] = field(default_factory=list)
    # Possible values: truncated_axis, inverted_axis, 3d_distortion,
    # area_misrepresentation, dual_axis_abuse, cherry_picked_window,
    # misleading_annotations, broken_scale, inappropriate_type,
    # missing_baseline, color_manipulation, missing_footnote

    # Footnote completeness
    has_source: bool = False
    has_date_range: bool = False
    has_methodology: bool = False
    has_units: bool = False

    notes: str = ""
    annotator: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── T3: Pairing Annotation ──

@dataclass
class PairingAnnotation:
    """Annotation for Non-GAAP ↔ GAAP pairing (T3 ground truth)."""
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    company: str = ""
    filing_year: str = ""

    # Non-GAAP side
    nongaap_chart_id: str = ""
    nongaap_metric: str = ""            # e.g., "Adjusted EBITDA"
    nongaap_page: int = 0

    # Expected GAAP counterpart
    expected_gaap_metric: str = ""      # e.g., "Net Income"

    # Pairing status
    gaap_chart_exists: bool = False     # Is there a paired GAAP chart?
    gaap_chart_id: str = ""             # Chart ID of the GAAP counterpart
    gaap_page: int = 0

    # Prominence comparison (if pair exists)
    nongaap_more_prominent: bool = False  # Is Non-GAAP visually more prominent?
    prominence_notes: str = ""            # e.g., "Non-GAAP title 2x larger font"

    # Comparability (if pair exists)
    same_time_window: bool = True
    same_chart_type: bool = True
    comparability_notes: str = ""

    # Reconciliation
    reconciliation_exists: bool = False
    reconciliation_page: int = 0

    # Overall
    sec_compliant: bool = True
    violation_type: str = ""            # e.g., "missing_pair", "undue_prominence", "not_comparable"
    notes: str = ""
    annotator: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── T4: Definition Consistency Annotation ──

@dataclass
class DefinitionInstance:
    """One instance of a metric definition in a specific section."""
    section: str = ""                   # "chart_footnote", "mda", "reconciliation"
    page: int = 0
    definition_text: str = ""           # Exact quote
    excluded_items: list[str] = field(default_factory=list)  # For Non-GAAP


@dataclass
class DefinitionAnnotation:
    """Annotation for metric definition consistency across sections (T4 ground truth)."""
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    company: str = ""
    filing_year: str = ""
    metric_name: str = ""               # e.g., "Adjusted EBITDA"

    # All definitions found
    definitions: list[DefinitionInstance] = field(default_factory=list)

    # Consistency judgment
    is_consistent: bool = True
    discrepancy_description: str = ""   # What's different
    risk_level: str = ""                # LOW, MEDIUM, HIGH

    notes: str = ""
    annotator: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── Annotation Storage ──

class AnnotationStore:
    """Load/save annotations as JSON files."""

    def __init__(self, base_dir: Path = ANNOTATIONS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, company: str, filing_year: str, annotation_type: str) -> Path:
        return self.base_dir / company / f"{filing_year}_{annotation_type}.json"

    def save_chart_annotations(self, company: str, filing_year: str,
                                annotations: list[ChartAnnotation]):
        path = self._file_path(company, filing_year, "charts")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(a) for a in annotations]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_chart_annotations(self, company: str, filing_year: str) -> list[ChartAnnotation]:
        path = self._file_path(company, filing_year, "charts")
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [ChartAnnotation(**d) for d in data]

    def save_pairing_annotations(self, company: str, filing_year: str,
                                  annotations: list[PairingAnnotation]):
        path = self._file_path(company, filing_year, "pairings")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(a) for a in annotations]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_pairing_annotations(self, company: str, filing_year: str) -> list[PairingAnnotation]:
        path = self._file_path(company, filing_year, "pairings")
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [PairingAnnotation(**d) for d in data]

    def save_definition_annotations(self, company: str, filing_year: str,
                                     annotations: list[DefinitionAnnotation]):
        path = self._file_path(company, filing_year, "definitions")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(a) for a in annotations]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_definition_annotations(self, company: str, filing_year: str) -> list[DefinitionAnnotation]:
        path = self._file_path(company, filing_year, "definitions")
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [DefinitionAnnotation(**{k: v for k, v in d.items()
                                         if k != "definitions"},
                                      definitions=[DefinitionInstance(**di) for di in d.get("definitions", [])])
                for d in data]

    def get_annotation_summary(self) -> dict:
        """Get summary of all annotations."""
        summary = {}
        for company_dir in self.base_dir.iterdir():
            if not company_dir.is_dir():
                continue
            company = company_dir.name
            summary[company] = {}
            for f in company_dir.glob("*.json"):
                parts = f.stem.split("_", 1)
                if len(parts) == 2:
                    year, atype = parts
                    data = json.loads(f.read_text(encoding="utf-8"))
                    summary[company][f"{year}_{atype}"] = len(data)
        return summary

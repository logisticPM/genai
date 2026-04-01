"""
PDF extractor — extract charts, tables, text, and section structure from SEC filings.

Usage:
    python -m data_tools.pdf_extractor --input data/filings/AAPL/2024_10K/filing.htm
    python -m data_tools.pdf_extractor --input path/to/filing.pdf --extract-all
"""
import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF

from .config import CHARTS_DIR


@dataclass
class PageContent:
    page_num: int
    has_text: bool
    has_images: bool
    text_length: int
    image_count: int
    detected_section: str | None = None


@dataclass
class ExtractedChart:
    chart_id: str
    page_num: int
    image_path: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    width: int
    height: int
    surrounding_text: str
    caption: str | None = None


@dataclass
class ExtractedSection:
    name: str
    title: str
    start_page: int
    end_page: int


class PDFExtractor:
    """Extract structured content from SEC filing PDFs."""

    # Common 10-K section patterns
    SECTION_PATTERNS = [
        (r"(?i)item\s*1[.\s]*business", "business"),
        (r"(?i)item\s*1a[.\s]*risk\s*factors", "risk_factors"),
        (r"(?i)item\s*7[.\s]*management.s\s*discussion", "mda"),
        (r"(?i)item\s*8[.\s]*financial\s*statements", "financial_statements"),
        (r"(?i)non[- ]?gaap", "nongaap_disclosure"),
        (r"(?i)reconciliation", "reconciliation"),
    ]

    # Minimum image size to be considered a chart (not a logo/icon)
    MIN_CHART_WIDTH = 200
    MIN_CHART_HEIGHT = 150

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(str(self.pdf_path))
        self.page_count = len(self.doc)

    def close(self):
        self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Text Extraction ──

    def extract_text(self, page_num: int) -> str:
        """Extract text from a single page."""
        page = self.doc[page_num]
        return page.get_text("text")

    def extract_all_text(self) -> dict[int, str]:
        """Extract text from all pages."""
        return {i: self.extract_text(i) for i in range(self.page_count)}

    def extract_text_range(self, start_page: int, end_page: int) -> str:
        """Extract text from a page range."""
        texts = []
        for i in range(start_page, min(end_page + 1, self.page_count)):
            texts.append(self.extract_text(i))
        return "\n\n".join(texts)

    # ── Table Extraction ──

    def extract_tables(self, page_num: int) -> list[list[list[str]]]:
        """Extract tables from a page using PyMuPDF's table detection."""
        page = self.doc[page_num]
        tables = page.find_tables()
        result = []
        for table in tables:
            rows = []
            for row in table.extract():
                rows.append([cell if cell else "" for cell in row])
            result.append(rows)
        return result

    # ── Image/Chart Extraction ──

    def extract_page_as_image(self, page_num: int, dpi: int = 200) -> bytes:
        """Render a page as a PNG image."""
        page = self.doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")

    def extract_charts(self, output_dir: Path | None = None) -> list[ExtractedChart]:
        """Extract all chart-sized images from the PDF."""
        if output_dir is None:
            output_dir = CHARTS_DIR / self.pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        charts = []
        for page_num in range(self.page_count):
            page = self.doc[page_num]
            images = page.get_images(full=True)

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    base_image = self.doc.extract_image(xref)
                except Exception:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < self.MIN_CHART_WIDTH or height < self.MIN_CHART_HEIGHT:
                    continue

                # Save image
                ext = base_image.get("ext", "png")
                chart_id = f"p{page_num + 1}_c{img_idx + 1}"
                img_path = output_dir / f"{chart_id}.{ext}"
                img_path.write_bytes(base_image["image"])

                # Get surrounding text
                page_text = page.get_text("text")
                surrounding = page_text[:500] if page_text else ""

                charts.append(ExtractedChart(
                    chart_id=chart_id,
                    page_num=page_num + 1,
                    image_path=str(img_path),
                    bbox=(0, 0, width, height),  # Approximate
                    width=width,
                    height=height,
                    surrounding_text=surrounding,
                ))

        return charts

    def extract_chart_pages_as_images(self, output_dir: Path | None = None,
                                       dpi: int = 200) -> list[dict]:
        """
        For pages that contain charts, render the full page as an image.
        More reliable than extracting individual embedded images.
        """
        if output_dir is None:
            output_dir = CHARTS_DIR / self.pdf_path.stem / "pages"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for page_num in range(self.page_count):
            page = self.doc[page_num]
            images = page.get_images(full=True)

            # Check if page has chart-sized images
            has_chart = any(
                self.doc.extract_image(img[0]).get("width", 0) >= self.MIN_CHART_WIDTH
                and self.doc.extract_image(img[0]).get("height", 0) >= self.MIN_CHART_HEIGHT
                for img in images
                if self._safe_extract_image(img[0]) is not None
            )

            if has_chart:
                img_bytes = self.extract_page_as_image(page_num, dpi=dpi)
                img_path = output_dir / f"page_{page_num + 1}.png"
                img_path.write_bytes(img_bytes)
                results.append({
                    "page_num": page_num + 1,
                    "image_path": str(img_path),
                    "chart_count": len(images),
                })

        return results

    def _safe_extract_image(self, xref: int) -> dict | None:
        try:
            return self.doc.extract_image(xref)
        except Exception:
            return None

    # ── Section Detection ──

    def detect_sections(self) -> list[ExtractedSection]:
        """Detect 10-K sections by matching heading patterns."""
        sections = []
        current_section = None

        for page_num in range(self.page_count):
            text = self.extract_text(page_num)

            for pattern, section_name in self.SECTION_PATTERNS:
                if re.search(pattern, text):
                    # Close previous section
                    if current_section:
                        current_section["end_page"] = page_num
                        sections.append(ExtractedSection(**current_section))

                    # Find the actual heading text
                    match = re.search(pattern, text)
                    line_start = text.rfind("\n", 0, match.start()) + 1
                    line_end = text.find("\n", match.end())
                    title = text[line_start:line_end].strip() if line_end > 0 else text[line_start:match.end()].strip()

                    current_section = {
                        "name": section_name,
                        "title": title[:100],
                        "start_page": page_num + 1,
                        "end_page": self.page_count,
                    }
                    break  # Only one section per page

        # Close last section
        if current_section:
            current_section["end_page"] = self.page_count
            sections.append(ExtractedSection(**current_section))

        return sections

    # ── Document Overview ──

    def get_overview(self) -> dict:
        """Get a summary of the PDF contents."""
        pages = []
        total_images = 0
        total_charts = 0

        for page_num in range(self.page_count):
            page = self.doc[page_num]
            text = page.get_text("text")
            images = page.get_images(full=True)

            chart_count = sum(
                1 for img in images
                if self._safe_extract_image(img[0]) is not None
                and self._safe_extract_image(img[0]).get("width", 0) >= self.MIN_CHART_WIDTH
            )

            pages.append(PageContent(
                page_num=page_num + 1,
                has_text=len(text.strip()) > 50,
                has_images=len(images) > 0,
                text_length=len(text),
                image_count=len(images),
            ))

            total_images += len(images)
            total_charts += chart_count

        sections = self.detect_sections()

        return {
            "file": str(self.pdf_path),
            "page_count": self.page_count,
            "total_images": total_images,
            "estimated_charts": total_charts,
            "sections": [asdict(s) for s in sections],
            "pages_with_images": sum(1 for p in pages if p.has_images),
            "pages_with_text": sum(1 for p in pages if p.has_text),
        }

    # ── Full Extraction Pipeline ──

    def extract_all(self, output_dir: Path) -> dict:
        """Run full extraction: text, charts, sections, tables."""
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting from {self.pdf_path.name} ({self.page_count} pages)")

        # Overview
        overview = self.get_overview()
        (output_dir / "overview.json").write_text(
            json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Sections detected: {len(overview['sections'])}")
        print(f"  Estimated charts: {overview['estimated_charts']}")

        # Sections
        sections = self.detect_sections()
        sections_data = {s.name: asdict(s) for s in sections}
        (output_dir / "sections.json").write_text(
            json.dumps(sections_data, indent=2), encoding="utf-8"
        )

        # Section texts
        texts_dir = output_dir / "section_texts"
        texts_dir.mkdir(exist_ok=True)
        for section in sections:
            text = self.extract_text_range(section.start_page - 1, section.end_page - 1)
            (texts_dir / f"{section.name}.txt").write_text(text, encoding="utf-8")
        print(f"  Section texts saved: {len(sections)}")

        # Charts
        charts = self.extract_charts(output_dir / "charts")
        charts_data = [asdict(c) for c in charts]
        (output_dir / "charts.json").write_text(
            json.dumps(charts_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Charts extracted: {len(charts)}")

        # Chart pages as full-page images
        chart_pages = self.extract_chart_pages_as_images(output_dir / "chart_pages")
        print(f"  Chart pages rendered: {len(chart_pages)}")

        # Tables (first 5 pages with tables for quick check)
        tables_data = {}
        for page_num in range(self.page_count):
            tables = self.extract_tables(page_num)
            if tables:
                tables_data[str(page_num + 1)] = tables
        (output_dir / "tables.json").write_text(
            json.dumps(tables_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Pages with tables: {len(tables_data)}")

        return overview


def main():
    parser = argparse.ArgumentParser(description="Extract content from SEC filing PDFs")
    parser.add_argument("--input", "-i", required=True, help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output directory (default: auto)")
    parser.add_argument("--overview", action="store_true", help="Print overview only")
    parser.add_argument("--extract-all", action="store_true", help="Run full extraction")
    parser.add_argument("--charts-only", action="store_true", help="Extract charts only")
    parser.add_argument("--sections-only", action="store_true", help="Detect sections only")
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    with PDFExtractor(pdf_path) as extractor:
        if args.overview:
            overview = extractor.get_overview()
            print(json.dumps(overview, indent=2))

        elif args.sections_only:
            sections = extractor.detect_sections()
            for s in sections:
                print(f"  {s.name}: pages {s.start_page}-{s.end_page} | {s.title}")

        elif args.charts_only:
            output_dir = Path(args.output) if args.output else None
            charts = extractor.extract_charts(output_dir)
            for c in charts:
                print(f"  {c.chart_id}: page {c.page_num}, {c.width}x{c.height}")

        elif args.extract_all:
            output_dir = Path(args.output) if args.output else CHARTS_DIR / pdf_path.stem
            extractor.extract_all(output_dir)

        else:
            overview = extractor.get_overview()
            print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()

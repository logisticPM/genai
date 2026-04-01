"""
Export annotations into evaluation-ready datasets.

Usage:
    python -m data_tools.export_dataset --format json --output data/eval_datasets/
    python -m data_tools.export_dataset --format csv --tier t3
"""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .config import DATA_DIR
from .annotation_models import AnnotationStore
from .company_registry import CompanyRegistry


def export_t2_dataset(store: AnnotationStore, registry: CompanyRegistry) -> list[dict]:
    """Export T2 chart misleader annotations as evaluation dataset."""
    dataset = []
    for ticker in registry.list_tickers():
        for year in ["2022", "2023", "2024"]:
            annotations = store.load_chart_annotations(ticker, year)
            for ann in annotations:
                dataset.append({
                    "id": f"{ticker}_{year}_{ann.chart_id}",
                    "company": ticker,
                    "year": year,
                    "chart_id": ann.chart_id,
                    "image_path": ann.image_path,
                    "chart_type": ann.chart_type,
                    "metric_name": ann.metric_name,
                    "is_gaap": ann.is_gaap,
                    "time_window": f"{ann.time_window_start}-{ann.time_window_end}",
                    "axis_origin": ann.axis_origin,
                    "misleaders": ann.misleaders,
                    "has_misleader": len(ann.misleaders) > 0,
                    "misleader_count": len(ann.misleaders),
                    "footnote_complete": all([ann.has_source, ann.has_date_range,
                                              ann.has_methodology, ann.has_units]),
                    "footnote_score": sum([ann.has_source, ann.has_date_range,
                                           ann.has_methodology, ann.has_units]) / 4,
                })
    return dataset


def export_t3_dataset(store: AnnotationStore, registry: CompanyRegistry) -> list[dict]:
    """Export T3 pairing annotations as evaluation dataset."""
    dataset = []
    for ticker in registry.list_tickers():
        for year in ["2022", "2023", "2024"]:
            annotations = store.load_pairing_annotations(ticker, year)
            for ann in annotations:
                dataset.append({
                    "id": f"{ticker}_{year}_{ann.nongaap_chart_id}",
                    "company": ticker,
                    "year": year,
                    "nongaap_chart_id": ann.nongaap_chart_id,
                    "nongaap_metric": ann.nongaap_metric,
                    "expected_gaap_metric": ann.expected_gaap_metric,
                    "gaap_chart_exists": ann.gaap_chart_exists,
                    "nongaap_more_prominent": ann.nongaap_more_prominent,
                    "same_time_window": ann.same_time_window,
                    "same_chart_type": ann.same_chart_type,
                    "reconciliation_exists": ann.reconciliation_exists,
                    "sec_compliant": ann.sec_compliant,
                    "violation_type": ann.violation_type,
                })
    return dataset


def export_t4_dataset(store: AnnotationStore, registry: CompanyRegistry) -> list[dict]:
    """Export T4 definition consistency annotations as evaluation dataset."""
    dataset = []
    for ticker in registry.list_tickers():
        for year in ["2022", "2023", "2024"]:
            annotations = store.load_definition_annotations(ticker, year)
            for ann in annotations:
                dataset.append({
                    "id": f"{ticker}_{year}_{ann.metric_name.replace(' ', '_')}",
                    "company": ticker,
                    "year": year,
                    "metric_name": ann.metric_name,
                    "num_definitions": len(ann.definitions),
                    "sections_with_definition": [d.section for d in ann.definitions],
                    "is_consistent": ann.is_consistent,
                    "discrepancy": ann.discrepancy_description,
                    "risk_level": ann.risk_level,
                    "excluded_items_by_section": {
                        d.section: d.excluded_items for d in ann.definitions
                    },
                })
    return dataset


def export_combined_dataset(store: AnnotationStore, registry: CompanyRegistry) -> dict:
    """Export all tiers as a combined dataset."""
    return {
        "t2_charts": export_t2_dataset(store, registry),
        "t3_pairings": export_t3_dataset(store, registry),
        "t4_definitions": export_t4_dataset(store, registry),
        "metadata": {
            "companies": registry.list_tickers(),
            "annotation_summary": store.get_annotation_summary(),
        },
    }


def save_as_json(data: list[dict] | dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output_path} ({len(data) if isinstance(data, list) else 'combined'} records)")


def save_as_csv(data: list[dict], output_path: Path):
    if not data:
        print(f"No data to save for {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Flatten lists for CSV
    flat_data = []
    for row in data:
        flat_row = {}
        for k, v in row.items():
            if isinstance(v, list):
                flat_row[k] = "; ".join(str(x) for x in v)
            elif isinstance(v, dict):
                flat_row[k] = json.dumps(v)
            else:
                flat_row[k] = v
        flat_data.append(flat_row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_data[0].keys())
        writer.writeheader()
        writer.writerows(flat_data)
    print(f"Saved {output_path} ({len(flat_data)} records)")


def main():
    parser = argparse.ArgumentParser(description="Export annotation datasets")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--tier", choices=["t2", "t3", "t4", "all"], default="all")
    parser.add_argument("--output", "-o", default="data/eval_datasets")
    args = parser.parse_args()

    output_dir = Path(args.output)
    store = AnnotationStore()
    registry = CompanyRegistry()

    save_fn = save_as_json if args.format == "json" else save_as_csv

    if args.tier in ("t2", "all"):
        data = export_t2_dataset(store, registry)
        save_fn(data, output_dir / f"t2_charts.{args.format}")

    if args.tier in ("t3", "all"):
        data = export_t3_dataset(store, registry)
        save_fn(data, output_dir / f"t3_pairings.{args.format}")

    if args.tier in ("t4", "all"):
        data = export_t4_dataset(store, registry)
        save_fn(data, output_dir / f"t4_definitions.{args.format}")

    if args.tier == "all" and args.format == "json":
        combined = export_combined_dataset(store, registry)
        save_as_json(combined, output_dir / "combined_dataset.json")

    # Print summary
    print(f"\nExport complete. Files in {output_dir}/")


if __name__ == "__main__":
    main()

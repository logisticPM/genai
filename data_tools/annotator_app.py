"""
Streamlit annotation tool for FinChartAudit ground truth labeling.

Run: streamlit run data_tools/annotator_app.py

Three annotation modes:
1. Chart Annotation (T2) — label misleader types per chart
2. Pairing Annotation (T3) — label Non-GAAP ↔ GAAP pairing status
3. Definition Annotation (T4) — label metric definition consistency across sections
"""
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from .config import DATA_DIR, FILINGS_DIR, CHARTS_DIR, ANNOTATIONS_DIR
from .company_registry import CompanyRegistry
from .annotation_models import (
    ChartAnnotation, PairingAnnotation, DefinitionAnnotation, DefinitionInstance,
    AnnotationStore,
)

store = AnnotationStore()
registry = CompanyRegistry()


def main():
    st.set_page_config(page_title="FinChartAudit Annotator", layout="wide")
    st.title("FinChartAudit — Annotation Tool")

    mode = st.sidebar.radio("Annotation Mode", [
        "T2: Chart Misleaders",
        "T3: Pairing Completeness",
        "T4: Definition Consistency",
        "Summary",
    ])

    # Company & year selection
    tickers = registry.list_tickers()
    if not tickers:
        st.warning("No companies registered. Run `company_registry.py` to add companies first.")
        return

    company = st.sidebar.selectbox("Company", tickers)
    year = st.sidebar.selectbox("Filing Year", ["2024", "2023", "2022"])

    if mode == "T2: Chart Misleaders":
        annotate_charts(company, year)
    elif mode == "T3: Pairing Completeness":
        annotate_pairings(company, year)
    elif mode == "T4: Definition Consistency":
        annotate_definitions(company, year)
    elif mode == "Summary":
        show_summary()


def annotate_charts(company: str, year: str):
    """T2 chart misleader annotation."""
    st.header(f"T2: Chart Misleader Annotation — {company} {year}")

    # Load existing annotations
    annotations = store.load_chart_annotations(company, year)
    annotated_ids = {a.chart_id for a in annotations}

    # Find charts
    charts_dir = CHARTS_DIR / f"{company}_{year}" if (CHARTS_DIR / f"{company}_{year}").exists() else None
    filing_dir = FILINGS_DIR / company / f"{year}_10K"
    charts_json = None

    # Try to find extracted charts
    for search_dir in [charts_dir, filing_dir]:
        if search_dir and (search_dir / "charts.json").exists():
            charts_json = search_dir / "charts.json"
            break

    if not charts_json:
        st.info("No extracted charts found. Run `pdf_extractor.py --extract-all` first, "
                "or upload chart images manually.")

        # Manual upload
        uploaded = st.file_uploader("Upload chart images", type=["png", "jpg", "jpeg"],
                                     accept_multiple_files=True)
        if not uploaded:
            return
        chart_images = [(f.name, f) for f in uploaded]
    else:
        charts_data = json.loads(charts_json.read_text(encoding="utf-8"))
        chart_images = [(c["chart_id"], Path(c["image_path"])) for c in charts_data
                        if Path(c["image_path"]).exists()]

    if not chart_images:
        st.warning("No chart images available.")
        return

    # Chart selector
    chart_names = [name for name, _ in chart_images]
    selected_idx = st.selectbox("Select chart", range(len(chart_names)),
                                 format_func=lambda i: f"{chart_names[i]} {'✓' if chart_names[i] in annotated_ids else ''}")

    chart_id, chart_source = chart_images[selected_idx]

    # Display chart
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Chart Image")
        if isinstance(chart_source, Path):
            st.image(str(chart_source), use_container_width=True)
        else:
            st.image(chart_source, use_container_width=True)

    with col2:
        st.subheader("Annotation")

        # Load existing annotation if available
        existing = next((a for a in annotations if a.chart_id == chart_id), None)

        chart_type = st.selectbox("Chart type",
            ["bar", "line", "pie", "3d_bar", "3d_pie", "area", "scatter", "dual_axis", "other"],
            index=0 if not existing else ["bar", "line", "pie", "3d_bar", "3d_pie", "area", "scatter", "dual_axis", "other"].index(existing.chart_type) if existing and existing.chart_type in ["bar", "line", "pie", "3d_bar", "3d_pie", "area", "scatter", "dual_axis", "other"] else 0)

        metric_name = st.text_input("Metric name", value=existing.metric_name if existing else "")
        is_gaap = st.checkbox("Is GAAP metric", value=existing.is_gaap if existing else True)

        col_tw1, col_tw2 = st.columns(2)
        with col_tw1:
            tw_start = st.text_input("Time window start", value=existing.time_window_start if existing else "")
        with col_tw2:
            tw_end = st.text_input("Time window end", value=existing.time_window_end if existing else "")

        axis_origin = st.number_input("Y-axis origin (leave 0 if starts from 0)",
                                       value=existing.axis_origin or 0.0 if existing else 0.0)

        st.markdown("**Misleader Types**")
        misleader_options = [
            "truncated_axis", "inverted_axis", "3d_distortion",
            "area_misrepresentation", "dual_axis_abuse", "cherry_picked_window",
            "misleading_annotations", "broken_scale", "inappropriate_type",
            "missing_baseline", "color_manipulation", "missing_footnote",
        ]
        selected_misleaders = []
        cols = st.columns(3)
        for i, m in enumerate(misleader_options):
            with cols[i % 3]:
                if st.checkbox(m.replace("_", " "),
                              value=m in existing.misleaders if existing else False,
                              key=f"misleader_{m}"):
                    selected_misleaders.append(m)

        st.markdown("**Footnote Completeness**")
        col_fn1, col_fn2 = st.columns(2)
        with col_fn1:
            has_source = st.checkbox("Has data source", value=existing.has_source if existing else False)
            has_date = st.checkbox("Has date range", value=existing.has_date_range if existing else False)
        with col_fn2:
            has_method = st.checkbox("Has methodology", value=existing.has_methodology if existing else False)
            has_units = st.checkbox("Has units", value=existing.has_units if existing else False)

        notes = st.text_area("Notes", value=existing.notes if existing else "")
        annotator = st.text_input("Annotator", value=existing.annotator if existing else "")

        if st.button("Save Annotation", type="primary"):
            ann = ChartAnnotation(
                chart_id=chart_id, company=company, filing_year=year,
                chart_type=chart_type, metric_name=metric_name, is_gaap=is_gaap,
                time_window_start=tw_start, time_window_end=tw_end,
                axis_origin=axis_origin if axis_origin != 0 else None,
                misleaders=selected_misleaders,
                has_source=has_source, has_date_range=has_date,
                has_methodology=has_method, has_units=has_units,
                notes=notes, annotator=annotator,
            )

            # Update or append
            annotations = [a for a in annotations if a.chart_id != chart_id]
            annotations.append(ann)
            store.save_chart_annotations(company, year, annotations)
            st.success(f"Saved annotation for {chart_id}")

    # Progress
    st.sidebar.markdown("---")
    st.sidebar.metric("Annotated", f"{len(annotated_ids)}/{len(chart_images)}")


def annotate_pairings(company: str, year: str):
    """T3 pairing completeness annotation."""
    st.header(f"T3: Pairing Annotation — {company} {year}")

    # Load chart annotations to find Non-GAAP charts
    chart_annotations = store.load_chart_annotations(company, year)
    nongaap_charts = [a for a in chart_annotations if not a.is_gaap]
    gaap_charts = [a for a in chart_annotations if a.is_gaap]

    if not nongaap_charts:
        st.info("No Non-GAAP charts annotated yet. Complete T2 annotation first to identify Non-GAAP charts.")
        return

    # Load existing pairing annotations
    pairing_annotations = store.load_pairing_annotations(company, year)
    annotated_ids = {a.nongaap_chart_id for a in pairing_annotations}

    st.markdown(f"**Non-GAAP charts found:** {len(nongaap_charts)} | "
                f"**GAAP charts found:** {len(gaap_charts)}")

    # Select Non-GAAP chart
    nongaap_options = [f"{c.chart_id}: {c.metric_name}" for c in nongaap_charts]
    selected_idx = st.selectbox("Select Non-GAAP chart to annotate",
                                 range(len(nongaap_options)),
                                 format_func=lambda i: f"{nongaap_options[i]} {'✓' if nongaap_charts[i].chart_id in annotated_ids else ''}")

    nongaap = nongaap_charts[selected_idx]
    existing = next((a for a in pairing_annotations if a.nongaap_chart_id == nongaap.chart_id), None)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Non-GAAP: {nongaap.metric_name}")
        if Path(nongaap.image_path).exists():
            st.image(nongaap.image_path, use_container_width=True)
        st.caption(f"Page {nongaap.page_num} | {nongaap.chart_type}")

    with col2:
        st.subheader("Pairing Status")

        expected_gaap = st.text_input("Expected GAAP counterpart metric",
                                       value=existing.expected_gaap_metric if existing else "")

        gaap_exists = st.checkbox("GAAP counterpart chart exists",
                                   value=existing.gaap_chart_exists if existing else False)

        gaap_chart_id = ""
        if gaap_exists:
            gaap_options = [f"{c.chart_id}: {c.metric_name}" for c in gaap_charts]
            if gaap_options:
                gaap_selected = st.selectbox("Select GAAP counterpart", gaap_options)
                gaap_chart_id = gaap_selected.split(":")[0].strip()
            else:
                gaap_chart_id = st.text_input("GAAP chart ID (manual)")

            nongaap_prominent = st.checkbox("Non-GAAP more visually prominent",
                                             value=existing.nongaap_more_prominent if existing else False)
            prominence_notes = st.text_input("Prominence notes",
                                              value=existing.prominence_notes if existing else "")
            same_window = st.checkbox("Same time window",
                                       value=existing.same_time_window if existing else True)
            same_type = st.checkbox("Same chart type",
                                     value=existing.same_chart_type if existing else True)
            comparability_notes = st.text_input("Comparability notes",
                                                 value=existing.comparability_notes if existing else "")
        else:
            nongaap_prominent = False
            prominence_notes = ""
            same_window = True
            same_type = True
            comparability_notes = ""

        recon_exists = st.checkbox("Reconciliation table exists",
                                    value=existing.reconciliation_exists if existing else False)
        recon_page = st.number_input("Reconciliation page", value=existing.reconciliation_page if existing else 0)

        sec_compliant = st.checkbox("SEC compliant (overall)",
                                     value=existing.sec_compliant if existing else True)

        violation_options = ["", "missing_pair", "undue_prominence", "not_comparable",
                            "missing_reconciliation", "multiple_violations"]
        violation_type = st.selectbox("Violation type", violation_options,
                                       index=violation_options.index(existing.violation_type) if existing and existing.violation_type in violation_options else 0)

        notes = st.text_area("Notes", value=existing.notes if existing else "")
        annotator = st.text_input("Annotator", value=existing.annotator if existing else "", key="t3_annotator")

        if st.button("Save Pairing Annotation", type="primary"):
            ann = PairingAnnotation(
                company=company, filing_year=year,
                nongaap_chart_id=nongaap.chart_id,
                nongaap_metric=nongaap.metric_name,
                nongaap_page=nongaap.page_num,
                expected_gaap_metric=expected_gaap,
                gaap_chart_exists=gaap_exists,
                gaap_chart_id=gaap_chart_id,
                nongaap_more_prominent=nongaap_prominent,
                prominence_notes=prominence_notes,
                same_time_window=same_window,
                same_chart_type=same_type,
                comparability_notes=comparability_notes,
                reconciliation_exists=recon_exists,
                reconciliation_page=recon_page,
                sec_compliant=sec_compliant,
                violation_type=violation_type,
                notes=notes, annotator=annotator,
            )
            pairing_annotations = [a for a in pairing_annotations if a.nongaap_chart_id != nongaap.chart_id]
            pairing_annotations.append(ann)
            store.save_pairing_annotations(company, year, pairing_annotations)
            st.success(f"Saved pairing annotation for {nongaap.metric_name}")

    st.sidebar.markdown("---")
    st.sidebar.metric("Annotated", f"{len(annotated_ids)}/{len(nongaap_charts)}")


def annotate_definitions(company: str, year: str):
    """T4 definition consistency annotation."""
    st.header(f"T4: Definition Consistency — {company} {year}")

    # Load existing
    def_annotations = store.load_definition_annotations(company, year)
    annotated_metrics = {a.metric_name for a in def_annotations}

    # Input metric name
    metric_name = st.text_input("Non-GAAP metric name to annotate",
                                 placeholder="e.g., Adjusted EBITDA")

    if not metric_name:
        if def_annotations:
            st.subheader("Existing Annotations")
            for a in def_annotations:
                status = "Consistent" if a.is_consistent else f"INCONSISTENT: {a.discrepancy_description}"
                st.markdown(f"- **{a.metric_name}**: {status} ({len(a.definitions)} instances)")
        return

    existing = next((a for a in def_annotations if a.metric_name == metric_name), None)

    st.subheader(f"Definition instances for: {metric_name}")

    sections = ["chart_footnote", "mda", "reconciliation", "risk_factors", "earnings_release", "other"]

    # Dynamic number of instances
    num_instances = st.number_input("Number of definition instances found",
                                     min_value=1, max_value=10,
                                     value=len(existing.definitions) if existing else 2)

    instances = []
    for i in range(num_instances):
        st.markdown(f"**Instance {i + 1}**")
        ex_inst = existing.definitions[i] if existing and i < len(existing.definitions) else None

        col1, col2 = st.columns([1, 2])
        with col1:
            section = st.selectbox(f"Section", sections,
                                    index=sections.index(ex_inst.section) if ex_inst and ex_inst.section in sections else 0,
                                    key=f"def_section_{i}")
            page = st.number_input(f"Page", value=ex_inst.page if ex_inst else 0, key=f"def_page_{i}")

        with col2:
            def_text = st.text_area(f"Definition text (exact quote)",
                                     value=ex_inst.definition_text if ex_inst else "",
                                     key=f"def_text_{i}", height=80)
            excluded = st.text_input(f"Excluded items (comma-separated)",
                                      value=", ".join(ex_inst.excluded_items) if ex_inst else "",
                                      key=f"def_excluded_{i}")

        instances.append(DefinitionInstance(
            section=section, page=page, definition_text=def_text,
            excluded_items=[x.strip() for x in excluded.split(",") if x.strip()],
        ))

    st.markdown("---")
    is_consistent = st.checkbox("Definitions are consistent",
                                 value=existing.is_consistent if existing else True)
    discrepancy = st.text_area("Discrepancy description (if inconsistent)",
                                value=existing.discrepancy_description if existing else "")
    risk = st.selectbox("Risk level", ["", "LOW", "MEDIUM", "HIGH"],
                         index=["", "LOW", "MEDIUM", "HIGH"].index(existing.risk_level) if existing and existing.risk_level in ["", "LOW", "MEDIUM", "HIGH"] else 0)
    notes = st.text_area("Notes", value=existing.notes if existing else "", key="t4_notes")
    annotator = st.text_input("Annotator", value=existing.annotator if existing else "", key="t4_annotator")

    if st.button("Save Definition Annotation", type="primary"):
        ann = DefinitionAnnotation(
            company=company, filing_year=year, metric_name=metric_name,
            definitions=instances,
            is_consistent=is_consistent,
            discrepancy_description=discrepancy,
            risk_level=risk, notes=notes, annotator=annotator,
        )
        def_annotations = [a for a in def_annotations if a.metric_name != metric_name]
        def_annotations.append(ann)
        store.save_definition_annotations(company, year, def_annotations)
        st.success(f"Saved definition annotation for {metric_name}")


def show_summary():
    """Show annotation progress summary."""
    st.header("Annotation Summary")

    summary = store.get_annotation_summary()
    if not summary:
        st.info("No annotations yet.")
        return

    for company, files in summary.items():
        st.subheader(company)
        for file_key, count in files.items():
            st.markdown(f"  - **{file_key}**: {count} annotations")


if __name__ == "__main__":
    main()

"""FinChartAudit — Streamlit Demo Application."""
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from PIL import Image

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def init_components():
    """Initialize VLM, OCR, memory (cached in session state)."""
    if "initialized" in st.session_state:
        return

    from finchartaudit.config import get_config
    from finchartaudit.vlm.claude_client import OpenRouterVLMClient
    from finchartaudit.memory.filing_memory import FilingMemory
    from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
    from finchartaudit.agents.t2_visual import T2VisualAgent

    config = get_config()

    if not config.openrouter_api_key:
        st.error("Set FCA_OPENROUTER_API_KEY in .env or environment")
        st.stop()

    vlm = OpenRouterVLMClient(
        api_key=config.openrouter_api_key,
        model=config.vlm_model,
        base_url=config.openrouter_base_url,
    )

    ocr = TraditionalOCRTool()
    memory = FilingMemory()

    agent = T2VisualAgent(vlm=vlm, memory=memory)
    agent.set_ocr_tool(ocr)

    st.session_state.vlm = vlm
    st.session_state.ocr = ocr
    st.session_state.memory = memory
    st.session_state.agent = agent
    st.session_state.config = config
    st.session_state.initialized = True


def reset_memory():
    """Reset memory for a new analysis."""
    from finchartaudit.memory.filing_memory import FilingMemory
    from finchartaudit.agents.t2_visual import T2VisualAgent

    memory = FilingMemory()
    agent = T2VisualAgent(vlm=st.session_state.vlm, memory=memory)
    agent.set_ocr_tool(st.session_state.ocr)
    st.session_state.memory = memory
    st.session_state.agent = agent


RISK_COLORS = {
    "HIGH": "#e74c3c",
    "CRITICAL": "#c0392b",
    "MEDIUM": "#f39c12",
    "LOW": "#3498db",
    "RiskLevel.HIGH": "#e74c3c",
    "RiskLevel.CRITICAL": "#c0392b",
    "RiskLevel.MEDIUM": "#f39c12",
    "RiskLevel.LOW": "#3498db",
}


def main():
    st.set_page_config(
        page_title="FinChartAudit",
        page_icon="📊",
        layout="wide",
    )

    with st.sidebar:
        st.title("FinChartAudit")
        st.caption("Cross-modal consistency verification for SEC financial filings")
        st.divider()

        page = st.radio("Page", ["Single Chart Audit", "Filing Scanner"], index=0)

        st.subheader("Configuration")
        init_components()

        config = st.session_state.config
        st.text(f"Model: {config.vlm_model}")
        st.text(f"OCR: {st.session_state.ocr._backend} "
                f"({'available' if st.session_state.ocr.is_available else 'unavailable'})")

        st.divider()
        st.caption("CS 6180 Generative AI | Final Project")

    if page == "Single Chart Audit":
        page_single_chart()
    else:
        page_filing_scanner()


def page_single_chart():
    st.header("Single Chart Audit")
    st.markdown("Upload a chart image to detect misleading visualization techniques.")

    uploaded = st.file_uploader(
        "Upload chart image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Drag and drop a chart image or click to browse",
    )

    with st.expander("Optional: Ground-truth textual context"):
        text_context = st.text_area(
            "Provide actual data values for vision+text analysis",
            placeholder="e.g., Revenue: Q1=120M, Q2=125M, Q3=118M, Q4=122M",
            height=80,
        )

    if not uploaded:
        st.info("Upload a chart image to begin, or try one of the sample charts below.")
        show_sample_selector()
        return

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(uploaded.read())
        image_path = f.name

    run_analysis(image_path, text_context)


def show_sample_selector():
    """Show a few sample Misviz charts for quick testing."""
    import json
    from pathlib import Path

    misviz_json = Path("data/misviz/misviz.json")
    if not misviz_json.exists():
        return

    data = json.loads(misviz_json.read_text(encoding="utf-8"))

    # Find sample charts by misleader type
    samples = {}
    for d in data:
        for m in d.get("misleader", []):
            if m not in samples:
                img_name = d["image_path"].split("/")[-1]
                img_path = Path("data/misviz/img") / img_name
                if img_path.exists():
                    samples[m] = {"path": str(img_path), "chart_type": d.get("chart_type", [])}
        if len(samples) >= 6:
            break

    if not samples:
        return

    st.subheader("Sample Charts (from Misviz)")
    cols = st.columns(min(len(samples), 3))
    for i, (misleader, info) in enumerate(list(samples.items())[:6]):
        with cols[i % 3]:
            try:
                img = Image.open(info["path"])
                st.image(img, caption=f"{misleader}", use_container_width=True)
                if st.button(f"Analyze", key=f"sample_{i}"):
                    run_analysis(info["path"], "")
            except Exception:
                pass


def run_analysis(image_path: str, text_context: str = ""):
    """Run T2 analysis on the uploaded chart."""
    reset_memory()
    agent = st.session_state.agent
    memory = st.session_state.memory

    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.subheader("Chart Image")
        try:
            img = Image.open(image_path)
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Cannot load image: {e}")
            return

    with col_results:
        st.subheader("Analysis")

        status = st.status("Running T2 Visual Encoding Audit...", expanded=True)
        start = time.time()
        try:
            with status:
                st.write("Sending image to VLM...")
                findings = agent.execute({
                    "image_path": image_path,
                    "page": 1,
                    "chart_id": "uploaded_chart",
                })
                elapsed = time.time() - start

                # Show what happened
                trace = memory.audit_trace.get_trace()
                tool_calls = [e for e in trace if e.action == "tool_call"]
                st.write(f"Completed: {len(tool_calls)} tool calls, {elapsed:.1f}s")

            status.update(label=f"Analysis complete ({elapsed:.1f}s)", state="complete")
        except Exception as e:
            status.update(label="Analysis failed", state="error")
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

        st.success(f"Analysis complete in {elapsed:.1f}s")

        # Summary metrics
        summary = memory.get_summary()
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Findings", summary["total_findings"])
        mcol2.metric("Charts Registered", summary["total_charts"])

        risk_dist = summary.get("findings_by_risk", {})
        high_count = sum(v for k, v in risk_dist.items() if "HIGH" in str(k) or "CRITICAL" in str(k))
        mcol3.metric("High/Critical", high_count)

    # Findings
    st.divider()

    tab_findings, tab_trace, tab_json = st.tabs(["Findings", "Audit Trace", "Raw JSON"])

    with tab_findings:
        if not findings:
            st.info("No misleading techniques detected.")
        else:
            for f in sorted(findings, key=lambda x: ["CRITICAL", "HIGH", "MEDIUM", "LOW"].index(
                    str(x.risk_level).replace("RiskLevel.", "")) if str(x.risk_level).replace("RiskLevel.", "") in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] else 3):
                risk_str = str(f.risk_level).replace("RiskLevel.", "")
                color = RISK_COLORS.get(risk_str, RISK_COLORS.get(str(f.risk_level), "#95a5a6"))

                # Find SEC rule reference in evidence
                sec_ref = ""
                for ev in f.evidence:
                    if isinstance(ev, str) and ev.startswith("SEC basis:"):
                        sec_ref = ev
                        break

                sec_html = (f'<br/><span style="color: #8e44ad; font-size: 0.85em;">'
                            f'{sec_ref}</span>' if sec_ref else "")
                category_badge = (f'<span style="background: #eee; padding: 2px 6px; '
                                  f'border-radius: 3px; font-size: 0.8em; margin-left: 8px;">'
                                  f'{f.category}</span>')

                st.markdown(
                    f'<div style="border-left: 4px solid {color}; padding: 12px; '
                    f'margin: 8px 0; background: #f8f9fa; border-radius: 4px;">'
                    f'<span style="color: {color}; font-weight: bold;">{risk_str}</span> '
                    f'&mdash; <strong>{f.subcategory}</strong>{category_badge} '
                    f'<span style="color: #666;">(confidence: {f.confidence:.0%})</span><br/>'
                    f'<span>{f.description}</span><br/>'
                    f'<span style="color: #27ae60; font-size: 0.9em;">'
                    f'Suggestion: {f.correction}</span>'
                    f'{sec_html}</div>',
                    unsafe_allow_html=True,
                )

    with tab_trace:
        trace = memory.audit_trace.get_trace()
        if not trace:
            st.info("No trace entries.")
        else:
            for i, entry in enumerate(trace, 1):
                icon = {
                    "tool_call": "🔧",
                    "tool_result": "📋",
                    "vlm_reasoning": "🧠",
                    "finding": "🎯",
                    "decision": "⚡",
                }.get(entry.action, "•")

                tool_info = f" **[{entry.tool_name}]**" if entry.tool_name else ""
                content = entry.output_summary or entry.input_summary or entry.decision or ""

                with st.expander(f"{icon} Step {i}: {entry.action}{tool_info}", expanded=(i <= 3)):
                    if entry.input_summary:
                        st.markdown(f"**Input:** `{entry.input_summary[:300]}`")
                    if entry.output_summary:
                        st.markdown(f"**Output:** `{entry.output_summary[:500]}`")
                    if entry.decision:
                        st.markdown(f"**Decision:** {entry.decision}")

    with tab_json:
        export = memory.export_json()
        st.json(export)
        st.download_button(
            "Download JSON",
            data=json.dumps(export, indent=2, default=str),
            file_name="finchartaudit_results.json",
            mime="application/json",
        )


def page_filing_scanner():
    """Page 2: Full filing scanner with T3."""
    st.header("Filing Scanner")
    st.markdown("Upload an SEC filing (HTML or PDF) to run T3 Non-GAAP compliance audit.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload SEC filing",
            type=["htm", "html", "pdf"],
            help="Upload an 8-K, 10-K, or DEF 14A filing",
        )
    with col2:
        ticker = st.text_input("Ticker (optional)", placeholder="e.g., MYE")
        filing_type = st.selectbox("Filing Type", ["8-K", "10-K", "DEF 14A", "20-F", "6-K"])

    with st.expander("Or select from downloaded filings"):
        filings_dir = Path("data/filings")
        if filings_dir.exists():
            companies = sorted([d.name for d in filings_dir.iterdir() if d.is_dir()])
            selected_company = st.selectbox("Company", [""] + companies)
            if selected_company:
                company_dir = filings_dir / selected_company
                filing_subdir = company_dir / "filing"
                files = sorted([
                    f.name for f in filing_subdir.glob("*")
                    if f.suffix in (".htm", ".html", ".pdf") and "_meta" not in f.name
                ]) if filing_subdir.exists() else []
                selected_file = st.selectbox("Filing", [""] + files)
                if selected_file and st.button("Load selected filing"):
                    st.session_state.scanner_file = str(filing_subdir / selected_file)
                    st.session_state.scanner_ticker = selected_company
                    st.rerun()

    file_path = None
    if uploaded:
        import tempfile
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(uploaded.read())
            file_path = f.name
    elif "scanner_file" in st.session_state:
        file_path = st.session_state.scanner_file
        ticker = ticker or st.session_state.get("scanner_ticker", "")

    if not file_path:
        st.info("Upload a filing or select from downloaded filings to begin.")
        return

    if st.button("Run Filing Audit", type="primary"):
        run_filing_analysis(file_path, ticker, filing_type)


def run_filing_analysis(file_path: str, ticker: str, filing_type: str):
    """Run T3 pairing analysis on a filing."""
    from finchartaudit.agents.orchestrator import Orchestrator

    reset_memory()
    memory = st.session_state.memory

    orchestrator = Orchestrator(vlm=st.session_state.vlm, memory=memory)
    orchestrator.set_ocr_tool(st.session_state.ocr)

    status = st.status("Running Filing Audit (T3 Pairing Analysis)...", expanded=True)
    start = time.time()

    try:
        with status:
            st.write(f"Analyzing {Path(file_path).name}...")
            findings = orchestrator.audit_filing(
                file_path=file_path, ticker=ticker, filing_type=filing_type)
            elapsed = time.time() - start
            st.write(f"Completed in {elapsed:.1f}s")
        status.update(label=f"Filing audit complete ({elapsed:.1f}s)", state="complete")
    except Exception as e:
        status.update(label="Audit failed", state="error")
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    st.success(f"Filing audit complete: {len(findings)} findings in {elapsed:.1f}s")

    summary = memory.get_summary()
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Findings", summary["total_findings"])
    mcol2.metric("T3 Pairing", len([f for f in findings if f.tier == "T3"]))
    risk_dist = summary.get("findings_by_risk", {})
    high_count = sum(v for k, v in risk_dist.items() if "HIGH" in str(k) or "CRITICAL" in str(k))
    mcol3.metric("High/Critical", high_count)
    mcol4.metric("Pairings Tracked", len(memory.pairing_matrix))

    st.divider()
    tab_findings, tab_trace, tab_json = st.tabs(["Findings", "Audit Trace", "Raw JSON"])

    with tab_findings:
        if not findings:
            st.success("No compliance violations detected.")
        else:
            for f in sorted(findings, key=lambda x: ["CRITICAL", "HIGH", "MEDIUM", "LOW"].index(
                    str(x.risk_level).replace("RiskLevel.", "")) if str(x.risk_level).replace("RiskLevel.", "") in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] else 3):
                risk_str = str(f.risk_level).replace("RiskLevel.", "")
                color = RISK_COLORS.get(risk_str, RISK_COLORS.get(str(f.risk_level), "#95a5a6"))
                tier_badge = f'<span style="background: #eee; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">{f.tier}</span>'
                sec_ref = ""
                for ev in f.evidence:
                    if isinstance(ev, str) and "SEC basis:" in ev:
                        sec_ref = ev
                        break
                sec_html = f'<br/><span style="color: #8e44ad; font-size: 0.85em;">{sec_ref}</span>' if sec_ref else ""

                st.markdown(
                    f'<div style="border-left: 4px solid {color}; padding: 12px; '
                    f'margin: 8px 0; background: #f8f9fa; border-radius: 4px;">'
                    f'{tier_badge} '
                    f'<span style="color: {color}; font-weight: bold;">{risk_str}</span> '
                    f'&mdash; <strong>{f.subcategory}</strong> '
                    f'<span style="color: #666;">({f.confidence:.0%})</span><br/>'
                    f'{f.description}<br/>'
                    f'<span style="color: #27ae60; font-size: 0.9em;">Suggestion: {f.correction}</span>'
                    f'{sec_html}</div>',
                    unsafe_allow_html=True,
                )

    with tab_trace:
        trace = memory.audit_trace.get_trace()
        for i, entry in enumerate(trace, 1):
            icon = {"tool_call": "🔧", "tool_result": "📋", "vlm_reasoning": "🧠",
                    "finding": "🎯", "decision": "⚡"}.get(entry.action, "•")
            tool_info = f" **[{entry.tool_name}]**" if entry.tool_name else ""
            with st.expander(f"{icon} Step {i}: {entry.action}{tool_info}", expanded=False):
                if entry.input_summary:
                    st.markdown(f"**Input:** `{entry.input_summary[:300]}`")
                if entry.output_summary:
                    st.markdown(f"**Output:** `{entry.output_summary[:500]}`")

    with tab_json:
        export = memory.export_json()
        st.json(export)
        st.download_button("Download JSON", data=json.dumps(export, indent=2, default=str),
                          file_name=f"finchartaudit_{ticker or 'filing'}_results.json",
                          mime="application/json")


if __name__ == "__main__":
    main()

"""
FinChartAudit — Capstone Demo
Run: streamlit run demo_app.py
"""
import json
import base64
import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "finchartaudit" / "results"

MISLEADER_LABELS = {
    "misrepresentation":               "Misrepresentation",
    "3d":                              "3D Distortion",
    "truncated axis":                  "Truncated Axis",
    "inappropriate use of pie chart":  "Pie Chart Misuse",
    "inconsistent tick intervals":     "Inconsistent Ticks",
    "dual axis":                       "Dual Axis Abuse",
    "inconsistent binning size":       "Inconsistent Bins",
    "discretized continuous variable": "Discretized Variable",
    "inappropriate use of line chart": "Line Chart Misuse",
    "inappropriate item order":        "Inappropriate Order",
    "inverted axis":                   "Inverted Axis",
    "inappropriate axis range":        "Axis Range Abuse",
}

MISLEADER_DESCRIPTIONS = {
    "misrepresentation":               "Bar or area sizes don't match labeled values",
    "3d":                              "3D effects distort visual comparison",
    "truncated axis":                  "Y-axis doesn't start at zero, exaggerating differences",
    "inappropriate use of pie chart":  "Pie chart used for non-part-to-whole data",
    "inconsistent tick intervals":     "Axis ticks are unevenly spaced",
    "dual axis":                       "Two y-axes with different scales mislead comparisons",
    "inconsistent binning size":       "Histogram bins have unequal widths",
    "discretized continuous variable": "Continuous data binned to hide distribution",
    "inappropriate use of line chart": "Line chart used for non-sequential data",
    "inappropriate item order":        "Ordering creates false impressions",
    "inverted axis":                   "Axis direction reversed, flipping perceived trend",
    "inappropriate axis range":        "Range set to exaggerate or minimize differences",
}

MODEL_DISPLAY = {
    "Claude | vision_only": ("Claude Haiku", "Vision Only"),
    "Claude | vision_text": ("Claude Haiku", "Vision + Context"),
    "Qwen   | vision_only": ("Qwen2.5-VL", "Vision Only"),
    "Qwen   | vision_text": ("Qwen2.5-VL", "Vision + Context"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_aggregated():
    p = RESULTS_DIR / "aggregated_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data
def load_sec_results():
    results = {}
    for label, fname in [
        ("Claude | vision_only", "sec_claude_vision_only.json"),
        ("Claude | vision_text", "sec_claude_vision_text.json"),
        ("Qwen | vision_only",   "sec_qwen_vision_only.json"),
        ("Qwen | vision_text",   "sec_qwen_vision_text.json"),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            results[label] = json.loads(p.read_text())
    return results


def img_to_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    if pil_img.mode in ("RGBA", "P", "CMYK"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def call_openrouter(api_key: str, b64_img: str, context: str = "") -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    taxonomy = "\n".join(f"- {k}: {v}" for k, v in MISLEADER_DESCRIPTIONS.items())
    if context:
        prompt = f"""You are an expert in data visualization. Detect misleading elements by comparing the chart against the provided data context.

## Misleader Taxonomy
{taxonomy}

## Ground-Truth / Context Data
{context}

## Output
Respond with valid JSON only:
{{
  "misleading": <true|false>,
  "misleader_types": [<zero or more types from the taxonomy>],
  "explanation": "<two to three sentences>"
}}"""
    else:
        prompt = f"""You are an expert in data visualization. Detect misleading elements in the chart image.

## Misleader Taxonomy
{taxonomy}

## Output
Respond with valid JSON only:
{{
  "misleading": <true|false>,
  "misleader_types": [<zero or more types from the taxonomy>],
  "explanation": "<two to three sentences>"
}}"""

    response = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
        ]}],
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        return {"misleading": None, "misleader_types": [], "explanation": raw, "parse_error": True}


# ── Page: Live Audit ──────────────────────────────────────────────────────────

def page_live_audit():
    st.markdown("## 🔍 Live Chart Audit")
    st.markdown(
        "Upload any financial chart and let Claude Haiku detect misleading visualization techniques in real time."
    )

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-...",
            help="Get a free key at openrouter.ai",
        )
        st.markdown("---")
        condition = st.radio(
            "Analysis Mode",
            ["Vision Only", "Vision + Data Context"],
            help="Vision Only: model sees only the image. Vision + Context: provide numeric data for cross-checking.",
        )
        context_text = ""
        if condition == "Vision + Data Context":
            context_text = st.text_area(
                "Paste the underlying data or table here",
                placeholder="e.g. Q1: $120M, Q2: $125M, Q3: $121M, Q4: $130M",
                height=120,
            )

    uploaded = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Uploaded chart", use_container_width=True)

        with col2:
            if not api_key:
                st.warning("Enter your OpenRouter API key in the sidebar to run live analysis.")
                st.info("**Example result (demo)**")
                _show_result({
                    "misleading": True,
                    "misleader_types": ["truncated axis", "dual axis"],
                    "explanation": "The y-axis starts at $800M instead of zero, visually exaggerating the revenue growth. Additionally, the dual y-axis uses different scales that make comparisons between the two metrics misleading.",
                })
            else:
                with st.spinner("Analyzing with Claude Haiku..."):
                    b64 = img_to_b64(img)
                    result = call_openrouter(api_key, b64, context_text if condition == "Vision + Data Context" else "")
                _show_result(result)
    else:
        st.markdown("---")
        st.markdown("#### Example: What the tool detects")
        _show_examples_grid()


def _show_result(result: dict):
    misleading = result.get("misleading")
    types = result.get("misleader_types", [])
    explanation = result.get("explanation", "")

    if misleading is True:
        st.error("⚠️ **Misleading Chart Detected**")
    elif misleading is False:
        st.success("✅ **Chart Appears Clean**")
    else:
        st.warning("⚠️ Could not parse model response")

    if types:
        st.markdown("**Detected Issues:**")
        for t in types:
            label = MISLEADER_LABELS.get(t, t)
            desc = MISLEADER_DESCRIPTIONS.get(t, "")
            st.markdown(f"- 🔴 **{label}** — {desc}")
    elif misleading is False:
        st.markdown("**No misleading patterns detected.**")

    st.markdown("**Model Explanation:**")
    st.info(explanation)


def _show_examples_grid():
    examples = [
        {
            "title": "Truncated Axis",
            "icon": "📊",
            "desc": "Y-axis starts at $800M instead of zero, making a 5% difference look like 50%.",
            "verdict": "MISLEADING",
        },
        {
            "title": "3D Distortion",
            "icon": "🎲",
            "desc": "3D pie chart makes the front slice appear larger due to perspective distortion.",
            "verdict": "MISLEADING",
        },
        {
            "title": "Dual Axis Abuse",
            "icon": "📈",
            "desc": "Two y-axes with different scales create a false impression of correlation.",
            "verdict": "MISLEADING",
        },
        {
            "title": "Clean Bar Chart",
            "icon": "✅",
            "desc": "Axis starts at zero, consistent ticks, appropriate chart type, clear labels.",
            "verdict": "CLEAN",
        },
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            color = "#ffeded" if ex["verdict"] == "MISLEADING" else "#edfff0"
            badge = "🔴 MISLEADING" if ex["verdict"] == "MISLEADING" else "🟢 CLEAN"
            st.markdown(
                f"""<div style="background:{color};padding:16px;border-radius:8px;margin-bottom:12px;">
                <h4>{ex['icon']} {ex['title']}</h4>
                <p style="font-size:0.9em">{ex['desc']}</p>
                <strong>{badge}</strong></div>""",
                unsafe_allow_html=True,
            )


# ── Page: Benchmark Results ───────────────────────────────────────────────────

def page_benchmark():
    st.markdown("## 📊 Benchmark Results — Misviz Dataset")
    st.markdown(
        "Evaluation of **Claude Haiku** and **Qwen2.5-VL** on the "
        "[Misviz-synth benchmark](https://github.com/UKPLab/arxiv2025-misviz) "
        "(271 stratified samples, 12 misleader types)."
    )

    data = load_aggregated()
    if not data:
        st.error("aggregated_summary.json not found in results/")
        return

    misviz = data.get("rq1_rq2_misviz", [])
    if not misviz:
        st.warning("No Misviz results found.")
        return

    # ── Overview metrics table ─────────────────────────────────────────────
    st.markdown("### Overall Performance")
    rows = []
    for r in misviz:
        model, condition = MODEL_DISPLAY.get(r["label"], (r["label"], ""))
        rows.append({
            "Model": model,
            "Condition": condition,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "TP": r["tp"],
            "FP": r["fp"],
            "FN": r["fn"],
            "TN": r["tn"],
        })
    df = pd.DataFrame(rows)

    def highlight_best(s):
        is_max = s == s.max()
        return ["background-color: #d4edda; font-weight:bold" if v else "" for v in is_max]

    styled = (
        df.style
        .apply(highlight_best, subset=["Accuracy", "Precision", "Recall", "F1"])
        .format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Bar chart comparison ───────────────────────────────────────────────
    st.markdown("### F1 Score Comparison")
    labels = [f"{r['Model']}\n{r['Condition']}" for r in rows]
    fig = go.Figure()
    metrics = ["Precision", "Recall", "F1"]
    colors  = ["#4C9BE8", "#F4845F", "#2ECC71"]
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=labels,
            y=df[metric],
            marker_color=color,
            text=[f"{v:.1%}" for v in df[metric]],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        height=380,
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-type F1 heatmap ────────────────────────────────────────────────
    st.markdown("### Per-Type F1 Score (Misleader Detection)")

    all_types = set()
    for r in misviz:
        all_types.update(r.get("per_misleader_type_f1", {}).keys())
    all_types = sorted(all_types)

    heat_data = []
    col_labels = []
    for r in misviz:
        model, cond = MODEL_DISPLAY.get(r["label"], (r["label"], ""))
        col_labels.append(f"{model}<br>{cond}")
        type_f1 = r.get("per_misleader_type_f1", {})
        heat_data.append([type_f1.get(t, 0) for t in all_types])

    type_labels = [MISLEADER_LABELS.get(t, t) for t in all_types]

    import numpy as np
    z = np.array(heat_data).T.tolist()

    fig2 = go.Figure(data=go.Heatmap(
        z=z,
        x=col_labels,
        y=type_labels,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="F1"),
    ))
    fig2.update_layout(height=480, margin=dict(t=20, b=20, l=200))
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Green = high F1 (model detects well). Red = low F1 (model struggles). "
        "Claude outperforms Qwen on most misleader types."
    )


# ── Page: SEC Cases ───────────────────────────────────────────────────────────

def page_sec():
    st.markdown("## 🏛️ SEC Filing Analysis (RQ3)")
    st.markdown(
        "Analysis of real SEC 10-K filings for **Non-GAAP prominence violations** "
        "and chart misleaders. Ground truth derived from SEC comment letters."
    )

    data = load_aggregated()
    if not data:
        st.error("Results not found.")
        return

    sec_results = data.get("rq3_sec", [])
    if not sec_results:
        st.warning("No SEC results found.")
        return

    # ── Overview table ─────────────────────────────────────────────────────
    st.markdown("### Overall Performance on SEC Filings")
    rows = []
    for r in sec_results:
        model, condition = MODEL_DISPLAY.get(r["label"], (r["label"], ""))
        rows.append({
            "Model": model,
            "Condition": condition,
            "Accuracy":  r.get("accuracy", 0),
            "Precision": r.get("precision", 0),
            "Recall":    r.get("recall", 0),
            "F1":        r.get("f1", 0),
        })
    df = pd.DataFrame(rows)

    def highlight_best(s):
        is_max = s == s.max()
        return ["background-color: #d4edda; font-weight:bold" if v else "" for v in is_max]

    styled = (
        df.style
        .apply(highlight_best, subset=["Accuracy", "Precision", "Recall", "F1"])
        .format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Per-ticker chart ───────────────────────────────────────────────────
    st.markdown("### Flag Rate by Company")
    st.caption("How often each model flagged filings per company (✓ = has known SEC violation)")

    best_result = sec_results[0] if sec_results else None
    if best_result and "per_ticker" in best_result:
        per_ticker = best_result["per_ticker"]
        tickers    = sorted(per_ticker.keys())
        flag_rates = [per_ticker[t]["flag_rate"] for t in tickers]
        has_gt     = [per_ticker[t]["has_gt_violation"] for t in tickers]

        colors = ["#E74C3C" if gt else "#3498DB" for gt in has_gt]
        labels = [f"{t} {'✓' if gt else ''}" for t, gt in zip(tickers, has_gt)]

        fig = go.Figure(go.Bar(
            x=labels,
            y=flag_rates,
            marker_color=colors,
            text=[f"{r:.0%}" for r in flag_rates],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(tickformat=".0%", title="Flag Rate", range=[0, 1.1]),
            height=350,
            margin=dict(t=20, b=20),
            showlegend=False,
        )
        fig.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper",
            text="🔴 Has GT violation  🔵 No violation",
            showarrow=False, font=dict(size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Example case ──────────────────────────────────────────────────────
    st.markdown("### Example: Real SEC Violation Case")
    sec_raw = load_sec_results()
    if sec_raw:
        model_key = list(sec_raw.keys())[0]
        result_data = sec_raw[model_key]
        ticker_results = result_data.get("results", {})

        gt_violators = [t for t, items in ticker_results.items()
                        if items and items[0].get("has_gt_violation")]

        if gt_violators:
            selected_ticker = st.selectbox("Select company", gt_violators)
            items = ticker_results[selected_ticker]

            flagged = [i for i in items if i.get("pred_misleading") or i.get("pred_violation")]
            clean   = [i for i in items if not i.get("pred_misleading") and not i.get("pred_violation")]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Filings Analyzed", len(items))
            col2.metric("Flagged as Suspicious", len(flagged), delta=f"{len(flagged)/len(items):.0%} flag rate")
            col3.metric("Ground Truth", "✓ Has Violation", delta_color="off")

            if flagged:
                st.markdown("**Sample flagged finding:**")
                ex = flagged[0]
                st.warning(f"📄 `{ex.get('file', 'unknown')}` — flagged as suspicious")
                st.info(ex.get("explanation", ""))


# ── Page: About ───────────────────────────────────────────────────────────────

def page_about():
    st.markdown("## ℹ️ About FinChartAudit")
    st.markdown("""
**FinChartAudit** is a research framework for evaluating how well vision-language models
detect misleading charts in SEC 10-K financial filings.

### Research Questions
| RQ | Question |
|----|---------|
| RQ1 | Can VLMs detect binary misleading vs. clean charts? |
| RQ2 | Can VLMs identify specific misleader types across 12 categories? |
| RQ3 | Can VLMs detect Non-GAAP prominence violations in real SEC filings? |

### Framework Architecture
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Models Evaluated**
- 🤖 Claude Haiku (Anthropic)
- 🤖 Qwen2.5-VL 8B (Alibaba)

**Conditions**
- Vision Only — model sees image only
- Vision + Text — model also sees ground-truth data table
""")
    with col2:
        st.markdown("""
**Datasets**
- 📊 Misviz-synth (81,814 synthetic charts)
  - 271 stratified samples used for eval
- 📄 SEC 10-K Filings
  - 13 companies, real filings
  - Ground truth from SEC comment letters

**Misleader Types: 12 categories**
""")

    cols = st.columns(3)
    for i, (key, label) in enumerate(MISLEADER_LABELS.items()):
        with cols[i % 3]:
            st.markdown(f"- **{label}**")

    st.markdown("---")
    st.markdown("""
### Team
CS 6180 Generative AI Capstone — Northeastern University, 2026

**GitHub:** [logisticPM/genai](https://github.com/logisticPM/genai)
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="FinChartAudit",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    padding: 24px 32px; border-radius: 12px; margin-bottom: 24px;">
            <h1 style="color: white; margin: 0; font-size: 2.2em;">📊 FinChartAudit</h1>
            <p style="color: #a0b4c8; margin: 4px 0 0 0; font-size: 1.1em;">
                Detecting Misleading Charts in SEC Financial Filings with Vision-Language Models
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "",
            ["🔍 Live Chart Audit", "📊 Benchmark Results", "🏛️ SEC Cases", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            "<small>CS 6180 Capstone · NEU 2026</small>",
            unsafe_allow_html=True,
        )

    if page == "🔍 Live Chart Audit":
        page_live_audit()
    elif page == "📊 Benchmark Results":
        page_benchmark()
    elif page == "🏛️ SEC Cases":
        page_sec()
    else:
        page_about()


if __name__ == "__main__":
    main()

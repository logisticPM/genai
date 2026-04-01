"""Misviz dataset configuration and paths."""
from pathlib import Path

# Base paths
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
MISVIZ_DIR = DATA_ROOT / "misviz"
MISVIZ_SYNTH_DIR = DATA_ROOT / "misviz_synth"

# JSON metadata files (from GitHub repo)
MISVIZ_JSON = MISVIZ_DIR / "misviz.json"
MISVIZ_SYNTH_JSON = MISVIZ_SYNTH_DIR / "misviz_synth.json"

# Image directories
MISVIZ_IMAGES_DIR = MISVIZ_DIR / "img"
MISVIZ_SYNTH_IMAGES_DIR = MISVIZ_SYNTH_DIR / "png"

# Data tables and axis metadata (from TUdatalib, Misviz-synth only)
MISVIZ_SYNTH_TABLES_DIR = MISVIZ_SYNTH_DIR / "data_tables"
MISVIZ_SYNTH_AXIS_DIR = MISVIZ_SYNTH_DIR / "axis_data"
MISVIZ_SYNTH_CODE_DIR = MISVIZ_SYNTH_DIR / "code"

# Evaluation output
EVAL_OUTPUT_DIR = DATA_ROOT / "eval_results"

# 12 Misviz misleader types (using exact labels from the dataset)
MISLEADER_TYPES = [
    "misrepresentation",
    "3d",
    "truncated axis",
    "inappropriate use of pie chart",
    "inconsistent binning size",
    "dual axis",
    "inconsistent tick intervals",
    "discretized continuous variable",
    "inappropriate use of line chart",
    "inappropriate item order",
    "inverted axis",
    "inappropriate axis range",
]

# HuggingFace dataset ID
HF_DATASET_ID = "UKPLab/misviz"

# TUdatalib URL (for synth data tables + axis metadata)
TUDATALIB_URL = "https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/5003"

# GitHub repo
GITHUB_REPO = "https://github.com/UKPLab/arxiv2025-misviz"

# Ensure directories
for d in [MISVIZ_DIR, MISVIZ_SYNTH_DIR, EVAL_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

"""
Misviz dataset loader — unified interface for both real-world and synthetic datasets.

Usage:
    from data_tools.misviz.loader import MisvizLoader

    # Load from local JSON files (after cloning GitHub repo)
    loader = MisvizLoader()
    synth_data = loader.load_synth()
    real_data = loader.load_real()

    # Load from HuggingFace (requires token)
    loader = MisvizLoader()
    hf_data = loader.load_from_huggingface()

    # Get a single instance with all its data
    instance = loader.get_synth_instance(0)
    print(instance["misleader"])       # ['truncated_axis']
    print(instance["data_table"])      # pandas DataFrame or dict
    print(instance["axis_metadata"])   # structured axis info
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import (
    MISVIZ_JSON, MISVIZ_SYNTH_JSON,
    MISVIZ_DIR, MISVIZ_IMAGES_DIR, MISVIZ_SYNTH_IMAGES_DIR,
    MISVIZ_SYNTH_TABLES_DIR, MISVIZ_SYNTH_AXIS_DIR,
    MISLEADER_TYPES,
)


@dataclass
class MisvizInstance:
    """A single chart instance from Misviz dataset."""
    instance_id: str
    image_path: str
    chart_type: list[str]
    misleader: list[str]
    split: str                          # train / val / test
    bbox: list = field(default_factory=list)
    is_misleading: bool = False

    # Synth-only fields
    data_table: dict | None = None      # Underlying data table
    axis_metadata: list[dict] | None = None  # Axis tick info
    code: str | None = None             # Matplotlib code

    # Derived
    image_url: str = ""
    source: str = ""                    # "synth" or "real"


class MisvizLoader:
    """Load Misviz datasets from local files or HuggingFace."""

    def __init__(self):
        self._synth_data: list[dict] | None = None
        self._real_data: list[dict] | None = None

    # ── Load from local JSON ──

    def load_synth(self) -> list[dict]:
        """Load Misviz-synth metadata from local JSON."""
        if self._synth_data is not None:
            return self._synth_data

        if not MISVIZ_SYNTH_JSON.exists():
            raise FileNotFoundError(
                f"Misviz-synth JSON not found at {MISVIZ_SYNTH_JSON}\n"
                f"Clone the repo: git clone https://github.com/UKPLab/arxiv2025-misviz\n"
                f"Then copy data/misviz_synth/misviz_synth.json to {MISVIZ_SYNTH_JSON}"
            )

        self._synth_data = json.loads(MISVIZ_SYNTH_JSON.read_text(encoding="utf-8"))
        print(f"Loaded Misviz-synth: {len(self._synth_data)} instances")
        return self._synth_data

    def load_real(self) -> list[dict]:
        """Load Misviz (real-world) metadata from local JSON."""
        if self._real_data is not None:
            return self._real_data

        if not MISVIZ_JSON.exists():
            raise FileNotFoundError(
                f"Misviz JSON not found at {MISVIZ_JSON}\n"
                f"Clone the repo: git clone https://github.com/UKPLab/arxiv2025-misviz\n"
                f"Then copy data/misviz/misviz.json to {MISVIZ_JSON}"
            )

        self._real_data = json.loads(MISVIZ_JSON.read_text(encoding="utf-8"))
        print(f"Loaded Misviz real: {len(self._real_data)} instances")
        return self._real_data

    # ── Load from HuggingFace ──

    def load_from_huggingface(self, split: str = "test") -> list[dict]:
        """Load from HuggingFace datasets (requires login + token)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        ds = load_dataset("UKPLab/misviz", split=split)
        return [dict(row) for row in ds]

    # ── Instance Access ──

    def get_synth_instance(self, idx: int) -> MisvizInstance:
        """Get a single synth instance with all associated data."""
        data = self.load_synth()
        raw = data[idx]

        instance = MisvizInstance(
            instance_id=str(idx),
            image_path=str(MISVIZ_SYNTH_IMAGES_DIR.parent / raw.get("image_path", "")),
            chart_type=raw.get("chart_type", []),
            misleader=raw.get("misleader", []),
            split=raw.get("split", ""),
            bbox=raw.get("bbox", []),
            is_misleading=len(raw.get("misleader", [])) > 0,
            source="synth",
        )

        # Load data table from explicit path in JSON
        table_path_str = raw.get("table_data_path", "")
        if table_path_str:
            table_path = MISVIZ_SYNTH_IMAGES_DIR.parent / table_path_str
            if table_path.exists():
                instance.data_table = self._load_data_table(table_path)

        # Load axis metadata from explicit path in JSON
        axis_path_str = raw.get("axis_data_path", "")
        if axis_path_str:
            axis_path = MISVIZ_SYNTH_IMAGES_DIR.parent / axis_path_str
            if axis_path.exists():
                instance.axis_metadata = self._load_axis_metadata(axis_path)

        # Load code from explicit path in JSON
        code_path_str = raw.get("code_path", "")
        if code_path_str:
            code_path = MISVIZ_SYNTH_IMAGES_DIR.parent / code_path_str
            if code_path.exists():
                instance.code = code_path.read_text(encoding="utf-8")

        return instance

    def get_real_instance(self, idx: int) -> MisvizInstance:
        """Get a single real-world instance."""
        data = self.load_real()
        raw = data[idx]

        return MisvizInstance(
            instance_id=str(idx),
            image_path=str(MISVIZ_DIR / raw.get("image_path", "")),
            chart_type=raw.get("chart_type", []),
            misleader=raw.get("misleader", []),
            split=raw.get("split", ""),
            bbox=raw.get("bbox", []),
            is_misleading=len(raw.get("misleader", [])) > 0,
            image_url=raw.get("image_url", ""),
            source="real",
        )

    # ── Iteration ──

    def iter_synth(self, split: str | None = None) -> Iterator[MisvizInstance]:
        """Iterate over synth instances, optionally filtered by split."""
        data = self.load_synth()
        for idx, raw in enumerate(data):
            if split and raw.get("split") != split:
                continue
            yield self.get_synth_instance(idx)

    def iter_real(self, split: str | None = None) -> Iterator[MisvizInstance]:
        """Iterate over real instances, optionally filtered by split."""
        data = self.load_real()
        for idx, raw in enumerate(data):
            if split and raw.get("split") != split:
                continue
            yield self.get_real_instance(idx)

    # ── Filtering ──

    def filter_by_misleader(self, data: list[dict], misleader_type: str) -> list[dict]:
        """Filter instances by misleader type."""
        return [d for d in data if misleader_type in d.get("misleader", [])]

    def filter_by_chart_type(self, data: list[dict], chart_type: str) -> list[dict]:
        """Filter instances by chart type."""
        return [d for d in data if chart_type in d.get("chart_type", [])]

    def filter_misleading_only(self, data: list[dict]) -> list[dict]:
        """Return only misleading instances."""
        return [d for d in data if len(d.get("misleader", [])) > 0]

    def filter_clean_only(self, data: list[dict]) -> list[dict]:
        """Return only non-misleading instances."""
        return [d for d in data if len(d.get("misleader", [])) == 0]

    # ── Statistics ──

    def get_stats(self, data: list[dict]) -> dict:
        """Get dataset statistics."""
        total = len(data)
        misleading = sum(1 for d in data if len(d.get("misleader", [])) > 0)

        # Per misleader type
        type_counts = {}
        for mtype in MISLEADER_TYPES:
            type_counts[mtype] = sum(
                1 for d in data if mtype in d.get("misleader", [])
            )

        # Per chart type
        chart_counts = {}
        for d in data:
            for ct in d.get("chart_type", []):
                chart_counts[ct] = chart_counts.get(ct, 0) + 1

        # Per split
        split_counts = {}
        for d in data:
            s = d.get("split", "unknown")
            split_counts[s] = split_counts.get(s, 0) + 1

        return {
            "total": total,
            "misleading": misleading,
            "clean": total - misleading,
            "misleading_ratio": misleading / total if total > 0 else 0,
            "per_misleader_type": type_counts,
            "per_chart_type": chart_counts,
            "per_split": split_counts,
        }

    # ── Private helpers ──

    def _find_data_table(self, raw: dict) -> Path | None:
        """Find the data table file for a synth instance."""
        image_path = raw.get("image_path", "")
        if not image_path:
            return None
        # Convention: tables stored alongside images with .csv or .json extension
        stem = Path(image_path).stem
        for ext in [".csv", ".json", ".tsv"]:
            candidate = MISVIZ_SYNTH_TABLES_DIR / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _find_axis_metadata(self, raw: dict) -> Path | None:
        """Find the axis metadata file for a synth instance."""
        image_path = raw.get("image_path", "")
        if not image_path:
            return None
        stem = Path(image_path).stem
        for ext in [".json", ".csv"]:
            candidate = MISVIZ_SYNTH_AXIS_DIR / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _find_code(self, raw: dict) -> Path | None:
        """Find the Matplotlib code file for a synth instance."""
        image_path = raw.get("image_path", "")
        if not image_path:
            return None
        stem = Path(image_path).stem
        from .config import MISVIZ_SYNTH_CODE_DIR
        candidate = MISVIZ_SYNTH_CODE_DIR / f"{stem}.py"
        return candidate if candidate.exists() else None

    def _load_data_table(self, path: Path) -> dict:
        """Load a data table file."""
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".csv":
            # Return as list of rows
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            if not lines:
                return {}
            headers = lines[0].split(",")
            rows = [dict(zip(headers, line.split(","))) for line in lines[1:]]
            return {"headers": headers, "rows": rows}
        return {}

    def _load_axis_metadata(self, path: Path) -> list[dict] | dict:
        """Load axis metadata. Format: {"axis": [...], "label": [...], ...}"""
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            # Convert parallel arrays to list of dicts if needed
            if isinstance(data, dict) and "axis" in data and "label" in data:
                entries = []
                axes = data["axis"]
                labels = data["label"]
                positions = data.get("relative_position", [None] * len(axes))
                for i in range(len(axes)):
                    entries.append({
                        "axis": axes[i],
                        "label": labels[i] if i < len(labels) else "",
                        "relative_position": positions[i] if i < len(positions) else None,
                    })
                return entries
            return data if isinstance(data, list) else [data]
        elif path.suffix == ".csv":
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            if len(lines) < 2:
                return []
            headers = [h.strip() for h in lines[0].split(",")]
            return [
                dict(zip(headers, [v.strip() for v in line.split(",")]))
                for line in lines[1:]
            ]
        return []

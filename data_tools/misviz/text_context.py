"""
Build textual context from Misviz-synth data tables for the vision+text experiment condition.

This module converts Misviz-synth's underlying data tables into natural language
text that serves as "ground truth context" in the 2x2 experiment:
  - Vision-only: chart image + misleader type definitions
  - Vision+text: chart image + misleader type definitions + data table as text

Usage:
    from data_tools.misviz.text_context import TextContextBuilder

    builder = TextContextBuilder()
    context = builder.build_context(data_table, axis_metadata)
    # Returns: "The chart is based on the following data:\n
    #           Year | Value\n 2020 | 45.2\n 2021 | 48.1\n..."
"""
import json
from pathlib import Path


class TextContextBuilder:
    """Build textual ground-truth context from Misviz-synth data."""

    def build_context(self, data_table: dict | None,
                      axis_metadata: list[dict] | None) -> str:
        """
        Build a complete textual context string from data table and axis metadata.
        Returns empty string if no data available.
        """
        parts = []

        if data_table:
            parts.append(self._format_data_table(data_table))

        if axis_metadata:
            parts.append(self._format_axis_metadata(axis_metadata))

        return "\n\n".join(parts)

    def build_context_from_instance(self, instance) -> str:
        """Build context from a MisvizInstance object."""
        return self.build_context(instance.data_table, instance.axis_metadata)

    def _format_data_table(self, data_table: dict) -> str:
        """Format data table as readable text."""
        if not data_table:
            return ""

        lines = ["The chart is based on the following data:"]

        # Handle dict with "headers" and "rows" (CSV-loaded format)
        if "headers" in data_table and "rows" in data_table:
            headers = data_table["headers"]
            lines.append(" | ".join(headers))
            lines.append("-" * (len(" | ".join(headers))))
            for row in data_table["rows"]:
                if isinstance(row, dict):
                    lines.append(" | ".join(str(row.get(h, "")) for h in headers))
                elif isinstance(row, list):
                    lines.append(" | ".join(str(v) for v in row))

        # Handle list of dicts
        elif isinstance(data_table, list) and data_table:
            if isinstance(data_table[0], dict):
                headers = list(data_table[0].keys())
                lines.append(" | ".join(headers))
                lines.append("-" * (len(" | ".join(headers))))
                for row in data_table:
                    lines.append(" | ".join(str(row.get(h, "")) for h in headers))

        # Handle raw dict (key-value pairs)
        elif isinstance(data_table, dict):
            for key, value in data_table.items():
                if key in ("headers", "rows"):
                    continue
                if isinstance(value, list):
                    lines.append(f"{key}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def _format_axis_metadata(self, axis_metadata) -> str:
        """Format axis metadata as readable text."""
        if not axis_metadata:
            return ""

        # Handle both list[dict] and raw dict formats
        entries = axis_metadata if isinstance(axis_metadata, list) else [axis_metadata]

        lines = ["Axis information:"]

        # Group by axis name
        axes = {}
        for entry in entries:
            if isinstance(entry, dict):
                axis_name = entry.get("Axis", entry.get("axis", "unknown"))
                if axis_name not in axes:
                    axes[axis_name] = []
                axes[axis_name].append(entry)

        for axis_name, axis_entries in axes.items():
            labels = [e.get("Label", e.get("label", "")) for e in axis_entries]
            lines.append(f"  {axis_name}-axis labels: {', '.join(str(l) for l in labels)}")

            # Check for numeric axis range
            try:
                numeric_labels = []
                for l in labels:
                    try:
                        numeric_labels.append(float(l))
                    except (ValueError, TypeError):
                        pass
                if len(numeric_labels) >= 2:
                    lines.append(f"  {axis_name}-axis range: {min(numeric_labels)} to {max(numeric_labels)}")
                    if min(numeric_labels) > 0 and axis_name.lower() in ("y", "y1"):
                        lines.append(f"  Note: {axis_name}-axis does not start from 0 (starts at {min(numeric_labels)})")
            except (ValueError, TypeError):
                pass

        return "\n".join(lines)


def build_experiment_prompt(chart_type: list[str], context: str = "",
                             include_definitions: bool = True) -> str:
    """
    Build the full prompt for the 2x2 experiment.

    Args:
        chart_type: Chart types detected in the image
        context: Textual ground-truth context (empty for vision-only condition)
        include_definitions: Whether to include misleader type definitions
    """
    parts = []

    parts.append(
        "Analyze this chart image for misleading visualization techniques. "
        "For each of the following misleader types, determine if it is present (yes/no) "
        "and provide a brief explanation."
    )

    if include_definitions:
        parts.append("\nMisleader types to check:")
        definitions = {
            "truncated_axis": "Y-axis does not start from 0, exaggerating differences between values",
            "inverted_axis": "Axis values are ordered in reverse (e.g., decreasing from bottom to top)",
            "misrepresentation": "Visual encoding (bar height, area, angle) does not match the actual data values",
            "3d_effects": "3D rendering distorts the visual perception of data values",
            "dual_axis": "Two different y-axes with different scales that can suggest misleading correlations",
            "inappropriate_pie_chart_use": "Pie chart used where it is not suitable (e.g., values don't sum to 100%)",
            "inappropriate_line_chart_use": "Line chart used for categorical or non-sequential data",
            "inconsistent_binning_size": "Histogram bins have unequal widths, distorting distribution perception",
            "inconsistent_tick_intervals": "Tick marks on axis are not evenly spaced",
            "discretized_continuous_variable": "Continuous data forced into discrete categories, losing information",
            "inappropriate_item_order": "Items ordered in a way that suggests a trend where none exists",
            "inappropriate_axis_range": "Axis range chosen to exaggerate or minimize variation",
        }
        for mtype, definition in definitions.items():
            parts.append(f"  - {mtype}: {definition}")

    if context:
        parts.append(f"\n--- Ground Truth Data ---\n{context}")

    parts.append(
        "\nFor each misleader type, respond in this exact JSON format:\n"
        "{\n"
        '  "truncated_axis": {"present": true/false, "confidence": 0.0-1.0, "explanation": "..."},\n'
        '  "inverted_axis": {"present": true/false, "confidence": 0.0-1.0, "explanation": "..."},\n'
        "  ...\n"
        "}"
    )

    return "\n".join(parts)

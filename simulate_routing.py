"""Phase 1: Offline simulation of routing strategies on existing LLM-OCR+Rules results.

Uses the 271-chart raw_results.json to simulate:
  Strategy A (baseline): Current LLM-OCR+Rules as-is
  Strategy B: Rule veto for structural types
  Strategy C: Rule veto + remove high-hallucination types without rule support
  Strategy D: Strategy C + inverted_axis rule FP fix

No API calls needed — pure post-processing on existing predictions.
"""
import json
from collections import defaultdict
from pathlib import Path

RAW_RESULTS = Path("data/eval_results/llm_ocr_rules/raw_results.json")

# ---------- Type classification ----------

# Structural types: OCR+Rules can verify, rules have veto power
TOOL_GROUNDED = {
    "truncated_axis",
    "inappropriate_axis_range",
    "inverted_axis",
    "dual_axis",
    "inconsistent_tick_intervals",
    "discretized_continuous_variable",
    "inconsistent_binning_size",
}

# Visual types: Rules cannot verify, pure VLM judgment
VISION_ONLY = {
    "misrepresentation",
    "3d",
    "inappropriate_use_of_pie_chart",
    "inappropriate_use_of_line_chart",
    "inappropriate_item_order",
}

# Mapping from predicted name (underscored) to ground truth name (spaced)
PRED_TO_GT = {
    "truncated_axis": "truncated axis",
    "inappropriate_axis_range": "inappropriate axis range",
    "inverted_axis": "inverted axis",
    "dual_axis": "dual axis",
    "inconsistent_tick_intervals": "inconsistent tick intervals",
    "discretized_continuous_variable": "discretized continuous variable",
    "inconsistent_binning_size": "inconsistent binning size",
    "misrepresentation": "misrepresentation",
    "3d": "3d",
    "inappropriate_use_of_pie_chart": "inappropriate use of pie chart",
    "inappropriate_use_of_line_chart": "inappropriate use of line chart",
    "inappropriate_item_order": "inappropriate item order",
}

GT_TO_PRED = {v: k for k, v in PRED_TO_GT.items()}


def normalize_pred(name: str) -> str:
    """Normalize prediction name: 'truncated axis' and 'truncated_axis' → 'truncated_axis'."""
    return name.strip().replace(" ", "_")


def denormalize_pred(name: str) -> str:
    """Convert normalized name back to ground truth format: 'truncated_axis' → 'truncated axis'."""
    return PRED_TO_GT.get(name, name.replace("_", " "))


# Rules that are ONLY present in rule_results when they flag positive.
# If OCR exists but these rules are absent → means "clean" (implicit veto).
IMPLICIT_CLEAN_RULES = {
    "inverted_axis",
    "inappropriate_axis_range",
    "dual_axis",
    "inconsistent_binning_size",
}

# Rules that always appear in rule_results when OCR values exist (flagged or clean).
# Must parse the text to determine verdict.
EXPLICIT_RULES = {
    "truncated_axis",
    "inconsistent_tick_intervals",  # via broken_scale
}


def parse_rule_flags(rule_results: list[str]) -> dict[str, bool]:
    """Parse rule_results strings to determine which rules flagged positive.

    Returns dict: rule_name -> True if rule says flagged, False if rule says clean.
    Rules only appear in results if they were run.
    """
    flags = {}
    for r in rule_results:
        r_lower = r.lower()
        # truncated_axis
        if r.startswith("truncated_axis:"):
            flags["truncated_axis"] = "instead of 0" in r_lower or "exaggerated" in r_lower
        # broken_scale → maps to inconsistent_tick_intervals
        elif r.startswith("broken_scale:"):
            flags["inconsistent_tick_intervals"] = "inconsistent" in r_lower and "consistent." not in r_lower
        # inverted_axis — only present when is_inverted=True in code
        elif r.startswith("inverted_axis:"):
            flags["inverted_axis"] = "inverted" in r_lower and "normal" not in r_lower
        # inappropriate_axis_range — only present when is_inappropriate=True
        elif r.startswith("inappropriate_axis_range:"):
            flags["inappropriate_axis_range"] = True
        # dual_axis — only present when has_dual_axis=True
        elif r.startswith("dual_axis:"):
            flags["dual_axis"] = True
        # inconsistent_binning — only present when is_inconsistent=True
        elif r.startswith("inconsistent_binning:"):
            flags["inconsistent_binning_size"] = True
    return flags


def compute_metrics(results: list[dict], predicted_key: str = "predicted") -> dict:
    """Compute binary + per-type metrics."""
    tp = fp = tn = fn = 0
    type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

    all_types = set(PRED_TO_GT.values())

    for r in results:
        gt_set = set(r["ground_truth"])
        pred_set = {denormalize_pred(normalize_pred(p)) for p in r[predicted_key]}

        # Binary: is the chart misleading at all?
        gt_positive = len(gt_set) > 0
        pred_positive = len(pred_set) > 0

        if gt_positive and pred_positive:
            tp += 1
        elif not gt_positive and pred_positive:
            fp += 1
        elif not gt_positive and not pred_positive:
            tn += 1
        else:
            fn += 1

        # Per-type
        for t in all_types:
            t_gt = t in gt_set
            t_pred = t in pred_set
            if t_gt and t_pred:
                type_stats[t]["tp"] += 1
            elif not t_gt and t_pred:
                type_stats[t]["fp"] += 1
            elif not t_gt and not t_pred:
                type_stats[t]["tn"] += 1
            else:
                type_stats[t]["fn"] += 1

    n = len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    per_type = {}
    for t in sorted(all_types):
        s = type_stats[t]
        t_prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        t_rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0
        per_type[t] = {
            "precision": round(t_prec, 4),
            "recall": round(t_rec, 4),
            "f1": round(t_f1, 4),
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
        }

    return {
        "accuracy": round((tp + tn) / n, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "per_type": per_type,
    }


def strategy_baseline(results: list[dict]) -> list[dict]:
    """Strategy A: no changes (current LLM-OCR+Rules)."""
    return results


def _apply_veto(pred_name_norm: str, rule_flags: dict, has_ocr: bool,
                 strict_types: set | None = None) -> bool:
    """Decide whether to KEEP a prediction (True) or VETO it (False).

    Args:
        pred_name_norm: Normalized prediction name (underscored)
        rule_flags: Parsed rule verdicts {rule_name: bool}
        has_ocr: Whether OCR axis values were extracted
        strict_types: If set, these types require explicit rule confirmation
                      (vetoed even when rule didn't run, as long as OCR exists)
    """
    if strict_types is None:
        strict_types = set()

    if pred_name_norm not in TOOL_GROUNDED:
        return True  # Visual type → always keep

    if not has_ocr:
        # No OCR → can't verify → keep unless strict
        return pred_name_norm not in strict_types

    # Has OCR. Check rule verdict.
    if pred_name_norm in rule_flags:
        # Rule explicitly ran → trust its verdict
        return rule_flags[pred_name_norm]

    # Rule absent. For implicit-clean rules, absence = clean → veto.
    if pred_name_norm in IMPLICIT_CLEAN_RULES:
        return False  # Rule would have appeared if flagged → veto

    # For explicit rules (truncated_axis, broken_scale), absence means rule didn't run
    # (e.g., OCR got non-numeric values). In strict mode, veto anyway.
    if pred_name_norm in strict_types:
        return False

    return True  # Default: keep


def strategy_rule_veto(results: list[dict]) -> list[dict]:
    """Strategy B: Rule veto for TOOL_GROUNDED types.

    If VLM predicts a structural misleader but rules say clean → remove prediction.
    For implicit-clean rules (inverted_axis, inappropriate_axis_range, etc.),
    absence of rule result when OCR exists = clean → veto.
    """
    out = []
    for r in results:
        rule_flags = parse_rule_flags(r["rule_results"])
        has_ocr = len(r.get("ocr_y_values", [])) > 0

        new_pred = []
        for p in r["predicted"]:
            p_norm = normalize_pred(p)
            if _apply_veto(p_norm, rule_flags, has_ocr):
                new_pred.append(p)

        out.append({**r, "predicted_routed": new_pred})
    return out


def strategy_rule_veto_strict(results: list[dict]) -> list[dict]:
    """Strategy C: Rule veto + strict mode for high-hallucination types.

    truncated_axis (96% hallucination) and inappropriate_axis_range (90% hallucination):
    require explicit rule confirmation. If rule didn't run → veto anyway.
    """
    STRICT = {"truncated_axis", "inappropriate_axis_range"}
    out = []
    for r in results:
        rule_flags = parse_rule_flags(r["rule_results"])
        has_ocr = len(r.get("ocr_y_values", [])) > 0

        new_pred = []
        for p in r["predicted"]:
            p_norm = normalize_pred(p)
            if _apply_veto(p_norm, rule_flags, has_ocr, strict_types=STRICT):
                new_pred.append(p)

        out.append({**r, "predicted_routed": new_pred})
    return out


def strategy_full_routing(results: list[dict]) -> list[dict]:
    """Strategy D: Strategy C + fix inverted_axis rule FP.

    inverted_axis rule has high FP because OCR reads Y-axis top→bottom as increasing.
    Heuristic: if values are evenly spaced and increasing, it's normal OCR order, not inverted.
    """
    STRICT = {"truncated_axis", "inappropriate_axis_range"}
    out = []

    for r in results:
        rule_flags = parse_rule_flags(r["rule_results"])
        has_ocr = len(r.get("ocr_y_values", [])) > 0
        ocr_y = r.get("ocr_y_values", [])

        # Fix inverted_axis rule FP
        if rule_flags.get("inverted_axis", False):
            numeric_vals = []
            for v in ocr_y:
                try:
                    numeric_vals.append(float(v))
                except (ValueError, TypeError):
                    pass

            if len(numeric_vals) >= 3:
                diffs = [numeric_vals[i+1] - numeric_vals[i] for i in range(len(numeric_vals)-1)]
                if diffs:
                    avg_diff = sum(diffs) / len(diffs)
                    if avg_diff > 0:
                        max_dev = max(abs(d - avg_diff) for d in diffs)
                        if max_dev < avg_diff * 0.5:
                            rule_flags["inverted_axis"] = False

        new_pred = []
        for p in r["predicted"]:
            p_norm = normalize_pred(p)
            if _apply_veto(p_norm, rule_flags, has_ocr, strict_types=STRICT):
                new_pred.append(p)

        out.append({**r, "predicted_routed": new_pred})
    return out


def strategy_aggressive(results: list[dict]) -> list[dict]:
    """Strategy E: Strategy D + strict mode for ALL tool-grounded types.

    Every structural prediction must have rule confirmation to survive.
    """
    STRICT = TOOL_GROUNDED  # All structural types require rule confirmation
    out = []

    for r in results:
        rule_flags = parse_rule_flags(r["rule_results"])
        has_ocr = len(r.get("ocr_y_values", [])) > 0
        ocr_y = r.get("ocr_y_values", [])

        # Fix inverted_axis rule FP
        if rule_flags.get("inverted_axis", False):
            numeric_vals = []
            for v in ocr_y:
                try:
                    numeric_vals.append(float(v))
                except (ValueError, TypeError):
                    pass
            if len(numeric_vals) >= 3:
                diffs = [numeric_vals[i+1] - numeric_vals[i] for i in range(len(numeric_vals)-1)]
                if diffs:
                    avg_diff = sum(diffs) / len(diffs)
                    if avg_diff > 0:
                        max_dev = max(abs(d - avg_diff) for d in diffs)
                        if max_dev < avg_diff * 0.5:
                            rule_flags["inverted_axis"] = False

        new_pred = []
        for p in r["predicted"]:
            p_norm = normalize_pred(p)
            if _apply_veto(p_norm, rule_flags, has_ocr, strict_types=STRICT):
                new_pred.append(p)

        out.append({**r, "predicted_routed": new_pred})
    return out


def strategy_optimal(results: list[dict]) -> list[dict]:
    """Strategy F: Per-type optimal routing based on simulation results.

    Per-type decisions:
      truncated_axis:        STRICT — require rule confirm (F1: 35.7%→69.0%)
      inappropriate_axis_range: KEEP VLM — rule is broken (0 TP when strict)
      inverted_axis:         KEEP VLM — rule has high FP itself
      dual_axis:             VETO if rule absent + OCR exists (F1: 75→77.8%)
      inconsistent_tick_intervals: KEEP VLM — rule coverage too low
      inconsistent_binning_size:   KEEP VLM — rule fires incorrectly
      discretized_continuous_variable: KEEP VLM — no rule available
      Visual types:          KEEP VLM always
    """
    # Only these types get rule-based filtering
    STRICT_TYPES = {"truncated_axis"}  # Must have rule confirmation
    VETO_TYPES = {"dual_axis"}         # Veto if implicit-clean

    out = []
    for r in results:
        rule_flags = parse_rule_flags(r["rule_results"])
        has_ocr = len(r.get("ocr_y_values", [])) > 0

        new_pred = []
        for p in r["predicted"]:
            p_norm = normalize_pred(p)

            if p_norm in STRICT_TYPES:
                # Only keep if rule explicitly confirms
                if rule_flags.get(p_norm, False):
                    new_pred.append(p)
            elif p_norm in VETO_TYPES and has_ocr:
                # Veto if implicit-clean rule says no
                if p_norm in rule_flags:
                    if rule_flags[p_norm]:
                        new_pred.append(p)
                elif p_norm in IMPLICIT_CLEAN_RULES:
                    pass  # Absent = clean → veto
                else:
                    new_pred.append(p)
            else:
                # Keep VLM prediction as-is
                new_pred.append(p)

        out.append({**r, "predicted_routed": new_pred})
    return out


def print_comparison(name: str, metrics: dict, baseline: dict):
    """Print metrics with delta from baseline."""
    delta_f1 = metrics["f1"] - baseline["f1"]
    delta_prec = metrics["precision"] - baseline["precision"]
    delta_rec = metrics["recall"] - baseline["recall"]
    delta_acc = metrics["accuracy"] - baseline["accuracy"]

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Binary:  Acc={metrics['accuracy']:.1%} ({delta_acc:+.1%})  "
          f"Prec={metrics['precision']:.1%} ({delta_prec:+.1%})  "
          f"Rec={metrics['recall']:.1%} ({delta_rec:+.1%})  "
          f"F1={metrics['f1']:.1%} ({delta_f1:+.1%})")
    print(f"  Counts:  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")

    # Per-type comparison
    print(f"\n  {'Type':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'dF1':>7} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*4} {'-'*4} {'-'*4}")

    for t in sorted(metrics["per_type"].keys()):
        m = metrics["per_type"][t]
        b = baseline["per_type"].get(t, {"f1": 0})
        delta = m["f1"] - b["f1"]
        marker = " +" if delta > 0.01 else (" -" if delta < -0.01 else "  ")
        print(f"  {t:<35} {m['precision']:>5.1%} {m['recall']:>5.1%} "
              f"{m['f1']:>5.1%} {delta:>+6.1%}{marker} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")


def main():
    data = json.loads(RAW_RESULTS.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} instances from {RAW_RESULTS}")

    # Strategy A: Baseline
    baseline_metrics = compute_metrics(data, "predicted")
    print_comparison("Strategy A: Baseline (LLM-OCR+Rules as-is)", baseline_metrics, baseline_metrics)

    # Strategy B: Rule veto
    routed_b = strategy_rule_veto(data)
    metrics_b = compute_metrics(routed_b, "predicted_routed")
    print_comparison("Strategy B: Rule Veto (implicit clean)", metrics_b, baseline_metrics)

    # Strategy C: Rule veto + strict high-hallucination
    routed_c = strategy_rule_veto_strict(data)
    metrics_c = compute_metrics(routed_c, "predicted_routed")
    print_comparison("Strategy C: B + Strict truncated/range", metrics_c, baseline_metrics)

    # Strategy D: Full routing
    routed_d = strategy_full_routing(data)
    metrics_d = compute_metrics(routed_d, "predicted_routed")
    print_comparison("Strategy D: C + inverted_axis FP fix", metrics_d, baseline_metrics)

    # Strategy E: Aggressive
    routed_e = strategy_aggressive(data)
    metrics_e = compute_metrics(routed_e, "predicted_routed")
    print_comparison("Strategy E: All structural require rule confirm", metrics_e, baseline_metrics)

    # Strategy F: Cherry-pick best per-type settings
    routed_f = strategy_optimal(data)
    metrics_f = compute_metrics(routed_f, "predicted_routed")
    print_comparison("Strategy F: Optimal per-type routing", metrics_f, baseline_metrics)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Strategy':<50} {'F1':>6} {'Prec':>6} {'Rec':>6} {'FP':>4}")
    print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6} {'-'*4}")
    for name, m in [
        ("A: Baseline", baseline_metrics),
        ("B: Rule Veto (implicit clean)", metrics_b),
        ("C: B + Strict truncated/range", metrics_c),
        ("D: C + Inverted Axis FP fix", metrics_d),
        ("E: All structural require confirm", metrics_e),
        ("F: Optimal per-type routing", metrics_f),
    ]:
        print(f"  {name:<50} {m['f1']:>5.1%} {m['precision']:>5.1%} {m['recall']:>5.1%} {m['fp']:>4}")

    # Save detailed results
    output = {
        "strategies": {
            "A_baseline": baseline_metrics,
            "B_rule_veto": metrics_b,
            "C_hallucination_suppression": metrics_c,
            "D_full_routing": metrics_d,
        }
    }
    out_path = Path("data/eval_results/routing_simulation.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

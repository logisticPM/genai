"""
Evaluation metrics for Misviz benchmark experiments.

Computes:
- Per-misleader-type accuracy, precision, recall, F1
- Exact match (EM): predicted set == ground truth set
- Partial match (PM): predicted set is subset of ground truth
- Binary classification: misleading vs. not misleading
- Confusion matrix

Usage:
    from data_tools.misviz.evaluator import MisvizEvaluator

    evaluator = MisvizEvaluator()
    evaluator.add_prediction(
        instance_id="0",
        ground_truth=["truncated_axis"],
        predicted=["truncated_axis", "misrepresentation"],
        confidences={"truncated_axis": 0.9, "misrepresentation": 0.6}
    )
    results = evaluator.compute_metrics()
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import MISLEADER_TYPES, EVAL_OUTPUT_DIR


@dataclass
class Prediction:
    instance_id: str
    ground_truth: list[str]
    predicted: list[str]
    confidences: dict[str, float] = field(default_factory=dict)
    condition: str = ""     # "vision_only" or "vision_text"
    model: str = ""         # "claude" or "qwen"


class MisvizEvaluator:
    """Compute evaluation metrics for Misviz experiments."""

    def __init__(self):
        self.predictions: list[Prediction] = []

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize misleader type names (underscore vs space, case)."""
        return name.strip().lower().replace("_", " ")

    def add_prediction(self, instance_id: str, ground_truth: list[str],
                        predicted: list[str],
                        confidences: dict[str, float] | None = None,
                        condition: str = "", model: str = ""):
        self.predictions.append(Prediction(
            instance_id=instance_id,
            ground_truth=[self._normalize(g) for g in ground_truth],
            predicted=[self._normalize(p) for p in predicted],
            confidences={self._normalize(k): v for k, v in (confidences or {}).items()},
            condition=condition,
            model=model,
        ))

    def compute_metrics(self) -> dict:
        """Compute all metrics."""
        if not self.predictions:
            return {}

        results = {
            "total_instances": len(self.predictions),
            "binary_classification": self._binary_metrics(),
            "exact_match": self._exact_match(),
            "partial_match": self._partial_match(),
            "per_misleader_type": self._per_type_metrics(),
            "confusion_summary": self._confusion_summary(),
        }

        return results

    def compute_2x2_comparison(self) -> dict:
        """Compute metrics grouped by the 2x2 experimental conditions."""
        groups = defaultdict(list)
        for pred in self.predictions:
            key = f"{pred.model}_{pred.condition}"
            groups[key].append(pred)

        results = {}
        for key, preds in groups.items():
            sub_eval = MisvizEvaluator()
            sub_eval.predictions = preds
            results[key] = sub_eval.compute_metrics()

        return results

    # ── Binary Classification (misleading vs. clean) ──

    def _binary_metrics(self) -> dict:
        tp = fp = tn = fn = 0
        for pred in self.predictions:
            gt_positive = len(pred.ground_truth) > 0
            pred_positive = len(pred.predicted) > 0

            if gt_positive and pred_positive:
                tp += 1
            elif not gt_positive and pred_positive:
                fp += 1
            elif not gt_positive and not pred_positive:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(self.predictions) if self.predictions else 0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    # ── Exact Match ──

    def _exact_match(self) -> dict:
        matches = sum(
            1 for pred in self.predictions
            if set(pred.ground_truth) == set(pred.predicted)
        )
        return {
            "score": round(matches / len(self.predictions), 4) if self.predictions else 0,
            "matches": matches,
            "total": len(self.predictions),
        }

    # ── Partial Match ──

    def _partial_match(self) -> dict:
        matches = sum(
            1 for pred in self.predictions
            if set(pred.predicted).issubset(set(pred.ground_truth))
        )
        return {
            "score": round(matches / len(self.predictions), 4) if self.predictions else 0,
            "matches": matches,
            "total": len(self.predictions),
        }

    # ── Per Misleader Type ──

    def _per_type_metrics(self) -> dict:
        results = {}
        for mtype in MISLEADER_TYPES:
            tp = fp = tn = fn = 0
            for pred in self.predictions:
                gt_has = mtype in pred.ground_truth
                pred_has = mtype in pred.predicted

                if gt_has and pred_has:
                    tp += 1
                elif not gt_has and pred_has:
                    fp += 1
                elif not gt_has and not pred_has:
                    tn += 1
                else:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[mtype] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": tp + fn,  # Number of ground truth positives
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }

        return results

    # ── Confusion Summary ──

    def _confusion_summary(self) -> dict:
        """Which misleader types are most often confused with each other?"""
        confusions = defaultdict(lambda: defaultdict(int))
        for pred in self.predictions:
            false_positives = set(pred.predicted) - set(pred.ground_truth)
            for fp_type in false_positives:
                for gt_type in pred.ground_truth:
                    confusions[gt_type][fp_type] += 1

        # Convert to regular dict for serialization
        return {k: dict(v) for k, v in confusions.items()}

    # ── Save / Load ──

    def save_results(self, experiment_name: str):
        """Save predictions and metrics to JSON."""
        output_dir = EVAL_OUTPUT_DIR / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        preds_file = output_dir / "predictions.json"
        preds_file.write_text(
            json.dumps([asdict(p) for p in self.predictions], indent=2),
            encoding="utf-8"
        )

        # Save metrics
        metrics = self.compute_metrics()
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8"
        )

        # Save 2x2 comparison if applicable
        comparison = self.compute_2x2_comparison()
        if len(comparison) > 1:
            comp_file = output_dir / "2x2_comparison.json"
            comp_file.write_text(
                json.dumps(comparison, indent=2),
                encoding="utf-8"
            )

        print(f"Results saved to {output_dir}")
        return metrics

    def save_predictions_only(self, path: Path):
        """Save just the predictions (for later metric computation)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(p) for p in self.predictions], indent=2),
            encoding="utf-8"
        )

    @classmethod
    def load_predictions(cls, path: Path) -> "MisvizEvaluator":
        """Load predictions from JSON and create evaluator."""
        data = json.loads(path.read_text(encoding="utf-8"))
        evaluator = cls()
        for d in data:
            evaluator.predictions.append(Prediction(**d))
        return evaluator

    # ── Pretty Print ──

    def print_summary(self):
        """Print a formatted summary of results."""
        metrics = self.compute_metrics()
        if not metrics:
            print("No predictions to evaluate.")
            return

        binary = metrics["binary_classification"]
        em = metrics["exact_match"]
        pm = metrics["partial_match"]

        print(f"\n{'='*60}")
        print(f"  Misviz Evaluation Summary ({metrics['total_instances']} instances)")
        print(f"{'='*60}")
        print(f"  Binary (misleading vs clean):")
        print(f"    Accuracy: {binary['accuracy']:.1%}  |  F1: {binary['f1']:.1%}")
        print(f"    Precision: {binary['precision']:.1%}  |  Recall: {binary['recall']:.1%}")
        print(f"  Exact Match:   {em['score']:.1%} ({em['matches']}/{em['total']})")
        print(f"  Partial Match: {pm['score']:.1%} ({pm['matches']}/{pm['total']})")
        print(f"\n  Per Misleader Type:")
        print(f"  {'Type':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
        print(f"  {'-'*63}")
        for mtype, mdata in metrics["per_misleader_type"].items():
            if mdata["support"] > 0:
                print(f"  {mtype:<35} {mdata['precision']:>6.1%} {mdata['recall']:>6.1%} "
                      f"{mdata['f1']:>6.1%} {mdata['support']:>8}")
        print()

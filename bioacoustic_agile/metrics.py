"""Evaluation tools and metric artifacts for bioacoustic detection."""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score


@dataclass
class EvaluationSummary:
    roc_auc: float
    pr_auc: float
    optimal_threshold: float
    precision_at_optimal: float
    recall_at_optimal: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "precision_at_optimal": round(self.precision_at_optimal, 4),
            "recall_at_optimal": round(self.recall_at_optimal, 4),
        }


def evaluate_detection_performance(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_f1_beta: float = 1.0,
) -> EvaluationSummary:
    """
    Evaluates bioacoustic detection model performance.
    Calculates optimal decision threshold using Precision-Recall curve analysis.
    """
    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F-beta scores to identify optimal hardware triggering threshold
    numerator = (1 + target_f1_beta**2) * (precisions * recalls)
    denominator = (target_f1_beta**2 * precisions) + recalls
    f_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    best_idx = np.argmax(f_scores[:-1])  # Exclude last boundary point
    optimal_threshold = float(thresholds[best_idx])

    return EvaluationSummary(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        optimal_threshold=optimal_threshold,
        precision_at_optimal=float(precisions[best_idx]),
        recall_at_optimal=float(recalls[best_idx]),
    )
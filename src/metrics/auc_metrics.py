import numpy as np
from typing import Dict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)

from .base_metric import BaseMetric
from ..utils.registry import Registry


@Registry.register_metric("ExplanationAUC")
class ExplanationAUC(BaseMetric):
    """
    Computes AUROC, AUPRC, AUP, AUR between:
        gt   : (N,T,D) binary {0,1}
        attr : (N,T,D) continuous [0,1] or any range

    Flattened NTD into a single binary classification problem.
    """
    name = "ExplanationAUC"

    def __init__(self):
        pass

    def compute(self, attributions, model=None, X=None, y=None, gt=None) -> Dict[str, float]:
        """
        Args:
            attributions : np.ndarray (N,T,D) continuous scores
            gt           : np.ndarray (N,T,D) binary mask {0,1}
        """
        if gt is None:
            raise ValueError("ExplanationAUC requires ground-truth mask `gt`.")

        # ---- Flatten ----
        attr = attributions.reshape(-1).astype(np.float32)
        y_true = gt.reshape(-1).astype(np.int32)

        # Basic checks
        if not ((y_true == 0) | (y_true == 1)).all():
            raise ValueError("gt must be binary {0,1}")

        out = {}

        # --------------------------
        # 1) AUROC
        # --------------------------
        try:
            auroc = roc_auc_score(y_true, attr)
        except ValueError:
            auroc = float("nan")     # happens if only one class present
        out["AUROC"] = float(auroc)

        # --------------------------
        # 2) AUPRC  (area under PR curve)
        # --------------------------
        try:
            auprc = average_precision_score(y_true, attr)
        except ValueError:
            auprc = float("nan")
        out["AUPRC"] = float(auprc)

        # --------------------------
        # 3) AUP (Area under Precision vs threshold)
        # --------------------------
        # precision_recall_curve returns precision(t), recall(t), thresholds
        precision, recall, _ = precision_recall_curve(y_true, attr)

        # AUP = integral over precision as recall decreases
        # (not standard, but you asked explicitly)
        # Use recall as x-axis
        try:
            AUP = auc(recall, precision)
        except ValueError:
            AUP = float("nan")
        out["AUP"] = float(AUP)

        # --------------------------
        # 4) AUR (Area under Recall curve)
        # --------------------------
        # For recall curve, recall vs threshold, use thresholds from PR curve.
        # But recall is monotonic; area under recall = mean recall
        # (equivalent to integral normalized to [0,1] range).
        AUR = float(np.trapz(recall, dx=1.0/len(recall)))
        out["AUR"] = AUR

        return out

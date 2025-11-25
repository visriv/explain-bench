import numpy as np
import torch
from typing import Iterable, Union, List, Dict
from .base_metric import BaseMetric
from ..utils.registry import Registry
import torch.nn.functional as F


@Registry.register_metric("CrossEntropy")
class CrossEntropy(BaseMetric):
    """
    CrossEntropy metric:
    For each k_ratio, mask the top-k important features and compute the
    actual cross-entropy loss (NOT the drop).
    """
    name = "CrossEntropy"   
    def __init__(self, k_ratio: Union[float, Iterable[float]] = 0.2):
        if isinstance(k_ratio, (list, tuple)):
            ratios = list(k_ratio)
        else:
            ratios = [float(k_ratio)]

        clean = []
        for r in ratios:
            r = float(r)
            if r <= 0: r = 1e-9
            if r > 1: r = 1.0
            if r not in clean:
                clean.append(r)
        self.k_ratio_list = clean

    def compute(self, attributions, model, X, y, gt=None) -> Dict[str, float]:
        """
        X: (N,T,D)
        y: (N,) integer class labels
        """
        net = model.torch_module()
        device = next(net.parameters()).device
        net.eval()

        N, T, D = attributions.shape
        P = T * D

        # flatten & rank attributions
        flat_attr = np.abs(attributions.reshape(N, -1))
        order_idx = np.argsort(-flat_attr, axis=1)

        X_flat_all = X.reshape(N, -1)

        # store results
        out: Dict[str, float] = {}

        y_t = torch.tensor(y, dtype=torch.long, device=device)

        for r in self.k_ratio_list:
            k = max(1, int(round(r * P)))

            # mask top-k
            X_flat = X_flat_all.copy()
            for i in range(N):
                X_flat[i, order_idx[i, :k]] = 0.0
            X_mask = X_flat.reshape(N, T, D)

            logits = net(torch.tensor(X_mask, dtype=torch.float32, device=device))
            ce = F.cross_entropy(logits, y_t, reduction="mean").item()

            out[f"crossentropy@{r:.2f}"] = float(ce)

        return out

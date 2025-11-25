import numpy as np
import torch
from typing import Iterable, Union, List, Dict
from .base_metric import BaseMetric
from ..utils.registry import Registry


@Registry.register_metric("Accuracy")
class Accuracy(BaseMetric):
    """
    Accuracy metric:
    For each k_ratio, mask the top-k important features and compute the 
    actual classification accuracy (not the drop).
    Supports multiclass naturally.
    """
    name = "Accuracy"

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
        net = model.torch_module()
        device = next(net.parameters()).device
        net.eval()

        y = np.array(y)
        N, T, D = attributions.shape
        P = T * D

        flat_attr = np.abs(attributions.reshape(N, -1))
        order_idx = np.argsort(-flat_attr, axis=1)
        X_flat_all = X.reshape(N, -1)

        out: Dict[str, float] = {}

        for r in self.k_ratio_list:
            k = max(1, int(round(r * P)))

            # mask top-k features
            X_flat = X_flat_all.copy()
            for i in range(N):
                X_flat[i, order_idx[i, :k]] = 0.0
            X_mask = X_flat.reshape(N, T, D)

            with torch.no_grad():
                logits = net(torch.tensor(X_mask, dtype=torch.float32, device=device))
                preds = logits.argmax(dim=1).cpu().numpy()

            acc = (preds == y).mean()
            out[f"accuracy@{r:.2f}"] = float(acc)

        return out

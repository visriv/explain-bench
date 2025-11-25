import numpy as np
import torch
from typing import Iterable, Union, List, Dict
from .base_metric import BaseMetric
from ..utils.registry import Registry


@Registry.register_metric("Sufficiency")
class Sufficiency(BaseMetric):
    """
    Sufficiency metric:
    For each k_ratio, keep ONLY the top-k important features and
    mask everything else. Measures how much model confidence remains.
    Lower values = better.
    """
    name = "Sufficiency"

    def __init__(self, k_ratio: Union[float, Iterable[float]] = 0.2):
        if isinstance(k_ratio, (list, tuple)):
            ratios = list(k_ratio)
        else:
            ratios = [float(k_ratio)]

        clean: List[float] = []
        for r in ratios:
            r = float(r)
            if r <= 0: r = 1e-9
            if r > 1: r = 1.0
            if r not in clean:
                clean.append(r)
        self.k_ratio_list = clean

    def compute(self, attributions, model, X, y=None, gt=None) -> Dict[str, float]:
        net = model.torch_module()
        device = next(net.parameters()).device
        net.eval()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        # original confidence
        with torch.no_grad():
            prob0 = net(X_t).softmax(-1).cpu().numpy()
        pred = prob0.argmax(axis=-1)

        N, T, D = attributions.shape
        P = T * D

        flat_attr = np.abs(attributions.reshape(N, -1))
        order_idx = np.argsort(-flat_attr, axis=1)

        out: Dict[str, float] = {}

        for r in self.k_ratio_list:
            k = max(1, int(round(r * P)))

            # keep top-k, mask everything else
            X_flat = np.zeros_like(X.reshape(N, -1))
            for i in range(N):
                keep_idx = order_idx[i, :k]
                X_flat[i, keep_idx] = X.reshape(N, -1)[i, keep_idx]
            X_keep = X_flat.reshape(N, T, D)

            # score
            with torch.no_grad():
                prob1 = net(torch.tensor(X_keep, dtype=torch.float32, device=device)).softmax(-1).cpu().numpy()

            suff = np.mean(prob0[np.arange(N), pred] - prob1[np.arange(N), pred])
            out[f"suff@{r:.2f}"] = float(suff)

        return out

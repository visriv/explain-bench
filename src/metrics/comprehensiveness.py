import numpy as np
import torch
from typing import Iterable, Union, List, Dict
from .base_metric import BaseMetric
from ..utils.registry import Registry


@Registry.register_metric("Comprehensiveness")
class Comprehensiveness(BaseMetric):
    """
    Comprehensiveness metric:
    For each k_ratio, remove (mask) the top-k important features and
    measure the drop in model confidence for the originally predicted class.
    """
    name = "Comprehensiveness"

    def __init__(self, k_ratio: Union[float, Iterable[float]] = 0.2):
        # normalize k_ratio to a list of valid floats in (0,1]
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

        # flatten attributions once and pre-sort
        flat_attr = np.abs(attributions.reshape(N, -1))
        order_idx = np.argsort(-flat_attr, axis=1)

        out: Dict[str, float] = {}

        for r in self.k_ratio_list:
            k = max(1, int(round(r * P)))

            # mask top-k
            X_flat = X.reshape(N, -1).copy()
            for i in range(N):
                X_flat[i, order_idx[i, :k]] = 0.0 # TODO: better masking strategy
            X_mask = X_flat.reshape(N, T, D)

            # score masked
            with torch.no_grad():
                prob1 = net(torch.tensor(X_mask, dtype=torch.float32, device=device)).softmax(-1).cpu().numpy()

            drop = np.mean(prob0[np.arange(N), pred] - prob1[np.arange(N), pred])
            out[f"comp@{r:.2f}"] = float(drop)

        return out

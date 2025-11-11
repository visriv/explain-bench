import numpy as np
import torch
from .base_metric import BaseMetric
from ..utils.registry import Registry
from typing import Iterable, Union, List, Dict

@Registry.register_metric("Faithfulness")
class Faithfulness(BaseMetric):
    """
    Faithfulness metric:
    For each k_ratio in k_ratio_list, mask top-k most attributed features
    and measure the average probability drop for the originally predicted class.
    """
    name = "Faithfulness"

    def __init__(self, k_ratio: Union[float, Iterable[float]] = 0.2):
        """
        Args:
            k_ratio: single fraction or list of fractions to mask (0 < k <= 1).
        """
        if isinstance(k_ratio, (list, tuple)):
            ratios = list(k_ratio)
        else:
            ratios = [float(k_ratio)]
        # sanitize & deduplicate while preserving order
        clean: List[float] = []
        for r in ratios:
            r = float(r)
            if r <= 0:
                r = 1e-9
            if r > 1:
                r = 1.0
            if r not in clean:
                clean.append(r)
        self.k_ratio_list = clean

    def compute(self, attributions, model, X, y=None, gt=None) -> Dict[str, float]:
        net = model.torch_module()
        device = next(net.parameters()).device
        net.eval()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        # Original prediction probs & labels
        with torch.no_grad():
            prob0 = net(X_t).softmax(-1).cpu().numpy()
        pred = prob0.argmax(axis=-1)

        N, T, D = attributions.shape
        P = T * D

        # Precompute absolute attributions and a single descending order per sample
        flat_attr = np.abs(attributions.reshape(N, -1))
        # argsort once; slice as needed for different k
        order_idx = np.argsort(-flat_attr, axis=1)   # decreasing importance

        # Prepare outputs
        out: Dict[str, float] = {}

        # Iterate over requested ratios
        for r in self.k_ratio_list:
            k = max(1, int(round(r * P)))

            # Mask the top-k features per sample
            X_flat = X.reshape(N, -1).copy()
            # vectorized assignment needs a loop (indices are ragged across rows)
            for i in range(N):
                X_flat[i, order_idx[i, :k]] = 0.0
            X_mask = X_flat.reshape(N, T, D)

            # Forward on masked inputs
            with torch.no_grad():
                prob1 = net(torch.tensor(X_mask, dtype=torch.float32, device=device)).softmax(-1).cpu().numpy()

            # Average probability drop for the originally predicted class
            drop = np.mean(prob0[np.arange(N), pred] - prob1[np.arange(N), pred])

            key = f"faithfulness_drop@{r:.2f}"
            out[key] = float(drop)

        return out

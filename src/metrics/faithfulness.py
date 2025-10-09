import numpy as np
from .base_metric import BaseMetric
from ..utils.registry import Registry

@Registry.register_metric("Faithfulness")
class Faithfulness(BaseMetric):
    name = "Faithfulness"
    def __init__(self, k_ratio=0.2):
        self.k_ratio = k_ratio

    def compute(self, attributions, model, X, y, gt=None):
        # deletion metric: mask top-k attribution features over time and see prob drop
        import torch
        net = model.torch_module()
        device = next(net.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            prob0 = net(X_t).softmax(-1).cpu().numpy()
            pred = prob0.argmax(axis=-1)

        N,T,D = attributions.shape
        k = max(1, int(self.k_ratio * T * D))
        flat = attributions.reshape(N, -1)
        idx = np.argsort(-np.abs(flat), axis=1)[:, :k]
        X_mask = X.copy().reshape(N, -1)
        for i in range(N):
            X_mask[i, idx[i]] = 0.0
        X_mask = X_mask.reshape(N, T, D)

        with torch.no_grad():
            prob1 = net(torch.tensor(X_mask, dtype=torch.float32, device=device)).softmax(-1).cpu().numpy()

        # average prob drop for predicted class
        drop = 0.0
        for i in range(N):
            drop += (prob0[i, pred[i]] - prob1[i, pred[i]])
        return {"faithfulness_drop": float(drop / N)}

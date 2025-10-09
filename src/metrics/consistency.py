import numpy as np
from .base_metric import BaseMetric
from ..utils.registry import Registry

@Registry.register_metric("Consistency")
class Consistency(BaseMetric):
    name = "Consistency"
    def compute(self, attributions, model, X, y):
        # run model twice with the same X and measure attribution similarity (cosine)
        # (in practice you'd compare across checkpoints or augmentations)
        eps = 1e-8
        A = attributions.reshape(len(X), -1)
        # pretend second attribution is with tiny noise and same explainer
        A2 = A + 1e-6*np.random.randn(*A.shape)
        num = (A*A2).sum(axis=1)
        den = (np.linalg.norm(A, axis=1)+eps) * (np.linalg.norm(A2, axis=1)+eps)
        cos = (num/den).mean()
        return {"consistency_cos": float(cos)}

import numpy as np
from .base_metric import BaseMetric
from ..utils.registry import Registry

@Registry.register_metric("Stability")
class Stability(BaseMetric):
    name = "Stability"
    def compute(self, attributions, model, X, y, gt=None):
        # sensitivity to small input noise (lower variance -> higher stability)
        rng = np.random.default_rng(0)
        noises = [rng.normal(0, 0.02, size=X.shape) for _ in range(3)]
        # naive: variance of attribution magnitude across jittered inputs
        mags = []
        for n in noises:
            mags.append(np.abs(attributions + n).mean())
        var = np.var(mags)
        return {"stability_var": float(var)}

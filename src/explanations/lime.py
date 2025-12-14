# Simple placeholder LIME-style: random perturbation sensitivity estimate.
import numpy as np
from .base_explainer import BaseExplainer
from ..utils.registry import Registry

@Registry.register_explainer("LIME")
class LIMEExplainer(BaseExplainer):
    name = "LIME"
    def __init__(self, samples=64, sigma=0.5, seed=42):
        self.samples=samples; self.sigma=sigma; self.rng=np.random.default_rng(seed)

    def explain(self, model, X):
        # crude sensitivity: perturb noise and measure change in logits for predicted class
        # logits0 = model.predict_class(X)  # predicted labels
        # to get class-prob change, we call torch net directly
        net = model.torch_module()
        import torch
        device = next(net.parameters()).device
        net.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            prob0 = net(X_t).softmax(-1)
            prob0_np = prob0.detach().cpu().numpy()  
            tgt = prob0.argmax(dim=-1)
            tgt_np = prob0.argmax(dim=-1).detach().cpu().numpy()  # (N,) numpy

        N,T,D = X.shape
        atts = np.zeros_like(X, dtype=np.float32)
        for s in range(self.samples):
            noise = self.rng.normal(0, self.sigma, size=X.shape).astype("float32")
            Xp = X + noise
            with torch.no_grad():
                probp = net(torch.tensor(Xp, dtype=torch.float32, device=device)).softmax(-1).cpu().numpy()
                
            # contribution magnitude = |Δp(target)|; apportion by |noise|
            for i in range(N):
                delta = abs(probp[i, tgt_np[i]] - prob0_np[i, tgt_np[i]])
                w = np.abs(noise[i]) + 1e-6
                atts[i] += (delta * w) / (self.samples)
        return atts

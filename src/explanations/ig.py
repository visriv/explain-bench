import torch
from captum.attr import IntegratedGradients
from .base_explainer import BaseExplainer
from ..utils.registry import Registry

@Registry.register_explainer("IG")
class IGExplainer(BaseExplainer):
    def __init__(self, steps: int = 32):
        self.steps = steps
    name = "IG"
    def explain(self, model, X):
        net = model.torch_module()
        device = next(net.parameters()).device
        net.eval()
        ig = IntegratedGradients(net)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        baseline = torch.zeros_like(X_t)

        logits = net(X_t)
        target = logits.argmax(dim=1)

        net.train()
        # Integrated gradients attribution for that target
        attrs = ig.attribute(X_t, baselines=baseline, n_steps=self.steps, target=target)
        net.eval()
        return attrs.detach().cpu().numpy()

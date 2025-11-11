import torch
from captum.attr import IntegratedGradients
from .base_explainer import BaseExplainer
from ..utils.registry import Registry

@Registry.register_explainer("IG")
class IGExplainer(BaseExplainer):
    name = "IG"
    def explain(self, model, X, steps=32):
        net = model.torch_module()
        device = next(net.parameters()).device
        net.eval()
        ig = IntegratedGradients(net)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        baseline = torch.zeros_like(X_t)
        # Forward pass to pick top predicted class for each sample
        with torch.no_grad():
            logits = net(X_t)
            target = logits.argmax(dim=1)

        # Integrated gradients attribution for that target
        attrs = ig.attribute(X_t, baselines=baseline, n_steps=steps, target=target)
        return attrs.detach().cpu().numpy()

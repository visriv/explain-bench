import torch
from .base_explainer import BaseExplainer
from ..utils.registry import Registry

@Registry.register_explainer("Grad")
class GradExplainer(BaseExplainer):
    name = "Grad"
    def explain(self, model, X):
        net = model.torch_module()
        device = next(net.parameters()).device
        net.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
        # assume predicted class target for simplicity
        logits = net(X_t)
        top = logits.argmax(dim=-1)
        atts = []
        for i in range(X_t.shape[0]):
            net.zero_grad(set_to_none=True)
            logits[i, top[i]].backward(retain_graph=True)
            atts.append(X_t.grad[i].detach().cpu().numpy())
            X_t.grad.zero_()
        return (torch.stack([torch.tensor(a) for a in atts]).cpu().numpy())

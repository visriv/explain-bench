import torch
import torch.nn as nn
from .base_explainer import BaseExplainer
from ..utils.registry import Registry
from tqdm import tqdm
@Registry.register_explainer("Grad")
class GradExplainer(BaseExplainer):
    name = "Grad"

    def explain(self, model, X):
        net = model.torch_module()
        device = next(net.parameters()).device

        # --- Save states and neutralize stochastic layers ---
        was_training = net.training
        # batchnorms we’ll force to eval, but keep model in train() for cuDNN RNN backward
        bn_layers = []
        dropout_layers = []

        for m in net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_layers.append((m, m.training))
                m.eval()  # freeze running stats/affine behavior
            elif isinstance(m, nn.Dropout):
                dropout_layers.append((m, m.p))
                m.p = 0.0  # no dropout, even in train()

        # cuDNN RNN backward requires training mode:
        net.train(True)

        # --- Prepare input with grad ---
        X_t = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)

        # Forward pass
        logits = net(X_t)                  # (N, C)
        top = logits.argmax(dim=-1)        # (N,)
        N = X_t.shape[0]
        # --- Per-sample gradient wrt input ---
        atts = []
        for i in tqdm(range(N), desc="IG grads", leave=False):
            net.zero_grad(set_to_none=True)
            if X_t.grad is not None:
                X_t.grad.zero_()
            # scalar target logit for sample i
            logits[i, top[i]].backward(retain_graph=True)
            atts.append(X_t.grad[i].detach().cpu().numpy())

        # --- Restore original states ---
        for m, was_train in bn_layers:
            if was_train:
                m.train(True)
            else:
                m.eval()

        for m, p in dropout_layers:
            m.p = p

        net.train(was_training)

        # (N, T, D) numpy
        import numpy as np
        return np.stack(atts, axis=0)

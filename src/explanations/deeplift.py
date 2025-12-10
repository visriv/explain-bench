import numpy as np
import torch
from captum.attr import DeepLift
from .base_explainer import BaseExplainer
from ..utils.registry import Registry


@Registry.register_explainer("DeepLIFT")
class DeepLIFTExplainer(BaseExplainer):
    """
    DeepLIFT for time-series explanations.
    
    Computes contributions relative to a reference input (baseline).
    Default baseline = zeros of same shape.
    """
    name = "DeepLift"

    def __init__(self, baseline="zero"):
        """
        Args:
            baseline: "zero" → use zero baseline (default)
                      "mean" → use mean over X as baseline (computed per batch)
                      np.ndarray → explicit baseline array of shape (T,D)
        """
        self.baseline = baseline

    # ------------------------------------------------------------------
    def _make_baseline(self, X):
        """
        Produce a baseline of shape (N,T,D).
        """
        if isinstance(self.baseline, str):
            if self.baseline == "zero":
                return np.zeros_like(X, dtype=np.float32)
            elif self.baseline == "mean":
                mean = X.mean(axis=0, keepdims=True)  # (1,T,D)
                return np.repeat(mean, X.shape[0], axis=0)
            else:
                raise ValueError(f"Unknown baseline type {self.baseline}")

        # explicit numpy baseline provided: shape must be (T,D)
        if isinstance(self.baseline, np.ndarray):
            assert self.baseline.shape == (X.shape[1], X.shape[2])
            return np.repeat(self.baseline[None, :, :], X.shape[0], axis=0)

        raise ValueError("Invalid baseline specification")

    # ------------------------------------------------------------------
    def explain(self, model, X):
        """
        Args:
            model: ExplainBench model wrapper containing torch_module()
            X: np.ndarray (N,T,D)

        Returns:
            attributions: np.ndarray (N,T,D)
        """
        net = model.torch_module()
        device = next(net.parameters()).device
        net.eval()

        X = np.asarray(X, dtype=np.float32)
        baselines = self._make_baseline(X)

        N, T, D = X.shape

        # Captum expects (N, channels, length, ...) → for time series we use (N, D, T)
        # but your framework convention is (N,T,D). Let's convert:
        X_ch = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32, device=device)      # (N, D, T)
        B_ch = torch.tensor(baselines.transpose(0, 2, 1), dtype=torch.float32, device=device)

        # DeepLIFT instance
        dl = DeepLift(net)

        # Compute attributions for the predicted class (standard XAI convention)
        with torch.no_grad():
            logits = net(X_ch)                  # shape (N,C)
            pred_classes = logits.argmax(dim=-1)

        # Captum expects target index tensor
        target_idx = pred_classes

        # Compute attributions
        # deepLIFT returns list if model has multiple inputs; here it's a single tensor
        attributions = dl.attribute(X_ch, baselines=B_ch, target=target_idx)  # (N, D, T)

        # Back to (N,T,D)
        attributions = attributions.detach().cpu().numpy().transpose(0, 2, 1)

        # Ensure dtype
        return attributions.astype("float32")

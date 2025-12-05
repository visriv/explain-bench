# src/explanations/fit.py
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn

from .base_explainer import BaseExplainer
from ..utils.registry import Registry


import logging
import pathlib



from src.explanations.base_explainer import BaseExplainer
from src.explanations.fit_generator.generator import GeneratorTrainingResults
from src.explanations.fit_generator.joint_generator import JointFeatureGenerator

@Registry.register_explainer("FIT")
class FITExplainer(BaseExplainer):
    """
    The explainer for FIT. The implementation is from layer6/WinIT
    https://github.com/layer6ai-labs/WinIT/blob/main/winit/explainer/fitexplainers.py

    FIT explainer adapted to the (N,T,D)->(N,T,D) schema.
    """

    name = "FIT"

    def __init__(
        self,
        feature_size: int,
        num_samples: int = 10,
        **kwargs
    ):
        """
        Args:
            feature_size: D (number of features)
            num_samples: counterfactual samples per feature
        """
        self.feature_size = feature_size
        self.n_samples = num_samples
        self.generator_path = kwargs.get("generator_path", None)
        self.epochs_gen = kwargs.get("epochs_gen", 300)
        # generator will be initialized lazily
        self.generator: JointFeatureGenerator | None = None
        self.log = logging.getLogger("ExplainBench")
        if kwargs:
            print(f"[FIT] Warning: unused kwargs={kwargs}")

    def _model_predict(self, model, x):
        """
        Run predict on base model. If the output is binary, i.e. num_class = 1, we will make it
        into a probability distribution by append (p, 1-p) to it.
        """
        x = x.permute(0, 2, 1)
        activation = nn.Softmax(dim=-1)
        logits = model.net.forward(x, return_all=False)
        p = activation(logits).to(x.device)
        # p = model.predict(x, return_all=False)
        if model.num_classes == 2:
            # Create a 'probability distribution' (p, 1 - p)
            prob_distribution = torch.cat((p, 1 - p), dim=1)
            return prob_distribution
        return p


    # ------------------------------------------------------------------
    # GENERATOR HELPERS
    # ------------------------------------------------------------------

    def _init_generators(self, device):
        import pathlib
        gen_path = pathlib.Path(self.generator_path) / "joint_generator"
        gen_path.mkdir(parents=True, exist_ok=True)
        self.generator = JointFeatureGenerator(
            feature_size=self.feature_size,
            device=device,
            gen_path=gen_path,
            hidden_size=self.feature_size * 3,
        )


    def _train_or_load_generator(self, train_loader=None, valid_loader=None):
        """
        If checkpoint exists → load.
        If not → require train_loader and valid_loader to train.
        """
        ckpt = self.generator._get_model_file_name()

        if ckpt.exists():
            self.log.info(f"[FIT] Loading existing generator checkpoint: {ckpt}")
            self.generator.load_generator()
        else:
            assert train_loader is not None and valid_loader is not None, \
                "FIT needs train_loader and valid_loader the first time you run it."
            self.log.info("[FIT] Training generator from scratch...")
            self.generator.train_generator(train_loader, valid_loader, num_epochs = self.epochs_gen)

    def explain(self, model, x):
        """
        X: numpy array (N,T,D)
        Returns numpy saliency (N,T,D)
        """
        net = model.torch_module()
        net.eval()

        self.device = next(net.parameters()).device
        device = self.device

        # ensure generator loaded
        if self.generator is None:
            self._init_generators(device)
            self._train_or_load_generator(train_loader=self.train_loader,
                                       valid_loader=self.valid_loader)
            
        self.generator.eval()
        self.generator.to(device)

        x = torch.tensor(x, dtype=torch.float32, device=device)
        x = x.permute(0, 2, 1)  # (N,T,D) -> (N,D,T)
        _, n_features, t_len = x.shape
        score = np.zeros(list(x.shape))

        for t in range(1, t_len):
            p_y_t = self._model_predict(model, x[:, :, : t + 1]).float()
            p_tm1 = self._model_predict(model, x[:, :, 0:t]).float()

            for i in range(n_features):
                mu_z, std_z = self.generator.get_z_mu_std(x[:, :, :t])
                x_hat_t, _ = self.generator.forward_conditional_multisample_from_z_mu_std(
                    x[:, :, :t], x[:, :, t], [i], mu_z, std_z, self.n_samples
                )
                x_hat = x[:, :, : t + 1].unsqueeze(0).repeat(self.n_samples, 1, 1, 1)
                x_hat[:, :, :, t] = x_hat_t[:, :, :, 0]
                x_hat = x_hat.reshape(-1, n_features, t + 1)
                y_hat_t = self._model_predict(model, x_hat)
                y_hat_t = y_hat_t.reshape(self.n_samples, -1, 1) #y_hat_t.shape[-1])

                first_term = torch.sum(
                    torch.nn.KLDivLoss(reduction="none")(torch.log(p_tm1), p_y_t), -1
                )
                p_y_t_expanded = p_y_t.unsqueeze(0).expand(self.n_samples, -1, -1)
                second_term = torch.sum(
                    torch.nn.KLDivLoss(reduction="none")(torch.log(y_hat_t), p_y_t_expanded), -1
                )
                div = first_term.unsqueeze(0) - second_term
                E_div = torch.mean(div, dim=0).detach().cpu().numpy()

                score[:, i, t] = 2.0 / (1 + np.exp(-5 * E_div)) - 1

        self.log.info("attribution done")
        return score.transpose(0, 2, 1)

# src/explanations/fit.py
from __future__ import annotations
from time import time
import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm

from .base_explainer import BaseExplainer
from ..utils.registry import Registry


import logging
import pathlib



from src.explanations.base_explainer import BaseExplainer
from src.explanations.fit_generator.generator import GeneratorTrainingResults, FeatureGenerator, BaseFeatureGenerator
from src.explanations.fit_generator.joint_generator import JointFeatureGenerator

@Registry.register_explainer("WINIT")
class WINITExplainer(BaseExplainer):
    """
    The explainer for WINIT. The implementation is from layer6/WinIT
    https://github.com/layer6ai-labs/WinIT/blob/main/winit/explainer/winitexplainers.py

    WINIT explainer adapted to the (N,T,D)->(N,T,D) schema.
    """

    name = "WINIT"

    def __init__(
        self,
        num_features: int,
        # path: pathlib.Path,
        window_size: int = 10,
        num_samples: int = 3,
        conditional: bool = False,
        joint: bool = False,
        metric: str = "pd",
        random_state: int | None = None,
        **kwargs,
    ):
        """
        Construtor

        Args:
            num_features:
                The number of features.
            path:
                The path indicating where the generator to be saved.
            train_loader:
                The train loader if we are using the data distribution instead of a generator
                for generating counterfactual. Default=None.
            window_size:
                The window size for the WinIT
            num_samples:
                The number of Monte-Carlo samples for generating counterfactuals.
            conditional:
                Indicate whether the individual feature generator we used are conditioned on
                the current features. Default=False
            joint:
                Indicate whether we are using the joint generator.
            metric:
                The metric for the measures of comparison of the two distributions for i(S)_a^b
            random_state:
                The random state.
            **kwargs:
                There should be no additional kwargs.
        """


        self.window_size = window_size
        self.num_samples = num_samples
        self.num_features = num_features
        # self.data_name = data_name
        self.joint = joint
        self.conditional = conditional
        self.metric = metric
        self.epochs_gen = kwargs.get("epochs_gen", 300)
        self.generators: BaseFeatureGenerator | None = None
        self.generator_path = kwargs.get("generator_path", None)
        self.data_distribution = None
        self.rng = np.random.default_rng(random_state)
        self.log = logging.getLogger("ExplainBench")
        if kwargs:
            print(f"[WINIT] Warning: unused kwargs={kwargs}")

    def _model_predict(self, model, x):
        """
        Run predict on base model. If the output is binary, i.e. num_class = 2, we will make it
        into a probability distribution by append (p, 1-p) to it.
        input: [N, D, T]
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
        if self.joint:
            gen_path = pathlib.Path(self.generator_path) / "joint_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = JointFeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=self.num_features * 3,
                prediction_size=self.window_size,
                data="not_mimic"
            )
        else:
            gen_path = pathlib.Path(self.generator_path) / "feature_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = FeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=50,
                prediction_size=self.window_size,
                conditional=self.conditional,
                data="not_mimic"
            )
        if self.train_loader is not None:
            self.data_distribution = (
                torch.stack([x[0] for x in self.train_loader.dataset]).detach().cpu().numpy()
            )

    def _train_or_load_generator(self, train_loader=None, valid_loader=None):
        """
        If checkpoint exists → load.
        If not → require train_loader and valid_loader to train.
        """
        ckpts = self.generators._get_model_file_name()

        if not isinstance(ckpts, list):
            ckpt = ckpts
            if ckpt.exists():
                self.log.info(f"[FIT] Loading existing generator checkpoint: {ckpt}")
                self.generator.load_generator()
            else:
                assert train_loader is not None and valid_loader is not None, \
                    "FIT needs train_loader and valid_loader the first time you run it."
                self.log.info("[FIT] Training joint generator from scratch...")
                self.generator.train_generator(train_loader, valid_loader, num_epochs = self.epochs_gen)

        else:
            for ckpt in ckpts:
                if ckpt.exists():
                    self.log.info(f"[WINIT] Loading existing generator checkpoint: {ckpt}")
                    self.generators.load_generator()
                else:
                    assert train_loader is not None and valid_loader is not None, \
                    "WINIT needs train_loader and valid_loader the first time you run it."
                    self.log.info("[WINIT] Training individual generators from scratch...")
                    self.generators.train_generator(train_loader, valid_loader, num_epochs = self.epochs_gen)
        

            



    def explain(self, model, x):
        """
        Compute the WinIT attribution.

        Args:
            x:
                The input Tensor of shape (N, T, D)

        Returns:
            The attribution Tensor of shape (N, D, T, window_size)
            The (i, j, k, l)-entry is the importance of observation (i, j, k - window_size + l + 1)
            to the prediction at time k

        """
        net = model.torch_module()
        net.eval()
        self.device = next(net.parameters()).device
        device = self.device
        
        # ensure generator loaded
        if self.generators is None:
            self._init_generators(device)
            self._train_or_load_generator(train_loader=self.train_loader,
                                       valid_loader=self.valid_loader)
            
        self.generators.eval()
        self.generators.to(device)


        with torch.no_grad():
            tic = time()

            batch_size, num_timesteps, num_features = x.shape
            x = torch.tensor(x, dtype=torch.float32, device=device)
            x = x.permute(0, 2, 1)  # (N, D, T)
            scores = []

            for t in tqdm(range(num_timesteps)):
                window_size = min(t, self.window_size)

                if t == 0:
                    scores.append(np.zeros((batch_size, num_features, self.window_size)))
                    continue

                # x = (num_sample, D, T)
                p_y = self._model_predict(model, x[:, :, : t + 1])

                iS_array = np.zeros((num_features, window_size, batch_size), dtype=float)
                for n in range(window_size):
                    time_past = t - n
                    time_forward = n + 1
                    counterfactuals = self._generate_counterfactuals(
                        time_forward, x[:, :, :time_past], x[:, :, time_past : t + 1]
                    )
                    # counterfactual shape = (D, num_samples, N, time_forward)
                    for f in range(num_features):
                        # repeat input for num samples
                        x_hat_in = (
                            x[:, :, : t + 1].unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
                        )  # (ns, N, D, time)
                        # replace unknown with counterfactuals
                        x_hat_in[:, :, f, time_past : t + 1] = counterfactuals[f, :, :, :]

                        # Compute Q = p(y_t | tilde(X)^S_{t-n:t})
                        p_y_hat = self._model_predict(
                            model, x_hat_in.reshape(self.num_samples * batch_size, num_features, t + 1)
                        )

                        # Compute P = p(y_t | X_{1:t})
                        p_y_exp = (
                            p_y.unsqueeze(0)
                            .repeat(self.num_samples, 1, 1)
                            .reshape(self.num_samples * batch_size, p_y.shape[-1])
                        )
                        iSab_sample = self._compute_metric(p_y_exp, p_y_hat).reshape(
                            self.num_samples, batch_size
                        )
                        iSab = torch.mean(iSab_sample, dim=0).detach().cpu().numpy()
                        # For KL, the metric can be unbounded. We clip it for numerical stability.
                        iSab = np.clip(iSab, -1e6, 1e6)
                        iS_array[f, n, :] = iSab

                # Compute the I(S) array
                b = iS_array[:, 1:, :] - iS_array[:, :-1, :]
                iS_array[:, 1:, :] = b

                score = iS_array[:, ::-1, :].transpose(2, 0, 1)  # (N, D, T)

                # Pad the scores when time forward is less than window size.
                if score.shape[2] < self.window_size:
                    score = np.pad(score, ((0, 0), (0, 0), (self.window_size - score.shape[2], 0)))
                scores.append(score)
            self.log.info(f"Batch done: Time elapsed: {(time() - tic):.4f}")

            scores = np.stack(scores).transpose((1, 2, 0, 3))  # (N, D, T, window_size)
            scores = np.mean(scores, axis = 3) # average over window size
            return scores

    def _compute_metric(self, p_y_exp: torch.Tensor, p_y_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric for comparisons of two distributions.

        Args:
            p_y_exp:
                The current expected distribution. Shape = (batch_size, num_states)
            p_y_hat:
                The modified (counterfactual) distribution. Shape = (batch_size, num_states)

        Returns:
            The result Tensor of shape (batch_size).

        """
        if self.metric == "kl":
            return torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp), -1)
        if self.metric == "js":
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return torch.sum((lhs + rhs) / 2, -1)
        if self.metric == "pd":
            diff = torch.abs(p_y_hat - p_y_exp)
            return torch.sum(diff, -1)
        raise Exception(f"unknown metric. {self.metric}")

    
    def _generate_counterfactuals(
        self, time_forward: int, x_in: torch.Tensor, x_current: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate the counterfactuals.

        Args:
            time_forward:
                Number of timesteps of counterfactuals we wish to generate.
            x_in:
                The past Tensor. Shape = (batch_size, num_features, num_times)
            x_current:
                The current Tensor if a conditional generator is used.
                Shape = (batch_size, num_features, time_forward). If the generator is not
                conditional, x_current is None.

        Returns:
            Counterfactual of shape (num_features, num_samples, batch_size, time_forward)

        """
        # x_in shape (bs, num_feature, num_time)
        # x_current shape (bs, num_feature, time_forward)
        # return counterfactuals shape (num_feature, num_samples, batchsize, time_forward)
        batch_size, _, num_time = x_in.shape
        if self.data_distribution is not None:
            # Random sample instead of using generator
            counterfactuals = torch.zeros(
                (self.num_features, self.num_samples, batch_size, time_forward), device=self.device
            )
            for f in range(self.num_features):
                values = self.data_distribution[:, f, :].reshape(-1)
                sampled = self.rng.choice(values, size=(self.num_samples, batch_size, time_forward))
                counterfactuals[f, :, :, :] = torch.tensor(
                    sampled,
                    dtype=torch.float32,
                    device=self.device,
                )
            return counterfactuals

        if isinstance(self.generators, FeatureGenerator):
            mu, std = self.generators.forward(x_current, x_in, deterministic=True)
            mu = mu[:, :, :time_forward]
            std = std[:, :, :time_forward]  # (bs, f, time_forward)
            counterfactuals = mu.unsqueeze(0) + torch.randn(
                self.num_samples, batch_size, self.num_features, time_forward, device=self.device
            ) * std.unsqueeze(0)
            return counterfactuals.permute(2, 0, 1, 3)

        if isinstance(self.generators, JointFeatureGenerator):
            counterfactuals = torch.zeros(
                (self.num_features, self.num_samples, batch_size, time_forward), device=self.device
            )
            for f in range(self.num_features):
                mu_z, std_z = self.generators.get_z_mu_std(x_in)
                gen_out, _ = self.generators.forward_conditional_multisample_from_z_mu_std(
                    x_in,
                    x_current,
                    list(set(range(self.num_features)) - {f}),
                    mu_z,
                    std_z,
                    self.num_samples,
                )
                # gen_out shape (ns, bs, num_feature, time_forward)
                counterfactuals[f, :, :, :] = gen_out[:, :, f, :]
            return counterfactuals

        raise ValueError("Unknown generator or no data distribution provided.")


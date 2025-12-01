# src/explanations/dynamask.py

import logging
import numpy as np
import torch

from .base_explainer import BaseExplainer
from ..utils.registry import Registry

# Dynamask dependencies from WinIT repo
from src.explanations.dynamask_utils.mask_group import MaskGroup
from src.explanations.dynamask_utils.perturbation import GaussianBlur, FadeMovingAverage
from src.explanations.dynamask_utils.losses import log_loss_multiple, cross_entropy_multiple


@Registry.register_explainer("Dynamask")
class DynamaskExplainer(BaseExplainer):
    """
    Dynamask explainer adapted to the (N,T,D)->(N,T,D) schema of ExplainBench.

    - Uses vectorised mask optimisation (MaskGroup.fit_multiple).
    - Computes extremal masks and extracts saliency.

    The explainer for Dynamask. The code was modified from the Dynamask repository.
    https://github.com/JonathanCrabbe/Dynamask
    This code fromm https://github.com/layer6ai-labs/WinIT/blob/main/winit/explainer/dynamaskexplainer.py
    
    As dynamask "training" does not have early stopping, we vectorize the dynamask
    generation to make it about 100x faster.

    """

    name = "Dynamask"

    def __init__(
        self,
        area_list=None,
        num_epoch: int = 200,
        num_class: int = 1,
        blur_type: str = "gaussian",
        deletion_mode: bool = False,
        size_reg_factor_dilation: float = 100.0,
        time_reg_factor: float = 1.0,
        loss: str = "logloss",
        use_last_timestep_only: bool = False,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        """
        Params match the WinIT implementation.
        """
        # Store params
        self.area_list = (
            area_list if area_list is not None else np.arange(0.25, 0.35, 0.01)
        )
        self.num_epoch = num_epoch
        self.num_class = num_class
        self.deletion_mode = deletion_mode
        self.size_reg_factor_dilation = size_reg_factor_dilation
        self.time_reg_factor = time_reg_factor
        self.use_last_timestep_only = use_last_timestep_only

        self.device = device
        # Perturbation type
        if blur_type == "gaussian":
            self.pert = GaussianBlur(
                self.device, sigma_max=1.0
            )  # This is the perturbation operator
        elif blur_type == "fadema":
            self.pert = FadeMovingAverage(self.device)
        else:
            raise Exception("Unknown blur_type " + blur_type)
        self.blur_type = blur_type

       
        # This is the list of masks area to consider
        self.area_list = area_list if area_list is not None else np.arange(0.25, 0.35, 0.01)

        self.num_epoch = num_epoch
        self.num_class = num_class
        self.deletion_mode = deletion_mode
        self.size_reg_factor_dilation = size_reg_factor_dilation
        self.time_reg_factor = time_reg_factor
        if loss == "logloss":
            self.loss = log_loss_multiple
        elif loss == "ce":
            self.loss = cross_entropy_multiple
        else:
            raise RuntimeError(f"Unrecognized loss {loss}")
        self.loss_str = loss
        self.use_last_timestep_only = use_last_timestep_only

        if len(kwargs):
            log = logging.getLogger("ExplainBench")
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    # ------------------------------------------------------------------------------
    # MAIN ENTRY POINT — REQUIRED BY YOUR SCHEMA
    # ------------------------------------------------------------------------------
    def explain(self, model, X):
        """
        Memory-safe wrapper for computing attributions on (N,T,D) numpy arrays.

        Args:
            model : your wrapper model
            X     : numpy array [N,T,D]

        Returns:
            numpy array [N,T,D]
        """

        model.net.eval()
        model.torch_module().zero_grad()
        device = next(model.net.parameters()).device
        self.device = device

        # convert to torch tensor ONCE
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        # choose an initial batch size 
        init_bs = X.shape[0]

        return self._explain_recursive(model, X_t, batch_size=init_bs)



    def _explain_recursive(self, model, X_t, batch_size):
        """
        Recursively compute attributions with automatic batch-size reduction on CUDA OOM.
        """

        N = X_t.shape[0]

        try:
            # Try running attribution in the current batch size
            return self._run_explain_in_batches(model, X_t, batch_size)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                new_bs = max(1, batch_size // 2)

                if new_bs == batch_size:
                    raise RuntimeError("OOM even at batch=1")

                print(f"[OOM] Reducing batch size {batch_size} → {new_bs}")
                torch.cuda.empty_cache()

                # Retry recursively
                return self._explain_recursive(model, X_t, new_bs)

            else:
                raise e

    def _run_explain_in_batches(self, model, X_t, batch_size):
        """
        Run attribution for X_t in fixed batches.

        Returns:
            numpy array [N,T,D]
        """

        N = X_t.shape[0]
        atts = []

        for i in range(0, N, batch_size):
            xb = X_t[i:i+batch_size]

            # Call your main attribution logic
            att = self._attribute_main(model, xb)   # should return numpy or torch

            # standardize output to numpy
            if isinstance(att, torch.Tensor):
                att = att.detach().cpu().numpy()

            atts.append(att)

            # clean up GPU memory
            torch.cuda.empty_cache()

        # concatenate along batch dimension
        return np.concatenate(atts, axis=0)


    def _attribute_main(self, model, x):

        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
   
        def f(x_in):
            # x_in (num_sample, num_time, num_feature)
            num_sample, num_times, num_features = x_in.shape
            # x_in = x_in.permute(0, 2, 1)
            out = model.net(x_in, return_all=True)  # (num_sample, num_state=1, num_time)
            out = model.net.activation(out)
            if self.num_class == 1:
                # stack
                out = out.reshape(num_sample, num_times)
                out = torch.stack([1 - out, out], dim=2)
            else:
                out = out.reshape(num_sample, self.num_class, num_times).permute(0, 2, 1)

            if self.use_last_timestep_only:
                out = out[:, -1:, :]
            return out  # (num_sample, num_time, 2 or num_state)

        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        # Convert (N,T,D) → (N,D,T)
        # x = x.permute(0, 2, 1) 
        # Fit the group of mask:
        mask_group = MaskGroup(
            self.pert, self.device, verbose=False, deletion_mode=self.deletion_mode
        )
        mask_group.fit_multiple(
            X=x,
            f=f,
            use_last_timestep_only=self.use_last_timestep_only,
            loss_function_multiple=self.loss,
            area_list=self.area_list,
            learning_rate=1.0,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=self.size_reg_factor_dilation,
            initial_mask_coeff=0.5,
            n_epoch=self.num_epoch,
            momentum=1.0,
            time_reg_factor=self.time_reg_factor,
        )

        # Extract the extremal mask:
        y_test = f(x).unsqueeze(0)
        thresh = cross_entropy_multiple(y_test, y_test)  # This is what we call epsilon in the paper
        mask = mask_group.get_extremal_mask_multiple(thresholds=thresh)
        mask_saliency = mask.permute(0, 2, 1)

        torch.backends.cudnn.enabled = orig_cudnn_setting

        return mask_saliency.detach().cpu().numpy()   
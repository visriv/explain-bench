import os
import pickle
import numpy as np
from .base_dataset import BaseDataset
from ..utils.registry import Registry


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@Registry.register_dataset("StateDataset")
class StateDataset(BaseDataset):
    """
    Loads the 'simulated_state_data' PKL set produced by your generator.

    Expected directory layout (defaults can be overridden via params):
      ./data/simulated_state_data/
        ├─ state_dataset_x_train.pkl          # np.ndarray (N_train, T, D) or (N_train, D, T)
        ├─ state_dataset_x_test.pkl           # np.ndarray (N_test,  T, D) or (N_test,  D, T)
        ├─ state_dataset_y_train.pkl          # np.ndarray (N_train,)
        ├─ state_dataset_y_test.pkl           # np.ndarray (N_test,)
        ├─ state_dataset_importance_train.pkl # np.ndarray (N_train, T, D) or (N_train, D, T)
        ├─ state_dataset_importance_test.pkl  # np.ndarray (N_test,  T, D) or (N_test,  D, T)
        ├─ state_dataset_logits_train.pkl     # np.ndarray (N_train, C) or per-step
        ├─ state_dataset_logits_test.pkl      # np.ndarray (N_test,  C) or per-step
        ├─ state_dataset_states_train.pkl     # any aux (e.g., latent states)
        └─ state_dataset_states_test.pkl

    Parameters
    ----------
    data_dir : str
        Root folder containing the PKLs. Default: "./data/simulated_state_data"
    transpose_to_t_first : bool
        If True and an array looks like (N, D, T), transpose to (N, T, D).
    cast_float32 : bool
        Cast X and importance to float32.
    cast_int64 : bool
        Cast y to int64.
    limit_train, limit_test : int or None
        Truncate for quick smoke tests.
    filemap : dict or None
        Optional override for filenames. Keys:
        {x_train,x_test,y_train,y_test,imp_train,imp_test,logits_train,logits_test,states_train,states_test}
    """

    def __init__(
        self,
        data_dir: str = "./data/simulated_state_data",
        transpose_to_t_first: bool = True,
        cast_float32: bool = True,
        cast_int64: bool = True,
        limit_train: int | None = None,
        limit_test: int | None = None,
        filemap: dict | None = None,
    ):
        self.data_dir = data_dir
        self.transpose_to_t_first = transpose_to_t_first
        self.cast_float32 = cast_float32
        self.cast_int64 = cast_int64
        self.limit_train = limit_train
        self.limit_test = limit_test

        # default filenames (match your screenshot)
        defaults = dict(
            x_train="state_dataset_x_train.pkl",
            x_test="state_dataset_x_test.pkl",
            y_train="state_dataset_y_train.pkl",
            y_test="state_dataset_y_test.pkl",
            imp_train="state_dataset_importance_train.pkl",
            imp_test="state_dataset_importance_test.pkl",
            logits_train="state_dataset_logits_train.pkl",
            logits_test="state_dataset_logits_test.pkl",
            states_train="state_dataset_states_train.pkl",
            states_test="state_dataset_states_test.pkl",
        )
        self.files = {**defaults, **(filemap or {})}

    # ---------- helpers ----------

    def _maybe_transpose_TD(self, arr: np.ndarray) -> np.ndarray:
        """Normalize to (N, T, D) if transpose_to_t_first is True."""
        if arr is None or arr.ndim != 3:
            return arr
        N, a, b = arr.shape
        if self.transpose_to_t_first and a > b:
            return arr.transpose(0, 2, 1)
        return arr

    def _prepare_xy(
        self, X: np.ndarray, y: np.ndarray, limit: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self._maybe_transpose_TD(X)
        if limit is not None:
            X = X[:limit]
            y = y[:limit]
        if self.cast_float32:
            X = X.astype("float32", copy=False)
        if self.cast_int64:
            y = y.astype("int64", copy=False)
        return X, y

    # ---------- public API ----------

    def load(self):
        """Return ((X_train, y_train), (X_test, y_test)) with X in (N, T, D)."""
        paths = {k: os.path.join(self.data_dir, v) for k, v in self.files.items()}

        # required
        for k in ["x_train", "x_test", "y_train", "y_test"]:
            if not os.path.exists(paths[k]):
                raise FileNotFoundError(f"Missing required file: {paths[k]}")

        X_tr = _load_pkl(paths["x_train"])
        X_te = _load_pkl(paths["x_test"])
        y_tr = _load_pkl(paths["y_train"])
        y_te = _load_pkl(paths["y_test"])

        X_tr, y_tr = self._prepare_xy(X_tr, y_tr, self.limit_train)
        X_te, y_te = self._prepare_xy(X_te, y_te, self.limit_test)

        # sanity
        if X_tr.ndim != 3 or X_te.ndim != 3:
            raise ValueError(f"X must be 3D (N,T,D). Got train {X_tr.shape}, test {X_te.shape}")
        if y_tr.ndim != 1 or y_te.ndim != 1:
            raise ValueError(f"y must be 1D. Got train {y_tr.shape}, test {y_te.shape}")

        return (X_tr, y_tr), (X_te, y_te)

    def load_aux(self) -> dict:
        """
        Returns auxiliary arrays including ground-truth importance for metrics:

        {
          "importance_train": np.ndarray (N_train, T, D),
          "importance_test":  np.ndarray (N_test,  T, D),
          "logits_train":     np.ndarray or list,
          "logits_test":      np.ndarray or list,
          "states_train":     any,
          "states_test":      any,
        }
        Missing files are simply omitted.
        """
        out = {}
        paths = {k: os.path.join(self.data_dir, v) for k, v in self.files.items()}

        def maybe_load(name, postproc=None):
            p = paths[name]
            if os.path.exists(p):
                arr = _load_pkl(p)
                if postproc is not None:
                    arr = postproc(arr)
                out[name] = arr

        # importance as (N,T,D), cast to float32 to match X
        def _prep_imp(arr):
            arr = self._maybe_transpose_TD(arr)
            return arr.astype("float32", copy=False) if self.cast_float32 else arr

        maybe_load("imp_train", postproc=_prep_imp)
        if "imp_train" in out:
            out["importance_train"] = out.pop("imp_train")

        maybe_load("imp_test", postproc=_prep_imp)
        if "imp_test" in out:
            out["importance_test"] = out.pop("imp_test")

        # logits & states (left as-is; shapes can vary by generator)
        maybe_load("logits_train")
        maybe_load("logits_test")
        maybe_load("states_train")
        maybe_load("states_test")

        return out

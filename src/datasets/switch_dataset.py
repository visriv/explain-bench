import os
import pickle
import numpy as np
from .base_dataset import BaseDataset
from ..utils.registry import Registry

def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@Registry.register_dataset("SwitchDataset")
class SwitchDataset(BaseDataset):
    """
    Loads time-series classification data saved as PKLs by your generator script.

    Expected files in `data_dir` (default names can be overridden via params):
      - state_dataset_x_train.pkl  -> np.ndarray, shape (N_train, T, D) or (N_train, D, T)
      - state_dataset_x_test.pkl   -> np.ndarray, shape (N_test,  T, D) or (N_test,  D, T)
      - state_dataset_y_train.pkl  -> np.ndarray, shape (N_train,)
      - state_dataset_y_test.pkl   -> np.ndarray, shape (N_test,)

    Optional (ignored by default, but kept for future metrics/analysis):
      - state_dataset_importance_train.pkl
      - state_dataset_importance_test.pkl
      - state_dataset_logits_train.pkl
      - state_dataset_logits_test.pkl
      - state_dataset_states_train.pkl
      - state_dataset_states_test.pkl

    Parameters
    ----------
    data_dir : str
        Folder where PKLs are stored.
    x_train_pkl, x_test_pkl, y_train_pkl, y_test_pkl : str
        Filenames of the core PKLs. Defaults match the generator in your screenshot.
    transpose_to_t_first : bool
        If True and input is (N, D, T), transpose to (N, T, D).
    cast_float32 : bool
        Cast X to float32.
    cast_int64 : bool
        Cast y to int64.
    limit_train, limit_test : int or None
        If set, truncate datasets (useful for quick smoke tests).
    """

    def __init__(
        self,
        data_dir="./data/simulated_switch_data",
        x_train_pkl="state_dataset_x_train.pkl",
        x_test_pkl="state_dataset_x_test.pkl",
        y_train_pkl="state_dataset_y_train.pkl",
        y_test_pkl="state_dataset_y_test.pkl",
        transpose_to_t_first=True,
        cast_float32=True,
        cast_int64=True,
        limit_train=None,
        limit_test=None,
    ):
        self.data_dir = data_dir
        self.paths = {
            "x_train": os.path.join(data_dir, x_train_pkl),
            "x_test":  os.path.join(data_dir, x_test_pkl),
            "y_train": os.path.join(data_dir, y_train_pkl),
            "y_test":  os.path.join(data_dir, y_test_pkl),
        }
        self.transpose_to_t_first = transpose_to_t_first
        self.cast_float32 = cast_float32
        self.cast_int64 = cast_int64
        self.limit_train = limit_train
        self.limit_test = limit_test

    def _maybe_transpose_TD(self, X):
        """
        Ensure shape is (N, T, D). If it looks like (N, D, T), transpose.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array for X, got shape {X.shape}")
        N, a, b = X.shape
        # Heuristic: if a < b, we assume a=T, b=D already; otherwise swap.
        # You can also remove heuristic and rely solely on self.transpose_to_t_first.
        if self.transpose_to_t_first and a > b:
            X = X.transpose(0, 2, 1)
        return X

    def _load_xy(self, x_path, y_path, limit=None):
        X = _load_pkl(x_path)
        y = _load_pkl(y_path)

        X = self._maybe_transpose_TD(X)

        if limit is not None:
            X = X[:limit]
            y = y[:limit]

        if self.cast_float32:
            X = X.astype("float32", copy=False)
        if self.cast_int64:
            y = y.astype("int64", copy=False)

        return X, y

    def load_aux(self):
        """Returns a dict of optional arrays; keys may include:
           importance_train, importance_test, logits_train, logits_test,
           states_train, states_test.
        """
        aux = {}
        def maybe(path, key, transpose=False):
            if os.path.exists(path):
                arr = _load_pkl(path)
                if transpose and arr.ndim == 3:
                    arr = self._maybe_transpose_TD(arr)
                aux[key] = arr

        # importance
        maybe(os.path.join(self.data_dir, "state_dataset_importance_train.pkl"),
              "importance_train", transpose=True)
        maybe(os.path.join(self.data_dir, "state_dataset_importance_test.pkl"),
              "importance_test", transpose=True)

        # optional extras (kept for future metrics)
        maybe(os.path.join(self.data_dir, "state_dataset_logits_train.pkl"), "logits_train")
        maybe(os.path.join(self.data_dir, "state_dataset_logits_test.pkl"), "logits_test")
        maybe(os.path.join(self.data_dir, "state_dataset_states_train.pkl"), "states_train")
        maybe(os.path.join(self.data_dir, "state_dataset_states_test.pkl"), "states_test")
        return aux
    
    def load(self):
        # Validate files exist
        for k, p in self.paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required file: {k} -> {p}")

        X_train, y_train = self._load_xy(self.paths["x_train"], self.paths["y_train"], self.limit_train)
        X_test,  y_test  = self._load_xy(self.paths["x_test"],  self.paths["y_test"],  self.limit_test)

        # Optional: sanity checks
        assert X_train.ndim == 3 and X_test.ndim == 3, "X must be (N, T, D)"
        assert y_train.ndim == 1 and y_test.ndim == 1, "y must be (N,)"

        return (X_train, y_train), (X_test, y_test)

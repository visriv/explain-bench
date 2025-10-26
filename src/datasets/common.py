# explainbench/datasets/common.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Iterable
import numpy as np
import torch
from torch.utils.data import Dataset

Array = np.ndarray

@dataclass
class SplitData:
    """Container for a dataset split set (train/val/test)."""
    # Regular shape: X: (N, T, D), y: (N,), times: Optional (N, T) or (T,) or None
    train: Tuple[Array, Optional[Array], Array]
    val:   Tuple[Array, Optional[Array], Array]
    test:  Tuple[Array, Optional[Array], Array]
    # Optional ground truth importance (N, T, D)
    gt: Optional[Dict[str, Array]] = None
    # Optional misc for model-specific needs (e.g., class names)
    meta: Optional[Dict[str, Any]] = None

class RegularTSDataset(Dataset):
    """PyTorch dataset for (N, T, D) [+ optional times (N, T)]."""
    def __init__(self, X: Array, y: Array, times: Optional[Array] = None):
        self.X = X
        self.y = y
        self.times = times

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]          # (T, D)
        y = self.y[idx]          # ()
        if self.times is None:
            return torch.from_numpy(x), None, torch.as_tensor(y, dtype=torch.long)
        else:
            t = self.times[idx]  # (T,) or (T,1)
            return torch.from_numpy(x), torch.from_numpy(t), torch.as_tensor(y, dtype=torch.long)

class ChunkedMVTS2Regular(Dataset):
    """
    Wraps Timex++ / MVTS 'chunked' format:  X: (T, N, D), times: (T, N), y: (N,)
    and exposes samples as (x_i: (T, D), t_i: (T,), y_i).
    """
    def __init__(self, X: torch.Tensor, times: torch.Tensor, y: torch.Tensor):
        # Expect torch tensors from Timex preprocess; move to CPU np arrays once
        assert X.dim() == 3 and times.dim() == 2 and y.dim() == 1
        self.T, self.N, self.D = X.shape
        self.X = X.cpu().permute(1, 0, 2).contiguous().numpy()     # (N, T, D)
        self.times = times.cpu().permute(1, 0).contiguous().numpy() # (N, T)
        self.y = y.cpu().numpy()

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        x = self.X[idx]          # (T, D)
        t = self.times[idx]      # (T,)
        y = self.y[idx]          # ()
        return torch.from_numpy(x), torch.from_numpy(t), torch.as_tensor(y, dtype=torch.long)

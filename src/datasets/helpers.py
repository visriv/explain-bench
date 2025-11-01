from __future__ import annotations
import importlib
import numpy as np
import torch
from .base_dataset import BaseDataset
from ..utils.registry import Registry

def _import(module, name):
    mod = importlib.import_module(module)
    return getattr(mod, name)

def _TNDtoNTD(x: torch.Tensor | np.ndarray) -> np.ndarray:
    # Convert (T,N,D) -> (N,T,D)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert x.ndim == 3, f"expected 3D, got {x.shape}"
    # if x.shape[0] <= x.shape[1]:  # (T,N,D)
    x = np.transpose(x, (1,0,2))
    return x.astype("float32", copy=False)

def _TNtoNT(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if x is None: return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert x.ndim == 2
    # if x.shape[0] <= x.shape[1]:  # (T,N) -> (N,T)
    x = np.transpose(x, (1,0))
    return x.astype("float32", copy=False)
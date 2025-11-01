# src/datasets/pam.py
from __future__ import annotations
import torch
import numpy as np
from .base_dataset import BaseDataset
from ..utils.registry import Registry
from .helpers import _TNDtoNTD, _TNtoNT     # you already have these
from .handlers.process_realdata import process_PAM



@Registry.register_dataset("PAM")
class PAM(BaseDataset):
    """
    Real-world PAM dataset loader.
    Returns numpy arrays in (N,T,D) and times in (N,T),
    identical output schema to FreqShape.load_splits().
    """
    def __init__(self, split_no: int = 1, base_path: str = "./data/PAM/", gethalf: bool = False):
        self.split_no = split_no
        self.base_path = base_path
        self.gethalf = gethalf
        self.task = "classification"
        # Timex reports 8 classes for PAM
        self.n_classes = 8

    def load_splits(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_chunk, val_chunk, test_chunk = process_PAM(
            split_no=self.split_no, device=device, base_path=self.base_path, gethalf=self.gethalf
        )

        # Convert (T,N,D) → (N,T,D), (T,N) → (N,T), y -> numpy int64
        Xtr = _TNDtoNTD(train_chunk.X)
        Ttr = _TNtoNT(train_chunk.time)
        ytr = train_chunk.y.detach().cpu().numpy().astype("int64")

        Xv  = _TNDtoNTD(val_chunk.X)
        Tv  = _TNtoNT(val_chunk.time)
        yv  = val_chunk.y.detach().cpu().numpy().astype("int64")

        Xte = _TNDtoNTD(test_chunk.X)
        Tte = _TNtoNT(test_chunk.time)
        yte = test_chunk.y.detach().cpu().numpy().astype("int64")

        # PAM does not ship GT explanation masks; return None
        gt = None

        return (Xtr, ytr, Ttr), (Xv, yv, Tv), (Xte, yte, Tte), gt

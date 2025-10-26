from __future__ import annotations
import importlib
import numpy as np
import torch
from .base_dataset import BaseDataset
from ..utils.registry import Registry
from .helpers import _TNDtoNTD, _TNtoNT

@Registry.register_dataset("PAM")
class PAM(BaseDataset):
    """Loads PAM via Timex++ process_PAM; returns (N,T,D) arrays."""
    def __init__(self, split_no=1, base_path="./data/PAM/", gethalf=True):
        self.split_no = split_no
        self.base_path = base_path
        self.gethalf = gethalf

    def load_splits(self):
        process_PAM = _import("txai.utils.data.preprocess", "process_PAM")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        trainPAM, val, test = process_PAM(split_no=self.split_no, device=device, base_path=self.base_path, gethalf=self.gethalf)

        Xtr = _TNDtoNTD(trainPAM.X);  Ttr = _TNtoNT(trainPAM.time); ytr = trainPAM.y.detach().cpu().numpy().astype('int64')
        Xv  = _TNDtoNTD(val.X);       Tv  = _TNtoNT(val.time);     yv  = val.y.detach().cpu().numpy().astype('int64')
        Xte = _TNDtoNTD(test.X);      Tte = _TNtoNT(test.time);    yte = test.y.detach().cpu().numpy().astype('int64')

        return (Xtr, ytr, Ttr), (Xv, yv, Tv), (Xte, yte, Tte), None

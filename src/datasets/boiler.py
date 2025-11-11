# src/datasets/boiler.py
from __future__ import annotations
import torch
import numpy as np
from .base_dataset import BaseDataset
from ..utils.registry import Registry
from .helpers import _TNDtoNTD, _TNtoNT     # you already have these
from .handlers.process_realdata import process_Boiler_OLD



@Registry.register_dataset("Boiler")
class Boiler(BaseDataset):
    """
    Real-world Boiler dataset loader.
    Returns numpy arrays in (N,T,D) and times in (N,T),
    identical output schema to FreqShape.load_splits().
    """
    def __init__(self, split_no: int = 1, base_path: str = "./data/Boiker/"):
        self.split_no = split_no
        self.base_path = base_path
        self.task = "classification"
        self.n_classes = 2

    def load_splits(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # process_Boiler_OLD returns: trainB (tuple), val (tuple or obj), test (tuple or obj)
        trainB, val_chunk, test_chunk = process_Boiler_OLD(
            split_no=self.split_no, device=device, base_path=self.base_path
        )

        # TRAIN
        Xtr = _TNDtoNTD(trainB[0])
        ytr = trainB[2].detach().cpu().numpy().astype("int64")
        Ttr = _TNtoNT(trainB[1])

        # VAL
        Xv = _TNDtoNTD(val_chunk[0])
        yv = val_chunk[2].detach().cpu().numpy().astype("int64")
        Tv = _TNtoNT(val_chunk[1])

        # TEST
        Xte = _TNDtoNTD(test_chunk[0])
        yte = test_chunk[2].detach().cpu().numpy().astype("int64")
        Tte = _TNtoNT(test_chunk[1])

        gt = None  # no ground-truth explanation masks for Boiler

        return (Xtr, ytr, Ttr), (Xv, yv, Tv), (Xte, yte, Tte), gt

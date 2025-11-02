from __future__ import annotations
import importlib
import numpy as np
import torch
from .base_dataset import BaseDataset
from ..utils.registry import Registry
from .helpers import _import, _TNDtoNTD, _TNtoNT
from src.datasets.handlers.process_synth import process_Synth

@Registry.register_dataset("SeqCombUV")
class SeqCombUV(BaseDataset):
    """Loads SeqCombUV via Timex++ process_Synth and returns numpy arrays in (N,T,D)."""
    def __init__(self, split_no=1, base_path="./data/SeqCombSingleBetter/"):
        self.split_no = split_no
        self.base_path = base_path

    def load_splits(self):
        # process_Synth = _import("txai.utils.data", "process_Synth")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        D = process_Synth(split_no=self.split_no, device=device, base_path=self.base_path)

        tr = D['train_loader']
        val = D['val']
        tes = D['test']

        Xtr = _TNDtoNTD(tr.X);  Ttr = _TNtoNT(tr.times); ytr = tr.y.detach().cpu().numpy().astype('int64')
        Xv  = _TNDtoNTD(val[0]); Tv  = _TNtoNT(val[1]);   yv  = val[2].detach().cpu().numpy().astype('int64')
        Xte = _TNDtoNTD(tes[0]); Tte = _TNtoNT(tes[1]);   yte = tes[2].detach().cpu().numpy().astype('int64')

        gt = None
        if 'gt_exps' in D and D['gt_exps'] is not None:
            ge = _TNDtoNTD(D['gt_exps'])
            gt = {'importance_train': ge[:Xtr.shape[0]]}  # at least provide train GT if aligned

        return (Xtr, ytr, Ttr), (Xv, yv, Tv), (Xte, yte, Tte), gt

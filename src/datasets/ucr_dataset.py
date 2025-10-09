# Placeholder for UCR/UEA loader. Implement download/cache as needed.
import numpy as np
from .base_dataset import BaseDataset
from ..utils.registry import Registry

@Registry.register_dataset("UCRDataset")
class UCRDataset(BaseDataset):
    def __init__(self, name="ECG200"):
        self.name=name

    def load(self):
        # TODO: Implement actual UCR dataset loading.
        # For now, return synthetic-shaped data.
        Xtr = np.random.randn(100, 96, 1).astype("float32")
        ytr = np.random.randint(0, 2, size=(100,)).astype("int64")
        Xte = np.random.randn(50, 96, 1).astype("float32")
        yte = np.random.randint(0, 2, size=(50,)).astype("int64")
        return (Xtr,ytr),(Xte,yte)

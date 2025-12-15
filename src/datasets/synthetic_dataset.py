import numpy as np
from .base_dataset import BaseDataset
from ..utils.registry import Registry
from .common import SplitData
@Registry.register_dataset("SyntheticDataset")
class SyntheticDataset(BaseDataset):
    """Simple sinusoid + noise dataset for classification."""
    def __init__(self, n_train=800, n_test=200, length=100, features=6, n_classes=2, task="classification", seed=42):
        self.n_train=n_train; self.n_test=n_test; self.length=length
        self.features=features; self.n_classes=n_classes; self.task=task
        self.rng = np.random.default_rng(seed)

    def _make(self, n):
        X = self.rng.normal(0, 0.3, size=(n, self.length, self.features))
        y = self.rng.integers(0, self.n_classes, size=(n,))
        # imprint class-dependent sinusoid on first feature
        t = np.linspace(0, 2*np.pi, self.length)
        for i in range(n):
            X[i,:,0] += (y[i]+1)*0.4*np.sin(2*t)  # stronger amplitude for higher class
        return X.astype("float32"), y.astype("int64")

    def load(self):
        return self._make(self.n_train), self._make(self.n_test)
    

    def load_splits(self):
        pass
    
    def load_full(self) -> SplitData:
        (Xtr, ytr), (Xte, yte) = self.load()
        # simple 80/20 split of train as val
        n = Xtr.shape[0]
        k = int(n * 0.8)
        Xv, yv = Xtr[k:], ytr[k:]
        Xtr, ytr = Xtr[:k], ytr[:k]
        return SplitData(
            train=(Xtr, None, ytr),
            val=(Xv, None, yv),
            test=(Xte, None, yte),
            gt=None,
            meta={"n_classes": self.n_classes, "input_dim": self.features, "max_len": self.length}
        )

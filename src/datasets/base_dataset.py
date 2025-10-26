from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """Minimal interface for all datasets."""
    @abstractmethod
    def load_splits(self):
        """Return (train, val, test, gt) where each split is (X, y, times or None).
        Shapes:
          X: (N, T, D)  y: (N,)  times: (N, T) or None
        gt: dict with optional keys like 'importance_train', 'importance_val', 'importance_test' (each (N,T,D)).
        """
        raise NotImplementedError

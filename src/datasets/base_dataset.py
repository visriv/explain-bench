from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """Interface for datasets. Should return (X_train, y_train), (X_test, y_test)."""
    @abstractmethod
    def load(self):
        pass

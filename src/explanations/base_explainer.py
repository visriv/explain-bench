from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    name = "BaseExplainer"
    @abstractmethod
    def explain(self, model, X):
        """Return attributions shaped like X [N, T, D] or [N, D, T]."""
        ...

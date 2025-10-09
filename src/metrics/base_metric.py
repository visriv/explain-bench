from abc import ABC, abstractmethod

class BaseMetric(ABC):
    name = "BaseMetric"

    @abstractmethod
    def compute(self, attributions, model, X, y, gt=None):
        """Return dict of metric_name->value.

        Parameters
        ----------
        attributions : np.ndarray  (N, T, D)
        model        : model wrapper
        X            : np.ndarray  (N, T, D)
        y            : np.ndarray  (N,)
        gt           : Optional[np.ndarray] (N, T, D) ground-truth importance
        """
        ...

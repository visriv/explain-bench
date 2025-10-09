from abc import ABC, abstractmethod

class BaseMetric(ABC):
    name = "BaseMetric"
    @abstractmethod
    def compute(self, attributions, model, X, y):
        """Return dict of metric_name->value"""
        ...

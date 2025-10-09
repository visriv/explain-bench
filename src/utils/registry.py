class Registry:
    """Simple name->class registry for datasets/models/explainers/metrics."""
    _datasets = {}
    _models = {}
    _explainers = {}
    _metrics = {}

    @classmethod
    def register_dataset(cls, name):
        def deco(kls):
            cls._datasets[name] = kls
            return kls
        return deco

    @classmethod
    def register_model(cls, name):
        def deco(kls):
            cls._models[name] = kls
            return kls
        return deco

    @classmethod
    def register_explainer(cls, name):
        def deco(kls):
            cls._explainers[name] = kls
            return kls
        return deco

    @classmethod
    def register_metric(cls, name):
        def deco(kls):
            cls._metrics[name] = kls
            return kls
        return deco

    @classmethod
    def get_dataset(cls, name): return cls._datasets[name]
    @classmethod
    def get_model(cls, name): return cls._models[name]
    @classmethod
    def get_explainer(cls, name): return cls._explainers[name]
    @classmethod
    def get_metric(cls, name): return cls._metrics[name]

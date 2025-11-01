from __future__ import annotations
import yaml
from .utils.logging_utils import ensure_dir
from .utils.registry import Registry
from . import datasets  # ensure registries are populated
from . import models, explanations, metrics
from typing import Any, Dict, List

class BenchmarkRunner:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def _build_dataset(self):
        dname = self.cfg['dataset']['name']
        dparams = self.cfg['dataset'].get('params', {})
        D = Registry.get_dataset(dname)
        return D(**dparams)

    def _infer_dims(self, dataset):
        (Xtr, ytr, _), *_ = dataset.load_splits()
        return Xtr.shape[-1], int(ytr.max() + 1)

    def _instantiate_model(self, name: str, params: Dict[str, Any], input_dim: int, num_classes: int):
        M = Registry.get_model(name)
        p = dict(params or {})
        p.setdefault('input_dim', input_dim)
        p.setdefault('num_classes', num_classes)
        m = M(**p)
        if not hasattr(m, "name"):
            try:
                m.name = name
            except Exception:
                pass
        return m

    def _build_models(self, dataset):
        input_dim, num_classes = self._infer_dims(dataset)

        if 'models' in self.cfg:
            cfg_models = self.cfg['models']
            # list form: - {name: "...", params: {...}}
            if isinstance(cfg_models, list):
                return [
                    self._instantiate_model(item['name'], item.get('params', {}), input_dim, num_classes)
                    for item in cfg_models
                ]
            # dict form: ModelName: {params...}
            if isinstance(cfg_models, dict):
                return {
                    name: self._instantiate_model(name, params or {}, input_dim, num_classes)
                    for name, params in cfg_models.items()
                }
            raise ValueError("Unsupported 'models' format; use list of {name,params} or dict name->params.")
        else:
            # single model (backward compatible)
            mname = self.cfg['model']['name']
            mparams = self.cfg['model'].get('params', {})
            return self._instantiate_model(mname, mparams, input_dim, num_classes)

    def run(self):
        dataset = self._build_dataset()
        models_obj = self._build_models(dataset)

        explainers = [Registry.get_explainer(n)() for n in self.cfg['explainers']]
        metrics = [Registry.get_metric(n)() for n in self.cfg['metrics']]

        outdir = self.cfg.get('output_dir', 'runs')
        ensure_dir(outdir)

        from .benchmark import Benchmark
        bm = Benchmark(dataset, models_obj, explainers, metrics, outdir)
        rows = bm.run()
        print('Results:', rows)

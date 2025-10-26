from __future__ import annotations
import yaml
from .benchmark import Benchmark
from .utils.logging_utils import ensure_dir
from .utils.registry import Registry
from . import datasets
from . import models   
from . import explanations
from . import metrics  


class BenchmarkRunner:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def _build_dataset(self):
        dname = self.cfg['dataset']['name']
        dparams = self.cfg['dataset'].get('params', {})
        D = Registry.get_dataset(dname)
        return D(**dparams)

    def run(self):
        dataset = self._build_dataset()

        mname = self.cfg['model']['name']
        mparams = self.cfg['model'].get('params', {})
        M = Registry.get_model(mname)
        # infer dims if missing
        if 'input_dim' not in mparams or 'num_classes' not in mparams:
            (Xtr, ytr, _), *_ = dataset.load_splits()
            mparams.setdefault('input_dim', Xtr.shape[-1])
            mparams.setdefault('num_classes', int(ytr.max() + 1))
        model = M(**mparams)

        explainers = [Registry.get_explainer(n)() for n in self.cfg['explainers']]
        metrics = [Registry.get_metric(n)() for n in self.cfg['metrics']]

        outdir = self.cfg.get('output_dir', 'results')
        ensure_dir(outdir)

        bm = Benchmark(dataset, model, explainers, metrics, outdir)
        rows = bm.run()
        print('Results:', rows)

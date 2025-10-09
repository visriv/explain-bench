from .utils.config_loader import load_config
from .utils.registry import Registry
from .benchmark import Benchmark
from .utils.logging_utils import ensure_dir

class BenchmarkRunner:
    def __init__(self, config_path):
        self.cfg = load_config(config_path)

    def run(self):
        ds_cls = Registry.get_dataset(self.cfg["dataset"]["name"])
        dataset = ds_cls(**self.cfg["dataset"]["params"])

        m_cls = Registry.get_model(self.cfg["model"]["name"])
        model = m_cls(**self.cfg["model"]["params"])

        explainers = [Registry.get_explainer(name)() for name in self.cfg["explainers"]]
        metrics = [Registry.get_metric(name)() for name in self.cfg["metrics"]]

        outdir = self.cfg.get("output_dir", "results")
        ensure_dir(outdir)
        bm = Benchmark(dataset, model, explainers, metrics, outdir)
        rows = bm.run()
        print("Results:", rows)

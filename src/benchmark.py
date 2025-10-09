import time, numpy as np
from .visualization.report import write_csv

class Benchmark:
    def __init__(self, dataset, model, explainers, metrics, output_dir):
        self.dataset = dataset
        self.model = model
        self.explainers = explainers
        self.metrics = metrics
        self.output_dir = output_dir

    def run(self):
        (X_train, y_train), (X_test, y_test) = self.dataset.load()
        # fit once
        self.model.fit(X_train, y_train)

        if hasattr(self.dataset, "load_aux"):
            aux = self.dataset.load_aux() or {}
        gt_test = aux.get("importance_test", None)

        rows = []
        for explainer in self.explainers:
            t0 = time.time()
            attributions = explainer.explain(self.model, X_test)  # expect [N, T, D]
            elapsed = time.time() - t0

            # Ensure shape [N, T, D]
            if attributions.shape[1] != X_test.shape[1]:
                # try to permute if [N, D, T]
                attributions = np.transpose(attributions, (0,2,1))

            metric_vals = {}
            for m in self.metrics:
                metric_vals.update(m.compute(attributions, self.model, X_test, y_test, gt=gt_test))

            row = {
                "explainer": getattr(explainer, "name", explainer.__class__.__name__),
                "time_sec": elapsed,
                **metric_vals
            }
            rows.append(row)

        # save
        write_csv(f"{self.output_dir}/benchmark_results.csv", rows, fieldnames=list(rows[0].keys()))
        return rows

from __future__ import annotations
import time, numpy as np

class Benchmark:
    def __init__(self, dataset, model, explainers, metrics, output_dir):
        self.dataset = dataset
        self.model = model
        self.explainers = explainers
        self.metrics = metrics
        self.output_dir = output_dir

    def run(self):
        (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _), gt = self.dataset.load_splits()
        self.model.fit(Xtr, ytr)

        rows = []
        gt_test = None if not gt else gt.get("importance_test") or gt.get("importance_train")  # fallback

        for explainer in self.explainers:
            t0 = time.time()
            attributions = explainer.explain(self.model, Xte)
            elapsed = time.time() - t0
            if attributions.shape[1] != Xte.shape[1]:
                attributions = np.transpose(attributions, (0,2,1))

            mvals = {}
            for m in self.metrics:
                mvals.update(m.compute(attributions, self.model, Xte, yte, gt=gt_test))

            rows.append({"explainer": getattr(explainer, "name", explainer.__class__.__name__),
                         "time_sec": elapsed, **mvals})
        return rows

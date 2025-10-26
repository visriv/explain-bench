from __future__ import annotations
import time
import numpy as np
import logging
from typing import List, Dict, Any

# --- simple logging setup (idempotent) ---
_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
log = logging.getLogger("ExplainBench")

def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        return "N/A"
    
def _model_name(m):
    return getattr(m, "name", None) or getattr(m, "__class__", type(m)).__name__

def _as_model_list(model_or_models) -> List[Any]:
    """
    Accept:
      - single model instance
      - list/tuple of model instances
      - dict name->model (optional)
    Return a list of (name, instance).
    """
    if isinstance(model_or_models, dict):
        return [(str(k), v) for k, v in model_or_models.items()]
    if isinstance(model_or_models, (list, tuple)):
        out = []
        for m in model_or_models:
            out.append((_model_name(m), m))
        return out
    # single instance
    m = model_or_models
    return [(_model_name(m), m)]


class Benchmark:
    def __init__(self, dataset, models, explainers, metrics, output_dir):
        self.dataset = dataset
        self.models = models
        self.explainers = explainers
        self.metrics = metrics
        self.output_dir = output_dir

    def run(self) -> List[Dict[str, Any]]:
        log.info("⚙️  Loading dataset splits…")
        t0 = time.time()
        (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _), gt = self.dataset.load_splits()
        log.info("✅ Dataset loaded in %.2fs", time.time() - t0)
        # log.info("   • train: X %s, y %s", _shape(Xtr), _shape(ytr))
        # log.info("   •   val: X %s, y %s", _shape(Xv),  _shape(yv))
        # log.info("   •  test: X %s, y %s", _shape(Xte), _shape(yte))
        if gt:
            gt_keys = ", ".join(sorted(gt.keys()))
            log.info("   • GT available")
        else:
            log.info("   • GT not available")


        rows = []
        gt_test = None if not gt else (gt.get("importance_test") or gt.get("importance_train"))
        if gt_test is not None:
            log.info("   • Using GT for metrics: %s", "importance_test" if "importance_test" in gt else "importance_train")

        mm = _as_model_list(self.models)
        print(mm)

         # ---- iterate over one or many models ----
        for mname, mdl in _as_model_list(self.model):
            log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            log.info("🤖 Fitting model: %s", mname)
            t0 = time.time()
            mdl.fit(Xtr, ytr)
            log.info("✅ Model fit complete")
            mdl.validate(Xv, yv)
            log.info("✅ Model validation complete")

            # Explain + evaluate
            log.info("🔍 Running %d explainer(s) and %d metric(s)…", len(self.explainers), len(self.metrics))
            for explainer in self.explainers:
                expl_name = getattr(explainer, "name", explainer.__class__.__name__)
                log.info("   ▶ Explainer: %s — generating attributions…", expl_name)
                t_expl = time.time()
                attributions = explainer.explain(self.model, Xte)
                expl_elapsed = time.time() - t_expl
                log.info("     • Done in %.2fs; attr shape %s", expl_elapsed, _shape(attributions))

                # Ensure (N,T,D)
                if attributions.shape[1] != Xte.shape[1]:
                    attributions = np.transpose(attributions, (0, 2, 1))

                # Metrics
                metric_vals: Dict[str, Any] = {}
                for m in self.metrics:
                    mname = getattr(m, "name", m.__class__.__name__)
                    log.info("     • Metric: %s …", mname)
                    t_m = time.time()
                    vals = m.compute(attributions, self.model, Xte, yte, gt=gt_test)
                    metric_time = time.time() - t_m
                    # log key → value (rounded if scalar)
                    pretty = {k: (float(v) if np.isscalar(v) else v) for k, v in vals.items()}
                    log.info("       -> %s (%.2fs)", ", ".join(f"{k}={pretty[k]:.4f}" if isinstance(pretty[k], (int,float)) else f"{k}={pretty[k]}" for k in pretty), metric_time)
                    metric_vals.update(vals)

                row = {
                    "explainer": expl_name,
                    "time_sec": expl_elapsed,
                    **metric_vals
                }
                rows.append(row)

        # Summary
        log.info("📄 Benchmark complete. Returning %d row(s).", len(rows))
        for r in rows:
            expl = r.pop("explainer")
            tsec = r.pop("time_sec")
            kv = ", ".join(f"{k}={float(v):.4f}" if isinstance(v, (int, float, np.floating)) else f"{k}={v}" for k, v in r.items())
            log.info("   • %s: time=%.2fs | %s", expl, tsec, kv)

        return rows

from __future__ import annotations
import time
import numpy as np
import logging
from typing import List, Dict, Any
from src.utils.train_utils import train_classifier, validate_classifier
from src.utils.path_utils import *
import os, json, time, re

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
      - dict name->model
    Return a list of (name, instance).
    """
    if isinstance(model_or_models, dict):
        return [(str(k), v) for k, v in model_or_models.items()]
    if isinstance(model_or_models, (list, tuple)):
        return [(_model_name(m), m) for m in model_or_models]
    return [(_model_name(model_or_models), model_or_models)]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _flat_with_prefix(prefix: str, d: dict) -> dict:
    """Turn {'a':1,'b':2} into {f'{prefix}a':1, f'{prefix}b':2} (None -> '')."""
    if not isinstance(d, dict):
        # print('reached here')
        # print(prefix)
        return {}
    out = {}
    for k, v in d.items():
        out[f"{prefix}{k}"] = "" if v is None else v
    return out

def _ensure_tsv_header(tsv_path: str, new_cols: list[str]) -> list[str]:
    """
    Ensure TSV has a header that includes all `new_cols`.
    - If file doesn't exist: create with these columns.
    - If exists: append any missing columns to the header (rewrite only the first line).
    Returns the final ordered list of columns.
    """
    if not os.path.isfile(tsv_path):
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("\t".join(new_cols) + "\n")
        return new_cols

    # Read existing header
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        lines = [""]

    old_cols = lines[0].split("\t") if lines[0] else []
    add_cols = [c for c in new_cols if c not in old_cols]
    if add_cols:
        lines[0] = "\t".join(old_cols + add_cols)
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines and lines[-1] != "" else ""))
        return old_cols + add_cols
    return old_cols


def dump_json(obj: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

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
        # build nested run root: runs/<DATA_DIR>/...
        data_dir_rel, data_cfg = dataset_signature(self.dataset)
        run_root = os.path.join(self.output_dir, data_dir_rel)
        ensure_dir(run_root)
        print('RUN ROOT: ', run_root)
        log.info("✅ Dataset loaded in %.2fs", time.time() - t0)

        tsv_path = os.path.join(run_root, "results.tsv")
        if not os.path.isfile(tsv_path):
            with open(tsv_path, "w", encoding="utf-8") as f:
                f.write("\t".join([
                    "data", "data_params",
                    "model", "model_params",
                    "explainer", "explainer_params",
                    "time_sec", "metrics_json"
                ]) + "\n")

        if gt:
            log.info("   • GT available")
        else:
            log.info("   • GT not available")

        rows = []
        gt_test = None if not gt else (gt.get("importance_test") or gt.get("importance_train"))
        if gt_test is not None:
            log.info("   • Using GT for metrics: %s",
                     "importance_test" if "importance_test" in gt else "importance_train")

        # ---- iterate over one or many models ----
        for mname, mdl in _as_model_list(self.models):
            log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            # per-model directory: runs/<DATA>/<MODEL>/<HPs>/...
            model_dir_rel, model_cfg = model_dir_name(mdl)
            model_dir = os.path.join(run_root, model_dir_rel)
            ensure_dir(model_dir)
            ckpt_path = os.path.join(model_dir, "checkpoint.pt")

            # snapshot (once) for reproducibility
            dump_json({"dataset": data_cfg, "model": model_cfg}, os.path.join(model_dir, "config.json"))

            if os.path.isfile(ckpt_path):
                log.info(f"[model] found checkpoint for {mname} under '{data_dir_rel}', loading (skip training)…")
                _ = load_checkpoint_if_exists(mdl, ckpt_path)
            else:
                log.info(f"[model] training {mname} on dataset '{data_dir_rel}'")
                t0 = time.time()
                train_classifier(mdl, Xtr, ytr, Xva=Xv, yva=yv, logger=log)
                log.info(f"[model] training complete in {time.time() - t0:.2f}s")
                save_checkpoint(mdl, ckpt_path, meta={"data_dir": data_dir_rel, "model_dir": model_dir_rel})
                log.info(f"[model] saved checkpoint → {ckpt_path}")

            # quick sanity validation after load/train
            val_out = validate_classifier(mdl, Xv, yv, logger=log)
            print(val_out)



            # Explain + evaluate
            log.info("🔍 Running %d explainer(s) and %d metric(s)…",
                     len(self.explainers), len(self.metrics))
            for explainer in self.explainers:
                # per-explainer directory: runs/<DATA>/<MODEL>/<EXPL>/<params>/...
                expl_dir_rel, expl_cfg = expl_dir_name(explainer)
                expl_dir = os.path.join(model_dir, expl_dir_rel)
                ensure_dir(expl_dir)

                # save explainer config
                dump_json({"dataset": data_cfg, "model": model_cfg, "explainer": expl_cfg},
                        os.path.join(expl_dir, "config.json"))

                # explain
                expl_name = getattr(explainer, "name", explainer.__class__.__name__)
                log.info("   ▶ Explainer: %s — generating attributions…", expl_name)
                t_expl = time.time()
                attributions = explainer.explain(mdl, Xte)
                expl_elapsed = time.time() - t_expl
                log.info("     • Done in %.2fs; attr shape %s", expl_elapsed, _shape(attributions))

                # Ensure (N,T,D)
                if attributions.shape[1] != Xte.shape[1]:
                    attributions = np.transpose(attributions, (0, 2, 1))

                # Metrics
                metric_vals: Dict[str, Any] = {}
                for m in self.metrics:
                    mname_metric = getattr(m, "name", m.__class__.__name__)
                    log.info("     • Metric: %s …", mname_metric)
                    t_m = time.time()
                    vals = m.compute(attributions, mdl, Xte, yte, gt=gt_test)
                    metric_time = time.time() - t_m
                    pretty = {k: (float(v) if np.isscalar(v) else v) for k, v in vals.items()}
                    log.info("       -> %s (%.2fs)",
                             ", ".join(f"{k}={pretty[k]:.4f}" if isinstance(pretty[k], (int, float))
                                       else f"{k}={pretty[k]}" for k in pretty),
                             metric_time)
                    metric_vals.update(vals)

                    # --- build a flat row: core identifiers + flattened params + flattened metrics
                    flat_row = {
                        "data": data_dir_rel,
                        "model": mname,
                        "explainer": expl_name,
                        "time_sec": f"{expl_elapsed:.6f}",
                    }
                    flat_row.update(_flat_with_prefix("data_", data_cfg))       # e.g., data_base_path, data_split_no
                    flat_row.update(_flat_with_prefix("model_", model_cfg))     # e.g., model_lr, model_epochs, model_batch_size
                    flat_row.update(_flat_with_prefix("expl_", expl_cfg))       # e.g., expl_sigma, expl_samples
                    flat_row.update(_flat_with_prefix("metric_", metric_vals))  # e.g., metric_faithfulness_drop, metric_consistency_cos
                    flat_row.update(_flat_with_prefix("", val_out))  # e.g., pr, auprc, auroc

                    # Establish TSV path once per dataset
                    tsv_path = os.path.join(run_root, "results.tsv")

                    # Ensure header includes all current columns (append missing ones in-place)
                    header = _ensure_tsv_header(tsv_path, list(flat_row.keys()))

                    # Write the row in header order (fill missing keys with "")
                    with open(tsv_path, "a", encoding="utf-8") as f:
                        f.write("\t".join(str(flat_row.get(col, "")) for col in header) + "\n")
                    print('written metrics to {run_root}/{tsv_path}')

                rows.append({
                    "model": mname,
                    "explainer": expl_name,
                    "time_sec": expl_elapsed,
                    **metric_vals
                })

        # Summary
        log.info("📄 Benchmark complete. Returning %d row(s).", len(rows))
        for r in rows:
            model_name = r.get("model")
            expl = r.get("explainer")
            tsec = r.get("time_sec")
            kv = {k: v for k, v in r.items() if k not in ("model", "explainer", "time_sec")}
            kvs = ", ".join(
                f"{k}={float(v):.4f}" if isinstance(v, (int, float, np.floating)) else f"{k}={v}"
                for k, v in kv.items()
            )
            log.info("   • [%s] %s: time=%.2fs | %s", model_name, expl, tsec, kvs)

        return rows

from __future__ import annotations
import time
import numpy as np
import logging
from typing import List, Dict, Any
from src.utils.train_utils import train_classifier, validate_classifier
from src.utils.path_utils import *
from src.utils.plot_samples import save_sample_plots, plot_explainer_samples
import os, json, time, re, pickle
from torch.utils.data import TensorDataset, DataLoader
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
        if v is None:
            out[f"{prefix}{k}"] = ""
        elif isinstance(v, (list, tuple)):
            out[f"{prefix}{k}"] = ",".join(str(x) for x in v)
        else:
            out[f"{prefix}{k}"] = v
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
    def __init__(self, 
                 dataset, 
                 models, 
                 config, 
                 explainers, 
                 metrics, 
                 output_dir):
        self.dataset = dataset
        self.models = models
        self.config = config
        self.explainers = explainers
        self.metrics = metrics
        self.output_dir = output_dir

    def run(self) -> List[Dict[str, Any]]:



        log.info("⚙️  Loading dataset splits…")
        t0 = time.time()
        (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _), gt = self.dataset.load_splits()
        # build nested run root: runs/<DATA_DIR>/...
        data_dir_rel, _ = dataset_signature(self.dataset)
        data_cfg = self.config['dataset']
        run_root = os.path.join(self.output_dir, data_dir_rel)
        ensure_dir(run_root)
        print('RUN ROOT: ', run_root)
        log.info("✅ Dataset loaded in %.2fs", time.time() - t0)

        tsv_path = os.path.join(run_root, "results.tsv")
        if not os.path.isfile(tsv_path):
            with open(tsv_path, "w", encoding="utf-8") as f:
                f.write("\t".join([
                    "data", 
                    "model", 
                    "explainer"
                ]) + "\n")

        if gt:
            log.info("   • GT available")
        else:
            log.info("   • GT not available")

        rows = []
        gt_test = None if not gt else (gt.get("importance_test"))# or gt.get("importance_train"))
        if gt_test is not None:
            log.info("   • Using GT for metrics: %s",
                     "importance_test" if "importance_test" in gt else "importance_train")


        plot_dir = os.path.join(run_root, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        save_sample_plots(Xtr, split_name="train", outdir=plot_dir)
        save_sample_plots(Xte, split_name="test", outdir=plot_dir, gt = gt_test )


        # ---- iterate over one or many models ----
        for mname, mdl in _as_model_list(self.models):
            log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            # per-model directory: runs/<DATA>/<MODEL>/<HPs>/...
            model_dir_rel, _ = model_dir_name(mdl)
            model_cfg = next((m for m in self.config['models'] if m.get("name") == mname), {})

            model_dir = os.path.join(run_root, model_dir_rel)
            ensure_dir(model_dir)
            ckpt_path = os.path.join(model_dir, "checkpoint.pt")

            # Match config entry by model name
            single_label = bool(model_cfg.get("single_label", True))

            # Handle single_label if dataset labels are temporal
            if single_label and ytr.ndim == 2:
                log.info(f"[model] Using single-label mode (last timestep) for {mname}")
                ytr = ytr[:, -1]
                yv  = yv[:,  -1]
                yte = yte[:, -1]
            else:
                ytr, yv, yte = ytr, yv, yte


            # snapshot (once) for reproducibility
            dump_json({"dataset": data_cfg, "model": model_cfg}, os.path.join(model_dir, "model_config.json"))

            if os.path.isfile(ckpt_path):
                log.info(f"[model] found checkpoint for {mname} under '{data_dir_rel}', loading (skip training)…")
                _ = load_checkpoint_if_exists(mdl, ckpt_path)
            else:
                log.info(f"[model] training {mname} on dataset '{data_dir_rel}'")
                t0 = time.time()
                train_classifier(mdl, Xtr, ytr, Xva=Xv, yva=yv, logger=log, early_stop=model_cfg["params"]["early_stop"], patience=model_cfg["params"]["patience"])
                log.info(f"[model] training complete in {time.time() - t0:.2f}s")
                save_checkpoint(mdl, ckpt_path, meta={"data_dir": data_dir_rel, "model_dir": model_dir_rel})
                log.info(f"[model] saved checkpoint → {ckpt_path}")

            # quick sanity validation after load/train
            val_loss, val_out = validate_classifier(mdl, Xv, yv, logger=log, return_loss=True)
            print(val_out)



            # Explain + evaluate
            log.info("🔍 Running %d explainer(s) and %d metric(s)…",
                     len(self.explainers), len(self.metrics))
            for explainer in self.explainers:
                # per-explainer directory: runs/<DATA>/<MODEL>/<EXPL>/<params>/...
                expl_dir_rel, _ = expl_dir_name(explainer)
                expl_cfg = next((e for e in self.config['explainers']))

                expl_dir = os.path.join(model_dir, expl_dir_rel)
                ensure_dir(expl_dir)


                # save explainer config
                dump_json({"dataset": data_cfg, "model": model_cfg, "explainer": expl_cfg},
                        os.path.join(expl_dir, "config.json"))
                attr_path = os.path.join(expl_dir, "attributions.pkl")
                expl_name = getattr(explainer, "name", explainer.__class__.__name__)


                # set generator directory for FIT/WINIT/RamBIT
                if expl_name in ["FIT", "WINIT"]:

                    # Convert to PyTorch format (N,T,D) → (N,D,T) because FIT expects (N,D,T)
                    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).permute(0, 2, 1)
                    Xv_t  = torch.tensor(Xv,  dtype=torch.float32).permute(0, 2, 1)
                    ytr_t = torch.tensor(ytr, dtype=torch.long)
                    yv_t  = torch.tensor(yv,  dtype=torch.long)

                    train_loader = DataLoader(
                        TensorDataset(Xtr_t, ytr_t),
                        batch_size=64,
                        shuffle=True
                    )
                    valid_loader = DataLoader(
                        TensorDataset(Xv_t, yv_t),
                        batch_size=64,
                        shuffle=False
                    )

                    explainer.generator_path = expl_dir
                    explainer.train_loader = train_loader
                    explainer.valid_loader = valid_loader


                    
                    
                if os.path.isfile(attr_path):
                    log.info(f"[explainer] found cache for {expl_name} under '{data_dir_rel}', loading attributions (skip generation)…")
                    with open(attr_path, "rb") as f:
                        attributions = pickle.load(f)          # expected to be np.ndarray (N,T,D) or (N,D,T)
                        plot_explainer_samples(
                            X=Xte,
                            y=yte,
                            attributions=attributions,
                            expl_name=expl_name,
                            expl_dir=expl_dir,
                        )
                else:
                    # explain
                    
                    log.info("   ▶ Explainer: %s — generating attributions…", expl_name)
                    t_expl = time.time()
                    attributions = explainer.explain(mdl, Xte)
                    expl_elapsed = time.time() - t_expl
                    log.info("     • Done in %.2fs; attr shape %s", expl_elapsed, _shape(attributions))
                    plot_explainer_samples(
                        X=Xte,
                        y=yte,
                        attributions=attributions,
                        expl_name=expl_name,
                        expl_dir=expl_dir,
                    )
                    # Ensure (N,T,D)
                    if attributions.shape[1] != Xte.shape[1]:
                        attributions = np.transpose(attributions, (0, 2, 1))

                    with open(attr_path, "wb") as f:
                        pickle.dump(attributions, f, protocol=pickle.HIGHEST_PROTOCOL)
                    log.info("     • Saved attributions to %s", attr_path)

                # --- build a flat row: core identifiers + flattened params + flattened metrics
                flat_row = {
                    "data": data_dir_rel,
                    "model": mname,
                    "explainer": expl_name
                }
                flat_row.update(_flat_with_prefix("model_", model_cfg['params']))     # e.g., model_lr, model_epochs, model_batch_size
                flat_row.update(_flat_with_prefix("expl_", expl_cfg['params']))       # e.g., expl_sigma, expl_samples
                    


                # Metrics
                metric_vals: Dict[str, Any] = {}
                for m in self.metrics:
                    metric_cfg = next((me for me in self.config['metrics']))
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

                    flat_row.update(_flat_with_prefix("metric_cfg_", metric_cfg['params']))  # e.g., k_ratio
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
                    **metric_vals
                })

        # Summary
        log.info("📄 Benchmark complete. Returning %d row(s).", len(rows))
        for r in rows:
            model_name = r.get("model")
            expl = r.get("explainer")
            kv = {k: v for k, v in r.items() if k not in ("model", "explainer")}
            kvs = ", ".join(
                f"{k}={float(v):.4f}" if isinstance(v, (int, float, np.floating)) else f"{k}={v}"
                for k, v in kv.items()
            )
            log.info("   • [%s] %s: | %s", model_name, expl, kvs)

        return rows

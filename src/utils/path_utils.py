from __future__ import annotations
import os, json, time, re
from typing import List, Dict, Any, Tuple
import logging
import torch
# --- simple logging setup (idempotent) ---
_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
log = logging.getLogger("ExplainBench")

def _sanitize(s: str) -> str:
    # file-system friendly
    s = re.sub(r"[^\w\-\.\=]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"

def _short_path_component_from_pathlike(p):
    if p is None:
        return None
    try:
        return os.path.basename(str(p).rstrip(os.sep))
    except Exception:
        return None

def _kv_str(d: Dict[str, Any], keys: List[str]) -> str:
    parts = []
    for k in sorted(keys):
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, float):
                v = f"{v:g}"
            elif isinstance(v, (list, tuple)):
                v = "-".join(str(x) for x in v)
            parts.append(f"{k}={v}")
    return ",".join(parts)

# ---------------------------------------------------------------------
# signatures for save paths
# ---------------------------------------------------------------------

def _model_name(m):
    return getattr(m, "name", None) or getattr(m, "__class__", type(m)).__name__



def dataset_signature(dataset) -> Tuple[str, Dict[str, Any]]:
    """
    Return (nested_dir, cfg_dict) for the dataset.
    Example: "FreqShape/split=1" (optionally adds more segments if available).
    """
    dname = dataset.__class__.__name__

    # gather a few attributes if they exist
    cfg = {}
    for k in ["split_no", "task", "n_classes", "length", "features", "layout"]:
        if hasattr(dataset, k):
            cfg[k] = getattr(dataset, k)
    for k in ["root", "base_path"]:
        if hasattr(dataset, k):
            cfg[k] = _short_path_component_from_pathlike(getattr(dataset, k))

    # nested pieces: dataset name, then most discriminative parts
    pieces = [ _sanitize(dname) ]
    if "split_no" in cfg:  pieces.append(f"split={cfg['split_no']}")
    # include a couple of optional, stable descriptors as deeper folders
    if "task"     in cfg:  pieces.append(f"task={cfg['task']}")
    if "n_classes" in cfg: pieces.append(f"classes={cfg['n_classes']}")

    # turn into nested path like "FreqShape/split=1/task=classification"
    nested_dir = os.path.join(*pieces)
    return nested_dir, cfg


def _get_torch_module(model) -> Tuple[nn.Module, torch.device, dict]:
    """
    Retrieve (net, device, hparams) from a model instance.
    - If model has `.torch_module()`, use it; otherwise assume the model itself is an nn.Module.
    - device: model.device if present else cuda-if-available else cpu.
    - hparams: {'lr','epochs','batch_size'} with sensible defaults if not present.
    """
    net = model.torch_module() if hasattr(model, "torch_module") else model
    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    net.to(device)

    hparams = {
        "lr": float(getattr(model, "lr", 1e-3)),
        "epochs": int(getattr(model, "epochs", 5)),
        "batch_size": int(getattr(model, "batch_size", 64)),
    }
    return net, device, hparams


def model_signature(model) -> Tuple[str, Dict[str, Any]]:
    mname = _model_name(model)
    _, _, hp = _get_torch_module(model)
    extras = {}
    # try to pull a few common architecture params if they exist
    for k in ["d_model", "nhead", "num_layers", "hidden_dim", "input_dim", "num_classes"]:
        if hasattr(model, k):
            extras[k] = getattr(model, k)
    cfg = {**hp, **extras}
    key = _sanitize(mname)
    suffix = _kv_str(cfg, ["d_model","nhead","num_layers","hidden_dim","input_dim","num_classes","lr","batch_size","epochs"])
    if suffix:
        key = f"{key}__{_sanitize(suffix)}"
    return key, cfg

def explainer_signature(explainer) -> Tuple[str, Dict[str, Any]]:
    ename = getattr(explainer, "name", explainer.__class__.__name__)
    # pull simple numeric/string params from __dict__ (if any)
    p = {}
    for k, v in getattr(explainer, "__dict__", {}).items():
        if isinstance(v, (int, float, bool, str)):
            p[k] = v
    key = _sanitize(ename)
    suffix = _kv_str(p, sorted(p.keys()))
    if suffix:
        key = f"{key}__{_sanitize(suffix)}"
    return key, p


def _short_float(v: float) -> str:
    # readable, filesystem-friendly floats
    s = f"{v:.0e}" if (v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e3)) else f"{v:g}"
    return s.replace("+", "")

def model_dir_name(model) -> Tuple[str, Dict[str, Any]]:
    """
    Compact, ordered model directory name.
    Example: "Transformer/dm=64_h=4_L=1_bs=32_lr=1e-3_ep=5"
    """
    name = _sanitize(_model_name(model))
    _, _, hp = _get_torch_module(model)
    extras = {}
    for k in ["d_model", "nhead", "num_layers", "hidden_dim", "input_dim", "num_classes"]:
        if hasattr(model, k):
            extras[k] = getattr(model, k)
    cfg = {**hp, **extras}

    # build short, stable ordering
    parts = [name]
    sub = []
    if "d_model"   in cfg: sub.append(f"dm={cfg['d_model']}")
    if "hidden_dim" in cfg: sub.append(f"hd={cfg['hidden_dim']}")
    if "nhead"     in cfg: sub.append(f"h={cfg['nhead']}")
    if "num_layers" in cfg: sub.append(f"L={cfg['num_layers']}")
    if "batch_size" in cfg: sub.append(f"bs={cfg['batch_size']}")
    if "lr"        in cfg: sub.append(f"lr={_short_float(cfg['lr'])}")
    if "epochs"    in cfg: sub.append(f"ep={cfg['epochs']}")
    # optional: include in/out dims when present
    if "input_dim" in cfg and cfg["input_dim"]: sub.append(f"in={cfg['input_dim']}")
    if "num_classes" in cfg and cfg["num_classes"]: sub.append(f"out={cfg['num_classes']}")

    if sub:
        parts.append("/" + "_".join(sub))  # subdir for hyperparams

    # join parts → "Transformer/dm=64_h=4_..."
    dir_rel = "".join(parts)
    return dir_rel, cfg

def expl_dir_name(explainer) -> Tuple[str, Dict[str, Any]]:
    """
    Compact explainer directory. Example: "Grad" or "LIME/samples=50_sigma=0.2"
    """
    ename = getattr(explainer, "name", explainer.__class__.__name__)
    ename = _sanitize(ename)
    p = {}
    for k, v in getattr(explainer, "__dict__", {}).items():
        if isinstance(v, (int, float, bool, str)):
            p[k] = v
    sub = []
    for k in sorted(p.keys()):
        val = p[k]
        if isinstance(val, float):
            val = _short_float(val)
        sub.append(f"{_sanitize(k)}={_sanitize(str(val))}")
    dir_rel = ename if not sub else os.path.join(ename, "_".join(sub))
    return dir_rel, p



# ---------------------------------------------------------------------
# saving / loading
# ---------------------------------------------------------------------

def save_checkpoint(model, ckpt_path: str, meta: Dict[str, Any]):
    net, device, _ = _get_torch_module(model)
    payload = {"state_dict": net.state_dict(), "meta": meta}
    torch.save(payload, ckpt_path)

def load_checkpoint_if_exists(model, ckpt_path: str) -> bool:
    if not os.path.isfile(ckpt_path):
        return False
    net, device, _ = _get_torch_module(model)
    log.info("🔁 Found checkpoint: %s — loading and skipping training.", ckpt_path)
    payload = torch.load(ckpt_path, map_location=device)
    sd = payload.get("state_dict", payload)
    net.load_state_dict(sd)
    return True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
# -------------------------
# unified trainer / validator
# -------------------------


# --- simple logging setup (idempotent) ---
_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
log = logging.getLogger("ExplainBench")

def _model_name(m):
    return getattr(m, "name", None) or getattr(m, "__class__", type(m)).__name__


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


def train_classifier(model, Xtr, ytr, Xva=None, yva=None, logger=log,
                     early_stop=True, patience=10):
    """
    Generic classifier trainer for (N,T,D) -> logits.
    Adds early stopping based on validation loss.
    """
    net, device, hp = _get_torch_module(model)
    ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long),
    )
    dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    ce = nn.CrossEntropyLoss()

    # --- Early Stopping State ---
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    net.train(True)
    for epoch in range(1, hp["epochs"] + 1):
        running_loss, seen = 0.0, 0
        pbar = tqdm(dl, desc=f"[{_model_name(model)}] Epoch {epoch}/{hp['epochs']}", leave=False)

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = net(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            bsz = xb.size(0)
            running_loss += loss.item() * bsz
            seen += bsz
            pbar.set_postfix(loss=f"{running_loss/max(seen,1):.4f}")

        train_loss = running_loss / max(seen, 1)
        tqdm.write(f"[{_model_name(model)}][Epoch {epoch}/{hp['epochs']}] "
                   f"train_loss={train_loss:.4f}")

        # -------------------------
        # VALIDATION & EARLY STOPPING
        # -------------------------
        if early_stop and Xva is not None and yva is not None:
            val_loss, _ = validate_classifier(
                model, Xva, yva, batch_size=2048, logger=logger, return_loss=True
            )

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1

            # patience reached → stop
            if wait >= patience:
                tqdm.write(
                    f"[{_model_name(model)}] Early stopping at epoch {epoch} "
                    f"(best val={best_val_loss:.4f})"
                )
                break

    # Restore best
    if early_stop and best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)


def validate_classifier(model, Xva, yva, *, batch_size: int = 32, logger=log, return_loss = False):
    """
    Unified validation for classifiers.
    Handles: 2-logit softmax, 1-logit sigmoid binary, or multiclass softmax.
    Prints a tidy summary and returns a dict of metrics.
    """
    if Xva is None or yva is None:
        logger.info("[val] skipped (no split)")
        return None

    net, device, _ = _get_torch_module(model)

    # SAVE current state and switch to eval
    was_training = net.training
    net.eval()

    with torch.no_grad():
        N = len(Xva)
        bs = batch_size or N
        logits_list = []
        for i in range(0, N, bs):
            xb = torch.from_numpy(Xva[i:i+bs]).float().to(device)
            logits_list.append(net(xb).detach().cpu())
        logits = torch.cat(logits_list, dim=0)

    # RESTORE previous state (critical for LSTM/RNN cuDNN backward)
    net.train(was_training)



    y_true = np.asarray(yva)
    # predictions & probs
    if logits.shape[1] == 1:  # Single class (sigmoid)
        probs = torch.sigmoid(logits).numpy().ravel()
        preds = (probs >= 0.5).astype(np.int64)
    else:
        probs_full = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs_full.argmax(axis=1).astype(np.int64)
        # probs = probs_full[:, 1] if probs_full.shape[1] == 2 else None

    prec = precision_score(y_true, preds, average="macro", zero_division=0)
    rec  = recall_score(y_true, preds, average="macro", zero_division=0)
    f1   = f1_score(y_true, preds, average="macro", zero_division=0)
    auroc = float("nan")

    try:
        if len(np.unique(y_true)) == 2:
            # binary classification
            auroc = roc_auc_score(y_true, probs[:, 1])
        else:
            # multiclass (OvR macro-average)
            auroc = roc_auc_score(y_true, probs_full, multi_class="ovr", average="macro")
    except Exception as e:
        auroc = float("nan")
        logger.warning(f"[metrics] AUROC failed: {e}")
    

    logger.info(f"[val] precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} auroc={auroc:.3f}")
    if return_loss:
        ce = nn.CrossEntropyLoss()
        val_loss = ce(torch.from_numpy(np.asarray(logits)).float(), torch.from_numpy(yva).long()).item()

    return val_loss, {"val_precision": prec, "val_recall": rec, "val_f1": f1, "val_auroc": auroc}

    

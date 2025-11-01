import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .base_model import BaseModel
from ..utils.registry import Registry
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F

class _TinyTransformer(nn.Module):
    def __init__(self, d_in, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):  # [B, T, D]
        z = self.proj(x)
        h = self.encoder(z)[:, -1, :]
        return self.fc(h)

@Registry.register_model("Transformer")
class Transformer(BaseModel):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=1, num_classes=2, lr=1e-3, epochs=3, batch_size=64, device=None):
        self.net = _TinyTransformer(input_dim, d_model, nhead, num_layers, num_classes)
        self.lr = float(lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, X_train, y_train):
        X = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_train, dtype=torch.long, device=self.device)
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = optim.Adam(self.net.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        self.net.train()

        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            seen = 0

            pbar = tqdm(dl, desc=f"Epoch {epoch}/{self.epochs}", leave=False)
            for xb, yb in pbar:
                opt.zero_grad()
                logits = self.net(xb)
                loss = ce(logits, yb)
                loss.backward()
                opt.step()

                # running average loss
                bsz = xb.size(0)
                running_loss += loss.item() * bsz
                seen += bsz
                pbar.set_postfix(loss=f"{running_loss/seen:.4f}")

            # nice one-line summary after each epoch
            avg_loss = running_loss / max(seen, 1)
            tqdm.write(f"[Epoch {epoch}/{self.epochs}] train_loss: {avg_loss:.4f}")

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.net(X)
            return logits.argmax(dim=-1).cpu().numpy()

    # def validate(self, Xva, yva, batch_size: int = 32):
    #     """
    #     Run validation on (Xva, yva), print metrics, and return a dict.
    #     - Supports binary (1-logit or 2-logit) and multiclass.
    #     - Uses batched inference to keep memory modest.
    #     """
    #     if Xva is None or yva is None:
    #         print("[val] skipped (no validation split provided)")
    #         return None

    #     self.net.eval()
    #     device = self.device

    #     # --- batched forward to collect logits ---
    #     with torch.no_grad():
    #         N = len(Xva)
    #         bs = batch_size or N
    #         logits_list = []
    #         for i in range(0, N, bs):
    #             Xb = torch.from_numpy(Xva[i:i+bs]).float().to(device)
    #             logits_b = self.net(Xb)   # (B, C) or (B, 1)
    #             logits_list.append(logits_b.detach().cpu())
    #         logits = torch.cat(logits_list, dim=0)              # (N, C) or (N, 1)

    #     # predicted labels
    #     if logits.shape[1] == 1:
    #         # single-logit binary: threshold at 0.5 on sigmoid
    #         probs = torch.sigmoid(logits).numpy().ravel()       # (N,)
    #         preds = (probs >= 0.5).astype(np.int64)
    #     else:
    #         # C >= 2
    #         probs_full = F.softmax(logits, dim=1).cpu().numpy()  # (N, C)
    #         preds = probs_full.argmax(axis=1).astype(np.int64)
    #         # if exactly 2 classes, keep positive-class prob for AUROC
    #         probs = probs_full[:, 1] if probs_full.shape[1] == 2 else None

    #     y_true = np.asarray(yva)

    #     prec = precision_score(y_true, preds, average="macro", zero_division=0)
    #     rec  = recall_score(y_true, preds, average="macro", zero_division=0)
    #     f1   = f1_score(y_true, preds, average="macro", zero_division=0)
    #     auroc = float("nan")
    #     if probs is not None and len(np.unique(y_true)) == 2:
    #         try:
    #             auroc = roc_auc_score(y_true, probs)
    #         except Exception:
    #             auroc = float("nan")

    #     print(f"[val] precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  auroc={auroc:.3f}")

    #     return {"val_precision": prec, "val_recall": rec, "val_f1": f1, "val_auroc": auroc}
    
    def torch_module(self):
        return self.net

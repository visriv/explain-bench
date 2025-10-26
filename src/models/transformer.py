import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .base_model import BaseModel
from ..utils.registry import Registry

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
        ds = TensorDataset(X,y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = optim.Adam(self.net.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(self.epochs):
            for xb,yb in dl:
                opt.zero_grad()
                logits = self.net(xb)
                loss = ce(logits, yb)
                loss.backward()
                opt.step()

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.net(X)
            return logits.argmax(dim=-1).cpu().numpy()

    def torch_module(self):
        return self.net

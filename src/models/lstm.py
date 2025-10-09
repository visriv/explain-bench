import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .base_model import BaseModel
from ..utils.registry import Registry

class _LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # x: [B, T, D]
        out,_ = self.lstm(x)
        h = out[:,-1,:]
        return self.fc(h)

@Registry.register_model("LSTM")
class LSTM(BaseModel):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, num_classes=2, lr=1e-3, epochs=3, batch_size=64, device=None):
        self.net = _LSTMNet(input_dim, hidden_dim, num_layers, num_classes)
        self.lr=lr; self.epochs=epochs; self.batch_size=batch_size
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

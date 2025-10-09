from explainbench.datasets.synthetic_dataset import SyntheticDataset
from explainbench.models.lstm import LSTM
from explainbench.explanations.grad import GradExplainer

def test_grad_explainer():
    (Xtr,ytr),(Xte,yte) = SyntheticDataset(n_train=50, n_test=8, length=12, features=3).load()
    model = LSTM(input_dim=3, num_classes=2, epochs=1, batch_size=8, hidden_dim=16)
    model.fit(Xtr,ytr)
    att = GradExplainer().explain(model, Xte)
    assert att.shape == Xte.shape

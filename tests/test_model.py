from explainbench.datasets.synthetic_dataset import SyntheticDataset
from explainbench.models.lstm import LSTM

def test_model_train_predict():
    (Xtr,ytr),(Xte,yte) = SyntheticDataset(n_train=50, n_test=10, length=12, features=3).load()
    model = LSTM(input_dim=3, num_classes=2, epochs=1, batch_size=16, hidden_dim=16)
    model.fit(Xtr,ytr)
    pred = model.predict(Xte)
    assert pred.shape == (10,)

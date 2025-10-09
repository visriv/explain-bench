from explainbench.datasets.synthetic_dataset import SyntheticDataset

def test_synth_shape():
    (Xtr,ytr),(Xte,yte) = SyntheticDataset(n_train=10, n_test=5, length=12, features=3).load()
    assert Xtr.shape == (10,12,3) and Xte.shape==(5,12,3)
    assert ytr.shape == (10,) and yte.shape==(5,)

# put this in src/datasets/datagen/gp_utils.py
import numpy as np

def rbf_kernel(t, lengthscale=20.0, variance=1.0):
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    d2 = (t - t.T) ** 2
    return variance * np.exp(-0.5 * d2 / (lengthscale ** 2))

def sample_gp(T, lengthscale=20.0, mean=0.0, variance=1.0, jitter=1e-6, rng=None):
    """Return a (T,) sample from a GP with RBF kernel."""
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(T, dtype=float)
    K = rbf_kernel(t, lengthscale=lengthscale, variance=variance)
    L = np.linalg.cholesky(K + jitter * np.eye(T))
    z = rng.standard_normal(T)
    return mean + L @ z  # (T,)

def sample_gp_features(T, D, lengthscales, means, var=0.1, rng=None):
    """Return X with shape (T, D)."""
    rng = np.random.default_rng() if rng is None else rng
    X = np.empty((T, D), dtype=float)
    for d in range(D):
        ell = lengthscales[d] if np.ndim(lengthscales) else lengthscales
        mu  = means[d]        if np.ndim(means)        else means
        X[:, d] = sample_gp(T, lengthscale=ell, mean=mu, variance=var, rng=rng)
    return X.astype("float32")

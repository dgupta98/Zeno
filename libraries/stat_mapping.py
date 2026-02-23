import numpy as np

def moment_init_linear(X: np.ndarray, y: np.ndarray, eps: float = 1e-6):
    """
    Interpretable mapping: w_j ≈ Cov(x_j, y) / Var(x_j)
    Works well when features are near-independent.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    var = (Xc ** 2).mean(axis=0) + eps
    cov = (Xc * yc[:, None]).mean(axis=0)
    w = cov / var
    b = float(y.mean() - X.mean(axis=0).dot(w))
    return w.astype(np.float32), np.float32(b)

def moment_init_logistic(X: np.ndarray, y: np.ndarray, eps: float = 1e-6):
    """
    Simple LDA-ish init for logistic:
    w ∝ (mu1 - mu0) / pooled_var ; b from log prior
    """
    y = y.astype(np.float32)
    X0 = X[y < 0.5]
    X1 = X[y >= 0.5]
    mu0 = X0.mean(axis=0) if len(X0) else np.zeros(X.shape[1], dtype=np.float32)
    mu1 = X1.mean(axis=0) if len(X1) else np.zeros(X.shape[1], dtype=np.float32)

    Xc = X - X.mean(axis=0, keepdims=True)
    pooled_var = (Xc ** 2).mean(axis=0) + eps

    w = (mu1 - mu0) / pooled_var
    p1 = float(max(eps, min(1 - eps, y.mean())))
    b = np.log(p1 / (1 - p1))
    return w.astype(np.float32), np.float32(b)
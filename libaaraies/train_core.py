import torch
import torch.nn as nn
import torch.optim as optim
from .metrics import mse, accuracy_from_logits

def fit_linear_sgd(X, y, w_init, b_init, steps=100, lr=0.1):
    w = torch.nn.Parameter(w_init.clone())
    b = torch.nn.Parameter(b_init.clone())
    opt = optim.SGD([w, b], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        yhat = X @ w + b
        loss = torch.mean((yhat - y) ** 2)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()

def fit_logistic_sgd(X, y, w_init, b_init, steps=100, lr=0.1):
    w = torch.nn.Parameter(w_init.clone())
    b = torch.nn.Parameter(b_init.clone())
    opt = optim.SGD([w, b], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        opt.zero_grad()
        logits = X @ w + b
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()

def eval_linear(X, y, w, b):
    yhat = X @ w + b
    return mse(yhat, y)

def eval_logistic(X, y, w, b):
    logits = X @ w + b
    return accuracy_from_logits(logits, y)
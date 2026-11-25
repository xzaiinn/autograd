"""Gradient checking utilities."""


import numpy as np

def num_grad(f, P, eps=1e-5):
    """Central-difference gradient of scalar f() with respect to array P.
    f takes no arguments and reads P, which is nudged in place."""
    g = np.zeros_like(P, dtype=float)
    for idx in np.ndindex(*P.shape):
        old = P[idx]
        P[idx] = old + eps
        up = f()
        P[idx] = old - eps
        down = f()
        P[idx] = old
        g[idx] = (up - down) / (2 * eps)
    return g

def rel_err(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12))
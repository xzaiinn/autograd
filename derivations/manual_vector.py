"""Manual gradient checks for vector-valued functions."""

import numpy as np  # type: ignore[import-not-found]
from gradcheck import num_grad, rel_err




def forward(W1, b1, W2, b2, x, t):

    z = W1 @ x + b1
    h = np.tanh(z)
    y = W2 @ h + b2
    L = ((y - t)**2)/2
    L = L.item()

    return L, z, h, y



def backward(W1, b1, W2, b2, x, t):

    z = W1 @ x + b1
    h = np.tanh(z)
    y = W2 @ h + b2
    L = ((y - t)**2)/2

    dy = y-t 
    dw2 = dy @ h.T
    db2 = dy
    dh = W2.T @ dy
    dz = (1 - h**2) * dh
    db1 = dz
    dw1 = dz @ x.T
    dx = W1.T @ dz

    return dw1, db1, dw2, db2, dx


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W1 = rng.normal(size=(2, 2)); b1 = rng.normal(size=(2, 1))
    W2 = rng.normal(size=(1, 2)); b2 = rng.normal(size=(1, 1))
    x = rng.normal(size=(2, 1));  t = 0.5

    loss = lambda: forward(W1, b1, W2, b2, x, t)[0]
    grads = backward(W1, b1, W2, b2, x, t)

    for name, P, ana in zip(["W1", "b1", "W2", "b2", "x"],
                            [W1, b1, W2, b2, x], grads):
        print(name, ana.shape, rel_err(ana, num_grad(loss, P)))
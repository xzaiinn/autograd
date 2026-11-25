import numpy as np

def loss (w1, w2, b1, b2, x=1.0, t=0.5):


    z = w1 * x + b1
    h = np.tanh(z)
    y = w2 * h + b2
    L = ((y-t)**2) / 2

    return L


def grads(w1, w2, b1, b2, x=1.0, t=0.5):

    z = w1 * x + b1
    h = np.tanh(z)
    y = w2 * h + b2
    L = ((y-t)**2) / 2

    dw1 = (y-t)*w2*(1 / np.cosh(z)**2) * x
    dw2 = (y-t)*h
    dx = (y-t)*w2*(1 / np.cosh(z)**2) * w1
    db1 = (y-t) * w2 * (1 / np.cosh(z)**2)
    db2 = y - t

    return dw1, dw2, db1, db2, dx

def num_grad(f, args, i, eps=1e-5):
    up = list(args)
    down = list(args)
    up[i] += eps
    down[i] -= eps
    return (f(*up) - f(*down)) / (2 * eps)

args = [0.5, 2.0, 0.1,-0.3, 1.7]            # w1, w2, x
names = ["w1", "w2", "b1","b2", "x"]
analytic = grads(*args)

for i, name in enumerate(names):
    num = num_grad(loss, args, i)
    ana = analytic[i]
    rel = abs(ana - num) / max(abs(ana), abs(num), 1e-12)
    print(name, ana, num, rel)


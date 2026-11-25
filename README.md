# autograd

A small reverse-mode automatic differentiation engine, written from scratch in plain Python,
with a tiny neural-network library on top. Every gradient is checked against a numerical
(finite-difference) estimate.

## What's here

| File | What it does |
|------|--------------|
| [`value.py`](value.py) | `Value`, a scalar that records the operations applied to it and backpropagates gradients through the resulting graph. Supports `+`, `-`, `*`, `/`, `**` (constant exponent) and `tanh`, including the reflected forms like `2 * a` and `1 / a`. |
| [`nn.py`](nn.py) | `Neuron`, `Layer` and `MLP`, built only from `Value`. Each neuron computes `tanh(w · x + b)`. |
| [`train.py`](train.py) | Trains a 3 → 4 → 4 → 1 MLP on a four-example toy dataset with squared-error loss and plain gradient descent. |
| [`test_mlp_grad.py`](test_mlp_grad.py) | Compares the gradients from `backward()` for every MLP parameter with central-difference estimates, then prints the worst relative error. |
| [`derivations/`](derivations/) | Gradients worked out by hand before the engine existed, each checked numerically. |

### `derivations/`

- [`manual_scalar.py`](derivations/manual_scalar.py): a one-hidden-unit network
  (`y = w2 · tanh(w1·x + b1) + b2`) with hand-derived partial derivatives, compared with
  finite differences.
- [`manual_vector.py`](derivations/manual_vector.py): the same network in matrix form
  (`W1`, `b1`, `W2`, `b2`), using matrix-calculus backprop.
- [`gradcheck.py`](derivations/gradcheck.py): helpers for those checks. `num_grad` gives a
  central-difference gradient for any NumPy array, and `rel_err` gives the maximum relative error.

## How it works

Each arithmetic operation on a `Value` returns a new `Value`. The new value remembers its
parents and holds a small `_backward` closure that applies the chain rule for that one
operation. Calling `loss.backward()`:

1. sorts the graph topologically with a depth-first search,
2. sets the output's gradient to `1.0`,
3. runs each node's `_backward` in reverse topological order.

Gradients are added with `+=`, not assigned. When a value is used more than once, the
contributions from every path therefore add up correctly. The same accumulation is why the
training loop resets `p.grad = 0.0` before each backward pass.

```python
from value import Value

a = Value(2.0)
b = Value(-3.0)
c = (a * b + a ** 2).tanh()
c.backward()
print(a.grad, b.grad)
```

## Running it

Requires Python 3.10+. NumPy is used only by the `derivations/` scripts. The engine itself
has no dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

python train.py                          # loss per step, then predictions vs. targets
python test_mlp_grad.py                  # worst relative gradient error for the MLP
python derivations/manual_scalar.py      # hand-derived scalar gradients vs. numerical
python derivations/manual_vector.py      # hand-derived matrix gradients vs. numerical
```

The gradient checks should report relative errors around `1e-9` or smaller.

## License

[MIT](LICENSE)

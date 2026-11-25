from value import Value
from nn import MLP

def make_loss(net, x, t):
    output = net(x)
    return 0.5 * (output - t) ** 2

xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

def total_loss(net):
    total = Value(0.0)
    for row, t in zip(xs, ys):
        inputs = [Value(v) for v in row]
        total += make_loss(net,inputs,t)
    return total


if __name__ == "__main__":
    net = MLP(3, [4, 4, 1])
    lr = 0.05

    for step in range(100):
        L = total_loss(net)
        print(L.data)
        for p in net.parameters():
            p.grad = 0.0
        L.backward()
        for p in net.parameters():
            p.data -= lr * p.grad
        print(step, L.data)
        
    for row, t in zip(xs, ys):
        pred = net([Value(v) for v in row])
        print(round(pred.data, 4), t)
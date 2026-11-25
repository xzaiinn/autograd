from value import Value
from nn import MLP

def make_loss(net, x, t):
    output = net(x)
    return 0.5 * (output - t) ** 2

net = MLP(3, [4, 4, 1])
x = [Value(0.5), Value(-1.2), Value(2.0)]
t = 0.3

loss = make_loss(net, x, t)
loss.backward()


analytic = [p.grad for p in net.parameters()]

eps = 1e-5
worst = 0.0
for i, p in enumerate(net.parameters()):
    old = p.data
    p.data = old + eps; up = make_loss(net, x, t).data
    p.data = old - eps; down = make_loss(net, x, t).data
    p.data = old
    num = (up - down) / (2 * eps)
    ana = analytic[i]
    rel = abs(ana - num) / max(abs(ana), abs(num), 1e-12)
    worst = max(worst, rel)
print(worst)
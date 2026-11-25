"""Scalar Value class for reverse-mode autodiff."""

import math as math
class Value:
    
    
    def __init__(self, data, parents=()):
        self.data = data
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None   # default: do nothing

    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, parents=(self, other))
        def _backward():
                self.grad +=  out.grad
                other.grad += out.grad
        out._backward = _backward
        return out


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, parents=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    
    def __pow__(self, n):
        assert isinstance(n, (int, float))   # constant exponents only, for now
        out = Value(self.data ** n, parents=(self,))

        def _backward():
            self.grad += (n * self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, parents=(self,))

        def _backward():
            self.grad += (1 - t**2) * out.grad
            

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    
    def __radd__(self, other):
        return self + other
    
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + other.__neg__()

    def __rsub__(self, other):   
        return other + self.__neg__()

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):   # for  1 / a
        return other * self ** -1

    
    def backward(self):
        order = []
        visited = set()

        def build(node):
            if node not in visited:
                 visited.add(node)
                 for parent in node._parents:
                      build(parent)
                 order.append(node)


                
        build(self)
        self.grad = 1.0
        for node in reversed(order):
             node._backward()
             
             

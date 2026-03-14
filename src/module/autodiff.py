import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda:None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, and op='{self._op}')"
    
    @staticmethod
    def _to_tensor(x):
        if isinstance(x, Tensor):
            return x
        else:
            return Tensor(x)
        
    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=float)

    
    # Basic Operator Overload

    def __add__(self, other):
        other = Tensor._to_tensor(other)
        out = Tensor(self.data + other.data, _children = (self,other), _op = "+")

        def _backward():
            # implementasinya
            pass
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = Tensor._to_tensor(other)
        out = Tensor(self.data * other.data, _children=(self,other), _op='*')

        def _backward():
            # implementasinya
            pass

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, exponent):
        if not isinstance(exponent, (int,float)):
            raise TypeError("Eksponen harus int atau float")
        out = Tensor(self.data ** exponent, _children=(self,), _op=f"**{exponent}")

        def _backward():
            # implementasinya
            pass
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = Tensor._to_tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")

        def _backward():
            # implementasi nanti
            pass

        out._backward = _backward
        return out
    
    def __rmatmul__(self,other):
        other = Tensor._to_tensor(other)
        return other.__matmul__(self)
    
    def dot(self,other):
        other = Tensor._to_tensor(other)
        out = Tensor(np.dot(self.data, other.data), _children=(self, other), _op="dot")

        def _backward():
            # implementasi nnti
            pass

        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,), _op="exp")

        def _backward():
            #implementasi
            pass

        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), _children=(self,), _op="log")

        def _backward():
            #implemetnasi
            pass

        out._backward = _backward
        return out
    

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _children=(self,), _op="sum")

        def _backward():
            #implementasi
            pass

        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), _children=(self,), _op="relu")

        def _backward():
            #implementasi
            pass

        out._backward = _backward
        return out
    

    # Reverse backprop
    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad harus ada buat output nonskalar, misal tensor dgn size > 1")
            grad = np.ones_like(self.data, dtype=float)
        else:
            grad = np.array(grad, dtype=float)

        # topological sort
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = self.grad + grad

        for node in reversed(topo):
            node._backward()

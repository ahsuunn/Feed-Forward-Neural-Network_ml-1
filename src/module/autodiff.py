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
        
    @staticmethod
    def _unbroadcast(grad, original_shape):
        """
        Fungsi helper untuk balikin dimensi gradien ke shape aslinya kalau terjadi broadcasting dari numpy
        """
        while len(grad.shape) > len(original_shape):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(original_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad
        
    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=float)

    
    # Basic Operator Overload

    def __add__(self, other):
        other = Tensor._to_tensor(other)
        out = Tensor(self.data + other.data, _children = (self,other), _op = "+")

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = Tensor._to_tensor(other)
        out = Tensor(self.data * other.data, _children=(self,other), _op='*')

        def _backward():
            grad_self_raw = other.data * out.grad
            grad_other_raw = self.data * out.grad

            self.grad += Tensor._unbroadcast(grad_self_raw, self.data.shape)
            other.grad += Tensor._unbroadcast(grad_other_raw, other.data.shape)

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, exponent):
        if not isinstance(exponent, (int,float)):
            raise TypeError("Eksponen harus int atau float")
        out = Tensor(self.data ** exponent, _children=(self,), _op=f"**{exponent}")

        def _backward():
            # turunan eksponen -> d(x^n)/dx = n * x^(n-1)
            self.grad += (exponent * (self.data ** (exponent-1))) * out.grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = Tensor._to_tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")

        def _backward():
            # turunan dari matrix multiplier menggunakan transpose .T
            # kalau C = A @ B -> dL/dA = dL/dC @ B.T -> dL/dB = A.T @ dL/dC
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out
    
    def __rmatmul__(self,other):
        other = Tensor._to_tensor(other)
        return other.__matmul__(self)
    
    def dot(self,other):
        other = Tensor._to_tensor(other)
        out = Tensor(np.dot(self.data, other.data), _children=(self, other), _op="dot")

        def _backward():
            # turunan dot sama dgn matmul, tp pake np.dot agar konsisten dgn forward pass
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)

        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,), _op="exp")

        def _backward():
            # turunan eksponen -> eksponen itu sendiri
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), _children=(self,), _op="log")

        def _backward():
            # turunan dari ln x -> 1/x
            self.grad += (1.0/self.data) * out.grad

        out._backward = _backward
        return out
    

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _children=(self,), _op="sum")

        def _backward():
            grad = out.grad
            # kalau sum ngurangin dimensi, perlu dibalikin dimensinya agar bs di-broadcast ke bentuk data aslinya
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            # turunan dari x1 + x2 + ... + xn -> 1 untuk setiap xi, jd pakai ones like
            self.grad += grad * np.ones_like(self.data)

        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), _children=(self,), _op="relu")

        def _backward():
            # turunan ReLU -> 1 jika x > 0, dan 0 jika x <= 0
            self.grad += (self.data > 0) * out.grad

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

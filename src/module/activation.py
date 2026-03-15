import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

class Linear(Activation):
    def forward(self, x):
        self.output = x
        return self.output

    def backward(self, grad):
        return grad * np.ones_like(self.output)

class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.input > 0).astype(float)

class Sigmoid(Activation):
    def forward(self, x):
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)

class Tanh(Activation):
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        return grad * (1.0 - self.output ** 2)

class Softmax(Activation):
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)

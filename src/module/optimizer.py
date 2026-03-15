import numpy as np


class Optimizer:
    def __init__(self, parameters, l1_lambda=0.0, l2_lambda=0.0):
        self.parameters = parameters
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def _apply_regularization(self):
        for p in self.parameters:
            if self.l2_lambda > 0:
                p.grad += self.l2_lambda * p.data
            if self.l1_lambda > 0:
                p.grad += self.l1_lambda * np.sign(p.data)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class SGD(Optimizer):
    def __init__(self, parameters, learning_rate=0.01, l1_lambda=0.0, l2_lambda=0.0):
        super().__init__(parameters, l1_lambda, l2_lambda)
        self.learning_rate = learning_rate

    def step(self):
        self._apply_regularization()
        for p in self.parameters:
            p.data -= self.learning_rate * p.grad


class Adam(Optimizer):
    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        l1_lambda=0.0,
        l2_lambda=0.0,
    ):
        super().__init__(parameters, l1_lambda, l2_lambda)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self._apply_regularization()
        self.t += 1
        for i, p in enumerate(self.parameters):
            # Update biased first and second moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)

            # Compute bias-corrected first and second moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters
            p.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

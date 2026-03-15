import numpy as np
from src.module.autodiff import Tensor


class RMSNorm:
    def __init__(self, size, learning_rate=0.01, eps=1e-8):
        self.learning_rate = learning_rate
        self.eps = eps
        self.gamma = Tensor(np.ones((1, size)))

    def forward(self, inputs):
        inputs = Tensor._to_tensor(inputs)

        x_squared = inputs**2.0

        # mean(x^2) = sum(x^2) / size
        # mean on last axis (features)
        mean_squared = x_squared.sum(axis=-1, keepdims=True) * (
            1.0 / inputs.data.shape[-1]
        )

        # 1 / sqrt(mean_squared + eps)
        inv_rms = (mean_squared + self.eps) ** -0.5

        return (inputs * inv_rms) * self.gamma

    def update(self, lambda_regularization=0.0):
        self.gamma.data -= self.learning_rate * self.gamma.grad
        self.gamma.zero_grad()

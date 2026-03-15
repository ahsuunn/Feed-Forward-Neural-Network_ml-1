import numpy as np
from src.module.autodiff import Tensor


class Layer:
    def __init__(
        self,
        input_size,
        output_size,
        initialization_type="random_uniform",
        learning_rate=0.01,
        regularization_type=None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.initialization_type = initialization_type
        self.learning_rate = learning_rate
        self.regularization_type = regularization_type

        # Initialize weights
        if self.initialization_type == "zero":
            w = np.zeros((input_size, output_size))
        elif self.initialization_type == "random_uniform":
            w = np.random.uniform(-0.5, 0.5, (input_size, output_size))
        elif self.initialization_type == "random_normal":
            w = np.random.randn(input_size, output_size)

        # Bonus
        elif self.initialization_type == "he":
            w = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif self.initialization_type == "xavier":
            w = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)

        else:
            w = np.random.randn(input_size, output_size)

        self.weights = Tensor(w)
        self.bias = Tensor(np.zeros((1, output_size)))

        self.inputs = None

    def forward(self, inputs):
        self.inputs = Tensor._to_tensor(inputs)
        return self.inputs @ self.weights + self.bias

    def update(self, lambda_regularization=0.0):
        if self.regularization_type == "l1":
            l1_penalty = lambda_regularization * np.sign(self.weights.data)
            self.weights.data -= self.learning_rate * (self.weights.grad + l1_penalty)
        elif self.regularization_type == "l2":
            l2_penalty = lambda_regularization * self.weights.data
            self.weights.data -= self.learning_rate * (self.weights.grad + l2_penalty)
        else:
            self.weights.data -= self.learning_rate * self.weights.grad

        self.bias.data -= self.learning_rate * self.bias.grad

        self.weights.zero_grad()
        self.bias.zero_grad()

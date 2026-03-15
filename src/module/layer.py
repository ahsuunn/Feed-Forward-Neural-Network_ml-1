import numpy as np
from module.autodiff import Tensor


class Layer:
    def __init__(
        self,
        input_size,
        output_size,
        initialization_type="random_uniform",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.initialization_type = initialization_type

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

    def get_parameters(self):
        return [self.weights, self.bias]

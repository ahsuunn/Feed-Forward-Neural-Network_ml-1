import numpy as np

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
            self.weights = np.zeros((input_size, output_size))
        elif self.initialization_type == "random_uniform":
            self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))
        elif self.initialization_type == "random_normal":
            self.weights = np.random.randn(input_size, output_size)
        elif self.initialization_type == "he":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size)

        self.bias = np.zeros((1, output_size))
        
        self.inputs = None
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad):
        self.gradient_weights = np.dot(self.inputs.T, grad)
        self.gradient_bias = np.sum(grad, axis=0, keepdims=True)

        input_grad = np.dot(grad, self.weights.T)
        return input_grad

    def update(self, lambda_regularization=0.0):
        if self.regularization_type == "l1":
            l1_penalty = lambda_regularization * np.sign(self.weights)
            self.weights -= self.learning_rate * (self.gradient_weights + l1_penalty)
        elif self.regularization_type == "l2":
            l2_penalty = lambda_regularization * self.weights
            self.weights -= self.learning_rate * (self.gradient_weights + l2_penalty)
        else:
            self.weights -= self.learning_rate * self.gradient_weights
            
        self.bias -= self.learning_rate * self.gradient_bias

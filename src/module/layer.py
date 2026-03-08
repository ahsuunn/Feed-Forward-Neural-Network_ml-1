class Layer:
    def __init__(
        self,
        input_size,
        output_size,
        initialization_type,
        learning_rate,
        regularization_type,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.initialization_type = initialization_type
        self.learning_rate = learning_rate
        self.regularization_type = regularization_type

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs, grad):
        self.gradient_weights = np.dot(inputs.T, grad)
        self.gradient_bias = np.sum(grad)

        input_grad = np.dot(grad, self.weights.T)
        return input_grad

    def update(self, lambda_regularization=0.0):
        if self.regularization_type == "l1":
            raise NotImplementedError
        elif regularization_type == "l2":
            raise NotImplementedError
        else:
            raise ValueError("Invalid regularization type")

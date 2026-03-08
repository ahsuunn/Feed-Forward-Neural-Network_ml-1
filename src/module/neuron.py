class Neuron:
    def __init__(self, input_size, initialization_type):
        """
        Args:
            input_size (int): Input size
            initialization_type (str): Initialization type
            Available Type:
                - zero
                - random_uniform(lowerbound, upperbound)
                - random_normal(mean, std, seed)
                - he
        """
        self.weights = np.random.randn(input_size)
        self.gradient_weights = np.zeros(input_size)

        self.bias = np.random.randn()
        self.gradient_bias = 0
        self.initialization_type = initialization_type

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, inputs, grad):
        self.weights -= learning_rate * np.dot(inputs.T, grad)
        self.bias -= learning_rate * np.sum(grad)

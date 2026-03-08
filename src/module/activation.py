class Activation:
    def __init__(self, activation_type):
        """
        Activation function type
        Available Type:
            - linear
            - relu
            - sigmoid
            - softmax
            - tanh
            - softmax
        Args:
            activation_type (str): Activation function type

        Raises:
            ValueError: If activation_type is not supported
        """
        self.activation_type = activation_type

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

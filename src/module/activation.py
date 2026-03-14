import numpy as np

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

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def _linear_forward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _linear_backward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _relu_forward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _relu_backward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _sigmoid_forward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _sigmoid_backward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _tanh_forward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _tanh_backward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _softmax_forward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
    
    def _softmax_backward(self, x: np.ndarray) -> np.ndarray:
        return NotImplementedError
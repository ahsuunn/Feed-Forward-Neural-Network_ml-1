import numpy as np

class Activation:
    def __init__(self, activation_type: str):
        supported = {"linear", "relu", "sigmoid", "tanh", "softmax"}
        if activation_type not in supported:
            raise ValueError(                
                f"Activation '{activation_type}' tidak dikenal. "
                f"Pilihan yang tersedia: {supported}"
            )
        self.activation_type = activation_type

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type == "linear":
            return self._linear_forward(x)
        elif self.activation_type == "relu":
            return self._relu_forward(x)
        elif self.activation_type == "sigmoid":
            return self._sigmoid_forward(x)
        elif self.activation_type == "tanh":
            return self._tanh_forward(x)
        elif self.activation_type == "softmax":
            return self._softmax_forward(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type == "linear":
            return self._linear_backward(x)
        elif self.activation_type == "relu":
            return self._relu_backward(x)
        elif self.activation_type == "sigmoid":
            return self._sigmoid_backward(x)
        elif self.activation_type == "tanh":
            return self._tanh_backward(x)
        elif self.activation_type == "softmax":
            return self._softmax_backward(x)
    
    def _linear_forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def _linear_backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    def _relu_forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def _sigmoid_forward(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))
    
    def _sigmoid_backward(self, x: np.ndarray) -> np.ndarray:
        s = self._sigmoid_forward(x)
        return s * (1 - s)
    
    def _tanh_forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def _tanh_backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    
    def _softmax_forward(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _softmax_backward(self, x: np.ndarray) -> np.ndarray:
        s = self._softmax_forward(x)          
        n = s.shape[-1]
        diag_s = np.einsum('bi,ij->bij', s, np.eye(n))   
        outer_s = np.einsum('bi,bj->bij', s, s)           
        return diag_s - outer_s   
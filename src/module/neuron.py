import numpy as np

class Neuron:
    def __init__(
        self,
        n_in: int,
        init_method: str = "random_normal",
        init_params: dict = None,
    ):
        self.n_in = n_in
        params = init_params or {}
        self.weights, self.bias = self._initialize(init_method, params)
        self.grad_w = np.zeros(n_in)
        self.grad_b = 0.0

    def _initialize(self, method: str, params: dict):
        return NotImplementedError

    def compute(self, x: np.ndarray) -> float:
        return NotImplementedError

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_bias(self) -> float:
        return self.bias

    def set_weights(self, weights: np.ndarray, bias: float):
        return NotImplementedError

    def set_gradients(self, grad_w: np.ndarray, grad_b: float):
        return NotImplementedError

    def reset_gradients(self):
        return NotImplementedError

    def __repr__(self) -> str:
        return NotImplementedError
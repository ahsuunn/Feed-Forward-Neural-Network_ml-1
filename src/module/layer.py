import numpy as np
from activation import Activation
from neuron import Neuron


class Layer:
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation_type: str,
        init_method: str = "random_normal",
        init_params: dict = None,
    ):
        self.n_in  = n_in
        self.n_out = n_out
        self.activation = Activation(activation_type)
        params = init_params or {}
        self.neurons = self._create_neurons(init_method, params)
        self.W, self.b = self._assemble_matrix()
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache_X = None
        self._cache_Z = None

    def _create_neurons(self, init_method: str, params: dict) -> list:
        return NotImplementedError

    def _assemble_matrix(self):
        return NotImplementedError

    def _distribute_to_neurons(self):
        return NotImplementedError

    def _distribute_gradients_to_neurons(self):
        return NotImplementedError

    def forward(self, X: np.ndarray) -> np.ndarray:
        return NotImplementedError

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return NotImplementedError
        
    def update(self, learning_rate: float, reg_type: str = None, lam: float = 0.0):
        return

    def get_weights(self) -> dict:
        return NotImplementedError

    def set_weights(self, W: np.ndarray, b: np.ndarray):
        return NotImplementedError

    def __repr__(self) -> str:
        pass
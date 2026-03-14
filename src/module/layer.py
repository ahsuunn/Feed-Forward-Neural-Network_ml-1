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
        neurons = []
        base_seed = params.get("seed", None)

        for i in range(self.n_out):
            neuron_params = params.copy()
            if base_seed is not None:
                neuron_params["seed"] = base_seed + i
            neurons.append(Neuron(self.n_in, init_method, neuron_params))

        return neurons

    def _assemble_matrix(self):
        W = np.array([n.get_weights() for n in self.neurons])
        b = np.array([n.get_bias()    for n in self.neurons])
        return W, b

    def _distribute_to_neurons(self):
        for i, neuron in enumerate(self.neurons):
            neuron.set_weights(self.W[i], self.b[i])

    def _distribute_gradients_to_neurons(self):
        for i, neuron in enumerate(self.neurons):
            neuron.set_gradients(self.dW[i], self.db[i])

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._cache_X = X
        Z = X @ self.W.T + self.b
        self._cache_Z = Z
        return self.activation.forward(Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        X = self._cache_X
        Z = self._cache_Z

        if self.activation.activation_type == "softmax":
            J  = self.activation.backward(Z)
            dZ = np.einsum('bij,bj->bi', J, dA)
        else:
            dZ = dA * self.activation.backward(Z)
        
        self.dW = dZ.T @ X
        self.db = np.sum(dZ, axis=0)
        self._distribute_gradients_to_neurons()
        return dZ @ self.W
        
    def update(self, learning_rate: float, reg_type: str = None, lam: float = 0.0):
        if reg_type == "l2":
            self.W -= learning_rate * (self.dW + lam * self.W)
        elif reg_type == "l1":
            self.W -= learning_rate * (self.dW + lam * np.sign(self.W))
        else:
            self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        self._distribute_to_neurons()

    def get_weights(self) -> dict:
        return {"W": self.W.copy(), "b": self.b.copy()}
        
    def set_weights(self, W: np.ndarray, b: np.ndarray):
        assert W.shape == (self.n_out, self.n_in)
        assert b.shape == (self.n_out,)
        self.W = W.copy()
        self.b = b.copy()
        self._distribute_to_neurons()     

    def __repr__(self) -> str:
        return (
            f"Layer(n_in={self.n_in}, n_out={self.n_out}, "
            f"activation='{self.activation.activation_type}', "
            f"neurons={len(self.neurons)})"
        )
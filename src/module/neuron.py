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
        seed = params.get("seed", None)
        rng = np.random.default_rng(seed)

        if method == "zero":
            weights = np.zeros(self.n_in)
        elif method == "random_normal":
            mean = params.get("mean", 0.0)
            std = params.get("std", 0.01)
            weights = rng.normal(loc=mean, scale=std, size=self.n_in)
        elif method == "random_uniform":
            low = params.get("low", -0.01)
            high = params.get("high", 0.01)
            weights = rng.uniform(low=low, high=high, size=self.n_in)
        else:         
            raise ValueError(
                f"Metode inisialisasi '{method}' tidak dikenal. "
                f"Pilihan yang tersedia: 'zero', 'random_normal', 'random_uniform'"
            )
        
        bias = 0.0
        return weights, bias

    def compute(self, x: np.ndarray) -> float:
        if x.shape != (self.n_in,):
            raise ValueError(
                f"Shape input {x.shape} tidak cocok dengan n_in={self.n_in}. "
                f"Ekspektasi: ({self.n_in},)"
            )
        return float(np.dot(self.weights, x) + self.bias)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_bias(self) -> float:
        return self.bias

    def set_weights(self, weights: np.ndarray, bias: float):
        if weights.shape != (self.n_in,):
            raise ValueError(
                f"Shape weights {weights.shape} tidak cocok. "
                f"Ekspektasi: ({self.n_in},)"
            )
        self.weights = weights.copy()
        self.bias    = float(bias)

    def set_gradients(self, grad_w: np.ndarray, grad_b: float):
        if grad_w.shape != (self.n_in,):
            raise ValueError(
                f"Shape grad_w {grad_w.shape} tidak cocok. "
                f"Ekspektasi: ({self.n_in},)"
            )
        self.grad_w = grad_w.copy()
        self.grad_b = float(grad_b)

    def reset_gradients(self):
        self.grad_w = np.zeros(self.n_in)
        self.grad_b = 0.0

    def __repr__(self) -> str:
        w_summary = np.round(self.weights[:3], 3)
        dots      = "..." if self.n_in > 3 else ""
        return (
            f"Neuron(n_in={self.n_in}, "
            f"weights=[{w_summary}{dots}], "
            f"bias={self.bias:.3f})"
        )
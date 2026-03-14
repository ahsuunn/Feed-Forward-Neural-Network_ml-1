import numpy as np

class LossFunction:
    def __init__(self, loss_function_type: str):
        """
        Loss function type
        Available Type:
            - mse
            - binary_crossentropy
            - categorical_crossentropy
        Args:
            loss_function_type (str): Loss function type
        """
        supported = {"mse", "binary_crossentropy", "categorical_crossentropy"}  
        if loss_function_type not in supported:
            raise ValueError(                
                f"Loss function '{loss_function_type}' tidak dikenal. "
                f"Pilihan yang tersedia: {supported}"
            )
        self.loss_function_type = loss_function_type

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.loss_function_type == "mse":
            return self._mse_forward(y_true, y_pred)
        elif self.loss_function_type == "binary_crossentropy":
            return self._binary_crossentropy_forward(y_true, y_pred)
        elif self.loss_function_type == "categorical_crossentropy":
            return self._categorical_crossentropy_forward(y_true, y_pred)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.loss_function_type == "mse":
            return self._mse_backward(y_true, y_pred)
        elif self.loss_function_type == "binary_crossentropy":
            return self._binary_crossentropy_backward(y_true, y_pred)
        elif self.loss_function_type == "categorical_crossentropy":
            return self._categorical_crossentropy_backward(y_true, y_pred)
    
    def _mse_forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def _mse_backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (2/n) * (y_pred - y_true)

    def _binary_crossentropy_forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(loss)
    
    def _binary_crossentropy_backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        return (1 / n) * (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped))
    
    def _categorical_crossentropy_forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0)
        per_sample_loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return np.mean(per_sample_loss)
    
    def _categorical_crossentropy_backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0)
        return -(1 / n) * (y_true / y_pred_clipped)


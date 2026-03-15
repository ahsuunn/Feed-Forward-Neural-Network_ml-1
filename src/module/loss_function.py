import numpy as np

class LossFunction:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError

class MSE(LossFunction):
    def forward(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size

class BinaryCrossEntropy(LossFunction):
    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / y_pred.shape[0]

class CategoricalCrossEntropy(LossFunction):
    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) / y_pred.shape[0]

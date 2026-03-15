import numpy as np
from module.autodiff import Tensor

class LossFunction:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

class MSE(LossFunction):
    def forward(self, y_true, y_pred):
        y_true = Tensor._to_tensor(y_true)
        diff = y_pred + (-1.0 * y_true)
        squared = diff ** 2.0
        return squared.sum() * (1.0 / y_true.data.size)

class BinaryCrossEntropy(LossFunction):
    def forward(self, y_true, y_pred):
        y_true = Tensor._to_tensor(y_true)
        eps = 1e-15
        term1 = (y_pred + eps).log() * y_true
        term2 = ((y_pred * -1.0) + (1.0 + eps)).log() * (1.0 + (-1.0 * y_true))
        return (term1 + term2).sum() * (-1.0 / y_true.data.size)

class CategoricalCrossEntropy(LossFunction):
    def forward(self, y_true, y_pred):
        y_true = Tensor._to_tensor(y_true)
        eps = 1e-15
        batch_size = y_true.data.shape[0]
        term = (y_pred + eps).log() * y_true
        return term.sum() * (-1.0 / batch_size)

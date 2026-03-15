import numpy as np


class Activation:
    def forward(self, x):
        raise NotImplementedError


class Linear(Activation):
    def forward(self, x):
        return x


class ReLU(Activation):
    def forward(self, x):
        return x.relu()


class Sigmoid(Activation):
    def forward(self, x):
        # 1 / (1 + exp(-x))
        return (1.0 + (-x).exp()) ** -1.0


class Tanh(Activation):
    def forward(self, x):
        # (exp(2x) - 1) / (exp(2x) + 1)
        exp2x = (x * 2.0).exp()
        return (exp2x + -1.0) * ((exp2x + 1.0) ** -1.0)


class Softmax(Activation):
    def forward(self, x):
        # Compute max purely on the data for numeric stability
        max_val = np.max(x.data, axis=-1, keepdims=True)
        shifted = x + (-max_val)
        exps = shifted.exp()
        return exps * (exps.sum(axis=-1, keepdims=True) ** -1.0)


# Bonus
class Swish(Activation):
    def forward(self, x):
        return x * (1.0 + (-x).exp()) ** -1.0


class LeakyReLU(Activation):
    def forward(self, x):
        return x.relu() + (-0.01 * (-x).relu())

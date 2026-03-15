import sys
import numpy as np


from .ffnn import FeedForwardNeuralNetwork
from .module.layer import Layer
from .module.activation import ReLU, Softmax
from .module.loss_function import CategoricalCrossEntropy

# Create dummy data
np.random.seed(42)
X = np.random.randn(100, 4)
y_labels = np.random.randint(0, 3, 100)
y = np.zeros((100, 3))
y[np.arange(100), y_labels] = 1

model = FeedForwardNeuralNetwork(verbose=True, seed=42)

model.add(
    Layer(input_size=4, output_size=8, initialization_type="he", learning_rate=0.01)
)
model.add(ReLU())
model.add(
    Layer(input_size=8, output_size=3, initialization_type="he", learning_rate=0.01)
)
model.add(Softmax())

model.compile(CategoricalCrossEntropy())

model.fit(X, y, epochs=10, batch_size=10, lambda_regularization=0.0)

print(model.predict(X[:2]))

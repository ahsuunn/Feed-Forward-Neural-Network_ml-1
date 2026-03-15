import sys
import numpy as np

from .ffnn import FeedForwardNeuralNetwork
from .module.layer import Layer
from .module.normalization import RMSNorm
from .module.activation import ReLU, Softmax
from .module.loss_function import CategoricalCrossEntropy

# Create dummy data
np.random.seed(42)
X = np.random.randn(100, 4)
y_labels = np.random.randint(0, 3, 100)
y = np.zeros((100, 3))
y[np.arange(100), y_labels] = 1

print("--- TRAINING WITHOUT RMSNorm ---")
model1 = FeedForwardNeuralNetwork(verbose=True, seed=42)
model1.add(Layer(input_size=4, output_size=16, initialization_type="he", learning_rate=0.01))
model1.add(ReLU())
model1.add(Layer(input_size=16, output_size=3, initialization_type="he", learning_rate=0.01))
model1.add(Softmax())

model1.compile(CategoricalCrossEntropy())
model1.fit(X, y, epochs=15, batch_size=10, lambda_regularization=0.0)

print("\n--- TRAINING WITH RMSNorm ---")
model2 = FeedForwardNeuralNetwork(verbose=True, seed=42)
model2.add(Layer(input_size=4, output_size=16, initialization_type="he", learning_rate=0.01))
model2.add(RMSNorm(size=16, learning_rate=0.01)) # <-- Added RMSNorm Feature
model2.add(ReLU())
model2.add(Layer(input_size=16, output_size=3, initialization_type="he", learning_rate=0.01))
model2.add(Softmax())

model2.compile(CategoricalCrossEntropy())
model2.fit(X, y, epochs=15, batch_size=10, lambda_regularization=0.0)

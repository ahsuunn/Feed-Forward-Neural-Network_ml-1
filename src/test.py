import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ffnn import FeedForwardNeuralNetwork
from module.layer import Layer
from module.normalization import RMSNorm
from module.activation import ReLU, Softmax
from module.loss_function import CategoricalCrossEntropy
from module.optimizer import Adam, SGD

# Create dummy data
np.random.seed(42)
X = np.random.randn(100, 4)
y_labels = np.random.randint(0, 3, 100)
y = np.zeros((100, 3))
y[np.arange(100), y_labels] = 1

print("Without RMSNorm")
model1 = FeedForwardNeuralNetwork(verbose=True, seed=42)
model1.add(Layer(input_size=4, output_size=16, initialization_type="he"))
model1.add(ReLU())
model1.add(Layer(input_size=16, output_size=3, initialization_type="he"))
model1.add(Softmax())

params1 = []
for layer in model1.layers:
    if hasattr(layer, "get_parameters"):
        params1.extend(layer.get_parameters())

model1.compile(CategoricalCrossEntropy(), optimizer=Adam(params1, learning_rate=0.01))
model1.fit(X, y, epochs=15, batch_size=10)
model1.plot_weight_distribution(0)
model1.plot_gradient_weight_distribution(0)

print("With RMSNorm")
model2 = FeedForwardNeuralNetwork(verbose=True, seed=42)
model2.add(Layer(input_size=4, output_size=16, initialization_type="he"))
model2.add(RMSNorm(size=16))
model2.add(ReLU())
model2.add(Layer(input_size=16, output_size=3, initialization_type="he"))
model2.add(Softmax())

params2 = []
for layer in model2.layers:
    if hasattr(layer, "get_parameters"):
        params2.extend(layer.get_parameters())

model2.compile(CategoricalCrossEntropy(), optimizer=Adam(params2, learning_rate=0.01))
model2.fit(X, y, epochs=15, batch_size=10)

model2.save("test_model.npy")

model3 = FeedForwardNeuralNetwork(verbose=True, seed=42)
model3.load("test_model.npy")

predictions2 = model2.predict(X)
predictions3 = model3.predict(X)
print("Loaded model prediction sum:", np.sum(predictions3.data))
print("Original model prediction sum:", np.sum(predictions2.data))

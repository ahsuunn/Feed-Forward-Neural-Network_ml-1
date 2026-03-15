import numpy as np
from .module.autodiff import Tensor
import matplotlib.pyplot as plt


class FeedForwardNeuralNetwork:
    def __init__(self, verbose=False, seed=None):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_function, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, X, y, epochs, batch_size):
        if self.loss_function is None:
            raise ValueError(
                "Model must be compiled with a loss function before fitting."
            )
        if self.optimizer is None:
            raise ValueError("Model must be compiled with an optimizer before fitting.")

        num_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                batch_loss = self.loss_function.forward(y_batch, y_pred)
                epoch_loss += batch_loss.data * X_batch.shape[0]

                # Backward pass
                batch_loss.backward()

                # Backend optimization
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss /= num_samples

            if self.verbose:
                if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}")

    def predict(self, X):
        return self.forward(X)

    def plot_weight_distribution(self, layer_idx, num_bins=50):
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError("Invalid layer index")

        layer = self.layers[layer_idx]

        if not hasattr(layer, "get_parameters"):
            raise ValueError("Layer does not have parameters")

        params = layer.get_parameters()
        if len(params) == 0:
            raise ValueError("Layer has no parameters")

        param = params[0]
        if not isinstance(param, Tensor):
            raise ValueError("Layer parameters are not tensors")

        data = param.data
        if data.ndim != 2:
            raise ValueError("Layer parameters are not 2D tensors")

        data = data.flatten()
        plt.hist(data, bins=num_bins)
        plt.title(f"Weight distribution for layer {layer_idx}")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.show()

    def plot_gradient_weight_distribution(self, layer_idx, num_bins=50):
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError("Invalid layer index")

        layer = self.layers[layer_idx]

        if not hasattr(layer, "get_parameters"):
            raise ValueError("Layer does not have parameters")

        params = layer.get_parameters()
        if len(params) == 0:
            raise ValueError("Layer has no parameters")

        param = params[0]
        if not isinstance(param, Tensor):
            raise ValueError("Layer parameters are not tensors")

        data = param.grad
        if data.ndim != 2:
            raise ValueError("Layer parameters are not 2D tensors")

        data = data.flatten()
        plt.hist(data, bins=num_bins)
        plt.title(f"Gradient weight distribution for layer {layer_idx}")
        plt.xlabel("Gradient weight")
        plt.ylabel("Frequency")
        plt.show()

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass

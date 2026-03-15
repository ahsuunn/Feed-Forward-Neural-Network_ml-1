import numpy as np


class FeedForwardNeuralNetwork:
    def __init__(self, verbose=False, seed=None):
        self.layers = []
        self.loss_function = None
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_function):
        self.loss_function = loss_function

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        pass

    def update(self, lambda_regularization=0.0):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lambda_regularization)

    def fit(self, X, y, epochs, batch_size, lambda_regularization=0.0):
        if self.loss_function is None:
            raise ValueError(
                "Model must be compiled with a loss function before fitting."
            )

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

                # Update weights
                self.update(lambda_regularization)

            epoch_loss /= num_samples

            if self.verbose:
                if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}")

    def predict(self, X):
        return self.forward(X)

    def plot_weight_distribution(self, layer_idx, num_bins=50):
        pass

    def plot_gradient_weight_distribution(self, layer_idx, num_bins=50):
        pass

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass

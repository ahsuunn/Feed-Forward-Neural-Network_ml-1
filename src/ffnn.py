import numpy as np
from tqdm import tqdm
from module.autodiff import Tensor
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

    def fit(self, X, y, epochs, batch_size, validation_data=None):
        if self.loss_function is None:
            raise ValueError(
                "Model must be compiled with a loss function before fitting."
            )
        if self.optimizer is None:
            raise ValueError("Model must be compiled with an optimizer before fitting.")

        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        # History tracker
        history = {"loss": [], "val_loss": []} if validation_data else {"loss": []}

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            # Setup tqdm
            batch_iterator = range(num_batches)
            if self.verbose == True:
                batch_iterator = tqdm(
                    batch_iterator, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
                )

            for i in batch_iterator:
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

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

                # Update tqdm
                if self.verbose == True:
                    batch_iterator.set_postfix(batch_loss=f"{batch_loss.data:.4f}")

            epoch_loss /= num_samples
            history["loss"].append(epoch_loss)

            # Validation calculation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.forward(X_val)
                v_loss = self.loss_function.forward(y_val, val_pred)
                history["val_loss"].append(v_loss.data)

                if self.verbose == True:
                    batch_iterator.set_postfix(
                        loss=f"{epoch_loss:.4f}", val_loss=f"{v_loss.data:.4f}"
                    )
            else:
                if self.verbose == True:
                    batch_iterator.set_postfix(loss=f"{epoch_loss:.4f}")

        return history

    def predict(self, X):
        return self.forward(X)

    def plot_distributions(self, layer_idx, num_bins=50):
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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Weight distribution
        data = np.asarray(param.data).ravel()
        data = data[np.isfinite(data)]
        if data.size > 0:
            ax1.hist(data, bins=num_bins)
            ax1.set_title(f"Weight distribution (Layer {layer_idx})")
            ax1.set_xlabel("Weight")
            ax1.set_ylabel("Frequency")
        else:
            ax1.set_title(f"Weight distribution (Layer {layer_idx}) - No data")

        # Gradient distribution
        grad_data = np.asarray(param.grad).ravel()
        grad_data = grad_data[np.isfinite(grad_data)]
        if grad_data.size > 0:
            ax2.hist(grad_data, bins=num_bins)
            ax2.set_title(f"Gradient distribution (Layer {layer_idx})")
            ax2.set_xlabel("Gradient")
            ax2.set_ylabel("Frequency")
        else:
            ax2.set_title(f"Gradient distribution (Layer {layer_idx}) - No data")

        plt.tight_layout()
        plt.show()

    def save(self, file_path):
        np.save(file_path, self.layers)

    def load(self, file_path):
        self.layers = np.load(file_path, allow_pickle=True)

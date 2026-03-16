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

                # Snapshot gradients
                for p in self.optimizer.parameters:
                    p.saved_grad = p.grad.copy()

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

    def get_layer_parameter(self, layer_idx):
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

        return param

    def _plot_histogram(self, ax, data, num_bins, title, xlabel):
        data = np.asarray(data).ravel()
        data = data[np.isfinite(data)]
        if data.size > 0:
            ax.hist(data, bins=num_bins)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency")
        else:
            ax.set_title(f"{title} - No data")

    def _get_plottable_layers(self, layer_indices=None):
        if layer_indices is not None:
            idxs = layer_indices
        else:
            idxs = range(len(self.layers))

        result = []
        for i in idxs:
            if i < 0 or i >= len(self.layers):
                raise ValueError(f"Invalid layer index: {i}")
            layer = self.layers[i]
            if hasattr(layer, "get_parameters"):
                params = layer.get_parameters()
                if params and isinstance(params[0], Tensor):
                    result.append((i, layer))
        return result

    def plot_weight_distributions(self, layer_indices=None, num_bins=50):
        plottable = self._get_plottable_layers(layer_indices)
        if not plottable:
            print("No plottable layers found.")
            return

        n = len(plottable)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (idx, layer) in zip(axes, plottable):
            param = layer.get_parameters()[0]
            self._plot_histogram(
                ax, param.data, num_bins, f"Weight distribution (Layer {idx})", "Weight"
            )

        plt.tight_layout()
        plt.show()

    def plot_gradient_distributions(self, layer_indices=None, num_bins=50):
        plottable = self._get_plottable_layers(layer_indices)

        # filter saved_grad
        filtered = []
        for idx, layer in plottable:
            for param in layer.get_parameters():
                if isinstance(param, Tensor) and param.saved_grad is not None:
                    filtered.append((idx, param))
                    break

        if not filtered:
            print("No plottable layers with gradients found.")
            return

        n = len(filtered)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (idx, param) in zip(axes, filtered):
            self._plot_histogram(
                ax,
                param.saved_grad,
                num_bins,
                f"Gradient distribution (Layer {idx})",
                "Gradient",
            )

        plt.tight_layout()
        plt.show()

    def save(self, file_path):
        np.save(file_path, self.layers)

    def load(self, file_path):
        self.layers = np.load(file_path, allow_pickle=True)

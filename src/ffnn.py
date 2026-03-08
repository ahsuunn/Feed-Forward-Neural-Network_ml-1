class FeedForwardNeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_type,
        loss_function_type,
        learning_rate,
        regularization_type,
        batch_size,
        epochs,
        verbose=False,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.regularization_type = regularization_type

        self.activation = Activation(activation_type)
        self.loss_function = LossFunction(loss_function_type)

        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def plot_weight_distribution(self, layer_name, num_bins=50):
        """
        Args:
            layer_name (str): Layer name
            num_bins (int, optional): Number of bins. Defaults to 50.
        """

    def plot_gradient_weight_distribution(self, layer_name, num_bins=50):
        """
        Args:
            layer_name (str): Layer name
            num_bins (int, optional): Number of bins. Defaults to 50.
        """

    def calculate_loss(self, y_true, y_pred, lamda_regularization=0.0):
        """
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            lamda_regularization (float, optional): Lambda regularization. Defaults to 0.0.
        """
        if self.regularization_type == "l1":
            raise NotImplementedError
        elif self.regularization_type == "l2":
            raise NotImplementedError
        else:
            raise ValueError("Invalid regularization type")

    def save(self, file_path):
        """
        Args:
            file_path (str): File path
        """

    def load(self, file_path):
        """
        Args:
            file_path (str): File path
        """

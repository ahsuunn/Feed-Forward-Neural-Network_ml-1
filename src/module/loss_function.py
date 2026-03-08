class LossFunction:
    def __init__(self, loss_function_type):
        """
        Loss function type
        Available Type:
            - mse
            - binary_crossentropy
            - categorical_crossentropy
        Args:
            loss_function_type (str): Loss function type
        """
        self.loss_function_type = loss_function_type

    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError

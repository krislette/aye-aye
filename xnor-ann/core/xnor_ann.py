import pandas as pd
import numpy as np


class XnorAnn:
    def __init__(
        self, fan_in: int, fan_out: int, hidden_neurons: int, learning_rate: float
    ) -> None:
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.target_epochs = 10000
        self.target_error = 1.00e-15

    def init_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Read excel first
        df = pd.read_excel("data/training_data.xlsx")

        # And then convert the dataframe to a numpy array
        training_data = df.to_numpy()

        # Params: orig_matrix, index to delete, axis=1 is column, axis=0 is row
        training_inputs = np.delete(training_data, 2, axis=1)

        # Colon (:) means all rows, and then (2) just column 2
        training_outputs = training_data[:, 2]

        return training_inputs, training_outputs

    def init_weights(self) -> np.ndarray:
        # Initialize weights using He initialization (for ReLU)
        mean = 0.0
        std_dev = np.sqrt(2 / self.fan_in)

        # Returns initial weights w the formula G(0.0, sqrt(2 / n)) [third param is size]
        return np.random.normal(mean, std_dev, self.fan_in)

    def relu(self, value: float) -> float:
        # Basic implementation of ReLU
        return value if value > 0 else 0

    def relu_derivative(self, value: float) -> float:
        # Basic implementation of ReLU derivative
        return 1.0 if value > 0 else 0.0

    def feedforward(self) -> None:
        # TODO: Implement feedforward
        pass

    def backpropagation(self) -> None:
        # TODO: Implement backprop
        pass

    def train(self) -> None:
        # TODO: Create main training logic
        pass

    def predict(self) -> None:
        # TODO: Create prediction logic
        pass

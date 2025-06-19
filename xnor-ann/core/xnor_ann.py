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

    def init_biases(self) -> np.ndarray:
        return np.zeros((1, self.hidden_neurons))

    def relu(self, values: np.ndarray) -> np.ndarray:
        # Compares maximum between two values (ReLU) on matrices
        # Ex: np.maximum(0, [-1, 2], [3, -4]) = [[0, 2], [3, 0]] <- ReLU!!!
        return np.maximum(0, values)

    def relu_derivative(self, values: np.ndarray) -> np.ndarray:
        # Compares if value is 0 or 1 (since derivative returns 0 or 1)
        # Ex: [[-1, 2], [3, -4]] > 0 ? [False, True, True, False]
        # Then .astype(int) converts booleans to int -> [0, 1, 1, 0] <- ReLU derivative!!!
        return (values > 0).astype(int)

    def feedforward(
        self, inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Formula: (i1 * w1) + (i2 * w2) + bias
        net_outputs = np.dot(inputs, weights) + biases
        outputs = self.relu(net_outputs)
        return net_outputs, outputs

    def backpropagation(self) -> None:
        # 
        pass

    def train(self) -> None:
        # TODO: Create main training logic
        pass

    def predict(self) -> None:
        # TODO: Create prediction logic
        pass

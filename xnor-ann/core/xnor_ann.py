import pandas as pd
import numpy as np


class XnorAnn:
    def __init__(self) -> None:
        # TODO: Initialize parameters
        pass

    def init_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Read excel first
        df = pd.read_excel("data/training_data.xlsx")

        # And then convert the dataframe to a numpy array
        training_data = df.to_numpy()

        # Params: orig_matrix, index to delete, axis=1 is column, axis=0 is row
        training_inputs = np.delete(training_data, 2, axis=1)
        print(training_inputs)

        # Colon (:) means all rows, and then (2) just column 2
        training_outputs = training_data[:, 2]
        print(training_outputs)

        return training_inputs, training_outputs

    def init_weights(self) -> None:
        # TODO: Init weights using He (for ReLU)
        pass

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

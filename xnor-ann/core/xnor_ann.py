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
        training_targets = training_data[:, 2].reshape(-1, 1)

        return training_inputs, training_targets

    def init_weights(self) -> tuple[np.ndarray, np.ndarray]:
        # Initialize weights using He initialization (for ReLU)
        mean = 0.0

        # For hidden layer first: 2 inputs x 3 neurons -> (2, 3)
        # Initial hidden weights w the formula G(0.0, sqrt(2 / n)) [third param is size]
        std_dev_hidden = np.sqrt(2 / self.fan_in)
        hidden_weights = np.random.normal(
            mean, std_dev_hidden, (self.fan_in, self.hidden_neurons)
        )

        # For output layer: 3 inputs x 1 output -> (3, 1)
        # Initial output weight w the formula G(0.0, sqrt(2 / n)) [third param is size]
        std_dev_output = np.sqrt(2 / self.hidden_neurons)
        output_weights = np.random.normal(
            mean, std_dev_output, (self.hidden_neurons, self.fan_out)
        )

        # Return both weight matrices (hidden and output)
        return hidden_weights, output_weights

    def init_biases(self) -> np.ndarray:
        # Init hidden biases with the size (1, 3) for 3 hidden neurons
        hidden_biases = np.zeros((1, self.hidden_neurons))

        # Then init output bias with the size of (1, 1) for 1 output neuron
        output_biases = np.zeros((1, self.fan_out))

        return hidden_biases, output_biases

    def relu(self, values: np.ndarray) -> np.ndarray:
        # Compares maximum between two values (ReLU) on matrices
        # Ex: np.maximum(0, [-1, 2], [3, -4]) = [[0, 2], [3, 0]] <- ReLU!!!
        return np.maximum(0, values)

    def relu_derivative(self, values: np.ndarray) -> np.ndarray:
        # Compares if value is 0 or 1 (since derivative returns 0 or 1)
        # Ex: [[-1, 2], [3, -4]] > 0 ? [False, True, True, False]
        # Then .astype(int) converts booleans to int -> [0, 1, 1, 0] <- ReLU derivative!!!
        return (values > 0).astype(int)

    def calculate_mse(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        # Get the total number of training entries first
        size = targets.shape[0]

        # And then calculate MSE: summation of -> each (output - target)^2 / 2 then average
        mse = np.sum(((outputs - targets) ** 2) / 2) / size
        return mse

    def feedforward(
        self,
        inputs: np.ndarray,
        hidden_weights: np.ndarray,
        output_weights: np.ndarray,
        hidden_biases: np.ndarray,
        output_biases: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # For hidden layer first
        # Formula: (i1 * w1) + (i2 * w2) + bias then active with ReLU
        # Performs (1x2) dot (2x3) + (1x3) = (1x3)
        hidden_net_outputs = np.dot(inputs, hidden_weights) + hidden_biases
        hidden_outputs = self.relu(hidden_net_outputs)

        # Then for output layer
        # Formula: (i1 * w1) + (i2 * w2) + bias then active with ReLU
        # Performs (1x3) dot (3x1) + (1x1) = (1x1)
        final_net_outputs = np.dot(hidden_outputs, output_weights) + output_biases
        final_outputs = self.relu(final_net_outputs)

        # Then return net outputs & activated outputs for hidden and output layers
        return hidden_net_outputs, hidden_outputs, final_net_outputs, final_outputs

    def backpropagation(
        self,
        net_outputs: np.ndarray,
        outputs: np.ndarray,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Get (dE/dO) and (dO/dnet)
        de_dout = outputs - targets
        dout_dnet = self.relu_derivative(net_outputs)

        # Then apply to the chain rule formula: (dE/dO) * (dO/dnet) * (dnet/d<input>)
        de_dw = np.dot((de_dout * dout_dnet), inputs)
        de_db = np.sum((de_dout * dout_dnet), axis=0, keepdims=True)

        return de_dw, de_db

    def update_parameters(
        self, weight_gradients: np.ndarray, bias_gradients: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        new_weights = weight_gradients - (self.learning_rate * weight_gradients)
        new_biases = bias_gradients - (self.learning_rate * bias_gradients)
        return new_weights, new_biases

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        inputs, targets = self.init_training_data()
        weights = self.init_weights()
        biases = self.init_biases()

        for epoch in range(self.target_epochs):
            # Step 1: Feedforward
            net_outputs, outputs = self.feedforward(inputs, weights, biases)

            # Step 2: Calculate MSE
            mean_squared_error = self.calculate_mse(outputs, targets)

            # Step 3: Backpropagation
            weight_gradients, bias_gradients = self.backpropagation(
                net_outputs, outputs, inputs, targets
            )

            # Step 4: Update weights and biases
            weights, biases = self.update_parameters(weight_gradients, bias_gradients)

            # Check if network converged
            if mean_squared_error <= self.target_error:
                with open("data/optimized_parameters.txt", "w") as file:
                    file.write(f"{weights},{biases}")
                return weights, biases

            # Print epoch every 100 for monitoring on the CLI
            if epoch % 100 == 0:
                print(f"Epoch: {epoch:<5} " f"| Error: {mean_squared_error:<12.8f} ")

    def predict(self) -> None:
        # TODO: Create prediction logic
        pass

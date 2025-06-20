import pandas as pd
import numpy as np
import os


class XnorAnn:
    def __init__(
        self, fan_in: int, fan_out: int, hidden_neurons: int, learning_rate: float
    ) -> None:
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.target_epochs = 1000000
        self.target_error = 1e-5

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
        mean = 0.0

        # Initialize hidden weights using He initialization (for leaky_relu)
        # For hidden layer first: 2 inputs x 3 neurons -> (2, 3)
        # Initial hidden weights w the formula G(0.0, sqrt(2 / n)) [third param is size]
        std_dev_hidden = np.sqrt(2 / self.fan_in)
        hidden_weights = np.random.normal(
            mean, std_dev_hidden, (self.fan_in, self.hidden_neurons)
        )

        # Initialize output weights using Xavier initialization (for Sigmoid)
        # For output layer: 3 inputs x 1 output -> (3, 1)
        # Initial output weight w the formula G(0.0, sqrt(2 / (fan_in + fan_out))) [third param is size]
        std_dev_output = np.sqrt(2 / (self.hidden_neurons + self.fan_out))
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

    def leaky_relu(self, values: np.ndarray) -> np.ndarray:
        # Leaky ReLU implementation for an array/matrix
        # Like IF() on excel, but for arrays, leak = 0.1
        return np.where(values > 0, values, 0.1 * values)

    def leaky_relu_derivative(self, values: np.ndarray) -> np.ndarray:
        # Leaky ReLU derivative where if it is less than 0, return the leak (0.1)
        return np.where(values > 0, 1, 0.1)

    def sigmoid(self, values: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-values))

    def sigmoid_derivative(self, values: np.ndarray) -> np.ndarray:
        sig = self.sigmoid(values)
        return sig * (1 - sig)

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
        # Formula: (i1 * w1) + (i2 * w2) + bias then active with leaky_relu
        # Performs (200x2) dot (2x3) + (1x3) = (200x3)
        hidden_net_outputs = np.dot(inputs, hidden_weights) + hidden_biases
        hidden_outputs = self.leaky_relu(hidden_net_outputs)

        # Then for output layer
        # Formula: (i1 * w1) + (i2 * w2) + bias then active with leaky_relu
        # Performs (200x3) dot (3x1) + (1x1) = (200x1)
        final_net_outputs = np.dot(hidden_outputs, output_weights) + output_biases
        final_outputs = self.sigmoid(final_net_outputs)

        # Then return net outputs & activated outputs for hidden and output layers
        return hidden_net_outputs, hidden_outputs, final_net_outputs, final_outputs

    def backpropagation(
        self,
        inputs: np.ndarray,
        hidden_net_outputs: np.ndarray,
        hidden_outputs: np.ndarray,
        final_net_outputs: np.ndarray,
        final_outputs: np.ndarray,
        targets: np.ndarray,
        output_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        size = inputs.shape[0]

        # 1.0: Start from output layer gradients
        # Calculate error derivative: dE/dO
        de_dout = final_outputs - targets

        # 1.1: Calculate activation derivative: dO/dNet
        dout_dnet = self.sigmoid_derivative(final_net_outputs)

        # Extra: Store (dE/dO) * (dO/dNet) since they are reusable (200x1)
        output_delta = de_dout * dout_dnet

        # 1.2: Chain rule to get output weight gradients: (dE/dO) * (dO/dNet) * (dNet/d<input>)
        # Output: (3x200) dot (200x1) = (3x1) gradients
        output_weight_gradients = np.dot(hidden_outputs.T, output_delta) / size

        # 1.3: Output bias gradients: dE/dB (1x1)
        output_bias_gradients = np.sum(output_delta, axis=0, keepdims=True) / size

        # 2.0: Then next are hidden layer gradients
        # (200x1) dot (1x3) = (200x3)
        de_dw = np.dot(output_delta, output_weights.T)

        # 2.1: Calculate activativation derivative
        dw_dnet = self.leaky_relu_derivative(hidden_net_outputs)

        # Extra: Store (dE/dW) * (dW/dNet) since they are reusable (200x3)
        hidden_delta = de_dw * dw_dnet

        # 2.2: Chain rule to get hidden weight gradients
        # Hidden: (2x200) dot (200x3) = (2x3) gradients
        hidden_weight_gradients = np.dot(inputs.T, hidden_delta) / size

        # 2.3: Hidden bias gradients: dE/dB (1x3)
        hidden_bias_gradients = np.sum(hidden_delta, axis=0, keepdims=True) / size

        return (
            hidden_weight_gradients,
            hidden_bias_gradients,
            output_weight_gradients,
            output_bias_gradients,
        )

    def update_parameters(
        self,
        hidden_weight_gradients: np.ndarray,
        hidden_bias_gradients: np.ndarray,
        output_weight_gradients: np.ndarray,
        output_bias_gradients: np.ndarray,
        hidden_weights: np.ndarray,
        hidden_biases: np.ndarray,
        output_weights: np.ndarray,
        output_biases: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get new weights using gradient descent formula (weights - (LR * gradients))
        new_hidden_weights = hidden_weights - (
            self.learning_rate * hidden_weight_gradients
        )
        new_output_weights = output_weights - (
            self.learning_rate * output_weight_gradients
        )

        # And get new biases using gradient descent formula (biases - (LR * gradients))
        new_hidden_biases = hidden_biases - (self.learning_rate * hidden_bias_gradients)
        new_output_biases = output_biases - (self.learning_rate * output_bias_gradients)

        return (
            new_hidden_weights,  # Shape: (2x3)
            new_hidden_biases,  # Shape: (1x3)
            new_output_weights,  # Shape: (3x1)
            new_output_biases,  # Shape: (1x1)
        )

    def save_parameters(
        self, hidden_weights, hidden_biases, output_weights, output_biases
    ):
        # Save trained params using nump save function
        os.makedirs("data", exist_ok=True)

        # Save each parameter array separately
        np.save("data/hidden_weights.npy", hidden_weights)
        np.save("data/hidden_biases.npy", hidden_biases)
        np.save("data/output_weights.npy", output_weights)
        np.save("data/output_biases.npy", output_biases)

        print("Parameters saved")

    def load_parameters(self):
        # Load trained parameters using load function of numpy
        try:
            hidden_weights = np.load("data/hidden_weights.npy")
            hidden_biases = np.load("data/hidden_biases.npy")
            output_weights = np.load("data/output_weights.npy")
            output_biases = np.load("data/output_biases.npy")

            return hidden_weights, hidden_biases, output_weights, output_biases
        except FileNotFoundError:
            raise FileNotFoundError(
                "Trained parameters not found. Please train the model first."
            )

    def train(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Initialize parameters for training
        inputs, targets = self.init_training_data()
        hidden_weights, output_weights = self.init_weights()
        hidden_biases, output_biases = self.init_biases()

        # Main training loop
        for epoch in range(self.target_epochs):
            # Step 1: Feedforward
            hidden_net_outputs, hidden_outputs, output_net_outputs, final_outputs = (
                self.feedforward(
                    inputs, hidden_weights, output_weights, hidden_biases, output_biases
                )
            )

            # Step 2: Calculate MSE
            mean_squared_error = self.calculate_mse(final_outputs, targets)

            # Step 3: Backpropagation
            (
                hidden_weight_gradients,
                hidden_bias_gradients,
                output_weight_gradients,
                output_bias_gradients,
            ) = self.backpropagation(
                inputs,
                hidden_net_outputs,
                hidden_outputs,
                output_net_outputs,
                final_outputs,
                targets,
                output_weights,
            )

            # Step 4: Update weights and biases
            hidden_weights, hidden_biases, output_weights, output_biases = (
                self.update_parameters(
                    hidden_weight_gradients,
                    hidden_bias_gradients,
                    output_weight_gradients,
                    output_bias_gradients,
                    hidden_weights,
                    hidden_biases,
                    output_weights,
                    output_biases,
                )
            )

            # Check if network converged
            if mean_squared_error <= self.target_error:
                print(f"Converged at epoch {epoch} with error {mean_squared_error}")
                break

            # Print epoch every 100 for monitoring on the CLI
            if epoch % 100 == 0:
                print(f"Epoch: {epoch:<5} " f"| Error: {mean_squared_error:<12.8f} ")

        # Always save parameters after training (whether converged or not)
        self.save_parameters(
            hidden_weights, hidden_biases, output_weights, output_biases
        )

        print(f"Final error: {mean_squared_error}")

        # Return final parameters even if not converged
        return hidden_weights, hidden_biases, output_weights, output_biases

    def predict(self, input: list[float]) -> float:
        try:
            # Load parameters using the load params mmethod
            hidden_weights, hidden_biases, output_weights, output_biases = (
                self.load_parameters()
            )
        except FileNotFoundError:
            print("Error: Trained parameters not found. Please train the model first.")
            return -1.0
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return -1.0

        # Prepare input (1 sample, 2 features)
        inputs = np.array(input).reshape(1, -1)

        # Feedforward using loaded parameters
        _, _, _, prediction = self.feedforward(
            inputs, hidden_weights, output_weights, hidden_biases, output_biases
        )

        # Return scalar prediction as float
        return float(prediction[0][0])

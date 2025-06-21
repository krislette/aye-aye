import pandas as pd
import numpy as np
import os
from gui.visuals import Visualizer
from core.gate_definitions import GateDefinitions


class LogicGateAnn:
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        hidden_neurons: int,
        learning_rate: float,
        gate_type: str,
    ) -> None:
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.gate_type = gate_type.upper()
        self.target_epochs = 1000000
        self.target_error = 1e-5

        # Validate gate type
        if self.gate_type not in GateDefinitions.get_available_gates():
            raise ValueError(f"Unsupported gate type: {gate_type}")

    def init_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Reads training data from Excel based on gate type
        try:
            # Try to read gate-specific Excel file first
            file_name = f"data/{self.gate_type.lower()}_training_data.xlsx"
            df = pd.read_excel(file_name)

            # Convert dataframe to numpy array
            training_data = df.to_numpy()

            # Split inputs and targets where target should be at last col
            training_inputs = training_data[:, :-1]
            training_targets = training_data[:, -1].reshape(-1, 1)

            print(f"> Loaded training data from {file_name}")

            # Return input and target matrices
            return training_inputs, training_targets
        except FileNotFoundError:
            print(f"> Excel file not found: {file_name}")
            print(f"> Generating {self.gate_type} training data programmatically...")

            # Fallback to programmatic generation
            training_inputs, training_targets, _ = GateDefinitions.get_gate_data(
                self.gate_type
            )

            # Save the generated data to Excel for future use
            combined_data = np.column_stack((training_inputs, training_targets))

            if self.gate_type == "NOT":
                df = pd.DataFrame(combined_data, columns=["Input 1", "Target"])
            else:
                df = pd.DataFrame(
                    combined_data, columns=["Input 1", "Input 2", "Target"]
                )

            # Then save to `data` dir
            os.makedirs("data", exist_ok=True)
            df.to_excel(file_name, index=False)
            print(f"> Generated and saved training data to {file_name}")

            return training_inputs, training_targets

    def init_weights(self) -> tuple[np.ndarray, np.ndarray]:
        # Init mean
        mean = 0.0

        # He initialization for hidden layer (Leaky ReLU)
        # Initial hidden weights w the formula G(0.0, sqrt(2 / n)) [third param is size]
        std_dev_hidden = np.sqrt(2 / self.fan_in)
        hidden_weights = np.random.normal(
            mean, std_dev_hidden, (self.fan_in, self.hidden_neurons)
        )

        # Xavier initialization for output layer (Sigmoid)
        # Initial output weight w the formula G(0.0, sqrt(2 / (fan_in + fan_out))) [third param is size]
        std_dev_output = np.sqrt(2 / (self.hidden_neurons + self.fan_out))
        output_weights = np.random.normal(
            mean, std_dev_output, (self.hidden_neurons, self.fan_out)
        )

        # Return both weight matrices (hidden and output)
        return hidden_weights, output_weights

    def init_biases(self) -> np.ndarray:
        # Init hidden biases with the size (1, 8) for 8 hidden neurons
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
        # Sigmoid implementation for matrices/arrays
        return 1 / (1 + np.exp(-values))

    def sigmoid_derivative(self, values: np.ndarray) -> np.ndarray:
        # Store sigmoid activation result then get derivative using the formula
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
        # Formula: (i1 * w1) + (i2 * w2) + ... + bias then activate with leaky ReLU
        # Performs (200x2) dot (2x8) + (1x8) = (200x8)
        hidden_net_outputs = np.dot(inputs, hidden_weights) + hidden_biases
        hidden_outputs = self.leaky_relu(hidden_net_outputs)

        # Then for output layer
        # Formula: (i1 * w1) + (i2 * w2) + ... + bias then activate with sigmoid
        # Performs (200x8) dot (8x1) + (1x1) = (200x1)
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

        # For 1: Simultaneously calculate (O-T)(0-0.25)(O1/2/3.../1)
        # 1.0: Start from output layer gradients
        # Calculate error derivative: dE/dO
        de_dout = final_outputs - targets

        # 1.1: Calculate activation derivative: dO/dNet
        dout_dnet = self.sigmoid_derivative(final_net_outputs)

        # Extra: Store (dE/dO) * (dO/dNet) since they are reusable (200x1)
        output_delta = de_dout * dout_dnet

        # 1.2: Chain rule to get output weight gradients: (dE/dO) * (dO/dNet) * (dNet/d<input>)
        # Output weight: (8x200) dot (200x1) = (8x1) gradients
        output_weight_gradients = np.dot(hidden_outputs.T, output_delta) / size

        # 1.3: Output bias gradients: dE/dB (1x1)
        output_bias_gradients = np.sum(output_delta, axis=0, keepdims=True) / size

        # For 2: Simultaneously calculate (O-T)(0-0.25)(W...)(1/leak)(I1/I2)
        # 2.0: Then next are hidden layer gradients
        # (200x1) dot (1x8) = (200x8)
        de_dw = np.dot(output_delta, output_weights.T)

        # 2.1: Calculate activativation derivative
        dw_dnet = self.leaky_relu_derivative(hidden_net_outputs)

        # Extra: Store (dE/dW) * (dW/dNet) since they are reusable (200x3)
        hidden_delta = de_dw * dw_dnet

        # 2.2: Chain rule to get hidden weight gradients
        # Hidden weight: (2x200) dot (200x8) = (2x8) gradients
        hidden_weight_gradients = np.dot(inputs.T, hidden_delta) / size

        # 2.3: Hidden bias gradients: dE/dB (1x8)
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
            new_hidden_weights,
            new_hidden_biases,
            new_output_weights,
            new_output_biases,
        )

    def get_parameter_filenames(self) -> tuple[str, str, str, str]:
        # Generate parameter filenames specific to the gate type
        return (
            f"data/{self.gate_type.lower()}_hidden_weights.npy",
            f"data/{self.gate_type.lower()}_hidden_biases.npy",
            f"data/{self.gate_type.lower()}_output_weights.npy",
            f"data/{self.gate_type.lower()}_output_biases.npy",
        )

    def save_parameters(
        self, hidden_weights, hidden_biases, output_weights, output_biases
    ):
        # Save trained parameters with gate-specific filenames
        os.makedirs("data", exist_ok=True)

        hw_file, hb_file, ow_file, ob_file = self.get_parameter_filenames()

        np.save(hw_file, hidden_weights)
        np.save(hb_file, hidden_biases)
        np.save(ow_file, output_weights)
        np.save(ob_file, output_biases)

        print(f"> {self.gate_type} parameters saved")

    def load_parameters(self):
        # Load trained parameters with gate-specific filenames
        hw_file, hb_file, ow_file, ob_file = self.get_parameter_filenames()

        try:
            hidden_weights = np.load(hw_file)
            hidden_biases = np.load(hb_file)
            output_weights = np.load(ow_file)
            output_biases = np.load(ob_file)

            return hidden_weights, hidden_biases, output_weights, output_biases
        except FileNotFoundError:
            raise FileNotFoundError(
                f"> Trained parameters for {self.gate_type} not found. Please train the model first."
            )

    def train(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Initialize parameters for training
        inputs, targets = self.init_training_data()
        hidden_weights, output_weights = self.init_weights()
        hidden_biases, output_biases = self.init_biases()
        visualizer = Visualizer()

        print(f"Training {self.gate_type} gate...")

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
                print(f"\n> Converged at epoch {epoch} with error {mean_squared_error}")
                break

            # Print epoch every 1000 for monitoring on the CLI
            if epoch % 1000 == 0:
                print(f"Epoch: {epoch:<8} | Error: {mean_squared_error:<12.8f}")
                visualizer.update_error(epoch, mean_squared_error)

        # Always save parameters after training (whether converged or not)
        self.save_parameters(
            hidden_weights, hidden_biases, output_weights, output_biases
        )

        # Call visualizer to create diagram output
        visualizer.plot_ann_diagram(
            self.gate_type,
            self.fan_in,
            self.hidden_neurons,
            self.fan_out,
            hidden_weights,
            output_weights,
            hidden_biases,
            output_biases,
        )

        # Then just print MSE
        print(f"> Final MSE: {mean_squared_error}")

        # Return final parameters even if not converged
        return hidden_weights, hidden_biases, output_weights, output_biases

    def predict(self, input: list[float]) -> float:
        try:
            # Load parameters using the load params method
            hidden_weights, hidden_biases, output_weights, output_biases = (
                self.load_parameters()
            )
        except FileNotFoundError:
            print(
                f"Error: Trained parameters for {self.gate_type} not found. Please train the model first."
            )
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

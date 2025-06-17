import numpy as np
from utils.parser import parse_excel
from core.forward_pass import perform_forward_pass
from core.backpropagation import perform_backpropagation
from core.activations import relu_derivative


class AndAnn:
    def __init__(self):
        self.fan_in = 2
        self.fan_out = 1
        self.learning_rate = 0.01
        self.target_epochs = 10000
        self.target_error = 1.00e-15
        self.weights = self.initialize_weights()
        self.bias = np.random.uniform(-1, 1)
        self.training_data = parse_excel("data/training_data.xlsx")

    def initialize_weights(self) -> list[float]:
        # Initialize weights using Xavier Uniform Initialization
        limit = np.sqrt(6 / (self.fan_in + self.fan_out))
        return np.random.uniform(-limit, limit, size=(self.fan_in, self.fan_out))

    def train(self) -> tuple[float, float, float]:
        # Initialize weights and biases
        w1_new, w2_new = self.weights[0], self.weights[1]
        b1_new = self.bias

        # Initialize variables for monitoring
        current_error = 1
        epoch = 0

        # Count training data rows for MSE computation later
        training_size = self.training_data.shape[0]

        # Main loop for epochs
        for epoch in range(self.target_epochs):
            # Early return if we converged to desired error or max epochs reached
            if current_error <= self.target_error or epoch >= self.target_epochs:
                # If the training is successful, write the parameters to a file
                with open("data/optimized_parameters.txt", "w") as file:
                    file.write(f"{w1_new.item()},{w2_new.item()},{b1_new.item()}")
                return w1_new, w2_new, b1_new

            # Initialize the previously new values into old values every new iter
            w1_old = w1_new
            w2_old = w2_new
            b1_old = b1_new

            # Gradients
            de_dw1 = 0
            de_dw2 = 0
            de_db1 = 0

            # For storing results of forward pass
            forward_results = []

            # Initialize error to zero every iter for fresh calculation
            current_error = 0

            # Step 1: Perform forward pass through the entire dataset
            for entry in self.training_data:
                # Extract inputs and target from training data
                i1, i2, target = entry

                # Forward pass for each data point
                net_output, output = perform_forward_pass(
                    i1, i2, w1_old, w2_old, b1_old
                )

                # Accumulate all results of forward pass
                forward_results.append((i1, i2, target, net_output, output))

                # Accumulate error for MSE calculation later
                current_error += ((target - output) ** 2) / 2

            # Then calculate gradients from the stored results
            for i1, i2, target, net_output, output in forward_results:
                de_dw1 += (output - target) * (relu_derivative(net_output)) * i1
                de_dw2 += (output - target) * (relu_derivative(net_output)) * i2
                de_db1 += (output - target) * (relu_derivative(net_output)) * 1

            # Average gradients
            de_dw1 /= training_size
            de_dw2 /= training_size
            de_db1 /= training_size

            # Step 2: Calculate MSE
            current_error /= training_size

            # Print epoch every 100 for monitoring on the CLI
            if epoch % 100 == 0:
                print(f"| Epoch: {epoch} | Error: {current_error} |")

            # Step 3: Perform backpropagation
            w1_new, w2_new, b1_new = perform_backpropagation(
                w1_old, w2_old, b1_old, de_dw1, de_dw2, de_db1, self.learning_rate
            )

        # Return negatives if failed to find optimized weights and bias
        return -1, -1, -1

    def predict(self, input: list[float]) -> float:
        try:
            with open("data/optimized_parameters.txt", "r") as file:
                # Get the optimized weights and biases (from training)
                parameters = file.read().strip().split(",")
                w1, w2, b1 = (
                    float(parameters[0]),
                    float(parameters[1]),
                    float(parameters[2]),
                )
        except FileNotFoundError:
            # If file not found, use initial weights and bias
            w1, w2, b1 = self.weights[0], self.weights[1], self.bias

        # Extract input and compute prediction
        i1, i2 = input
        _, output = perform_forward_pass(i1, i2, w1, w2, b1)

        return output

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
        w1_new, w2_new = self.weights[0][0], self.weights[1][0]
        b1_new = self.bias

        # Main loop for epochs
        for epoch in range(self.target_epochs):
            # Initialize error to zero every iter for fresh calculation
            accumulated_error = 0

            # Main training loop
            for entry in self.training_data:
                # Extract inputs and target from training data
                i1, i2, target = entry

                # Initialize the previously new values into old values every new iter
                w1_old = w1_new
                w2_old = w2_new
                b1_old = b1_new

                # Step 1: Perform forward pass
                net_output, output = perform_forward_pass(
                    i1, i2, w1_old, w2_old, b1_old
                )

                # Step 2: Accumulate error (For MSE)
                accumulated_error += ((target - output) ** 2) / 2

                # Step 3: Perform backpropagation
                de_dout = output - target
                dout_dnet = relu_derivative(net_output)

                de_dw1 = perform_backpropagation(de_dout, dout_dnet, i1)
                de_dw2 = perform_backpropagation(de_dout, dout_dnet, i2)
                de_db1 = perform_backpropagation(de_dout, dout_dnet, 1)

                # Calculate gradients
                w1_new = w1_old - (self.learning_rate * de_dw1)
                w2_new = w2_old - (self.learning_rate * de_dw2)
                b1_new = b1_old - (self.learning_rate * de_db1)

            # Get MSE
            training_size = self.training_data.shape[0]
            mean_squared_error = accumulated_error / training_size

            # Check if ANN converged to desired error
            if mean_squared_error <= self.target_error:
                # If the training is successful, write the parameters to a file
                with open("data/optimized_parameters.txt", "w") as file:
                    file.write(f"{w1_new},{w2_new},{b1_new}")
                return w1_new, w2_new, b1_new

            # Print epoch every 100 for monitoring on the CLI
            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch:<5} "
                    f"| Error: {mean_squared_error:<12.8f} "
                    f"| Weights: [{w1_new:<12.8f}, {w2_new:<12.8f}] "
                    f"| Bias: [{b1_new:<12.8f}]"
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

import pandas as pd
import numpy as np


def parse_excel(file: str) -> list[list]:
    # Parse the training data from the excel file
    df = pd.read_excel(file)
    training_data = df.to_numpy()
    return training_data


def initialize_weights(fan_in: int, fan_out: int) -> list[float]:
    # Initialize weighst using Xavier Uniform Initialization
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


def perform_pe(i1: int, i2: int, w1: float, w2: float, b1: float) -> float:
    # Perform neuron calculation
    return (i1 * w1) + (i2 * w2) + b1


def relu(output: float) -> float:
    # Relu utility
    # return output if output > 0 else 0
    return max(output, 0)


def relu_derivative(output: float) -> int:
    # Fix: Relu activation is different from relu derivative REMEMBER!!!
    return 1 if output > 0 else 0


def train() -> tuple[float, float, float]:
    # Preprocessing
    training_data = parse_excel("training_data.xlsx")

    # Initialization
    weights = initialize_weights(2, 1).flatten()
    w1_new, w2_new = weights[0], weights[1]
    b1_new = np.random.uniform(1, 10)
    learning_rate = 0.01

    # Targets
    target_error = 1.00e-15
    target_epochs = 5000

    # Monitored variables
    current_error = 1
    epoch = 0

    # Number of entries in training data
    entries_count = training_data.shape[0]

    while current_error > target_error and epoch < target_epochs:
        # Initialization
        current_error = 0

        print("-" * 80)
        print(f"Current weights: {w1_new}, {w2_new}, Current Bias: {b1_new}")

        # Main training loop
        for entry in training_data:
            w1_old = w1_new
            w2_old = w2_new
            b1_old = b1_new

            # GD Step 1: Forward pass
            net_output = perform_pe(entry[0], entry[1], w1_new, w2_new, b1_new)
            output = relu(net_output)

            # GD Step 2: Calculate MSE (accumulator first)
            current_error += ((entry[2] - output) ** 2) / 2

            # GD Step 3: Perform back propagation
            de_dout = output - entry[2]
            dout_dnet = relu_derivative(net_output)

            de_dw1 = de_dout * dout_dnet * (entry[0])
            de_dw2 = de_dout * dout_dnet * (entry[1])
            de_db1 = de_dout * dout_dnet * 1

            w1_new = w1_old - (learning_rate * de_dw1)
            w2_new = w2_old - (learning_rate * de_dw2)
            b1_new = b1_old - (learning_rate * de_db1)

        # Continue: Calculate MSE by averaging squared differences
        current_error /= entries_count

        # Try early return if we converge early
        if current_error <= target_error:
            return w1_new, w2_new, b1_new

        print(f"Updated weights: {w1_new}, {w2_new}, Updated Bias: {b1_new}")
        print(f"Current Error: {current_error}")

        epoch += 1

    return w1_new, w2_new, b1_new


def main():
    # Train model and get optimized weights and biases
    optimized_w1, optimized_w2, optimized_b1 = train()

    # Print optimized weights and biases
    print("-" * 80)
    print("Training result:")
    print(f"Weights (1, 2): {optimized_w1}, {optimized_w2}, Bias: {optimized_b1}")

    # Try optimized weights and biases on sample AND inputs
    and_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

    print("-" * 80)
    print("Test:")
    for inputs in and_inputs:
        output = relu(
            perform_pe(inputs[0], inputs[1], optimized_w1, optimized_w2, optimized_b1)
        )
        print(f"A: {inputs[0]}, B: {inputs[1]}, Output: {output}")


if __name__ == "__main__":
    main()

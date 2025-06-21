from core.logic_gate_ann import LogicGateAnn
from core.gate_definitions import GateDefinitions
import numpy as np
import sys


def main() -> None:
    # If user didn't add any argument
    if len(sys.argv) < 2:
        print("Enter: python main.py <train/predict> [gate_type]")
        print(f"Available gates: {', '.join(GateDefinitions.get_available_gates())}")
        return

    action = sys.argv[1].lower()

    # Get gate type from command line or prompt user
    if len(sys.argv) >= 3:
        gate_type = sys.argv[2].upper()
    else:
        available_gates = GateDefinitions.get_available_gates()
        print(f"Available gates: {', '.join(available_gates)}")
        gate_type = input("Enter gate type: ").strip().upper()

    # Validate gate type
    if gate_type not in GateDefinitions.get_available_gates():
        print(
            f"Invalid gate type. Available: {', '.join(GateDefinitions.get_available_gates())}"
        )
        return

    # Set fan_in depending on gate type
    fan_in = 1 if gate_type == "NOT" else 2

    # Then Instantiate
    ann = LogicGateAnn(fan_in, 1, 8, 0.1, gate_type)

    # If User wants to train
    if action == "train":
        print(f"Training the {gate_type} ANN...")
        hidden_weights, hidden_biases, output_weights, output_biases = ann.train()

        np.set_printoptions(precision=4, suppress=True)
        print(f"\n-> {gate_type} Hidden Weights:")
        print(hidden_weights)
        print(f"\n-> {gate_type} Hidden Biases:")
        print(hidden_biases)
        print(f"\n-> {gate_type} Output Weights:")
        print(output_weights)
        print(f"\n-> {gate_type} Output Biases:")
        print(output_biases)
    # Else if user wants to predict
    elif action == "predict":
        try:
            if gate_type == "NOT":
                i1 = float(input("Enter input (0 or 1): ").strip())
                i2 = 0  # NOT gate only uses first input
                inputs = [i1]
            else:
                i1 = float(input("Enter first input (0 or 1): ").strip())
                i2 = float(input("Enter second input (0 or 1): ").strip())
                inputs = [i1, i2]

            output = ann.predict(inputs)
            if gate_type == "NOT":
                print(f"{gate_type} Gate - Input: {i1}, Output: {output:.4f}")
            else:
                print(f"{gate_type} Gate - Inputs: [{i1}, {i2}], Output: {output:.4f}")

        except ValueError:
            print("Invalid input. Please enter numeric values.")
    else:
        print("Invalid action. Use 'train' or 'predict'")


if __name__ == "__main__":
    main()

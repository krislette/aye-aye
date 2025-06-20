from core.xnor_ann import XnorAnn
import sys


def main() -> None:
    # Params: fan_in=2 inputs, fan_out=1 output, 3 hidden neurons, learning rate=0.1
    xnor_ann = XnorAnn(2, 1, 8, 0.1)

    # If user wants to train
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("Training the XNOR ANN...")

        # Call train method from class
        (
            hidden_weights,
            hidden_biases,
            output_weights,
            output_biases,
        ) = xnor_ann.train()

        print("\nTraining completed. Optimized parameters:")
        print(f"Hidden Weights:\n{hidden_weights}")
        print(f"Hidden Biases:\n{hidden_biases}")
        print(f"Output Weights:\n{output_weights}")
        print(f"Output Biases:\n{output_biases}")
    # If user wants to predict
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        try:
            # Get user input
            i1 = float(input("Enter first input (0 or 1): ").strip())
            i2 = float(input("Enter second input (0 or 1): ").strip())

            # Then call predict method from class
            output = xnor_ann.predict([i1, i2])
            print(f"Inputs: [{i1}, {i2}], Output: {output}")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    # If invalid CLI run command
    else:
        print(
            "Enter: python main.py <argument> where argument can be 'train' or 'predict'"
        )


if __name__ == "__main__":
    main()

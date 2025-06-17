import sys
from core.ann import AndAnn


def main() -> None:
    # Instantiate ANN
    and_ann = AndAnn()

    # If user wants to train, train the ANN...
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Get parameters from ANN
        w1, w2, b1 = and_ann.train()

        # Check if the result of training is -1 (meaning didn't converge)
        if w1 == -1:
            print("Training failed to converge within target epochs.")
        else:
            print(f"Optimized Weights: [{w1}, {w2}], Optimized Bias: [{b1}]")
    # Else if user wants to predict given that ANN is trained...
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        try:
            # Get user input and then print
            i1 = float(input("Enter first input (0 or 1): ").strip())
            i2 = float(input("Enter second input (0 or 1): ").strip())
            output = and_ann.predict([i1, i2])
            print(f"Inputs: [{i1}, {i2}], Output: {output}")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    # Else print instruction to user
    else:
        print(
            "Enter: python main.py <argument> where argument can be 'train' or 'predict'"
        )


if __name__ == "__main__":
    main()

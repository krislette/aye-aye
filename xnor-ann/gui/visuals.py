import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os


class Visualizer:
    def __init__(self) -> None:
        # Init list to store error values over epochs
        self.epochs = []
        self.errors = []

        # Create figure and axis only once
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("MSE (Error)")
        self.ax.set_title("Live Training Error")
        (self.line,) = self.ax.plot([], [], color="blue", label="Error")
        self.ax.grid(True)
        self.ax.legend()

        # Show improvements even when error is very small (e.g. 1e-5)
        self.ax.set_yscale("log")

        # Show in interactive mode
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def update_error(self, epoch: int, error: float) -> None:
        # Append new values
        self.epochs.append(epoch)
        self.errors.append(error)

        # Update line data
        self.line.set_xdata(self.epochs)
        self.line.set_ydata(self.errors)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_ann_diagram(
        self,
        fan_in: int,
        hidden_neurons: int,
        fan_out: int,
        hidden_weights: np.ndarray,
        output_weights: np.ndarray,
        hidden_biases: np.ndarray,
        output_biases: np.ndarray,
    ) -> None:
        # Structure: 2 input -> 8 hidden -> 1 output
        input_size = fan_in
        hidden_size = hidden_neurons
        output_size = fan_out

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        # X-coordinates for each layer
        x_input = 0.1
        x_hidden = 0.5
        x_output = 0.9

        def get_y_coords(n: int) -> list[float]:
            return list(np.linspace(0.1, 0.9, n))

        def center_y_coords(layer_size: int, total_neurons: int) -> list[float]:
            """Centers a layer vertically relative to the largest layer."""
            full_range = np.linspace(0.1, 0.9, total_neurons)
            center_index = len(full_range) // 2
            offset = layer_size // 2
            return list(
                full_range[center_index - offset : center_index - offset + layer_size]
            )

        # Vertical alignment
        input_y = center_y_coords(input_size, hidden_size)
        hidden_y = get_y_coords(hidden_size)
        output_y = center_y_coords(output_size, hidden_size)

        # Draw input layer
        for y in input_y:
            ax.add_patch(
                mpatches.Circle((x_input, y), 0.03, fill=True, color="lightblue")
            )
        ax.text(x_input, 1.0, "Input", ha="center")

        # Draw hidden layer
        ax.text(x_hidden, 1.0, "Hidden (ReLU)", ha="center")
        for idx, y in enumerate(hidden_y):
            ax.add_patch(
                mpatches.Circle((x_hidden, y), 0.03, fill=True, color="lightgreen")
            )
            bias_val = hidden_biases[0][len(hidden_y) - 1 - idx]
            ax.text(
                x_hidden + 0.04,
                y,
                f"b:{bias_val:.2f}",
                fontsize=6,
                color="purple",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        # Draw output layer
        ax.text(x_output, 1.0, "Output (Sigmoid)", ha="center")
        for idx, y in enumerate(output_y):
            ax.add_patch(
                mpatches.Circle((x_output, y), 0.03, fill=True, color="orange")
            )
            bias_val = output_biases[0][idx]
            ax.text(
                x_output + 0.04,
                y,
                f"b:{bias_val:.2f}",
                fontsize=6,
                color="purple",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        # Reverse for printing
        input_y = input_y[::-1]
        hidden_y = hidden_y[::-1]

        # Input -> Hidden connections with offsets on labels to avoid clashing
        for y1_idx, y1 in enumerate(input_y):
            for y2_idx, y2 in enumerate(hidden_y):
                ax.plot([x_input, x_hidden], [y1, y2], "gray", linewidth=0.5)
                weight_val = hidden_weights[y1_idx][y2_idx]

                # Position label exactly like output layer - at midpoint of connection
                mid_x = (x_input + x_hidden) / 2
                mid_y = (y1 + y2) / 2

                # Small offset to prevent overlap when multiple lines cross
                # Offset based on which input and which hidden neuron
                offset_x = 0.05 * (y1_idx - 0.5)
                offset_y = 0.0005 * (y2_idx - 3.5)

                ax.text(
                    mid_x + offset_x,
                    mid_y + offset_y,
                    f"{weight_val:.2f}",
                    fontsize=6,
                    color="blue",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )

        # Hidden -> Output connections
        for h_idx, y1 in enumerate(hidden_y):
            for o_idx, y2 in enumerate(output_y):
                ax.plot([x_hidden, x_output], [y1, y2], "gray", linewidth=0.5)
                weight_val = output_weights[h_idx][o_idx]
                mid_x = (x_hidden + x_output) / 2
                mid_y = (y1 + y2) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    f"{weight_val:.2f}",
                    fontsize=6,
                    color="blue",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )

        # Save figure with padding so nothing gets cut off
        os.makedirs("out", exist_ok=True)
        plt.savefig("out/ann_structure.png", bbox_inches="tight", pad_inches=0.3)
        print("> ANN diagram saved to out/ann_structure.png")

        # Prevent closing
        plt.ioff()
        plt.show()

import numpy as np


class GateDefinitions:
    def get_gate_data(gate_type: str) -> tuple[np.ndarray, np.ndarray, str]:
        # Returns training data for the specified gate type
        gate_type = gate_type.upper()

        if gate_type == "AND":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            targets = np.array([[0], [0], [0], [1]])
            return inputs, targets, "AND"
        elif gate_type == "OR":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            targets = np.array([[0], [1], [1], [1]])
            return inputs, targets, "OR"
        elif gate_type == "XOR":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            targets = np.array([[0], [1], [1], [0]])
            return inputs, targets, "XOR"
        elif gate_type == "NOT":
            inputs = np.array([[0], [1]])
            targets = np.array([[1], [0]])
            return inputs, targets, "NOT"
        elif gate_type == "NAND":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            targets = np.array([[1], [1], [1], [0]])
            return inputs, targets, "NAND"
        elif gate_type == "NOR":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            targets = np.array([[1], [0], [0], [0]])
            return inputs, targets, "NOR"
        elif gate_type == "XNOR":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            targets = np.array([[1], [0], [0], [1]])
            return inputs, targets, "XNOR"
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")

    def get_available_gates() -> list[str]:
        # Returns list of available gate types
        return ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR"]

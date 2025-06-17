def relu(value: float) -> float:
    return value if value > 0 else 0


def relu_derivative(value: float) -> int:
    return 1 if value > 0 else 0

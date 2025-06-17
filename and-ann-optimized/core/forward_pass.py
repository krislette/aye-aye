from core.activations import relu


def perform_pe(i1: float, i2: float, w1: float, w2: float, b1: float) -> float:
    return (i1 * w1) + (i2 * w2) + b1


def perform_forward_pass(
    i1: float, i2: float, w1: float, w2: float, b1: float
) -> tuple[float, float]:
    net_output = perform_pe(i1, i2, w1, w2, b1)
    output = relu(net_output)
    return net_output, output

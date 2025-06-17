def perform_backpropagation(de_dout: float, dout_dnet: float, input: float) -> float:
    # This follows the formula: (dE/dO) * (dO/dnet) * (dnet/d<input>) [chain rule]
    return de_dout * dout_dnet * input

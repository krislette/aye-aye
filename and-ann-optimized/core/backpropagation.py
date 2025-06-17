def perform_backpropagation(
    w1_old: float,
    w2_old: float,
    b1_old: float,
    de_dw1: float,
    de_dw2: float,
    de_db1: float,
    learning_rate: float,
) -> tuple[float, float, float]:
    w1_new = w1_old - (learning_rate * de_dw1)
    w2_new = w2_old - (learning_rate * de_dw2)
    b1_new = b1_old - (learning_rate * de_db1)
    return w1_new, w2_new, b1_new

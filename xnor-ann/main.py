from core.xnor_ann import XnorAnn


def main() -> None:
    # Params: Fan in, fan out, hidden neurons, LR
    xnor_ann = XnorAnn(2, 1, 3, 0.01)
    xnor_ann.init_weights()


if __name__ == "__main__":
    main()

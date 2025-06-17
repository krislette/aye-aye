import pandas as pd


def parse_excel(path: str) -> list[list]:
    df = pd.read_excel(path)
    return df.to_numpy()

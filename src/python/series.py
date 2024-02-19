from collections import Counter
import pandas as pd


def collaps_series(*args,  dtype=float):
    serie = Counter()
    for arg in args:
        for index, value in arg.items():
            serie[index] += value
    return pd.Series(serie,  dtype=dtype)

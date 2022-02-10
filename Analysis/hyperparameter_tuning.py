from typing import Union, Iterable

import pandas as pd


def mean_relative_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return ((y_pred - y_true).abs() / y_true.abs()).mean()


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return (y_pred - y_true).abs().mean()
